//! Path simulation: one step kernel per model ([`PathModel`]), streaming path
//! functionals ([`PathFunctional`]) and the generic chunked driver.

use quantlib_models::{BlackScholesModel, HestonModel};
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use super::core::{ChunkSums, HestonScheme, Moments, MonteCarloConfig, Sample, chunk_rng};
use super::heston::{HestonState, HestonStepper};

/// Normals drawn per step, at most.
const MAX_NORMALS: usize = 2;

/// One model's state and time step.
pub(crate) trait PathModel: Sync {
    type State: Copy;
    /// Normals consumed per step (at most `MAX_NORMALS`).
    const NORMALS: usize;

    fn initial(&self) -> Self::State;
    fn step(&self, state: &mut Self::State, z: &[f64; MAX_NORMALS]);
    fn spot(state: &Self::State) -> f64;
}

/// Reduces one path to a [`Sample`] while it is simulated, so paths are never stored.
pub(crate) trait PathFunctional: Sync {
    type Acc: Copy;

    fn start(&self, spot: f64) -> Self::Acc;
    /// Called with the spot after every step.
    fn observe(&self, _acc: &mut Self::Acc, _spot: f64) {}
    fn finish(&self, acc: &Self::Acc, terminal: f64) -> Sample;
}

/// Simulates `config.n_samples()` samples of `n_steps` steps and returns the
/// moments of the functional's samples.
///
/// Samples are split into chunks of `config.chunk_size`; chunk `i` draws from
/// its own stream `chunk_rng(seed, i)` and chunk moments are merged in chunk
/// order, so the result is bit-identical serially, in parallel, and for any
/// number of threads.
pub(crate) fn simulate<M: PathModel, F: PathFunctional>(
    config: &MonteCarloConfig,
    seed: u64,
    model: &M,
    functional: &F,
    n_steps: usize,
) -> Moments {
    let n_samples = config.n_samples();
    let n_chunks = n_samples.div_ceil(config.chunk_size);
    let run_chunk = |chunk: usize| {
        let first = chunk * config.chunk_size;
        let count = config.chunk_size.min(n_samples - first);
        let mut rng = chunk_rng(seed, chunk);
        let mut sums = ChunkSums::default();
        for _ in 0..count {
            let sample = if config.antithetic {
                let (up, down) = simulate_antithetic_pair(model, functional, n_steps, &mut rng);
                up.mean(down)
            } else {
                simulate_path(model, functional, n_steps, &mut rng)
            };
            sums.push(sample);
        }
        sums.moments()
    };

    let chunks: Vec<Moments> = if config.parallel {
        (0..n_chunks).into_par_iter().map(run_chunk).collect()
    } else {
        (0..n_chunks).map(run_chunk).collect()
    };
    chunks.into_iter().fold(Moments::default(), Moments::merge)
}

#[inline(always)]
fn draw<M: PathModel>(rng: &mut impl Rng) -> [f64; MAX_NORMALS] {
    let mut z = [0.0; MAX_NORMALS];
    for value in z.iter_mut().take(M::NORMALS) {
        *value = rng.sample(StandardNormal);
    }
    z
}

#[inline(always)]
fn simulate_path<M: PathModel, F: PathFunctional>(
    model: &M,
    functional: &F,
    n_steps: usize,
    rng: &mut impl Rng,
) -> Sample {
    let mut state = model.initial();
    let mut acc = functional.start(M::spot(&state));
    for _ in 0..n_steps {
        model.step(&mut state, &draw::<M>(rng));
        functional.observe(&mut acc, M::spot(&state));
    }
    functional.finish(&acc, M::spot(&state))
}

#[inline(always)]
fn simulate_antithetic_pair<M: PathModel, F: PathFunctional>(
    model: &M,
    functional: &F,
    n_steps: usize,
    rng: &mut impl Rng,
) -> (Sample, Sample) {
    let (mut up, mut down) = (model.initial(), model.initial());
    let mut acc_up = functional.start(M::spot(&up));
    let mut acc_down = functional.start(M::spot(&down));
    for _ in 0..n_steps {
        let z = draw::<M>(rng);
        model.step(&mut up, &z);
        model.step(&mut down, &z.map(|value| -value));
        functional.observe(&mut acc_up, M::spot(&up));
        functional.observe(&mut acc_down, M::spot(&down));
    }
    (
        functional.finish(&acc_up, M::spot(&up)),
        functional.finish(&acc_down, M::spot(&down)),
    )
}

/// Full paths, row-major: path `i` is `values[i * (n_steps + 1)..][..n_steps + 1]`.
/// Uses the same chunks and streams as pricing (without antithetic pairing).
pub(crate) fn record_paths<M: PathModel>(
    config: &MonteCarloConfig,
    seed: u64,
    model: &M,
    n_steps: usize,
) -> Vec<f64> {
    let width = n_steps + 1;
    let mut values = vec![0.0; config.n_paths * width];
    let fill_chunk = |(chunk, rows): (usize, &mut [f64])| {
        let mut rng = chunk_rng(seed, chunk);
        for row in rows.chunks_exact_mut(width) {
            let mut state = model.initial();
            row[0] = M::spot(&state);
            for value in &mut row[1..] {
                model.step(&mut state, &draw::<M>(&mut rng));
                *value = M::spot(&state);
            }
        }
    };

    let chunk_len = config.chunk_size * width;
    if config.parallel {
        values
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(fill_chunk);
    } else {
        values
            .chunks_mut(chunk_len)
            .enumerate()
            .for_each(fill_chunk);
    }
    values
}

/// Exact log-normal step: `S ← S·exp((r − q − σ²/2)Δ + σ√Δ·z)`.
pub(crate) struct BlackScholesPath {
    spot: f64,
    drift: f64,
    diffusion: f64,
}

impl BlackScholesPath {
    pub(crate) fn new(model: &BlackScholesModel, spot: f64, dt: f64) -> Self {
        Self {
            spot,
            drift: (model.r - model.q - 0.5 * model.sigma * model.sigma) * dt,
            diffusion: model.sigma * dt.sqrt(),
        }
    }
}

impl PathModel for BlackScholesPath {
    type State = f64;
    const NORMALS: usize = 1;

    fn initial(&self) -> f64 {
        self.spot
    }

    #[inline]
    fn step(&self, spot: &mut f64, z: &[f64; MAX_NORMALS]) {
        *spot *= (self.drift + self.diffusion * z[0]).exp();
    }

    fn spot(spot: &f64) -> f64 {
        *spot
    }
}

pub(crate) struct HestonPath {
    spot: f64,
    v0: f64,
    stepper: HestonStepper,
}

impl HestonPath {
    pub(crate) fn new(model: &HestonModel, spot: f64, dt: f64, scheme: HestonScheme) -> Self {
        Self {
            spot,
            v0: model.v0,
            stepper: HestonStepper::new(model, dt, scheme),
        }
    }
}

impl PathModel for HestonPath {
    type State = HestonState;
    const NORMALS: usize = 2;

    fn initial(&self) -> HestonState {
        HestonState {
            spot: self.spot,
            variance: self.v0,
        }
    }

    #[inline]
    fn step(&self, state: &mut HestonState, z: &[f64; MAX_NORMALS]) {
        self.stepper.step(state, z[0], z[1]);
    }

    fn spot(state: &HestonState) -> f64 {
        state.spot
    }
}

/// Payoff of the terminal spot; the control variate is `S_T` itself.
pub(crate) struct Terminal<P> {
    pub(crate) payoff: P,
}

impl<P: Fn(f64) -> f64 + Sync> PathFunctional for Terminal<P> {
    type Acc = ();

    fn start(&self, _spot: f64) {}

    #[inline]
    fn finish(&self, _acc: &(), terminal: f64) -> Sample {
        Sample {
            y: (self.payoff)(terminal),
            x: terminal,
        }
    }
}

/// Control variate of an Asian payoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AsianControl {
    /// The same payoff on the geometric average (closed form under BS).
    GeometricPayoff,
    /// The arithmetic average of the spot (its mean is known for any model).
    ArithmeticAverage,
}

/// Running sum and log-sum over the monitoring points `S_0, S_Δ, …, S_T`.
pub(crate) struct Average<P> {
    pub(crate) geometric: bool,
    pub(crate) control: AsianControl,
    pub(crate) payoff: P,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AverageAcc {
    sum: f64,
    log_sum: f64,
    count: f64,
}

impl<P: Fn(f64) -> f64 + Sync> PathFunctional for Average<P> {
    type Acc = AverageAcc;

    fn start(&self, spot: f64) -> AverageAcc {
        AverageAcc {
            sum: spot,
            log_sum: spot.ln(),
            count: 1.0,
        }
    }

    #[inline]
    fn observe(&self, acc: &mut AverageAcc, spot: f64) {
        acc.sum += spot;
        acc.log_sum += spot.ln();
        acc.count += 1.0;
    }

    #[inline]
    fn finish(&self, acc: &AverageAcc, _terminal: f64) -> Sample {
        let arithmetic = acc.sum / acc.count;
        let geometric = (acc.log_sum / acc.count).exp();
        let average = if self.geometric {
            geometric
        } else {
            arithmetic
        };
        Sample {
            y: (self.payoff)(average),
            x: match self.control {
                AsianControl::GeometricPayoff => (self.payoff)(geometric),
                AsianControl::ArithmeticAverage => arithmetic,
            },
        }
    }
}
