use quantlib_models::{BlackScholesModel, HestonModel};
use rayon::prelude::*;

use quantlib_core::norm_cdf;

use super::core::{HestonScheme, MonteCarloConfig, NormalRng};

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlackScholesSimulator {
    config: MonteCarloConfig,
}

impl BlackScholesSimulator {
    pub(crate) fn new(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    pub(crate) fn simulate_terminal(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        if self.should_parallelize_terminal() {
            self.simulate_terminal_parallel(spot, model, ttm, rng)
        } else {
            self.simulate_terminal_serial(spot, model, ttm, rng)
        }
    }

    pub(crate) fn simulate_paths(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        if self.config.parallel {
            self.simulate_paths_parallel(spot, model, ttm, rng)
        } else {
            self.simulate_paths_serial(spot, model, ttm, rng)
        }
    }

    fn should_parallelize_terminal(&self) -> bool {
        if !self.config.parallel {
            return false;
        }

        // Exact BS terminal pricing does little work per path. For typical batch sizes,
        // Rayon scheduling overhead dominates and serial execution is faster.
        let threads = rayon::current_num_threads().max(1);
        let min_work = self
            .config
            .chunk_size
            .saturating_mul(threads)
            .saturating_mul(256);
        self.config.n_paths >= min_work
    }

    fn simulate_terminal_serial(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let drift = (model.r - model.q - 0.5 * model.sigma * model.sigma) * ttm;
        let diffusion = model.sigma * ttm.sqrt();

        let mut spots = Vec::with_capacity(self.config.n_paths);
        for _ in 0..self.config.n_paths {
            let z = rng.standard_normal();
            spots.push(spot * (drift + diffusion * z).exp());
        }
        spots
    }

    fn simulate_terminal_parallel(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let drift = (model.r - model.q - 0.5 * model.sigma * model.sigma) * ttm;
        let diffusion = model.sigma * ttm.sqrt();
        let chunk_size = self.config.chunk_size.min(self.config.n_paths).max(1);
        let mut spots = vec![0.0; self.config.n_paths];

        for chunk in spots.chunks_mut(chunk_size) {
            let chunk_len = chunk.len();
            let mut normals = Vec::with_capacity(chunk_len);
            for _ in 0..chunk_len {
                normals.push(rng.standard_normal());
            }

            chunk
                .par_iter_mut()
                .zip(normals.par_iter().copied())
                .for_each(|(s, z)| *s = spot * (drift + diffusion * z).exp());
        }
        spots
    }

    fn simulate_paths_serial(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let dt = ttm / self.config.n_steps as f64;
        let drift_step = (model.r - model.q - 0.5 * model.sigma * model.sigma) * dt;
        let diffusion_step = model.sigma * dt.sqrt();

        let mut paths = vec![vec![0.0; self.config.n_steps + 1]; self.config.n_paths];
        for path in &mut paths {
            path[0] = spot;
            for step in 0..self.config.n_steps {
                let z = rng.standard_normal();
                path[step + 1] = path[step] * (drift_step + diffusion_step * z).exp();
            }
        }

        paths
    }

    fn simulate_paths_parallel(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let dt = ttm / self.config.n_steps as f64;
        let drift_step = (model.r - model.q - 0.5 * model.sigma * model.sigma) * dt;
        let diffusion_step = model.sigma * dt.sqrt();
        let chunk_size = self.config.chunk_size.min(self.config.n_paths).max(1);

        let mut paths = vec![vec![0.0; self.config.n_steps + 1]; self.config.n_paths];
        for path_chunk in paths.chunks_mut(chunk_size) {
            let chunk_len = path_chunk.len();
            let mut normals = Vec::with_capacity(chunk_len * self.config.n_steps);
            for _ in 0..(chunk_len * self.config.n_steps) {
                normals.push(rng.standard_normal());
            }

            path_chunk
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_idx, path)| {
                    let base = local_idx * self.config.n_steps;
                    path[0] = spot;
                    for step in 0..self.config.n_steps {
                        let z = normals[base + step];
                        path[step + 1] = path[step] * (drift_step + diffusion_step * z).exp();
                    }
                });
        }

        paths
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HestonSimulator {
    config: MonteCarloConfig,
}

impl HestonSimulator {
    pub(crate) fn new(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    pub(crate) fn simulate_terminal(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<f64> {
        let stepper = HestonStepper::new(model, ttm / self.config.n_steps as f64, self.config);
        let n_steps = self.config.n_steps;
        let terminal = |normals: &mut dyn FnMut() -> (f64, f64)| {
            let mut state = HestonState::new(spot, model.v0);
            for _ in 0..n_steps {
                let (z_s, z_v) = normals();
                stepper.step(&mut state, z_s, z_v);
            }
            state.spot
        };

        if !self.config.parallel {
            return (0..self.config.n_paths)
                .map(|_| terminal(&mut || (rng.standard_normal(), rng.standard_normal())))
                .collect();
        }

        let mut terminals = vec![0.0; self.config.n_paths];
        for chunk in terminals.chunks_mut(self.chunk_size()) {
            let normals = draw_normals(rng, chunk.len() * n_steps * 2);
            chunk.par_iter_mut().enumerate().for_each(|(index, value)| {
                let mut path_normals = normals[index * n_steps * 2..].as_chunks::<2>().0.iter();
                *value = terminal(&mut || {
                    let pair = path_normals.next().expect("two normals per step");
                    (pair[0], pair[1])
                });
            });
        }
        terminals
    }

    pub(crate) fn simulate_paths(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
        rng: &mut NormalRng,
    ) -> Vec<Vec<f64>> {
        let stepper = HestonStepper::new(model, ttm / self.config.n_steps as f64, self.config);
        let n_steps = self.config.n_steps;
        let fill = |path: &mut [f64], normals: &mut dyn FnMut() -> (f64, f64)| {
            let mut state = HestonState::new(spot, model.v0);
            path[0] = spot;
            for value in &mut path[1..] {
                let (z_s, z_v) = normals();
                stepper.step(&mut state, z_s, z_v);
                *value = state.spot;
            }
        };

        let mut paths = vec![vec![0.0; n_steps + 1]; self.config.n_paths];
        if !self.config.parallel {
            for path in &mut paths {
                fill(path, &mut || (rng.standard_normal(), rng.standard_normal()));
            }
            return paths;
        }

        for chunk in paths.chunks_mut(self.chunk_size()) {
            let normals = draw_normals(rng, chunk.len() * n_steps * 2);
            chunk.par_iter_mut().enumerate().for_each(|(index, path)| {
                let mut path_normals = normals[index * n_steps * 2..].as_chunks::<2>().0.iter();
                fill(path, &mut || {
                    let pair = path_normals.next().expect("two normals per step");
                    (pair[0], pair[1])
                });
            });
        }
        paths
    }

    fn chunk_size(&self) -> usize {
        self.config.chunk_size.min(self.config.n_paths).max(1)
    }
}

/// Normals for one parallel chunk, drawn in the same order the serial loop uses.
fn draw_normals(rng: &mut NormalRng, count: usize) -> Vec<f64> {
    (0..count).map(|_| rng.standard_normal()).collect()
}

#[derive(Debug, Clone, Copy)]
struct HestonState {
    spot: f64,
    variance: f64,
}

impl HestonState {
    fn new(spot: f64, variance: f64) -> Self {
        Self { spot, variance }
    }
}

/// One Heston time step, with every per-step constant precomputed.
///
/// Each step consumes two independent standard normals: `z_s` drives the
/// stock and `z_v` the variance (the QE scheme turns it into its uniform via
/// `U = Φ(z_v)`), so both schemes share the RNG stream layout.
#[derive(Debug, Clone, Copy)]
struct HestonStepper {
    scheme: HestonScheme,
    dt: f64,
    sqrt_dt: f64,
    drift: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    rho_orth: f64,
    qe: QeConstants,
}

/// Andersen (2008) QE constants for γ₁ = γ₂ = ½.
#[derive(Debug, Clone, Copy)]
struct QeConstants {
    /// `e^{−κΔ}`.
    decay: f64,
    /// `(1 − e^{−κΔ}) / κ`, which tends to `Δ` as `κ → 0`.
    decay_integral: f64,
    k0: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
}

/// Switching level of `ψ = s²/m²` between the quadratic and exponential branches.
const QE_PSI_CRITICAL: f64 = 1.5;

/// Below this vol-of-vol the variance is treated as deterministic, where the
/// QE coefficients (which divide by σ) are ill-defined.
const DETERMINISTIC_SIGMA: f64 = 1.0e-8;

impl HestonStepper {
    fn new(model: &HestonModel, dt: f64, config: MonteCarloConfig) -> Self {
        let (kappa, theta, sigma, rho) = (model.kappa, model.theta, model.sigma, model.rho);
        let decay = (-kappa * dt).exp();
        let decay_integral = if kappa * dt < 1.0e-12 {
            dt
        } else {
            -(-kappa * dt).exp_m1() / kappa
        };

        let (gamma1, gamma2) = (0.5, 0.5);
        let (k0, k1, k2) = if sigma > DETERMINISTIC_SIGMA {
            let lead = kappa * rho / sigma - 0.5;
            (
                -rho * kappa * theta * dt / sigma,
                gamma1 * dt * lead - rho / sigma,
                gamma2 * dt * lead + rho / sigma,
            )
        } else {
            (0.0, -gamma1 * dt, -gamma2 * dt)
        };

        Self {
            scheme: config.heston_scheme,
            dt,
            sqrt_dt: dt.sqrt(),
            drift: (model.r - model.q) * dt,
            kappa,
            theta,
            sigma,
            rho,
            rho_orth: (1.0 - rho * rho).max(0.0).sqrt(),
            qe: QeConstants {
                decay,
                decay_integral,
                k0,
                k1,
                k2,
                k3: gamma1 * dt * (1.0 - rho * rho),
                k4: gamma2 * dt * (1.0 - rho * rho),
            },
        }
    }

    #[inline]
    fn step(&self, state: &mut HestonState, z_s: f64, z_v: f64) {
        match self.scheme {
            HestonScheme::FullTruncation => self.full_truncation_step(state, z_s, z_v),
            HestonScheme::QuadraticExponential => self.qe_step(state, z_s, z_v),
        }
    }

    /// Lord, Koekkoek & van Dijk (2010): the variance is never floored; only
    /// its positive part enters the drift and diffusion.
    fn full_truncation_step(&self, state: &mut HestonState, z_s: f64, z_v: f64) {
        let v_pos = state.variance.max(0.0);
        let vol = v_pos.sqrt();
        let d_w1 = self.sqrt_dt * z_s;
        let d_w2 = self.sqrt_dt * (self.rho * z_s + self.rho_orth * z_v);

        state.spot *= (self.drift - 0.5 * v_pos * self.dt + vol * d_w1).exp();
        state.variance += self.kappa * (self.theta - v_pos) * self.dt + self.sigma * vol * d_w2;
    }

    /// Andersen (2008) quadratic-exponential scheme with martingale correction.
    fn qe_step(&self, state: &mut HestonState, z_s: f64, z_v: f64) {
        let qe = &self.qe;
        let v = state.variance;

        // Exact conditional mean and variance of v_{n+1} given v_n.
        let mean = self.theta + (v - self.theta) * qe.decay;
        let sigma_sq = self.sigma * self.sigma;
        let var = v * sigma_sq * qe.decay * qe.decay_integral
            + 0.5 * self.theta * sigma_sq * (1.0 - qe.decay) * qe.decay_integral;

        if self.sigma <= DETERMINISTIC_SIGMA || mean <= 0.0 || var <= 0.0 {
            // Deterministic variance: integrate it with the trapezoidal rule.
            let v_next = mean.max(0.0);
            let integrated = 0.5 * (v + v_next) * self.dt;
            state.spot *= (self.drift - 0.5 * integrated + integrated.sqrt() * z_s).exp();
            state.variance = v_next;
            return;
        }

        let a_coeff = qe.k2 + 0.5 * qe.k4;
        let psi = var / (mean * mean);
        let (v_next, log_mgf) = if psi <= QE_PSI_CRITICAL {
            let inv = 2.0 / psi;
            let b_sq = inv - 1.0 + inv.sqrt() * (inv - 1.0).sqrt();
            let a = mean / (1.0 + b_sq);
            let b = b_sq.sqrt();
            let v_next = a * (b + z_v) * (b + z_v);
            // ln E[e^{A·v_{n+1}}], finite only for A < 1/(2a).
            let denom = 1.0 - 2.0 * a_coeff * a;
            let log_mgf = (denom > 0.0).then(|| a_coeff * b_sq * a / denom - 0.5 * denom.ln());
            (v_next, log_mgf)
        } else {
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / mean;
            let u = norm_cdf(z_v);
            let v_next = if u <= p {
                0.0
            } else {
                ((1.0 - p) / (1.0 - u)).ln() / beta
            };
            // ln E[e^{A·v_{n+1}}], finite only for A < β.
            let log_mgf = (a_coeff < beta).then(|| (p + beta * (1.0 - p) / (beta - a_coeff)).ln());
            (v_next, log_mgf)
        };

        // Martingale-corrected K₀* keeps E[S_{n+1} | S_n] = S_n e^{(r−q)Δ}; when
        // the moment generating function does not exist, keep Andersen's K₀.
        let k0 = log_mgf.map_or(qe.k0, |log_mgf| -log_mgf - (qe.k1 + 0.5 * qe.k3) * v);
        let diffusion = (qe.k3 * v + qe.k4 * v_next).max(0.0).sqrt();
        state.spot *= (self.drift + k0 + qe.k1 * v + qe.k2 * v_next + diffusion * z_s).exp();
        state.variance = v_next;
    }
}
