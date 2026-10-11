use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use quantlib_core::QuantLibError;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) const DEFAULT_PATH_CHUNK_SIZE: usize = 4_096;

/// Discretization of the Heston variance process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HestonScheme {
    /// Euler with full truncation (Lord, Koekkoek & van Dijk, 2010): the
    /// variance is never floored, only its positive part enters drift and
    /// diffusion. Needs fine time steps.
    FullTruncation,
    /// Andersen (2008) quadratic-exponential scheme with martingale
    /// correction. Low bias even with 10–30 steps per year.
    #[default]
    QuadraticExponential,
}

impl FromStr for HestonScheme {
    type Err = QuantLibError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "full_truncation" | "ft" => Ok(Self::FullTruncation),
            "quadratic_exponential" | "qe" => Ok(Self::QuadraticExponential),
            _ => Err(QuantLibError::Validation(
                "heston_scheme must be 'qe' (quadratic_exponential) or 'full_truncation'"
                    .to_owned(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MonteCarloConfig {
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: Option<u64>,
    pub parallel: bool,
    /// Paths per independent random stream (and per parallel task). Results
    /// depend on it, but not on the number of threads.
    pub chunk_size: usize,
    pub heston_scheme: HestonScheme,
    /// Pair every normal vector `Z` with `−Z`; each pair is one sample.
    pub antithetic: bool,
    /// Use the product's control variate (see [`super::MonteCarloEngine`]).
    pub control_variate: bool,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            n_paths: 10_000,
            n_steps: 252,
            seed: None,
            parallel: true,
            chunk_size: DEFAULT_PATH_CHUNK_SIZE,
            heston_scheme: HestonScheme::default(),
            antithetic: false,
            control_variate: false,
        }
    }
}

impl MonteCarloConfig {
    pub fn new(n_paths: usize, n_steps: usize, seed: Option<u64>) -> Result<Self, QuantLibError> {
        let config = Self {
            n_paths,
            n_steps,
            seed,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if self.n_paths == 0 {
            return Err(QuantLibError::Validation("n_paths must be >= 1".to_owned()));
        }
        if self.n_steps == 0 {
            return Err(QuantLibError::Validation("n_steps must be >= 1".to_owned()));
        }
        if self.chunk_size == 0 {
            return Err(QuantLibError::Validation(
                "chunk_size must be >= 1".to_owned(),
            ));
        }
        Ok(())
    }

    /// Independent samples: paths, or antithetic pairs of paths.
    pub(crate) fn n_samples(&self) -> usize {
        if self.antithetic {
            self.n_paths.div_ceil(2)
        } else {
            self.n_paths
        }
    }

    /// Paths actually simulated (an odd antithetic count is rounded up).
    pub(crate) fn simulated_paths(&self) -> usize {
        if self.antithetic {
            2 * self.n_samples()
        } else {
            self.n_paths
        }
    }

    pub(crate) fn resolved_seed(&self) -> u64 {
        self.seed.unwrap_or_else(default_seed)
    }
}

/// The random stream of chunk `chunk`: Xoshiro256++ seeded from a SplitMix64
/// mix of `(seed, chunk)`, so chunks are independent and reproducible
/// whatever thread runs them.
pub(crate) fn chunk_rng(seed: u64, chunk: usize) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(splitmix64(seed ^ splitmix64(chunk as u64)))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Monte Carlo price with its standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McEstimate {
    pub price: f64,
    /// Standard error of `price` (NaN with fewer than two samples).
    pub std_error: f64,
    /// Paths simulated (both legs of antithetic pairs count).
    pub n_paths: usize,
}

/// One sample: the payoff `y` and the control variate `x`, both undiscounted.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Sample {
    pub y: f64,
    pub x: f64,
}

impl Sample {
    pub(crate) fn mean(self, other: Self) -> Self {
        Self {
            y: 0.5 * (self.y + other.y),
            x: 0.5 * (self.x + other.x),
        }
    }
}

/// Means and (co)moments of `(y, x)`, mergeable across chunks (Chan et al.).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Moments {
    n: f64,
    mean_y: f64,
    mean_x: f64,
    m2_y: f64,
    m2_x: f64,
    c_xy: f64,
}

/// Division-free accumulator for one chunk: sums of `(y, x)` shifted by the
/// chunk's first sample, which keeps the sums of squares well conditioned.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ChunkSums {
    n: f64,
    shift: Sample,
    sum_y: f64,
    sum_x: f64,
    sum_yy: f64,
    sum_xx: f64,
    sum_xy: f64,
}

impl ChunkSums {
    #[inline]
    pub(crate) fn push(&mut self, sample: Sample) {
        if self.n == 0.0 {
            self.shift = sample;
        }
        let (dy, dx) = (sample.y - self.shift.y, sample.x - self.shift.x);
        self.n += 1.0;
        self.sum_y += dy;
        self.sum_x += dx;
        self.sum_yy += dy * dy;
        self.sum_xx += dx * dx;
        self.sum_xy += dy * dx;
    }

    pub(crate) fn moments(&self) -> Moments {
        if self.n == 0.0 {
            return Moments::default();
        }
        let (mean_dy, mean_dx) = (self.sum_y / self.n, self.sum_x / self.n);
        Moments {
            n: self.n,
            mean_y: self.shift.y + mean_dy,
            mean_x: self.shift.x + mean_dx,
            m2_y: (self.sum_yy - self.sum_y * mean_dy).max(0.0),
            m2_x: (self.sum_xx - self.sum_x * mean_dx).max(0.0),
            c_xy: self.sum_xy - self.sum_y * mean_dx,
        }
    }
}

impl Moments {
    pub(crate) fn merge(self, other: Self) -> Self {
        if self.n == 0.0 {
            return other;
        }
        if other.n == 0.0 {
            return self;
        }
        let n = self.n + other.n;
        let dy = other.mean_y - self.mean_y;
        let dx = other.mean_x - self.mean_x;
        let weight = self.n * other.n / n;
        Self {
            n,
            mean_y: self.mean_y + dy * other.n / n,
            mean_x: self.mean_x + dx * other.n / n,
            m2_y: self.m2_y + other.m2_y + dy * dy * weight,
            m2_x: self.m2_x + other.m2_x + dx * dx * weight,
            c_xy: self.c_xy + other.c_xy + dy * dx * weight,
        }
    }

    /// Discounted mean and standard error, with the control variate applied
    /// when its expectation is given: `ȳ − β(x̄ − E[X])`, `β = Cov(y, x)/Var(x)`.
    pub(crate) fn estimate(
        &self,
        discount: f64,
        control_mean: Option<f64>,
        n_paths: usize,
    ) -> McEstimate {
        let (mean, m2) = match control_mean {
            Some(expected) if self.m2_x > 0.0 => {
                let beta = self.c_xy / self.m2_x;
                (
                    self.mean_y - beta * (self.mean_x - expected),
                    (self.m2_y - beta * self.c_xy).max(0.0),
                )
            }
            _ => (self.mean_y, self.m2_y.max(0.0)),
        };
        let std_error = if self.n >= 2.0 {
            (m2 / (self.n - 1.0) / self.n).sqrt()
        } else {
            f64::NAN
        };
        McEstimate {
            price: discount * mean,
            std_error: discount * std_error,
            n_paths,
        }
    }
}

fn default_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos() as u64,
        Err(err) => err.duration().as_nanos() as u64,
    }
}
