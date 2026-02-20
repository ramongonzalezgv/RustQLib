use std::time::{SystemTime, UNIX_EPOCH};

use quantlib_core::QuantLibError;

pub(crate) const DEFAULT_PATH_CHUNK_SIZE: usize = 4_096;

#[derive(Debug, Clone, Copy)]
pub struct MonteCarloConfig {
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: Option<u64>,
    pub parallel: bool,
    pub chunk_size: usize,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            n_paths: 10_000,
            n_steps: 252,
            seed: None,
            parallel: true,
            chunk_size: DEFAULT_PATH_CHUNK_SIZE,
        }
    }
}

impl MonteCarloConfig {
    pub fn new(n_paths: usize, n_steps: usize, seed: Option<u64>) -> Result<Self, QuantLibError> {
        let config = Self {
            n_paths,
            n_steps,
            seed,
            parallel: true,
            chunk_size: DEFAULT_PATH_CHUNK_SIZE,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
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

    pub(crate) fn rng(&self) -> NormalRng {
        let seed = self.seed.unwrap_or_else(default_seed);
        NormalRng::new(seed)
    }
}

pub(crate) fn discount_and_average(
    payoffs: &[f64],
    rate: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    if payoffs.is_empty() {
        return Err(QuantLibError::Engine(
            "empty payoff vector in MonteCarloEngine".to_owned(),
        ));
    }

    let mean = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
    Ok((-rate * ttm).exp() * mean)
}

fn default_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos() as u64,
        Err(err) => err.duration().as_nanos() as u64,
    }
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let adjusted = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state: adjusted }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64_open01(&mut self) -> f64 {
        let mut u = (self.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
        if u <= f64::EPSILON {
            u = f64::EPSILON;
        }
        if u >= 1.0 {
            u = 1.0 - f64::EPSILON;
        }
        u
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NormalRng {
    uniform: XorShift64,
    spare: Option<f64>,
}

impl NormalRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            uniform: XorShift64::new(seed),
            spare: None,
        }
    }

    pub(crate) fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }

        let u1 = self.uniform.next_f64_open01();
        let u2 = self.uniform.next_f64_open01();

        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;

        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.spare = Some(z1);
        z0
    }
}
