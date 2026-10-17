use core::f64::consts::PI;
use core::str::FromStr;
use std::fmt;
use std::sync::Arc;

use num_complex::Complex64;
use quantlib_core::{OptionType, PricingEngine, QuantLibError};
use quantlib_models::FourierModel;
use quantlib_products::EuropeanOption;
use rustfft::{Fft, FftPlanner};

use crate::fourier::{FourierStrip, price_european, validate_price, validate_strip};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationKind {
    Linear,
    /// Local high-order (8-point) Lagrange on the uniform log-strike grid. It is
    /// O(1) per strike and more accurate than the global cubic spline it replaced.
    #[default]
    Cubic,
}

impl FromStr for InterpolationKind {
    type Err = QuantLibError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "cubic" => Ok(Self::Cubic),
            _ => Err(QuantLibError::Validation(
                "interp must be 'linear' or 'cubic'".to_owned(),
            )),
        }
    }
}

/// Points in the local Lagrange stencil of [`InterpolationKind::Cubic`].
const STENCIL: usize = 8;

/// Lewis (2001) call price on an inverse-FFT log-strike grid.
///
/// The FFT plan and every model-independent quadrature array are built once in
/// [`FFTEngine::new`] and shared between clones.
#[derive(Clone)]
pub struct FFTEngine {
    n: usize,
    b: f64,
    interp: InterpolationKind,
    grid: Arc<FftGrid>,
}

struct FftGrid {
    fft: Arc<dyn Fft<f64>>,
    /// CF arguments `x_j − i/2`.
    nodes: Vec<Complex64>,
    /// `(−1)^j · w_j·dx/3 / (x_j² + ¼)`: phase `e^{−i·shift·x_j}`, Simpson weight and kernel.
    factors: Vec<f64>,
    /// Log-strike of the first output point (`−shift`) and grid spacing.
    k0: f64,
    dk: f64,
}

impl fmt::Debug for FFTEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FFTEngine")
            .field("n", &self.n)
            .field("b", &self.b)
            .field("interp", &self.interp)
            .finish()
    }
}

impl Default for FFTEngine {
    fn default() -> Self {
        Self::new(
            Self::DEFAULT_N,
            Self::DEFAULT_B,
            InterpolationKind::default(),
        )
        .expect("default FFT settings are valid")
    }
}

impl FFTEngine {
    pub const DEFAULT_N: usize = 1 << 12;
    pub const DEFAULT_B: f64 = 200.0;

    pub fn new(n: usize, b: f64, interp: InterpolationKind) -> Result<Self, QuantLibError> {
        if n < 8 {
            return Err(QuantLibError::Validation("n must be >= 8".to_owned()));
        }
        if !b.is_finite() || b <= 0.0 {
            return Err(QuantLibError::Validation("b must be > 0".to_owned()));
        }

        let dx = b / n as f64;
        let dk = 2.0 * PI / b;
        // shift·x_j = π·j, so the phase factor is exactly (−1)^j.
        let (nodes, factors) = (0..n)
            .map(|j| {
                let x = j as f64 * dx;
                // Carr–Madan Simpson weights, 0-indexed: 3 − (−1)^j − δ_{j0}
                // = 1, 4, 2, 4, …, 2, 4 (divided by 3 below).
                let (sign, simpson) = match j {
                    0 => (1.0, 1.0),
                    _ if j % 2 == 1 => (-1.0, 4.0),
                    _ => (1.0, 2.0),
                };
                let factor = sign * simpson * dx / 3.0 / (x * x + 0.25);
                (Complex64::new(x, -0.5), factor)
            })
            .unzip();

        let grid = FftGrid {
            fft: FftPlanner::<f64>::new().plan_fft_inverse(n),
            nodes,
            factors,
            k0: -(n as f64) * dk / 2.0,
            dk,
        };

        Ok(Self {
            n,
            b,
            interp,
            grid: Arc::new(grid),
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn b(&self) -> f64 {
        self.b
    }

    pub fn interp(&self) -> InterpolationKind {
        self.interp
    }

    /// Unit prices for many strikes at one exact maturity: one CF sweep and one
    /// FFT, then an O(1) interpolation per strike. For forward-quoted models
    /// (Black-76) `spot` is the forward.
    pub fn price_strip<M: FourierModel>(
        &self,
        model: &M,
        spot: f64,
        ttm: f64,
        strikes: &[f64],
        types: &[OptionType],
    ) -> Result<Vec<f64>, QuantLibError> {
        validate_strip(model, spot, ttm, strikes, types)?;
        self.strip(model, spot, ttm, strikes, types)
    }

    /// Lewis integral `∫ Re[e^{iuk} φ(u − i/2)] / (u² + ¼) du`, `k = ln(S/K)`, on the log-strike grid.
    fn integral_grid<M: FourierModel>(&self, model: &M, ttm: f64) -> Vec<f64> {
        let grid = &*self.grid;
        let fft = &grid.fft;
        let mut buffer = vec![Complex64::new(0.0, 0.0); self.n + fft.get_inplace_scratch_len()];
        let (values, scratch) = buffer.split_at_mut(self.n);

        for ((value, &u), &factor) in values.iter_mut().zip(&grid.nodes).zip(&grid.factors) {
            *value = model.log_return_cf(u, ttm) * factor;
        }
        fft.process_with_scratch(values, scratch);

        buffer.truncate(self.n);
        buffer.into_iter().map(|value| value.re).collect()
    }

    /// Interpolates the integral grid at log-moneyness `k`; allocation-free and O(1).
    fn interpolate(&self, values: &[f64], k: f64) -> Result<f64, QuantLibError> {
        let FftGrid { k0, dk, .. } = *self.grid;
        let k_max = k0 + dk * (self.n - 1) as f64;
        if !(k0..=k_max).contains(&k) {
            return Err(QuantLibError::Engine(format!(
                "log-moneyness {k} is outside FFT interpolation domain [{k0}, {k_max}]"
            )));
        }

        let position = (k - k0) / dk;
        match self.interp {
            InterpolationKind::Linear => {
                let i = (position.floor() as usize).min(self.n - 2);
                let t = position - i as f64;
                Ok(values[i] + t * (values[i + 1] - values[i]))
            }
            InterpolationKind::Cubic => {
                // Nodes centred on the bracketing interval, clamped inside the grid.
                let i =
                    (position.floor() as usize).clamp(STENCIL / 2 - 1, self.n - 1 - STENCIL / 2);
                let start = i + 1 - STENCIL / 2;
                let t = position - start as f64;
                let mut value = 0.0;
                for a in 0..STENCIL {
                    let mut weight = 1.0;
                    for b in (0..STENCIL).filter(|&b| b != a) {
                        weight *= (t - b as f64) / (a as f64 - b as f64);
                    }
                    value += weight * values[start + a];
                }
                Ok(value)
            }
        }
    }
}

impl FourierStrip for FFTEngine {
    fn strip<M: FourierModel>(
        &self,
        model: &M,
        spot: f64,
        ttm: f64,
        strikes: &[f64],
        types: &[OptionType],
    ) -> Result<Vec<f64>, QuantLibError> {
        let values = self.integral_grid(model, ttm);
        let discount = (-model.rate() * ttm).exp();
        let discounted_spot = spot * (-model.dividend_yield() * ttm).exp();

        // Sized up front: a `Result` collect cannot pre-size and would regrow.
        let mut prices = Vec::with_capacity(strikes.len());
        for (&strike, &option_type) in strikes.iter().zip(types) {
            let integral = self.interpolate(&values, (spot / strike).ln())?;
            let call = discounted_spot - (spot * strike).sqrt() * discount * integral / PI;
            let price = match option_type {
                OptionType::Call => call,
                OptionType::Put => call + strike * discount - discounted_spot,
            };
            prices.push(validate_price(price, "FFT")?);
        }
        Ok(prices)
    }
}

impl<M: FourierModel> PricingEngine<EuropeanOption, M> for FFTEngine {
    fn calculate_price(&self, product: &EuropeanOption, model: &M) -> Result<f64, QuantLibError> {
        price_european(self, product, model)
    }
}
