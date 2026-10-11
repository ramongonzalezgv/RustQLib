use core::f64::consts::PI;

use num_complex::Complex64;
use quantlib_core::{OptionType, PricingEngine, QuantLibError};
use quantlib_models::FourierModel;
use quantlib_products::EuropeanOption;

use crate::fourier::{FourierStrip, price_european, validate_price, validate_strip};

/// Fang & Oosterlee (2008) COS method.
///
/// Puts are priced from the cosine expansion of the log-return density (their
/// payoff is bounded, so truncation is benign) and calls follow from put–call
/// parity. The truncation interval is `c₁ ± L·√(c₂ + √c₄)` around each strike's
/// log-moneyness; its width does not depend on the strike, so a strip evaluates
/// the characteristic function once and only rebuilds the payoff coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosEngine {
    /// Number of cosine terms.
    pub n: usize,
    /// Truncation range multiplier.
    pub l: f64,
}

impl Default for CosEngine {
    fn default() -> Self {
        Self { n: 256, l: 12.0 }
    }
}

impl CosEngine {
    pub fn new(n: usize, l: f64) -> Result<Self, QuantLibError> {
        let engine = Self { n, l };
        engine.validate()?;
        Ok(engine)
    }

    fn validate(&self) -> Result<(), QuantLibError> {
        if self.n < 2 {
            return Err(QuantLibError::Validation("n must be >= 2".to_owned()));
        }
        if !self.l.is_finite() || self.l <= 0.0 {
            return Err(QuantLibError::Validation("l must be > 0".to_owned()));
        }
        Ok(())
    }

    /// Unit prices for many strikes at one exact maturity. For forward-quoted
    /// models (Black-76) `spot` is the forward.
    pub fn price_strip<M: FourierModel>(
        &self,
        model: &M,
        spot: f64,
        ttm: f64,
        strikes: &[f64],
        types: &[OptionType],
    ) -> Result<Vec<f64>, QuantLibError> {
        self.validate()?;
        validate_strip(model, spot, ttm, strikes, types)?;
        self.strip(model, spot, ttm, strikes, types)
    }
}

impl FourierStrip for CosEngine {
    fn strip<M: FourierModel>(
        &self,
        model: &M,
        spot: f64,
        ttm: f64,
        strikes: &[f64],
        types: &[OptionType],
    ) -> Result<Vec<f64>, QuantLibError> {
        let cumulants = model.cumulants(ttm);
        let half_width = self.l * (cumulants.c2 + cumulants.c4.max(0.0).sqrt()).sqrt();
        if !half_width.is_finite() || half_width <= 0.0 || !cumulants.c1.is_finite() {
            return Err(QuantLibError::Engine(format!(
                "COS truncation range is degenerate (cumulants {cumulants:?})"
            )));
        }
        let width = 2.0 * half_width;
        let omega = |k: usize| k as f64 * PI / width;

        // Re[φ(ω_k) e^{iω_k(x−a)}]. With a = x + c₁ − w the phase argument
        // x − a = w − c₁ is the same for every strike.
        let shift = half_width - cumulants.c1;
        let mut coefficients: Vec<f64> = (0..self.n)
            .map(|k| {
                let u = Complex64::new(omega(k), 0.0);
                (model.log_return_cf(u, ttm) * Complex64::new(0.0, u.re * shift).exp()).re
            })
            .collect();
        coefficients[0] *= 0.5;

        let discount = (-model.rate() * ttm).exp();
        let discounted_spot = spot * (-model.dividend_yield() * ttm).exp();

        let mut prices = Vec::with_capacity(strikes.len());
        for (&strike, &option_type) in strikes.iter().zip(types) {
            let a = (spot / strike).ln() + cumulants.c1 - half_width;
            let put = if a >= 0.0 {
                // S_T > K on the whole interval.
                0.0
            } else if a + width <= 0.0 {
                // S_T < K on the whole interval: the put is a short forward. Using
                // the exact forward instead of the series keeps truncated tail
                // mass (e.g. rare jumps at short maturities) out of the price.
                strike * discount - discounted_spot
            } else {
                strike * discount * put_series(&coefficients, a, width, omega)
            };
            let price = match option_type {
                OptionType::Put => put,
                OptionType::Call => put + discounted_spot - strike * discount,
            };
            // Parity can leave round-off below zero for far-OTM calls.
            prices.push(validate_price(price.max(0.0), "COS")?);
        }
        Ok(prices)
    }
}

/// `Σ' A_k U_k` for the put payoff `(1 − e^y)⁺` on `y ∈ [a, a + width]` with
/// `a < 0 < b`, where `U_k = 2/(b−a)·(ψ_k(a, 0) − χ_k(a, 0))`.
fn put_series(coefficients: &[f64], a: f64, width: f64, omega: impl Fn(usize) -> f64) -> f64 {
    let exp_a = a.exp();
    // cos/sin(ω_k (0 − a)) by rotating e^{iθ} instead of calling trig per term.
    let step = Complex64::from_polar(1.0, -PI * a / width);
    let mut rotation = Complex64::new(1.0, 0.0);

    let mut sum = 0.0;
    for (k, &coefficient) in coefficients.iter().enumerate() {
        let w = omega(k);
        let (cos, sin) = (rotation.re, rotation.im);
        let chi = (cos - exp_a + w * sin) / (1.0 + w * w);
        let psi = if k == 0 { -a } else { sin / w };
        sum += coefficient * (psi - chi);
        rotation *= step;
    }

    2.0 / width * sum
}

impl<M: FourierModel> PricingEngine<EuropeanOption, M> for CosEngine {
    fn calculate_price(&self, product: &EuropeanOption, model: &M) -> Result<f64, QuantLibError> {
        self.validate()?;
        price_european(self, product, model)
    }
}
