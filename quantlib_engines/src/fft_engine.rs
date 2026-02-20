use core::f64::consts::PI;
use core::str::FromStr;

use num_complex::Complex64;
use quantlib_core::{OptionType, PricingEngine, QuantLibError};
use quantlib_models::{BatesModel, BlackModel, BlackScholesModel, HestonModel, KouModel};
use quantlib_products::EuropeanOption;
use rustfft::FftPlanner;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationKind {
    Linear,
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

#[derive(Debug, Clone, Copy)]
pub struct FFTEngine {
    pub n: usize,
    pub b: f64,
    pub interp: InterpolationKind,
}

impl Default for FFTEngine {
    fn default() -> Self {
        Self {
            n: 1 << 12,
            b: 200.0,
            interp: InterpolationKind::Cubic,
        }
    }
}

impl FFTEngine {
    pub fn new(n: usize, b: f64, interp: InterpolationKind) -> Result<Self, QuantLibError> {
        if n < 8 {
            return Err(QuantLibError::Validation("n must be >= 8".to_owned()));
        }
        if !b.is_finite() || b <= 0.0 {
            return Err(QuantLibError::Validation("b must be > 0".to_owned()));
        }

        Ok(Self { n, b, interp })
    }

    fn intrinsic(option_type: OptionType, spot_or_forward: f64, strike: f64) -> f64 {
        match option_type {
            OptionType::Call => (spot_or_forward - strike).max(0.0),
            OptionType::Put => (strike - spot_or_forward).max(0.0),
        }
    }

    fn simpson_weight(index: usize, n: usize) -> f64 {
        if index == 0 || index + 1 == n {
            1.0
        } else if index.is_multiple_of(2) {
            2.0
        } else {
            4.0
        }
    }

    fn fft_integral_grid<F>(
        &self,
        characteristic_fn: F,
    ) -> Result<(Vec<f64>, Vec<f64>), QuantLibError>
    where
        F: Fn(Complex64) -> Complex64,
    {
        let n_as_f64 = self.n as f64;
        let dx = self.b / n_as_f64;
        let dk = 2.0 * PI / self.b;
        let shift = n_as_f64 * dk / 2.0;

        let mut integrand = vec![Complex64::new(0.0, 0.0); self.n];
        for (index, value) in integrand.iter_mut().enumerate() {
            let x = index as f64 * dx;
            let weight = Self::simpson_weight(index, self.n);
            let u = Complex64::new(x, -0.5);
            let phi = characteristic_fn(u);
            let kernel = phi / (x * x + 0.25);
            let phase = Complex64::new(0.0, -shift * x).exp();

            *value = phase * kernel * (weight * dx / 3.0);
        }

        let mut planner = FftPlanner::<f64>::new();
        let inverse_fft = planner.plan_fft_inverse(self.n);
        inverse_fft.process(&mut integrand);

        let mut ks = Vec::with_capacity(self.n);
        let mut values = Vec::with_capacity(self.n);
        for (index, item) in integrand.into_iter().enumerate() {
            ks.push(-shift + dk * index as f64);
            values.push(item.re);
        }

        Ok((ks, values))
    }

    fn black_scholes_characteristic(
        model: &BlackScholesModel,
        ttm: f64,
        u: Complex64,
    ) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let sigma_sq = model.sigma * model.sigma;
        let drift = (model.r - model.q - 0.5 * sigma_sq) * ttm;
        (i * u * drift - 0.5 * u * u * sigma_sq * ttm).exp()
    }

    fn black_characteristic(model: &BlackModel, forward: f64, ttm: f64, u: Complex64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let sigma_sq = model.sigma * model.sigma;
        (i * u * forward.ln() - 0.5 * u * u * sigma_sq * ttm).exp()
    }

    fn interpolate_at(&self, xs: &[f64], ys: &[f64], x: f64) -> Result<f64, QuantLibError> {
        if xs.len() != ys.len() || xs.len() < 2 {
            return Err(QuantLibError::Engine(
                "invalid interpolation grid".to_owned(),
            ));
        }

        if x < xs[0] || x > xs[xs.len() - 1] {
            return Err(QuantLibError::Engine(format!(
                "log-moneyness {x} is outside FFT interpolation domain [{}, {}]",
                xs[0],
                xs[xs.len() - 1]
            )));
        }

        if (x - xs[0]).abs() <= f64::EPSILON {
            return Ok(ys[0]);
        }
        if (x - xs[xs.len() - 1]).abs() <= f64::EPSILON {
            return Ok(ys[ys.len() - 1]);
        }

        match self.interp {
            InterpolationKind::Linear => linear_interpolate(xs, ys, x),
            InterpolationKind::Cubic => cubic_spline_interpolate(xs, ys, x),
        }
    }

    fn validate_price(price: f64) -> Result<f64, QuantLibError> {
        if !price.is_finite() {
            return Err(QuantLibError::Engine(
                "FFT pricing produced a non-finite result".to_owned(),
            ));
        }
        Ok(price)
    }
}

impl PricingEngine<EuropeanOption, HestonModel> for FFTEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let ttm = product.ttm();

        if ttm <= 0.0 {
            return Ok(
                Self::intrinsic(product.option_type, spot, product.k) * f64::from(product.qty)
            );
        }

        if product.k <= 0.0 {
            let unit_price = match product.option_type {
                OptionType::Call => spot * (-(model.q) * ttm).exp(),
                OptionType::Put => 0.0,
            };
            return Ok(unit_price * f64::from(product.qty));
        }

        let log_moneyness = (spot / product.k).ln();
        let (ks, integral_grid) =
            self.fft_integral_grid(|u| model.characteristic_function(u, ttm))?;
        let integral_value = self.interpolate_at(&ks, &integral_grid, log_moneyness)?;

        let discounted_spot = spot * (-(model.q) * ttm).exp();
        let discounted_strike = product.k * (-(model.r) * ttm).exp();

        let mut call_price = discounted_spot
            - (spot * product.k).sqrt() * (-(model.r) * ttm).exp() * integral_value / PI;

        if product.option_type == OptionType::Put {
            call_price += discounted_strike - discounted_spot;
        }

        Self::validate_price(call_price * f64::from(product.qty))
    }
}

impl PricingEngine<EuropeanOption, BatesModel> for FFTEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BatesModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let ttm = product.ttm();

        if ttm <= 0.0 {
            return Ok(
                Self::intrinsic(product.option_type, spot, product.k) * f64::from(product.qty)
            );
        }

        if product.k <= 0.0 {
            let unit_price = match product.option_type {
                OptionType::Call => spot * (-(model.q) * ttm).exp(),
                OptionType::Put => 0.0,
            };
            return Ok(unit_price * f64::from(product.qty));
        }

        let log_moneyness = (spot / product.k).ln();
        let (ks, integral_grid) =
            self.fft_integral_grid(|u| model.characteristic_function(u, ttm))?;
        let integral_value = self.interpolate_at(&ks, &integral_grid, log_moneyness)?;

        let discounted_spot = spot * (-(model.q) * ttm).exp();
        let discounted_strike = product.k * (-(model.r) * ttm).exp();

        let mut call_price = discounted_spot
            - (spot * product.k).sqrt() * (-(model.r) * ttm).exp() * integral_value / PI;

        if product.option_type == OptionType::Put {
            call_price += discounted_strike - discounted_spot;
        }

        Self::validate_price(call_price * f64::from(product.qty))
    }
}

impl PricingEngine<EuropeanOption, KouModel> for FFTEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &KouModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let ttm = product.ttm();

        if ttm <= 0.0 {
            return Ok(
                Self::intrinsic(product.option_type, spot, product.k) * f64::from(product.qty)
            );
        }

        if product.k <= 0.0 {
            let unit_price = match product.option_type {
                OptionType::Call => spot * (-(model.q) * ttm).exp(),
                OptionType::Put => 0.0,
            };
            return Ok(unit_price * f64::from(product.qty));
        }

        let log_moneyness = (spot / product.k).ln();
        let (ks, integral_grid) =
            self.fft_integral_grid(|u| model.characteristic_function(u, ttm))?;
        let integral_value = self.interpolate_at(&ks, &integral_grid, log_moneyness)?;

        let discounted_spot = spot * (-(model.q) * ttm).exp();
        let discounted_strike = product.k * (-(model.r) * ttm).exp();

        let mut call_price = discounted_spot
            - (spot * product.k).sqrt() * (-(model.r) * ttm).exp() * integral_value / PI;

        if product.option_type == OptionType::Put {
            call_price += discounted_strike - discounted_spot;
        }

        Self::validate_price(call_price * f64::from(product.qty))
    }
}

impl PricingEngine<EuropeanOption, BlackScholesModel> for FFTEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let ttm = product.ttm();

        if ttm <= 0.0 {
            return Ok(
                Self::intrinsic(product.option_type, spot, product.k) * f64::from(product.qty)
            );
        }

        if product.k <= 0.0 {
            let unit_price = match product.option_type {
                OptionType::Call => spot * (-(model.q) * ttm).exp(),
                OptionType::Put => 0.0,
            };
            return Ok(unit_price * f64::from(product.qty));
        }

        let log_moneyness = (spot / product.k).ln();
        let (ks, integral_grid) =
            self.fft_integral_grid(|u| Self::black_scholes_characteristic(model, ttm, u))?;
        let integral_value = self.interpolate_at(&ks, &integral_grid, log_moneyness)?;

        let discounted_spot = spot * (-(model.q) * ttm).exp();
        let discounted_strike = product.k * (-(model.r) * ttm).exp();

        let mut call_price = discounted_spot
            - (spot * product.k).sqrt() * (-(model.r) * ttm).exp() * integral_value / PI;

        if product.option_type == OptionType::Put {
            call_price += discounted_strike - discounted_spot;
        }

        Self::validate_price(call_price * f64::from(product.qty))
    }
}

impl PricingEngine<EuropeanOption, BlackModel> for FFTEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let forward = product.require_forward()?;
        let ttm = product.ttm();

        if ttm <= 0.0 {
            return Ok(
                Self::intrinsic(product.option_type, forward, product.k) * f64::from(product.qty)
            );
        }

        if product.k <= 0.0 {
            let unit_price = match product.option_type {
                OptionType::Call => forward * (-(model.r) * ttm).exp(),
                OptionType::Put => 0.0,
            };
            return Ok(unit_price * f64::from(product.qty));
        }

        let log_moneyness = (forward / product.k).ln();
        let (ks, integral_grid) =
            self.fft_integral_grid(|u| Self::black_characteristic(model, forward, ttm, u))?;
        let integral_value = self.interpolate_at(&ks, &integral_grid, log_moneyness)?;

        let discount = (-(model.r) * ttm).exp();
        let mut call_price =
            discount * (forward - (forward * product.k).sqrt() * integral_value / PI);

        if product.option_type == OptionType::Put {
            call_price += discount * (product.k - forward);
        }

        Self::validate_price(call_price * f64::from(product.qty))
    }
}

fn linear_interpolate(xs: &[f64], ys: &[f64], x: f64) -> Result<f64, QuantLibError> {
    let idx = locate_interval(xs, x)?;
    let x0 = xs[idx];
    let x1 = xs[idx + 1];
    let y0 = ys[idx];
    let y1 = ys[idx + 1];

    let width = x1 - x0;
    if width <= 0.0 {
        return Err(QuantLibError::Engine(
            "FFT interpolation grid must be strictly increasing".to_owned(),
        ));
    }

    Ok(y0 + (x - x0) / width * (y1 - y0))
}

fn cubic_spline_interpolate(xs: &[f64], ys: &[f64], x: f64) -> Result<f64, QuantLibError> {
    let n = xs.len();
    if n < 3 {
        return linear_interpolate(xs, ys, x);
    }

    let mut h = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        h[i] = xs[i + 1] - xs[i];
        if h[i] <= 0.0 {
            return Err(QuantLibError::Engine(
                "FFT interpolation grid must be strictly increasing".to_owned(),
            ));
        }
    }

    let mut alpha = vec![0.0; n];
    for i in 1..(n - 1) {
        alpha[i] = 3.0 / h[i] * (ys[i + 1] - ys[i]) - 3.0 / h[i - 1] * (ys[i] - ys[i - 1]);
    }

    let mut l = vec![0.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];
    l[0] = 1.0;

    for i in 1..(n - 1) {
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
        if l[i].abs() <= 1.0e-14 {
            return Err(QuantLibError::Engine(
                "cubic spline system is ill-conditioned".to_owned(),
            ));
        }

        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n - 1] = 1.0;

    let mut b = vec![0.0; n - 1];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n - 1];

    for j in (0..=(n - 2)).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    let idx = locate_interval(xs, x)?;
    let dx = x - xs[idx];

    Ok(ys[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx)
}

fn locate_interval(xs: &[f64], x: f64) -> Result<usize, QuantLibError> {
    if xs.len() < 2 {
        return Err(QuantLibError::Engine(
            "FFT interpolation grid requires at least 2 points".to_owned(),
        ));
    }

    let idx = match xs.binary_search_by(|value| value.total_cmp(&x)) {
        Ok(exact) => exact.min(xs.len() - 2),
        Err(insert) => insert.saturating_sub(1).min(xs.len() - 2),
    };

    Ok(idx)
}
