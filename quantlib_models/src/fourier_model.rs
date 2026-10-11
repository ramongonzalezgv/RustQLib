use num_complex::Complex64;
use quantlib_core::QuantLibError;

use crate::{BatesModel, BlackModel, BlackScholesModel, HestonModel, KouModel};

/// First, second and fourth cumulants of `ln(S_T / S_0)`.
///
/// Fourier engines use them to size truncation ranges (e.g. the COS interval).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cumulants {
    pub c1: f64,
    pub c2: f64,
    pub c4: f64,
}

/// A model priced through the characteristic function of its log-return.
///
/// Forward-quoted models (Black-76) set [`FourierModel::FORWARD_QUOTED`] and
/// describe `ln(F_T / F_0)` with `q = r`, so engines treat the forward as the
/// spot without a separate code path.
pub trait FourierModel {
    /// Whether the model expects a forward-quoted product instead of a spot one.
    const FORWARD_QUOTED: bool = false;

    /// CF of `ln(S_T / S_0)` under the risk-neutral measure (includes the `(r − q)T` drift).
    fn log_return_cf(&self, u: Complex64, ttm: f64) -> Complex64;
    fn rate(&self) -> f64;
    fn dividend_yield(&self) -> f64;
    fn validate(&self) -> Result<(), QuantLibError>;

    /// Cumulants of `ln(S_T / S_0)`. The default differentiates the cumulant
    /// generating function `ln φ(−it)` numerically at `t = 0`.
    fn cumulants(&self, ttm: f64) -> Cumulants {
        numerical_cumulants(|u| self.log_return_cf(u, ttm))
    }
}

/// Central finite differences of `K(t) = ln E[e^{tX}] = ln φ(−it)` at `t = 0`.
pub fn numerical_cumulants<F>(cf: F) -> Cumulants
where
    F: Fn(Complex64) -> Complex64,
{
    let cgf = |t: f64| cf(Complex64::new(0.0, -t)).re.ln();
    let k0 = cgf(0.0);

    let h = 1.0e-3;
    let (kp, km) = (cgf(h), cgf(-h));
    let c1 = (kp - km) / (2.0 * h);
    let c2 = (kp - 2.0 * k0 + km) / (h * h);

    let h4 = 5.0e-2;
    let c4 =
        (cgf(2.0 * h4) - 4.0 * cgf(h4) + 6.0 * k0 - 4.0 * cgf(-h4) + cgf(-2.0 * h4)) / h4.powi(4);

    Cumulants { c1, c2, c4 }
}

fn black_scholes_cf(sigma: f64, drift: f64, u: Complex64, ttm: f64) -> Complex64 {
    let i = Complex64::new(0.0, 1.0);
    let sigma_sq = sigma * sigma;
    (i * u * (drift - 0.5 * sigma_sq) * ttm - 0.5 * u * u * sigma_sq * ttm).exp()
}

fn black_scholes_cumulants(sigma: f64, drift: f64, ttm: f64) -> Cumulants {
    let sigma_sq = sigma * sigma;
    Cumulants {
        c1: (drift - 0.5 * sigma_sq) * ttm,
        c2: sigma_sq * ttm,
        c4: 0.0,
    }
}

impl FourierModel for BlackScholesModel {
    fn log_return_cf(&self, u: Complex64, ttm: f64) -> Complex64 {
        black_scholes_cf(self.sigma, self.r - self.q, u, ttm)
    }

    fn rate(&self) -> f64 {
        self.r
    }

    fn dividend_yield(&self) -> f64 {
        self.q
    }

    fn validate(&self) -> Result<(), QuantLibError> {
        BlackScholesModel::validate(self)
    }

    fn cumulants(&self, ttm: f64) -> Cumulants {
        black_scholes_cumulants(self.sigma, self.r - self.q, ttm)
    }
}

/// Black-76: Black–Scholes on the forward with `q = r`.
impl FourierModel for BlackModel {
    const FORWARD_QUOTED: bool = true;

    fn log_return_cf(&self, u: Complex64, ttm: f64) -> Complex64 {
        black_scholes_cf(self.sigma, 0.0, u, ttm)
    }

    fn rate(&self) -> f64 {
        self.r
    }

    fn dividend_yield(&self) -> f64 {
        self.r
    }

    fn validate(&self) -> Result<(), QuantLibError> {
        BlackModel::validate(self)
    }

    fn cumulants(&self, ttm: f64) -> Cumulants {
        black_scholes_cumulants(self.sigma, 0.0, ttm)
    }
}

impl FourierModel for HestonModel {
    fn log_return_cf(&self, u: Complex64, ttm: f64) -> Complex64 {
        self.characteristic_function(u, ttm)
    }

    fn rate(&self) -> f64 {
        self.r
    }

    fn dividend_yield(&self) -> f64 {
        self.q
    }

    fn validate(&self) -> Result<(), QuantLibError> {
        HestonModel::validate(self)
    }
}

impl FourierModel for BatesModel {
    fn log_return_cf(&self, u: Complex64, ttm: f64) -> Complex64 {
        self.characteristic_function(u, ttm)
    }

    fn rate(&self) -> f64 {
        self.r
    }

    fn dividend_yield(&self) -> f64 {
        self.q
    }

    fn validate(&self) -> Result<(), QuantLibError> {
        BatesModel::validate(self)
    }
}

impl FourierModel for KouModel {
    fn log_return_cf(&self, u: Complex64, ttm: f64) -> Complex64 {
        self.characteristic_function(u, ttm)
    }

    fn rate(&self) -> f64 {
        self.r
    }

    fn dividend_yield(&self) -> f64 {
        self.q
    }

    fn validate(&self) -> Result<(), QuantLibError> {
        KouModel::validate(self)
    }

    /// Closed form: the n-th cumulant of a compound Poisson process is `λT·E[Jⁿ]`,
    /// and for the double exponential `E[Jⁿ] = n!·(p/η₁ⁿ + (−1)ⁿ(1−p)/η₂ⁿ)`.
    fn cumulants(&self, ttm: f64) -> Cumulants {
        let (p, eta1, eta2) = (self.p_up, self.eta1, self.eta2);
        let jump_moment = |n: i32, factorial: f64| {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            factorial * (p / eta1.powi(n) + sign * (1.0 - p) / eta2.powi(n))
        };
        let lambda_t = self.lambda_jump * ttm;
        let jump_mean = p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0;
        let sigma_sq = self.sigma * self.sigma;

        Cumulants {
            c1: (self.r - self.q - self.lambda_jump * jump_mean - 0.5 * sigma_sq) * ttm
                + lambda_t * jump_moment(1, 1.0),
            c2: sigma_sq * ttm + lambda_t * jump_moment(2, 2.0),
            c4: lambda_t * jump_moment(4, 24.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FourierModel, numerical_cumulants};
    use crate::{BlackScholesModel, KouModel};

    fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn numerical_cumulants_match_black_scholes_closed_form() {
        let model = BlackScholesModel::new(0.25, 0.03, 0.01).expect("model");
        let exact = model.cumulants(1.5);
        let numeric = numerical_cumulants(|u| model.log_return_cf(u, 1.5));
        assert_close(numeric.c1, exact.c1, 1.0e-9, "c1");
        assert_close(numeric.c2, exact.c2, 1.0e-6, "c2");
        assert_close(numeric.c4, exact.c4, 1.0e-6, "c4");
    }

    #[test]
    fn kou_closed_form_cumulants_match_numerical() {
        let model = KouModel::new(0.2, 0.03, 0.01, 0.6, 0.4, 9.0, 6.0).expect("model");
        let exact = model.cumulants(1.3);
        let numeric = numerical_cumulants(|u| model.log_return_cf(u, 1.3));
        assert_close(exact.c1, numeric.c1, 1.0e-8, "c1");
        assert_close(exact.c2, numeric.c2, 1.0e-6, "c2");
        assert_close(exact.c4, numeric.c4, 1.0e-5, "c4");
    }
}
