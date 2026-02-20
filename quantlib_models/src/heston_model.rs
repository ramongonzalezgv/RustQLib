use std::collections::BTreeMap;

use num_complex::Complex64;
use quantlib_core::{QuantLibError, StochasticModel};

#[derive(Debug, Clone, Copy)]
pub struct HestonModel {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub rho: f64,
    pub v0: f64,
    pub r: f64,
    pub q: f64,
}

impl HestonModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
        r: f64,
        q: f64,
    ) -> Result<Self, QuantLibError> {
        let model = Self {
            kappa,
            theta,
            sigma,
            rho,
            v0,
            r,
            q,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.kappa.is_finite()
            || !self.theta.is_finite()
            || !self.sigma.is_finite()
            || !self.rho.is_finite()
            || !self.v0.is_finite()
            || !self.r.is_finite()
            || !self.q.is_finite()
        {
            return Err(QuantLibError::Validation(
                "Heston parameters must be finite".to_owned(),
            ));
        }

        if self.kappa < 0.0 {
            return Err(QuantLibError::Validation(
                "kappa must be non-negative".to_owned(),
            ));
        }
        if self.theta < 0.0 {
            return Err(QuantLibError::Validation(
                "theta must be non-negative".to_owned(),
            ));
        }
        if self.sigma < 0.0 {
            return Err(QuantLibError::Validation(
                "sigma (vol-of-vol) must be non-negative".to_owned(),
            ));
        }
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err(QuantLibError::Validation(
                "rho must be within [-1, 1]".to_owned(),
            ));
        }
        if self.v0 < 0.0 {
            return Err(QuantLibError::Validation(
                "v0 must be non-negative".to_owned(),
            ));
        }

        Ok(())
    }

    pub fn characteristic_function(&self, u: Complex64, ttm: f64) -> Complex64 {
        if ttm <= 0.0 {
            return Complex64::new(1.0, 0.0);
        }

        let i = Complex64::new(0.0, 1.0);
        let sigma_sq = self.sigma * self.sigma;

        // Handle near-deterministic vol-of-vol with the sigma -> 0 limit.
        if sigma_sq <= 1.0e-16 {
            let integrated_variance = self.theta * ttm
                + (self.v0 - self.theta) * (1.0 - (-self.kappa * ttm).exp())
                    / self.kappa.max(1.0e-16);
            let drift = (self.r - self.q) * ttm;
            return (i * u * drift - 0.5 * (u * u + i * u) * integrated_variance).exp();
        }

        let drift = self.kappa - self.rho * self.sigma * u * i;
        let d = ((self.rho * self.sigma * u * i - self.kappa).powu(2) + sigma_sq * (u * i + u * u))
            .sqrt();
        let g = (drift - d) / (drift + d);
        let exp_dt = (-d * ttm).exp();
        let g_term = (Complex64::new(1.0, 0.0) - g * exp_dt) / (Complex64::new(1.0, 0.0) - g);

        let c = (self.r - self.q) * u * i * ttm
            + (self.kappa * self.theta / sigma_sq) * ((drift - d) * ttm - 2.0 * g_term.ln());
        let d_term = ((drift - d) / sigma_sq)
            * ((Complex64::new(1.0, 0.0) - exp_dt) / (Complex64::new(1.0, 0.0) - g * exp_dt));

        (c + d_term * self.v0).exp()
    }
}

impl StochasticModel for HestonModel {
    fn get_parameters(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("kappa".to_owned(), self.kappa.to_string()),
            ("theta".to_owned(), self.theta.to_string()),
            ("sigma".to_owned(), self.sigma.to_string()),
            ("rho".to_owned(), self.rho.to_string()),
            ("v0".to_owned(), self.v0.to_string()),
            ("r".to_owned(), self.r.to_string()),
            ("q".to_owned(), self.q.to_string()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::HestonModel;

    #[test]
    fn validates_rho_range() {
        let bad = HestonModel::new(1.0, 0.04, 0.3, 1.2, 0.04, 0.02, 0.0);
        assert!(bad.is_err());
    }

    #[test]
    fn characteristic_function_is_one_at_origin() {
        let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01).expect("model");
        let phi0 = model.characteristic_function(Complex64::new(0.0, 0.0), 1.0);
        assert!(
            (phi0 - Complex64::new(1.0, 0.0)).norm() < 1.0e-12,
            "phi0={phi0:?}"
        );
    }

    #[test]
    fn characteristic_function_respects_first_moment() {
        let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01).expect("model");
        let ttm = 1.25;
        let u = Complex64::new(0.0, -1.0);
        let phi = model.characteristic_function(u, ttm);
        let expected = ((model.r - model.q) * ttm).exp();
        assert!(
            (phi.re - expected).abs() < 1.0e-10,
            "phi={phi:?}, expected={expected}"
        );
        assert!(phi.im.abs() < 1.0e-10, "phi={phi:?}");
    }

    #[test]
    fn rejects_nan_inputs() {
        let model = HestonModel::new(f64::NAN, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01);
        assert!(model.is_err());
    }
}
