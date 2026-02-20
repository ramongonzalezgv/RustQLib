use std::collections::BTreeMap;

use num_complex::Complex64;
use quantlib_core::{QuantLibError, StochasticModel};

#[derive(Debug, Clone, Copy)]
pub struct KouModel {
    pub sigma: f64,
    pub r: f64,
    pub q: f64,
    pub lambda_jump: f64,
    pub p_up: f64,
    pub eta1: f64,
    pub eta2: f64,
}

impl KouModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sigma: f64,
        r: f64,
        q: f64,
        lambda_jump: f64,
        p_up: f64,
        eta1: f64,
        eta2: f64,
    ) -> Result<Self, QuantLibError> {
        let model = Self {
            sigma,
            r,
            q,
            lambda_jump,
            p_up,
            eta1,
            eta2,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.sigma.is_finite()
            || !self.r.is_finite()
            || !self.q.is_finite()
            || !self.lambda_jump.is_finite()
            || !self.p_up.is_finite()
            || !self.eta1.is_finite()
            || !self.eta2.is_finite()
        {
            return Err(QuantLibError::Validation(
                "Kou parameters must be finite".to_owned(),
            ));
        }

        if self.sigma < 0.0 {
            return Err(QuantLibError::Validation(
                "sigma must be non-negative".to_owned(),
            ));
        }
        if self.lambda_jump < 0.0 {
            return Err(QuantLibError::Validation(
                "lambda_jump must be non-negative".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.p_up) {
            return Err(QuantLibError::Validation(
                "p_up must be within [0, 1]".to_owned(),
            ));
        }
        if self.eta1 <= 1.0 {
            return Err(QuantLibError::Validation(
                "eta1 must be > 1 to ensure finite jump compensator".to_owned(),
            ));
        }
        if self.eta2 <= 0.0 {
            return Err(QuantLibError::Validation(
                "eta2 must be positive".to_owned(),
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

        let jump_mean = self.p_up * self.eta1 / (self.eta1 - 1.0)
            + (1.0 - self.p_up) * self.eta2 / (self.eta2 + 1.0)
            - 1.0;
        let iu = i * u;
        let jump_cf = self.p_up * self.eta1 / (Complex64::new(self.eta1, 0.0) - iu)
            + (1.0 - self.p_up) * self.eta2 / (Complex64::new(self.eta2, 0.0) + iu);

        let drift = (self.r - self.q - self.lambda_jump * jump_mean - 0.5 * sigma_sq) * ttm;
        let diffusion = -0.5 * sigma_sq * u * u * ttm;
        let jump = self.lambda_jump * ttm * (jump_cf - Complex64::new(1.0, 0.0));

        (i * u * drift + diffusion + jump).exp()
    }
}

impl StochasticModel for KouModel {
    fn get_parameters(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("sigma".to_owned(), self.sigma.to_string()),
            ("r".to_owned(), self.r.to_string()),
            ("q".to_owned(), self.q.to_string()),
            ("lambda_jump".to_owned(), self.lambda_jump.to_string()),
            ("p_up".to_owned(), self.p_up.to_string()),
            ("eta1".to_owned(), self.eta1.to_string()),
            ("eta2".to_owned(), self.eta2.to_string()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::KouModel;

    #[test]
    fn validates_probability_bounds() {
        let bad = KouModel::new(0.2, 0.02, 0.0, 0.5, 1.2, 10.0, 5.0);
        assert!(bad.is_err());
    }

    #[test]
    fn validates_eta_constraints() {
        let bad_eta1 = KouModel::new(0.2, 0.02, 0.0, 0.5, 0.4, 1.0, 5.0);
        assert!(bad_eta1.is_err());

        let bad_eta2 = KouModel::new(0.2, 0.02, 0.0, 0.5, 0.4, 10.0, 0.0);
        assert!(bad_eta2.is_err());
    }

    #[test]
    fn characteristic_function_is_one_at_origin() {
        let model = KouModel::new(0.2, 0.02, 0.01, 0.4, 0.35, 12.0, 7.0).expect("model");
        let phi0 = model.characteristic_function(Complex64::new(0.0, 0.0), 1.0);
        assert!(
            (phi0 - Complex64::new(1.0, 0.0)).norm() < 1.0e-12,
            "phi0={phi0:?}"
        );
    }

    #[test]
    fn characteristic_function_respects_first_moment() {
        let model = KouModel::new(0.25, 0.03, 0.01, 0.6, 0.4, 9.0, 6.0).expect("model");
        let ttm = 1.3;
        let phi = model.characteristic_function(Complex64::new(0.0, -1.0), ttm);
        let expected = ((model.r - model.q) * ttm).exp();

        assert!(
            (phi.re - expected).abs() < 1.0e-10,
            "phi={phi:?}, expected={expected}"
        );
        assert!(phi.im.abs() < 1.0e-10, "phi={phi:?}");
    }

    #[test]
    fn reduces_to_black_scholes_when_jump_intensity_is_zero() {
        let model = KouModel::new(0.2, 0.03, 0.01, 0.0, 0.4, 10.0, 5.0).expect("model");

        let u = Complex64::new(1.5, -0.25);
        let ttm = 1.1;
        let i = Complex64::new(0.0, 1.0);
        let sigma_sq = model.sigma * model.sigma;

        let kou_phi = model.characteristic_function(u, ttm);
        let bs_phi = (i * u * (model.r - model.q - 0.5 * sigma_sq) * ttm
            - 0.5 * sigma_sq * u * u * ttm)
            .exp();

        assert!(
            (kou_phi - bs_phi).norm() < 1.0e-12,
            "kou={kou_phi:?}, bs={bs_phi:?}"
        );
    }

    #[test]
    fn rejects_nan_inputs() {
        let model = KouModel::new(f64::NAN, 0.03, 0.01, 0.5, 0.4, 10.0, 5.0);
        assert!(model.is_err());
    }
}
