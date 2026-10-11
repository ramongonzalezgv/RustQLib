use num_complex::Complex64;
use quantlib_core::QuantLibError;

use crate::heston_model::HestonModel;

#[derive(Debug, Clone, Copy)]
pub struct BatesModel {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub rho: f64,
    pub v0: f64,
    pub r: f64,
    pub q: f64,
    pub lambda_jump: f64,
    pub mu_j: f64,
    pub sigma_j: f64,
}

impl BatesModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
        r: f64,
        q: f64,
        lambda_jump: f64,
        mu_j: f64,
        sigma_j: f64,
    ) -> Result<Self, QuantLibError> {
        let model = Self {
            kappa,
            theta,
            sigma,
            rho,
            v0,
            r,
            q,
            lambda_jump,
            mu_j,
            sigma_j,
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
            || !self.lambda_jump.is_finite()
            || !self.mu_j.is_finite()
            || !self.sigma_j.is_finite()
        {
            return Err(QuantLibError::Validation(
                "Bates parameters must be finite".to_owned(),
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
        if self.lambda_jump < 0.0 {
            return Err(QuantLibError::Validation(
                "lambda_jump must be non-negative".to_owned(),
            ));
        }
        if self.sigma_j < 0.0 {
            return Err(QuantLibError::Validation(
                "sigma_j must be non-negative".to_owned(),
            ));
        }

        Ok(())
    }

    pub fn characteristic_function(&self, u: Complex64, ttm: f64) -> Complex64 {
        if ttm <= 0.0 {
            return Complex64::new(1.0, 0.0);
        }

        let heston_component = HestonModel {
            kappa: self.kappa,
            theta: self.theta,
            sigma: self.sigma,
            rho: self.rho,
            v0: self.v0,
            r: self.r,
            q: self.q,
        }
        .characteristic_function(u, ttm);

        let i = Complex64::new(0.0, 1.0);
        let sigma_j_sq = self.sigma_j * self.sigma_j;
        let jump_mean = (self.mu_j + 0.5 * sigma_j_sq).exp() - 1.0;
        let jump_log_cf = (i * u * self.mu_j - 0.5 * sigma_j_sq * u * u).exp();
        let jump_exponent =
            self.lambda_jump * ttm * (jump_log_cf - Complex64::new(1.0, 0.0) - i * u * jump_mean);

        heston_component * jump_exponent.exp()
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::HestonModel;

    use super::BatesModel;

    #[test]
    fn validates_rho_range() {
        let bad = BatesModel::new(1.0, 0.04, 0.3, 1.2, 0.04, 0.02, 0.0, 0.5, -0.1, 0.2);
        assert!(bad.is_err());
    }

    #[test]
    fn rejects_negative_jump_intensity() {
        let bad = BatesModel::new(1.0, 0.04, 0.3, -0.5, 0.04, 0.02, 0.0, -0.1, -0.1, 0.2);
        assert!(bad.is_err());
    }

    #[test]
    fn characteristic_function_is_one_at_origin() {
        let model =
            BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01, 0.3, -0.1, 0.2).expect("model");
        let phi0 = model.characteristic_function(Complex64::new(0.0, 0.0), 1.0);
        assert!(
            (phi0 - Complex64::new(1.0, 0.0)).norm() < 1.0e-12,
            "phi0={phi0:?}"
        );
    }

    #[test]
    fn characteristic_function_respects_first_moment() {
        let model = BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01, 0.4, -0.05, 0.25)
            .expect("model");
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
    fn reduces_to_heston_when_jump_intensity_is_zero() {
        let bates = BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01, 0.0, -0.05, 0.25)
            .expect("model");
        let heston = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01).expect("model");

        let u = Complex64::new(1.7, -0.3);
        let ttm = 1.8;
        let bates_phi = bates.characteristic_function(u, ttm);
        let heston_phi = heston.characteristic_function(u, ttm);

        assert!(
            (bates_phi - heston_phi).norm() < 1.0e-12,
            "bates={bates_phi:?}, heston={heston_phi:?}"
        );
    }

    #[test]
    fn rejects_nan_inputs() {
        let model = BatesModel::new(
            f64::NAN,
            0.04,
            0.6,
            -0.7,
            0.04,
            0.02,
            0.01,
            0.4,
            -0.05,
            0.25,
        );
        assert!(model.is_err());
    }
}
