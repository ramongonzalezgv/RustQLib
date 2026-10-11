use quantlib_core::QuantLibError;

#[derive(Debug, Clone, Copy)]
pub struct BlackScholesModel {
    pub sigma: f64,
    pub r: f64,
    pub q: f64,
}

impl BlackScholesModel {
    pub fn new(sigma: f64, r: f64, q: f64) -> Result<Self, QuantLibError> {
        let model = Self { sigma, r, q };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(QuantLibError::Validation(
                "sigma must be finite and positive".to_owned(),
            ));
        }
        if !self.r.is_finite() {
            return Err(QuantLibError::Validation("r must be finite".to_owned()));
        }
        if !self.q.is_finite() {
            return Err(QuantLibError::Validation("q must be finite".to_owned()));
        }
        Ok(())
    }

    pub fn d1(&self, spot: f64, strike: f64, ttm: f64) -> f64 {
        ((spot / strike).ln() + (self.r - self.q + 0.5 * self.sigma * self.sigma) * ttm)
            / (self.sigma * ttm.sqrt())
    }

    pub fn d2(&self, spot: f64, strike: f64, ttm: f64) -> f64 {
        self.d1(spot, strike, ttm) - self.sigma * ttm.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::BlackScholesModel;

    #[test]
    fn rejects_nan_inputs() {
        assert!(BlackScholesModel::new(f64::NAN, 0.03, 0.0).is_err());
        assert!(BlackScholesModel::new(0.2, f64::NAN, 0.0).is_err());
        assert!(BlackScholesModel::new(0.2, 0.03, f64::NAN).is_err());
    }
}
