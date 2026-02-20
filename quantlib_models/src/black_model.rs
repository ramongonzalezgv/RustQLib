use std::collections::BTreeMap;

use quantlib_core::{QuantLibError, StochasticModel};

#[derive(Debug, Clone, Copy)]
pub struct BlackModel {
    pub sigma: f64,
    pub r: f64,
}

impl BlackModel {
    pub fn new(sigma: f64, r: f64) -> Result<Self, QuantLibError> {
        let model = Self { sigma, r };
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
        Ok(())
    }

    pub fn d1(&self, forward: f64, strike: f64, ttm: f64) -> f64 {
        ((forward / strike).ln() + 0.5 * self.sigma * self.sigma * ttm) / (self.sigma * ttm.sqrt())
    }

    pub fn d2(&self, forward: f64, strike: f64, ttm: f64) -> f64 {
        self.d1(forward, strike, ttm) - self.sigma * ttm.sqrt()
    }
}

impl StochasticModel for BlackModel {
    fn get_parameters(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("sigma".to_owned(), self.sigma.to_string()),
            ("r".to_owned(), self.r.to_string()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::BlackModel;

    #[test]
    fn d1_is_finite_for_valid_inputs() {
        let model = BlackModel::new(0.2, 0.03).expect("valid model");
        let d1 = model.d1(100.0, 100.0, 1.0);
        assert!(d1.is_finite());
    }

    #[test]
    fn rejects_nan_inputs() {
        assert!(BlackModel::new(f64::NAN, 0.03).is_err());
        assert!(BlackModel::new(0.2, f64::NAN).is_err());
    }
}
