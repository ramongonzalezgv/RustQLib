use std::str::FromStr;
use std::sync::Arc;

use quantlib_core::{FinancialProduct, MarketCalendar, OptionType, QuantLibError, SimpleDate};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AveragingType {
    Arithmetic,
    Geometric,
}

impl FromStr for AveragingType {
    type Err = QuantLibError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "arithmetic" => Ok(Self::Arithmetic),
            "geometric" => Ok(Self::Geometric),
            _ => Err(QuantLibError::Validation(
                "averaging type must be 'arithmetic' or 'geometric'".to_owned(),
            )),
        }
    }
}

impl core::fmt::Display for AveragingType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Arithmetic => write!(f, "arithmetic"),
            Self::Geometric => write!(f, "geometric"),
        }
    }
}

#[derive(Clone)]
pub struct AsianOption {
    pub s: f64,
    pub k: f64,
    pub expiry_date: SimpleDate,
    pub valuation_date: SimpleDate,
    pub option_type: OptionType,
    pub qty: i32,
    pub averaging_type: AveragingType,
    pub calendar: Option<Arc<dyn MarketCalendar>>,
}

impl AsianOption {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s: f64,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        averaging_type: AveragingType,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let valuation_date = valuation_date.unwrap_or_else(SimpleDate::today_utc);
        let option = Self {
            s,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            averaging_type,
            calendar,
        };
        option.validate()?;
        Ok(option)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_date_str(
        s: f64,
        k: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        averaging_type: &str,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let expiry = SimpleDate::parse(expiry_date)?;
        let valuation = valuation_date.map(SimpleDate::parse).transpose()?;
        let option_type = OptionType::from_str(option_type)?;
        let averaging_type = AveragingType::from_str(averaging_type)?;

        Self::new(
            s,
            k,
            expiry,
            valuation,
            option_type,
            qty,
            averaging_type,
            calendar,
        )
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.s.is_finite() || !self.k.is_finite() {
            return Err(QuantLibError::Validation(
                "spot price (S) and strike (K) must be finite".to_owned(),
            ));
        }

        if self.s < 0.0 || self.k < 0.0 {
            return Err(QuantLibError::Validation(
                "spot price (S) and strike (K) must be non-negative".to_owned(),
            ));
        }

        if self.valuation_date > self.expiry_date {
            return Err(QuantLibError::Validation(format!(
                "valuation date ({}) cannot be after expiry ({})",
                self.valuation_date, self.expiry_date
            )));
        }

        Ok(())
    }

    pub fn days_to_maturity(&self) -> i64 {
        match &self.calendar {
            Some(calendar) => calendar.count_days(self.valuation_date, self.expiry_date),
            None => self.valuation_date.days_between(&self.expiry_date),
        }
    }

    pub fn ttm(&self) -> f64 {
        let denominator = match &self.calendar {
            Some(calendar) => calendar.days_in_year(),
            None => 365.0,
        };
        (self.days_to_maturity() as f64 / denominator).max(0.0)
    }

    pub fn average_from_paths(&self, spot_paths: &[Vec<f64>]) -> Result<Vec<f64>, QuantLibError> {
        let mut averages = Vec::with_capacity(spot_paths.len());

        for path in spot_paths {
            if path.is_empty() {
                return Err(QuantLibError::Validation(
                    "spot paths for Asian payoff cannot contain empty paths".to_owned(),
                ));
            }

            let avg = match self.averaging_type {
                AveragingType::Arithmetic => path.iter().sum::<f64>() / path.len() as f64,
                AveragingType::Geometric => {
                    if path.iter().any(|v| *v <= 0.0) {
                        return Err(QuantLibError::Validation(
                            "geometric averaging requires strictly positive prices".to_owned(),
                        ));
                    }

                    let mean_log = path.iter().map(|v| v.ln()).sum::<f64>() / path.len() as f64;
                    mean_log.exp()
                }
            };

            averages.push(avg);
        }

        Ok(averages)
    }

    pub fn payoff_from_paths(&self, spot_paths: &[Vec<f64>]) -> Result<Vec<f64>, QuantLibError> {
        let averages = self.average_from_paths(spot_paths)?;
        Ok(self.payoff(&averages))
    }
}

impl FinancialProduct for AsianOption {
    fn payoff(&self, average_prices: &[f64]) -> Vec<f64> {
        let qty = self.qty as f64;

        average_prices
            .iter()
            .map(|avg| match self.option_type {
                OptionType::Call => qty * (avg - self.k).max(0.0),
                OptionType::Put => qty * (self.k - avg).max(0.0),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use quantlib_core::{OptionType, SimpleDate};

    use super::{AsianOption, AveragingType};

    #[test]
    fn arithmetic_average_payoff_from_paths() {
        let option = AsianOption::new(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            AveragingType::Arithmetic,
            None,
        )
        .expect("valid option");

        let paths = vec![vec![90.0, 100.0, 110.0], vec![80.0, 90.0, 100.0]];
        let payoff = option.payoff_from_paths(&paths).expect("payoff");

        assert_eq!(payoff.len(), 2);
        assert!((payoff[0] - 0.0).abs() < 1.0e-12);
        assert!((payoff[1] - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn geometric_average_requires_positive_prices() {
        let option = AsianOption::new(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            AveragingType::Geometric,
            None,
        )
        .expect("valid option");

        let paths = vec![vec![100.0, 0.0, 110.0]];
        let err = option
            .payoff_from_paths(&paths)
            .expect_err("geometric average must reject non-positive prices");
        assert!(
            err.to_string()
                .contains("requires strictly positive prices")
        );
    }
}
