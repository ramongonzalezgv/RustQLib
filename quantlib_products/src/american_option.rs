use std::sync::Arc;

use quantlib_core::{FinancialProduct, MarketCalendar, OptionType, QuantLibError, SimpleDate};

use crate::{EuropeanOption, UnderlyingQuote};

#[derive(Clone)]
pub struct AmericanOption {
    pub european: EuropeanOption,
}

impl AmericanOption {
    #[allow(clippy::too_many_arguments)]
    pub fn new_spot(
        s: f64,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let european = EuropeanOption::new_spot(
            s,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )?;

        Ok(Self { european })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_forward(
        f: f64,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let european = EuropeanOption::new_forward(
            f,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )?;

        Ok(Self { european })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        underlying: UnderlyingQuote,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let european = EuropeanOption::new(
            underlying,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )?;

        Ok(Self { european })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_date_str_spot(
        s: f64,
        k: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let european = EuropeanOption::from_date_str_spot(
            s,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )?;

        Ok(Self { european })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_date_str_forward(
        f: f64,
        k: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let european = EuropeanOption::from_date_str_forward(
            f,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )?;

        Ok(Self { european })
    }

    pub fn days_to_maturity(&self) -> i64 {
        self.european.days_to_maturity()
    }

    pub fn ttm(&self) -> f64 {
        self.european.ttm()
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        self.european.validate()
    }
}

impl FinancialProduct for AmericanOption {
    fn payoff(&self, spot_prices: &[f64]) -> Vec<f64> {
        self.european.payoff(spot_prices)
    }
}

#[cfg(test)]
mod tests {
    use quantlib_core::{FinancialProduct, OptionType, SimpleDate};

    use super::AmericanOption;

    #[test]
    fn american_option_payoff_matches_european_contract_shape() {
        let option = AmericanOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            2,
            None,
        )
        .expect("valid product");

        let payoff = option.payoff(&[90.0, 100.0, 110.0]);
        assert_eq!(payoff, vec![0.0, 0.0, 20.0]);
    }
}
