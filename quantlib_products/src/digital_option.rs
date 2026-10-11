use std::str::FromStr;
use std::sync::Arc;

use quantlib_core::{FinancialProduct, MarketCalendar, OptionType, QuantLibError, SimpleDate};

use crate::{EuropeanOption, UnderlyingQuote};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalPayoffType {
    CashOrNothing,
    AssetOrNothing,
}

impl std::fmt::Display for DigitalPayoffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CashOrNothing => write!(f, "cash"),
            Self::AssetOrNothing => write!(f, "asset"),
        }
    }
}

impl FromStr for DigitalPayoffType {
    type Err = QuantLibError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cash" | "cash_or_nothing" | "cash-or-nothing" | "cashornothing" => {
                Ok(Self::CashOrNothing)
            }
            "asset" | "asset_or_nothing" | "asset-or-nothing" | "assetornothing" => {
                Ok(Self::AssetOrNothing)
            }
            _ => Err(QuantLibError::Validation(
                "digital payoff type must be 'cash' or 'asset'".to_owned(),
            )),
        }
    }
}

#[derive(Clone)]
pub struct DigitalOption {
    pub european: EuropeanOption,
    pub payout: f64,
    pub payoff_type: DigitalPayoffType,
}

impl DigitalOption {
    #[allow(clippy::too_many_arguments)]
    pub fn new_spot(
        s: f64,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        payout: f64,
        payoff_type: DigitalPayoffType,
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

        let option = Self {
            european,
            payout,
            payoff_type,
        };
        option.validate()?;
        Ok(option)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_forward(
        f: f64,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        payout: f64,
        payoff_type: DigitalPayoffType,
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

        let option = Self {
            european,
            payout,
            payoff_type,
        };
        option.validate()?;
        Ok(option)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        underlying: UnderlyingQuote,
        k: f64,
        expiry_date: SimpleDate,
        valuation_date: Option<SimpleDate>,
        option_type: OptionType,
        qty: i32,
        payout: f64,
        payoff_type: DigitalPayoffType,
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

        let option = Self {
            european,
            payout,
            payoff_type,
        };
        option.validate()?;
        Ok(option)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_date_str_spot(
        s: f64,
        k: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        payout: f64,
        payoff_type: &str,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let option_type = OptionType::from_str(option_type)?;
        let payoff_type = DigitalPayoffType::from_str(payoff_type)?;
        let expiry = SimpleDate::parse(expiry_date)?;
        let valuation = valuation_date.map(SimpleDate::parse).transpose()?;

        Self::new_spot(
            s,
            k,
            expiry,
            valuation,
            option_type,
            qty,
            payout,
            payoff_type,
            calendar,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_date_str_forward(
        f: f64,
        k: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        payout: f64,
        payoff_type: &str,
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let option_type = OptionType::from_str(option_type)?;
        let payoff_type = DigitalPayoffType::from_str(payoff_type)?;
        let expiry = SimpleDate::parse(expiry_date)?;
        let valuation = valuation_date.map(SimpleDate::parse).transpose()?;

        Self::new_forward(
            f,
            k,
            expiry,
            valuation,
            option_type,
            qty,
            payout,
            payoff_type,
            calendar,
        )
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        self.european.validate()?;

        if !self.payout.is_finite() {
            return Err(QuantLibError::Validation(
                "digital payout must be finite".to_owned(),
            ));
        }
        if self.payout < 0.0 {
            return Err(QuantLibError::Validation(
                "digital payout must be non-negative".to_owned(),
            ));
        }

        Ok(())
    }

    pub fn strike(&self) -> f64 {
        self.european.k
    }

    pub fn option_type(&self) -> OptionType {
        self.european.option_type
    }

    pub fn days_to_maturity(&self) -> i64 {
        self.european.days_to_maturity()
    }

    pub fn ttm(&self) -> f64 {
        self.european.ttm()
    }

    pub fn spot(&self) -> Option<f64> {
        self.european.spot()
    }

    pub fn forward(&self) -> Option<f64> {
        self.european.forward()
    }

    pub fn require_spot(&self) -> Result<f64, QuantLibError> {
        self.european.require_spot().map_err(|_| {
            QuantLibError::Engine(
                "spot-based pricing requested for a forward-quoted DigitalOption".to_owned(),
            )
        })
    }

    pub fn require_forward(&self) -> Result<f64, QuantLibError> {
        self.european.require_forward().map_err(|_| {
            QuantLibError::Engine(
                "forward-based pricing requested for a spot-quoted DigitalOption".to_owned(),
            )
        })
    }

    pub fn unit_payoff_at_maturity(&self, underlying_price: f64) -> f64 {
        let is_itm = match self.option_type() {
            OptionType::Call => underlying_price > self.strike(),
            OptionType::Put => underlying_price < self.strike(),
        };

        if !is_itm {
            return 0.0;
        }

        match self.payoff_type {
            DigitalPayoffType::CashOrNothing => self.payout,
            DigitalPayoffType::AssetOrNothing => self.payout * underlying_price,
        }
    }
}

impl FinancialProduct for DigitalOption {
    fn payoff(&self, spot_prices: &[f64]) -> Vec<f64> {
        let qty = f64::from(self.european.qty);
        spot_prices
            .iter()
            .map(|spot| qty * self.unit_payoff_at_maturity(*spot))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use quantlib_core::{FinancialProduct, OptionType, SimpleDate};

    use super::{DigitalOption, DigitalPayoffType};

    #[test]
    fn cash_or_nothing_payoff_uses_fixed_payout() {
        let option = DigitalOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            2,
            15.0,
            DigitalPayoffType::CashOrNothing,
            None,
        )
        .expect("option");

        let payoff = option.payoff(&[99.0, 101.0]);
        assert_eq!(payoff, vec![0.0, 30.0]);
    }

    #[test]
    fn asset_or_nothing_payoff_scales_underlying() {
        let option = DigitalOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            1,
            1.5,
            DigitalPayoffType::AssetOrNothing,
            None,
        )
        .expect("option");

        let payoff = option.payoff(&[90.0, 110.0]);
        assert_eq!(payoff, vec![135.0, 0.0]);
    }

    #[test]
    fn rejects_negative_payout() {
        let err = DigitalOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            -1.0,
            DigitalPayoffType::CashOrNothing,
            None,
        )
        .err()
        .expect("must reject negative payout");

        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn parses_payoff_type_aliases() {
        let cash = "cash-or-nothing"
            .parse::<DigitalPayoffType>()
            .expect("cash");
        let asset = "asset_or_nothing"
            .parse::<DigitalPayoffType>()
            .expect("asset");

        assert_eq!(cash, DigitalPayoffType::CashOrNothing);
        assert_eq!(asset, DigitalPayoffType::AssetOrNothing);
    }
}
