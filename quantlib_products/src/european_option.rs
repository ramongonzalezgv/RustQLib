use std::{collections::BTreeMap, str::FromStr, sync::Arc};

use quantlib_core::{FinancialProduct, MarketCalendar, OptionType, QuantLibError, SimpleDate};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnderlyingQuote {
    Spot(f64),
    Forward(f64),
}

#[derive(Clone)]
pub struct EuropeanOption {
    pub underlying: UnderlyingQuote,
    pub k: f64,
    pub expiry_date: SimpleDate,
    pub valuation_date: SimpleDate,
    pub option_type: OptionType,
    pub qty: i32,
    pub calendar: Option<Arc<dyn MarketCalendar>>,
}

impl EuropeanOption {
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
        Self::new(
            UnderlyingQuote::Spot(s),
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )
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
        Self::new(
            UnderlyingQuote::Forward(f),
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
        )
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
        let valuation_date = valuation_date.unwrap_or_else(SimpleDate::today_utc);
        let option = Self {
            underlying,
            k,
            expiry_date,
            valuation_date,
            option_type,
            qty,
            calendar,
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
        calendar: Option<Arc<dyn MarketCalendar>>,
    ) -> Result<Self, QuantLibError> {
        let expiry = SimpleDate::parse(expiry_date)?;
        let valuation = valuation_date.map(SimpleDate::parse).transpose()?;
        let option_type = OptionType::from_str(option_type)?;

        Self::new_spot(s, k, expiry, valuation, option_type, qty, calendar)
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
        let expiry = SimpleDate::parse(expiry_date)?;
        let valuation = valuation_date.map(SimpleDate::parse).transpose()?;
        let option_type = OptionType::from_str(option_type)?;

        Self::new_forward(f, k, expiry, valuation, option_type, qty, calendar)
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        match self.underlying {
            UnderlyingQuote::Spot(s) => {
                if !s.is_finite() {
                    return Err(QuantLibError::Validation(
                        "underlying price must be finite".to_owned(),
                    ));
                }
                if s < 0.0 {
                    return Err(QuantLibError::Validation(
                        "spot price must be non-negative".to_owned(),
                    ));
                }
            }
            UnderlyingQuote::Forward(f) => {
                if !f.is_finite() {
                    return Err(QuantLibError::Validation(
                        "underlying price must be finite".to_owned(),
                    ));
                }
                if f < 0.0 {
                    return Err(QuantLibError::Validation(
                        "forward price must be non-negative".to_owned(),
                    ));
                }
            }
        }

        if !self.k.is_finite() {
            return Err(QuantLibError::Validation(
                "strike price must be finite".to_owned(),
            ));
        }
        if self.k < 0.0 {
            return Err(QuantLibError::Validation(
                "strike price must be non-negative".to_owned(),
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
        let days = self.days_to_maturity() as f64;
        let denominator = match &self.calendar {
            Some(calendar) => calendar.days_in_year(),
            None => 365.0,
        };

        (days / denominator).max(0.0)
    }

    pub fn spot(&self) -> Option<f64> {
        match self.underlying {
            UnderlyingQuote::Spot(s) => Some(s),
            UnderlyingQuote::Forward(_) => None,
        }
    }

    pub fn forward(&self) -> Option<f64> {
        match self.underlying {
            UnderlyingQuote::Spot(_) => None,
            UnderlyingQuote::Forward(f) => Some(f),
        }
    }

    pub fn require_spot(&self) -> Result<f64, QuantLibError> {
        self.spot().ok_or_else(|| {
            QuantLibError::Engine(
                "spot-based pricing requested for a forward-quoted EuropeanOption".to_owned(),
            )
        })
    }

    pub fn require_forward(&self) -> Result<f64, QuantLibError> {
        self.forward().ok_or_else(|| {
            QuantLibError::Engine(
                "forward-based pricing requested for a spot-quoted EuropeanOption".to_owned(),
            )
        })
    }
}

impl FinancialProduct for EuropeanOption {
    fn payoff(&self, spot_prices: &[f64]) -> Vec<f64> {
        let qty = self.qty as f64;
        spot_prices
            .iter()
            .map(|spot| match self.option_type {
                OptionType::Call => qty * (spot - self.k).max(0.0),
                OptionType::Put => qty * (self.k - spot).max(0.0),
            })
            .collect()
    }

    fn get_parameters(&self) -> BTreeMap<String, String> {
        let mut params = BTreeMap::from([
            ("K".to_owned(), self.k.to_string()),
            ("expiry_date".to_owned(), self.expiry_date.format_ymd()),
            ("option_type".to_owned(), self.option_type.to_string()),
            ("qty".to_owned(), self.qty.to_string()),
        ]);

        match self.underlying {
            UnderlyingQuote::Spot(s) => {
                params.insert("quote_type".to_owned(), "spot".to_owned());
                params.insert("S".to_owned(), s.to_string());
            }
            UnderlyingQuote::Forward(f) => {
                params.insert("quote_type".to_owned(), "forward".to_owned());
                params.insert("F".to_owned(), f.to_string());
            }
        }

        params
    }
}
