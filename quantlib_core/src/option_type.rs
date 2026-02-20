use core::fmt;
use core::str::FromStr;

use crate::error::QuantLibError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl fmt::Display for OptionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call => write!(f, "call"),
            Self::Put => write!(f, "put"),
        }
    }
}

impl FromStr for OptionType {
    type Err = QuantLibError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "call" => Ok(Self::Call),
            "put" => Ok(Self::Put),
            _ => Err(QuantLibError::Validation(
                "option type must be 'call' or 'put'".to_owned(),
            )),
        }
    }
}
