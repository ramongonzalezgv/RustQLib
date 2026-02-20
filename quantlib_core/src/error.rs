use std::fmt;

#[derive(Debug, Clone)]
pub enum QuantLibError {
    Validation(String),
    DateParse(String),
    Serialization(String),
    Engine(String),
}

impl fmt::Display for QuantLibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(msg) => write!(f, "validation error: {msg}"),
            Self::DateParse(msg) => write!(f, "date parse error: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::Engine(msg) => write!(f, "engine error: {msg}"),
        }
    }
}

impl std::error::Error for QuantLibError {}
