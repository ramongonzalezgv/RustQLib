use crate::error::QuantLibError;

pub trait FinancialProduct {
    fn payoff(&self, spot_prices: &[f64]) -> Vec<f64>;
}

pub trait PricingEngine<P, M> {
    fn calculate_price(&self, product: &P, model: &M) -> Result<f64, QuantLibError>;
}
