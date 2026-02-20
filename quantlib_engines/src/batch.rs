use quantlib_core::{PricingEngine, QuantLibError};
use rayon::prelude::*;

pub fn calculate_prices_parallel<E, P, M>(
    engine: &E,
    products: &[P],
    model: &M,
) -> Result<Vec<f64>, QuantLibError>
where
    E: PricingEngine<P, M> + Sync,
    P: Sync,
    M: Sync,
{
    let prices: Vec<Result<f64, QuantLibError>> = products
        .par_iter()
        .map(|product| engine.calculate_price(product, model))
        .collect();

    prices.into_iter().collect()
}

pub fn calculate_prices_parallel_per_model<E, P, M>(
    engine: &E,
    products: &[P],
    models: &[M],
) -> Result<Vec<f64>, QuantLibError>
where
    E: PricingEngine<P, M> + Sync,
    P: Sync,
    M: Sync,
{
    if products.len() != models.len() {
        return Err(QuantLibError::Validation(format!(
            "products length ({}) must match models length ({})",
            products.len(),
            models.len()
        )));
    }

    let prices: Vec<Result<f64, QuantLibError>> = products
        .par_iter()
        .zip(models.par_iter())
        .map(|(product, model)| engine.calculate_price(product, model))
        .collect();

    prices.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use quantlib_core::{OptionType, PricingEngine, SimpleDate};
    use quantlib_models::BlackScholesModel;
    use quantlib_products::EuropeanOption;

    use crate::{AnalyticalEngine, batch};

    #[test]
    fn parallel_batch_matches_individual_pricing() {
        let expiry = SimpleDate::new(2027, 2, 10).expect("date");
        let valuation = SimpleDate::new(2026, 2, 10).expect("date");
        let products = vec![
            EuropeanOption::new_spot(
                100.0,
                95.0,
                expiry,
                Some(valuation),
                OptionType::Call,
                1,
                None,
            )
            .expect("option"),
            EuropeanOption::new_spot(
                100.0,
                105.0,
                expiry,
                Some(valuation),
                OptionType::Put,
                2,
                None,
            )
            .expect("option"),
        ];

        let model = BlackScholesModel::new(0.25, 0.03, 0.01).expect("model");
        let engine = AnalyticalEngine;

        let expected = products
            .iter()
            .map(|product| engine.calculate_price(product, &model).expect("price"))
            .collect::<Vec<_>>();
        let batch_prices =
            batch::calculate_prices_parallel(&engine, &products, &model).expect("batch pricing");

        assert_eq!(batch_prices.len(), expected.len());
        for (got, want) in batch_prices.iter().zip(expected.iter()) {
            assert!((*got - *want).abs() < 1.0e-12, "got={got}, want={want}");
        }
    }
}
