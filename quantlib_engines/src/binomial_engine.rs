use quantlib_core::{OptionType, PricingEngine, QuantLibError};
use quantlib_models::BlackScholesModel;
use quantlib_products::{AmericanOption, EuropeanOption};

#[derive(Debug, Clone)]
pub struct BinomialLevels {
    pub price: f64,
    pub level1: Option<Vec<f64>>,
    pub level2: Option<Vec<f64>>,
    pub up_factor: f64,
    pub down_factor: f64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BinomialEngine {
    pub n_steps: usize,
}

impl BinomialEngine {
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps: n_steps.max(1),
        }
    }

    pub fn price_and_levels_european(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
        n_steps: Option<usize>,
    ) -> Result<BinomialLevels, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let steps = n_steps.unwrap_or(self.n_steps).max(1);
        price_and_levels(
            BinomialInput {
                spot,
                strike: product.k,
                option_type: product.option_type,
                qty: product.qty,
                ttm: product.ttm(),
                n_steps: steps,
                american_exercise: false,
            },
            model,
        )
    }

    pub fn price_and_levels_american(
        &self,
        product: &AmericanOption,
        model: &BlackScholesModel,
        n_steps: Option<usize>,
    ) -> Result<BinomialLevels, QuantLibError> {
        product.validate()?;
        model.validate()?;

        let spot = product.european.require_spot()?;
        let steps = n_steps.unwrap_or(self.n_steps).max(1);
        price_and_levels(
            BinomialInput {
                spot,
                strike: product.european.k,
                option_type: product.european.option_type,
                qty: product.european.qty,
                ttm: product.ttm(),
                n_steps: steps,
                american_exercise: true,
            },
            model,
        )
    }
}

impl PricingEngine<EuropeanOption, BlackScholesModel> for BinomialEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        Ok(self
            .price_and_levels_european(product, model, Some(self.n_steps))?
            .price)
    }
}

impl PricingEngine<AmericanOption, BlackScholesModel> for BinomialEngine {
    fn calculate_price(
        &self,
        product: &AmericanOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        Ok(self
            .price_and_levels_american(product, model, Some(self.n_steps))?
            .price)
    }
}

#[derive(Debug, Clone, Copy)]
struct BinomialInput {
    spot: f64,
    strike: f64,
    option_type: OptionType,
    qty: i32,
    ttm: f64,
    n_steps: usize,
    american_exercise: bool,
}

fn price_and_levels(
    input: BinomialInput,
    model: &BlackScholesModel,
) -> Result<BinomialLevels, QuantLibError> {
    let spot = input.spot;
    let strike = input.strike;
    let option_type = input.option_type;
    let qty = input.qty;
    let ttm = input.ttm;
    let n_steps = input.n_steps;
    let american_exercise = input.american_exercise;
    if n_steps == 0 {
        return Err(QuantLibError::Validation(
            "binomial n_steps must be >= 1".to_owned(),
        ));
    }

    if ttm <= 0.0 {
        let intrinsic = intrinsic_value(option_type, spot, strike, qty);
        return Ok(BinomialLevels {
            price: intrinsic,
            level1: None,
            level2: None,
            up_factor: 1.0,
            down_factor: 1.0,
        });
    }

    let dt = ttm / n_steps as f64;
    if dt <= 0.0 {
        return Err(QuantLibError::Engine(
            "invalid dt in binomial engine".to_owned(),
        ));
    }

    let sigma = model.sigma;
    let r = model.r;
    let q = model.q;

    let sigma_sqrt_dt = sigma * dt.sqrt();
    let mut up = sigma_sqrt_dt.exp();
    let mut down = 1.0 / up;
    let discount = (-r * dt).exp();

    // Degenerate sigma branch: deterministic risk-neutral drift.
    let deterministic = (up - down).abs() < 1.0e-14;
    let p = if deterministic {
        let drift_up = ((r - q) * dt).exp();
        up = drift_up;
        down = drift_up;
        1.0
    } else {
        (((r - q) * dt).exp() - down) / (up - down)
    };

    if !(0.0..=1.0).contains(&p) {
        return Err(QuantLibError::Engine(format!(
            "risk-neutral probability out of range: {p}"
        )));
    }

    let mut values = terminal_payoffs(spot, strike, option_type, qty, up, down, n_steps);

    let mut level1 = None;
    let mut level2 = None;

    for i in (0..n_steps).rev() {
        let mut next = vec![0.0; i + 1];

        for j in 0..=i {
            let continuation = discount * (p * values[j] + (1.0 - p) * values[j + 1]);
            if american_exercise {
                let node_spot = spot * up.powf((i - j) as f64) * down.powf(j as f64);
                let exercise = intrinsic_value(option_type, node_spot, strike, qty);
                next[j] = continuation.max(exercise);
            } else {
                next[j] = continuation;
            }
        }

        if i == 2 {
            level2 = Some(next.clone());
        }
        if i == 1 {
            level1 = Some(next.clone());
        }

        values = next;
    }

    Ok(BinomialLevels {
        price: values[0],
        level1,
        level2,
        up_factor: up,
        down_factor: down,
    })
}

fn terminal_payoffs(
    spot: f64,
    strike: f64,
    option_type: OptionType,
    qty: i32,
    up: f64,
    down: f64,
    n_steps: usize,
) -> Vec<f64> {
    let mut payoffs = vec![0.0; n_steps + 1];
    for (j, payoff) in payoffs.iter_mut().enumerate() {
        let terminal_spot = spot * up.powf((n_steps - j) as f64) * down.powf(j as f64);
        *payoff = intrinsic_value(option_type, terminal_spot, strike, qty);
    }
    payoffs
}

fn intrinsic_value(option_type: OptionType, spot: f64, strike: f64, qty: i32) -> f64 {
    let unsigned = match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
    };
    unsigned * f64::from(qty)
}

#[cfg(test)]
mod tests {
    use quantlib_core::{OptionType, PricingEngine, SimpleDate};
    use quantlib_models::BlackScholesModel;
    use quantlib_products::{AmericanOption, EuropeanOption};

    use crate::analytical_engine::AnalyticalEngine;

    use super::BinomialEngine;

    #[test]
    fn binomial_european_converges_to_analytical() {
        let option = EuropeanOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            None,
        )
        .expect("option");

        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
        let binom = BinomialEngine::new(500);
        let analytical = AnalyticalEngine;

        let tree_price = binom.calculate_price(&option, &model).expect("tree");
        let bs_price = analytical.calculate_price(&option, &model).expect("bs");

        assert!(
            (tree_price - bs_price).abs() < 0.1,
            "tree={tree_price}, bs={bs_price}"
        );
    }

    #[test]
    fn american_put_not_cheaper_than_european_put() {
        let euro_put = EuropeanOption::new_spot(
            90.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            1,
            None,
        )
        .expect("option");

        let amer_put = AmericanOption::new_spot(
            90.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            1,
            None,
        )
        .expect("option");

        let model = BlackScholesModel::new(0.2, 0.05, 0.0).expect("model");
        let binom = BinomialEngine::new(500);

        let euro_price = binom.calculate_price(&euro_put, &model).expect("euro");
        let amer_price = binom.calculate_price(&amer_put, &model).expect("amer");

        assert!(
            amer_price + 1.0e-10 >= euro_price,
            "amer={amer_price}, euro={euro_price}"
        );
    }
}
