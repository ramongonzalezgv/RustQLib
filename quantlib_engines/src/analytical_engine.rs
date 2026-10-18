use quantlib_core::{OptionType, PricingEngine, QuantLibError, norm_cdf, norm_pdf};
use quantlib_models::{BlackModel, BlackScholesModel, SabrModel};
use quantlib_products::{
    AsianOption, AveragingType, DigitalOption, DigitalPayoffType, EuropeanOption,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct AnalyticalEngine;

impl AnalyticalEngine {
    fn intrinsic(option_type: OptionType, spot_or_forward: f64, strike: f64) -> f64 {
        match option_type {
            OptionType::Call => (spot_or_forward - strike).max(0.0),
            OptionType::Put => (strike - spot_or_forward).max(0.0),
        }
    }

    fn price_digital_black_scholes(
        &self,
        product: &DigitalOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        let spot = product.require_spot()?;
        let ttm = product.ttm();
        if ttm <= 0.0 {
            return Ok(product.unit_payoff_at_maturity(spot));
        }

        let strike = product.strike();
        let d1 = model.d1(spot, strike, ttm);
        let d2 = model.d2(spot, strike, ttm);
        let discount = (-(model.r) * ttm).exp();
        let discounted_spot = spot * (-(model.q) * ttm).exp();

        let unit_price = match (product.payoff_type, product.option_type()) {
            (DigitalPayoffType::CashOrNothing, OptionType::Call) => {
                product.payout * discount * norm_cdf(d2)
            }
            (DigitalPayoffType::CashOrNothing, OptionType::Put) => {
                product.payout * discount * norm_cdf(-d2)
            }
            (DigitalPayoffType::AssetOrNothing, OptionType::Call) => {
                product.payout * discounted_spot * norm_cdf(d1)
            }
            (DigitalPayoffType::AssetOrNothing, OptionType::Put) => {
                product.payout * discounted_spot * norm_cdf(-d1)
            }
        };

        Ok(unit_price)
    }

    fn price_digital_black(
        &self,
        product: &DigitalOption,
        model: &BlackModel,
    ) -> Result<f64, QuantLibError> {
        let forward = product.require_forward()?;
        let ttm = product.ttm();
        if ttm <= 0.0 {
            return Ok(product.unit_payoff_at_maturity(forward));
        }

        let strike = product.strike();
        let d1 = model.d1(forward, strike, ttm);
        let d2 = model.d2(forward, strike, ttm);
        let discount = (-(model.r) * ttm).exp();

        let unit_price = match (product.payoff_type, product.option_type()) {
            (DigitalPayoffType::CashOrNothing, OptionType::Call) => {
                product.payout * discount * norm_cdf(d2)
            }
            (DigitalPayoffType::CashOrNothing, OptionType::Put) => {
                product.payout * discount * norm_cdf(-d2)
            }
            (DigitalPayoffType::AssetOrNothing, OptionType::Call) => {
                product.payout * discount * forward * norm_cdf(d1)
            }
            (DigitalPayoffType::AssetOrNothing, OptionType::Put) => {
                product.payout * discount * forward * norm_cdf(-d1)
            }
        };

        Ok(unit_price)
    }

    fn price_sabr(
        &self,
        product: &EuropeanOption,
        model: &SabrModel,
    ) -> Result<f64, QuantLibError> {
        let ttm = product.ttm();
        let forward = product.require_forward()?;

        if ttm <= 0.0 {
            return Ok(Self::intrinsic(product.option_type, forward, product.k));
        }

        let sigma = model.implied_volatility(forward, product.k, ttm)?;
        let discount = (-(model.r()) * ttm).exp();
        // Shifted SABR is Black/Bachelier on F + s, K + s.
        let (forward, strike) = (forward + model.shift(), product.k + model.shift());

        let unit_price = if model.beta().abs() < 1.0e-12 {
            self.price_sabr_normal(product.option_type, forward, strike, sigma, discount, ttm)
        } else {
            self.price_sabr_lognormal(product.option_type, forward, strike, sigma, discount, ttm)
        };

        Ok(unit_price)
    }

    fn price_sabr_normal(
        &self,
        option_type: OptionType,
        forward: f64,
        strike: f64,
        sigma: f64,
        discount: f64,
        ttm: f64,
    ) -> f64 {
        let vol_term = sigma * ttm.sqrt();
        if vol_term <= 0.0 {
            return discount * Self::intrinsic(option_type, forward, strike);
        }

        let d = (forward - strike) / vol_term;
        let pdf = norm_pdf(d);
        let cdf = norm_cdf(d);

        match option_type {
            OptionType::Call => discount * ((forward - strike) * cdf + vol_term * pdf),
            OptionType::Put => discount * ((strike - forward) * norm_cdf(-d) + vol_term * pdf),
        }
    }

    fn price_sabr_lognormal(
        &self,
        option_type: OptionType,
        forward: f64,
        strike: f64,
        sigma: f64,
        discount: f64,
        ttm: f64,
    ) -> f64 {
        let vol_term = sigma * ttm.sqrt();
        if vol_term <= 0.0 {
            return discount * Self::intrinsic(option_type, forward, strike);
        }

        let d1 = ((forward / strike).ln() + 0.5 * sigma * sigma * ttm) / vol_term;
        let d2 = d1 - vol_term;

        match option_type {
            OptionType::Call => discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2)),
            OptionType::Put => discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1)),
        }
    }
}

/// Black–Scholes vanilla price for a unit quantity; `ttm <= 0` returns intrinsic value.
pub fn black_scholes_price(
    spot: f64,
    strike: f64,
    ttm: f64,
    option_type: OptionType,
    model: &BlackScholesModel,
) -> f64 {
    if ttm <= 0.0 {
        return AnalyticalEngine::intrinsic(option_type, spot, strike);
    }

    let d1 = model.d1(spot, strike, ttm);
    let d2 = model.d2(spot, strike, ttm);
    let discounted_spot = spot * (-model.q * ttm).exp();
    let discounted_strike = strike * (-model.r * ttm).exp();

    match option_type {
        OptionType::Call => discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2),
        OptionType::Put => discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1),
    }
}

/// Black-76 vanilla price for a unit quantity; `ttm <= 0` returns intrinsic value.
pub fn black_price(
    forward: f64,
    strike: f64,
    ttm: f64,
    option_type: OptionType,
    model: &BlackModel,
) -> f64 {
    if ttm <= 0.0 {
        return AnalyticalEngine::intrinsic(option_type, forward, strike);
    }

    let d1 = model.d1(forward, strike, ttm);
    let d2 = model.d2(forward, strike, ttm);
    let discount = (-model.r * ttm).exp();

    match option_type {
        OptionType::Call => discount * (forward * norm_cdf(d1) - strike * norm_cdf(d2)),
        OptionType::Put => discount * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1)),
    }
}

impl PricingEngine<EuropeanOption, BlackScholesModel> for AnalyticalEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;
        let unit_price = black_scholes_price(
            product.require_spot()?,
            product.k,
            product.ttm(),
            product.option_type,
            model,
        );
        Ok(unit_price * f64::from(product.qty))
    }
}

impl PricingEngine<EuropeanOption, BlackModel> for AnalyticalEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;
        let unit_price = black_price(
            product.require_forward()?,
            product.k,
            product.ttm(),
            product.option_type,
            model,
        );
        Ok(unit_price * f64::from(product.qty))
    }
}

impl PricingEngine<EuropeanOption, SabrModel> for AnalyticalEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &SabrModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;
        let unit_price = self.price_sabr(product, model)?;
        Ok(unit_price * f64::from(product.qty))
    }
}

impl PricingEngine<DigitalOption, BlackScholesModel> for AnalyticalEngine {
    fn calculate_price(
        &self,
        product: &DigitalOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;
        let unit_price = self.price_digital_black_scholes(product, model)?;
        Ok(unit_price * f64::from(product.european.qty))
    }
}

impl PricingEngine<DigitalOption, BlackModel> for AnalyticalEngine {
    fn calculate_price(
        &self,
        product: &DigitalOption,
        model: &BlackModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;
        let unit_price = self.price_digital_black(product, model)?;
        Ok(unit_price * f64::from(product.european.qty))
    }
}

/// Geometric-average Asian under Black–Scholes (Kemna & Vorst 1990), for a
/// unit quantity.
///
/// `n_steps = Some(n)` averages the `n + 1` equally spaced fixings
/// `S_0, S_{T/n}, …, S_T` (the Monte Carlo engine's monitoring);
/// `None` is continuous monitoring. `ln G` is normal with mean
/// `ln S_0 + (r − q − σ²/2)·T/2` and variance `σ²T·(2n + 1)/(6(n + 1))`
/// (`σ²T/3` in the continuous limit), so the price is Black-76 on `G`.
pub fn geometric_asian_black_scholes(
    spot: f64,
    strike: f64,
    ttm: f64,
    model: &BlackScholesModel,
    option_type: OptionType,
    n_steps: Option<usize>,
) -> f64 {
    let discount = (-model.r * ttm).exp();
    if ttm <= 0.0 {
        return AnalyticalEngine::intrinsic(option_type, spot, strike);
    }

    let sigma_sq = model.sigma * model.sigma;
    let variance_factor = n_steps.map_or(1.0 / 3.0, |n| {
        let n = n.max(1) as f64;
        (2.0 * n + 1.0) / (6.0 * (n + 1.0))
    });
    let mean_log = spot.ln() + (model.r - model.q - 0.5 * sigma_sq) * 0.5 * ttm;
    let var_log = sigma_sq * ttm * variance_factor;
    let forward = (mean_log + 0.5 * var_log).exp();
    let sd = var_log.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * var_log) / sd;
    let d2 = d1 - sd;
    discount
        * match option_type {
            OptionType::Call => forward * norm_cdf(d1) - strike * norm_cdf(d2),
            OptionType::Put => strike * norm_cdf(-d2) - forward * norm_cdf(-d1),
        }
}

/// Continuously monitored geometric Asians only; arithmetic averages have no
/// closed form (use `MonteCarloEngine`).
impl PricingEngine<AsianOption, BlackScholesModel> for AnalyticalEngine {
    fn calculate_price(
        &self,
        product: &AsianOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        product.validate()?;
        model.validate()?;
        if product.averaging_type != AveragingType::Geometric {
            return Err(QuantLibError::Engine(
                "analytical engine prices only geometric AsianOptions under BlackScholesModel; \
                 use MonteCarloEngine for arithmetic averages"
                    .to_owned(),
            ));
        }
        let unit = geometric_asian_black_scholes(
            product.s,
            product.k,
            product.ttm(),
            model,
            product.option_type,
            None,
        );
        Ok(unit * f64::from(product.qty))
    }
}

#[cfg(test)]
mod tests {
    use quantlib_core::{OptionType, PricingEngine, SimpleDate};
    use quantlib_models::{BlackModel, BlackScholesModel, SabrModel};
    use quantlib_products::{AsianOption, AveragingType, DigitalOption, DigitalPayoffType};

    use quantlib_core::norm_cdf;

    use super::AnalyticalEngine;

    #[test]
    fn analytical_prices_cash_digital_bs() {
        let option = DigitalOption::new_spot(
            100.0,
            95.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            3,
            12.5,
            DigitalPayoffType::CashOrNothing,
            None,
        )
        .expect("option");
        let model = BlackScholesModel::new(0.25, 0.03, 0.01).expect("model");
        let engine = AnalyticalEngine;

        let ttm = option.ttm();
        let d2 = model.d2(option.require_spot().expect("spot"), option.strike(), ttm);
        let expected_unit = option.payout * (-model.r * ttm).exp() * norm_cdf(d2);
        let expected = expected_unit * 3.0;

        let price = engine.calculate_price(&option, &model).expect("price");
        assert!((price - expected).abs() < 1.0e-10);
    }

    #[test]
    fn analytical_prices_asset_digital_black() {
        let option = DigitalOption::new_forward(
            101.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            2,
            1.2,
            DigitalPayoffType::AssetOrNothing,
            None,
        )
        .expect("option");
        let model = BlackModel::new(0.2, 0.01).expect("model");
        let engine = AnalyticalEngine;

        let ttm = option.ttm();
        let forward = option.require_forward().expect("forward");
        let d1 = model.d1(forward, option.strike(), ttm);
        let expected_unit = option.payout * (-model.r * ttm).exp() * forward * norm_cdf(-d1);
        let expected = expected_unit * 2.0;

        let price = engine.calculate_price(&option, &model).expect("price");
        assert!((price - expected).abs() < 1.0e-10);
    }

    #[test]
    fn analytical_black_requires_forward() {
        let option = quantlib_products::EuropeanOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            None,
        )
        .expect("valid product");

        let model = BlackModel::new(0.2, 0.01).expect("model");
        let engine = AnalyticalEngine;
        let err = engine
            .calculate_price(&option, &model)
            .expect_err("must fail without forward");
        assert!(err.to_string().contains("forward-based pricing requested"));
    }

    #[test]
    fn analytical_sabr_prices_european_option() {
        let option = quantlib_products::EuropeanOption::new_forward(
            101.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            None,
        )
        .expect("valid product");

        let model = SabrModel::normal(0.2, -0.2, 0.5, 0.01, None).expect("model");
        let engine = AnalyticalEngine;
        let price = engine
            .calculate_price(&option, &model)
            .expect("SABR analytical price");
        assert!(price.is_finite() && price > 0.0);
    }

    #[test]
    fn analytical_engine_rejects_arithmetic_asian_option() {
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
        .expect("valid product");
        let model = BlackScholesModel::new(0.2, 0.01, 0.0).expect("valid model");
        let engine = AnalyticalEngine;

        let err = engine
            .calculate_price(&option, &model)
            .expect_err("must reject");
        assert!(
            err.to_string()
                .contains("use MonteCarloEngine for arithmetic averages"),
            "unexpected error: {err}"
        );
    }
}
