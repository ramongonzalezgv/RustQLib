mod asian_engine;
mod core;
mod digital_engine;
mod european_engine;
mod simulators;

pub use asian_engine::MonteCarloAsianEngine;
pub use core::MonteCarloConfig;
pub use digital_engine::MonteCarloDigitalEngine;
pub use european_engine::MonteCarloEuropeanEngine;

use quantlib_core::{PricingEngine, QuantLibError};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::{AmericanOption, AsianOption, DigitalOption, EuropeanOption};
use simulators::{BlackScholesSimulator, HestonSimulator};

#[derive(Debug, Clone, Copy)]
pub struct MonteCarloEngine {
    pub n_paths: usize,
    pub n_steps: usize,
    pub seed: Option<u64>,
    pub parallel: bool,
    pub chunk_size: usize,
}

impl Default for MonteCarloEngine {
    fn default() -> Self {
        Self::from_config(MonteCarloConfig::default())
    }
}

impl MonteCarloEngine {
    pub fn new(n_paths: usize, n_steps: usize, seed: Option<u64>) -> Result<Self, QuantLibError> {
        let config = MonteCarloConfig::new(n_paths, n_steps, seed)?;
        Ok(Self::from_config(config))
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        self.config().validate()
    }

    /// Simulates `n_paths` full spot paths of `n_steps + 1` points each
    /// (including `spot` at t = 0) under Black–Scholes. Meant for plotting and
    /// diagnostics; pricing uses the engines, which do not keep the paths.
    pub fn sample_paths_black_scholes(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
    ) -> Result<Vec<Vec<f64>>, QuantLibError> {
        self.validate_path_inputs(spot, ttm)?;
        model.validate()?;
        let config = self.config();
        let mut rng = config.rng();
        Ok(BlackScholesSimulator::new(config).simulate_paths(spot, model, ttm, &mut rng))
    }

    /// Heston counterpart of [`Self::sample_paths_black_scholes`].
    pub fn sample_paths_heston(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
    ) -> Result<Vec<Vec<f64>>, QuantLibError> {
        self.validate_path_inputs(spot, ttm)?;
        model.validate()?;
        let config = self.config();
        let mut rng = config.rng();
        Ok(HestonSimulator::new(config).simulate_paths(spot, model, ttm, &mut rng))
    }

    fn validate_path_inputs(&self, spot: f64, ttm: f64) -> Result<(), QuantLibError> {
        self.validate()?;
        if !(spot.is_finite() && spot > 0.0) {
            return Err(QuantLibError::Validation(
                "spot must be finite and positive".to_owned(),
            ));
        }
        if !(ttm.is_finite() && ttm > 0.0) {
            return Err(QuantLibError::Validation(
                "time to maturity must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }

    fn config(&self) -> MonteCarloConfig {
        MonteCarloConfig {
            n_paths: self.n_paths,
            n_steps: self.n_steps,
            seed: self.seed,
            parallel: self.parallel,
            chunk_size: self.chunk_size,
        }
    }

    fn from_config(config: MonteCarloConfig) -> Self {
        Self {
            n_paths: config.n_paths,
            n_steps: config.n_steps,
            seed: config.seed,
            parallel: config.parallel,
            chunk_size: config.chunk_size,
        }
    }
}

impl PricingEngine<EuropeanOption, BlackScholesModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;
        let engine = MonteCarloEuropeanEngine::from_config(self.config());
        engine.calculate_price(product, model)
    }
}

impl PricingEngine<AsianOption, BlackScholesModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        product: &AsianOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;
        let engine = MonteCarloAsianEngine::from_config(self.config());
        engine.calculate_price(product, model)
    }
}

impl PricingEngine<DigitalOption, BlackScholesModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        product: &DigitalOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;
        let engine = MonteCarloDigitalEngine::from_config(self.config());
        engine.calculate_price(product, model)
    }
}

impl PricingEngine<AmericanOption, BlackScholesModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        _product: &AmericanOption,
        _model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        Err(QuantLibError::Engine(
            "MonteCarloEngine does not support AmericanOption under BlackScholesModel".to_owned(),
        ))
    }
}

impl PricingEngine<EuropeanOption, HestonModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;
        let engine = MonteCarloEuropeanEngine::from_config(self.config());
        engine.calculate_price(product, model)
    }
}

impl PricingEngine<AsianOption, HestonModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        product: &AsianOption,
        model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;
        let engine = MonteCarloAsianEngine::from_config(self.config());
        engine.calculate_price(product, model)
    }
}

impl PricingEngine<DigitalOption, HestonModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        product: &DigitalOption,
        model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;
        let engine = MonteCarloDigitalEngine::from_config(self.config());
        engine.calculate_price(product, model)
    }
}

impl PricingEngine<AmericanOption, HestonModel> for MonteCarloEngine {
    fn calculate_price(
        &self,
        _product: &AmericanOption,
        _model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        Err(QuantLibError::Engine(
            "MonteCarloEngine does not support AmericanOption under HestonModel".to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use quantlib_core::{OptionType, PricingEngine, SimpleDate};
    use quantlib_models::{BlackScholesModel, HestonModel};
    use quantlib_products::{
        AsianOption, AveragingType, DigitalOption, DigitalPayoffType, EuropeanOption,
    };

    use crate::analytical_engine::AnalyticalEngine;

    use super::{
        MonteCarloAsianEngine, MonteCarloDigitalEngine, MonteCarloEngine, MonteCarloEuropeanEngine,
    };

    #[test]
    fn mc_with_seed_is_reproducible() {
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

        let engine = MonteCarloEngine::new(50_000, 252, Some(42)).expect("engine");
        let p1 = engine.calculate_price(&option, &model).expect("price");
        let p2 = engine.calculate_price(&option, &model).expect("price");

        assert!((p1 - p2).abs() < 1.0e-12, "p1={p1}, p2={p2}");
    }

    #[test]
    fn mc_parallel_matches_serial_for_seeded_bs() {
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

        let serial = MonteCarloEngine::new(30_000, 252, Some(42))
            .expect("engine")
            .with_parallel(false);
        let parallel = MonteCarloEngine::new(30_000, 252, Some(42))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(1_024);

        let p_serial = serial.calculate_price(&option, &model).expect("serial");
        let p_parallel = parallel.calculate_price(&option, &model).expect("parallel");
        assert!(
            (p_serial - p_parallel).abs() < 1.0e-12,
            "serial={p_serial}, parallel={p_parallel}"
        );
    }

    #[test]
    fn mc_parallel_matches_serial_for_seeded_heston() {
        let option = AsianOption::new(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            1,
            AveragingType::Arithmetic,
            None,
        )
        .expect("option");
        let model = HestonModel::new(1.5, 0.04, 0.4, -0.6, 0.04, 0.05, 0.01).expect("model");

        let serial = MonteCarloEngine::new(20_000, 252, Some(777))
            .expect("engine")
            .with_parallel(false);
        let parallel = MonteCarloEngine::new(20_000, 252, Some(777))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(512);

        let p_serial = serial.calculate_price(&option, &model).expect("serial");
        let p_parallel = parallel.calculate_price(&option, &model).expect("parallel");
        assert!(
            (p_serial - p_parallel).abs() < 1.0e-12,
            "serial={p_serial}, parallel={p_parallel}"
        );
    }

    #[test]
    fn mc_european_is_close_to_analytical_for_large_paths() {
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

        let mc_engine = MonteCarloEngine::new(200_000, 252, Some(7)).expect("mc");
        let analytical = AnalyticalEngine;

        let mc_price = mc_engine
            .calculate_price(&option, &model)
            .expect("mc price");
        let bs_price = analytical
            .calculate_price(&option, &model)
            .expect("bs price");

        assert!(
            (mc_price - bs_price).abs() < 0.15,
            "mc={mc_price}, bs={bs_price}"
        );
    }

    #[test]
    fn mc_asian_with_zero_vol_matches_discounted_deterministic_payoff() {
        let asian = AsianOption::new(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            AveragingType::Arithmetic,
            None,
        )
        .expect("asian option");

        // Use near-zero vol because model constructor enforces sigma > 0.
        let model = BlackScholesModel::new(1.0e-12, 0.05, 0.01).expect("model");
        let engine = MonteCarloEngine::new(5_000, 252, Some(9)).expect("engine");

        let price = engine.calculate_price(&asian, &model).expect("price");

        let t = asian.ttm();
        let dt = t / 252.0;
        let drift_step = (model.r - model.q - 0.5 * model.sigma * model.sigma) * dt;
        let growth = drift_step.exp();
        let mut sum = 0.0;
        let mut s = asian.s;
        for _ in 0..=252 {
            sum += s;
            s *= growth;
        }
        let avg = sum / 253.0;
        let expected = (-model.r * t).exp() * (avg - asian.k).max(0.0);

        assert!(
            (price - expected).abs() < 1.0e-3,
            "price={price}, expected={expected}"
        );
    }

    #[test]
    fn mc_heston_european_deterministic_zero_variance() {
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

        let model = HestonModel::new(1.5, 0.0, 0.4, -0.6, 0.0, 0.05, 0.01).expect("model");
        let engine = MonteCarloEngine::new(10_000, 252, Some(777)).expect("engine");

        let price = engine.calculate_price(&option, &model).expect("price");

        let t = option.ttm();
        let s_t = option.require_spot().expect("spot quote") * ((model.r - model.q) * t).exp();
        let expected = (-model.r * t).exp() * (s_t - option.k).max(0.0);

        assert!(
            (price - expected).abs() < 1.0e-10,
            "price={price}, expected={expected}"
        );
    }

    #[test]
    fn dedicated_european_engine_matches_facade() {
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

        let facade = MonteCarloEngine::new(40_000, 252, Some(42))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(1_024);
        let specialized = MonteCarloEuropeanEngine::new(40_000, 252, Some(42))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(1_024);

        let p_facade = facade.calculate_price(&option, &model).expect("facade");
        let p_specialized = specialized
            .calculate_price(&option, &model)
            .expect("specialized");

        assert!((p_facade - p_specialized).abs() < 1.0e-12);
    }

    #[test]
    fn dedicated_asian_engine_matches_facade() {
        let option = AsianOption::new(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            1,
            AveragingType::Arithmetic,
            None,
        )
        .expect("option");
        let model = HestonModel::new(1.5, 0.04, 0.4, -0.6, 0.04, 0.05, 0.01).expect("model");

        let facade = MonteCarloEngine::new(20_000, 252, Some(777))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(512);
        let specialized = MonteCarloAsianEngine::new(20_000, 252, Some(777))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(512);

        let p_facade = facade.calculate_price(&option, &model).expect("facade");
        let p_specialized = specialized
            .calculate_price(&option, &model)
            .expect("specialized");

        assert!((p_facade - p_specialized).abs() < 1.0e-12);
    }

    #[test]
    fn mc_digital_bs_is_close_to_analytical() {
        let option = DigitalOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Call,
            1,
            10.0,
            DigitalPayoffType::CashOrNothing,
            None,
        )
        .expect("option");
        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");

        let mc_engine = MonteCarloEngine::new(400_000, 252, Some(99)).expect("mc");
        let analytical = AnalyticalEngine;

        let mc_price = mc_engine
            .calculate_price(&option, &model)
            .expect("mc price");
        let bs_price = analytical
            .calculate_price(&option, &model)
            .expect("bs price");

        assert!(
            (mc_price - bs_price).abs() < 0.03,
            "mc={mc_price}, bs={bs_price}"
        );
    }

    #[test]
    fn dedicated_digital_engine_matches_facade() {
        let option = DigitalOption::new_spot(
            100.0,
            100.0,
            SimpleDate::new(2027, 2, 10).expect("date"),
            Some(SimpleDate::new(2026, 2, 10).expect("date")),
            OptionType::Put,
            1,
            1.0,
            DigitalPayoffType::AssetOrNothing,
            None,
        )
        .expect("option");
        let model = HestonModel::new(1.5, 0.04, 0.4, -0.6, 0.04, 0.05, 0.01).expect("model");

        let facade = MonteCarloEngine::new(30_000, 252, Some(808))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(512);
        let specialized = MonteCarloDigitalEngine::new(30_000, 252, Some(808))
            .expect("engine")
            .with_parallel(true)
            .with_chunk_size(512);

        let p_facade = facade.calculate_price(&option, &model).expect("facade");
        let p_specialized = specialized
            .calculate_price(&option, &model)
            .expect("specialized");

        assert!((p_facade - p_specialized).abs() < 1.0e-12);
    }
}
