use quantlib_core::{PricingEngine, QuantLibError};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::AsianOption;

use super::core::{HestonScheme, MonteCarloConfig, discount_and_average};
use super::simulators::{BlackScholesSimulator, HestonSimulator};

#[derive(Debug, Clone, Copy, Default)]
pub struct MonteCarloAsianEngine {
    pub config: MonteCarloConfig,
}

impl MonteCarloAsianEngine {
    pub fn new(n_paths: usize, n_steps: usize, seed: Option<u64>) -> Result<Self, QuantLibError> {
        Ok(Self {
            config: MonteCarloConfig::new(n_paths, n_steps, seed)?,
        })
    }

    pub fn from_config(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.config = self.config.with_parallel(parallel);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.config = self.config.with_chunk_size(chunk_size);
        self
    }

    pub fn with_heston_scheme(mut self, heston_scheme: HestonScheme) -> Self {
        self.config = self.config.with_heston_scheme(heston_scheme);
        self
    }
}

impl PricingEngine<AsianOption, BlackScholesModel> for MonteCarloAsianEngine {
    fn calculate_price(
        &self,
        product: &AsianOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        self.config.validate()?;
        product.validate()?;
        model.validate()?;

        let ttm = product.ttm();
        let mut rng = self.config.rng();
        let paths =
            BlackScholesSimulator::new(self.config).simulate_paths(product.s, model, ttm, &mut rng);
        let payoffs = product.payoff_from_paths(&paths)?;

        discount_and_average(&payoffs, model.r, ttm)
    }
}

impl PricingEngine<AsianOption, HestonModel> for MonteCarloAsianEngine {
    fn calculate_price(
        &self,
        product: &AsianOption,
        model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        self.config.validate()?;
        product.validate()?;
        model.validate()?;

        let ttm = product.ttm();
        let mut rng = self.config.rng();
        let paths =
            HestonSimulator::new(self.config).simulate_paths(product.s, model, ttm, &mut rng);
        let payoffs = product.payoff_from_paths(&paths)?;

        discount_and_average(&payoffs, model.r, ttm)
    }
}
