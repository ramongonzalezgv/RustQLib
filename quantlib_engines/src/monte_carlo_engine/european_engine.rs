use quantlib_core::{FinancialProduct, PricingEngine, QuantLibError};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::EuropeanOption;

use super::core::{HestonScheme, MonteCarloConfig, discount_and_average};
use super::simulators::{BlackScholesSimulator, HestonSimulator};

#[derive(Debug, Clone, Copy, Default)]
pub struct MonteCarloEuropeanEngine {
    pub config: MonteCarloConfig,
}

impl MonteCarloEuropeanEngine {
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

impl PricingEngine<EuropeanOption, BlackScholesModel> for MonteCarloEuropeanEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
    ) -> Result<f64, QuantLibError> {
        self.config.validate()?;
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let ttm = product.ttm();
        let mut rng = self.config.rng();
        let final_spots =
            BlackScholesSimulator::new(self.config).simulate_terminal(spot, model, ttm, &mut rng);
        let payoffs = product.payoff(&final_spots);

        discount_and_average(&payoffs, model.r, ttm)
    }
}

impl PricingEngine<EuropeanOption, HestonModel> for MonteCarloEuropeanEngine {
    fn calculate_price(
        &self,
        product: &EuropeanOption,
        model: &HestonModel,
    ) -> Result<f64, QuantLibError> {
        self.config.validate()?;
        product.validate()?;
        model.validate()?;

        let spot = product.require_spot()?;
        let ttm = product.ttm();
        let mut rng = self.config.rng();
        let final_spots =
            HestonSimulator::new(self.config).simulate_terminal(spot, model, ttm, &mut rng);
        let payoffs = product.payoff(&final_spots);

        discount_and_average(&payoffs, model.r, ttm)
    }
}
