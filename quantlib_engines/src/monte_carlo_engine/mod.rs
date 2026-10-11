//! Monte Carlo pricing: one engine, one generic chunked driver ([`paths`]),
//! streaming payoffs, per-chunk random streams and standard errors.

mod core;
mod heston;
mod paths;

pub use core::{HestonScheme, McEstimate, MonteCarloConfig};

use quantlib_core::{OptionType, PricingEngine, QuantLibError};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::{AsianOption, AveragingType, DigitalOption, EuropeanOption};

use crate::analytical_engine::geometric_asian_black_scholes;
use paths::{
    AsianControl, Average, BlackScholesPath, HestonPath, PathFunctional, PathModel, Terminal,
    record_paths, simulate,
};

/// Monte Carlo engine for European, digital and Asian options under
/// Black–Scholes and Heston.
///
/// Use [`MonteCarloEngine::price_with_stats`] for the price with its standard
/// error. Variance reduction is opt-in:
/// - `antithetic`: each normal vector `Z` is paired with `−Z`.
/// - `control_variate`: European and digital options use the discounted
///   terminal spot (`E = S_0 e^{−qT}`); arithmetic Asians under Black–Scholes
///   use the geometric Asian (closed form); other Asians use the arithmetic
///   average of the spot (`E[S_t] = S_0 e^{(r−q)t}`).
///
/// Black–Scholes European and digital options are simulated with one exact
/// log-normal step; Heston and Asian options use `n_steps` steps.
#[derive(Debug, Clone, Copy, Default)]
pub struct MonteCarloEngine {
    pub config: MonteCarloConfig,
}

/// Prices one product/model pair with statistics; see
/// [`MonteCarloEngine::price_with_stats`].
pub trait MonteCarloPricing<P, M> {
    fn price_with_stats(&self, product: &P, model: &M) -> Result<McEstimate, QuantLibError>;
}

/// Simulated spot paths, row-major with `n_steps + 1` points each (`S_0` first).
#[derive(Debug, Clone, PartialEq)]
pub struct SimulatedPaths {
    pub n_steps: usize,
    pub values: Vec<f64>,
}

impl SimulatedPaths {
    pub fn len(&self) -> usize {
        self.values.len() / (self.n_steps + 1)
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn path(&self, index: usize) -> &[f64] {
        let width = self.n_steps + 1;
        &self.values[index * width..(index + 1) * width]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.values.chunks_exact(self.n_steps + 1)
    }
}

impl MonteCarloEngine {
    pub fn new(n_paths: usize, n_steps: usize, seed: Option<u64>) -> Result<Self, QuantLibError> {
        Ok(Self::from_config(MonteCarloConfig::new(
            n_paths, n_steps, seed,
        )?))
    }

    pub fn from_config(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = chunk_size.max(1);
        self
    }

    pub fn with_heston_scheme(mut self, heston_scheme: HestonScheme) -> Self {
        self.config.heston_scheme = heston_scheme;
        self
    }

    pub fn with_antithetic(mut self, antithetic: bool) -> Self {
        self.config.antithetic = antithetic;
        self
    }

    pub fn with_control_variate(mut self, control_variate: bool) -> Self {
        self.config.control_variate = control_variate;
        self
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        self.config.validate()
    }

    /// The price with its standard error and path count.
    pub fn price_with_stats<P, M>(
        &self,
        product: &P,
        model: &M,
    ) -> Result<McEstimate, QuantLibError>
    where
        Self: MonteCarloPricing<P, M>,
    {
        MonteCarloPricing::price_with_stats(self, product, model)
    }

    /// Simulates `n_paths` full spot paths under Black–Scholes. Meant for
    /// plotting and diagnostics; pricing streams payoffs and keeps no paths.
    pub fn sample_paths_black_scholes(
        &self,
        spot: f64,
        model: &BlackScholesModel,
        ttm: f64,
    ) -> Result<SimulatedPaths, QuantLibError> {
        self.validate_path_inputs(spot, ttm)?;
        model.validate()?;
        let n_steps = self.config.n_steps;
        let path = BlackScholesPath::new(model, spot, ttm / n_steps as f64);
        Ok(self.record(&path, n_steps))
    }

    /// Heston counterpart of [`Self::sample_paths_black_scholes`].
    pub fn sample_paths_heston(
        &self,
        spot: f64,
        model: &HestonModel,
        ttm: f64,
    ) -> Result<SimulatedPaths, QuantLibError> {
        self.validate_path_inputs(spot, ttm)?;
        model.validate()?;
        let n_steps = self.config.n_steps;
        let path = HestonPath::new(model, spot, ttm / n_steps as f64, self.config.heston_scheme);
        Ok(self.record(&path, n_steps))
    }

    fn record<M: PathModel>(&self, path: &M, n_steps: usize) -> SimulatedPaths {
        SimulatedPaths {
            n_steps,
            values: record_paths(&self.config, self.config.resolved_seed(), path, n_steps),
        }
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

    /// Runs the driver and discounts; `control_mean` is `E[X]` for the
    /// functional's control, used when control variates are enabled.
    fn estimate<M: PathModel, F: PathFunctional>(
        &self,
        path: &M,
        functional: &F,
        n_steps: usize,
        discount: f64,
        control_mean: f64,
    ) -> McEstimate {
        let moments = simulate(
            &self.config,
            self.config.resolved_seed(),
            path,
            functional,
            n_steps,
        );
        let control = self.config.control_variate.then_some(control_mean);
        moments.estimate(discount, control, self.config.simulated_paths())
    }

    /// Asian options under either model: running averages over `n_steps` steps.
    fn price_asian<M: PathModel>(
        &self,
        product: &AsianOption,
        path: &M,
        rate: f64,
        dividend_yield: f64,
        control: AsianControl,
        geometric_control_mean: f64,
    ) -> McEstimate {
        let ttm = product.ttm();
        let n_steps = self.config.n_steps;
        let (strike, qty) = (product.k, f64::from(product.qty));
        let option_type = product.option_type;
        let functional = Average {
            geometric: product.averaging_type == AveragingType::Geometric,
            control,
            payoff: move |average: f64| qty * vanilla_payoff(option_type, average, strike),
        };
        let control_mean = match control {
            AsianControl::GeometricPayoff => geometric_control_mean,
            AsianControl::ArithmeticAverage => {
                expected_arithmetic_average(product.s, rate - dividend_yield, ttm, n_steps)
            }
        };
        self.estimate(
            path,
            &functional,
            n_steps,
            (-rate * ttm).exp(),
            control_mean,
        )
    }
}

impl<P, M> PricingEngine<P, M> for MonteCarloEngine
where
    MonteCarloEngine: MonteCarloPricing<P, M>,
{
    fn calculate_price(&self, product: &P, model: &M) -> Result<f64, QuantLibError> {
        Ok(self.price_with_stats(product, model)?.price)
    }
}

fn vanilla_payoff(option_type: OptionType, underlying: f64, strike: f64) -> f64 {
    match option_type {
        OptionType::Call => (underlying - strike).max(0.0),
        OptionType::Put => (strike - underlying).max(0.0),
    }
}

/// `E[(1/(n+1)) Σ_{i=0}^{n} S_{iT/n}]` with `E[S_t] = S_0 e^{(r−q)t}`.
fn expected_arithmetic_average(spot: f64, carry: f64, ttm: f64, n_steps: usize) -> f64 {
    let dt = ttm / n_steps as f64;
    let sum: f64 = (0..=n_steps).map(|i| (carry * dt * i as f64).exp()).sum();
    spot * sum / (n_steps + 1) as f64
}

impl MonteCarloPricing<EuropeanOption, BlackScholesModel> for MonteCarloEngine {
    fn price_with_stats(
        &self,
        product: &EuropeanOption,
        model: &BlackScholesModel,
    ) -> Result<McEstimate, QuantLibError> {
        self.validate()?;
        product.validate()?;
        model.validate()?;
        let (spot, ttm) = (product.require_spot()?, product.ttm());
        let (strike, qty, option_type) = (product.k, f64::from(product.qty), product.option_type);
        let functional = Terminal {
            payoff: move |s_t: f64| qty * vanilla_payoff(option_type, s_t, strike),
        };
        let forward = spot * ((model.r - model.q) * ttm).exp();
        Ok(self.estimate(
            &BlackScholesPath::new(model, spot, ttm),
            &functional,
            1,
            (-model.r * ttm).exp(),
            forward,
        ))
    }
}

impl MonteCarloPricing<EuropeanOption, HestonModel> for MonteCarloEngine {
    fn price_with_stats(
        &self,
        product: &EuropeanOption,
        model: &HestonModel,
    ) -> Result<McEstimate, QuantLibError> {
        self.validate()?;
        product.validate()?;
        model.validate()?;
        let (spot, ttm) = (product.require_spot()?, product.ttm());
        let n_steps = self.config.n_steps;
        let (strike, qty, option_type) = (product.k, f64::from(product.qty), product.option_type);
        let functional = Terminal {
            payoff: move |s_t: f64| qty * vanilla_payoff(option_type, s_t, strike),
        };
        let path = HestonPath::new(model, spot, ttm / n_steps as f64, self.config.heston_scheme);
        let forward = spot * ((model.r - model.q) * ttm).exp();
        Ok(self.estimate(&path, &functional, n_steps, (-model.r * ttm).exp(), forward))
    }
}

impl MonteCarloPricing<DigitalOption, BlackScholesModel> for MonteCarloEngine {
    fn price_with_stats(
        &self,
        product: &DigitalOption,
        model: &BlackScholesModel,
    ) -> Result<McEstimate, QuantLibError> {
        self.validate()?;
        product.validate()?;
        model.validate()?;
        let (spot, ttm) = (product.require_spot()?, product.ttm());
        let qty = f64::from(product.european.qty);
        let functional = Terminal {
            payoff: |s_t: f64| qty * product.unit_payoff_at_maturity(s_t),
        };
        let forward = spot * ((model.r - model.q) * ttm).exp();
        Ok(self.estimate(
            &BlackScholesPath::new(model, spot, ttm),
            &functional,
            1,
            (-model.r * ttm).exp(),
            forward,
        ))
    }
}

impl MonteCarloPricing<DigitalOption, HestonModel> for MonteCarloEngine {
    fn price_with_stats(
        &self,
        product: &DigitalOption,
        model: &HestonModel,
    ) -> Result<McEstimate, QuantLibError> {
        self.validate()?;
        product.validate()?;
        model.validate()?;
        let (spot, ttm) = (product.require_spot()?, product.ttm());
        let n_steps = self.config.n_steps;
        let qty = f64::from(product.european.qty);
        let functional = Terminal {
            payoff: |s_t: f64| qty * product.unit_payoff_at_maturity(s_t),
        };
        let path = HestonPath::new(model, spot, ttm / n_steps as f64, self.config.heston_scheme);
        let forward = spot * ((model.r - model.q) * ttm).exp();
        Ok(self.estimate(&path, &functional, n_steps, (-model.r * ttm).exp(), forward))
    }
}

impl MonteCarloPricing<AsianOption, BlackScholesModel> for MonteCarloEngine {
    fn price_with_stats(
        &self,
        product: &AsianOption,
        model: &BlackScholesModel,
    ) -> Result<McEstimate, QuantLibError> {
        self.validate()?;
        product.validate()?;
        model.validate()?;
        let ttm = product.ttm();
        let n_steps = self.config.n_steps;
        let path = BlackScholesPath::new(model, product.s, ttm / n_steps as f64);
        // The geometric Asian is the natural control for an arithmetic one.
        let (control, geometric_mean) = if product.averaging_type == AveragingType::Arithmetic {
            let unit = geometric_asian_black_scholes(
                product.s,
                product.k,
                ttm,
                model,
                product.option_type,
                Some(n_steps),
            );
            (
                AsianControl::GeometricPayoff,
                f64::from(product.qty) * unit * (model.r * ttm).exp(),
            )
        } else {
            (AsianControl::ArithmeticAverage, 0.0)
        };
        Ok(self.price_asian(product, &path, model.r, model.q, control, geometric_mean))
    }
}

impl MonteCarloPricing<AsianOption, HestonModel> for MonteCarloEngine {
    fn price_with_stats(
        &self,
        product: &AsianOption,
        model: &HestonModel,
    ) -> Result<McEstimate, QuantLibError> {
        self.validate()?;
        product.validate()?;
        model.validate()?;
        let ttm = product.ttm();
        let n_steps = self.config.n_steps;
        let path = HestonPath::new(
            model,
            product.s,
            ttm / n_steps as f64,
            self.config.heston_scheme,
        );
        Ok(self.price_asian(
            product,
            &path,
            model.r,
            model.q,
            AsianControl::ArithmeticAverage,
            0.0,
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

    use super::MonteCarloEngine;
    use super::core::{ChunkSums, Moments, Sample};
    use crate::analytical_engine::AnalyticalEngine;

    fn dates() -> (SimpleDate, SimpleDate) {
        (
            SimpleDate::new(2026, 2, 10).expect("date"),
            SimpleDate::new(2027, 2, 10).expect("date"),
        )
    }

    fn european(option_type: OptionType) -> EuropeanOption {
        let (valuation, expiry) = dates();
        EuropeanOption::new_spot(100.0, 100.0, expiry, Some(valuation), option_type, 1, None)
            .expect("option")
    }

    fn asian(option_type: OptionType, averaging: AveragingType) -> AsianOption {
        let (valuation, expiry) = dates();
        AsianOption::new(
            100.0,
            100.0,
            expiry,
            Some(valuation),
            option_type,
            1,
            averaging,
            None,
        )
        .expect("option")
    }

    #[test]
    fn merged_chunk_moments_match_one_pass() {
        let samples: Vec<Sample> = (0..1000)
            .map(|i| {
                let t = f64::from(i);
                Sample {
                    y: (t * 0.37).sin() * 10.0 + 5.0,
                    x: (t * 0.11).cos() + t * 1.0e-3,
                }
            })
            .collect();

        let mut whole = ChunkSums::default();
        samples.iter().for_each(|s| whole.push(*s));
        let whole = whole.moments();
        let merged = samples
            .chunks(77)
            .map(|chunk| {
                let mut sums = ChunkSums::default();
                chunk.iter().for_each(|s| sums.push(*s));
                sums.moments()
            })
            .fold(Moments::default(), Moments::merge);

        for control in [None, Some(0.4)] {
            let (a, b) = (
                whole.estimate(0.9, control, 1000),
                merged.estimate(0.9, control, 1000),
            );
            assert!((a.price - b.price).abs() < 1.0e-12, "{a:?} vs {b:?}");
            assert!(
                (a.std_error - b.std_error).abs() < 1.0e-12,
                "{a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn mc_with_seed_is_reproducible() {
        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
        let engine = MonteCarloEngine::new(50_000, 252, Some(42)).expect("engine");
        let option = european(OptionType::Call);
        let p1 = engine.calculate_price(&option, &model).expect("price");
        let p2 = engine.calculate_price(&option, &model).expect("price");
        assert_eq!(p1.to_bits(), p2.to_bits());
    }

    #[test]
    fn mc_parallel_matches_serial_for_seeded_bs() {
        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
        let option = european(OptionType::Call);
        let engine = MonteCarloEngine::new(30_000, 252, Some(42))
            .expect("engine")
            .with_chunk_size(1_024);

        let serial = engine
            .with_parallel(false)
            .price_with_stats(&option, &model);
        let parallel = engine.with_parallel(true).price_with_stats(&option, &model);
        assert_eq!(serial.expect("serial"), parallel.expect("parallel"));
    }

    #[test]
    fn mc_parallel_matches_serial_for_seeded_heston() {
        let model = HestonModel::new(1.5, 0.04, 0.4, -0.6, 0.04, 0.05, 0.01).expect("model");
        let option = asian(OptionType::Put, AveragingType::Arithmetic);
        let engine = MonteCarloEngine::new(20_000, 252, Some(777))
            .expect("engine")
            .with_chunk_size(512)
            .with_antithetic(true)
            .with_control_variate(true);

        let serial = engine
            .with_parallel(false)
            .price_with_stats(&option, &model);
        let parallel = engine.with_parallel(true).price_with_stats(&option, &model);
        assert_eq!(serial.expect("serial"), parallel.expect("parallel"));
    }

    #[test]
    fn mc_european_is_close_to_analytical_for_large_paths() {
        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
        let option = european(OptionType::Call);
        let estimate = MonteCarloEngine::new(200_000, 252, Some(7))
            .expect("mc")
            .price_with_stats(&option, &model)
            .expect("mc price");
        let bs = AnalyticalEngine
            .calculate_price(&option, &model)
            .expect("bs price");
        assert!(
            (estimate.price - bs).abs() < 4.0 * estimate.std_error,
            "mc={estimate:?}, bs={bs}"
        );
    }

    #[test]
    fn mc_asian_with_zero_vol_matches_discounted_deterministic_payoff() {
        let option = asian(OptionType::Call, AveragingType::Arithmetic);
        // Near-zero vol because the model constructor enforces sigma > 0.
        let model = BlackScholesModel::new(1.0e-12, 0.05, 0.01).expect("model");
        let engine = MonteCarloEngine::new(5_000, 252, Some(9)).expect("engine");
        let price = engine.calculate_price(&option, &model).expect("price");

        let t = option.ttm();
        let growth = ((model.r - model.q) * t / 252.0).exp();
        let mut sum = 0.0;
        let mut s = option.s;
        for _ in 0..=252 {
            sum += s;
            s *= growth;
        }
        let expected = (-model.r * t).exp() * (sum / 253.0 - option.k).max(0.0);
        assert!(
            (price - expected).abs() < 1.0e-6,
            "price={price}, expected={expected}"
        );
    }

    #[test]
    fn mc_heston_european_deterministic_zero_variance() {
        let option = european(OptionType::Call);
        let model = HestonModel::new(1.5, 0.0, 0.4, -0.6, 0.0, 0.05, 0.01).expect("model");
        let estimate = MonteCarloEngine::new(10_000, 252, Some(777))
            .expect("engine")
            .price_with_stats(&option, &model)
            .expect("price");

        let t = option.ttm();
        let s_t = 100.0 * ((model.r - model.q) * t).exp();
        let expected = (-model.r * t).exp() * (s_t - option.k).max(0.0);
        assert!(
            (estimate.price - expected).abs() < 1.0e-10,
            "price={}, expected={expected}",
            estimate.price
        );
        assert!(estimate.std_error < 1.0e-10, "{estimate:?}");
    }

    #[test]
    fn mc_digital_bs_is_close_to_analytical() {
        let (valuation, expiry) = dates();
        let option = DigitalOption::new_spot(
            100.0,
            100.0,
            expiry,
            Some(valuation),
            OptionType::Call,
            1,
            10.0,
            DigitalPayoffType::CashOrNothing,
            None,
        )
        .expect("option");
        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
        let estimate = MonteCarloEngine::new(400_000, 252, Some(99))
            .expect("mc")
            .price_with_stats(&option, &model)
            .expect("mc price");
        let bs = AnalyticalEngine
            .calculate_price(&option, &model)
            .expect("bs price");
        assert!(
            (estimate.price - bs).abs() < 4.0 * estimate.std_error,
            "mc={estimate:?}, bs={bs}"
        );
    }

    #[test]
    fn antithetic_counts_both_legs_of_each_pair() {
        let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
        let estimate = MonteCarloEngine::new(10_001, 1, Some(3))
            .expect("engine")
            .with_antithetic(true)
            .price_with_stats(&european(OptionType::Call), &model)
            .expect("price");
        assert_eq!(estimate.n_paths, 10_002);
    }
}
