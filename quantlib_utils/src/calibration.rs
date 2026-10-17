use std::collections::BTreeMap;

use quantlib_core::{CalibrationResult, OptionType, QuantLibError};
use quantlib_engines::{FFTEngine, InterpolationKind, black_price, black_scholes_price};
use quantlib_models::{BlackModel, BlackScholesModel, HestonModel, SabrModel};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const PENALTY_OBJECTIVE: f64 = 1.0e300;
type EvaluationVectors = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

pub const BLACK_PARAMETER_NAMES: [&str; 1] = ["sigma"];
pub const BLACK_SCHOLES_PARAMETER_NAMES: [&str; 1] = ["sigma"];
pub const SABR_PARAMETER_NAMES: [&str; 4] = ["alpha", "beta", "rho", "nu"];
pub const HESTON_PARAMETER_NAMES: [&str; 5] = ["kappa", "theta", "sigma", "rho", "v0"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    DifferentialEvolution,
    NelderMead,
    HybridGlobalLocal,
}

#[derive(Debug, Clone, Copy)]
pub enum ObjectiveKind {
    LeastSquares,
    RelativeLeastSquares,
    Huber { delta: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationTarget {
    Price,
    Volatility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VanillaMarketValueKind {
    Price,
    ImpliedVolatility,
}

#[derive(Debug, Clone, Copy)]
pub struct DifferentialEvolutionConfig {
    pub max_generations: usize,
    pub population_size: usize,
    pub differential_weight: f64,
    pub crossover_probability: f64,
    pub tolerance: f64,
    pub seed: u64,
    pub parallel_objective: bool,
}

impl Default for DifferentialEvolutionConfig {
    fn default() -> Self {
        Self {
            max_generations: 350,
            population_size: 48,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            tolerance: 1.0e-6,
            seed: 42,
            parallel_objective: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NelderMeadConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub initial_simplex_scale: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            max_iterations: 300,
            tolerance: 1.0e-7,
            initial_simplex_scale: 0.08,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HestonFftConfig {
    pub n: usize,
    pub b: f64,
    pub interp: InterpolationKind,
}

impl Default for HestonFftConfig {
    fn default() -> Self {
        Self {
            n: 1 << 11,
            b: FFTEngine::DEFAULT_B,
            interp: InterpolationKind::Cubic,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CalibrationConfig {
    pub method: CalibrationMethod,
    pub objective: ObjectiveKind,
    pub differential_evolution: DifferentialEvolutionConfig,
    pub nelder_mead: NelderMeadConfig,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            method: CalibrationMethod::HybridGlobalLocal,
            objective: ObjectiveKind::LeastSquares,
            differential_evolution: DifferentialEvolutionConfig::default(),
            nelder_mead: NelderMeadConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VanillaOptionCalibrationQuote {
    pub underlying: f64,
    pub strike: f64,
    pub maturity: f64,
    pub option_type: OptionType,
    pub market_value: f64,
    pub market_value_kind: VanillaMarketValueKind,
    pub weight: f64,
}

impl VanillaOptionCalibrationQuote {
    #[allow(clippy::too_many_arguments)]
    pub fn price(
        underlying: f64,
        strike: f64,
        maturity: f64,
        option_type: OptionType,
        market_value: f64,
        weight: f64,
    ) -> Self {
        Self {
            underlying,
            strike,
            maturity,
            option_type,
            market_value,
            market_value_kind: VanillaMarketValueKind::Price,
            weight,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility(
        underlying: f64,
        strike: f64,
        maturity: f64,
        option_type: OptionType,
        market_value: f64,
        weight: f64,
    ) -> Self {
        Self {
            underlying,
            strike,
            maturity,
            option_type,
            market_value,
            market_value_kind: VanillaMarketValueKind::ImpliedVolatility,
            weight,
        }
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.underlying.is_finite() || self.underlying <= 0.0 {
            return Err(QuantLibError::Validation(
                "quote underlying must be finite and positive".to_owned(),
            ));
        }

        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(QuantLibError::Validation(
                "quote strike must be finite and positive".to_owned(),
            ));
        }

        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err(QuantLibError::Validation(
                "quote maturity must be finite and positive".to_owned(),
            ));
        }

        if !self.market_value.is_finite() {
            return Err(QuantLibError::Validation(
                "quote market value must be finite".to_owned(),
            ));
        }
        match self.market_value_kind {
            VanillaMarketValueKind::Price => {
                if self.market_value < 0.0 {
                    return Err(QuantLibError::Validation(
                        "price quote market value must be non-negative".to_owned(),
                    ));
                }
            }
            VanillaMarketValueKind::ImpliedVolatility => {
                if self.market_value <= 0.0 {
                    return Err(QuantLibError::Validation(
                        "implied-volatility quote market value must be positive".to_owned(),
                    ));
                }
            }
        }

        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(QuantLibError::Validation(
                "quote weight must be finite and positive".to_owned(),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SabrCalibrationQuote {
    pub forward: f64,
    pub strike: f64,
    pub maturity: f64,
    pub market_volatility: f64,
    pub weight: f64,
}

impl SabrCalibrationQuote {
    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.forward.is_finite() || self.forward <= 0.0 {
            return Err(QuantLibError::Validation(
                "SABR quote forward must be finite and positive".to_owned(),
            ));
        }

        if !self.strike.is_finite() || self.strike <= 0.0 {
            return Err(QuantLibError::Validation(
                "SABR quote strike must be finite and positive".to_owned(),
            ));
        }

        if !self.maturity.is_finite() || self.maturity <= 0.0 {
            return Err(QuantLibError::Validation(
                "SABR quote maturity must be finite and positive".to_owned(),
            ));
        }

        if !self.market_volatility.is_finite() || self.market_volatility <= 0.0 {
            return Err(QuantLibError::Validation(
                "SABR quote market volatility must be finite and positive".to_owned(),
            ));
        }

        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(QuantLibError::Validation(
                "SABR quote weight must be finite and positive".to_owned(),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_params: Vec<f64>,
    pub best_objective: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub converged: bool,
    pub message: String,
}

pub fn default_black_bounds() -> [(f64, f64); 1] {
    [(1.0e-4, 3.0)]
}

pub fn default_black_scholes_bounds() -> [(f64, f64); 1] {
    [(1.0e-4, 3.0)]
}

pub fn default_sabr_bounds() -> [(f64, f64); 4] {
    [(1.0e-4, 3.0), (0.01, 0.99), (-0.999, 0.999), (1.0e-4, 3.0)]
}

pub fn default_heston_bounds() -> [(f64, f64); 5] {
    [
        (1.0e-4, 20.0),
        (1.0e-5, 2.0),
        (1.0e-4, 5.0),
        (-0.999, 0.999),
        (1.0e-5, 2.0),
    ]
}

/// Black-76 σ calibration with fixed `r`; `Volatility` is rejected until implied-vol calibration (#19).
pub fn calibrate_black_model(
    quotes: &[VanillaOptionCalibrationQuote],
    target: CalibrationTarget,
    initial_sigma: f64,
    r: f64,
    bounds: Option<[(f64, f64); 1]>,
    config: &CalibrationConfig,
) -> Result<(BlackModel, CalibrationResult), QuantLibError> {
    reject_volatility_target(target)?;
    validate_vanilla_quotes(quotes)?;
    validate_vanilla_quote_kinds(quotes, target)?;

    let bounds = bounds.unwrap_or_else(default_black_bounds).to_vec();
    let initial = vec![initial_sigma];
    validate_bounds(&bounds, &initial, &BLACK_PARAMETER_NAMES)?;
    BlackModel::new(initial_sigma, r)?;

    let evaluate = |model: &BlackModel| {
        evaluate_vanilla_quotes(quotes, |q| {
            black_price(q.underlying, q.strike, q.maturity, q.option_type, model)
        })
    };
    let objective = |params: &[f64]| -> Result<f64, QuantLibError> {
        let (_, market_values, errors, weights) = evaluate(&BlackModel::new(params[0], r)?);
        Ok(compute_objective(
            config.objective,
            &errors,
            &market_values,
            &weights,
        ))
    };

    let optimization = optimize_bounded(&objective, &initial, &bounds, config)?;
    let model = BlackModel::new(optimization.best_params[0], r)?;

    let (model_values, market_values, errors, _) = evaluate(&model);
    let result = build_calibration_result(
        &BLACK_PARAMETER_NAMES,
        &optimization.best_params,
        &optimization,
        errors,
        market_values,
        model_values,
    );

    Ok((model, result))
}

/// Black–Scholes σ calibration with fixed `r`, `q`; `Volatility` is rejected until #19.
pub fn calibrate_black_scholes_model(
    quotes: &[VanillaOptionCalibrationQuote],
    target: CalibrationTarget,
    initial_sigma: f64,
    r: f64,
    q: f64,
    bounds: Option<[(f64, f64); 1]>,
    config: &CalibrationConfig,
) -> Result<(BlackScholesModel, CalibrationResult), QuantLibError> {
    reject_volatility_target(target)?;
    validate_vanilla_quotes(quotes)?;
    validate_vanilla_quote_kinds(quotes, target)?;

    let bounds = bounds.unwrap_or_else(default_black_scholes_bounds).to_vec();
    let initial = vec![initial_sigma];
    validate_bounds(&bounds, &initial, &BLACK_SCHOLES_PARAMETER_NAMES)?;
    BlackScholesModel::new(initial_sigma, r, q)?;

    let evaluate = |model: &BlackScholesModel| {
        evaluate_vanilla_quotes(quotes, |quote| {
            black_scholes_price(
                quote.underlying,
                quote.strike,
                quote.maturity,
                quote.option_type,
                model,
            )
        })
    };
    let objective = |params: &[f64]| -> Result<f64, QuantLibError> {
        let (_, market_values, errors, weights) =
            evaluate(&BlackScholesModel::new(params[0], r, q)?);
        Ok(compute_objective(
            config.objective,
            &errors,
            &market_values,
            &weights,
        ))
    };

    let optimization = optimize_bounded(&objective, &initial, &bounds, config)?;
    let model = BlackScholesModel::new(optimization.best_params[0], r, q)?;

    let (model_values, market_values, errors, _) = evaluate(&model);
    let result = build_calibration_result(
        &BLACK_SCHOLES_PARAMETER_NAMES,
        &optimization.best_params,
        &optimization,
        errors,
        market_values,
        model_values,
    );

    Ok((model, result))
}

#[allow(clippy::too_many_arguments)]
pub fn calibrate_sabr_model(
    quotes: &[SabrCalibrationQuote],
    initial_alpha: f64,
    initial_beta: f64,
    initial_rho: f64,
    initial_nu: f64,
    r: f64,
    bounds: Option<[(f64, f64); 4]>,
    config: &CalibrationConfig,
) -> Result<(SabrModel, CalibrationResult), QuantLibError> {
    validate_sabr_quotes(quotes)?;
    if !r.is_finite() {
        return Err(QuantLibError::Validation(
            "SABR calibration rate r must be finite".to_owned(),
        ));
    }

    let bounds = bounds.unwrap_or_else(default_sabr_bounds).to_vec();
    let initial = vec![initial_alpha, initial_beta, initial_rho, initial_nu];
    validate_bounds(&bounds, &initial, &SABR_PARAMETER_NAMES)?;

    let objective = |params: &[f64]| -> Result<f64, QuantLibError> {
        let model = SabrModel::general(params[0], params[1], params[2], params[3], r, None)?;
        let (_, market_values, errors, weights) = evaluate_sabr_quotes(&model, quotes)?;
        Ok(compute_objective(
            config.objective,
            &errors,
            &market_values,
            &weights,
        ))
    };

    let optimization = optimize_bounded(&objective, &initial, &bounds, config)?;
    let model = SabrModel::general(
        optimization.best_params[0],
        optimization.best_params[1],
        optimization.best_params[2],
        optimization.best_params[3],
        r,
        None,
    )?;

    let (model_values, market_values, errors, _) = evaluate_sabr_quotes(&model, quotes)?;
    let result = build_calibration_result(
        &SABR_PARAMETER_NAMES,
        &optimization.best_params,
        &optimization,
        errors,
        market_values,
        model_values,
    );

    Ok((model, result))
}

pub fn calibrate_heston_model(
    quotes: &[VanillaOptionCalibrationQuote],
    initial_model: HestonModel,
    bounds: Option<[(f64, f64); 5]>,
    config: &CalibrationConfig,
    fft_config: HestonFftConfig,
) -> Result<(HestonModel, CalibrationResult), QuantLibError> {
    validate_vanilla_quotes(quotes)?;
    validate_vanilla_quote_kinds(quotes, CalibrationTarget::Price)?;

    let bounds = bounds.unwrap_or_else(default_heston_bounds).to_vec();
    let initial = vec![
        initial_model.kappa,
        initial_model.theta,
        initial_model.sigma,
        initial_model.rho,
        initial_model.v0,
    ];
    validate_bounds(&bounds, &initial, &HESTON_PARAMETER_NAMES)?;
    initial_model.validate()?;
    let (r, q) = (initial_model.r, initial_model.q);

    let engine = FFTEngine::new(fft_config.n, fft_config.b, fft_config.interp)?;
    let strips = group_heston_strips(quotes);

    let objective = |params: &[f64]| -> Result<f64, QuantLibError> {
        let model = HestonModel::new(params[0], params[1], params[2], params[3], params[4], r, q)?;

        let (_, market_values, errors, weights) =
            evaluate_heston_quotes(&engine, &model, &strips, quotes)?;
        Ok(compute_objective(
            config.objective,
            &errors,
            &market_values,
            &weights,
        ))
    };

    let optimization = optimize_bounded(&objective, &initial, &bounds, config)?;
    let model = HestonModel::new(
        optimization.best_params[0],
        optimization.best_params[1],
        optimization.best_params[2],
        optimization.best_params[3],
        optimization.best_params[4],
        r,
        q,
    )?;

    let (model_values, market_values, errors, _) =
        evaluate_heston_quotes(&engine, &model, &strips, quotes)?;

    let result = build_calibration_result(
        &HESTON_PARAMETER_NAMES,
        &optimization.best_params,
        &optimization,
        errors,
        market_values,
        model_values,
    );

    Ok((model, result))
}

fn validate_vanilla_quotes(quotes: &[VanillaOptionCalibrationQuote]) -> Result<(), QuantLibError> {
    if quotes.is_empty() {
        return Err(QuantLibError::Validation(
            "at least one calibration quote is required".to_owned(),
        ));
    }

    for (index, quote) in quotes.iter().enumerate() {
        quote.validate().map_err(|error| {
            QuantLibError::Validation(format!("invalid quote at index {index}: {error}"))
        })?;
    }

    Ok(())
}

fn validate_vanilla_quote_kinds(
    quotes: &[VanillaOptionCalibrationQuote],
    target: CalibrationTarget,
) -> Result<(), QuantLibError> {
    for (index, quote) in quotes.iter().enumerate() {
        let expected = match target {
            CalibrationTarget::Price => VanillaMarketValueKind::Price,
            CalibrationTarget::Volatility => VanillaMarketValueKind::ImpliedVolatility,
        };

        if quote.market_value_kind != expected {
            let expected_str = match expected {
                VanillaMarketValueKind::Price => "price",
                VanillaMarketValueKind::ImpliedVolatility => "implied_volatility",
            };
            let found_str = match quote.market_value_kind {
                VanillaMarketValueKind::Price => "price",
                VanillaMarketValueKind::ImpliedVolatility => "implied_volatility",
            };
            return Err(QuantLibError::Validation(format!(
                "quote at index {index} has market_value_kind='{found_str}', expected '{expected_str}' for target {target:?}"
            )));
        }
    }

    Ok(())
}

fn validate_sabr_quotes(quotes: &[SabrCalibrationQuote]) -> Result<(), QuantLibError> {
    if quotes.is_empty() {
        return Err(QuantLibError::Validation(
            "at least one SABR calibration quote is required".to_owned(),
        ));
    }

    for (index, quote) in quotes.iter().enumerate() {
        quote.validate().map_err(|error| {
            QuantLibError::Validation(format!("invalid SABR quote at index {index}: {error}"))
        })?;
    }

    Ok(())
}

fn validate_bounds(
    bounds: &[(f64, f64)],
    initial: &[f64],
    names: &[&str],
) -> Result<(), QuantLibError> {
    if bounds.len() != initial.len() || bounds.len() != names.len() {
        return Err(QuantLibError::Validation(format!(
            "bounds ({})/initial ({})/names ({}) length mismatch",
            bounds.len(),
            initial.len(),
            names.len()
        )));
    }

    for (index, ((lower, upper), initial_value)) in bounds.iter().zip(initial.iter()).enumerate() {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(QuantLibError::Validation(format!(
                "bound for {} must be finite",
                names[index]
            )));
        }
        if lower > upper {
            return Err(QuantLibError::Validation(format!(
                "bound for {} must satisfy lower <= upper",
                names[index]
            )));
        }

        if !initial_value.is_finite() {
            return Err(QuantLibError::Validation(format!(
                "initial value for {} must be finite",
                names[index]
            )));
        }

        if *initial_value < *lower || *initial_value > *upper {
            return Err(QuantLibError::Validation(format!(
                "initial value for {} ({initial_value}) must be inside bounds [{lower}, {upper}]",
                names[index]
            )));
        }
    }

    Ok(())
}

fn reject_volatility_target(target: CalibrationTarget) -> Result<(), QuantLibError> {
    if target == CalibrationTarget::Volatility {
        return Err(QuantLibError::Validation(
            "Volatility target needs implied-vol calibration (#19); the constant-vol model would only fit a flat vol".to_owned(),
        ));
    }
    Ok(())
}

fn evaluate_vanilla_quotes(
    quotes: &[VanillaOptionCalibrationQuote],
    price: impl Fn(&VanillaOptionCalibrationQuote) -> f64,
) -> EvaluationVectors {
    let model_values: Vec<f64> = quotes.iter().map(price).collect();
    let market_values: Vec<f64> = quotes.iter().map(|quote| quote.market_value).collect();
    let errors = model_values
        .iter()
        .zip(&market_values)
        .map(|(model_value, market_value)| model_value - market_value)
        .collect();
    let weights = quotes.iter().map(|quote| quote.weight).collect();

    (model_values, market_values, errors, weights)
}

fn evaluate_sabr_quotes(
    model: &SabrModel,
    quotes: &[SabrCalibrationQuote],
) -> Result<EvaluationVectors, QuantLibError> {
    let mut model_values = Vec::with_capacity(quotes.len());
    let mut market_values = Vec::with_capacity(quotes.len());
    let mut errors = Vec::with_capacity(quotes.len());
    let mut weights = Vec::with_capacity(quotes.len());

    for quote in quotes {
        let model_value = model.implied_volatility(quote.forward, quote.strike, quote.maturity)?;
        let error = model_value - quote.market_volatility;

        model_values.push(model_value);
        market_values.push(quote.market_volatility);
        errors.push(error);
        weights.push(quote.weight);
    }

    Ok((model_values, market_values, errors, weights))
}

/// Quotes sharing one exact maturity and underlying, priced from one FFT strip.
struct HestonStrip {
    underlying: f64,
    maturity: f64,
    quote_indices: Vec<usize>,
    strikes: Vec<f64>,
    option_types: Vec<OptionType>,
}

/// Groups quotes by bit-exact `(maturity, underlying)` once, before optimization.
fn group_heston_strips(quotes: &[VanillaOptionCalibrationQuote]) -> Vec<HestonStrip> {
    let mut strips: Vec<HestonStrip> = Vec::new();
    let mut index_by_key = BTreeMap::new();

    for (quote_index, quote) in quotes.iter().enumerate() {
        let key = (quote.maturity.to_bits(), quote.underlying.to_bits());
        let strip_index = *index_by_key.entry(key).or_insert_with(|| {
            strips.push(HestonStrip {
                underlying: quote.underlying,
                maturity: quote.maturity,
                quote_indices: Vec::new(),
                strikes: Vec::new(),
                option_types: Vec::new(),
            });
            strips.len() - 1
        });

        let strip = &mut strips[strip_index];
        strip.quote_indices.push(quote_index);
        strip.strikes.push(quote.strike);
        strip.option_types.push(quote.option_type);
    }

    strips
}

fn evaluate_heston_quotes(
    engine: &FFTEngine,
    model: &HestonModel,
    strips: &[HestonStrip],
    quotes: &[VanillaOptionCalibrationQuote],
) -> Result<EvaluationVectors, QuantLibError> {
    let mut model_values = vec![0.0; quotes.len()];
    for strip in strips {
        let prices = engine.price_strip(
            model,
            strip.underlying,
            strip.maturity,
            &strip.strikes,
            &strip.option_types,
        )?;
        for (&quote_index, price) in strip.quote_indices.iter().zip(prices) {
            model_values[quote_index] = price;
        }
    }

    let market_values: Vec<f64> = quotes.iter().map(|quote| quote.market_value).collect();
    let errors = model_values
        .iter()
        .zip(&market_values)
        .map(|(model_value, market_value)| model_value - market_value)
        .collect();
    let weights = quotes.iter().map(|quote| quote.weight).collect();

    Ok((model_values, market_values, errors, weights))
}

fn optimize_bounded<F>(
    objective: &F,
    initial_guess: &[f64],
    bounds: &[(f64, f64)],
    config: &CalibrationConfig,
) -> Result<OptimizationResult, QuantLibError>
where
    F: Fn(&[f64]) -> Result<f64, QuantLibError> + Sync,
{
    match config.method {
        CalibrationMethod::DifferentialEvolution => differential_evolution(
            objective,
            initial_guess,
            bounds,
            &config.differential_evolution,
        ),
        CalibrationMethod::NelderMead => {
            nelder_mead(objective, initial_guess, bounds, &config.nelder_mead)
        }
        CalibrationMethod::HybridGlobalLocal => {
            let global = differential_evolution(
                objective,
                initial_guess,
                bounds,
                &config.differential_evolution,
            )?;

            let mut local =
                nelder_mead(objective, &global.best_params, bounds, &config.nelder_mead)?;
            local.evaluations += global.evaluations;
            local.iterations += global.iterations;
            if !local.converged {
                if global.converged {
                    local.converged = true;
                    local.message = format!(
                        "global stage converged; local refinement stopped early: {}",
                        local.message
                    );
                } else {
                    local.message = format!(
                        "global stage did not converge; local stage: {}",
                        local.message
                    );
                }
            }
            Ok(local)
        }
    }
}

fn differential_evolution<F>(
    objective: &F,
    initial_guess: &[f64],
    bounds: &[(f64, f64)],
    config: &DifferentialEvolutionConfig,
) -> Result<OptimizationResult, QuantLibError>
where
    F: Fn(&[f64]) -> Result<f64, QuantLibError> + Sync,
{
    if bounds.is_empty() {
        return Err(QuantLibError::Validation(
            "differential evolution requires at least one parameter".to_owned(),
        ));
    }

    let dim = bounds.len();
    let population_size = config.population_size.max(8).max(dim * 6);
    let max_generations = config.max_generations.max(1);
    let crossover_probability = config.crossover_probability.clamp(0.05, 1.0);
    let differential_weight = config.differential_weight.clamp(0.1, 1.9);

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut population = vec![vec![0.0; dim]; population_size];

    for individual in &mut population {
        for (coordinate, (lower, upper)) in individual.iter_mut().zip(bounds.iter()) {
            *coordinate = if (*upper - *lower).abs() < 1.0e-14 {
                *lower
            } else {
                rng.random_range(*lower..=*upper)
            };
        }
    }

    population[0] = project_to_bounds(initial_guess, bounds);

    let mut evaluations = 0;
    let mut objective_values =
        evaluate_population(objective, &population, config.parallel_objective);
    evaluations += objective_values.len();

    let (mut best_index, mut best_objective) = argmin_index(&objective_values)
        .ok_or_else(|| QuantLibError::Engine("empty population objective values".to_owned()))?;

    for generation in 0..max_generations {
        let mut trials = Vec::with_capacity(population_size);

        for candidate_index in 0..population_size {
            let [a, b, c] =
                random_three_distinct_indices(population_size, candidate_index, &mut rng);

            let random_dimension = rng.random_range(0..dim);
            let mut trial = population[candidate_index].clone();

            for dimension in 0..dim {
                let should_cross =
                    rng.random::<f64>() < crossover_probability || dimension == random_dimension;
                if should_cross {
                    let mutated = population[a][dimension]
                        + differential_weight
                            * (population[b][dimension] - population[c][dimension]);
                    trial[dimension] = clamp(mutated, bounds[dimension].0, bounds[dimension].1);
                }
            }

            trials.push(trial);
        }

        let trial_values = evaluate_population(objective, &trials, config.parallel_objective);
        evaluations += trial_values.len();

        for (index, trial_value) in trial_values.iter().enumerate() {
            if *trial_value <= objective_values[index] {
                population[index] = trials[index].clone();
                objective_values[index] = *trial_value;
            }
        }

        let (new_best_index, new_best_value) = argmin_index(&objective_values)
            .ok_or_else(|| QuantLibError::Engine("empty objective values".to_owned()))?;
        best_index = new_best_index;
        best_objective = new_best_value;

        if has_de_converged(&objective_values, config.tolerance) {
            return Ok(OptimizationResult {
                best_params: population[best_index].clone(),
                best_objective,
                iterations: generation + 1,
                evaluations,
                converged: true,
                message: "differential evolution converged by population objective spread"
                    .to_owned(),
            });
        }
    }

    Ok(OptimizationResult {
        best_params: population[best_index].clone(),
        best_objective,
        iterations: max_generations,
        evaluations,
        converged: false,
        message: "differential evolution hit max generations".to_owned(),
    })
}

fn nelder_mead<F>(
    objective: &F,
    initial_guess: &[f64],
    bounds: &[(f64, f64)],
    config: &NelderMeadConfig,
) -> Result<OptimizationResult, QuantLibError>
where
    F: Fn(&[f64]) -> Result<f64, QuantLibError> + Sync,
{
    if bounds.is_empty() {
        return Err(QuantLibError::Validation(
            "nelder-mead requires at least one parameter".to_owned(),
        ));
    }

    let dim = bounds.len();
    let max_iterations = config.max_iterations.max(1);
    let simplex_scale = config.initial_simplex_scale.max(1.0e-4);

    let mut simplex = Vec::with_capacity(dim + 1);
    simplex.push(project_to_bounds(initial_guess, bounds));

    for dimension in 0..dim {
        let mut vertex = simplex[0].clone();
        let width = (bounds[dimension].1 - bounds[dimension].0).abs();
        let step = simplex_scale * width.max(1.0);

        vertex[dimension] = clamp(
            vertex[dimension] + step,
            bounds[dimension].0,
            bounds[dimension].1,
        );
        if (vertex[dimension] - simplex[0][dimension]).abs() < 1.0e-14 && width > 1.0e-14 {
            vertex[dimension] = clamp(
                simplex[0][dimension] - step,
                bounds[dimension].0,
                bounds[dimension].1,
            );
        }

        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex
        .iter()
        .map(|vertex| evaluate_objective(objective, vertex))
        .collect();
    let mut evaluations = values.len();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let shrink = 0.5;

    for iteration in 0..max_iterations {
        order_simplex(&mut simplex, &mut values);

        if has_nm_converged(&simplex, &values, config.tolerance) {
            return Ok(OptimizationResult {
                best_params: simplex[0].clone(),
                best_objective: values[0],
                iterations: iteration + 1,
                evaluations,
                converged: true,
                message: "nelder-mead converged by simplex size and objective spread".to_owned(),
            });
        }

        let centroid = centroid_except_worst(&simplex);
        let worst = simplex[dim].clone();

        let reflected = project_to_bounds(
            &add_scaled(&centroid, &subtract(&centroid, &worst), alpha),
            bounds,
        );
        let reflected_value = evaluate_objective(objective, &reflected);
        evaluations += 1;

        if reflected_value < values[0] {
            let expanded = project_to_bounds(
                &add_scaled(&centroid, &subtract(&reflected, &centroid), gamma),
                bounds,
            );
            let expanded_value = evaluate_objective(objective, &expanded);
            evaluations += 1;

            if expanded_value < reflected_value {
                simplex[dim] = expanded;
                values[dim] = expanded_value;
            } else {
                simplex[dim] = reflected;
                values[dim] = reflected_value;
            }
            continue;
        }

        if reflected_value < values[dim - 1] {
            simplex[dim] = reflected;
            values[dim] = reflected_value;
            continue;
        }

        let contract_source = if reflected_value < values[dim] {
            reflected
        } else {
            worst.clone()
        };

        let contracted = project_to_bounds(
            &add_scaled(&centroid, &subtract(&contract_source, &centroid), rho),
            bounds,
        );
        let contracted_value = evaluate_objective(objective, &contracted);
        evaluations += 1;

        if contracted_value < values[dim] {
            simplex[dim] = contracted;
            values[dim] = contracted_value;
            continue;
        }

        let best = simplex[0].clone();
        for index in 1..=dim {
            simplex[index] = project_to_bounds(
                &add_scaled(&best, &subtract(&simplex[index], &best), shrink),
                bounds,
            );
            values[index] = evaluate_objective(objective, &simplex[index]);
            evaluations += 1;
        }
    }

    order_simplex(&mut simplex, &mut values);

    Ok(OptimizationResult {
        best_params: simplex[0].clone(),
        best_objective: values[0],
        iterations: max_iterations,
        evaluations,
        converged: false,
        message: "nelder-mead hit max iterations".to_owned(),
    })
}

fn evaluate_population<F>(objective: &F, population: &[Vec<f64>], parallel: bool) -> Vec<f64>
where
    F: Fn(&[f64]) -> Result<f64, QuantLibError> + Sync,
{
    if parallel {
        population
            .par_iter()
            .map(|individual| evaluate_objective(objective, individual))
            .collect()
    } else {
        population
            .iter()
            .map(|individual| evaluate_objective(objective, individual))
            .collect()
    }
}

fn evaluate_objective<F>(objective: &F, params: &[f64]) -> f64
where
    F: Fn(&[f64]) -> Result<f64, QuantLibError> + Sync,
{
    match objective(params) {
        Ok(value) if value.is_finite() => value,
        _ => PENALTY_OBJECTIVE,
    }
}

fn has_de_converged(objective_values: &[f64], tolerance: f64) -> bool {
    if objective_values.len() < 2 {
        return true;
    }

    let mean = objective_values.iter().sum::<f64>() / objective_values.len() as f64;
    let variance = objective_values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / objective_values.len() as f64;

    variance.sqrt() <= tolerance.max(1.0e-14) * (1.0 + mean.abs())
}

fn has_nm_converged(simplex: &[Vec<f64>], values: &[f64], tolerance: f64) -> bool {
    if simplex.is_empty() || values.is_empty() {
        return true;
    }

    let best_value = values[0];
    let max_value_spread = values
        .iter()
        .map(|value| (value - best_value).abs())
        .fold(0.0_f64, f64::max);

    let best_vertex = &simplex[0];
    let max_relative_distance = simplex
        .iter()
        .map(|vertex| {
            vertex
                .iter()
                .zip(best_vertex.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs() / (1.0 + rhs.abs()))
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);

    let value_tolerance = tolerance.max(1.0e-14) * (1.0 + best_value.abs());
    max_value_spread <= value_tolerance && max_relative_distance <= tolerance.max(1.0e-14)
}

fn argmin_index(values: &[f64]) -> Option<(usize, f64)> {
    values
        .iter()
        .enumerate()
        .min_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, value)| (index, *value))
}

fn random_three_distinct_indices(
    upper: usize,
    excluded: usize,
    rng: &mut ChaCha8Rng,
) -> [usize; 3] {
    let mut picks = [usize::MAX; 3];
    let mut filled = 0;

    while filled < 3 {
        let candidate = rng.random_range(0..upper);
        if candidate == excluded || picks[..filled].contains(&candidate) {
            continue;
        }

        picks[filled] = candidate;
        filled += 1;
    }

    picks
}

fn order_simplex(simplex: &mut [Vec<f64>], values: &mut [f64]) {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|lhs, rhs| {
        values[*lhs]
            .partial_cmp(&values[*rhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_simplex = indices
        .iter()
        .map(|index| simplex[*index].clone())
        .collect::<Vec<_>>();
    let sorted_values = indices
        .iter()
        .map(|index| values[*index])
        .collect::<Vec<_>>();

    simplex.clone_from_slice(&sorted_simplex);
    values.clone_from_slice(&sorted_values);
}

fn centroid_except_worst(simplex: &[Vec<f64>]) -> Vec<f64> {
    let dim = simplex[0].len();
    let mut centroid = vec![0.0; dim];

    for vertex in simplex.iter().take(simplex.len() - 1) {
        for (coordinate, value) in centroid.iter_mut().zip(vertex.iter()) {
            *coordinate += *value;
        }
    }

    let scale = 1.0 / (simplex.len() as f64 - 1.0);
    for coordinate in &mut centroid {
        *coordinate *= scale;
    }

    centroid
}

fn subtract(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect()
}

fn add_scaled(base: &[f64], delta: &[f64], scale: f64) -> Vec<f64> {
    base.iter()
        .zip(delta.iter())
        .map(|(value, shift)| value + scale * shift)
        .collect()
}

fn project_to_bounds(point: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    point
        .iter()
        .zip(bounds.iter())
        .map(|(value, (lower, upper))| clamp(*value, *lower, *upper))
        .collect()
}

fn clamp(value: f64, lower: f64, upper: f64) -> f64 {
    if value < lower {
        lower
    } else if value > upper {
        upper
    } else {
        value
    }
}

fn compute_objective(
    objective_kind: ObjectiveKind,
    errors: &[f64],
    market_values: &[f64],
    weights: &[f64],
) -> f64 {
    match objective_kind {
        ObjectiveKind::LeastSquares => errors
            .iter()
            .zip(weights.iter())
            .map(|(error, weight)| weight * error * error)
            .sum(),
        ObjectiveKind::RelativeLeastSquares => errors
            .iter()
            .zip(market_values.iter())
            .zip(weights.iter())
            .map(|((error, market), weight)| {
                let scaled_error = if market.abs() > 1.0e-10 {
                    error / market
                } else {
                    *error
                };
                weight * scaled_error * scaled_error
            })
            .sum(),
        ObjectiveKind::Huber { delta } => {
            let delta = delta.max(1.0e-8);
            errors
                .iter()
                .zip(weights.iter())
                .map(|(error, weight)| {
                    let abs_error = error.abs();
                    let loss = if abs_error <= delta {
                        0.5 * abs_error * abs_error
                    } else {
                        delta * (abs_error - 0.5 * delta)
                    };
                    weight * loss
                })
                .sum()
        }
    }
}

fn build_calibration_result(
    parameter_names: &[&str],
    parameters: &[f64],
    optimization: &OptimizationResult,
    errors: Vec<f64>,
    market_values: Vec<f64>,
    model_values: Vec<f64>,
) -> CalibrationResult {
    let rmse = if errors.is_empty() {
        0.0
    } else {
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt()
    };

    let mut mape_sum = 0.0;
    let mut mape_count = 0;
    for (error, market) in errors.iter().zip(market_values.iter()) {
        if market.abs() > 1.0e-10 {
            mape_sum += (error / market).abs();
            mape_count += 1;
        }
    }
    let mape = if mape_count > 0 {
        mape_sum / mape_count as f64
    } else {
        f64::INFINITY
    };

    let max_error = errors
        .iter()
        .map(|error| error.abs())
        .fold(0.0_f64, f64::max);

    let calibrated_params = parameter_names
        .iter()
        .zip(parameters.iter())
        .map(|(name, value)| ((*name).to_owned(), *value))
        .collect::<BTreeMap<_, _>>();

    CalibrationResult {
        success: optimization.converged,
        calibrated_params,
        objective_value: optimization.best_objective,
        errors,
        market_values,
        model_values,
        rmse,
        mape,
        max_error,
        iterations: optimization.iterations,
        message: optimization.message.clone(),
    }
}

#[cfg(test)]
mod tests {
    use quantlib_core::{OptionType, QuantLibError};
    use quantlib_engines::{InterpolationKind, black_price, black_scholes_price};

    use super::{
        CalibrationConfig, CalibrationMethod, CalibrationTarget, DifferentialEvolutionConfig,
        HestonFftConfig, NelderMeadConfig, ObjectiveKind, SabrCalibrationQuote,
        VanillaMarketValueKind, VanillaOptionCalibrationQuote, calibrate_black_model,
        calibrate_black_scholes_model, calibrate_heston_model, calibrate_sabr_model,
        default_heston_bounds,
    };

    #[test]
    fn calibrates_black_scholes_from_price_quotes() {
        let true_sigma = 0.24;
        let true_r = 0.03;
        let true_q = 0.01;

        let maturities = [0.25, 0.5, 1.0, 1.5];
        let strikes = [85.0, 95.0, 100.0, 105.0, 115.0];
        let mut quotes = Vec::new();

        let true_model = quantlib_models::BlackScholesModel::new(true_sigma, true_r, true_q)
            .expect("true model");

        for maturity in maturities {
            for strike in strikes {
                for option_type in [OptionType::Call, OptionType::Put] {
                    let quote = VanillaOptionCalibrationQuote {
                        underlying: 100.0,
                        strike,
                        maturity,
                        option_type,
                        market_value: black_scholes_price(
                            100.0,
                            strike,
                            maturity,
                            option_type,
                            &true_model,
                        ),
                        market_value_kind: VanillaMarketValueKind::Price,
                        weight: 1.0,
                    };
                    quotes.push(quote);
                }
            }
        }

        let config = CalibrationConfig {
            method: CalibrationMethod::HybridGlobalLocal,
            objective: ObjectiveKind::LeastSquares,
            ..CalibrationConfig::default()
        };

        let (calibrated, result) = calibrate_black_scholes_model(
            &quotes,
            CalibrationTarget::Price,
            0.18,
            true_r,
            true_q,
            None,
            &config,
        )
        .expect("calibration");

        assert!(result.success, "{}", result.message);
        assert!((calibrated.sigma - true_sigma).abs() < 1.0e-2);
    }

    #[test]
    fn calibrates_black_from_price_quotes() {
        let true_sigma = 0.27;
        let true_r = 0.02;

        let maturities = [0.3, 0.75, 1.25];
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let true_model = quantlib_models::BlackModel::new(true_sigma, true_r).expect("true model");

        let mut quotes = Vec::new();
        for maturity in maturities {
            for strike in strikes {
                for option_type in [OptionType::Call, OptionType::Put] {
                    let quote = VanillaOptionCalibrationQuote {
                        underlying: 100.0,
                        strike,
                        maturity,
                        option_type,
                        market_value: black_price(
                            100.0,
                            strike,
                            maturity,
                            option_type,
                            &true_model,
                        ),
                        market_value_kind: VanillaMarketValueKind::Price,
                        weight: 1.0,
                    };
                    quotes.push(quote);
                }
            }
        }

        let config = CalibrationConfig {
            method: CalibrationMethod::HybridGlobalLocal,
            objective: ObjectiveKind::LeastSquares,
            ..CalibrationConfig::default()
        };

        let (calibrated, result) = calibrate_black_model(
            &quotes,
            CalibrationTarget::Price,
            0.18,
            true_r,
            None,
            &config,
        )
        .expect("calibration");

        assert!(result.success, "{}", result.message);
        assert!((calibrated.sigma - true_sigma).abs() < 1.5e-2);
    }

    #[test]
    fn calibrates_sabr_surface() {
        let true_model =
            quantlib_models::SabrModel::general(0.35, 0.6, -0.3, 0.8, 0.01, None).expect("sabr");

        let maturities = [0.5, 1.0, 2.0];
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let mut quotes = Vec::new();

        for maturity in maturities {
            for strike in strikes {
                let market_vol = true_model
                    .implied_volatility(100.0, strike, maturity)
                    .expect("vol");
                quotes.push(SabrCalibrationQuote {
                    forward: 100.0,
                    strike,
                    maturity,
                    market_volatility: market_vol,
                    weight: 1.0,
                });
            }
        }

        let config = CalibrationConfig {
            method: CalibrationMethod::HybridGlobalLocal,
            objective: ObjectiveKind::RelativeLeastSquares,
            ..CalibrationConfig::default()
        };

        let bounds = [(0.05, 1.0), (0.6, 0.6), (-0.95, 0.2), (0.1, 1.6)];

        let (calibrated, result) =
            calibrate_sabr_model(&quotes, 0.2, 0.6, -0.1, 0.5, 0.01, Some(bounds), &config)
                .expect("calibration");

        assert!(result.success, "{}", result.message);

        let params = calibrated.get_parameters_map();
        let alpha = params
            .get("alpha")
            .expect("alpha")
            .parse::<f64>()
            .expect("alpha parse");
        let beta = params
            .get("beta")
            .expect("beta")
            .parse::<f64>()
            .expect("beta parse");
        let rho = params
            .get("rho")
            .expect("rho")
            .parse::<f64>()
            .expect("rho parse");
        let nu = params
            .get("nu")
            .expect("nu")
            .parse::<f64>()
            .expect("nu parse");

        assert!((alpha - 0.35).abs() < 3.0e-2);
        assert!((beta - 0.6).abs() < 3.0e-2);
        assert!((rho - -0.3).abs() < 3.0e-2);
        assert!((nu - 0.8).abs() < 4.0e-2);
    }

    #[test]
    fn calibrates_heston_surface_from_fft_prices() {
        let true_model = quantlib_models::HestonModel::new(1.8, 0.05, 0.45, -0.65, 0.04, 0.02, 0.0)
            .expect("heston");

        let fft = quantlib_engines::FFTEngine::new(1 << 9, 120.0, InterpolationKind::Cubic)
            .expect("engine");

        let maturities: [f64; 2] = [0.5, 1.0];
        let strikes = [90.0, 100.0, 110.0];

        let mut quotes = Vec::new();
        for maturity in maturities {
            for strike in strikes {
                for option_type in [OptionType::Call, OptionType::Put] {
                    let market_value = fft
                        .price_strip(&true_model, 100.0, maturity, &[strike], &[option_type])
                        .expect("price")[0];

                    quotes.push(VanillaOptionCalibrationQuote {
                        underlying: 100.0,
                        strike,
                        maturity,
                        option_type,
                        market_value,
                        market_value_kind: VanillaMarketValueKind::Price,
                        weight: 1.0,
                    });
                }
            }
        }

        let mut bounds = default_heston_bounds();
        // Keep weakly identified long-term moments fixed in this compact synthetic set.
        bounds[0] = (true_model.kappa, true_model.kappa);
        bounds[1] = (true_model.theta, true_model.theta);

        let initial_model =
            quantlib_models::HestonModel::new(1.8, 0.05, 0.7, -0.2, 0.08, 0.02, 0.0)
                .expect("initial");

        let nm = NelderMeadConfig {
            max_iterations: 120,
            tolerance: 1.0e-6,
            ..NelderMeadConfig::default()
        };
        let config = CalibrationConfig {
            method: CalibrationMethod::NelderMead,
            objective: ObjectiveKind::LeastSquares,
            differential_evolution: DifferentialEvolutionConfig::default(),
            nelder_mead: nm,
        };

        let fft_config = HestonFftConfig {
            n: 1 << 9,
            b: 120.0,
            interp: InterpolationKind::Cubic,
        };

        let (calibrated, result) =
            calibrate_heston_model(&quotes, initial_model, Some(bounds), &config, fft_config)
                .expect("calibration");

        assert!(result.objective_value.is_finite(), "{}", result.message);
        assert!((calibrated.kappa - true_model.kappa).abs() < 1.0e-12);
        assert!((calibrated.theta - true_model.theta).abs() < 1.0e-12);
        assert!((calibrated.sigma - true_model.sigma).abs() < 1.0e-1);
        assert!((calibrated.rho - true_model.rho).abs() < 8.0e-2);
        assert!((calibrated.v0 - true_model.v0).abs() < 2.0e-2);
    }

    #[test]
    fn supports_robust_huber_objective() {
        let quotes = vec![
            VanillaOptionCalibrationQuote {
                underlying: 100.0,
                strike: 100.0,
                maturity: 1.0,
                option_type: OptionType::Call,
                market_value: 11.2,
                market_value_kind: VanillaMarketValueKind::Price,
                weight: 1.0,
            },
            VanillaOptionCalibrationQuote {
                underlying: 100.0,
                strike: 140.0,
                maturity: 1.0,
                option_type: OptionType::Call,
                market_value: 0.35,
                market_value_kind: VanillaMarketValueKind::Price,
                weight: 1.0,
            },
        ];

        let config = CalibrationConfig {
            method: CalibrationMethod::NelderMead,
            objective: ObjectiveKind::Huber { delta: 0.1 },
            ..CalibrationConfig::default()
        };

        let (_, result) = calibrate_black_scholes_model(
            &quotes,
            CalibrationTarget::Price,
            0.2,
            0.03,
            0.0,
            None,
            &config,
        )
        .expect("calibration");

        assert!(result.objective_value.is_finite());
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn rejects_quote_kind_target_mismatch() {
        let quotes = vec![VanillaOptionCalibrationQuote::implied_volatility(
            100.0,
            100.0,
            1.0,
            OptionType::Call,
            0.2,
            1.0,
        )];

        let error = calibrate_black_model(
            &quotes,
            CalibrationTarget::Price,
            0.2,
            0.03,
            None,
            &CalibrationConfig::default(),
        )
        .expect_err("target mismatch should fail");

        match error {
            QuantLibError::Validation(message) => {
                assert!(
                    message.contains("market_value_kind"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Calibrates with every parameter pinned to `model`, so the result's
    /// model values are `model` priced at the quotes.
    fn heston_model_values(
        quotes: &[VanillaOptionCalibrationQuote],
        model: quantlib_models::HestonModel,
        fft_config: HestonFftConfig,
    ) -> Vec<f64> {
        let params = [model.kappa, model.theta, model.sigma, model.rho, model.v0];
        let config = CalibrationConfig {
            method: CalibrationMethod::NelderMead,
            nelder_mead: NelderMeadConfig {
                max_iterations: 1,
                ..NelderMeadConfig::default()
            },
            ..CalibrationConfig::default()
        };
        let (_, result) = calibrate_heston_model(
            quotes,
            model,
            Some(params.map(|p| (p, p))),
            &config,
            fft_config,
        )
        .expect("calibration");
        result.model_values
    }

    #[test]
    fn heston_calibration_prices_quotes_at_their_exact_maturity() {
        let model = quantlib_models::HestonModel::new(1.8, 0.05, 0.45, -0.65, 0.04, 0.02, 0.0)
            .expect("heston");
        let fft_config = HestonFftConfig::default();
        let fft = quantlib_engines::FFTEngine::new(fft_config.n, fft_config.b, fft_config.interp)
            .expect("engine");

        // T = 1/12 used to become 30 days (T = 0.08219), and T = 0.001 was
        // clamped up to one day.
        let cases = [
            (1.0 / 12.0, 100.0, OptionType::Call),
            (1.0 / 12.0, 95.0, OptionType::Put),
            (0.001, 100.0, OptionType::Call),
            (0.75, 110.0, OptionType::Call),
        ];
        let quotes: Vec<_> = cases
            .iter()
            .map(|&(maturity, strike, option_type)| {
                VanillaOptionCalibrationQuote::price(100.0, strike, maturity, option_type, 1.0, 1.0)
            })
            .collect();

        let model_values = heston_model_values(&quotes, model, fft_config);
        for (&(maturity, strike, option_type), &value) in cases.iter().zip(&model_values) {
            let direct = fft
                .price_strip(&model, 100.0, maturity, &[strike], &[option_type])
                .expect("direct price")[0];
            let rel = (value - direct).abs() / direct.abs();
            assert!(
                rel < 1.0e-10,
                "T={maturity}, K={strike}: calibration={value}, direct={direct}, rel={rel:e}"
            );
        }
    }

    #[test]
    fn heston_calibration_keeps_quote_order_across_maturities_and_spots() {
        let model = quantlib_models::HestonModel::new(1.8, 0.05, 0.45, -0.65, 0.04, 0.02, 0.0)
            .expect("heston");
        let fft_config = HestonFftConfig::default();
        let fft = quantlib_engines::FFTEngine::new(fft_config.n, fft_config.b, fft_config.interp)
            .expect("engine");

        // Interleaved maturities and two underlyings at the same maturity.
        let cases = [
            (100.0, 1.0, 105.0, OptionType::Call),
            (100.0, 0.5, 95.0, OptionType::Put),
            (120.0, 1.0, 105.0, OptionType::Call),
            (100.0, 1.0, 90.0, OptionType::Put),
            (100.0, 0.5, 100.0, OptionType::Call),
        ];
        let quotes: Vec<_> = cases
            .iter()
            .map(|&(spot, maturity, strike, option_type)| {
                VanillaOptionCalibrationQuote::price(spot, strike, maturity, option_type, 1.0, 1.0)
            })
            .collect();

        let model_values = heston_model_values(&quotes, model, fft_config);
        for (&(spot, maturity, strike, option_type), &value) in cases.iter().zip(&model_values) {
            let direct = fft
                .price_strip(&model, spot, maturity, &[strike], &[option_type])
                .expect("direct price")[0];
            assert!(
                (value - direct).abs() <= 1.0e-12 * direct.abs(),
                "S={spot}, T={maturity}, K={strike}: calibration={value}, direct={direct}"
            );
        }
    }
}

#[cfg(test)]
mod de_base_index_tests {
    use std::sync::Mutex;

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::{DifferentialEvolutionConfig, differential_evolution};

    /// DE/rand/1 base index `a` must be uniform over the P-1 indices other than the target.
    /// Decodes (a, b, c) from the recorded first generation of a 1-D DE run:
    /// trial_i = x_a + F (x_b - x_c). The sorted-indices bug makes `a` the smallest of the three.
    #[test]
    fn de_base_index_is_uniform() {
        const P: usize = 10;
        const F: f64 = 0.5;
        const RUNS: u64 = 100_000;
        let config = |seed| DifferentialEvolutionConfig {
            max_generations: 1,
            population_size: P,
            differential_weight: F,
            crossover_probability: 1.0,
            seed,
            parallel_objective: false,
            ..DifferentialEvolutionConfig::default()
        };
        let mut counts = [[0u32; P]; P];
        let mut kept = [0u32; P];
        let mut guess_rng = ChaCha8Rng::seed_from_u64(1 << 40);
        for seed in 0..RUNS {
            let seen = Mutex::new(Vec::new());
            let objective = |x: &[f64]| {
                seen.lock().unwrap().push(x[0]);
                Ok(x[0])
            };
            // Random guess so population[0] is exchangeable with the other members.
            let guess = guess_rng.random_range(-1.0..=1.0);
            differential_evolution(&objective, &[guess], &[(-1.0, 1.0)], &config(seed))
                .expect("de");
            let seen = seen.into_inner().unwrap();
            let (pop, trials) = (&seen[..P], &seen[P..2 * P]);
            for (i, &trial) in trials.iter().enumerate() {
                if trial.abs() >= 1.0 {
                    continue; // clamped to a bound: not decodable
                }
                let mut hits = Vec::new();
                for a in (0..P).filter(|&a| a != i) {
                    for b in (0..P).filter(|&b| b != i && b != a) {
                        for c in (0..P).filter(|&c| c != i && c != a && c != b) {
                            if (pop[a] + F * (pop[b] - pop[c]) - trial).abs() < 1e-12 {
                                hits.push(a);
                            }
                        }
                    }
                }
                assert_eq!(hits.len(), 1, "ambiguous or undecodable trial");
                counts[i][hits[0]] += 1;
                kept[i] += 1;
            }
        }
        let p = 1.0 / (P as f64 - 1.0);
        for i in 0..P {
            let n = f64::from(kept[i]);
            let bound = 4.0 * (p * (1.0 - p) / n).sqrt();
            for a in (0..P).filter(|&a| a != i) {
                let freq = f64::from(counts[i][a]) / n;
                assert!(
                    (freq - p).abs() <= bound,
                    "target {i}: base index {a} frequency {freq:.4} vs {p:.4} +- {bound:.4}"
                );
            }
        }
    }
}
