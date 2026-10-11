//! Platform-independent request handling for the pricing playground.
//!
//! Everything here is plain Rust so it can be unit-tested natively; `bindings`
//! only converts between `JsValue` and these types.

use quantlib_core::{FinancialProduct, OptionType, PricingEngine, QuantLibError, SimpleDate};
use quantlib_engines::{AnalyticalEngine, BinomialEngine, FFTEngine, MonteCarloEngine};
use quantlib_models::{
    BatesModel, BlackModel, BlackScholesModel, HestonModel, KouModel, SabrModel,
};
use quantlib_products::{
    AmericanOption, AsianOption, AveragingType, DigitalOption, DigitalPayoffType, EuropeanOption,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request types (mirrored in web/src/types.ts)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProductKind {
    European,
    American,
    Digital,
    Asian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EngineKind {
    Analytical,
    Fft,
    Binomial,
    MonteCarlo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelKind {
    BlackScholes,
    Black76,
    Heston,
    Bates,
    Kou,
    Sabr,
}

impl ModelKind {
    /// Forward-quoted models take the underlying as a forward price.
    pub fn is_forward_quoted(self) -> bool {
        matches!(self, Self::Black76 | Self::Sabr)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum ModelSpec {
    BlackScholes {
        sigma: f64,
    },
    Black76 {
        sigma: f64,
    },
    Heston {
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
    },
    Bates {
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
        lambda: f64,
        mu_j: f64,
        sigma_j: f64,
    },
    Kou {
        sigma: f64,
        lambda: f64,
        p_up: f64,
        eta1: f64,
        eta2: f64,
    },
    Sabr {
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
    },
}

impl ModelSpec {
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::BlackScholes { .. } => ModelKind::BlackScholes,
            Self::Black76 { .. } => ModelKind::Black76,
            Self::Heston { .. } => ModelKind::Heston,
            Self::Bates { .. } => ModelKind::Bates,
            Self::Kou { .. } => ModelKind::Kou,
            Self::Sabr { .. } => ModelKind::Sabr,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CallPut {
    Call,
    Put,
}

impl From<CallPut> for OptionType {
    fn from(value: CallPut) -> Self {
        match value {
            CallPut::Call => OptionType::Call,
            CallPut::Put => OptionType::Put,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DigitalPayoff {
    Cash,
    Asset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Averaging {
    Arithmetic,
    Geometric,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProductSpec {
    pub kind: ProductKind,
    pub option_type: CallPut,
    pub strike: f64,
    /// ISO date, `YYYY-MM-DD`. Always explicit: `SystemTime` is unavailable
    /// on `wasm32-unknown-unknown`.
    pub valuation_date: String,
    pub expiry_date: String,
    #[serde(default = "default_digital_payoff")]
    pub digital_payoff: DigitalPayoff,
    #[serde(default = "default_payout")]
    pub payout: f64,
    #[serde(default = "default_averaging")]
    pub averaging: Averaging,
}

fn default_digital_payoff() -> DigitalPayoff {
    DigitalPayoff::Cash
}
fn default_payout() -> f64 {
    1.0
}
fn default_averaging() -> Averaging {
    Averaging::Arithmetic
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarketSpec {
    /// Spot for spot-quoted models, forward for Black-76 and SABR.
    pub underlying: f64,
    pub r: f64,
    #[serde(default)]
    pub q: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineSettings {
    pub binomial_steps: usize,
    pub mc_paths: usize,
    pub mc_steps: usize,
    pub mc_seed: u64,
    pub fft_n: usize,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            binomial_steps: 500,
            mc_paths: 50_000,
            mc_steps: 252,
            mc_seed: 42,
            fft_n: 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PricingRequest {
    pub product: ProductSpec,
    pub model: ModelSpec,
    pub market: MarketSpec,
    #[serde(default)]
    pub settings: EngineSettings,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Quote {
    pub price: f64,
    /// Standard error of `price`; only Monte Carlo has one.
    pub stderr: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Capability {
    pub product: ProductKind,
    pub model: ModelKind,
    pub engines: Vec<EngineKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Histogram {
    pub edges: Vec<f64>,
    /// Normalised to a density (integrates to 1 over the edges).
    pub density: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathSample {
    pub times: Vec<f64>,
    pub paths: Vec<Vec<f64>>,
    /// Pointwise quantiles over all simulated paths: 5%, 25%, 50%, 75%, 95%.
    pub quantiles: Vec<Vec<f64>>,
    pub terminal: Histogram,
    /// Exact terminal density at the histogram bin centres (Black–Scholes only).
    pub model_density: Option<Vec<f64>>,
}

pub type ApiResult<T> = Result<T, String>;

fn err(e: QuantLibError) -> String {
    e.to_string()
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

const ALL_PRODUCTS: [ProductKind; 4] = [
    ProductKind::European,
    ProductKind::American,
    ProductKind::Digital,
    ProductKind::Asian,
];
const ALL_MODELS: [ModelKind; 6] = [
    ModelKind::BlackScholes,
    ModelKind::Black76,
    ModelKind::Heston,
    ModelKind::Bates,
    ModelKind::Kou,
    ModelKind::Sabr,
];
/// Order also defines which engine serves as the reference in comparisons.
pub const ALL_ENGINES: [EngineKind; 4] = [
    EngineKind::Analytical,
    EngineKind::Fft,
    EngineKind::Binomial,
    EngineKind::MonteCarlo,
];

/// Whether the library implements `engine` for `(product, model)`.
pub fn supports(product: ProductKind, model: ModelKind, engine: EngineKind) -> bool {
    use EngineKind as E;
    use ModelKind as M;
    use ProductKind as P;
    match (product, model) {
        (P::European, M::BlackScholes) => true,
        (P::European, M::Black76) => matches!(engine, E::Analytical | E::Fft),
        (P::European, M::Heston) => matches!(engine, E::Fft | E::MonteCarlo),
        (P::European, M::Bates | M::Kou) => engine == E::Fft,
        (P::European, M::Sabr) => engine == E::Analytical,
        (P::American, M::BlackScholes) => engine == E::Binomial,
        (P::Digital, M::BlackScholes) => matches!(engine, E::Analytical | E::MonteCarlo),
        (P::Digital, M::Black76) => engine == E::Analytical,
        (P::Digital, M::Heston) => engine == E::MonteCarlo,
        (P::Asian, M::BlackScholes | M::Heston) => engine == E::MonteCarlo,
        _ => false,
    }
}

pub fn capabilities() -> Vec<Capability> {
    let mut out = Vec::new();
    for product in ALL_PRODUCTS {
        for model in ALL_MODELS {
            let engines: Vec<_> = ALL_ENGINES
                .into_iter()
                .filter(|e| supports(product, model, *e))
                .collect();
            if !engines.is_empty() {
                out.push(Capability {
                    product,
                    model,
                    engines,
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

enum Model {
    BlackScholes(BlackScholesModel),
    Black76(BlackModel),
    Heston(HestonModel),
    Bates(BatesModel),
    Kou(KouModel),
    Sabr(SabrModel),
}

fn build_model(spec: &ModelSpec, market: &MarketSpec) -> ApiResult<Model> {
    let (r, q) = (market.r, market.q);
    let model = match *spec {
        ModelSpec::BlackScholes { sigma } => {
            Model::BlackScholes(BlackScholesModel::new(sigma, r, q).map_err(err)?)
        }
        ModelSpec::Black76 { sigma } => Model::Black76(BlackModel::new(sigma, r).map_err(err)?),
        ModelSpec::Heston {
            kappa,
            theta,
            sigma,
            rho,
            v0,
        } => Model::Heston(HestonModel::new(kappa, theta, sigma, rho, v0, r, q).map_err(err)?),
        ModelSpec::Bates {
            kappa,
            theta,
            sigma,
            rho,
            v0,
            lambda,
            mu_j,
            sigma_j,
        } => Model::Bates(
            BatesModel::new(kappa, theta, sigma, rho, v0, r, q, lambda, mu_j, sigma_j)
                .map_err(err)?,
        ),
        ModelSpec::Kou {
            sigma,
            lambda,
            p_up,
            eta1,
            eta2,
        } => Model::Kou(KouModel::new(sigma, r, q, lambda, p_up, eta1, eta2).map_err(err)?),
        ModelSpec::Sabr {
            alpha,
            beta,
            rho,
            nu,
        } => {
            // The library's General SABR requires 0 < beta < 1; the endpoints
            // are separate lognormal (beta = 1) and normal (beta = 0) variants.
            let model = if beta == 1.0 {
                SabrModel::lognormal(alpha, rho, nu, r, None)
            } else if beta == 0.0 {
                SabrModel::normal(alpha, rho, nu, r, None)
            } else {
                SabrModel::general(alpha, beta, rho, nu, r, None)
            };
            Model::Sabr(model.map_err(err)?)
        }
    };
    Ok(model)
}

struct Dates {
    valuation: SimpleDate,
    expiry: SimpleDate,
}

fn parse_dates(product: &ProductSpec) -> ApiResult<Dates> {
    let valuation = SimpleDate::parse(&product.valuation_date).map_err(err)?;
    let expiry = SimpleDate::parse(&product.expiry_date).map_err(err)?;
    if expiry <= valuation {
        return Err("expiry date must be after the valuation date".to_owned());
    }
    Ok(Dates { valuation, expiry })
}

fn european(req: &PricingRequest, underlying: f64, strike: f64) -> ApiResult<EuropeanOption> {
    let p = &req.product;
    let d = parse_dates(p)?;
    let option = if req.model.kind().is_forward_quoted() {
        EuropeanOption::new_forward(
            underlying,
            strike,
            d.expiry,
            Some(d.valuation),
            p.option_type.into(),
            1,
            None,
        )
    } else {
        EuropeanOption::new_spot(
            underlying,
            strike,
            d.expiry,
            Some(d.valuation),
            p.option_type.into(),
            1,
            None,
        )
    };
    option.map_err(err)
}

/// Time to maturity in years, using the same convention as the products.
pub fn ttm(req: &PricingRequest) -> ApiResult<f64> {
    Ok(european(req, req.market.underlying, req.product.strike)?.ttm())
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

/// Number of independent batches used to estimate the MC standard error.
const MC_BATCHES: u64 = 8;

/// Prices the request with one engine.
pub fn price(req: &PricingRequest, engine: EngineKind) -> ApiResult<Quote> {
    price_at(req, engine, req.market.underlying)
}

/// Prices the request with the underlying replaced by `underlying`.
pub fn price_at(req: &PricingRequest, engine: EngineKind, underlying: f64) -> ApiResult<Quote> {
    let product = req.product.kind;
    let model_kind = req.model.kind();
    if !supports(product, model_kind, engine) {
        return Err(format!(
            "{engine:?} engine does not support {product:?} options under {model_kind:?}"
        ));
    }
    let model = build_model(&req.model, &req.market)?;
    let s = &req.settings;

    if engine == EngineKind::MonteCarlo {
        return monte_carlo(req, &model, underlying);
    }

    let exact = |price: Result<f64, QuantLibError>| -> ApiResult<Quote> {
        Ok(Quote {
            price: price.map_err(err)?,
            stderr: None,
        })
    };

    match (product, engine) {
        (ProductKind::European, _) => {
            let option = european(req, underlying, req.product.strike)?;
            match (engine, &model) {
                (EngineKind::Analytical, Model::BlackScholes(m)) => {
                    exact(AnalyticalEngine.calculate_price(&option, m))
                }
                (EngineKind::Analytical, Model::Black76(m)) => {
                    exact(AnalyticalEngine.calculate_price(&option, m))
                }
                (EngineKind::Analytical, Model::Sabr(m)) => {
                    exact(AnalyticalEngine.calculate_price(&option, m))
                }
                (EngineKind::Fft, _) => {
                    let fft = FFTEngine::new(s.fft_n, FFTEngine::DEFAULT_B, Default::default())
                        .map_err(err)?;
                    match &model {
                        Model::BlackScholes(m) => exact(fft.calculate_price(&option, m)),
                        Model::Black76(m) => exact(fft.calculate_price(&option, m)),
                        Model::Heston(m) => exact(fft.calculate_price(&option, m)),
                        Model::Bates(m) => exact(fft.calculate_price(&option, m)),
                        Model::Kou(m) => exact(fft.calculate_price(&option, m)),
                        Model::Sabr(_) => unreachable!("filtered by supports()"),
                    }
                }
                (EngineKind::Binomial, Model::BlackScholes(m)) => {
                    exact(BinomialEngine::new(s.binomial_steps).calculate_price(&option, m))
                }
                _ => unreachable!("filtered by supports()"),
            }
        }
        (ProductKind::American, EngineKind::Binomial) => {
            let option = AmericanOption {
                european: european(req, underlying, req.product.strike)?,
            };
            match &model {
                Model::BlackScholes(m) => {
                    exact(BinomialEngine::new(s.binomial_steps).calculate_price(&option, m))
                }
                _ => unreachable!("filtered by supports()"),
            }
        }
        (ProductKind::Digital, EngineKind::Analytical) => {
            let option = digital(req, underlying)?;
            match &model {
                Model::BlackScholes(m) => exact(AnalyticalEngine.calculate_price(&option, m)),
                Model::Black76(m) => exact(AnalyticalEngine.calculate_price(&option, m)),
                _ => unreachable!("filtered by supports()"),
            }
        }
        _ => unreachable!("filtered by supports()"),
    }
}

fn digital(req: &PricingRequest, underlying: f64) -> ApiResult<DigitalOption> {
    let european = european(req, underlying, req.product.strike)?;
    let payoff_type = match req.product.digital_payoff {
        DigitalPayoff::Cash => DigitalPayoffType::CashOrNothing,
        DigitalPayoff::Asset => DigitalPayoffType::AssetOrNothing,
    };
    DigitalOption::new(
        european.underlying,
        european.k,
        european.expiry_date,
        Some(european.valuation_date),
        european.option_type,
        1,
        req.product.payout,
        payoff_type,
        None,
    )
    .map_err(err)
}

fn asian(req: &PricingRequest, underlying: f64) -> ApiResult<AsianOption> {
    let d = parse_dates(&req.product)?;
    let averaging = match req.product.averaging {
        Averaging::Arithmetic => AveragingType::Arithmetic,
        Averaging::Geometric => AveragingType::Geometric,
    };
    AsianOption::new(
        underlying,
        req.product.strike,
        d.expiry,
        Some(d.valuation),
        req.product.option_type.into(),
        1,
        averaging,
        None,
    )
    .map_err(err)
}

/// Monte Carlo with the paths split into `MC_BATCHES` independently seeded
/// batches; the spread of the batch means gives the standard error.
fn monte_carlo(req: &PricingRequest, model: &Model, underlying: f64) -> ApiResult<Quote> {
    let s = &req.settings;
    let batch_paths = (s.mc_paths as u64 / MC_BATCHES).max(1) as usize;
    let product = req.product.kind;

    let mut estimates = Vec::with_capacity(MC_BATCHES as usize);
    for batch in 0..MC_BATCHES {
        // wasm32 has no threads: keep the engine serial everywhere so native
        // and WASM results are identical for the same seed.
        let engine =
            MonteCarloEngine::new(batch_paths, s.mc_steps, Some(s.mc_seed.wrapping_add(batch)))
                .map_err(err)?
                .with_parallel(false);

        let price = match (product, model) {
            (ProductKind::European, Model::BlackScholes(m)) => {
                engine.calculate_price(&european(req, underlying, req.product.strike)?, m)
            }
            (ProductKind::European, Model::Heston(m)) => {
                engine.calculate_price(&european(req, underlying, req.product.strike)?, m)
            }
            (ProductKind::Digital, Model::BlackScholes(m)) => {
                engine.calculate_price(&digital(req, underlying)?, m)
            }
            (ProductKind::Digital, Model::Heston(m)) => {
                engine.calculate_price(&digital(req, underlying)?, m)
            }
            (ProductKind::Asian, Model::BlackScholes(m)) => {
                engine.calculate_price(&asian(req, underlying)?, m)
            }
            (ProductKind::Asian, Model::Heston(m)) => {
                engine.calculate_price(&asian(req, underlying)?, m)
            }
            _ => unreachable!("filtered by supports()"),
        };
        estimates.push(price.map_err(err)?);
    }

    let n = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let var = estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Ok(Quote {
        price: mean,
        stderr: Some((var / n).sqrt()),
    })
}

/// Today's value across a grid of underlying levels (`None` where pricing fails).
pub fn price_curve(
    req: &PricingRequest,
    engine: EngineKind,
    underlyings: &[f64],
) -> Vec<Option<f64>> {
    underlyings
        .iter()
        .map(|&u| price_at(req, engine, u).ok().map(|q| q.price))
        .collect()
}

/// Payoff at expiry across a grid of underlying levels (for Asian options,
/// levels are read as the realised average).
pub fn payoff_curve(req: &PricingRequest, underlyings: &[f64]) -> ApiResult<Vec<f64>> {
    let u = req.market.underlying;
    Ok(match req.product.kind {
        ProductKind::European | ProductKind::American => {
            european(req, u, req.product.strike)?.payoff(underlyings)
        }
        ProductKind::Digital => digital(req, u)?.payoff(underlyings),
        ProductKind::Asian => asian(req, u)?.payoff(underlyings),
    })
}

// ---------------------------------------------------------------------------
// Implied volatility smile
// ---------------------------------------------------------------------------

/// The engine used for smiles: the most accurate one the model supports for
/// European options.
pub fn smile_engine(model: ModelKind) -> Option<EngineKind> {
    [EngineKind::Analytical, EngineKind::Fft]
        .into_iter()
        .find(|e| supports(ProductKind::European, model, *e))
}

/// Black implied volatility of European options at each strike, using
/// out-of-the-money options (puts below the forward, calls above).
pub fn smile(req: &PricingRequest, strikes: &[f64]) -> ApiResult<Vec<Option<f64>>> {
    let engine = smile_engine(req.model.kind())
        .ok_or_else(|| "no European engine for this model".to_owned())?;
    let t = ttm(req)?;
    let (r, q) = (req.market.r, req.market.q);
    let forward = if req.model.kind().is_forward_quoted() {
        req.market.underlying
    } else {
        req.market.underlying * ((r - q) * t).exp()
    };
    let discount = (-r * t).exp();

    let mut european_req = req.clone();
    european_req.product.kind = ProductKind::European;

    Ok(strikes
        .iter()
        .map(|&k| {
            let option_type = if k < forward {
                CallPut::Put
            } else {
                CallPut::Call
            };
            let mut r = european_req.clone();
            r.product.strike = k;
            r.product.option_type = option_type;
            let price = price(&r, engine).ok()?.price;
            black_implied_vol(price / discount, forward, k, t, option_type.into())
        })
        .collect())
}

/// Undiscounted Black price.
fn black(forward: f64, strike: f64, t: f64, sigma: f64, option_type: OptionType) -> f64 {
    let sd = sigma * t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
    let d2 = d1 - sd;
    match option_type {
        OptionType::Call => {
            forward * quantlib_core::norm_cdf(d1) - strike * quantlib_core::norm_cdf(d2)
        }
        OptionType::Put => {
            strike * quantlib_core::norm_cdf(-d2) - forward * quantlib_core::norm_cdf(-d1)
        }
    }
}

/// Black implied volatility by bisection (robust; #19 will add Jäckel's
/// "Let's Be Rational"). `None` when the price is outside the arbitrage bounds.
pub fn black_implied_vol(
    undiscounted_price: f64,
    forward: f64,
    strike: f64,
    t: f64,
    option_type: OptionType,
) -> Option<f64> {
    let intrinsic = match option_type {
        OptionType::Call => (forward - strike).max(0.0),
        OptionType::Put => (strike - forward).max(0.0),
    };
    let upper = match option_type {
        OptionType::Call => forward,
        OptionType::Put => strike,
    };
    if !undiscounted_price.is_finite()
        || undiscounted_price <= intrinsic
        || undiscounted_price >= upper
    {
        return None;
    }

    let (mut lo, mut hi) = (1e-8, 10.0);
    if black(forward, strike, t, hi, option_type) < undiscounted_price {
        return None;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if black(forward, strike, t, mid, option_type) < undiscounted_price {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    Some(0.5 * (lo + hi))
}

// ---------------------------------------------------------------------------
// Monte Carlo paths for plotting
// ---------------------------------------------------------------------------

/// Upper bound on paths simulated for the chart (memory: paths × steps).
const MAX_CHART_PATHS: usize = 10_000;
const MAX_CHART_STEPS: usize = 252;
const HISTOGRAM_BINS: usize = 60;

/// Simulates paths for the chart. Returns `n_display` sample paths plus
/// quantile bands and the terminal distribution over all simulated paths.
pub fn mc_paths(req: &PricingRequest, n_display: usize) -> ApiResult<PathSample> {
    let s = &req.settings;
    let n_paths = s.mc_paths.clamp(n_display.max(2), MAX_CHART_PATHS);
    let n_steps = s.mc_steps.clamp(1, MAX_CHART_STEPS);
    let engine = MonteCarloEngine::new(n_paths, n_steps, Some(s.mc_seed))
        .map_err(err)?
        .with_parallel(false);
    let t = ttm(req)?;
    let spot = req.market.underlying;

    let model = build_model(&req.model, &req.market)?;
    let (paths, density_model) = match &model {
        Model::BlackScholes(m) => (
            engine.sample_paths_black_scholes(spot, m, t).map_err(err)?,
            Some(*m),
        ),
        Model::Heston(m) => (engine.sample_paths_heston(spot, m, t).map_err(err)?, None),
        _ => return Err("path simulation is available for Black–Scholes and Heston".to_owned()),
    };

    let times: Vec<f64> = (0..=n_steps)
        .map(|i| t * i as f64 / n_steps as f64)
        .collect();

    let mut quantiles: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(n_steps + 1)).collect();
    let mut column = vec![0.0; paths.len()];
    for step in 0..=n_steps {
        for (c, p) in column.iter_mut().zip(&paths) {
            *c = p[step];
        }
        column.sort_by(f64::total_cmp);
        for (band, level) in quantiles.iter_mut().zip([0.05, 0.25, 0.5, 0.75, 0.95]) {
            band.push(quantile_sorted(&column, level));
        }
    }

    let mut terminal: Vec<f64> = paths.iter().map(|p| p[n_steps]).collect();
    terminal.sort_by(f64::total_cmp);
    // Clip the histogram range to the 0.1%–99.9% quantiles so a few extreme
    // paths do not flatten the plot.
    let (lo, hi) = (
        quantile_sorted(&terminal, 0.001),
        quantile_sorted(&terminal, 0.999),
    );
    let histogram = histogram(&terminal, lo, hi, HISTOGRAM_BINS);

    let model_density = density_model.map(|m| {
        let mu = spot.ln() + (m.r - m.q - 0.5 * m.sigma * m.sigma) * t;
        let sd = m.sigma * t.sqrt();
        histogram
            .edges
            .windows(2)
            .map(|w| {
                let x = 0.5 * (w[0] + w[1]);
                let z = (x.ln() - mu) / sd;
                quantlib_core::norm_pdf(z) / (x * sd)
            })
            .collect()
    });

    let stride = (paths.len() / n_display.max(1)).max(1);
    let display = paths.into_iter().step_by(stride).take(n_display).collect();

    Ok(PathSample {
        times,
        paths: display,
        quantiles,
        terminal: histogram,
        model_density,
    })
}

fn quantile_sorted(sorted: &[f64], level: f64) -> f64 {
    let pos = level * (sorted.len() - 1) as f64;
    let (i, frac) = (pos.floor() as usize, pos.fract());
    let next = sorted[(i + 1).min(sorted.len() - 1)];
    sorted[i] + frac * (next - sorted[i])
}

fn histogram(values: &[f64], lo: f64, hi: f64, bins: usize) -> Histogram {
    let width = (hi - lo).max(f64::EPSILON) / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| lo + width * i as f64).collect();
    let mut counts = vec![0usize; bins];
    for &v in values {
        if v >= lo && v <= hi {
            let i = (((v - lo) / width) as usize).min(bins - 1);
            counts[i] += 1;
        }
    }
    let norm = values.len() as f64 * width;
    Histogram {
        edges,
        density: counts.into_iter().map(|c| c as f64 / norm).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn bs_request() -> PricingRequest {
        serde_json::from_str(
            r#"{
                "product": {
                    "kind": "european", "optionType": "call", "strike": 100,
                    "valuationDate": "2026-01-01", "expiryDate": "2027-01-01"
                },
                "model": { "kind": "blackScholes", "sigma": 0.2 },
                "market": { "underlying": 100, "r": 0.05, "q": 0 },
                "settings": {
                    "binomialSteps": 2000, "mcPaths": 80000, "mcSteps": 1,
                    "mcSeed": 42, "fftN": 4096
                }
            }"#,
        )
        .expect("request JSON")
    }

    #[test]
    fn every_engine_agrees_on_black_scholes_reference() {
        // S=K=100, T=1, r=5%, sigma=20%: 10.450583572185565.
        let req = bs_request();
        let reference = 10.450_583_572_185_565;
        let analytical = price(&req, EngineKind::Analytical).unwrap();
        assert!((analytical.price - reference).abs() < 1e-9);
        assert!((price(&req, EngineKind::Fft).unwrap().price - reference).abs() < 5e-3);
        assert!((price(&req, EngineKind::Binomial).unwrap().price - reference).abs() < 5e-3);
        let mc = price(&req, EngineKind::MonteCarlo).unwrap();
        let stderr = mc.stderr.expect("MC reports a standard error");
        assert!((mc.price - reference).abs() < 3.0 * stderr, "{mc:?}");
    }

    #[test]
    fn unsupported_combination_is_an_error() {
        let mut req = bs_request();
        req.model = ModelSpec::Kou {
            sigma: 0.2,
            lambda: 1.0,
            p_up: 0.4,
            eta1: 10.0,
            eta2: 5.0,
        };
        assert!(price(&req, EngineKind::Binomial).is_err());
        assert!(price(&req, EngineKind::Fft).is_ok());
    }

    #[test]
    fn capabilities_cover_every_supported_pair() {
        let caps = capabilities();
        assert!(caps.iter().any(|c| c.product == ProductKind::American
            && c.model == ModelKind::BlackScholes
            && c.engines == vec![EngineKind::Binomial]));
        assert!(caps.iter().all(|c| !c.engines.is_empty()));
    }

    #[test]
    fn flat_smile_for_black_scholes() {
        let req = bs_request();
        let vols = smile(&req, &[70.0, 90.0, 100.0, 110.0, 140.0]).unwrap();
        for v in vols {
            assert!((v.unwrap() - 0.2).abs() < 1e-8, "{v:?}");
        }
    }

    #[test]
    fn paths_and_histogram_are_consistent() {
        let mut req = bs_request();
        req.settings.mc_paths = 5_000;
        req.settings.mc_steps = 50;
        let sample = mc_paths(&req, 100).unwrap();
        assert_eq!(sample.paths.len(), 100);
        assert_eq!(sample.times.len(), 51);
        assert_eq!(sample.quantiles.len(), 5);
        let mass: f64 = sample
            .terminal
            .density
            .iter()
            .zip(sample.terminal.edges.windows(2))
            .map(|(d, w)| d * (w[1] - w[0]))
            .sum();
        assert!(mass > 0.99 && mass <= 1.0 + 1e-12, "mass={mass}");
        assert!(sample.model_density.is_some());
    }

    #[test]
    fn sabr_accepts_beta_endpoints() {
        let mut req = bs_request();
        for beta in [0.0, 0.5, 1.0] {
            req.model = ModelSpec::Sabr {
                alpha: if beta == 0.0 { 20.0 } else { 0.2 },
                beta,
                rho: -0.3,
                nu: 0.4,
            };
            let quote = price(&req, EngineKind::Analytical);
            assert!(quote.is_ok_and(|q| q.price > 0.0), "beta={beta}");
        }
    }

    #[test]
    fn digital_american_and_asian_requests_price() {
        let mut req = bs_request();
        req.product.kind = ProductKind::Digital;
        assert!(price(&req, EngineKind::Analytical).unwrap().price > 0.0);
        req.product.kind = ProductKind::American;
        req.product.option_type = CallPut::Put;
        assert!(price(&req, EngineKind::Binomial).unwrap().price > 0.0);
        req.product.kind = ProductKind::Asian;
        req.settings.mc_paths = 4_000;
        req.settings.mc_steps = 12;
        assert!(price(&req, EngineKind::MonteCarlo).unwrap().price > 0.0);
    }
}
