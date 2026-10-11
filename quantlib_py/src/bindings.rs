use std::str::FromStr;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use quantlib_core::{OptionType, PricingEngine, QuantLibError, SimpleDate};
use quantlib_engines::{
    AnalyticalEngine, CosEngine as RustCosEngine, FFTEngine as RustFFTEngine, HestonScheme,
    InterpolationKind, MonteCarloEngine, calculate_prices_parallel,
    calculate_prices_parallel_per_model,
};
use quantlib_models::{BatesModel, BlackModel, BlackScholesModel, HestonModel, KouModel};
use quantlib_products::{
    AsianOption, AveragingType, DigitalOption, DigitalPayoffType, EuropeanOption,
};
use quantlib_utils::{CalibrationMethod, CalibrationTarget, ObjectiveKind};

pub(crate) fn to_py_err(error: QuantLibError) -> PyErr {
    match error {
        QuantLibError::Validation(message) | QuantLibError::DateParse(message) => {
            PyValueError::new_err(message)
        }
        QuantLibError::Serialization(message) | QuantLibError::Engine(message) => {
            PyRuntimeError::new_err(message)
        }
    }
}

pub(crate) fn parse_option_type(option_type: &str) -> PyResult<OptionType> {
    OptionType::from_str(option_type).map_err(to_py_err)
}

pub(crate) fn parse_averaging_type(averaging_type: &str) -> PyResult<AveragingType> {
    AveragingType::from_str(averaging_type).map_err(to_py_err)
}

pub(crate) fn parse_digital_payoff_type(payoff_type: &str) -> PyResult<DigitalPayoffType> {
    DigitalPayoffType::from_str(payoff_type).map_err(to_py_err)
}

pub(crate) fn parse_date(value: &str) -> PyResult<SimpleDate> {
    SimpleDate::parse(value).map_err(to_py_err)
}

pub(crate) fn parse_fft_interpolation(value: &str) -> PyResult<InterpolationKind> {
    value.parse::<InterpolationKind>().map_err(to_py_err)
}

/// Default Heston Monte Carlo scheme for the Python API.
pub(crate) const DEFAULT_HESTON_SCHEME: &str = "qe";

fn heston_scheme_name(scheme: HestonScheme) -> &'static str {
    match scheme {
        HestonScheme::QuadraticExponential => "qe",
        HestonScheme::FullTruncation => "full_truncation",
    }
}

pub(crate) fn parse_calibration_method(value: &str) -> PyResult<CalibrationMethod> {
    match value.to_ascii_lowercase().as_str() {
        "hybrid" | "hybrid_global_local" => Ok(CalibrationMethod::HybridGlobalLocal),
        "differential_evolution" | "de" => Ok(CalibrationMethod::DifferentialEvolution),
        "nelder_mead" | "nm" => Ok(CalibrationMethod::NelderMead),
        _ => Err(PyValueError::new_err(
            "method must be one of: 'hybrid', 'differential_evolution', 'nelder_mead'",
        )),
    }
}

pub(crate) fn parse_calibration_target(value: &str) -> PyResult<CalibrationTarget> {
    match value.to_ascii_lowercase().as_str() {
        "price" => Ok(CalibrationTarget::Price),
        "volatility" | "vol" => Ok(CalibrationTarget::Volatility),
        _ => Err(PyValueError::new_err(
            "target must be one of: 'price', 'volatility'",
        )),
    }
}

pub(crate) fn parse_calibration_objective(
    value: &str,
    huber_delta: Option<f64>,
) -> PyResult<ObjectiveKind> {
    match value.to_ascii_lowercase().as_str() {
        "least_squares" | "lsq" => Ok(ObjectiveKind::LeastSquares),
        "relative_least_squares" | "relative" => Ok(ObjectiveKind::RelativeLeastSquares),
        "huber" => {
            let delta = huber_delta.unwrap_or(1.0e-3);
            if !delta.is_finite() || delta <= 0.0 {
                return Err(PyValueError::new_err(
                    "huber_delta must be finite and positive",
                ));
            }
            Ok(ObjectiveKind::Huber { delta })
        }
        _ => Err(PyValueError::new_err(
            "objective must be one of: 'least_squares', 'relative_least_squares', 'huber'",
        )),
    }
}

pub(crate) fn require_contiguous<'py, T: numpy::Element>(
    array: &'py PyReadonlyArray1<'py, T>,
    name: &str,
) -> PyResult<&'py [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a contiguous 1D numpy array with matching dtype"
        ))
    })
}

pub(crate) fn price_products_single_model<E, P, M>(
    engine: E,
    products: Vec<P>,
    model: M,
    parallel: bool,
) -> Result<Vec<f64>, QuantLibError>
where
    E: PricingEngine<P, M> + Sync,
    P: Sync,
    M: Sync,
{
    if parallel {
        calculate_prices_parallel(&engine, &products, &model)
    } else {
        products
            .iter()
            .map(|product| engine.calculate_price(product, &model))
            .collect()
    }
}

pub(crate) fn price_products_per_model<E, P, M>(
    engine: E,
    products: Vec<P>,
    models: Vec<M>,
    parallel: bool,
) -> Result<Vec<f64>, QuantLibError>
where
    E: PricingEngine<P, M> + Sync,
    P: Sync,
    M: Sync,
{
    if parallel {
        calculate_prices_parallel_per_model(&engine, &products, &models)
    } else {
        products
            .iter()
            .zip(models.iter())
            .map(|(product, model)| engine.calculate_price(product, model))
            .collect()
    }
}

#[pyclass(module = "quantlib_py", name = "BlackScholesModel", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyBlackScholesModel {
    pub(crate) inner: BlackScholesModel,
}

#[pymethods]
impl PyBlackScholesModel {
    #[new]
    pub(crate) fn new(sigma: f64, r: f64, q: f64) -> PyResult<Self> {
        let inner = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    #[getter]
    fn q(&self) -> f64 {
        self.inner.q
    }

    fn __repr__(&self) -> String {
        format!(
            "BlackScholesModel(sigma={}, r={}, q={})",
            self.inner.sigma, self.inner.r, self.inner.q
        )
    }
}

#[pyclass(module = "quantlib_py", name = "BlackModel", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyBlackModel {
    pub(crate) inner: BlackModel,
}

#[pymethods]
impl PyBlackModel {
    #[new]
    pub(crate) fn new(sigma: f64, r: f64) -> PyResult<Self> {
        let inner = BlackModel::new(sigma, r).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    fn __repr__(&self) -> String {
        format!("BlackModel(sigma={}, r={})", self.inner.sigma, self.inner.r)
    }
}

#[pyclass(module = "quantlib_py", name = "HestonModel", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyHestonModel {
    pub(crate) inner: HestonModel,
}

#[pymethods]
impl PyHestonModel {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
        r: f64,
        q: f64,
    ) -> PyResult<Self> {
        let inner = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    #[getter]
    fn v0(&self) -> f64 {
        self.inner.v0
    }

    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    #[getter]
    fn q(&self) -> f64 {
        self.inner.q
    }

    fn __repr__(&self) -> String {
        format!(
            "HestonModel(kappa={}, theta={}, sigma={}, rho={}, v0={}, r={}, q={})",
            self.inner.kappa,
            self.inner.theta,
            self.inner.sigma,
            self.inner.rho,
            self.inner.v0,
            self.inner.r,
            self.inner.q
        )
    }
}

#[pyclass(module = "quantlib_py", name = "BatesModel", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyBatesModel {
    pub(crate) inner: BatesModel,
}

#[pymethods]
impl PyBatesModel {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
        r: f64,
        q: f64,
        lambda_jump: f64,
        mu_j: f64,
        sigma_j: f64,
    ) -> PyResult<Self> {
        let inner = BatesModel::new(
            kappa,
            theta,
            sigma,
            rho,
            v0,
            r,
            q,
            lambda_jump,
            mu_j,
            sigma_j,
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    #[getter]
    fn v0(&self) -> f64 {
        self.inner.v0
    }

    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    #[getter]
    fn q(&self) -> f64 {
        self.inner.q
    }

    #[getter]
    fn lambda_jump(&self) -> f64 {
        self.inner.lambda_jump
    }

    #[getter]
    fn mu_j(&self) -> f64 {
        self.inner.mu_j
    }

    #[getter]
    fn sigma_j(&self) -> f64 {
        self.inner.sigma_j
    }

    fn __repr__(&self) -> String {
        format!(
            "BatesModel(kappa={}, theta={}, sigma={}, rho={}, v0={}, r={}, q={}, lambda_jump={}, mu_j={}, sigma_j={})",
            self.inner.kappa,
            self.inner.theta,
            self.inner.sigma,
            self.inner.rho,
            self.inner.v0,
            self.inner.r,
            self.inner.q,
            self.inner.lambda_jump,
            self.inner.mu_j,
            self.inner.sigma_j
        )
    }
}

#[pyclass(module = "quantlib_py", name = "KouModel", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyKouModel {
    pub(crate) inner: KouModel,
}

#[pymethods]
impl PyKouModel {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sigma: f64,
        r: f64,
        q: f64,
        lambda_jump: f64,
        p_up: f64,
        eta1: f64,
        eta2: f64,
    ) -> PyResult<Self> {
        let inner = KouModel::new(sigma, r, q, lambda_jump, p_up, eta1, eta2).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[getter]
    fn r(&self) -> f64 {
        self.inner.r
    }

    #[getter]
    fn q(&self) -> f64 {
        self.inner.q
    }

    #[getter]
    fn lambda_jump(&self) -> f64 {
        self.inner.lambda_jump
    }

    #[getter]
    fn p_up(&self) -> f64 {
        self.inner.p_up
    }

    #[getter]
    fn eta1(&self) -> f64 {
        self.inner.eta1
    }

    #[getter]
    fn eta2(&self) -> f64 {
        self.inner.eta2
    }

    fn __repr__(&self) -> String {
        format!(
            "KouModel(sigma={}, r={}, q={}, lambda_jump={}, p_up={}, eta1={}, eta2={})",
            self.inner.sigma,
            self.inner.r,
            self.inner.q,
            self.inner.lambda_jump,
            self.inner.p_up,
            self.inner.eta1,
            self.inner.eta2
        )
    }
}

#[pyclass(module = "quantlib_py", name = "EuropeanOptionSpot", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyEuropeanOptionSpot {
    pub(crate) inner: EuropeanOption,
}

#[pymethods]
impl PyEuropeanOptionSpot {
    #[new]
    #[pyo3(signature = (spot, strike, expiry_date, valuation_date = None, option_type = "call", qty = 1))]
    pub(crate) fn new(
        spot: f64,
        strike: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
    ) -> PyResult<Self> {
        let expiry = parse_date(expiry_date)?;
        let valuation = valuation_date.map(parse_date).transpose()?;
        let option_type = parse_option_type(option_type)?;

        let inner =
            EuropeanOption::new_spot(spot, strike, expiry, valuation, option_type, qty, None)
                .map_err(to_py_err)?;

        Ok(Self { inner })
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot().unwrap_or_default()
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.k
    }

    #[getter]
    fn option_type(&self) -> String {
        self.inner.option_type.to_string()
    }

    #[getter]
    fn qty(&self) -> i32 {
        self.inner.qty
    }

    #[getter]
    fn expiry_date(&self) -> String {
        self.inner.expiry_date.format_ymd()
    }

    #[getter]
    fn valuation_date(&self) -> String {
        self.inner.valuation_date.format_ymd()
    }

    fn days_to_maturity(&self) -> i64 {
        self.inner.days_to_maturity()
    }

    fn time_to_maturity(&self) -> f64 {
        self.inner.ttm()
    }

    fn __repr__(&self) -> String {
        format!(
            "EuropeanOptionSpot(spot={}, strike={}, expiry_date='{}', valuation_date='{}', option_type='{}', qty={})",
            self.spot(),
            self.strike(),
            self.expiry_date(),
            self.valuation_date(),
            self.option_type(),
            self.qty()
        )
    }
}

#[pyclass(module = "quantlib_py", name = "EuropeanOptionForward", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyEuropeanOptionForward {
    pub(crate) inner: EuropeanOption,
}

#[pymethods]
impl PyEuropeanOptionForward {
    #[new]
    #[pyo3(signature = (forward, strike, expiry_date, valuation_date = None, option_type = "call", qty = 1))]
    pub(crate) fn new(
        forward: f64,
        strike: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
    ) -> PyResult<Self> {
        let expiry = parse_date(expiry_date)?;
        let valuation = valuation_date.map(parse_date).transpose()?;
        let option_type = parse_option_type(option_type)?;

        let inner =
            EuropeanOption::new_forward(forward, strike, expiry, valuation, option_type, qty, None)
                .map_err(to_py_err)?;

        Ok(Self { inner })
    }

    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward().unwrap_or_default()
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.k
    }

    #[getter]
    fn option_type(&self) -> String {
        self.inner.option_type.to_string()
    }

    #[getter]
    fn qty(&self) -> i32 {
        self.inner.qty
    }

    #[getter]
    fn expiry_date(&self) -> String {
        self.inner.expiry_date.format_ymd()
    }

    #[getter]
    fn valuation_date(&self) -> String {
        self.inner.valuation_date.format_ymd()
    }

    fn days_to_maturity(&self) -> i64 {
        self.inner.days_to_maturity()
    }

    fn time_to_maturity(&self) -> f64 {
        self.inner.ttm()
    }

    fn __repr__(&self) -> String {
        format!(
            "EuropeanOptionForward(forward={}, strike={}, expiry_date='{}', valuation_date='{}', option_type='{}', qty={})",
            self.forward(),
            self.strike(),
            self.expiry_date(),
            self.valuation_date(),
            self.option_type(),
            self.qty()
        )
    }
}

#[pyclass(module = "quantlib_py", name = "DigitalOptionSpot", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDigitalOptionSpot {
    pub(crate) inner: DigitalOption,
}

#[pymethods]
impl PyDigitalOptionSpot {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (spot, strike, expiry_date, valuation_date = None, option_type = "call", qty = 1, payout = 1.0, payoff_type = "cash"))]
    pub(crate) fn new(
        spot: f64,
        strike: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        payout: f64,
        payoff_type: &str,
    ) -> PyResult<Self> {
        let expiry = parse_date(expiry_date)?;
        let valuation = valuation_date.map(parse_date).transpose()?;
        let option_type = parse_option_type(option_type)?;
        let payoff_type = parse_digital_payoff_type(payoff_type)?;

        let inner = DigitalOption::new_spot(
            spot,
            strike,
            expiry,
            valuation,
            option_type,
            qty,
            payout,
            payoff_type,
            None,
        )
        .map_err(to_py_err)?;

        Ok(Self { inner })
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.spot().unwrap_or_default()
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike()
    }

    #[getter]
    fn option_type(&self) -> String {
        self.inner.option_type().to_string()
    }

    #[getter]
    fn qty(&self) -> i32 {
        self.inner.european.qty
    }

    #[getter]
    fn payout(&self) -> f64 {
        self.inner.payout
    }

    #[getter]
    fn payoff_type(&self) -> String {
        self.inner.payoff_type.to_string()
    }

    #[getter]
    fn expiry_date(&self) -> String {
        self.inner.european.expiry_date.format_ymd()
    }

    #[getter]
    fn valuation_date(&self) -> String {
        self.inner.european.valuation_date.format_ymd()
    }

    fn days_to_maturity(&self) -> i64 {
        self.inner.days_to_maturity()
    }

    fn time_to_maturity(&self) -> f64 {
        self.inner.ttm()
    }

    fn __repr__(&self) -> String {
        format!(
            "DigitalOptionSpot(spot={}, strike={}, expiry_date='{}', valuation_date='{}', option_type='{}', qty={}, payout={}, payoff_type='{}')",
            self.spot(),
            self.strike(),
            self.expiry_date(),
            self.valuation_date(),
            self.option_type(),
            self.qty(),
            self.payout(),
            self.payoff_type()
        )
    }
}

#[pyclass(module = "quantlib_py", name = "DigitalOptionForward", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDigitalOptionForward {
    pub(crate) inner: DigitalOption,
}

#[pymethods]
impl PyDigitalOptionForward {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (forward, strike, expiry_date, valuation_date = None, option_type = "call", qty = 1, payout = 1.0, payoff_type = "cash"))]
    pub(crate) fn new(
        forward: f64,
        strike: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        payout: f64,
        payoff_type: &str,
    ) -> PyResult<Self> {
        let expiry = parse_date(expiry_date)?;
        let valuation = valuation_date.map(parse_date).transpose()?;
        let option_type = parse_option_type(option_type)?;
        let payoff_type = parse_digital_payoff_type(payoff_type)?;

        let inner = DigitalOption::new_forward(
            forward,
            strike,
            expiry,
            valuation,
            option_type,
            qty,
            payout,
            payoff_type,
            None,
        )
        .map_err(to_py_err)?;

        Ok(Self { inner })
    }

    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward().unwrap_or_default()
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike()
    }

    #[getter]
    fn option_type(&self) -> String {
        self.inner.option_type().to_string()
    }

    #[getter]
    fn qty(&self) -> i32 {
        self.inner.european.qty
    }

    #[getter]
    fn payout(&self) -> f64 {
        self.inner.payout
    }

    #[getter]
    fn payoff_type(&self) -> String {
        self.inner.payoff_type.to_string()
    }

    #[getter]
    fn expiry_date(&self) -> String {
        self.inner.european.expiry_date.format_ymd()
    }

    #[getter]
    fn valuation_date(&self) -> String {
        self.inner.european.valuation_date.format_ymd()
    }

    fn days_to_maturity(&self) -> i64 {
        self.inner.days_to_maturity()
    }

    fn time_to_maturity(&self) -> f64 {
        self.inner.ttm()
    }

    fn __repr__(&self) -> String {
        format!(
            "DigitalOptionForward(forward={}, strike={}, expiry_date='{}', valuation_date='{}', option_type='{}', qty={}, payout={}, payoff_type='{}')",
            self.forward(),
            self.strike(),
            self.expiry_date(),
            self.valuation_date(),
            self.option_type(),
            self.qty(),
            self.payout(),
            self.payoff_type()
        )
    }
}

#[pyclass(module = "quantlib_py", name = "AsianOptionSpot", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyAsianOptionSpot {
    pub(crate) inner: AsianOption,
}

#[pymethods]
impl PyAsianOptionSpot {
    #[new]
    #[pyo3(signature = (spot, strike, expiry_date, valuation_date = None, option_type = "call", qty = 1, averaging_type = "arithmetic"))]
    pub(crate) fn new(
        spot: f64,
        strike: f64,
        expiry_date: &str,
        valuation_date: Option<&str>,
        option_type: &str,
        qty: i32,
        averaging_type: &str,
    ) -> PyResult<Self> {
        let expiry = parse_date(expiry_date)?;
        let valuation = valuation_date.map(parse_date).transpose()?;
        let option_type = parse_option_type(option_type)?;
        let averaging_type = parse_averaging_type(averaging_type)?;

        let inner = AsianOption::new(
            spot,
            strike,
            expiry,
            valuation,
            option_type,
            qty,
            averaging_type,
            None,
        )
        .map_err(to_py_err)?;

        Ok(Self { inner })
    }

    #[getter]
    fn spot(&self) -> f64 {
        self.inner.s
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.k
    }

    #[getter]
    fn option_type(&self) -> String {
        self.inner.option_type.to_string()
    }

    #[getter]
    fn qty(&self) -> i32 {
        self.inner.qty
    }

    #[getter]
    fn averaging_type(&self) -> String {
        self.inner.averaging_type.to_string()
    }

    #[getter]
    fn expiry_date(&self) -> String {
        self.inner.expiry_date.format_ymd()
    }

    #[getter]
    fn valuation_date(&self) -> String {
        self.inner.valuation_date.format_ymd()
    }

    fn days_to_maturity(&self) -> i64 {
        self.inner.days_to_maturity()
    }

    fn time_to_maturity(&self) -> f64 {
        self.inner.ttm()
    }

    fn __repr__(&self) -> String {
        format!(
            "AsianOptionSpot(spot={}, strike={}, expiry_date='{}', valuation_date='{}', option_type='{}', qty={}, averaging_type='{}')",
            self.spot(),
            self.strike(),
            self.expiry_date(),
            self.valuation_date(),
            self.option_type(),
            self.qty(),
            self.averaging_type()
        )
    }
}

#[pyclass(module = "quantlib_py", name = "AnalyticalEngine", from_py_object)]
#[derive(Default, Clone, Copy)]
pub(crate) struct PyAnalyticalEngine {
    pub(crate) inner: AnalyticalEngine,
}

#[pymethods]
impl PyAnalyticalEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: AnalyticalEngine,
        }
    }

    fn price_european_bs(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        self.price_european_bs(py, option, model)
    }

    #[pyo3(signature = (options, model, parallel = true))]
    fn price_european_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel = true))]
    fn price_european_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        self.price_european_bs_batch(py, options, model, parallel)
    }

    #[pyo3(signature = (options, models, parallel = true))]
    fn price_european_bs_batch_per_model(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        models: Vec<Py<PyBlackScholesModel>>,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        if options.len() != models.len() {
            return Err(PyValueError::new_err(format!(
                "options length ({}) must match models length ({})",
                options.len(),
                models.len()
            )));
        }

        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let local_models = models
            .into_iter()
            .map(|model| model.borrow(py).inner)
            .collect::<Vec<_>>();

        let engine = self.inner;

        py.detach(move || price_products_per_model(engine, products, local_models, parallel))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, models, parallel = true))]
    fn price_european_batch_per_model(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        models: Vec<Py<PyBlackScholesModel>>,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        self.price_european_bs_batch_per_model(py, options, models, parallel)
    }

    fn price_european_black(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionForward,
        model: &PyBlackModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel = true))]
    fn price_european_black_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionForward>>,
        model: &PyBlackModel,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, models, parallel = true))]
    fn price_european_black_batch_per_model(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionForward>>,
        models: Vec<Py<PyBlackModel>>,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        if options.len() != models.len() {
            return Err(PyValueError::new_err(format!(
                "options length ({}) must match models length ({})",
                options.len(),
                models.len()
            )));
        }

        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let local_models = models
            .into_iter()
            .map(|model| model.borrow(py).inner)
            .collect::<Vec<_>>();

        let engine = self.inner;

        py.detach(move || price_products_per_model(engine, products, local_models, parallel))
            .map_err(to_py_err)
    }

    fn price_digital_bs(
        &self,
        py: Python<'_>,
        option: &PyDigitalOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel = true))]
    fn price_digital_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyDigitalOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel))
            .map_err(to_py_err)
    }

    fn price_digital_black(
        &self,
        py: Python<'_>,
        option: &PyDigitalOptionForward,
        model: &PyBlackModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel = true))]
    fn price_digital_black_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyDigitalOptionForward>>,
        model: &PyBlackModel,
        parallel: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel))
            .map_err(to_py_err)
    }
}

#[pyclass(module = "quantlib_py", name = "FFTEngine", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyFFTEngine {
    pub(crate) inner: RustFFTEngine,
}

#[pymethods]
impl PyFFTEngine {
    #[new]
    #[pyo3(signature = (n = 4_096, b = 200.0, interp = "cubic"))]
    fn new(n: usize, b: f64, interp: &str) -> PyResult<Self> {
        let interp = parse_fft_interpolation(interp)?;
        let inner = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn b(&self) -> f64 {
        self.inner.b()
    }

    #[getter]
    fn interp(&self) -> &'static str {
        match self.inner.interp() {
            InterpolationKind::Linear => "linear",
            InterpolationKind::Cubic => "cubic",
        }
    }

    fn price_european_heston(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyHestonModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european_bates(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBatesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european_kou(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyKouModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european_bs(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_european_black(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionForward,
        model: &PyBlackModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_black_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionForward>>,
        model: &PyBlackModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_heston_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyHestonModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_bates_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBatesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_kou_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyKouModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner.clone();

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "FFTEngine(n={}, b={}, interp='{}')",
            self.inner.n(),
            self.inner.b(),
            self.interp()
        )
    }
}

/// Fang–Oosterlee COS engine: much faster than FFT for single strikes or
/// short strike lists (Greeks by bumping, calibration with scattered quotes).
#[pyclass(module = "quantlib_py", name = "CosEngine", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyCosEngine {
    pub(crate) inner: RustCosEngine,
}

#[pymethods]
impl PyCosEngine {
    #[new]
    #[pyo3(signature = (n = 256, l = 12.0))]
    fn new(n: usize, l: f64) -> PyResult<Self> {
        let inner = RustCosEngine::new(n, l).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn l(&self) -> f64 {
        self.inner.l
    }

    fn price_european_heston(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyHestonModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european_bates(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBatesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european_kou(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyKouModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    fn price_european_bs(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_european_black(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionForward,
        model: &PyBlackModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_black_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionForward>>,
        model: &PyBlackModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_heston_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyHestonModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_bates_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBatesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_kou_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyKouModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("CosEngine(n={}, l={})", self.inner.n, self.inner.l)
    }
}

#[pyclass(module = "quantlib_py", name = "MonteCarloEngine", from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyMonteCarloEngine {
    pub(crate) inner: MonteCarloEngine,
}

#[pymethods]
impl PyMonteCarloEngine {
    #[new]
    #[pyo3(signature = (n_paths = 10_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, heston_scheme = DEFAULT_HESTON_SCHEME))]
    pub(crate) fn new(
        n_paths: usize,
        n_steps: usize,
        seed: Option<u64>,
        parallel: bool,
        chunk_size: usize,
        heston_scheme: &str,
    ) -> PyResult<Self> {
        let heston_scheme = heston_scheme.parse::<HestonScheme>().map_err(to_py_err)?;
        let inner = MonteCarloEngine::new(n_paths, n_steps, seed)
            .map_err(to_py_err)?
            .with_parallel(parallel)
            .with_chunk_size(chunk_size)
            .with_heston_scheme(heston_scheme);

        Ok(Self { inner })
    }

    #[getter]
    fn n_paths(&self) -> usize {
        self.inner.n_paths
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    #[getter]
    fn parallel(&self) -> bool {
        self.inner.parallel
    }

    #[getter]
    fn chunk_size(&self) -> usize {
        self.inner.chunk_size
    }

    #[getter]
    fn heston_scheme(&self) -> &'static str {
        heston_scheme_name(self.inner.heston_scheme)
    }

    fn price_european_bs(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_european_heston(
        &self,
        py: Python<'_>,
        option: &PyEuropeanOptionSpot,
        model: &PyHestonModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_european_heston_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyEuropeanOptionSpot>>,
        model: &PyHestonModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_digital_bs(
        &self,
        py: Python<'_>,
        option: &PyDigitalOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_digital_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyDigitalOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_digital_heston(
        &self,
        py: Python<'_>,
        option: &PyDigitalOptionSpot,
        model: &PyHestonModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_digital_heston_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyDigitalOptionSpot>>,
        model: &PyHestonModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_asian_bs(
        &self,
        py: Python<'_>,
        option: &PyAsianOptionSpot,
        model: &PyBlackScholesModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_asian_bs_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyAsianOptionSpot>>,
        model: &PyBlackScholesModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn price_asian_heston(
        &self,
        py: Python<'_>,
        option: &PyAsianOptionSpot,
        model: &PyHestonModel,
    ) -> PyResult<f64> {
        let option = option.inner.clone();
        let model = model.inner;
        let engine = self.inner;

        py.detach(move || engine.calculate_price(&option, &model))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (options, model, parallel_batch = true))]
    fn price_asian_heston_batch(
        &self,
        py: Python<'_>,
        options: Vec<Py<PyAsianOptionSpot>>,
        model: &PyHestonModel,
        parallel_batch: bool,
    ) -> PyResult<Vec<f64>> {
        let products = options
            .into_iter()
            .map(|option| option.borrow(py).inner.clone())
            .collect::<Vec<_>>();

        let model = model.inner;
        let engine = self.inner;

        py.detach(move || price_products_single_model(engine, products, model, parallel_batch))
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "MonteCarloEngine(n_paths={}, n_steps={}, seed={:?}, parallel={}, chunk_size={}, heston_scheme='{}')",
            self.inner.n_paths,
            self.inner.n_steps,
            self.inner.seed,
            self.inner.parallel,
            self.inner.chunk_size,
            self.heston_scheme()
        )
    }
}

pub(crate) fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlackScholesModel>()?;
    m.add_class::<PyBlackModel>()?;
    m.add_class::<PyHestonModel>()?;
    m.add_class::<PyBatesModel>()?;
    m.add_class::<PyKouModel>()?;
    m.add_class::<PyEuropeanOptionSpot>()?;
    m.add_class::<PyEuropeanOptionForward>()?;
    m.add_class::<PyDigitalOptionSpot>()?;
    m.add_class::<PyDigitalOptionForward>()?;
    m.add_class::<PyAsianOptionSpot>()?;
    m.add_class::<PyAnalyticalEngine>()?;
    m.add_class::<PyFFTEngine>()?;
    m.add_class::<PyCosEngine>()?;
    m.add_class::<PyMonteCarloEngine>()?;
    Ok(())
}
