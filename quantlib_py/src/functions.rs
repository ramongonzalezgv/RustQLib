use std::str::FromStr;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use quantlib_core::{CalibrationResult, OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{AnalyticalEngine, FFTEngine as RustFFTEngine, MonteCarloEngine};
use quantlib_models::{BatesModel, BlackModel, BlackScholesModel, HestonModel, KouModel};
use quantlib_products::{AsianOption, EuropeanOption};
use quantlib_utils::{
    CalibrationConfig, DifferentialEvolutionConfig, HestonFftConfig, NelderMeadConfig,
    SabrCalibrationQuote, VanillaMarketValueKind, VanillaOptionCalibrationQuote,
    calibrate_black_model, calibrate_black_scholes_model, calibrate_heston_model,
    calibrate_sabr_model,
};

use crate::bindings::{
    PyAsianOptionSpot, PyBatesModel, PyBlackModel, PyBlackScholesModel, PyDigitalOptionForward,
    PyDigitalOptionSpot, PyEuropeanOptionForward, PyEuropeanOptionSpot, PyHestonModel, PyKouModel,
    PyMonteCarloEngine, parse_averaging_type, parse_calibration_method,
    parse_calibration_objective, parse_calibration_target, parse_date, parse_fft_interpolation,
    price_products_per_model, price_products_single_model, require_contiguous, to_py_err,
};

fn resolve_weights(length: usize, weights: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
    let weights = weights.unwrap_or_else(|| vec![1.0; length]);
    if weights.len() != length {
        return Err(PyValueError::new_err(format!(
            "weights length ({}) must match quotes length ({length})",
            weights.len()
        )));
    }
    Ok(weights)
}

#[allow(clippy::too_many_arguments)]
fn make_calibration_config(
    method: &str,
    objective: &str,
    huber_delta: Option<f64>,
    max_generations: usize,
    population_size: usize,
    nm_max_iterations: usize,
    tolerance: f64,
    seed: u64,
    parallel_objective: bool,
) -> PyResult<CalibrationConfig> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PyValueError::new_err(
            "tolerance must be finite and positive",
        ));
    }

    let method = parse_calibration_method(method)?;
    let objective = parse_calibration_objective(objective, huber_delta)?;

    let de = DifferentialEvolutionConfig {
        max_generations,
        population_size,
        tolerance,
        seed,
        parallel_objective,
        ..DifferentialEvolutionConfig::default()
    };
    let nm = NelderMeadConfig {
        max_iterations: nm_max_iterations,
        tolerance,
        ..NelderMeadConfig::default()
    };

    Ok(CalibrationConfig {
        method,
        objective,
        differential_evolution: de,
        nelder_mead: nm,
    })
}

fn calibration_result_to_dict<'py>(
    py: Python<'py>,
    result: &CalibrationResult,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    let params = PyDict::new(py);
    for (name, value) in &result.calibrated_params {
        params.set_item(name, *value)?;
    }

    payload.set_item("success", result.success)?;
    payload.set_item("message", &result.message)?;
    payload.set_item("iterations", result.iterations)?;
    payload.set_item("objective_value", result.objective_value)?;
    payload.set_item("rmse", result.rmse)?;
    payload.set_item("mape", result.mape)?;
    payload.set_item("max_error", result.max_error)?;
    payload.set_item("calibrated_params", params)?;
    payload.set_item("errors", result.errors.clone())?;
    payload.set_item("market_values", result.market_values.clone())?;
    payload.set_item("model_values", result.model_values.clone())?;

    Ok(payload)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, sigma, r, q, valuation_date = None))]
fn price_european_analytical_bs(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    q: f64,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyBlackScholesModel::new(sigma, r, q)?;
    let engine = AnalyticalEngine;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, option_types, qtys, sigma, r, q, parallel = true))]
fn price_european_analytical_bs_batch(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    q: f64,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let n = spots.len();
    if strikes.len() != n || expiry_dates.len() != n || option_types.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={n}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(n);
        for i in 0..n {
            let option_type = OptionType::from_str(&option_types[i])?;
            let expiry = SimpleDate::parse(&expiry_dates[i])?;
            let option = EuropeanOption::new_spot(
                spots[i],
                strikes[i],
                expiry,
                Some(valuation),
                option_type,
                qtys[i],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(AnalyticalEngine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, is_call, qtys, sigma, r, q, parallel = true))]
fn price_european_analytical_bs_batch_fast(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    is_call: Vec<bool>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    q: f64,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let n = spots.len();
    if strikes.len() != n || expiry_dates.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={n}, strikes={}, expiry_dates={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(n);
        for i in 0..n {
            let option_type = if is_call[i] {
                OptionType::Call
            } else {
                OptionType::Put
            };
            let expiry = SimpleDate::parse(&expiry_dates[i])?;
            let option = EuropeanOption::new_spot(
                spots[i],
                strikes[i],
                expiry,
                Some(valuation),
                option_type,
                qtys[i],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(AnalyticalEngine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, option_types, qtys, sigmas, rates, dividends, parallel = true))]
fn price_european_analytical_bs_batch_per_model(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigmas: Vec<f64>,
    rates: Vec<f64>,
    dividends: Vec<f64>,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let n = spots.len();
    if strikes.len() != n
        || expiry_dates.len() != n
        || option_types.len() != n
        || qtys.len() != n
        || sigmas.len() != n
        || rates.len() != n
        || dividends.len() != n
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={n}, strikes={}, expiry_dates={}, option_types={}, qtys={}, sigmas={}, rates={}, dividends={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len(),
            sigmas.len(),
            rates.len(),
            dividends.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(n);
        let mut models = Vec::with_capacity(n);

        for i in 0..n {
            let option_type = OptionType::from_str(&option_types[i])?;
            let expiry = SimpleDate::parse(&expiry_dates[i])?;
            let option = EuropeanOption::new_spot(
                spots[i],
                strikes[i],
                expiry,
                Some(valuation),
                option_type,
                qtys[i],
                None,
            )?;
            products.push(option);

            models.push(BlackScholesModel::new(sigmas[i], rates[i], dividends[i])?);
        }

        price_products_per_model(AnalyticalEngine, products, models, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forward, strike, expiry_date, option_type, qty, sigma, r, valuation_date = None))]
fn price_european_analytical_black(
    py: Python<'_>,
    forward: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyEuropeanOptionForward::new(
        forward,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
    )?;
    let model = PyBlackModel::new(sigma, r)?;
    let engine = AnalyticalEngine;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forwards, strikes, expiry_dates, valuation_date, option_types, qtys, sigma, r, parallel = true))]
fn price_european_analytical_black_batch(
    py: Python<'_>,
    forwards: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let n = forwards.len();
    if strikes.len() != n || expiry_dates.len() != n || option_types.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got forwards={n}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackModel::new(sigma, r).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(n);
        for i in 0..n {
            let option_type = OptionType::from_str(&option_types[i])?;
            let expiry = SimpleDate::parse(&expiry_dates[i])?;
            let option = EuropeanOption::new_forward(
                forwards[i],
                strikes[i],
                expiry,
                Some(valuation),
                option_type,
                qtys[i],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(AnalyticalEngine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forwards, strikes, expiry_dates, valuation_date, is_call, qtys, sigma, r, parallel = true))]
fn price_european_analytical_black_batch_fast(
    py: Python<'_>,
    forwards: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    is_call: Vec<bool>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let n = forwards.len();
    if strikes.len() != n || expiry_dates.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got forwards={n}, strikes={}, expiry_dates={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackModel::new(sigma, r).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(n);
        for i in 0..n {
            let option_type = if is_call[i] {
                OptionType::Call
            } else {
                OptionType::Put
            };
            let expiry = SimpleDate::parse(&expiry_dates[i])?;
            let option = EuropeanOption::new_forward(
                forwards[i],
                strikes[i],
                expiry,
                Some(valuation),
                option_type,
                qtys[i],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(AnalyticalEngine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forwards, strikes, expiry_dates, valuation_date, option_types, qtys, sigmas, rates, parallel = true))]
fn price_european_analytical_black_batch_per_model(
    py: Python<'_>,
    forwards: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigmas: Vec<f64>,
    rates: Vec<f64>,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let n = forwards.len();
    if strikes.len() != n
        || expiry_dates.len() != n
        || option_types.len() != n
        || qtys.len() != n
        || sigmas.len() != n
        || rates.len() != n
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got forwards={n}, strikes={}, expiry_dates={}, option_types={}, qtys={}, sigmas={}, rates={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len(),
            sigmas.len(),
            rates.len(),
        )));
    }

    let valuation = parse_date(valuation_date)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(n);
        let mut models = Vec::with_capacity(n);

        for i in 0..n {
            let option_type = OptionType::from_str(&option_types[i])?;
            let expiry = SimpleDate::parse(&expiry_dates[i])?;
            let option = EuropeanOption::new_forward(
                forwards[i],
                strikes[i],
                expiry,
                Some(valuation),
                option_type,
                qtys[i],
                None,
            )?;
            products.push(option);

            models.push(BlackModel::new(sigmas[i], rates[i])?);
        }

        price_products_per_model(AnalyticalEngine, products, models, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, sigma, r, q, payout = 1.0, payoff_type = "cash", valuation_date = None))]
fn price_digital_analytical_bs(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    q: f64,
    payout: f64,
    payoff_type: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyDigitalOptionSpot::new(
        spot,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
        payout,
        payoff_type,
    )?;
    let model = PyBlackScholesModel::new(sigma, r, q)?;
    let engine = AnalyticalEngine;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forward, strike, expiry_date, option_type, qty, sigma, r, payout = 1.0, payoff_type = "cash", valuation_date = None))]
fn price_digital_analytical_black(
    py: Python<'_>,
    forward: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    payout: f64,
    payoff_type: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyDigitalOptionForward::new(
        forward,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
        payout,
        payoff_type,
    )?;
    let model = PyBlackModel::new(sigma, r)?;
    let engine = AnalyticalEngine;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, sigma, r, q, payout = 1.0, payoff_type = "cash", n_paths = 100_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, valuation_date = None))]
fn price_digital_monte_carlo_bs(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    q: f64,
    payout: f64,
    payoff_type: &str,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel: bool,
    chunk_size: usize,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyDigitalOptionSpot::new(
        spot,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
        payout,
        payoff_type,
    )?;
    let model = PyBlackScholesModel::new(sigma, r, q)?;
    let engine = PyMonteCarloEngine::new(n_paths, n_steps, seed, parallel, chunk_size)?;

    py.detach(move || engine.inner.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, kappa, theta, sigma, rho, v0, r, q, payout = 1.0, payoff_type = "cash", n_paths = 100_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, valuation_date = None))]
fn price_digital_monte_carlo_heston(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    payout: f64,
    payoff_type: &str,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel: bool,
    chunk_size: usize,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyDigitalOptionSpot::new(
        spot,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
        payout,
        payoff_type,
    )?;
    let model = PyHestonModel::new(kappa, theta, sigma, rho, v0, r, q)?;
    let engine = PyMonteCarloEngine::new(n_paths, n_steps, seed, parallel, chunk_size)?;

    py.detach(move || engine.inner.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, sigma, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, valuation_date = None))]
fn price_european_monte_carlo_bs(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel: bool,
    chunk_size: usize,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyBlackScholesModel::new(sigma, r, q)?;
    let engine = PyMonteCarloEngine::new(n_paths, n_steps, seed, parallel, chunk_size)?;

    py.detach(move || engine.inner.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, kappa, theta, sigma, rho, v0, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, valuation_date = None))]
fn price_european_monte_carlo_heston(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel: bool,
    chunk_size: usize,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyHestonModel::new(kappa, theta, sigma, rho, v0, r, q)?;
    let engine = PyMonteCarloEngine::new(n_paths, n_steps, seed, parallel, chunk_size)?;

    py.detach(move || engine.inner.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, sigma, r, q, n = 4_096, b = 200.0, interp = "cubic", valuation_date = None))]
fn price_european_fft_bs(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    q: f64,
    n: usize,
    b: f64,
    interp: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyBlackScholesModel::new(sigma, r, q)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forwards, strikes, expiry_dates, valuation_date, option_types, qtys, sigma, r, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_black_batch(
    py: Python<'_>,
    forwards: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let rows = forwards.len();
    if strikes.len() != rows
        || expiry_dates.len() != rows
        || option_types.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got forwards={rows}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackModel::new(sigma, r).map_err(to_py_err)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(rows);
        for index in 0..rows {
            let option_type = OptionType::from_str(&option_types[index])?;
            let expiry = SimpleDate::parse(&expiry_dates[index])?;
            let option = EuropeanOption::new_forward(
                forwards[index],
                strikes[index],
                expiry,
                Some(valuation),
                option_type,
                qtys[index],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(engine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forward, strike, expiry_date, option_type, qty, sigma, r, n = 4_096, b = 200.0, interp = "cubic", valuation_date = None))]
fn price_european_fft_black(
    py: Python<'_>,
    forward: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    n: usize,
    b: f64,
    interp: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyEuropeanOptionForward::new(
        forward,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
    )?;
    let model = PyBlackModel::new(sigma, r)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, option_types, qtys, sigma, r, q, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_bs_batch(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    q: f64,
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let rows = spots.len();
    if strikes.len() != rows
        || expiry_dates.len() != rows
        || option_types.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={rows}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(rows);
        for index in 0..rows {
            let option_type = OptionType::from_str(&option_types[index])?;
            let expiry = SimpleDate::parse(&expiry_dates[index])?;
            let option = EuropeanOption::new_spot(
                spots[index],
                strikes[index],
                expiry,
                Some(valuation),
                option_type,
                qtys[index],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(engine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, kappa, theta, sigma, rho, v0, r, q, n = 4_096, b = 200.0, interp = "cubic", valuation_date = None))]
fn price_european_fft_heston(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n: usize,
    b: f64,
    interp: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyHestonModel::new(kappa, theta, sigma, rho, v0, r, q)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, option_types, qtys, kappa, theta, sigma, rho, v0, r, q, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_heston_batch(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let rows = spots.len();
    if strikes.len() != rows
        || expiry_dates.len() != rows
        || option_types.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={rows}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).map_err(to_py_err)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(rows);
        for index in 0..rows {
            let option_type = OptionType::from_str(&option_types[index])?;
            let expiry = SimpleDate::parse(&expiry_dates[index])?;
            let option = EuropeanOption::new_spot(
                spots[index],
                strikes[index],
                expiry,
                Some(valuation),
                option_type,
                qtys[index],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(engine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, kappa, theta, sigma, rho, v0, r, q, lambda_jump, mu_j, sigma_j, n = 4_096, b = 200.0, interp = "cubic", valuation_date = None))]
fn price_european_fft_bates(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
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
    n: usize,
    b: f64,
    interp: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyBatesModel::new(
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
    )?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, option_types, qtys, kappa, theta, sigma, rho, v0, r, q, lambda_jump, mu_j, sigma_j, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_bates_batch(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
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
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let rows = spots.len();
    if strikes.len() != rows
        || expiry_dates.len() != rows
        || option_types.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={rows}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BatesModel::new(
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
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(rows);
        for index in 0..rows {
            let option_type = OptionType::from_str(&option_types[index])?;
            let expiry = SimpleDate::parse(&expiry_dates[index])?;
            let option = EuropeanOption::new_spot(
                spots[index],
                strikes[index],
                expiry,
                Some(valuation),
                option_type,
                qtys[index],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(engine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, sigma, r, q, lambda_jump, p_up, eta1, eta2, n = 4_096, b = 200.0, interp = "cubic", valuation_date = None))]
fn price_european_fft_kou(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    sigma: f64,
    r: f64,
    q: f64,
    lambda_jump: f64,
    p_up: f64,
    eta1: f64,
    eta2: f64,
    n: usize,
    b: f64,
    interp: &str,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option =
        PyEuropeanOptionSpot::new(spot, strike, expiry_date, valuation_date, option_type, qty)?;
    let model = PyKouModel::new(sigma, r, q, lambda_jump, p_up, eta1, eta2)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || engine.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_dates, valuation_date, option_types, qtys, sigma, r, q, lambda_jump, p_up, eta1, eta2, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_kou_batch(
    py: Python<'_>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    expiry_dates: Vec<String>,
    valuation_date: &str,
    option_types: Vec<String>,
    qtys: Vec<i32>,
    sigma: f64,
    r: f64,
    q: f64,
    lambda_jump: f64,
    p_up: f64,
    eta1: f64,
    eta2: f64,
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Vec<f64>> {
    let rows = spots.len();
    if strikes.len() != rows
        || expiry_dates.len() != rows
        || option_types.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={rows}, strikes={}, expiry_dates={}, option_types={}, qtys={}",
            strikes.len(),
            expiry_dates.len(),
            option_types.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = KouModel::new(sigma, r, q, lambda_jump, p_up, eta1, eta2).map_err(to_py_err)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    py.detach(move || {
        let mut products = Vec::with_capacity(rows);
        for index in 0..rows {
            let option_type = OptionType::from_str(&option_types[index])?;
            let expiry = SimpleDate::parse(&expiry_dates[index])?;
            let option = EuropeanOption::new_spot(
                spots[index],
                strikes[index],
                expiry,
                Some(valuation),
                option_type,
                qtys[index],
                None,
            )?;
            products.push(option);
        }

        price_products_single_model(engine, products, model, parallel)
    })
    .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, averaging_type, sigma, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, valuation_date = None))]
fn price_asian_monte_carlo_bs(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    averaging_type: &str,
    sigma: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel: bool,
    chunk_size: usize,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyAsianOptionSpot::new(
        spot,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
        averaging_type,
    )?;
    let model = PyBlackScholesModel::new(sigma, r, q)?;
    let engine = PyMonteCarloEngine::new(n_paths, n_steps, seed, parallel, chunk_size)?;

    py.detach(move || engine.inner.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spot, strike, expiry_date, option_type, qty, averaging_type, kappa, theta, sigma, rho, v0, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel = true, chunk_size = 4_096, valuation_date = None))]
fn price_asian_monte_carlo_heston(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    expiry_date: &str,
    option_type: &str,
    qty: i32,
    averaging_type: &str,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel: bool,
    chunk_size: usize,
    valuation_date: Option<&str>,
) -> PyResult<f64> {
    let option = PyAsianOptionSpot::new(
        spot,
        strike,
        expiry_date,
        valuation_date,
        option_type,
        qty,
        averaging_type,
    )?;
    let model = PyHestonModel::new(kappa, theta, sigma, rho, v0, r, q)?;
    let engine = PyMonteCarloEngine::new(n_paths, n_steps, seed, parallel, chunk_size)?;

    py.detach(move || engine.inner.calculate_price(&option.inner, &model.inner))
        .map_err(to_py_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, sigma, r, q, parallel = true))]
fn price_european_analytical_bs_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    sigma: f64,
    r: f64,
    q: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let n = spots.len();
    if strikes.len() != n || expiry_days.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={n}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;

    let mut products = Vec::with_capacity(n);
    for i in 0..n {
        let option_type = if is_call[i] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[i]);
        let option = EuropeanOption::new_spot(
            spots[i],
            strikes[i],
            expiry,
            Some(valuation),
            option_type,
            qtys[i],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices = price_products_single_model(AnalyticalEngine, products, model, parallel)
        .map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (forwards, strikes, expiry_days, is_call, qtys, valuation_date, sigma, r, parallel = true))]
fn price_european_analytical_black_batch_numpy<'py>(
    py: Python<'py>,
    forwards: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    sigma: f64,
    r: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let forwards = require_contiguous(&forwards, "forwards")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let n = forwards.len();
    if strikes.len() != n || expiry_days.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got forwards={n}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackModel::new(sigma, r).map_err(to_py_err)?;

    let mut products = Vec::with_capacity(n);
    for i in 0..n {
        let option_type = if is_call[i] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[i]);
        let option = EuropeanOption::new_forward(
            forwards[i],
            strikes[i],
            expiry,
            Some(valuation),
            option_type,
            qtys[i],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices = price_products_single_model(AnalyticalEngine, products, model, parallel)
        .map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, sigma, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel_paths = true, chunk_size = 4_096, parallel_batch = true))]
fn price_european_monte_carlo_bs_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    sigma: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel_paths: bool,
    chunk_size: usize,
    parallel_batch: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let n = spots.len();
    if strikes.len() != n || expiry_days.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={n}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;
    let engine = MonteCarloEngine::new(n_paths, n_steps, seed)
        .map_err(to_py_err)?
        .with_parallel(parallel_paths)
        .with_chunk_size(chunk_size);

    let mut products = Vec::with_capacity(n);
    for i in 0..n {
        let option_type = if is_call[i] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[i]);
        let option = EuropeanOption::new_spot(
            spots[i],
            strikes[i],
            expiry,
            Some(valuation),
            option_type,
            qtys[i],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel_batch).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, kappa, theta, sigma, rho, v0, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel_paths = true, chunk_size = 4_096, parallel_batch = true))]
fn price_european_monte_carlo_heston_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel_paths: bool,
    chunk_size: usize,
    parallel_batch: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let n = spots.len();
    if strikes.len() != n || expiry_days.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={n}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).map_err(to_py_err)?;
    let engine = MonteCarloEngine::new(n_paths, n_steps, seed)
        .map_err(to_py_err)?
        .with_parallel(parallel_paths)
        .with_chunk_size(chunk_size);

    let mut products = Vec::with_capacity(n);
    for i in 0..n {
        let option_type = if is_call[i] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[i]);
        let option = EuropeanOption::new_spot(
            spots[i],
            strikes[i],
            expiry,
            Some(valuation),
            option_type,
            qtys[i],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel_batch).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, kappa, theta, sigma, rho, v0, r, q, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_heston_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let rows = spots.len();
    if strikes.len() != rows
        || expiry_days.len() != rows
        || is_call.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={rows}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).map_err(to_py_err)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    let mut products = Vec::with_capacity(rows);
    for index in 0..rows {
        let option_type = if is_call[index] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[index]);
        let option = EuropeanOption::new_spot(
            spots[index],
            strikes[index],
            expiry,
            Some(valuation),
            option_type,
            qtys[index],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, kappa, theta, sigma, rho, v0, r, q, lambda_jump, mu_j, sigma_j, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_bates_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
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
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let rows = spots.len();
    if strikes.len() != rows
        || expiry_days.len() != rows
        || is_call.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={rows}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = BatesModel::new(
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
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    let mut products = Vec::with_capacity(rows);
    for index in 0..rows {
        let option_type = if is_call[index] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[index]);
        let option = EuropeanOption::new_spot(
            spots[index],
            strikes[index],
            expiry,
            Some(valuation),
            option_type,
            qtys[index],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, sigma, r, q, lambda_jump, p_up, eta1, eta2, n = 4_096, b = 200.0, interp = "cubic", parallel = true))]
fn price_european_fft_kou_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    sigma: f64,
    r: f64,
    q: f64,
    lambda_jump: f64,
    p_up: f64,
    eta1: f64,
    eta2: f64,
    n: usize,
    b: f64,
    interp: &str,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let rows = spots.len();
    if strikes.len() != rows
        || expiry_days.len() != rows
        || is_call.len() != rows
        || qtys.len() != rows
    {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={rows}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let model = KouModel::new(sigma, r, q, lambda_jump, p_up, eta1, eta2).map_err(to_py_err)?;
    let interp = parse_fft_interpolation(interp)?;
    let engine = RustFFTEngine::new(n, b, interp).map_err(to_py_err)?;

    let mut products = Vec::with_capacity(rows);
    for index in 0..rows {
        let option_type = if is_call[index] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[index]);
        let option = EuropeanOption::new_spot(
            spots[index],
            strikes[index],
            expiry,
            Some(valuation),
            option_type,
            qtys[index],
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, averaging_type, sigma, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel_paths = true, chunk_size = 4_096, parallel_batch = true))]
fn price_asian_monte_carlo_bs_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    averaging_type: &str,
    sigma: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel_paths: bool,
    chunk_size: usize,
    parallel_batch: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let n = spots.len();
    if strikes.len() != n || expiry_days.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={n}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let averaging_type = parse_averaging_type(averaging_type)?;
    let model = BlackScholesModel::new(sigma, r, q).map_err(to_py_err)?;
    let engine = MonteCarloEngine::new(n_paths, n_steps, seed)
        .map_err(to_py_err)?
        .with_parallel(parallel_paths)
        .with_chunk_size(chunk_size);

    let mut products = Vec::with_capacity(n);
    for i in 0..n {
        let option_type = if is_call[i] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[i]);
        let option = AsianOption::new(
            spots[i],
            strikes[i],
            expiry,
            Some(valuation),
            option_type,
            qtys[i],
            averaging_type,
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel_batch).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (spots, strikes, expiry_days, is_call, qtys, valuation_date, averaging_type, kappa, theta, sigma, rho, v0, r, q, n_paths = 100_000, n_steps = 252, seed = None, parallel_paths = true, chunk_size = 4_096, parallel_batch = true))]
fn price_asian_monte_carlo_heston_batch_numpy<'py>(
    py: Python<'py>,
    spots: PyReadonlyArray1<'py, f64>,
    strikes: PyReadonlyArray1<'py, f64>,
    expiry_days: PyReadonlyArray1<'py, i64>,
    is_call: PyReadonlyArray1<'py, bool>,
    qtys: PyReadonlyArray1<'py, i32>,
    valuation_date: &str,
    averaging_type: &str,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    v0: f64,
    r: f64,
    q: f64,
    n_paths: usize,
    n_steps: usize,
    seed: Option<u64>,
    parallel_paths: bool,
    chunk_size: usize,
    parallel_batch: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let spots = require_contiguous(&spots, "spots")?;
    let strikes = require_contiguous(&strikes, "strikes")?;
    let expiry_days = require_contiguous(&expiry_days, "expiry_days")?;
    let is_call = require_contiguous(&is_call, "is_call")?;
    let qtys = require_contiguous(&qtys, "qtys")?;

    let n = spots.len();
    if strikes.len() != n || expiry_days.len() != n || is_call.len() != n || qtys.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input arrays must have the same length, got spots={n}, strikes={}, expiry_days={}, is_call={}, qtys={}",
            strikes.len(),
            expiry_days.len(),
            is_call.len(),
            qtys.len()
        )));
    }

    let valuation = parse_date(valuation_date)?;
    let averaging_type = parse_averaging_type(averaging_type)?;
    let model = HestonModel::new(kappa, theta, sigma, rho, v0, r, q).map_err(to_py_err)?;
    let engine = MonteCarloEngine::new(n_paths, n_steps, seed)
        .map_err(to_py_err)?
        .with_parallel(parallel_paths)
        .with_chunk_size(chunk_size);

    let mut products = Vec::with_capacity(n);
    for i in 0..n {
        let option_type = if is_call[i] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        let expiry = valuation.add_days(expiry_days[i]);
        let option = AsianOption::new(
            spots[i],
            strikes[i],
            expiry,
            Some(valuation),
            option_type,
            qtys[i],
            averaging_type,
            None,
        )
        .map_err(to_py_err)?;
        products.push(option);
    }

    let prices =
        price_products_single_model(engine, products, model, parallel_batch).map_err(to_py_err)?;
    Ok(prices.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    forwards,
    strikes,
    maturities,
    is_call,
    market_values,
    target = "price",
    weights = None,
    initial_sigma = 0.2,
    initial_r = 0.02,
    sigma_bounds = (1.0e-4, 3.0),
    r_bounds = (-0.2, 0.2),
    method = "hybrid",
    objective = "least_squares",
    huber_delta = None,
    max_generations = 300,
    population_size = 48,
    nm_max_iterations = 300,
    tolerance = 1.0e-6,
    seed = 42,
    parallel_objective = true
))]
fn calibrate_black<'py>(
    py: Python<'py>,
    forwards: Vec<f64>,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    is_call: Vec<bool>,
    market_values: Vec<f64>,
    target: &str,
    weights: Option<Vec<f64>>,
    initial_sigma: f64,
    initial_r: f64,
    sigma_bounds: (f64, f64),
    r_bounds: (f64, f64),
    method: &str,
    objective: &str,
    huber_delta: Option<f64>,
    max_generations: usize,
    population_size: usize,
    nm_max_iterations: usize,
    tolerance: f64,
    seed: u64,
    parallel_objective: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = forwards.len();
    if strikes.len() != n || maturities.len() != n || is_call.len() != n || market_values.len() != n
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got forwards={n}, strikes={}, maturities={}, is_call={}, market_values={}",
            strikes.len(),
            maturities.len(),
            is_call.len(),
            market_values.len()
        )));
    }

    let weights = resolve_weights(n, weights)?;
    let target = parse_calibration_target(target)?;
    let market_value_kind = match target {
        quantlib_utils::CalibrationTarget::Price => VanillaMarketValueKind::Price,
        quantlib_utils::CalibrationTarget::Volatility => VanillaMarketValueKind::ImpliedVolatility,
    };
    let config = make_calibration_config(
        method,
        objective,
        huber_delta,
        max_generations,
        population_size,
        nm_max_iterations,
        tolerance,
        seed,
        parallel_objective,
    )?;

    let mut quotes = Vec::with_capacity(n);
    for index in 0..n {
        let option_type = if is_call[index] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        quotes.push(VanillaOptionCalibrationQuote {
            underlying: forwards[index],
            strike: strikes[index],
            maturity: maturities[index],
            option_type,
            market_value: market_values[index],
            market_value_kind,
            weight: weights[index],
        });
    }

    let bounds = [sigma_bounds, r_bounds];
    let (model, result) = py
        .detach(move || {
            calibrate_black_model(
                &quotes,
                target,
                initial_sigma,
                initial_r,
                Some(bounds),
                &config,
            )
        })
        .map_err(to_py_err)?;

    let payload = calibration_result_to_dict(py, &result)?;
    let model_dict = PyDict::new(py);
    model_dict.set_item("sigma", model.sigma)?;
    model_dict.set_item("r", model.r)?;
    payload.set_item("model", model_dict)?;
    payload.set_item("model_name", "BlackModel")?;
    Ok(payload)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    spots,
    strikes,
    maturities,
    is_call,
    market_values,
    target = "price",
    weights = None,
    initial_sigma = 0.2,
    initial_r = 0.02,
    initial_q = 0.0,
    sigma_bounds = (1.0e-4, 3.0),
    r_bounds = (-0.2, 0.2),
    q_bounds = (-0.2, 0.2),
    method = "hybrid",
    objective = "least_squares",
    huber_delta = None,
    max_generations = 300,
    population_size = 48,
    nm_max_iterations = 300,
    tolerance = 1.0e-6,
    seed = 42,
    parallel_objective = true
))]
fn calibrate_black_scholes<'py>(
    py: Python<'py>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    is_call: Vec<bool>,
    market_values: Vec<f64>,
    target: &str,
    weights: Option<Vec<f64>>,
    initial_sigma: f64,
    initial_r: f64,
    initial_q: f64,
    sigma_bounds: (f64, f64),
    r_bounds: (f64, f64),
    q_bounds: (f64, f64),
    method: &str,
    objective: &str,
    huber_delta: Option<f64>,
    max_generations: usize,
    population_size: usize,
    nm_max_iterations: usize,
    tolerance: f64,
    seed: u64,
    parallel_objective: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = spots.len();
    if strikes.len() != n || maturities.len() != n || is_call.len() != n || market_values.len() != n
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={n}, strikes={}, maturities={}, is_call={}, market_values={}",
            strikes.len(),
            maturities.len(),
            is_call.len(),
            market_values.len()
        )));
    }

    let weights = resolve_weights(n, weights)?;
    let target = parse_calibration_target(target)?;
    let market_value_kind = match target {
        quantlib_utils::CalibrationTarget::Price => VanillaMarketValueKind::Price,
        quantlib_utils::CalibrationTarget::Volatility => VanillaMarketValueKind::ImpliedVolatility,
    };
    let config = make_calibration_config(
        method,
        objective,
        huber_delta,
        max_generations,
        population_size,
        nm_max_iterations,
        tolerance,
        seed,
        parallel_objective,
    )?;

    let mut quotes = Vec::with_capacity(n);
    for index in 0..n {
        let option_type = if is_call[index] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        quotes.push(VanillaOptionCalibrationQuote {
            underlying: spots[index],
            strike: strikes[index],
            maturity: maturities[index],
            option_type,
            market_value: market_values[index],
            market_value_kind,
            weight: weights[index],
        });
    }

    let bounds = [sigma_bounds, r_bounds, q_bounds];
    let (model, result) = py
        .detach(move || {
            calibrate_black_scholes_model(
                &quotes,
                target,
                initial_sigma,
                initial_r,
                initial_q,
                Some(bounds),
                &config,
            )
        })
        .map_err(to_py_err)?;

    let payload = calibration_result_to_dict(py, &result)?;
    let model_dict = PyDict::new(py);
    model_dict.set_item("sigma", model.sigma)?;
    model_dict.set_item("r", model.r)?;
    model_dict.set_item("q", model.q)?;
    payload.set_item("model", model_dict)?;
    payload.set_item("model_name", "BlackScholesModel")?;
    Ok(payload)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    forwards,
    strikes,
    maturities,
    market_vols,
    r = 0.0,
    weights = None,
    initial_alpha = 0.2,
    initial_beta = 0.5,
    initial_rho = -0.2,
    initial_nu = 0.5,
    alpha_bounds = (1.0e-4, 3.0),
    beta_bounds = (0.01, 0.99),
    rho_bounds = (-0.999, 0.999),
    nu_bounds = (1.0e-4, 3.0),
    method = "hybrid",
    objective = "relative_least_squares",
    huber_delta = None,
    max_generations = 350,
    population_size = 48,
    nm_max_iterations = 320,
    tolerance = 1.0e-6,
    seed = 42,
    parallel_objective = true
))]
fn calibrate_sabr<'py>(
    py: Python<'py>,
    forwards: Vec<f64>,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    market_vols: Vec<f64>,
    r: f64,
    weights: Option<Vec<f64>>,
    initial_alpha: f64,
    initial_beta: f64,
    initial_rho: f64,
    initial_nu: f64,
    alpha_bounds: (f64, f64),
    beta_bounds: (f64, f64),
    rho_bounds: (f64, f64),
    nu_bounds: (f64, f64),
    method: &str,
    objective: &str,
    huber_delta: Option<f64>,
    max_generations: usize,
    population_size: usize,
    nm_max_iterations: usize,
    tolerance: f64,
    seed: u64,
    parallel_objective: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = forwards.len();
    if strikes.len() != n || maturities.len() != n || market_vols.len() != n {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got forwards={n}, strikes={}, maturities={}, market_vols={}",
            strikes.len(),
            maturities.len(),
            market_vols.len()
        )));
    }

    let weights = resolve_weights(n, weights)?;
    let config = make_calibration_config(
        method,
        objective,
        huber_delta,
        max_generations,
        population_size,
        nm_max_iterations,
        tolerance,
        seed,
        parallel_objective,
    )?;

    let mut quotes = Vec::with_capacity(n);
    for index in 0..n {
        quotes.push(SabrCalibrationQuote {
            forward: forwards[index],
            strike: strikes[index],
            maturity: maturities[index],
            market_volatility: market_vols[index],
            weight: weights[index],
        });
    }

    let bounds = [alpha_bounds, beta_bounds, rho_bounds, nu_bounds];
    let (model, result) = py
        .detach(move || {
            calibrate_sabr_model(
                &quotes,
                initial_alpha,
                initial_beta,
                initial_rho,
                initial_nu,
                r,
                Some(bounds),
                &config,
            )
        })
        .map_err(to_py_err)?;

    let payload = calibration_result_to_dict(py, &result)?;
    let model_dict = PyDict::new(py);
    let params = model.get_parameters_map();
    for (name, value) in params {
        let parsed = value.parse::<f64>().map_err(|_| {
            PyValueError::new_err(format!(
                "failed to parse calibrated SABR parameter '{name}'"
            ))
        })?;
        model_dict.set_item(name, parsed)?;
    }
    payload.set_item("model", model_dict)?;
    payload.set_item("model_name", "SABRModel")?;
    Ok(payload)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    spots,
    strikes,
    maturities,
    is_call,
    market_prices,
    weights = None,
    initial_kappa = 1.5,
    initial_theta = 0.04,
    initial_sigma = 0.4,
    initial_rho = -0.5,
    initial_v0 = 0.04,
    initial_r = 0.02,
    initial_q = 0.0,
    kappa_bounds = (1.0e-4, 20.0),
    theta_bounds = (1.0e-5, 2.0),
    sigma_bounds = (1.0e-4, 5.0),
    rho_bounds = (-0.999, 0.999),
    v0_bounds = (1.0e-5, 2.0),
    r_bounds = (-0.2, 0.2),
    q_bounds = (-0.2, 0.2),
    method = "hybrid",
    objective = "least_squares",
    huber_delta = None,
    fft_n = 1024,
    fft_b = 180.0,
    fft_interp = "cubic",
    max_generations = 240,
    population_size = 36,
    nm_max_iterations = 260,
    tolerance = 1.0e-6,
    seed = 42,
    parallel_objective = true
))]
fn calibrate_heston<'py>(
    py: Python<'py>,
    spots: Vec<f64>,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    is_call: Vec<bool>,
    market_prices: Vec<f64>,
    weights: Option<Vec<f64>>,
    initial_kappa: f64,
    initial_theta: f64,
    initial_sigma: f64,
    initial_rho: f64,
    initial_v0: f64,
    initial_r: f64,
    initial_q: f64,
    kappa_bounds: (f64, f64),
    theta_bounds: (f64, f64),
    sigma_bounds: (f64, f64),
    rho_bounds: (f64, f64),
    v0_bounds: (f64, f64),
    r_bounds: (f64, f64),
    q_bounds: (f64, f64),
    method: &str,
    objective: &str,
    huber_delta: Option<f64>,
    fft_n: usize,
    fft_b: f64,
    fft_interp: &str,
    max_generations: usize,
    population_size: usize,
    nm_max_iterations: usize,
    tolerance: f64,
    seed: u64,
    parallel_objective: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = spots.len();
    if strikes.len() != n || maturities.len() != n || is_call.len() != n || market_prices.len() != n
    {
        return Err(PyValueError::new_err(format!(
            "all input vectors must have the same length, got spots={n}, strikes={}, maturities={}, is_call={}, market_prices={}",
            strikes.len(),
            maturities.len(),
            is_call.len(),
            market_prices.len()
        )));
    }

    let weights = resolve_weights(n, weights)?;
    let config = make_calibration_config(
        method,
        objective,
        huber_delta,
        max_generations,
        population_size,
        nm_max_iterations,
        tolerance,
        seed,
        parallel_objective,
    )?;
    let interp = parse_fft_interpolation(fft_interp)?;
    let fft_config = HestonFftConfig {
        n: fft_n,
        b: fft_b,
        interp,
    };

    let initial_model = HestonModel::new(
        initial_kappa,
        initial_theta,
        initial_sigma,
        initial_rho,
        initial_v0,
        initial_r,
        initial_q,
    )
    .map_err(to_py_err)?;

    let mut quotes = Vec::with_capacity(n);
    for index in 0..n {
        let option_type = if is_call[index] {
            OptionType::Call
        } else {
            OptionType::Put
        };
        quotes.push(VanillaOptionCalibrationQuote {
            underlying: spots[index],
            strike: strikes[index],
            maturity: maturities[index],
            option_type,
            market_value: market_prices[index],
            market_value_kind: VanillaMarketValueKind::Price,
            weight: weights[index],
        });
    }

    let bounds = [
        kappa_bounds,
        theta_bounds,
        sigma_bounds,
        rho_bounds,
        v0_bounds,
        r_bounds,
        q_bounds,
    ];

    let (model, result) = py
        .detach(move || {
            calibrate_heston_model(&quotes, initial_model, Some(bounds), &config, fft_config)
        })
        .map_err(to_py_err)?;

    let payload = calibration_result_to_dict(py, &result)?;
    let model_dict = PyDict::new(py);
    model_dict.set_item("kappa", model.kappa)?;
    model_dict.set_item("theta", model.theta)?;
    model_dict.set_item("sigma", model.sigma)?;
    model_dict.set_item("rho", model.rho)?;
    model_dict.set_item("v0", model.v0)?;
    model_dict.set_item("r", model.r)?;
    model_dict.set_item("q", model.q)?;
    payload.set_item("model", model_dict)?;
    payload.set_item("model_name", "HestonModel")?;
    Ok(payload)
}

pub(crate) fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_european_analytical_bs, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_analytical_bs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        price_european_analytical_bs_batch_fast,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        price_european_analytical_bs_batch_per_model,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(price_european_analytical_black, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_analytical_black_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        price_european_analytical_black_batch_fast,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        price_european_analytical_black_batch_per_model,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(price_digital_analytical_bs, m)?)?;
    m.add_function(wrap_pyfunction!(price_digital_analytical_black, m)?)?;

    m.add_function(wrap_pyfunction!(price_european_monte_carlo_bs, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_monte_carlo_heston, m)?)?;
    m.add_function(wrap_pyfunction!(price_digital_monte_carlo_bs, m)?)?;
    m.add_function(wrap_pyfunction!(price_digital_monte_carlo_heston, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_bs, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_bs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_black, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_black_batch, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_heston, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_heston_batch, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_bates, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_bates_batch, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_kou, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_kou_batch, m)?)?;
    m.add_function(wrap_pyfunction!(price_asian_monte_carlo_bs, m)?)?;
    m.add_function(wrap_pyfunction!(price_asian_monte_carlo_heston, m)?)?;
    m.add_function(wrap_pyfunction!(
        price_european_analytical_bs_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        price_european_analytical_black_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        price_european_monte_carlo_bs_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        price_european_monte_carlo_heston_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_heston_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_bates_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(price_european_fft_kou_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(price_asian_monte_carlo_bs_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        price_asian_monte_carlo_heston_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(calibrate_black, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_sabr, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_heston, m)?)?;
    Ok(())
}
