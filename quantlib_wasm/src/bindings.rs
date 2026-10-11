//! `wasm-bindgen` exports. Requests and responses cross the boundary as plain
//! JS objects whose shapes are defined by the serde types in [`crate::api`].

use serde::Serialize;
use serde::de::DeserializeOwned;
use wasm_bindgen::prelude::*;

use crate::api::{self, EngineKind, ModelKind, PricingRequest};

fn from_js<T: DeserializeOwned>(value: JsValue) -> Result<T, JsError> {
    serde_wasm_bindgen::from_value(value).map_err(|e| JsError::new(&e.to_string()))
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsError> {
    // JSON-compatible: `None` becomes `null` and maps become plain objects.
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsError::new(&e.to_string()))
}

fn api_err(message: String) -> JsError {
    JsError::new(&message)
}

/// Supported engines for every (product, model) pair.
#[wasm_bindgen]
pub fn capabilities() -> Result<JsValue, JsError> {
    to_js(&api::capabilities())
}

/// Price with one engine: `{ price, stderr }`.
#[wasm_bindgen]
pub fn price(request: JsValue, engine: JsValue) -> Result<JsValue, JsError> {
    let request: PricingRequest = from_js(request)?;
    let engine: EngineKind = from_js(engine)?;
    to_js(&api::price(&request, engine).map_err(api_err)?)
}

/// Time to maturity in years.
#[wasm_bindgen]
pub fn ttm(request: JsValue) -> Result<f64, JsError> {
    let request: PricingRequest = from_js(request)?;
    api::ttm(&request).map_err(api_err)
}

/// Today's value at each underlying level (`null` where pricing fails).
#[wasm_bindgen(js_name = priceCurve)]
pub fn price_curve(
    request: JsValue,
    engine: JsValue,
    underlyings: Vec<f64>,
) -> Result<JsValue, JsError> {
    let request: PricingRequest = from_js(request)?;
    let engine: EngineKind = from_js(engine)?;
    to_js(&api::price_curve(&request, engine, &underlyings))
}

/// Payoff at expiry at each underlying level.
#[wasm_bindgen(js_name = payoffCurve)]
pub fn payoff_curve(request: JsValue, underlyings: Vec<f64>) -> Result<Vec<f64>, JsError> {
    let request: PricingRequest = from_js(request)?;
    api::payoff_curve(&request, &underlyings).map_err(api_err)
}

/// Engine used for the smile of `model` (`null` if none).
#[wasm_bindgen(js_name = smileEngine)]
pub fn smile_engine(model: JsValue) -> Result<JsValue, JsError> {
    let model: ModelKind = from_js(model)?;
    to_js(&api::smile_engine(model))
}

/// Black implied volatility at each strike (`null` where it does not exist).
#[wasm_bindgen]
pub fn smile(request: JsValue, strikes: Vec<f64>) -> Result<JsValue, JsError> {
    let request: PricingRequest = from_js(request)?;
    to_js(&api::smile(&request, &strikes).map_err(api_err)?)
}

/// Sample paths, quantile bands and terminal distribution.
#[wasm_bindgen(js_name = mcPaths)]
pub fn mc_paths(request: JsValue, n_display: usize) -> Result<JsValue, JsError> {
    let request: PricingRequest = from_js(request)?;
    to_js(&api::mc_paths(&request, n_display).map_err(api_err)?)
}
