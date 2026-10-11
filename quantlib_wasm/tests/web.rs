//! Runs in a real WASM runtime (`wasm-pack test --node quantlib_wasm`) and
//! checks that WASM prices equal the native ones computed by the same code.
#![cfg(target_arch = "wasm32")]

use quantlib_wasm::api::{self, EngineKind, PricingRequest};
use wasm_bindgen_test::wasm_bindgen_test;

fn request() -> PricingRequest {
    serde_json::from_str(
        r#"{
            "product": {
                "kind": "european", "optionType": "call", "strike": 100,
                "valuationDate": "2026-01-01", "expiryDate": "2027-01-01"
            },
            "model": { "kind": "blackScholes", "sigma": 0.2 },
            "market": { "underlying": 100, "r": 0.05, "q": 0 },
            "settings": {
                "binomialSteps": 500, "mcPaths": 8000, "mcSteps": 1,
                "mcSeed": 42, "fftN": 4096
            }
        }"#,
    )
    .unwrap()
}

#[wasm_bindgen_test]
fn analytical_matches_native() {
    let p = api::price(&request(), EngineKind::Analytical)
        .unwrap()
        .price;
    assert!((p - 10.450_583_572_185_565).abs() < 1e-12, "wasm={p}");
}

#[wasm_bindgen_test]
fn engines_match_native() {
    // Native values from the same request on aarch64-apple-darwin. Native
    // `exp`/`ln` come from the system libm and WASM uses Rust's own, which can
    // differ in the last ulp, so compare to 1e-12 relative, not bit-for-bit.
    let req = request();
    for (engine, expected) in NATIVE_REFERENCE {
        let p = api::price(&req, engine).unwrap().price;
        assert!(
            ((p - expected) / expected).abs() < 1e-12,
            "{engine:?}: wasm={p}, native={expected}"
        );
    }
}

const NATIVE_REFERENCE: [(EngineKind, f64); 3] = [
    (EngineKind::Fft, 10.450_583_572_186_233),
    (EngineKind::Binomial, 10.446_585_136_446_54),
    (EngineKind::MonteCarlo, 10.437_035_752_423_261),
];
