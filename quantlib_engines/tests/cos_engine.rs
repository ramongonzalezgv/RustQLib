use quantlib_core::{OptionType, PricingEngine, SimpleDate, norm_cdf};
use quantlib_engines::{CosEngine, FFTEngine, InterpolationKind};
use quantlib_models::{BatesModel, BlackModel, BlackScholesModel, HestonModel, KouModel};
use quantlib_products::EuropeanOption;

fn black_scholes_call(spot: f64, strike: f64, ttm: f64, sigma: f64, r: f64, q: f64) -> f64 {
    let sd = sigma * ttm.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * sigma * sigma) * ttm) / sd;
    spot * (-q * ttm).exp() * norm_cdf(d1) - strike * (-r * ttm).exp() * norm_cdf(d1 - sd)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn cos_heston_matches_fang_oosterlee_reference() {
    // Fang & Oosterlee (2008), Heston test case.
    let model = HestonModel::new(1.5768, 0.0398, 0.5751, -0.5711, 0.0175, 0.0, 0.0).expect("model");
    let price = CosEngine::default()
        .price_strip(&model, 100.0, 1.0, &[100.0], &[OptionType::Call])
        .expect("cos price")[0];

    // The reference itself sits ~1.6e-8 above the value both COS (large L)
    // and a 32k-point FFT converge to. N = 160 is not enough here: the
    // fat left tail needs the c4-widened interval, which in turn needs N ≈ 256.
    let err = (price - 5.785155450).abs();
    assert!(err < 1.0e-7, "cos={price}, err={err:e}");
}

#[test]
fn cos_black_scholes_matches_analytic_with_128_terms() {
    let (spot, sigma, r, q) = (100.0, 0.2, 0.03, 0.01);
    let model = BlackScholesModel::new(sigma, r, q).expect("model");
    let engine = CosEngine::new(128, 12.0).expect("engine");
    let strikes: Vec<f64> = (0..=30).map(|i| 50.0 + 5.0 * f64::from(i)).collect();
    let types = vec![OptionType::Call; strikes.len()];

    for ttm in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0] {
        let prices = engine
            .price_strip(&model, spot, ttm, &strikes, &types)
            .expect("cos strip");
        for (&strike, &price) in strikes.iter().zip(&prices) {
            let expected = black_scholes_call(spot, strike, ttm, sigma, r, q);
            let err = (price - expected).abs();
            assert!(
                err < 1.0e-10,
                "T={ttm}, K={strike}: cos={price}, analytic={expected}, err={err:e}"
            );
        }
    }
}

#[test]
fn cos_black_prices_forward_quoted_options() {
    let (forward, sigma, r) = (102.0, 0.25, 0.03);
    let model = BlackModel::new(sigma, r).expect("model");
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");
    let option = EuropeanOption::new_forward(
        forward,
        95.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");

    let price = CosEngine::default()
        .calculate_price(&option, &model)
        .expect("cos price");
    let expected = black_scholes_call(forward, 95.0, option.ttm(), sigma, r, r);
    assert!(
        (price - expected).abs() < 1.0e-10,
        "cos={price}, analytic={expected}"
    );
}

#[test]
fn cos_matches_fft_for_heston_bates_and_kou() {
    // A fine FFT grid is the reference: the default 4096-point grid is
    // interpolation-limited at short maturities (see fft_black_and_bs_engine.rs).
    let fft = FFTEngine::new(1 << 15, 1200.0, InterpolationKind::Cubic).expect("fft");
    let cos = CosEngine::default();

    let heston = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("heston");
    let bates =
        BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01, 0.3, -0.1, 0.2).expect("bates");
    let kou = KouModel::new(0.2, 0.03, 0.01, 0.4, 0.35, 12.0, 7.0).expect("kou");

    let strikes: Vec<f64> = (0..10).map(|i| 60.0 + 10.0 * f64::from(i)).collect();
    let types: Vec<OptionType> = (0..10)
        .map(|i| {
            if i < 5 {
                OptionType::Put
            } else {
                OptionType::Call
            }
        })
        .collect();

    for ttm in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] {
        let diffs = [
            max_abs_diff(
                &cos.price_strip(&heston, 100.0, ttm, &strikes, &types)
                    .expect("cos"),
                &fft.price_strip(&heston, 100.0, ttm, &strikes, &types)
                    .expect("fft"),
            ),
            max_abs_diff(
                &cos.price_strip(&bates, 100.0, ttm, &strikes, &types)
                    .expect("cos"),
                &fft.price_strip(&bates, 100.0, ttm, &strikes, &types)
                    .expect("fft"),
            ),
            max_abs_diff(
                &cos.price_strip(&kou, 100.0, ttm, &strikes, &types)
                    .expect("cos"),
                &fft.price_strip(&kou, 100.0, ttm, &strikes, &types)
                    .expect("fft"),
            ),
        ];
        for (name, diff) in ["heston", "bates", "kou"].iter().zip(diffs) {
            assert!(diff < 1.0e-6, "{name}, T={ttm}: max |cos - fft| = {diff:e}");
        }
    }
}

#[test]
fn cos_short_maturity_and_deep_otm_are_finite_non_negative_and_satisfy_parity() {
    let cos = CosEngine::default();
    let heston = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("heston");
    let bates =
        BatesModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01, 0.3, -0.1, 0.2).expect("bates");
    let kou = KouModel::new(0.2, 0.03, 0.01, 0.4, 0.35, 12.0, 7.0).expect("kou");

    let spot = 100.0;
    let strikes = [50.0, 80.0, 100.0, 125.0, 200.0];
    let calls = [OptionType::Call; 5];
    let puts = [OptionType::Put; 5];

    let check = |name: &str, ttm: f64, call: Vec<f64>, put: Vec<f64>, r: f64, q: f64| {
        for ((&strike, &c), &p) in strikes.iter().zip(&call).zip(&put) {
            assert!(
                c.is_finite() && c >= 0.0,
                "{name} T={ttm} K={strike}: call={c}"
            );
            assert!(
                p.is_finite() && p >= 0.0,
                "{name} T={ttm} K={strike}: put={p}"
            );
            let parity = c - p - (spot * (-q * ttm).exp() - strike * (-r * ttm).exp());
            assert!(
                parity.abs() < 1.0e-10,
                "{name} T={ttm} K={strike}: parity residual {parity:e}"
            );
        }
    };

    for ttm in [1.0 / 365.0, 1.0] {
        check(
            "heston",
            ttm,
            cos.price_strip(&heston, spot, ttm, &strikes, &calls)
                .expect("call"),
            cos.price_strip(&heston, spot, ttm, &strikes, &puts)
                .expect("put"),
            heston.r,
            heston.q,
        );
        check(
            "bates",
            ttm,
            cos.price_strip(&bates, spot, ttm, &strikes, &calls)
                .expect("call"),
            cos.price_strip(&bates, spot, ttm, &strikes, &puts)
                .expect("put"),
            bates.r,
            bates.q,
        );
        check(
            "kou",
            ttm,
            cos.price_strip(&kou, spot, ttm, &strikes, &calls)
                .expect("call"),
            cos.price_strip(&kou, spot, ttm, &strikes, &puts)
                .expect("put"),
            kou.r,
            kou.q,
        );
    }
}

#[test]
fn cos_single_option_matches_strip_and_scales_with_quantity() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2026, 8, 10).expect("date");
    let option = EuropeanOption::new_spot(
        100.0,
        105.0,
        expiry,
        Some(valuation),
        OptionType::Put,
        4,
        None,
    )
    .expect("option");
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    let engine = CosEngine::default();

    let single = engine.calculate_price(&option, &model).expect("price");
    let strip = engine
        .price_strip(&model, 100.0, option.ttm(), &[105.0], &[OptionType::Put])
        .expect("strip");
    assert!((single - 4.0 * strip[0]).abs() < 1.0e-12);
}

#[test]
fn cos_rejects_invalid_settings_and_inputs() {
    assert!(CosEngine::new(1, 12.0).is_err());
    assert!(CosEngine::new(128, 0.0).is_err());
    assert!(CosEngine::new(128, f64::NAN).is_err());

    let model = BlackScholesModel::new(0.2, 0.03, 0.0).expect("model");
    let engine = CosEngine::default();
    assert!(
        engine
            .price_strip(&model, 100.0, 1.0, &[90.0, 100.0], &[OptionType::Call])
            .is_err()
    );
    assert!(
        engine
            .price_strip(&model, 100.0, 0.0, &[90.0], &[OptionType::Call])
            .is_err()
    );
    assert!(
        engine
            .price_strip(&model, 100.0, 1.0, &[-5.0], &[OptionType::Call])
            .is_err()
    );
}
