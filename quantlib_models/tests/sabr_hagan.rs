//! Hagan et al. (2002), "Managing Smile Risk": implied-vol approximations.

use proptest::prelude::*;
use quantlib_models::SabrModel;

const R: f64 = 0.01;

fn general(alpha: f64, beta: f64, rho: f64, nu: f64) -> SabrModel {
    SabrModel::general(alpha, beta, rho, nu, R, None).expect("general SABR")
}

fn lognormal(alpha: f64, rho: f64, nu: f64) -> SabrModel {
    SabrModel::lognormal(alpha, rho, nu, R, None).expect("lognormal SABR")
}

fn normal(alpha: f64, rho: f64, nu: f64) -> SabrModel {
    SabrModel::normal(alpha, rho, nu, R, None).expect("normal SABR")
}

fn vol(model: &SabrModel, f: f64, k: f64, t: f64) -> f64 {
    model.implied_volatility(f, k, t).expect("implied vol")
}

/// (alpha, beta, rho, nu, strike, ttm, vol) with F = 100. Vols from QuantLib
/// 1.43 `ql.sabrVolatility(strike, forward, ttm, alpha, beta, nu, rho)`, which
/// implements eq. (2.17a); alpha = 0.25 * F^(1 - beta).
const QUANTLIB_REFERENCE: [(f64, f64, f64, f64, f64, f64, f64); 13] = [
    (
        6.279716078773949,
        0.3,
        -0.6,
        0.2,
        50.0,
        0.25,
        0.3598706152423521,
    ),
    (
        6.279716078773949,
        0.3,
        0.4,
        0.8,
        120.0,
        5.0,
        0.3357795253972751,
    ),
    (
        6.279716078773949,
        0.3,
        -0.6,
        0.8,
        200.0,
        5.0,
        0.24549841389199312,
    ),
    (2.5, 0.5, -0.6, 0.8, 50.0, 5.0, 0.49773035063654564),
    (2.5, 0.5, 0.4, 0.2, 120.0, 0.25, 0.2470565318267584),
    (2.5, 0.5, -0.6, 0.2, 200.0, 0.25, 0.17970111999837143),
    (2.5, 0.5, 0.4, 0.8, 100.0, 5.0, 0.31398046875),
    (
        0.39622329811527834,
        0.9,
        0.4,
        0.8,
        50.0,
        0.25,
        0.3087497455402886,
    ),
    (
        0.39622329811527834,
        0.9,
        -0.6,
        0.2,
        120.0,
        5.0,
        0.23117446211007,
    ),
    (
        0.39622329811527834,
        0.9,
        0.4,
        0.2,
        200.0,
        5.0,
        0.28594846817416525,
    ),
    (0.25, 1.0, -0.6, 0.8, 50.0, 5.0, 0.4217841797077001),
    (0.25, 1.0, 0.4, 0.2, 120.0, 0.25, 0.2584002853456622),
    (0.25, 1.0, -0.6, 0.8, 200.0, 0.25, 0.25794195161404354),
];

#[test]
fn matches_quantlib_reference_values() {
    for (alpha, beta, rho, nu, k, t, expected) in QUANTLIB_REFERENCE {
        let model = if beta == 1.0 {
            lognormal(alpha, rho, nu)
        } else {
            general(alpha, beta, rho, nu)
        };
        let v = vol(&model, 100.0, k, t);
        assert!(
            (v - expected).abs() < 1e-12,
            "beta={beta} rho={rho} nu={nu} K={k} T={t}: {v} vs QuantLib {expected}"
        );
    }
}

#[test]
fn lognormal_smile_is_flat_without_vol_of_vol() {
    // nu -> 0, rho = 0: sigma_B(K) = alpha (1 + nu^2 T / 12) for every K.
    let model = lognormal(0.2, 0.0, 1e-4);
    for k in [50.0, 70.0, 100.0, 140.0, 200.0] {
        let v = vol(&model, 100.0, k, 1.0);
        assert!((v - 0.2).abs() < 1e-6, "K={k}: {v}");
    }
}

#[test]
fn normal_smile_is_flat_without_vol_of_vol() {
    let model = normal(20.0, 0.0, 1e-4);
    for k in [50.0, 70.0, 100.0, 140.0, 200.0] {
        let v = vol(&model, 100.0, k, 1.0);
        assert!((v - 20.0).abs() < 20.0 * 1e-6, "K={k}: {v}");
    }
}

#[test]
fn normal_atm_vol_has_only_the_vol_of_vol_correction() {
    let (alpha, nu, t) = (20.0, 0.4, 2.0);
    for rho in [-0.7, 0.0, 0.5] {
        let v = vol(&normal(alpha, rho, nu), 100.0, 100.0, t);
        let expected = alpha * (1.0 + (2.0 - 3.0 * rho * rho) * nu * nu * t / 24.0);
        assert!(
            (v - expected).abs() < 1e-14 * alpha,
            "rho={rho}: {v} vs {expected}"
        );
    }
}

#[test]
fn general_sabr_is_continuous_at_beta_one() {
    let near_one = general(0.25, 1.0 - 1e-9, -0.4, 0.6);
    let exact = lognormal(0.25, -0.4, 0.6);
    for i in 0..=30 {
        let k = 100.0 * (0.5 + 1.5 * f64::from(i) / 30.0);
        let (a, b) = (vol(&near_one, 100.0, k, 2.0), vol(&exact, 100.0, k, 2.0));
        assert!((a - b).abs() < 1e-7, "K={k}: {a} vs {b}");
    }
}

#[test]
fn smile_is_continuous_across_the_money() {
    let models = [
        general(2.5, 0.5, -0.3, 0.5),
        lognormal(0.25, 0.4, 0.8),
        normal(20.0, -0.5, 0.6),
    ];
    for model in &models {
        let atm = vol(model, 100.0, 100.0, 1.5);
        // The smile has a genuine slope at the money, so compare against the
        // local linear trend: any jump at an ATM switch would show up here.
        let h = 0.1;
        let slope =
            (vol(model, 100.0, 100.0 + h, 1.5) - vol(model, 100.0, 100.0 - h, 1.5)) / (2.0 * h);
        for eps in [1e-12, 1e-9, 1e-6] {
            for k in [100.0 * (1.0 + eps), 100.0 * (1.0 - eps)] {
                let v = vol(model, 100.0, k, 1.5);
                let trend = atm + slope * (k - 100.0);
                assert!(
                    (v - trend).abs() < 1e-8 * atm,
                    "{model:?} K={k}: {v} vs ATM trend {trend}"
                );
            }
        }
    }
}

#[test]
fn rejects_degenerate_correlation() {
    assert!(SabrModel::lognormal(0.2, 1.0, 0.5, R, None).is_err());
    assert!(SabrModel::normal(0.2, -1.0, 0.5, R, None).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// With rho < 0 and beta < 1 the smile slopes down at the money, and the
    /// vol is finite and positive for K/F in [0.2, 5].
    #[test]
    fn negative_rho_gives_downward_skew_and_positive_vols(
        atm_vol in 0.1..0.5f64,
        beta in 0.01..0.99f64,
        rho in -0.95..-0.01f64,
        nu in 0.05..1.0f64,
        t in 0.1..2.0f64,
    ) {
        let f = 100.0;
        // alpha such that the backbone vol alpha F^(beta-1) equals atm_vol.
        let model = general(atm_vol * f64::powf(f, 1.0 - beta), beta, rho, nu);

        let below = vol(&model, f, 0.99 * f, t);
        let above = vol(&model, f, 1.01 * f, t);
        prop_assert!(below > above, "not decreasing at ATM: {} <= {}", below, above);

        for i in 0..=48 {
            let k = f * 0.2 * f64::powf(25.0, f64::from(i) / 48.0);
            let v = vol(&model, f, k, t);
            prop_assert!(v.is_finite() && v > 0.0, "K={}: vol={}", k, v);
        }
    }
}
