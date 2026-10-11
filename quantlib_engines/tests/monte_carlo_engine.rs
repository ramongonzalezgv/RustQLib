mod common;

use common::{mean_and_stderr, spot_option};
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::{AnalyticalEngine, CosEngine, FFTEngine, HestonScheme, MonteCarloEngine};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::EuropeanOption;

/// Prices `option` with `n_seeds` independent seeds and returns the mean and
/// its standard error. The engine does not expose a per-run standard error
/// yet (#11), so it is estimated from the spread across seeds.
fn mc_estimate<M>(
    option: &EuropeanOption,
    model: &M,
    n_paths: usize,
    n_steps: usize,
    n_seeds: u64,
) -> (f64, f64)
where
    MonteCarloEngine: PricingEngine<EuropeanOption, M>,
{
    mc_estimate_with(
        option,
        model,
        n_paths,
        n_steps,
        n_seeds,
        HestonScheme::default(),
    )
}

fn mc_estimate_with<M>(
    option: &EuropeanOption,
    model: &M,
    n_paths: usize,
    n_steps: usize,
    n_seeds: u64,
    scheme: HestonScheme,
) -> (f64, f64)
where
    MonteCarloEngine: PricingEngine<EuropeanOption, M>,
{
    let prices: Vec<f64> = (1..=n_seeds)
        .map(|seed| {
            MonteCarloEngine::new(n_paths, n_steps, Some(seed))
                .unwrap()
                .with_heston_scheme(scheme)
                .calculate_price(option, model)
                .unwrap()
        })
        .collect();
    mean_and_stderr(&prices)
}

/// Lord et al. / Andersen test case: 2κθ = 0.12 < σ² = 1, so the variance
/// spends a lot of time near zero and naive Euler schemes are badly biased.
fn feller_violating_heston() -> HestonModel {
    HestonModel::new(1.5, 0.04, 1.0, -0.9, 0.04, 0.02, 0.0).expect("model")
}

fn fourier_reference(option: &EuropeanOption, model: &HestonModel) -> f64 {
    CosEngine::new(1024, 16.0)
        .unwrap()
        .calculate_price(option, model)
        .unwrap()
}

#[test]
fn black_scholes_mc_within_three_stderr_of_analytic() {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    for (k, option_type) in [
        (90.0, OptionType::Call),
        (100.0, OptionType::Call),
        (110.0, OptionType::Put),
    ] {
        let option = spot_option(100.0, k, 365, option_type);
        let analytic = AnalyticalEngine.calculate_price(&option, &model).unwrap();
        let (mc, stderr) = mc_estimate(&option, &model, 20_000, 1, 20);
        assert!(
            (mc - analytic).abs() < 3.0 * stderr,
            "K={k} {option_type:?}: mc={mc}, analytic={analytic}, stderr={stderr}"
        );
    }
}

#[test]
fn heston_mc_within_three_stderr_of_fourier() {
    let model = HestonModel::new(2.0, 0.04, 0.3, -0.7, 0.04, 0.03, 0.0).expect("model");
    let fft = FFTEngine::default();
    for (k, option_type) in [(100.0, OptionType::Call), (90.0, OptionType::Put)] {
        let option = spot_option(100.0, k, 365, option_type);
        let reference = fft.calculate_price(&option, &model).unwrap();
        let (mc, stderr) = mc_estimate(&option, &model, 10_000, 200, 16);
        assert!(
            (mc - reference).abs() < 3.0 * stderr,
            "K={k} {option_type:?}: mc={mc}, fourier={reference}, stderr={stderr}"
        );
    }
}

#[test]
fn same_seed_gives_identical_price() {
    let option = spot_option(100.0, 100.0, 365, OptionType::Call);
    let bs = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let heston = HestonModel::new(2.0, 0.04, 0.3, -0.7, 0.04, 0.03, 0.0).expect("model");
    let engine = MonteCarloEngine::new(5_000, 50, Some(7)).unwrap();

    let bs_a: f64 = engine.calculate_price(&option, &bs).unwrap();
    let bs_b: f64 = engine.calculate_price(&option, &bs).unwrap();
    let heston_a: f64 = engine.calculate_price(&option, &heston).unwrap();
    let heston_b: f64 = engine.calculate_price(&option, &heston).unwrap();
    assert_eq!(bs_a.to_bits(), bs_b.to_bits());
    assert_eq!(heston_a.to_bits(), heston_b.to_bits());

    let other_seed: f64 = MonteCarloEngine::new(5_000, 50, Some(8))
        .unwrap()
        .calculate_price(&option, &bs)
        .unwrap();
    assert_ne!(bs_a, other_seed);
}

#[test]
fn sampled_paths_have_expected_shape_and_terminal_mean() {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let engine = MonteCarloEngine::new(20_000, 12, Some(3)).unwrap();
    let paths = engine
        .sample_paths_black_scholes(100.0, &model, 1.0)
        .unwrap();
    assert_eq!(paths.len(), 20_000);
    assert!(paths.iter().all(|p| p.len() == 13 && p[0] == 100.0));

    // E[S_T] = S e^{(r-q)T}; check within 4 standard errors.
    let terminal: Vec<f64> = paths.iter().map(|p| p[12]).collect();
    let n = terminal.len() as f64;
    let mean = terminal.iter().sum::<f64>() / n;
    let sd = (terminal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let expected = 100.0 * (0.04f64).exp();
    assert!((mean - expected).abs() < 4.0 * sd / n.sqrt(), "mean={mean}");

    let heston = HestonModel::new(2.0, 0.04, 0.3, -0.7, 0.04, 0.03, 0.0).expect("model");
    let paths = MonteCarloEngine::new(100, 50, Some(3))
        .unwrap()
        .sample_paths_heston(100.0, &heston, 0.5)
        .unwrap();
    assert!(
        paths
            .iter()
            .all(|p| p.len() == 51 && p.iter().all(|s| *s > 0.0))
    );
}

#[test]
fn heston_qe_is_unbiased_with_coarse_steps_when_feller_fails() {
    let model = feller_violating_heston();
    let option = spot_option(100.0, 100.0, 365, OptionType::Call);
    let reference = fourier_reference(&option, &model);

    // 500k paths, 32 steps per year.
    let (mc, stderr) = mc_estimate_with(
        &option,
        &model,
        25_000,
        32,
        20,
        HestonScheme::QuadraticExponential,
    );
    assert!(
        (mc - reference).abs() < 3.0 * stderr,
        "qe={mc}, fourier={reference}, stderr={stderr}"
    );
}

#[test]
fn heston_full_truncation_bias_shrinks_with_steps_when_feller_fails() {
    let model = feller_violating_heston();
    let option = spot_option(100.0, 100.0, 365, OptionType::Call);
    let reference = fourier_reference(&option, &model);

    // 200k paths per step count. Bias is +0.74 / +0.11 / +0.01 against a
    // standard error of ~0.016, so the ordering is unambiguous.
    let biases: Vec<(usize, f64, f64)> = [12, 52, 252]
        .into_iter()
        .map(|steps| {
            let (mc, stderr) = mc_estimate_with(
                &option,
                &model,
                20_000,
                steps,
                10,
                HestonScheme::FullTruncation,
            );
            (steps, mc - reference, stderr)
        })
        .collect();

    for pair in biases.windows(2) {
        assert!(
            pair[1].1.abs() < pair[0].1.abs(),
            "bias must shrink with steps: {biases:?}"
        );
    }
    let (_, bias, stderr) = biases[2];
    assert!(
        bias.abs() < 3.0 * stderr,
        "252 steps: bias={bias}, stderr={stderr}, all={biases:?}"
    );
}

#[test]
fn heston_qe_terminal_spot_is_a_martingale() {
    // A call struck near zero pays S_T - K, so its price checks
    // E[S_T]·e^{-rT} = S_0·e^{-qT} - K·e^{-rT} under the QE scheme.
    let model = feller_violating_heston();
    let strike = 1.0e-6;
    let option = spot_option(100.0, strike, 365, OptionType::Call);
    let ttm = option.ttm();
    let expected = 100.0 * (-model.q * ttm).exp() - strike * (-model.r * ttm).exp();

    let (mc, stderr) = mc_estimate_with(
        &option,
        &model,
        25_000,
        32,
        20,
        HestonScheme::QuadraticExponential,
    );
    assert!(
        (mc - expected).abs() < 3.0 * stderr,
        "forward={mc}, expected={expected}, stderr={stderr}"
    );
}
