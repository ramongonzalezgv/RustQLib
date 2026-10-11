mod common;

use common::spot_option;
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::{
    AnalyticalEngine, CosEngine, FFTEngine, HestonScheme, MonteCarloEngine, MonteCarloPricing,
    geometric_asian_black_scholes,
};
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::{AsianOption, AveragingType, EuropeanOption};

/// Price and standard error from one engine run over `n_paths * n_seeds`
/// paths (the seed count only scales the path count).
fn mc_estimate<M>(
    option: &EuropeanOption,
    model: &M,
    n_paths: usize,
    n_steps: usize,
    n_seeds: u64,
) -> (f64, f64)
where
    MonteCarloEngine: MonteCarloPricing<EuropeanOption, M>,
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
    MonteCarloEngine: MonteCarloPricing<EuropeanOption, M>,
{
    let estimate = MonteCarloEngine::new(n_paths * n_seeds as usize, n_steps, Some(1))
        .unwrap()
        .with_heston_scheme(scheme)
        .price_with_stats(option, model)
        .unwrap();
    (estimate.price, estimate.std_error)
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

fn asian_option(option_type: OptionType, averaging: AveragingType) -> AsianOption {
    let valuation = common::valuation();
    AsianOption::new(
        100.0,
        100.0,
        valuation.add_days(365),
        Some(valuation),
        option_type,
        1,
        averaging,
        None,
    )
    .expect("asian option")
}

#[test]
fn seeded_prices_are_bit_identical_for_any_thread_count() {
    let heston = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    let option = asian_option(OptionType::Call, AveragingType::Arithmetic);
    let engine = MonteCarloEngine::new(20_000, 52, Some(11))
        .unwrap()
        .with_chunk_size(1_000)
        .with_antithetic(true);

    let serial = engine
        .with_parallel(false)
        .price_with_stats(&option, &heston)
        .unwrap();
    for threads in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let parallel = pool.install(|| engine.price_with_stats(&option, &heston).unwrap());
        assert_eq!(parallel, serial, "{threads} threads");
    }
}

#[test]
fn standard_error_has_nominal_coverage() {
    // 200 independent runs of 20k paths: |MC − BS| < 1.96·stderr should hold
    // about 95% of the time.
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let option = spot_option(100.0, 100.0, 365, OptionType::Call);
    let analytic = AnalyticalEngine.calculate_price(&option, &model).unwrap();

    let covered = (1..=200u64)
        .filter(|&seed| {
            let estimate = MonteCarloEngine::new(20_000, 1, Some(seed))
                .unwrap()
                .price_with_stats(&option, &model)
                .unwrap();
            (estimate.price - analytic).abs() < 1.96 * estimate.std_error
        })
        .count();
    let fraction = covered as f64 / 200.0;
    assert!((0.90..=0.99).contains(&fraction), "coverage={fraction}");
}

#[test]
fn antithetic_variates_cut_the_standard_error() {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let option = spot_option(100.0, 100.0, 365, OptionType::Call);
    let analytic = AnalyticalEngine.calculate_price(&option, &model).unwrap();
    let engine = MonteCarloEngine::new(200_000, 1, Some(5)).unwrap();

    let plain = engine.price_with_stats(&option, &model).unwrap();
    let antithetic = engine
        .with_antithetic(true)
        .price_with_stats(&option, &model)
        .unwrap();
    assert_eq!(antithetic.n_paths, plain.n_paths);
    // Pairing Z with −Z scales the standard error by √(1 + ρ), where
    // ρ = corr(f(Z), f(−Z)) = −0.473 for this ATM call (by quadrature), so the
    // best possible cut is 27.4%; the estimate must match that ratio.
    let ratio = antithetic.std_error / plain.std_error;
    assert!(
        (ratio - 0.726).abs() < 0.03,
        "ratio={ratio}, plain={plain:?}, antithetic={antithetic:?}"
    );
    assert!((antithetic.price - analytic).abs() < 3.0 * antithetic.std_error);
}

#[test]
fn geometric_asian_closed_form_matches_monte_carlo() {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    for n_steps in [12, 52, 252] {
        for option_type in [OptionType::Call, OptionType::Put] {
            let option = asian_option(option_type, AveragingType::Geometric);
            let closed = geometric_asian_black_scholes(
                100.0,
                100.0,
                option.ttm(),
                &model,
                option_type,
                Some(n_steps),
            );
            let mc = MonteCarloEngine::new(1_000_000, n_steps, Some(21))
                .unwrap()
                .price_with_stats(&option, &model)
                .unwrap();
            assert!(
                (mc.price - closed).abs() < 3.0 * mc.std_error,
                "n={n_steps} {option_type:?}: closed={closed}, mc={mc:?}"
            );
        }
    }
}

#[test]
fn analytical_engine_prices_continuous_geometric_asians_only() {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let geometric = asian_option(OptionType::Call, AveragingType::Geometric);
    let price = AnalyticalEngine
        .calculate_price(&geometric, &model)
        .unwrap();
    let continuous = geometric_asian_black_scholes(
        100.0,
        100.0,
        geometric.ttm(),
        &model,
        OptionType::Call,
        None,
    );
    assert_eq!(price, continuous);
    // The discrete price converges to the continuous one as fixings grow.
    let fine = geometric_asian_black_scholes(
        100.0,
        100.0,
        geometric.ttm(),
        &model,
        OptionType::Call,
        Some(100_000),
    );
    assert!(
        (fine - continuous).abs() < 1.0e-4,
        "fine={fine}, continuous={continuous}"
    );

    let arithmetic = asian_option(OptionType::Call, AveragingType::Arithmetic);
    assert!(
        AnalyticalEngine
            .calculate_price(&arithmetic, &model)
            .is_err()
    );
}

#[test]
fn geometric_control_variate_cuts_arithmetic_asian_error_tenfold() {
    // ATM arithmetic Asian, sigma = 0.2, T = 1, 252 fixings.
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let option = asian_option(OptionType::Call, AveragingType::Arithmetic);
    let engine = MonteCarloEngine::new(50_000, 252, Some(8)).unwrap();

    let plain = engine.price_with_stats(&option, &model).unwrap();
    let controlled = engine
        .with_control_variate(true)
        .price_with_stats(&option, &model)
        .unwrap();
    assert!(
        controlled.std_error * 10.0 < plain.std_error,
        "plain={plain:?}, controlled={controlled:?}"
    );
    // Both estimate the same price.
    let combined = (plain.std_error.powi(2) + controlled.std_error.powi(2)).sqrt();
    assert!((plain.price - controlled.price).abs() < 3.0 * combined);
}

/// Reference check from #11: the control-variate estimate against a 1e7-path
/// plain Monte Carlo price. Slow (2.5e9 steps); run with
/// `cargo test --release -- --ignored control_variate_matches_large_plain_reference`.
#[test]
#[ignore]
fn control_variate_matches_large_plain_reference() {
    let model = BlackScholesModel::new(0.2, 0.05, 0.01).expect("model");
    let option = asian_option(OptionType::Call, AveragingType::Arithmetic);
    let reference = MonteCarloEngine::new(10_000_000, 252, Some(1))
        .unwrap()
        .price_with_stats(&option, &model)
        .unwrap();
    let controlled = MonteCarloEngine::new(100_000, 252, Some(2))
        .unwrap()
        .with_control_variate(true)
        .price_with_stats(&option, &model)
        .unwrap();
    let combined = (reference.std_error.powi(2) + controlled.std_error.powi(2)).sqrt();
    assert!(
        (reference.price - controlled.price).abs() < 3.0 * combined,
        "reference={reference:?}, controlled={controlled:?}"
    );
}

#[test]
fn european_control_variate_reduces_heston_error() {
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    let option = spot_option(100.0, 90.0, 365, OptionType::Call);
    let reference = FFTEngine::default()
        .calculate_price(&option, &model)
        .unwrap();
    let engine = MonteCarloEngine::new(100_000, 32, Some(4)).unwrap();

    let plain = engine.price_with_stats(&option, &model).unwrap();
    let controlled = engine
        .with_control_variate(true)
        .price_with_stats(&option, &model)
        .unwrap();
    assert!(
        controlled.std_error < 0.5 * plain.std_error,
        "{plain:?} vs {controlled:?}"
    );
    assert!((controlled.price - reference).abs() < 3.0 * controlled.std_error);
}
