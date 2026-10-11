mod common;

use common::{mean_and_stderr, spot_option};
use quantlib_core::{OptionType, PricingEngine};
use quantlib_engines::{AnalyticalEngine, FFTEngine, MonteCarloEngine};
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
    let prices: Vec<f64> = (1..=n_seeds)
        .map(|seed| {
            MonteCarloEngine::new(n_paths, n_steps, Some(seed))
                .unwrap()
                .calculate_price(option, model)
                .unwrap()
        })
        .collect();
    mean_and_stderr(&prices)
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
    // Feller condition holds (2*kappa*theta > sigma^2), so the variance
    // flooring in the Euler scheme rarely triggers and its bias (#6) stays
    // well below the statistical error at this path count.
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
