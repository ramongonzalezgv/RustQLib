use criterion::{Criterion, black_box, criterion_group, criterion_main};
use quantlib_core::OptionType;
use quantlib_models::HestonModel;
use quantlib_utils::{
    CalibrationConfig, CalibrationMethod, HestonFftConfig, NelderMeadConfig, ObjectiveKind,
    VanillaOptionCalibrationQuote, calibrate_heston_model,
};

/// 5 maturities × 20 strikes of synthetic call quotes.
fn surface() -> Vec<VanillaOptionCalibrationQuote> {
    let mut quotes = Vec::new();
    for maturity in [0.1, 0.25, 0.5, 1.0, 2.0] {
        for i in 0..20 {
            let strike = 70.0 + 3.0 * f64::from(i);
            quotes.push(VanillaOptionCalibrationQuote::price(
                100.0,
                strike,
                maturity,
                OptionType::Call,
                5.0,
                1.0,
            ));
        }
    }
    quotes
}

/// With every parameter pinned and one Nelder–Mead iteration, the calibration
/// performs a fixed number of objective evaluations, so this tracks the cost of
/// one objective evaluation over the surface.
fn bench_heston_objective(c: &mut Criterion) {
    let quotes = surface();
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    let params = [
        model.kappa,
        model.theta,
        model.sigma,
        model.rho,
        model.v0,
        model.r,
        model.q,
    ];
    let bounds = params.map(|p| (p, p));
    let config = CalibrationConfig {
        method: CalibrationMethod::NelderMead,
        objective: ObjectiveKind::LeastSquares,
        nelder_mead: NelderMeadConfig {
            max_iterations: 1,
            ..NelderMeadConfig::default()
        },
        ..CalibrationConfig::default()
    };

    c.bench_function("heston_objective_5x20_surface", |b| {
        b.iter(|| {
            let result = calibrate_heston_model(
                black_box(&quotes),
                model,
                Some(bounds),
                &config,
                HestonFftConfig::default(),
            )
            .expect("calibration");
            black_box(result)
        });
    });
}

criterion_group!(calibration_benches, bench_heston_objective);
criterion_main!(calibration_benches);
