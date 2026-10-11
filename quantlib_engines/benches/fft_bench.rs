use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{CosEngine, FFTEngine, InterpolationKind};
use quantlib_models::HestonModel;
use quantlib_products::EuropeanOption;

fn baseline_option_and_model() -> (EuropeanOption, HestonModel) {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let option = EuropeanOption::new_spot(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");

    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");
    (option, model)
}

fn deep_otm_option_and_model() -> (EuropeanOption, HestonModel) {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");

    let option = EuropeanOption::new_spot(
        70.0,
        130.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option");

    let model = HestonModel::new(2.2, 0.04, 0.55, -0.65, 0.05, 0.02, 0.005).expect("model");
    (option, model)
}

fn bench_fft_resolution(c: &mut Criterion) {
    let (option, model) = baseline_option_and_model();

    let mut group = c.benchmark_group("fft_heston_resolution");
    group.sample_size(20);

    for &n in &[1_024_usize, 2_048, 4_096] {
        let engine = FFTEngine::new(n, 180.0, InterpolationKind::Linear).expect("engine");
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let price = engine
                    .calculate_price(black_box(&option), black_box(&model))
                    .expect("price");
                black_box(price)
            });
        });
    }

    group.finish();
}

fn bench_fft_bandwidth(c: &mut Criterion) {
    let (option, model) = baseline_option_and_model();

    let mut group = c.benchmark_group("fft_heston_bandwidth");
    group.sample_size(20);

    for &bandwidth in &[160.0_f64, 220.0, 280.0] {
        let engine = FFTEngine::new(4_096, bandwidth, InterpolationKind::Linear).expect("engine");
        group.bench_with_input(
            BenchmarkId::from_parameter(bandwidth),
            &bandwidth,
            |b, _| {
                b.iter(|| {
                    let price = engine
                        .calculate_price(black_box(&option), black_box(&model))
                        .expect("price");
                    black_box(price)
                });
            },
        );
    }

    group.finish();
}

fn bench_fft_interpolation(c: &mut Criterion) {
    let (option, model) = baseline_option_and_model();

    let mut group = c.benchmark_group("fft_heston_interpolation");
    group.sample_size(20);

    for &(name, interp) in &[
        ("linear", InterpolationKind::Linear),
        ("cubic", InterpolationKind::Cubic),
    ] {
        let engine = FFTEngine::new(4_096, 200.0, interp).expect("engine");
        group.bench_function(name, |b| {
            b.iter(|| {
                let price = engine
                    .calculate_price(black_box(&option), black_box(&model))
                    .expect("price");
                black_box(price)
            });
        });
    }

    group.finish();
}

fn bench_fft_deep_otm_stress(c: &mut Criterion) {
    let (option, model) = deep_otm_option_and_model();
    let engine = FFTEngine::new(4_096, 260.0, InterpolationKind::Cubic).expect("engine");

    c.bench_function("fft_heston_deep_otm_stress", |b| {
        b.iter(|| {
            let price = engine
                .calculate_price(black_box(&option), black_box(&model))
                .expect("price");
            black_box(price)
        });
    });
}

fn bench_cos_vs_fft_single(c: &mut Criterion) {
    let (option, model) = baseline_option_and_model();
    let fft = FFTEngine::default();
    let cos = CosEngine::default();

    let mut group = c.benchmark_group("heston_single_price");
    group.bench_function("fft_4096", |b| {
        b.iter(|| black_box(fft.calculate_price(black_box(&option), black_box(&model))))
    });
    group.bench_function("cos_256", |b| {
        b.iter(|| black_box(cos.calculate_price(black_box(&option), black_box(&model))))
    });
    group.finish();
}

fn bench_cos_vs_fft_strip(c: &mut Criterion) {
    let (_, model) = baseline_option_and_model();
    let fft = FFTEngine::default();
    let cos = CosEngine::default();
    let strikes: Vec<f64> = (0..50).map(|i| 60.0 + 2.0 * f64::from(i)).collect();
    let types = vec![OptionType::Call; strikes.len()];

    let mut group = c.benchmark_group("heston_strip_50");
    group.bench_function("fft_4096", |b| {
        b.iter(|| black_box(fft.price_strip(&model, 100.0, 1.0, black_box(&strikes), &types)))
    });
    group.bench_function("cos_256", |b| {
        b.iter(|| black_box(cos.price_strip(&model, 100.0, 1.0, black_box(&strikes), &types)))
    });
    group.finish();
}

criterion_group!(
    fft_benches,
    bench_fft_resolution,
    bench_fft_bandwidth,
    bench_fft_interpolation,
    bench_fft_deep_otm_stress,
    bench_cos_vs_fft_single,
    bench_cos_vs_fft_strip
);
criterion_main!(fft_benches);
