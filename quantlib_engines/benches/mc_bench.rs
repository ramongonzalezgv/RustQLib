use criterion::{Criterion, black_box, criterion_group, criterion_main};
use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::MonteCarloEngine;
use quantlib_models::{BlackScholesModel, HestonModel};
use quantlib_products::{AsianOption, AveragingType, EuropeanOption};

fn dates() -> (SimpleDate, SimpleDate) {
    (
        SimpleDate::new(2026, 2, 10).expect("date"),
        SimpleDate::new(2027, 2, 10).expect("date"),
    )
}

fn european() -> EuropeanOption {
    let (valuation, expiry) = dates();
    EuropeanOption::new_spot(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        None,
    )
    .expect("option")
}

/// Single-thread throughput of the exact one-step BS terminal path.
fn bench_bs_terminal_throughput(c: &mut Criterion) {
    let option = european();
    let model = BlackScholesModel::new(0.2, 0.03, 0.01).expect("model");
    let engine = MonteCarloEngine::new(10_000_000, 1, Some(7))
        .expect("engine")
        .with_parallel(false);

    let mut group = c.benchmark_group("mc_bs_terminal_1e7");
    group.sample_size(10);
    group.bench_function("serial", |b| {
        b.iter(|| black_box(engine.calculate_price(black_box(&option), &model)))
    });
    group.finish();
}

/// Serial vs parallel Heston European, 200k paths × 252 steps.
fn bench_heston_scaling(c: &mut Criterion) {
    let option = european();
    let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model");

    let mut group = c.benchmark_group("mc_heston_200k_252");
    group.sample_size(10);
    for parallel in [false, true] {
        let engine = MonteCarloEngine::new(200_000, 252, Some(7))
            .expect("engine")
            .with_parallel(parallel);
        let name = if parallel { "parallel" } else { "serial" };
        group.bench_function(name, |b| {
            b.iter(|| black_box(engine.calculate_price(black_box(&option), &model)))
        });
    }
    group.finish();
}

/// Arithmetic Asian under BS, 100k paths × 252 steps, parallel.
fn bench_asian(c: &mut Criterion) {
    let (valuation, expiry) = dates();
    let option = AsianOption::new(
        100.0,
        100.0,
        expiry,
        Some(valuation),
        OptionType::Call,
        1,
        AveragingType::Arithmetic,
        None,
    )
    .expect("option");
    let model = BlackScholesModel::new(0.2, 0.03, 0.01).expect("model");
    let engine = MonteCarloEngine::new(100_000, 252, Some(7)).expect("engine");

    let mut group = c.benchmark_group("mc_asian_bs_100k_252");
    group.sample_size(10);
    group.bench_function("parallel", |b| {
        b.iter(|| black_box(engine.calculate_price(black_box(&option), &model)))
    });
    group.finish();
}

criterion_group!(
    mc_benches,
    bench_bs_terminal_throughput,
    bench_heston_scaling,
    bench_asian
);
criterion_main!(mc_benches);
