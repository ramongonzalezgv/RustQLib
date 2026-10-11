//! Heap-allocation counts of the Fourier engines' pricing paths.
//!
//! Lives in its own test binary because it installs a counting global
//! allocator. Counts are per thread, so allocations made concurrently by the
//! test harness or other tests do not leak into a measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::{CosEngine, FFTEngine, InterpolationKind};
use quantlib_models::HestonModel;
use quantlib_products::EuropeanOption;

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `try_with` because the thread-local may already be gone during thread teardown.
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = allocations();
    let value = f();
    (value, allocations() - before)
}

fn heston() -> HestonModel {
    HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0).expect("model")
}

fn option() -> EuropeanOption {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let expiry = SimpleDate::new(2027, 2, 10).expect("date");
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

#[test]
fn fft_price_allocates_a_constant_handful_of_buffers() {
    let (model, option) = (heston(), option());

    for interp in [InterpolationKind::Linear, InterpolationKind::Cubic] {
        let engine = FFTEngine::new(1 << 12, 200.0, interp).expect("engine");
        engine.calculate_price(&option, &model).expect("warm-up");

        // Work buffer, real-part grid and the one-element result: no planner,
        // no per-point or interpolation allocations.
        let (_, single) = count_allocations(|| engine.calculate_price(&option, &model));
        assert!(single <= 3, "{interp:?}: {single} allocations per price");

        // Interpolating 50 strikes costs no more than one.
        let strikes: Vec<f64> = (0..50).map(|i| 60.0 + 2.0 * f64::from(i)).collect();
        let types = vec![OptionType::Call; strikes.len()];
        let (_, strip) =
            count_allocations(|| engine.price_strip(&model, 100.0, 1.0, &strikes, &types));
        assert_eq!(
            strip, single,
            "{interp:?}: strip allocations {strip} vs single {single}"
        );
    }
}

#[test]
fn cos_price_allocates_a_constant_handful_of_buffers() {
    let (model, option) = (heston(), option());
    let engine = CosEngine::default();
    engine.calculate_price(&option, &model).expect("warm-up");

    // Series coefficients and the one-element result.
    let (_, single) = count_allocations(|| engine.calculate_price(&option, &model));
    assert!(single <= 2, "{single} allocations per price");
}
