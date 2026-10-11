//! Peak heap use of Monte Carlo pricing: payoffs are streamed, so memory must
//! not grow with the number of paths. Own test binary because it installs a
//! tracking global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use quantlib_core::{OptionType, PricingEngine, SimpleDate};
use quantlib_engines::MonteCarloEngine;
use quantlib_models::BlackScholesModel;
use quantlib_products::{AsianOption, AveragingType};

struct TrackingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let current = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(current, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

#[test]
fn arithmetic_asian_peak_heap_does_not_grow_with_paths() {
    let valuation = SimpleDate::new(2026, 2, 10).expect("date");
    let option = AsianOption::new(
        100.0,
        100.0,
        SimpleDate::new(2027, 2, 10).expect("date"),
        Some(valuation),
        OptionType::Call,
        1,
        AveragingType::Arithmetic,
        None,
    )
    .expect("option");
    let model = BlackScholesModel::new(0.2, 0.03, 0.01).expect("model");
    let engine = MonteCarloEngine::new(1_000_000, 12, Some(7)).expect("engine");
    // Warm up the Rayon pool so its one-off allocations are not measured.
    engine.calculate_price(&option, &model).expect("warm-up");

    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    engine.calculate_price(&option, &model).expect("price");
    let growth = PEAK.load(Ordering::Relaxed) - baseline;

    // Storing the paths would need 1e6 × 13 × 8 B ≈ 104 MB; streaming needs
    // one small moment record per 4096-path chunk.
    assert!(growth < 1 << 20, "peak heap grew by {growth} bytes");
}
