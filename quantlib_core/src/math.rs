//! Standard normal distribution functions shared by every engine.
//!
//! `erfc` comes from the pure-Rust `libm` crate (a port of musl), so there is
//! no `unsafe` and no system linkage: the crate builds on Windows MSVC and
//! `wasm32-unknown-unknown`.

use std::f64::consts::FRAC_1_SQRT_2;

/// 1 / sqrt(2 pi).
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// sqrt(2 pi).
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Standard normal density.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF, `0.5 * erfc(-x / sqrt(2))`.
///
/// Unlike `0.5 * (1 + erf(x / sqrt(2)))` this keeps full relative precision in
/// the left tail (no cancellation), e.g. `norm_cdf(-10) = 7.6e-24`.
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * FRAC_1_SQRT_2)
}

/// Inverse standard normal CDF.
///
/// Acklam's rational approximation (relative error < 1.15e-9) followed by one
/// Halley step on `norm_cdf`, which brings it to near machine precision.
/// Returns `-inf` / `+inf` at 0 / 1 and `NaN` outside [0, 1].
pub fn inv_norm_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;

    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    let tail = |q: f64| {
        let q = (-2.0 * q.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    let x = if p < P_LOW {
        tail(p)
    } else if p > 1.0 - P_LOW {
        -tail(1.0 - p)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };

    // Halley refinement. In the upper tail refine on the complement so the
    // residual is not swamped by rounding of values close to 1.
    let (e, sign) = if p > 0.5 {
        (norm_cdf(-x) - (1.0 - p), -1.0)
    } else {
        (norm_cdf(x) - p, 1.0)
    };
    let u = sign * e * SQRT_2PI * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(actual: f64, expected: f64) -> f64 {
        ((actual - expected) / expected).abs()
    }

    #[test]
    fn cdf_left_tail_has_full_relative_precision() {
        // Reference values from mpmath (50 digits).
        assert!(rel_err(norm_cdf(-10.0), 7.619_853_024_160_527e-24) < 1e-13);
        assert!(rel_err(norm_cdf(-5.0), 2.866_515_718_791_939e-7) < 1e-14);
        assert!(rel_err(norm_cdf(-1.0), 0.158_655_253_931_457_05) < 1e-15);
        assert!(norm_cdf(-37.5) > 0.0);
    }

    #[test]
    fn cdf_is_symmetric() {
        for i in -800..=800 {
            let x = f64::from(i) / 100.0;
            let s = norm_cdf(x) + norm_cdf(-x);
            assert!((s - 1.0).abs() <= 2.3e-16, "x={x}: sum={s}");
        }
    }

    #[test]
    fn pdf_matches_definition() {
        assert!((norm_pdf(0.0) - FRAC_1_SQRT_2PI).abs() < 1e-17);
        assert!(rel_err(norm_pdf(2.0), 0.053_990_966_513_188_06) < 1e-15);
    }

    #[test]
    fn inverse_round_trips() {
        // Lower tail and centre: the CDF value carries full information.
        for i in -800..=200 {
            let x = f64::from(i) / 100.0;
            let back = inv_norm_cdf(norm_cdf(x));
            let tol = 1e-12 * x.abs().max(1.0);
            assert!((back - x).abs() < tol, "x={x}: back={back}");
        }
        // Upper tail: 1 - p is rounded to the f64 grid near 1, so only check
        // symmetry where that rounding is negligible.
        for p in [1e-8, 1e-4, 0.01, 0.3] {
            let (lo, hi) = (inv_norm_cdf(p), inv_norm_cdf(1.0 - p));
            assert!((lo + hi).abs() < 1e-8, "p={p}: {lo} vs {hi}");
        }
    }

    #[test]
    fn inverse_edge_cases() {
        assert_eq!(inv_norm_cdf(0.5), 0.0);
        assert_eq!(inv_norm_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(inv_norm_cdf(1.0), f64::INFINITY);
        assert!(inv_norm_cdf(-0.1).is_nan());
        assert!(inv_norm_cdf(1.1).is_nan());
        assert!(inv_norm_cdf(f64::NAN).is_nan());
    }
}
