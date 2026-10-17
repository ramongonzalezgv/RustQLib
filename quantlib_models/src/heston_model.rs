use num_complex::Complex64;
use quantlib_core::QuantLibError;

#[derive(Debug, Clone, Copy)]
pub struct HestonModel {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub rho: f64,
    pub v0: f64,
    pub r: f64,
    pub q: f64,
}

impl HestonModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        v0: f64,
        r: f64,
        q: f64,
    ) -> Result<Self, QuantLibError> {
        let model = Self {
            kappa,
            theta,
            sigma,
            rho,
            v0,
            r,
            q,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        if !self.kappa.is_finite()
            || !self.theta.is_finite()
            || !self.sigma.is_finite()
            || !self.rho.is_finite()
            || !self.v0.is_finite()
            || !self.r.is_finite()
            || !self.q.is_finite()
        {
            return Err(QuantLibError::Validation(
                "Heston parameters must be finite".to_owned(),
            ));
        }

        if self.kappa < 0.0 {
            return Err(QuantLibError::Validation(
                "kappa must be non-negative".to_owned(),
            ));
        }
        if self.theta < 0.0 {
            return Err(QuantLibError::Validation(
                "theta must be non-negative".to_owned(),
            ));
        }
        if self.sigma < 0.0 {
            return Err(QuantLibError::Validation(
                "sigma (vol-of-vol) must be non-negative".to_owned(),
            ));
        }
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err(QuantLibError::Validation(
                "rho must be within [-1, 1]".to_owned(),
            ));
        }
        if self.v0 < 0.0 {
            return Err(QuantLibError::Validation(
                "v0 must be non-negative".to_owned(),
            ));
        }

        Ok(())
    }

    pub fn characteristic_function(&self, u: Complex64, ttm: f64) -> Complex64 {
        if ttm <= 0.0 {
            return Complex64::new(1.0, 0.0);
        }

        let i = Complex64::new(0.0, 1.0);
        let sigma_sq = self.sigma * self.sigma;

        // Handle near-deterministic vol-of-vol with the sigma -> 0 limit.
        if sigma_sq <= 1.0e-16 {
            // (1 - e^{-kT})/k -> T as k -> 0; the division is 0/0 there.
            let kt = self.kappa * ttm;
            let decay = if kt < 1.0e-8 {
                ttm
            } else {
                -(-kt).exp_m1() / self.kappa
            };
            let integrated_variance = self.theta * ttm + (self.v0 - self.theta) * decay;
            let drift = (self.r - self.q) * ttm;
            return (i * u * drift - 0.5 * (u * u + i * u) * integrated_variance).exp();
        }

        // Same formulation as before with g = (drift − d)/(drift + d) eliminated,
        // which removes two complex divisions from the inner FFT/COS loop.
        let iu = i * u;
        let drift = self.kappa - self.rho * self.sigma * iu;
        let d = principal_sqrt(drift * drift + sigma_sq * (iu + u * u));
        let (minus, plus) = (drift - d, drift + d);
        let exp_dt = (-d * ttm).exp();
        // (drift + d)·(1 − g·e^{−dT})
        let denom = plus - minus * exp_dt;
        // (1 − g·e^{−dT}) / (1 − g)
        let g_term = denom / (2.0 * d);

        let c = (self.r - self.q) * iu * ttm
            + (self.kappa * self.theta / sigma_sq) * (minus * ttm - 2.0 * fast_ln(g_term));
        let d_term = (minus / sigma_sq) * ((1.0 - exp_dt) * plus / denom);

        (c + d_term * self.v0).exp()
    }
}

// The helpers below take |z| from `norm_sqr` rather than `hypot`, which is
// slow in most libms. Heston CF arguments stay far from the f64 overflow range.

/// Principal square root without the polar round trip (`atan2` + `sin_cos`)
/// that `Complex64::sqrt` takes; same branch cut along the negative real axis.
fn principal_sqrt(z: Complex64) -> Complex64 {
    let r = z.norm_sqr().sqrt();
    if r == 0.0 {
        return Complex64::new(0.0, z.im);
    }
    if z.re >= 0.0 {
        let t = (0.5 * (r + z.re)).sqrt();
        Complex64::new(t, z.im / (2.0 * t))
    } else {
        let t = (0.5 * (r - z.re)).sqrt();
        Complex64::new(z.im.abs() / (2.0 * t), t.copysign(z.im))
    }
}

/// Principal logarithm, as `Complex64::ln`.
fn fast_ln(z: Complex64) -> Complex64 {
    Complex64::new(0.5 * z.norm_sqr().ln(), z.arg())
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::HestonModel;

    #[test]
    fn validates_rho_range() {
        let bad = HestonModel::new(1.0, 0.04, 0.3, 1.2, 0.04, 0.02, 0.0);
        assert!(bad.is_err());
    }

    #[test]
    fn characteristic_function_is_one_at_origin() {
        let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01).expect("model");
        let phi0 = model.characteristic_function(Complex64::new(0.0, 0.0), 1.0);
        assert!(
            (phi0 - Complex64::new(1.0, 0.0)).norm() < 1.0e-12,
            "phi0={phi0:?}"
        );
    }

    #[test]
    fn characteristic_function_respects_first_moment() {
        let model = HestonModel::new(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01).expect("model");
        let ttm = 1.25;
        let u = Complex64::new(0.0, -1.0);
        let phi = model.characteristic_function(u, ttm);
        let expected = ((model.r - model.q) * ttm).exp();
        assert!(
            (phi.re - expected).abs() < 1.0e-10,
            "phi={phi:?}, expected={expected}"
        );
        assert!(phi.im.abs() < 1.0e-10, "phi={phi:?}");
    }

    #[test]
    fn principal_sqrt_matches_num_complex() {
        for &(re, im) in &[
            (3.0, 4.0),
            (-3.0, 4.0),
            (-3.0, -4.0),
            (3.0, -4.0),
            (-2.0, 0.0),
            (-2.0, -0.0),
            (1.0e-30, 1.0e30),
            (0.0, -5.0),
        ] {
            let z = Complex64::new(re, im);
            let (ours, theirs) = (super::principal_sqrt(z), z.sqrt());
            assert!(
                (ours - theirs).norm() <= 1.0e-15 * theirs.norm(),
                "z={z:?}: ours={ours:?}, num_complex={theirs:?}"
            );
            assert_eq!(ours.im.is_sign_negative(), theirs.im.is_sign_negative());
        }
    }

    #[test]
    fn rejects_nan_inputs() {
        let model = HestonModel::new(f64::NAN, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01);
        assert!(model.is_err());
    }
}
