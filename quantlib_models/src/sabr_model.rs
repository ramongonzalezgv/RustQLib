use std::collections::BTreeMap;

use quantlib_core::QuantLibError;

#[derive(Debug, Clone, Copy)]
pub struct GeneralSabrModel {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
    pub r: f64,
    pub f0: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct NormalSabrModel {
    pub alpha: f64,
    pub rho: f64,
    pub nu: f64,
    pub r: f64,
    pub f0: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct LognormalSabrModel {
    pub alpha: f64,
    pub rho: f64,
    pub nu: f64,
    pub r: f64,
    pub f0: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ShiftedSabrModel {
    pub base_model: Box<SabrModel>,
    pub shift: f64,
}

#[derive(Debug, Clone)]
pub enum SabrModel {
    General(GeneralSabrModel),
    Normal(NormalSabrModel),
    Lognormal(LognormalSabrModel),
    Shifted(ShiftedSabrModel),
}

impl GeneralSabrModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
        r: f64,
        f0: Option<f64>,
    ) -> Result<Self, QuantLibError> {
        validate_common_sabr(alpha, rho, nu)?;
        if !beta.is_finite() {
            return Err(QuantLibError::Validation(
                "beta must be finite for General SABR".to_owned(),
            ));
        }
        if !(0.0 < beta && beta < 1.0) {
            return Err(QuantLibError::Validation(
                "beta must be in (0, 1) for General SABR".to_owned(),
            ));
        }
        if !r.is_finite() {
            return Err(QuantLibError::Validation("r must be finite".to_owned()));
        }
        validate_optional_f0(f0)?;

        Ok(Self {
            alpha,
            beta,
            rho,
            nu,
            r,
            f0,
        })
    }
}

impl NormalSabrModel {
    pub fn new(
        alpha: f64,
        rho: f64,
        nu: f64,
        r: f64,
        f0: Option<f64>,
    ) -> Result<Self, QuantLibError> {
        validate_common_sabr(alpha, rho, nu)?;
        if !r.is_finite() {
            return Err(QuantLibError::Validation("r must be finite".to_owned()));
        }
        validate_optional_f0(f0)?;
        Ok(Self {
            alpha,
            rho,
            nu,
            r,
            f0,
        })
    }
}

impl LognormalSabrModel {
    pub fn new(
        alpha: f64,
        rho: f64,
        nu: f64,
        r: f64,
        f0: Option<f64>,
    ) -> Result<Self, QuantLibError> {
        validate_common_sabr(alpha, rho, nu)?;
        if !r.is_finite() {
            return Err(QuantLibError::Validation("r must be finite".to_owned()));
        }
        validate_optional_f0(f0)?;
        Ok(Self {
            alpha,
            rho,
            nu,
            r,
            f0,
        })
    }
}

impl ShiftedSabrModel {
    pub fn new(base_model: SabrModel, shift: f64) -> Result<Self, QuantLibError> {
        base_model.validate()?;

        if !shift.is_finite() || shift <= 0.0 {
            return Err(QuantLibError::Validation(
                "shift must be finite and positive".to_owned(),
            ));
        }

        Ok(Self {
            base_model: Box::new(base_model),
            shift,
        })
    }
}

impl SabrModel {
    #[allow(clippy::too_many_arguments)]
    pub fn general(
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
        r: f64,
        f0: Option<f64>,
    ) -> Result<Self, QuantLibError> {
        Ok(Self::General(GeneralSabrModel::new(
            alpha, beta, rho, nu, r, f0,
        )?))
    }

    pub fn normal(
        alpha: f64,
        rho: f64,
        nu: f64,
        r: f64,
        f0: Option<f64>,
    ) -> Result<Self, QuantLibError> {
        Ok(Self::Normal(NormalSabrModel::new(alpha, rho, nu, r, f0)?))
    }

    pub fn lognormal(
        alpha: f64,
        rho: f64,
        nu: f64,
        r: f64,
        f0: Option<f64>,
    ) -> Result<Self, QuantLibError> {
        Ok(Self::Lognormal(LognormalSabrModel::new(
            alpha, rho, nu, r, f0,
        )?))
    }

    pub fn shifted(base_model: SabrModel, shift: f64) -> Result<Self, QuantLibError> {
        Ok(Self::Shifted(ShiftedSabrModel::new(base_model, shift)?))
    }

    pub fn validate(&self) -> Result<(), QuantLibError> {
        match self {
            Self::General(model) => {
                validate_common_sabr(model.alpha, model.rho, model.nu)?;
                if !model.beta.is_finite() {
                    return Err(QuantLibError::Validation(
                        "beta must be finite for General SABR".to_owned(),
                    ));
                }
                if !(0.0 < model.beta && model.beta < 1.0) {
                    return Err(QuantLibError::Validation(
                        "beta must be in (0, 1) for General SABR".to_owned(),
                    ));
                }
                if !model.r.is_finite() {
                    return Err(QuantLibError::Validation("r must be finite".to_owned()));
                }
                validate_optional_f0(model.f0)
            }
            Self::Normal(model) => {
                validate_common_sabr(model.alpha, model.rho, model.nu)?;
                if !model.r.is_finite() {
                    return Err(QuantLibError::Validation("r must be finite".to_owned()));
                }
                validate_optional_f0(model.f0)
            }
            Self::Lognormal(model) => {
                validate_common_sabr(model.alpha, model.rho, model.nu)?;
                if !model.r.is_finite() {
                    return Err(QuantLibError::Validation("r must be finite".to_owned()));
                }
                validate_optional_f0(model.f0)
            }
            Self::Shifted(model) => {
                model.base_model.validate()?;
                if !model.shift.is_finite() || model.shift <= 0.0 {
                    return Err(QuantLibError::Validation(
                        "shift must be finite and positive".to_owned(),
                    ));
                }
                Ok(())
            }
        }
    }

    pub fn beta(&self) -> f64 {
        match self {
            Self::General(model) => model.beta,
            Self::Normal(_) => 0.0,
            Self::Lognormal(_) => 1.0,
            Self::Shifted(model) => model.base_model.beta(),
        }
    }

    /// Total displacement applied to F and K (0 for unshifted variants).
    pub fn shift(&self) -> f64 {
        match self {
            Self::Shifted(model) => model.shift + model.base_model.shift(),
            _ => 0.0,
        }
    }

    pub fn r(&self) -> f64 {
        match self {
            Self::General(model) => model.r,
            Self::Normal(model) => model.r,
            Self::Lognormal(model) => model.r,
            Self::Shifted(model) => model.base_model.r(),
        }
    }

    pub fn implied_volatility(
        &self,
        forward: f64,
        strike: f64,
        ttm: f64,
    ) -> Result<f64, QuantLibError> {
        self.validate()?;

        match self {
            Self::General(model) => implied_vol_general(model, forward, strike, ttm),
            Self::Normal(model) => implied_vol_normal(model, forward, strike, ttm),
            Self::Lognormal(model) => implied_vol_lognormal(model, forward, strike, ttm),
            Self::Shifted(model) => {
                let f_shifted = forward + model.shift;
                let k_shifted = strike + model.shift;
                if f_shifted <= 0.0 || k_shifted <= 0.0 {
                    return Err(QuantLibError::Validation(
                        "shifted F and K must be positive".to_owned(),
                    ));
                }

                model
                    .base_model
                    .implied_volatility(f_shifted, k_shifted, ttm)
            }
        }
    }

    pub fn get_parameters_map(&self) -> BTreeMap<String, String> {
        match self {
            Self::General(model) => {
                let mut params = BTreeMap::from([
                    ("alpha".to_owned(), model.alpha.to_string()),
                    ("beta".to_owned(), model.beta.to_string()),
                    ("rho".to_owned(), model.rho.to_string()),
                    ("nu".to_owned(), model.nu.to_string()),
                    ("r".to_owned(), model.r.to_string()),
                ]);
                if let Some(f0) = model.f0 {
                    params.insert("F0".to_owned(), f0.to_string());
                }
                params
            }
            Self::Normal(model) => {
                let mut params = BTreeMap::from([
                    ("alpha".to_owned(), model.alpha.to_string()),
                    ("beta".to_owned(), "0".to_owned()),
                    ("rho".to_owned(), model.rho.to_string()),
                    ("nu".to_owned(), model.nu.to_string()),
                    ("r".to_owned(), model.r.to_string()),
                ]);
                if let Some(f0) = model.f0 {
                    params.insert("F0".to_owned(), f0.to_string());
                }
                params
            }
            Self::Lognormal(model) => {
                let mut params = BTreeMap::from([
                    ("alpha".to_owned(), model.alpha.to_string()),
                    ("beta".to_owned(), "1".to_owned()),
                    ("rho".to_owned(), model.rho.to_string()),
                    ("nu".to_owned(), model.nu.to_string()),
                    ("r".to_owned(), model.r.to_string()),
                ]);
                if let Some(f0) = model.f0 {
                    params.insert("F0".to_owned(), f0.to_string());
                }
                params
            }
            Self::Shifted(model) => {
                let mut params = model.base_model.get_parameters_map();
                params.insert("shift".to_owned(), model.shift.to_string());
                params
            }
        }
    }
}

fn validate_common_sabr(alpha: f64, rho: f64, nu: f64) -> Result<(), QuantLibError> {
    if !alpha.is_finite() || !rho.is_finite() || !nu.is_finite() {
        return Err(QuantLibError::Validation(
            "alpha, rho, and nu must be finite".to_owned(),
        ));
    }
    if alpha <= 0.0 {
        return Err(QuantLibError::Validation(
            "alpha must be positive".to_owned(),
        ));
    }
    // Hagan's x(z) divides by 1 - rho and 1 + rho: |rho| = 1 is degenerate.
    if !(rho > -1.0 && rho < 1.0) {
        return Err(QuantLibError::Validation(
            "rho must be strictly between -1 and 1".to_owned(),
        ));
    }
    if nu <= 0.0 {
        return Err(QuantLibError::Validation("nu must be positive".to_owned()));
    }
    Ok(())
}

fn validate_fk_ttm(forward: f64, strike: f64, ttm: f64) -> Result<(), QuantLibError> {
    if !forward.is_finite() || !strike.is_finite() || !ttm.is_finite() {
        return Err(QuantLibError::Validation(
            "forward, strike, and time to maturity must be finite".to_owned(),
        ));
    }
    if forward <= 0.0 || strike <= 0.0 {
        return Err(QuantLibError::Validation(
            "forward and strike must be positive".to_owned(),
        ));
    }
    if ttm <= 0.0 {
        return Err(QuantLibError::Validation(
            "time to maturity must be positive".to_owned(),
        ));
    }
    Ok(())
}

fn validate_optional_f0(f0: Option<f64>) -> Result<(), QuantLibError> {
    if let Some(value) = f0
        && !value.is_finite()
    {
        return Err(QuantLibError::Validation("f0 must be finite".to_owned()));
    }
    Ok(())
}

/// `z / x(z)` from Hagan et al. (2002), with
/// `x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho))`.
///
/// `x(z)` is evaluated through `ln_1p` on a cancellation-free argument
/// (using `sqrt(D) - 1 = (z^2 - 2 rho z) / (sqrt(D) + 1)`), separately for
/// each sign of `z` via `(sqrt(D) + z - rho)(sqrt(D) - z + rho) = 1 - rho^2`.
/// Near `z = 0` the ratio uses its Taylor expansion, so the smile is one
/// continuous formula across the money.
fn z_over_x(z: f64, rho: f64) -> f64 {
    if z.abs() < 1.0e-6 {
        // Error O(z^3) < 1e-18.
        return 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0;
    }
    let sqrt_d = (1.0 - 2.0 * rho * z + z * z).sqrt();
    let sqrt_d_minus_1 = (z * z - 2.0 * rho * z) / (sqrt_d + 1.0);
    let x = if z >= 0.0 {
        ((sqrt_d_minus_1 + z) / (1.0 - rho)).ln_1p()
    } else {
        -((sqrt_d_minus_1 - z) / (1.0 + rho)).ln_1p()
    };
    z / x
}

/// Hagan et al. (2002), "Managing Smile Risk", eq. (2.17a): Black (lognormal)
/// implied volatility for `0 <= beta <= 1`. At `K = F` it reduces to the ATM
/// formula (2.18) without a separate branch.
fn hagan_lognormal_vol(
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> f64 {
    let one_minus_beta = 1.0 - beta;
    let log_fk = (forward / strike).ln();
    let fk_pow = (forward * strike).powf(0.5 * one_minus_beta);

    let omb2_log2 = one_minus_beta * one_minus_beta * log_fk * log_fk;
    let moneyness_series = 1.0 + omb2_log2 / 24.0 + omb2_log2 * omb2_log2 / 1920.0;

    let z = nu / alpha * fk_pow * log_fk;

    let time_correction = 1.0
        + (one_minus_beta * one_minus_beta * alpha * alpha / (24.0 * fk_pow * fk_pow)
            + rho * beta * nu * alpha / (4.0 * fk_pow)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0)
            * ttm;

    alpha / (fk_pow * moneyness_series) * z_over_x(z, rho) * time_correction
}

/// Hagan et al. (2002) normal (Bachelier) implied volatility for `beta = 0`:
/// `alpha * z/x(z) * (1 + (2 - 3 rho^2) nu^2 T / 24)`, `z = nu/alpha (F - K)`.
/// The `rho beta nu alpha / 4` and `beta (2 - beta) alpha^2` terms of the
/// general expansion vanish at `beta = 0`.
fn hagan_normal_vol(alpha: f64, rho: f64, nu: f64, forward: f64, strike: f64, ttm: f64) -> f64 {
    let z = nu / alpha * (forward - strike);
    alpha * z_over_x(z, rho) * (1.0 + (2.0 - 3.0 * rho * rho) * nu * nu * ttm / 24.0)
}

fn implied_vol_general(
    model: &GeneralSabrModel,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    validate_fk_ttm(forward, strike, ttm)?;
    Ok(hagan_lognormal_vol(
        model.alpha,
        model.beta,
        model.rho,
        model.nu,
        forward,
        strike,
        ttm,
    ))
}

fn implied_vol_normal(
    model: &NormalSabrModel,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    validate_fk_ttm(forward, strike, ttm)?;
    Ok(hagan_normal_vol(
        model.alpha,
        model.rho,
        model.nu,
        forward,
        strike,
        ttm,
    ))
}

fn implied_vol_lognormal(
    model: &LognormalSabrModel,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    validate_fk_ttm(forward, strike, ttm)?;
    Ok(hagan_lognormal_vol(
        model.alpha,
        1.0,
        model.rho,
        model.nu,
        forward,
        strike,
        ttm,
    ))
}

#[cfg(test)]
mod tests {
    use super::{GeneralSabrModel, SabrModel};

    #[test]
    fn general_sabr_validation_rejects_beta_bounds() {
        let bad = GeneralSabrModel::new(0.2, 1.0, -0.3, 0.5, 0.01, None);
        assert!(bad.is_err());
    }

    #[test]
    fn normal_sabr_implied_vol_is_positive() {
        let model = SabrModel::normal(0.2, -0.2, 0.5, 0.01, None).expect("model");
        let vol = model.implied_volatility(100.0, 110.0, 1.0).expect("vol");
        assert!(vol > 0.0);
    }

    #[test]
    fn shifted_sabr_delegates_to_base_with_shifted_strikes() {
        let base = SabrModel::lognormal(0.25, -0.1, 0.6, 0.01, None).expect("base");
        let shifted = SabrModel::shifted(base.clone(), 0.03).expect("shifted");

        let vol_shifted = shifted
            .implied_volatility(0.02, 0.01, 2.0)
            .expect("shifted vol");
        let vol_base = base
            .implied_volatility(0.05, 0.04, 2.0)
            .expect("base vol with shifted F/K");

        assert!((vol_shifted - vol_base).abs() < 1.0e-12);
    }

    #[test]
    fn z_over_x_is_finite_and_positive_over_wide_range() {
        for rho in [-0.99, 0.0, 0.99] {
            for i in -500..=500 {
                let z = f64::from(i) / 10.0;
                let ratio = super::z_over_x(z, rho);
                assert!(
                    ratio.is_finite() && ratio > 0.0,
                    "z={z}, rho={rho}: {ratio}"
                );
            }
        }
    }

    #[test]
    fn z_over_x_series_matches_closed_form_at_the_switch() {
        // Both sides of the |z| = 1e-6 switch agree to O(z^3).
        for rho in [-0.9, -0.3, 0.0, 0.5, 0.9] {
            let below = super::z_over_x(0.999_999e-6, rho);
            let above = super::z_over_x(1.000_001e-6, rho);
            assert!(
                (below - above).abs() < 1e-12,
                "rho={rho}: {below} vs {above}"
            );
        }
    }

    #[test]
    fn rejects_nan_parameters() {
        assert!(SabrModel::normal(f64::NAN, -0.2, 0.5, 0.01, None).is_err());
        assert!(SabrModel::lognormal(0.2, f64::NAN, 0.5, 0.01, None).is_err());
        assert!(SabrModel::general(0.2, 0.5, -0.2, f64::NAN, 0.01, None).is_err());
    }
}
