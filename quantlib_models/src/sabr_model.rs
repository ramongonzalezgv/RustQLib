use std::collections::BTreeMap;

use quantlib_core::{QuantLibError, StochasticModel};

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

impl StochasticModel for SabrModel {
    fn get_parameters(&self) -> BTreeMap<String, String> {
        self.get_parameters_map()
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
    if !(-1.0..=1.0).contains(&rho) {
        return Err(QuantLibError::Validation(
            "rho must be between -1 and 1".to_owned(),
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

fn compute_chi(z: f64, rho: f64) -> f64 {
    let discriminant = (1.0 - 2.0 * rho * z + z * z).max(0.0);
    let sqrt_term = discriminant.sqrt();

    let numerator = sqrt_term + z - rho;
    let denominator = 1.0 - rho;

    if denominator.abs() < 1.0e-12 {
        return z;
    }

    if numerator <= 0.0 {
        return z.abs();
    }

    (numerator / denominator).ln()
}

fn implied_vol_general(
    model: &GeneralSabrModel,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    validate_fk_ttm(forward, strike, ttm)?;

    if (forward - strike).abs() < 1.0e-10 {
        return Ok(atm_vol_general(model, forward, ttm));
    }

    let z = (model.nu / model.alpha)
        * (forward * strike).powf((1.0 - model.beta) / 2.0)
        * (forward / strike).ln();
    let chi_z = compute_chi(z, model.rho);

    if chi_z.abs() < 1.0e-10 {
        return Ok(atm_vol_general(model, forward, ttm));
    }

    let main_term = model.alpha / (forward * strike).powf((1.0 - model.beta) / 2.0) * z / chi_z;

    let log_fk = (forward / strike).ln();
    let fk_power = (forward * strike).powf((1.0 - model.beta) / 2.0);

    let term1 = (((1.0 - model.beta).powi(2) / 24.0) * (model.alpha * model.alpha)
        / (fk_power * fk_power)
        + model.rho * model.beta * model.nu * model.alpha / (4.0 * fk_power)
        + (2.0 - 3.0 * model.rho * model.rho) * model.nu * model.nu / 24.0)
        * ttm;

    let term2 = if z.abs() > 1.0e-6 {
        ((1.0 - model.beta).powi(2) * log_fk * log_fk / 24.0)
            + ((1.0 - model.beta).powi(4) * log_fk.powi(4) / 1920.0)
    } else {
        0.0
    };

    Ok(main_term * (1.0 + term1 + term2))
}

fn atm_vol_general(model: &GeneralSabrModel, forward: f64, ttm: f64) -> f64 {
    let fk_power = forward.powf(1.0 - model.beta);

    let correction = (((1.0 - model.beta).powi(2) / 24.0) * (model.alpha * model.alpha)
        / (fk_power * fk_power)
        + model.rho * model.beta * model.nu * model.alpha / (4.0 * fk_power)
        + (2.0 - 3.0 * model.rho * model.rho) * model.nu * model.nu / 24.0)
        * ttm;

    model.alpha / fk_power * (1.0 + correction)
}

fn implied_vol_normal(
    model: &NormalSabrModel,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    validate_fk_ttm(forward, strike, ttm)?;

    if (forward - strike).abs() < 1.0e-10 {
        return Ok(atm_vol_normal(model, ttm));
    }

    let z = (model.nu / model.alpha) * (forward - strike);
    let chi_z = compute_chi(z, model.rho);

    if chi_z.abs() < 1.0e-10 {
        return Ok(atm_vol_normal(model, ttm));
    }

    let main_term = model.alpha * z / chi_z;
    let correction = (model.rho * model.nu * model.alpha / 4.0
        + (2.0 - 3.0 * model.rho * model.rho) * model.nu * model.nu / 24.0)
        * ttm;

    Ok(main_term * (1.0 + correction))
}

fn atm_vol_normal(model: &NormalSabrModel, ttm: f64) -> f64 {
    let correction = (model.rho * model.nu * model.alpha / 4.0
        + (2.0 - 3.0 * model.rho * model.rho) * model.nu * model.nu / 24.0)
        * ttm;

    model.alpha * (1.0 + correction)
}

fn implied_vol_lognormal(
    model: &LognormalSabrModel,
    forward: f64,
    strike: f64,
    ttm: f64,
) -> Result<f64, QuantLibError> {
    validate_fk_ttm(forward, strike, ttm)?;

    if (forward - strike).abs() < 1.0e-10 {
        return Ok(atm_vol_lognormal(model, ttm));
    }

    let log_fk = (forward / strike).ln();
    let z = (model.nu / model.alpha) * log_fk;
    let chi_z = compute_chi(z, model.rho);

    if chi_z.abs() < 1.0e-10 {
        return Ok(atm_vol_lognormal(model, ttm));
    }

    let main_term = model.alpha * z / chi_z;

    let term1 = (model.rho * model.nu * model.alpha / 4.0
        + (2.0 - 3.0 * model.rho * model.rho) * model.nu * model.nu / 24.0)
        * ttm;

    let term2 = if z.abs() > 1.0e-6 {
        (log_fk * log_fk) / 24.0 + log_fk.powi(4) / 1920.0
    } else {
        0.0
    };

    Ok(main_term * (1.0 + term1 + term2))
}

fn atm_vol_lognormal(model: &LognormalSabrModel, ttm: f64) -> f64 {
    let correction = (model.rho * model.nu * model.alpha / 4.0
        + (2.0 - 3.0 * model.rho * model.rho) * model.nu * model.nu / 24.0)
        * ttm;

    model.alpha * (1.0 + correction)
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
    fn rejects_nan_parameters() {
        assert!(SabrModel::normal(f64::NAN, -0.2, 0.5, 0.01, None).is_err());
        assert!(SabrModel::lognormal(0.2, f64::NAN, 0.5, 0.01, None).is_err());
        assert!(SabrModel::general(0.2, 0.5, -0.2, f64::NAN, 0.01, None).is_err());
    }
}
