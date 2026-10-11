//! Shared plumbing for the Fourier engines (FFT, COS).

use quantlib_core::{OptionType, QuantLibError};
use quantlib_models::FourierModel;
use quantlib_products::EuropeanOption;

/// Unit-quantity prices of one maturity strip, as computed by a Fourier engine.
pub(crate) trait FourierStrip {
    fn strip<M: FourierModel>(
        &self,
        model: &M,
        spot: f64,
        ttm: f64,
        strikes: &[f64],
        types: &[OptionType],
    ) -> Result<Vec<f64>, QuantLibError>;
}

/// Spot for spot-quoted models, forward for forward-quoted ones (Black-76).
pub(crate) fn underlying<M: FourierModel>(product: &EuropeanOption) -> Result<f64, QuantLibError> {
    if M::FORWARD_QUOTED {
        product.require_forward()
    } else {
        product.require_spot()
    }
}

/// Validates inputs, handles the expiry and zero-strike shortcuts, and delegates
/// everything else to the engine's strip pricer with a single strike.
pub(crate) fn price_european<E, M>(
    engine: &E,
    product: &EuropeanOption,
    model: &M,
) -> Result<f64, QuantLibError>
where
    E: FourierStrip,
    M: FourierModel,
{
    product.validate()?;
    model.validate()?;

    let spot = underlying::<M>(product)?;
    let ttm = product.ttm();
    let qty = f64::from(product.qty);

    if ttm <= 0.0 {
        let intrinsic = match product.option_type {
            OptionType::Call => (spot - product.k).max(0.0),
            OptionType::Put => (product.k - spot).max(0.0),
        };
        return Ok(intrinsic * qty);
    }

    if product.k <= 0.0 {
        let unit_price = match product.option_type {
            OptionType::Call => spot * (-model.dividend_yield() * ttm).exp(),
            OptionType::Put => 0.0,
        };
        return Ok(unit_price * qty);
    }

    let prices = engine.strip(model, spot, ttm, &[product.k], &[product.option_type])?;
    Ok(prices[0] * qty)
}

/// Checks the strip arguments shared by every Fourier engine.
pub(crate) fn validate_strip<M: FourierModel>(
    model: &M,
    spot: f64,
    ttm: f64,
    strikes: &[f64],
    types: &[OptionType],
) -> Result<(), QuantLibError> {
    model.validate()?;
    if !spot.is_finite() || spot <= 0.0 {
        return Err(QuantLibError::Validation(
            "spot must be finite and positive".to_owned(),
        ));
    }
    if !ttm.is_finite() || ttm <= 0.0 {
        return Err(QuantLibError::Validation(
            "ttm must be finite and positive".to_owned(),
        ));
    }
    if strikes.len() != types.len() {
        return Err(QuantLibError::Validation(
            "strikes and types must have the same length".to_owned(),
        ));
    }
    if strikes.iter().any(|k| !k.is_finite() || *k <= 0.0) {
        return Err(QuantLibError::Validation(
            "strikes must be finite and positive".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_price(price: f64, engine: &str) -> Result<f64, QuantLibError> {
    if !price.is_finite() {
        return Err(QuantLibError::Engine(format!(
            "{engine} pricing produced a non-finite result"
        )));
    }
    Ok(price)
}
