//! Heston variance discretizations: full truncation and Andersen QE.

use quantlib_core::norm_cdf;
use quantlib_models::HestonModel;

use super::core::HestonScheme;

#[derive(Debug, Clone, Copy)]
pub(crate) struct HestonState {
    pub(crate) spot: f64,
    pub(crate) variance: f64,
}

/// One Heston time step, with every per-step constant precomputed.
///
/// Each step consumes two independent standard normals: `z_s` drives the
/// stock and `z_v` the variance (the QE scheme turns it into its uniform via
/// `U = Φ(z_v)`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct HestonStepper {
    scheme: HestonScheme,
    dt: f64,
    sqrt_dt: f64,
    drift: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    rho_orth: f64,
    qe: QeConstants,
}

/// Andersen (2008) QE constants for γ₁ = γ₂ = ½.
#[derive(Debug, Clone, Copy)]
struct QeConstants {
    /// `e^{−κΔ}`.
    decay: f64,
    /// `(1 − e^{−κΔ}) / κ`, which tends to `Δ` as `κ → 0`.
    decay_integral: f64,
    k0: f64,
    k1: f64,
    k2: f64,
    k3: f64,
    k4: f64,
}

/// Switching level of `ψ = s²/m²` between the quadratic and exponential branches.
const QE_PSI_CRITICAL: f64 = 1.5;

/// Below this vol-of-vol the variance is treated as deterministic, where the
/// QE coefficients (which divide by σ) are ill-defined.
const DETERMINISTIC_SIGMA: f64 = 1.0e-8;

impl HestonStepper {
    pub(crate) fn new(model: &HestonModel, dt: f64, scheme: HestonScheme) -> Self {
        let (kappa, theta, sigma, rho) = (model.kappa, model.theta, model.sigma, model.rho);
        let decay = (-kappa * dt).exp();
        let decay_integral = if kappa * dt < 1.0e-12 {
            dt
        } else {
            -(-kappa * dt).exp_m1() / kappa
        };

        let (gamma1, gamma2) = (0.5, 0.5);
        // Non-finite for σ ≤ DETERMINISTIC_SIGMA, where `qe_step` never reads them.
        let lead = kappa * rho / sigma - 0.5;

        Self {
            scheme,
            dt,
            sqrt_dt: dt.sqrt(),
            drift: (model.r - model.q) * dt,
            kappa,
            theta,
            sigma,
            rho,
            rho_orth: (1.0 - rho * rho).max(0.0).sqrt(),
            qe: QeConstants {
                decay,
                decay_integral,
                k0: -rho * kappa * theta * dt / sigma,
                k1: gamma1 * dt * lead - rho / sigma,
                k2: gamma2 * dt * lead + rho / sigma,
                k3: gamma1 * dt * (1.0 - rho * rho),
                k4: gamma2 * dt * (1.0 - rho * rho),
            },
        }
    }

    #[inline]
    pub(crate) fn step(&self, state: &mut HestonState, z_s: f64, z_v: f64) {
        match self.scheme {
            HestonScheme::FullTruncation => self.full_truncation_step(state, z_s, z_v),
            HestonScheme::QuadraticExponential => self.qe_step(state, z_s, z_v),
        }
    }

    /// Lord, Koekkoek & van Dijk (2010): the variance is never floored; only
    /// its positive part enters the drift and diffusion.
    fn full_truncation_step(&self, state: &mut HestonState, z_s: f64, z_v: f64) {
        let v_pos = state.variance.max(0.0);
        let vol = v_pos.sqrt();
        let d_w1 = self.sqrt_dt * z_s;
        let d_w2 = self.sqrt_dt * (self.rho * z_s + self.rho_orth * z_v);

        state.spot *= (self.drift - 0.5 * v_pos * self.dt + vol * d_w1).exp();
        state.variance += self.kappa * (self.theta - v_pos) * self.dt + self.sigma * vol * d_w2;
    }

    /// Andersen (2008) quadratic-exponential scheme with martingale correction.
    fn qe_step(&self, state: &mut HestonState, z_s: f64, z_v: f64) {
        let qe = &self.qe;
        let v = state.variance;

        // Exact conditional mean and variance of v_{n+1} given v_n.
        let mean = self.theta + (v - self.theta) * qe.decay;
        let sigma_sq = self.sigma * self.sigma;
        let var = v * sigma_sq * qe.decay * qe.decay_integral
            + 0.5 * self.theta * sigma_sq * (1.0 - qe.decay) * qe.decay_integral;

        if self.sigma <= DETERMINISTIC_SIGMA || mean <= 0.0 || var <= 0.0 {
            // Deterministic variance: integrate it with the trapezoidal rule.
            let v_next = mean.max(0.0);
            let integrated = 0.5 * (v + v_next) * self.dt;
            state.spot *= (self.drift - 0.5 * integrated + integrated.sqrt() * z_s).exp();
            state.variance = v_next;
            return;
        }

        let a_coeff = qe.k2 + 0.5 * qe.k4;
        let psi = var / (mean * mean);
        let (v_next, log_mgf) = if psi <= QE_PSI_CRITICAL {
            let inv = 2.0 / psi;
            let b_sq = inv - 1.0 + inv.sqrt() * (inv - 1.0).sqrt();
            let a = mean / (1.0 + b_sq);
            let b = b_sq.sqrt();
            let v_next = a * (b + z_v) * (b + z_v);
            // ln E[e^{A·v_{n+1}}], finite only for A < 1/(2a).
            let denom = 1.0 - 2.0 * a_coeff * a;
            let log_mgf = (denom > 0.0).then(|| a_coeff * b_sq * a / denom - 0.5 * denom.ln());
            (v_next, log_mgf)
        } else {
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / mean;
            let u = norm_cdf(z_v);
            let v_next = if u <= p {
                0.0
            } else {
                ((1.0 - p) / (1.0 - u)).ln() / beta
            };
            // ln E[e^{A·v_{n+1}}], finite only for A < β.
            let log_mgf = (a_coeff < beta).then(|| (p + beta * (1.0 - p) / (beta - a_coeff)).ln());
            (v_next, log_mgf)
        };

        // Martingale-corrected K₀* keeps E[S_{n+1} | S_n] = S_n e^{(r−q)Δ}; when
        // the moment generating function does not exist, keep Andersen's K₀
        // (that step then breaks the martingale, biasing control variates slightly).
        let k0 = log_mgf.map_or(qe.k0, |log_mgf| -log_mgf - (qe.k1 + 0.5 * qe.k3) * v);
        let diffusion = (qe.k3 * v + qe.k4 * v_next).max(0.0).sqrt();
        state.spot *= (self.drift + k0 + qe.k1 * v + qe.k2 * v_next + diffusion * z_s).exp();
        state.variance = v_next;
    }
}
