import math
import unittest
from datetime import date, timedelta

try:
    import quantlib_py as ql
except Exception as exc:  # pragma: no cover
    ql = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class TestCalibrationBindings(unittest.TestCase):
    VALUATION_DATE = date(2026, 1, 1)

    @classmethod
    def setUpClass(cls) -> None:
        if ql is None:
            raise unittest.SkipTest(f"quantlib_py import failed: {IMPORT_ERROR}")

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def _black_price(
        cls,
        forward: float,
        strike: float,
        maturity: float,
        sigma: float,
        r: float,
        is_call: bool,
    ) -> float:
        vol_term = sigma * math.sqrt(maturity)
        discount = math.exp(-r * maturity)
        if vol_term <= 1.0e-12:
            intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
            return discount * intrinsic

        d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * maturity) / vol_term
        d2 = d1 - vol_term
        if is_call:
            return discount * (forward * cls._norm_cdf(d1) - strike * cls._norm_cdf(d2))
        return discount * (strike * cls._norm_cdf(-d2) - forward * cls._norm_cdf(-d1))

    @classmethod
    def _bs_price(
        cls,
        spot: float,
        strike: float,
        maturity: float,
        sigma: float,
        r: float,
        q: float,
        is_call: bool,
    ) -> float:
        vol_term = sigma * math.sqrt(maturity)
        discounted_spot = spot * math.exp(-q * maturity)
        discounted_strike = strike * math.exp(-r * maturity)
        if vol_term <= 1.0e-12:
            intrinsic = max(discounted_spot - discounted_strike, 0.0) if is_call else max(
                discounted_strike - discounted_spot, 0.0
            )
            return intrinsic

        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * maturity) / vol_term
        d2 = d1 - vol_term
        if is_call:
            return discounted_spot * cls._norm_cdf(d1) - discounted_strike * cls._norm_cdf(d2)
        return discounted_strike * cls._norm_cdf(-d2) - discounted_spot * cls._norm_cdf(-d1)

    @staticmethod
    def _sabr_implied_vol(
        forward: float,
        strike: float,
        maturity: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        if abs(forward - strike) < 1.0e-10:
            fk_power = forward ** (1.0 - beta)
            correction = (
                (((1.0 - beta) ** 2) / 24.0) * (alpha * alpha) / (fk_power * fk_power)
                + rho * beta * nu * alpha / (4.0 * fk_power)
                + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
            ) * maturity
            return alpha / fk_power * (1.0 + correction)

        z = (nu / alpha) * ((forward * strike) ** ((1.0 - beta) / 2.0)) * math.log(forward / strike)
        discriminant = max(0.0, 1.0 - 2.0 * rho * z + z * z)
        numerator = math.sqrt(discriminant) + z - rho
        denominator = 1.0 - rho
        x_z = math.log(numerator / denominator) if numerator > 0.0 and abs(denominator) > 1.0e-12 else z

        if abs(x_z) < 1.0e-12:
            return alpha

        fk_power = (forward * strike) ** ((1.0 - beta) / 2.0)
        main_term = alpha / fk_power * z / x_z
        log_fk = math.log(forward / strike)
        term1 = (
            (((1.0 - beta) ** 2) / 24.0) * (alpha * alpha) / (fk_power * fk_power)
            + rho * beta * nu * alpha / (4.0 * fk_power)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * maturity
        term2 = ((1.0 - beta) ** 2) * log_fk * log_fk / 24.0 + ((1.0 - beta) ** 4) * (
            log_fk**4
        ) / 1920.0
        return main_term * (1.0 + term1 + term2)

    @classmethod
    def _expiry(cls, maturity: float) -> str:
        return (cls.VALUATION_DATE + timedelta(days=round(maturity * 365.0))).isoformat()

    def test_calibrate_black_from_prices(self) -> None:
        true_sigma = 0.27
        true_r = 0.02

        forwards = []
        strikes = []
        maturities = []
        is_call = []
        market_values = []

        for maturity in (0.3, 0.8, 1.2):
            for strike in (85.0, 95.0, 100.0, 110.0, 120.0):
                for call in (True, False):
                    forwards.append(100.0)
                    strikes.append(strike)
                    maturities.append(maturity)
                    is_call.append(call)
                    market_values.append(
                        self._black_price(100.0, strike, maturity, true_sigma, true_r, call)
                    )

        result = ql.calibrate_black(
            forwards=forwards,
            strikes=strikes,
            maturities=maturities,
            is_call=is_call,
            market_values=market_values,
            target="price",
            initial_sigma=0.16,
            r=true_r,
            method="hybrid",
            objective="least_squares",
            max_generations=160,
            population_size=36,
            nm_max_iterations=180,
            tolerance=1.0e-6,
            seed=123,
        )

        self.assertAlmostEqual(result["model"]["sigma"], true_sigma, delta=2.0e-2)
        self.assertEqual(result["model"]["r"], true_r)
        self.assertEqual(list(result["calibrated_params"]), ["sigma"])
        self.assertLess(result["rmse"], 2.0e-2)

    def test_calibrate_black_scholes_from_prices(self) -> None:
        true_sigma = 0.23
        true_r = 0.03
        true_q = 0.01

        spots = []
        strikes = []
        maturities = []
        is_call = []
        market_values = []

        for maturity in (0.25, 0.5, 1.0):
            for strike in (85.0, 95.0, 100.0, 105.0, 115.0):
                for call in (True, False):
                    spots.append(100.0)
                    strikes.append(strike)
                    maturities.append(maturity)
                    is_call.append(call)
                    market_values.append(
                        self._bs_price(100.0, strike, maturity, true_sigma, true_r, true_q, call)
                    )

        result = ql.calibrate_black_scholes(
            spots=spots,
            strikes=strikes,
            maturities=maturities,
            is_call=is_call,
            market_values=market_values,
            target="price",
            initial_sigma=0.15,
            r=true_r,
            q=true_q,
            method="hybrid",
            objective="least_squares",
            max_generations=180,
            population_size=40,
            nm_max_iterations=200,
            tolerance=1.0e-6,
            seed=7,
        )

        self.assertAlmostEqual(result["model"]["sigma"], true_sigma, delta=1.5e-2)
        self.assertEqual(result["model"]["r"], true_r)
        self.assertEqual(result["model"]["q"], true_q)
        self.assertLess(result["rmse"], 2.0e-2)

    def test_calibrate_sabr_surface(self) -> None:
        true_alpha = 0.35
        true_beta = 0.60
        true_rho = -0.30
        true_nu = 0.80

        forwards = []
        strikes = []
        maturities = []
        market_vols = []

        for maturity in (0.5, 1.0, 2.0):
            for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
                forwards.append(100.0)
                strikes.append(strike)
                maturities.append(maturity)
                market_vols.append(
                    self._sabr_implied_vol(
                        100.0,
                        strike,
                        maturity,
                        true_alpha,
                        true_beta,
                        true_rho,
                        true_nu,
                    )
                )

        result = ql.calibrate_sabr(
            forwards=forwards,
            strikes=strikes,
            maturities=maturities,
            market_vols=market_vols,
            initial_alpha=0.20,
            initial_beta=0.60,
            initial_rho=-0.10,
            initial_nu=0.50,
            beta_bounds=(0.60, 0.60),
            method="hybrid",
            objective="relative_least_squares",
            max_generations=180,
            population_size=42,
            nm_max_iterations=220,
            tolerance=1.0e-6,
            seed=11,
        )

        self.assertAlmostEqual(result["model"]["alpha"], true_alpha, delta=3.0e-2)
        self.assertAlmostEqual(result["model"]["beta"], true_beta, delta=1.0e-12)
        self.assertAlmostEqual(result["model"]["rho"], true_rho, delta=3.0e-2)
        self.assertAlmostEqual(result["model"]["nu"], true_nu, delta=4.0e-2)
        self.assertLess(result["rmse"], 1.0e-3)

    def test_calibrate_heston_prices_partial(self) -> None:
        true_model = ql.HestonModel(1.8, 0.05, 0.45, -0.65, 0.04, 0.02, 0.0)

        spots = []
        strikes = []
        maturities = []
        is_call = []
        market_prices = []

        for maturity in (0.5, 1.0):
            expiry_date = self._expiry(maturity)
            for strike in (90.0, 100.0, 110.0):
                for call in (True, False):
                    spots.append(100.0)
                    strikes.append(strike)
                    maturities.append(maturity)
                    is_call.append(call)
                    market_prices.append(
                        ql.price_european_fft_heston(
                            spot=100.0,
                            strike=strike,
                            expiry_date=expiry_date,
                            option_type="call" if call else "put",
                            qty=1,
                            kappa=true_model.kappa,
                            theta=true_model.theta,
                            sigma=true_model.sigma,
                            rho=true_model.rho,
                            v0=true_model.v0,
                            r=true_model.r,
                            q=true_model.q,
                            valuation_date=self.VALUATION_DATE.isoformat(),
                            n=512,
                            b=120.0,
                            interp="cubic",
                        )
                    )

        result = ql.calibrate_heston(
            spots=spots,
            strikes=strikes,
            maturities=maturities,
            is_call=is_call,
            market_prices=market_prices,
            initial_kappa=1.8,
            initial_theta=0.05,
            initial_sigma=0.70,
            initial_rho=-0.20,
            initial_v0=0.08,
            r=0.02,
            q=0.0,
            kappa_bounds=(1.8, 1.8),
            theta_bounds=(0.05, 0.05),
            method="nelder_mead",
            objective="least_squares",
            fft_n=512,
            fft_b=120.0,
            fft_interp="cubic",
            nm_max_iterations=140,
            tolerance=1.0e-6,
            seed=19,
        )

        self.assertTrue(math.isfinite(result["objective_value"]))
        self.assertAlmostEqual(result["model"]["sigma"], 0.45, delta=1.0e-1)
        self.assertAlmostEqual(result["model"]["rho"], -0.65, delta=8.0e-2)
        self.assertAlmostEqual(result["model"]["v0"], 0.04, delta=2.0e-2)


if __name__ == "__main__":
    unittest.main()
