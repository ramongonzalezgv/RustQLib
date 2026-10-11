import unittest

try:
    import quantlib_py as ql
except Exception as exc:  # pragma: no cover - handled by SkipTest
    ql = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


VALUATION_DATE = "2026-02-10"
EXPIRY_DATE = "2027-02-10"


class TestMonteCarloStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if ql is None:
            raise unittest.SkipTest(f"quantlib_py import failed: {IMPORT_ERROR}")

    def test_price_with_stats_matches_price_and_brackets_analytic(self) -> None:
        option = ql.EuropeanOptionSpot(100.0, 100.0, EXPIRY_DATE, VALUATION_DATE, "call")
        model = ql.BlackScholesModel(0.2, 0.05, 0.01)
        engine = ql.MonteCarloEngine(n_paths=100_000, n_steps=1, seed=3)

        estimate = engine.price_european_bs_with_stats(option, model)
        self.assertEqual(estimate.n_paths, 100_000)
        self.assertGreater(estimate.std_error, 0.0)
        self.assertEqual(estimate.price, engine.price_european_bs(option, model))
        self.assertIn("McEstimate(price=", repr(estimate))

        analytic = ql.AnalyticalEngine().price_european_bs(option, model)
        self.assertLess(abs(estimate.price - analytic), 4.0 * estimate.std_error)

    def test_variance_reduction_flags(self) -> None:
        option = ql.AsianOptionSpot(
            100.0, 100.0, EXPIRY_DATE, VALUATION_DATE, "call", 1, "arithmetic"
        )
        model = ql.BlackScholesModel(0.2, 0.05, 0.01)
        plain = ql.MonteCarloEngine(n_paths=20_000, n_steps=52, seed=1)
        controlled = ql.MonteCarloEngine(
            n_paths=20_000, n_steps=52, seed=1, antithetic=True, control_variate=True
        )
        self.assertTrue(controlled.antithetic)
        self.assertTrue(controlled.control_variate)
        self.assertIn("control_variate=true", repr(controlled))

        a = plain.price_asian_bs_with_stats(option, model)
        b = controlled.price_asian_bs_with_stats(option, model)
        self.assertLess(b.std_error * 10.0, a.std_error)
        self.assertLess(abs(a.price - b.price), 3.0 * (a.std_error**2 + b.std_error**2) ** 0.5)


if __name__ == "__main__":
    unittest.main()
