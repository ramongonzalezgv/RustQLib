import unittest

try:
    import quantlib_py as ql
except Exception as exc:  # pragma: no cover - handled by SkipTest
    ql = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class TestHestonMonteCarloScheme(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if ql is None:
            raise unittest.SkipTest(f"quantlib_py import failed: {IMPORT_ERROR}")

    def test_default_is_qe_and_scheme_is_selectable(self) -> None:
        self.assertEqual(ql.MonteCarloEngine().heston_scheme, "qe")
        engine = ql.MonteCarloEngine(heston_scheme="full_truncation")
        self.assertEqual(engine.heston_scheme, "full_truncation")
        self.assertIn("heston_scheme='full_truncation'", repr(engine))
        with self.assertRaises(ValueError):
            ql.MonteCarloEngine(heston_scheme="absorption")

    def test_qe_matches_fourier_when_feller_fails(self) -> None:
        # 2*kappa*theta = 0.12 < sigma^2 = 1.
        model = ql.HestonModel(1.5, 0.04, 1.0, -0.9, 0.04, 0.02, 0.0)
        option = ql.EuropeanOptionSpot(100.0, 100.0, "2027-02-10", "2026-02-10", "call")
        reference = ql.CosEngine(n=1024, l=16.0).price_european_heston(option, model)
        mc = ql.MonteCarloEngine(n_paths=200_000, n_steps=32, seed=7)
        price = mc.price_european_heston(option, model)
        # Loose bound: ~5 standard errors at 200k paths.
        self.assertAlmostEqual(price, reference, delta=0.08)


if __name__ == "__main__":
    unittest.main()
