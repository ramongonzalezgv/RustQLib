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


class TestCosEngineBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if ql is None:
            raise unittest.SkipTest(f"quantlib_py import failed: {IMPORT_ERROR}")

    def spot_option(self, strike: float, option_type: str = "call"):
        return ql.EuropeanOptionSpot(100.0, strike, EXPIRY_DATE, VALUATION_DATE, option_type)

    def test_defaults_and_validation(self) -> None:
        engine = ql.CosEngine()
        self.assertEqual(engine.n, 256)
        self.assertEqual(engine.l, 12.0)
        self.assertEqual(repr(engine), "CosEngine(n=256, l=12)")
        with self.assertRaises(ValueError):
            ql.CosEngine(n=1)
        with self.assertRaises(ValueError):
            ql.CosEngine(l=0.0)

    def test_matches_fft_for_fourier_models(self) -> None:
        cos, fft = ql.CosEngine(), ql.FFTEngine()
        option = self.spot_option(105.0)
        cases = [
            ("heston", ql.HestonModel(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0)),
            ("bates", ql.BatesModel(2.0, 0.04, 0.6, -0.7, 0.04, 0.02, 0.01, 0.3, -0.1, 0.2)),
            ("kou", ql.KouModel(0.2, 0.03, 0.01, 0.4, 0.35, 12.0, 7.0)),
            ("bs", ql.BlackScholesModel(0.2, 0.03, 0.01)),
        ]
        for name, model in cases:
            with self.subTest(model=name):
                cos_price = getattr(cos, f"price_european_{name}")(option, model)
                fft_price = getattr(fft, f"price_european_{name}")(option, model)
                self.assertAlmostEqual(cos_price, fft_price, delta=1e-4)

    def test_batch_matches_single_prices(self) -> None:
        cos = ql.CosEngine()
        model = ql.HestonModel(2.0, 0.04, 0.6, -0.7, 0.04, 0.01, 0.0)
        options = [self.spot_option(k, "put") for k in (80.0, 100.0, 120.0)]
        batch = cos.price_european_heston_batch(options, model, parallel_batch=True)
        singles = [cos.price_european_heston(o, model) for o in options]
        for b, s in zip(batch, singles):
            self.assertAlmostEqual(b, s, delta=1e-12)

    def test_black_forward(self) -> None:
        cos, fft = ql.CosEngine(), ql.FFTEngine()
        option = ql.EuropeanOptionForward(102.0, 100.0, EXPIRY_DATE, VALUATION_DATE, "call")
        model = ql.BlackModel(0.25, 0.03)
        self.assertAlmostEqual(
            cos.price_european_black(option, model),
            fft.price_european_black(option, model),
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
