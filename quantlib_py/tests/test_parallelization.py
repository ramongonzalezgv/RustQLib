import os
import platform
import statistics
import sys
import time
import unittest

import numpy as np

try:
    import quantlib_py as ql
except Exception as exc:  # pragma: no cover - handled by SkipTest
    ql = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


VALUATION_DATE = "2026-02-11"


class TestPythonBindingsParallelization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if ql is None:
            raise unittest.SkipTest(f"quantlib_py import failed: {IMPORT_ERROR}")

        cls.cpu_count = os.cpu_count() or 1
        cls.cpu_affinity = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        )
        cls.rayon_threads = os.environ.get("RAYON_NUM_THREADS")
        cls.omp_threads = os.environ.get("OMP_NUM_THREADS")
        cls.mkl_threads = os.environ.get("MKL_NUM_THREADS")
        cls.openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS")
        cls.numexpr_threads = os.environ.get("NUMEXPR_NUM_THREADS")
        # Shared CI runners are too noisy for wall-clock ratios; prices are still checked there.
        cls.can_assert_speedup = (
            cls.cpu_count >= 2 and cls.rayon_threads != "1" and not os.environ.get("CI")
        )

        cls._print_section(
            "parallelization-tests environment",
            [
                ("python", sys.version.split()[0]),
                ("numpy", np.__version__),
                ("platform", platform.platform()),
                ("module", getattr(ql, "__file__", "<unknown>")),
                ("pid", str(os.getpid())),
                ("cpu_count", str(cls.cpu_count)),
                ("cpu_affinity", str(cls.cpu_affinity) if cls.cpu_affinity else "n/a"),
                ("RAYON_NUM_THREADS", cls.rayon_threads or "unset"),
                ("OMP_NUM_THREADS", cls.omp_threads or "unset"),
                ("MKL_NUM_THREADS", cls.mkl_threads or "unset"),
                ("OPENBLAS_NUM_THREADS", cls.openblas_threads or "unset"),
                ("NUMEXPR_NUM_THREADS", cls.numexpr_threads or "unset"),
                (
                    "speedup_assertions",
                    "enabled" if cls.can_assert_speedup else "disabled",
                ),
            ],
        )

    @staticmethod
    def _run_and_time(fn, *, repeat: int = 1, warmup: int = 0):
        for _ in range(warmup):
            fn()

        times = []
        last_value = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            last_value = fn()
            times.append(time.perf_counter() - t0)

        return last_value, statistics.median(times), times

    @staticmethod
    def _assert_prices_match(reference: np.ndarray, candidate: np.ndarray) -> None:
        np.testing.assert_allclose(reference, candidate, rtol=0.0, atol=1.0e-12)

    @staticmethod
    def _speedup(baseline: float, candidate: float) -> float:
        if candidate <= 0.0:
            return float("inf")
        return baseline / candidate

    @classmethod
    def _fmt_speedup(cls, baseline: float, candidate: float) -> str:
        ratio = cls._speedup(baseline, candidate)
        if baseline <= 0.0:
            return f"{ratio:.3f}x"
        faster_pct = (1.0 - (candidate / baseline)) * 100.0
        return f"{ratio:.3f}x ({faster_pct:.1f}% faster)"

    @staticmethod
    def _print_section(title: str, rows: list[tuple[str, str]]) -> None:
        max_label = max(len(label) for label, _ in rows)
        print(f"\n[{title}]")
        for label, value in rows:
            print(f"  {label:<{max_label}} : {value}")

    @staticmethod
    def _fmt_int(value: int) -> str:
        return f"{value:,}"

    @staticmethod
    def _fmt_rate(units: float) -> str:
        return f"{units:,.0f}/s"

    def test_analytical_bs_numpy_batch_parallel_matches_and_is_faster(self) -> None:
        n_options = 1_000_000
        spots = np.linspace(80.0, 120.0, n_options, dtype=np.float64)
        strikes = np.linspace(90.0, 110.0, n_options, dtype=np.float64)
        expiry_days = np.full(n_options, 365, dtype=np.int64)
        is_call = (np.arange(n_options) % 2) == 0
        qtys = np.ones(n_options, dtype=np.int32)
        valuation_date = VALUATION_DATE
        repeat = 2
        warmup = 1

        self._print_section(
            "Analytical BS NumPy Batch - workload",
            [
                ("valuation_date", valuation_date),
                ("n_options", self._fmt_int(n_options)),
                ("spots_range", f"{spots[0]:.2f} -> {spots[-1]:.2f}"),
                ("strikes_range", f"{strikes[0]:.2f} -> {strikes[-1]:.2f}"),
                ("expiry_days", f"{int(expiry_days.min())} -> {int(expiry_days.max())}"),
                ("is_call_true_ratio", f"{float(np.mean(is_call)):.2%}"),
                ("qtys_total", self._fmt_int(int(qtys.sum()))),
                ("input_dtypes", "spots/strikes=float64, expiry_days=int64, is_call=bool, qtys=int32"),
                (
                    "pricing_work",
                    f"{self._fmt_int(n_options)} closed-form prices per timed run",
                ),
                (
                    "timing_plan",
                    f"serial {repeat} runs (+{warmup} warmup), "
                    f"parallel {repeat} runs (+{warmup} warmup)",
                ),
            ],
        )

        def price_serial() -> np.ndarray:
            return np.asarray(
                ql.price_european_analytical_bs_batch_numpy(
                    spots,
                    strikes,
                    expiry_days,
                    is_call,
                    qtys,
                    valuation_date,
                    0.2,
                    0.03,
                    0.0,
                    False,
                )
            )

        def price_parallel() -> np.ndarray:
            return np.asarray(
                ql.price_european_analytical_bs_batch_numpy(
                    spots,
                    strikes,
                    expiry_days,
                    is_call,
                    qtys,
                    valuation_date,
                    0.2,
                    0.03,
                    0.0,
                    True,
                )
            )

        serial_prices, serial_median, serial_times = self._run_and_time(
            price_serial, repeat=repeat, warmup=warmup
        )
        parallel_prices, parallel_median, parallel_times = self._run_and_time(
            price_parallel, repeat=repeat, warmup=warmup
        )

        self._assert_prices_match(serial_prices, parallel_prices)

        speedup = self._speedup(serial_median, parallel_median)
        serial_options_per_sec = n_options / serial_median
        parallel_options_per_sec = n_options / parallel_median
        self._print_section(
            "Analytical BS NumPy Batch",
            [
                (
                    "serial",
                    f"median={serial_median:.4f}s "
                    f"(runs={', '.join(f'{t:.4f}s' for t in serial_times)})",
                ),
                (
                    "parallel",
                    f"median={parallel_median:.4f}s "
                    f"(runs={', '.join(f'{t:.4f}s' for t in parallel_times)})",
                ),
                (
                    "throughput_serial",
                    f"{self._fmt_rate(serial_options_per_sec)} options",
                ),
                (
                    "throughput_parallel",
                    f"{self._fmt_rate(parallel_options_per_sec)} options",
                ),
                ("speedup", self._fmt_speedup(serial_median, parallel_median)),
            ],
        )

        if self.can_assert_speedup:
            self.assertGreaterEqual(
                speedup,
                1.05,
                msg=(
                    "Expected analytical parallel mode to be measurably faster.\n"
                    f"serial_times={serial_times}, parallel_times={parallel_times}, "
                    f"speedup={speedup:.3f}x"
                ),
            )

    def test_monte_carlo_asian_bs_numpy_parallel_paths_and_batch(self) -> None:
        n_options = 6
        spots = np.linspace(90.0, 110.0, n_options, dtype=np.float64)
        strikes = np.linspace(92.0, 108.0, n_options, dtype=np.float64)
        expiry_days = np.full(n_options, 365, dtype=np.int64)
        is_call = (np.arange(n_options) % 2) == 0
        qtys = np.ones(n_options, dtype=np.int32)

        base_kwargs = dict(
            spots=spots,
            strikes=strikes,
            expiry_days=expiry_days,
            is_call=is_call,
            qtys=qtys,
            valuation_date=VALUATION_DATE,
            averaging_type="arithmetic",
            sigma=0.22,
            r=0.03,
            q=0.0,
            n_paths=30_000,
            n_steps=252,
            seed=2026,
            chunk_size=4_096,
        )
        total_paths = n_options * base_kwargs["n_paths"]
        total_path_steps = total_paths * base_kwargs["n_steps"]

        self._print_section(
            "Monte Carlo Asian BS NumPy Batch - workload",
            [
                ("valuation_date", base_kwargs["valuation_date"]),
                ("n_options", self._fmt_int(n_options)),
                ("n_paths_per_option", self._fmt_int(base_kwargs["n_paths"])),
                ("n_steps_per_path", self._fmt_int(base_kwargs["n_steps"])),
                ("seed", str(base_kwargs["seed"])),
                ("chunk_size", self._fmt_int(base_kwargs["chunk_size"])),
                ("spots_range", f"{spots[0]:.2f} -> {spots[-1]:.2f}"),
                ("strikes_range", f"{strikes[0]:.2f} -> {strikes[-1]:.2f}"),
                ("is_call_true_ratio", f"{float(np.mean(is_call)):.2%}"),
                ("averaging_type", base_kwargs["averaging_type"]),
                ("parallel_modes", "serial, parallel_paths, parallel_batch, paths+batch"),
                ("total_paths_per_run", self._fmt_int(total_paths)),
                ("total_path_steps_per_run", self._fmt_int(total_path_steps)),
                (
                    "normal_draws_per_run",
                    f"{self._fmt_int(total_path_steps)} (1 normal per step for BS paths)",
                ),
                ("timing_plan", "1 timed run per mode (+tiny warmup for thread-pool setup)"),
            ],
        )

        # Warm up to avoid one-time thread-pool initialization noise in timings.
        ql.price_asian_monte_carlo_bs_batch_numpy(
            parallel_paths=False,
            parallel_batch=False,
            **{**base_kwargs, "n_paths": 2_000, "n_steps": 16},
        )
        ql.price_asian_monte_carlo_bs_batch_numpy(
            parallel_paths=True,
            parallel_batch=True,
            **{**base_kwargs, "n_paths": 2_000, "n_steps": 16},
        )

        def price(parallel_paths: bool, parallel_batch: bool) -> np.ndarray:
            return np.asarray(
                ql.price_asian_monte_carlo_bs_batch_numpy(
                    parallel_paths=parallel_paths,
                    parallel_batch=parallel_batch,
                    **base_kwargs,
                )
            )

        serial_prices, serial_time, _ = self._run_and_time(
            lambda: price(False, False), repeat=1
        )
        path_prices, path_time, _ = self._run_and_time(lambda: price(True, False), repeat=1)
        batch_prices, batch_time, _ = self._run_and_time(lambda: price(False, True), repeat=1)
        full_prices, full_time, _ = self._run_and_time(lambda: price(True, True), repeat=1)

        self._assert_prices_match(serial_prices, path_prices)
        self._assert_prices_match(serial_prices, batch_prices)
        self._assert_prices_match(serial_prices, full_prices)

        self._print_section(
            "Monte Carlo Asian BS NumPy Batch",
            [
                (
                    "serial",
                    f"{serial_time:.4f}s, "
                    f"paths={self._fmt_rate(total_paths / serial_time)}, "
                    f"path_steps={self._fmt_rate(total_path_steps / serial_time)}",
                ),
                (
                    "parallel_paths",
                    f"{path_time:.4f}s, {self._fmt_speedup(serial_time, path_time)} vs serial, "
                    f"paths={self._fmt_rate(total_paths / path_time)}, "
                    f"path_steps={self._fmt_rate(total_path_steps / path_time)}",
                ),
                (
                    "parallel_batch",
                    f"{batch_time:.4f}s, {self._fmt_speedup(serial_time, batch_time)} vs serial, "
                    f"paths={self._fmt_rate(total_paths / batch_time)}, "
                    f"path_steps={self._fmt_rate(total_path_steps / batch_time)}",
                ),
                (
                    "paths+batch",
                    f"{full_time:.4f}s, {self._fmt_speedup(serial_time, full_time)} vs serial, "
                    f"paths={self._fmt_rate(total_paths / full_time)}, "
                    f"path_steps={self._fmt_rate(total_path_steps / full_time)}",
                ),
            ],
        )

        if self.can_assert_speedup:
            full_speedup = serial_time / full_time
            self.assertGreaterEqual(
                full_speedup,
                1.30,
                msg=(
                    "Expected Monte Carlo Asian full parallel mode to be clearly faster.\n"
                    f"serial={serial_time:.4f}s, path_only={path_time:.4f}s, "
                    f"batch_only={batch_time:.4f}s, full={full_time:.4f}s, "
                    f"speedup={full_speedup:.3f}x"
                ),
            )
            self.assertLessEqual(
                full_time,
                1.25 * min(path_time, batch_time),  # timing noise on shared CI runners
                msg=(
                    "Expected full parallel mode (paths+batch) to be no slower (25% slack) than "
                    f"individual parallel modes.\npath_only={path_time:.4f}s, "
                    f"batch_only={batch_time:.4f}s, full={full_time:.4f}s"
                ),
            )

    def test_monte_carlo_heston_numpy_parallel_matches_and_is_faster(self) -> None:
        n_options = 5
        spots = np.linspace(90.0, 110.0, n_options, dtype=np.float64)
        strikes = np.linspace(92.0, 108.0, n_options, dtype=np.float64)
        expiry_days = np.full(n_options, 365, dtype=np.int64)
        is_call = (np.arange(n_options) % 2) == 0
        qtys = np.ones(n_options, dtype=np.int32)

        base_kwargs = dict(
            spots=spots,
            strikes=strikes,
            expiry_days=expiry_days,
            is_call=is_call,
            qtys=qtys,
            valuation_date=VALUATION_DATE,
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=-0.7,
            v0=0.04,
            r=0.03,
            q=0.0,
            n_paths=15_000,
            n_steps=252,
            seed=4_321,
            chunk_size=4_096,
        )
        total_paths = n_options * base_kwargs["n_paths"]
        total_path_steps = total_paths * base_kwargs["n_steps"]
        total_normals = total_path_steps * 2

        self._print_section(
            "Monte Carlo Heston NumPy Batch - workload",
            [
                ("valuation_date", base_kwargs["valuation_date"]),
                ("n_options", self._fmt_int(n_options)),
                ("n_paths_per_option", self._fmt_int(base_kwargs["n_paths"])),
                ("n_steps_per_path", self._fmt_int(base_kwargs["n_steps"])),
                ("seed", str(base_kwargs["seed"])),
                ("chunk_size", self._fmt_int(base_kwargs["chunk_size"])),
                ("spots_range", f"{spots[0]:.2f} -> {spots[-1]:.2f}"),
                ("strikes_range", f"{strikes[0]:.2f} -> {strikes[-1]:.2f}"),
                ("is_call_true_ratio", f"{float(np.mean(is_call)):.2%}"),
                (
                    "heston_params",
                    "kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, r=0.03, q=0.0",
                ),
                ("parallel_modes", "serial vs paths+batch"),
                ("total_paths_per_run", self._fmt_int(total_paths)),
                ("total_path_steps_per_run", self._fmt_int(total_path_steps)),
                (
                    "normal_draws_per_run",
                    f"{self._fmt_int(total_normals)} "
                    "(2 normals per Heston step)",
                ),
                ("timing_plan", "1 timed run per mode (+tiny warmup for thread-pool setup)"),
            ],
        )

        ql.price_european_monte_carlo_heston_batch_numpy(
            parallel_paths=False,
            parallel_batch=False,
            **{**base_kwargs, "n_paths": 2_000, "n_steps": 16},
        )
        ql.price_european_monte_carlo_heston_batch_numpy(
            parallel_paths=True,
            parallel_batch=True,
            **{**base_kwargs, "n_paths": 2_000, "n_steps": 16},
        )

        def price(parallel_paths: bool, parallel_batch: bool) -> np.ndarray:
            return np.asarray(
                ql.price_european_monte_carlo_heston_batch_numpy(
                    parallel_paths=parallel_paths,
                    parallel_batch=parallel_batch,
                    **base_kwargs,
                )
            )

        serial_prices, serial_time, _ = self._run_and_time(
            lambda: price(False, False), repeat=1
        )
        parallel_prices, parallel_time, _ = self._run_and_time(
            lambda: price(True, True), repeat=1
        )

        self._assert_prices_match(serial_prices, parallel_prices)

        speedup = self._speedup(serial_time, parallel_time)
        self._print_section(
            "Monte Carlo Heston NumPy Batch",
            [
                (
                    "serial",
                    f"{serial_time:.4f}s, "
                    f"paths={self._fmt_rate(total_paths / serial_time)}, "
                    f"path_steps={self._fmt_rate(total_path_steps / serial_time)}, "
                    f"normals={self._fmt_rate(total_normals / serial_time)}",
                ),
                (
                    "parallel",
                    f"{parallel_time:.4f}s, "
                    f"paths={self._fmt_rate(total_paths / parallel_time)}, "
                    f"path_steps={self._fmt_rate(total_path_steps / parallel_time)}, "
                    f"normals={self._fmt_rate(total_normals / parallel_time)}",
                ),
                ("speedup", self._fmt_speedup(serial_time, parallel_time)),
            ],
        )

        if self.can_assert_speedup:
            self.assertGreaterEqual(
                speedup,
                1.20,
                msg=(
                    "Expected Monte Carlo Heston parallel mode to be faster.\n"
                    f"serial={serial_time:.4f}s, parallel={parallel_time:.4f}s, "
                    f"speedup={speedup:.3f}x"
                ),
            )


if __name__ == "__main__":
    unittest.main()
