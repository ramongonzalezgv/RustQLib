#!/usr/bin/env python3
"""
Full quantlib_py walkthrough script.

This file is intentionally verbose and print-heavy so users can run it and
understand:
1) What each API expects
2) What each API returns
3) How class-based and function-based interfaces relate to each other

Run (from repo root):
    cd quantlib_py
    maturin develop
    cd ..
    python quantlib_py/notebooks/quantlib_py_full_walkthrough.py
"""

from __future__ import annotations

import pprint

import numpy as np

try:
    import quantlib_py as ql
except Exception as exc:  # pragma: no cover - this is setup guidance for users
    raise RuntimeError(
        "Could not import quantlib_py.\n"
        "Build/install the extension first:\n"
        "  cd quantlib_py && maturin develop\n"
        "Then run this script again."
    ) from exc


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def subsection(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def show(label: str, value) -> None:
    print(f"{label}: {value}")


def show_vector(label: str, values) -> None:
    print(f"{label} (len={len(values)}):")
    for i, v in enumerate(values):
        print(f"  [{i}] {v}")


def show_numpy(label: str, arr: np.ndarray) -> None:
    print(f"{label}:")
    print(f"  dtype={arr.dtype}, shape={arr.shape}")
    print(f"  values={arr}")


def show_calibration_result(label: str, result: dict) -> None:
    subsection(label)
    show("success", result["success"])
    show("message", result["message"])
    show("iterations", result["iterations"])
    show("objective_value", result["objective_value"])
    show("rmse", result["rmse"])
    show("mape", result["mape"])
    show("max_error", result["max_error"])
    show("calibrated model", result["model"])


def sabr_hagan_implied_vol(forward: float, strike: float, maturity: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan-style SABR approximation used only to generate synthetic calibration targets.
    This keeps the walkthrough self-contained.
    """
    if abs(forward - strike) < 1.0e-10:
        fk_power = forward ** (1.0 - beta)
        correction = (
            (((1.0 - beta) ** 2) / 24.0) * (alpha * alpha) / (fk_power * fk_power)
            + rho * beta * nu * alpha / (4.0 * fk_power)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * maturity
        return alpha / fk_power * (1.0 + correction)

    z = (nu / alpha) * ((forward * strike) ** ((1.0 - beta) / 2.0)) * np.log(forward / strike)
    discriminant = max(0.0, 1.0 - 2.0 * rho * z + z * z)
    numerator = np.sqrt(discriminant) + z - rho
    denominator = 1.0 - rho
    chi_z = np.log(numerator / denominator) if numerator > 0.0 and abs(denominator) > 1.0e-12 else z

    if abs(chi_z) < 1.0e-12:
        return alpha

    fk_power = (forward * strike) ** ((1.0 - beta) / 2.0)
    log_fk = np.log(forward / strike)
    main_term = alpha / fk_power * z / chi_z
    term1 = (
        (((1.0 - beta) ** 2) / 24.0) * (alpha * alpha) / (fk_power * fk_power)
        + rho * beta * nu * alpha / (4.0 * fk_power)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * maturity
    term2 = ((1.0 - beta) ** 2) * log_fk * log_fk / 24.0 + ((1.0 - beta) ** 4) * (log_fk**4) / 1920.0
    return main_term * (1.0 + term1 + term2)


def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    section("0) Setup: shared market data and input vectors")
    valuation_date = "2026-01-02"
    expiry_dates = ["2026-04-02", "2026-07-02", "2026-10-02", "2027-01-02"]
    option_types = ["call", "put", "call", "put"]
    is_call = [True, False, True, False]
    qtys = [1, 2, 1, 3]
    spots = [95.0, 100.0, 105.0, 110.0]
    forwards = [96.0, 101.0, 106.0, 111.0]
    strikes = [100.0, 100.0, 100.0, 100.0]

    # Black-Scholes parameters
    sigma_bs = 0.20
    r_bs = 0.03
    q_bs = 0.01

    # Black (forward) parameters
    sigma_black = 0.22
    r_black = 0.03

    # Heston parameters
    kappa = 1.5
    theta = 0.04
    sigma_heston = 0.35
    rho = -0.65
    v0 = 0.04
    r_heston = 0.03
    q_heston = 0.01

    # Fast runtimes for demonstration. Increase these for production-grade MC accuracy.
    mc_n_paths = 5_000
    mc_n_steps = 64
    mc_seed = 42
    mc_chunk_size = 1_024

    # FFT settings. Larger n gives more resolution but takes longer.
    fft_n = 2_048
    fft_b = 200.0
    fft_interp = "cubic"

    show("valuation_date", valuation_date)
    show("expiry_dates", expiry_dates)
    show("option_types", option_types)
    show("is_call", is_call)
    show("qtys", qtys)
    show("spots", spots)
    show("forwards", forwards)
    show("strikes", strikes)

    section("1) Class constructors and getters")
    subsection("1.1 Models")
    bs_model = ql.BlackScholesModel(sigma=sigma_bs, r=r_bs, q=q_bs)
    black_model = ql.BlackModel(sigma=sigma_black, r=r_black)
    heston_model = ql.HestonModel(
        kappa=kappa, theta=theta, sigma=sigma_heston, rho=rho, v0=v0, r=r_heston, q=q_heston
    )

    show("BlackScholesModel repr", bs_model)
    show("BlackScholesModel.sigma", bs_model.sigma)
    show("BlackScholesModel.r", bs_model.r)
    show("BlackScholesModel.q", bs_model.q)

    show("BlackModel repr", black_model)
    show("BlackModel.sigma", black_model.sigma)
    show("BlackModel.r", black_model.r)

    show("HestonModel repr", heston_model)
    show("HestonModel.kappa", heston_model.kappa)
    show("HestonModel.theta", heston_model.theta)
    show("HestonModel.sigma", heston_model.sigma)
    show("HestonModel.rho", heston_model.rho)
    show("HestonModel.v0", heston_model.v0)
    show("HestonModel.r", heston_model.r)
    show("HestonModel.q", heston_model.q)

    # Bates model (Heston with log-normal jumps)
    bates_model = ql.BatesModel(
        kappa=kappa,
        theta=theta,
        sigma=sigma_heston,
        rho=rho,
        v0=v0,
        r=r_heston,
        q=q_heston,
        lambda_jump=0.5,
        mu_j=0.0,
        sigma_j=0.2,
    )

    show("BatesModel repr", bates_model)
    show("BatesModel.kappa", bates_model.kappa)
    show("BatesModel.theta", bates_model.theta)
    show("BatesModel.sigma", bates_model.sigma)
    show("BatesModel.rho", bates_model.rho)
    show("BatesModel.v0", bates_model.v0)
    show("BatesModel.r", bates_model.r)
    show("BatesModel.q", bates_model.q)
    show("BatesModel.lambda_jump", bates_model.lambda_jump)
    show("BatesModel.mu_j", bates_model.mu_j)
    show("BatesModel.sigma_j", bates_model.sigma_j)

    # Kou model (pure jump-diffusion with double exponential jumps)
    kou_model = ql.KouModel(
        sigma=sigma_bs, r=r_bs, q=q_bs, lambda_jump=0.5, p_up=0.4, eta1=5.0, eta2=10.0
    )

    show("KouModel repr", kou_model)
    show("KouModel.sigma", kou_model.sigma)
    show("KouModel.r", kou_model.r)
    show("KouModel.q", kou_model.q)
    show("KouModel.lambda_jump", kou_model.lambda_jump)
    show("KouModel.p_up", kou_model.p_up)
    show("KouModel.eta1", kou_model.eta1)
    show("KouModel.eta2", kou_model.eta2)

    subsection("1.2 Products")
    european_spot = ql.EuropeanOptionSpot(
        spot=100.0,
        strike=100.0,
        expiry_date="2026-07-02",
        valuation_date=valuation_date,
        option_type="call",
        qty=1,
    )
    european_forward = ql.EuropeanOptionForward(
        forward=101.0,
        strike=100.0,
        expiry_date="2026-07-02",
        valuation_date=valuation_date,
        option_type="put",
        qty=2,
    )
    asian_spot = ql.AsianOptionSpot(
        spot=100.0,
        strike=100.0,
        expiry_date="2026-07-02",
        valuation_date=valuation_date,
        option_type="call",
        qty=1,
        averaging_type="arithmetic",
    )

    show("EuropeanOptionSpot repr", european_spot)
    show("EuropeanOptionSpot.spot", european_spot.spot)
    show("EuropeanOptionSpot.strike", european_spot.strike)
    show("EuropeanOptionSpot.option_type", european_spot.option_type)
    show("EuropeanOptionSpot.qty", european_spot.qty)
    show("EuropeanOptionSpot.expiry_date", european_spot.expiry_date)
    show("EuropeanOptionSpot.valuation_date", european_spot.valuation_date)
    show("EuropeanOptionSpot.days_to_maturity()", european_spot.days_to_maturity())
    show("EuropeanOptionSpot.time_to_maturity()", european_spot.time_to_maturity())

    show("EuropeanOptionForward repr", european_forward)
    show("EuropeanOptionForward.forward", european_forward.forward)
    show("EuropeanOptionForward.strike", european_forward.strike)
    show("EuropeanOptionForward.option_type", european_forward.option_type)
    show("EuropeanOptionForward.qty", european_forward.qty)
    show("EuropeanOptionForward.expiry_date", european_forward.expiry_date)
    show("EuropeanOptionForward.valuation_date", european_forward.valuation_date)
    show("EuropeanOptionForward.days_to_maturity()", european_forward.days_to_maturity())
    show("EuropeanOptionForward.time_to_maturity()", european_forward.time_to_maturity())

    show("AsianOptionSpot repr", asian_spot)
    show("AsianOptionSpot.spot", asian_spot.spot)
    show("AsianOptionSpot.strike", asian_spot.strike)
    show("AsianOptionSpot.option_type", asian_spot.option_type)
    show("AsianOptionSpot.qty", asian_spot.qty)
    show("AsianOptionSpot.averaging_type", asian_spot.averaging_type)
    show("AsianOptionSpot.expiry_date", asian_spot.expiry_date)
    show("AsianOptionSpot.valuation_date", asian_spot.valuation_date)
    show("AsianOptionSpot.days_to_maturity()", asian_spot.days_to_maturity())
    show("AsianOptionSpot.time_to_maturity()", asian_spot.time_to_maturity())

    # Prepare lists for class batch APIs.
    spot_options = [
        ql.EuropeanOptionSpot(spots[i], strikes[i], expiry_dates[i], valuation_date, option_types[i], qtys[i])
        for i in range(len(spots))
    ]
    forward_options = [
        ql.EuropeanOptionForward(
            forwards[i], strikes[i], expiry_dates[i], valuation_date, option_types[i], qtys[i]
        )
        for i in range(len(forwards))
    ]
    asian_options = [
        ql.AsianOptionSpot(
            spots[i], strikes[i], expiry_dates[i], valuation_date, option_types[i], qtys[i], "arithmetic"
        )
        for i in range(len(spots))
    ]

    section("2) Class engine APIs")
    subsection("2.1 AnalyticalEngine")
    analytical = ql.AnalyticalEngine()

    # Single-instrument BS pricing
    p_bs_single = analytical.price_european_bs(european_spot, bs_model)
    show("AnalyticalEngine.price_european_bs(single)", p_bs_single)

    # Alias of price_european_bs
    p_bs_single_alias = analytical.price_european(european_spot, bs_model)
    show("AnalyticalEngine.price_european(single alias)", p_bs_single_alias)

    # Batch BS (shared model)
    p_bs_batch = analytical.price_european_bs_batch(spot_options, bs_model, parallel=True)
    show_vector("AnalyticalEngine.price_european_bs_batch(parallel=True)", p_bs_batch)

    # Alias of price_european_bs_batch
    p_bs_batch_alias = analytical.price_european_batch(spot_options, bs_model, parallel=False)
    show_vector("AnalyticalEngine.price_european_batch(parallel=False alias)", p_bs_batch_alias)

    # Batch BS (one model per option)
    bs_models_per_option = [
        ql.BlackScholesModel(sigma=0.18 + 0.01 * i, r=r_bs + 0.001 * i, q=q_bs)
        for i in range(len(spot_options))
    ]
    p_bs_per_model = analytical.price_european_bs_batch_per_model(
        spot_options, bs_models_per_option, parallel=True
    )
    show_vector("AnalyticalEngine.price_european_bs_batch_per_model(parallel=True)", p_bs_per_model)

    # Alias of price_european_bs_batch_per_model
    p_bs_per_model_alias = analytical.price_european_batch_per_model(
        spot_options, bs_models_per_option, parallel=False
    )
    show_vector(
        "AnalyticalEngine.price_european_batch_per_model(parallel=False alias)", p_bs_per_model_alias
    )

    # Black (forward-quoted) single and batch
    p_black_single = analytical.price_european_black(european_forward, black_model)
    show("AnalyticalEngine.price_european_black(single)", p_black_single)

    p_black_batch = analytical.price_european_black_batch(forward_options, black_model, parallel=True)
    show_vector("AnalyticalEngine.price_european_black_batch(parallel=True)", p_black_batch)

    black_models_per_option = [
        ql.BlackModel(sigma=0.20 + 0.01 * i, r=r_black + 0.001 * i) for i in range(len(forward_options))
    ]
    p_black_per_model = analytical.price_european_black_batch_per_model(
        forward_options, black_models_per_option, parallel=False
    )
    show_vector(
        "AnalyticalEngine.price_european_black_batch_per_model(parallel=False)", p_black_per_model
    )

    subsection("2.2 FFTEngine")
    fft_engine = ql.FFTEngine(n=fft_n, b=fft_b, interp=fft_interp)
    show("FFTEngine repr", fft_engine)
    show("FFTEngine.n", fft_engine.n)
    show("FFTEngine.b", fft_engine.b)
    show("FFTEngine.interp", fft_engine.interp)

    # Single-instrument pricing for all supported model/product combinations.
    show("FFTEngine.price_european_bs(single)", fft_engine.price_european_bs(european_spot, bs_model))
    show(
        "FFTEngine.price_european_black(single)",
        fft_engine.price_european_black(european_forward, black_model),
    )
    show(
        "FFTEngine.price_european_heston(single)",
        fft_engine.price_european_heston(european_spot, heston_model),
    )
    show(
        "FFTEngine.price_european_bates(single)",
        fft_engine.price_european_bates(european_spot, bates_model),
    )
    show(
        "FFTEngine.price_european_kou(single)",
        fft_engine.price_european_kou(european_spot, kou_model),
    )

    # Batch pricing for all supported combinations.
    show_vector(
        "FFTEngine.price_european_bs_batch(parallel_batch=True)",
        fft_engine.price_european_bs_batch(spot_options, bs_model, parallel_batch=True),
    )
    show_vector(
        "FFTEngine.price_european_black_batch(parallel_batch=False)",
        fft_engine.price_european_black_batch(forward_options, black_model, parallel_batch=False),
    )
    show_vector(
        "FFTEngine.price_european_heston_batch(parallel_batch=True)",
        fft_engine.price_european_heston_batch(spot_options, heston_model, parallel_batch=True),
    )
    show_vector(
        "FFTEngine.price_european_bates_batch(parallel_batch=True)",
        fft_engine.price_european_bates_batch(spot_options, bates_model, parallel_batch=True),
    )
    show_vector(
        "FFTEngine.price_european_kou_batch(parallel_batch=True)",
        fft_engine.price_european_kou_batch(spot_options, kou_model, parallel_batch=True),
    )

    subsection("2.3 MonteCarloEngine")
    mc_engine = ql.MonteCarloEngine(
        n_paths=mc_n_paths,
        n_steps=mc_n_steps,
        seed=mc_seed,
        parallel=True,
        chunk_size=mc_chunk_size,
    )

    show("MonteCarloEngine repr", mc_engine)
    show("MonteCarloEngine.n_paths", mc_engine.n_paths)
    show("MonteCarloEngine.n_steps", mc_engine.n_steps)
    show("MonteCarloEngine.seed", mc_engine.seed)
    show("MonteCarloEngine.parallel", mc_engine.parallel)
    show("MonteCarloEngine.chunk_size", mc_engine.chunk_size)

    # Single-instrument pricing
    show("MonteCarloEngine.price_european_bs(single)", mc_engine.price_european_bs(european_spot, bs_model))
    show(
        "MonteCarloEngine.price_european_heston(single)",
        mc_engine.price_european_heston(european_spot, heston_model),
    )
    show("MonteCarloEngine.price_asian_bs(single)", mc_engine.price_asian_bs(asian_spot, bs_model))
    show(
        "MonteCarloEngine.price_asian_heston(single)",
        mc_engine.price_asian_heston(asian_spot, heston_model),
    )

    # Batch pricing
    show_vector(
        "MonteCarloEngine.price_european_bs_batch(parallel_batch=True)",
        mc_engine.price_european_bs_batch(spot_options, bs_model, parallel_batch=True),
    )
    show_vector(
        "MonteCarloEngine.price_european_heston_batch(parallel_batch=False)",
        mc_engine.price_european_heston_batch(spot_options, heston_model, parallel_batch=False),
    )
    show_vector(
        "MonteCarloEngine.price_asian_bs_batch(parallel_batch=True)",
        mc_engine.price_asian_bs_batch(asian_options, bs_model, parallel_batch=True),
    )
    show_vector(
        "MonteCarloEngine.price_asian_heston_batch(parallel_batch=False)",
        mc_engine.price_asian_heston_batch(asian_options, heston_model, parallel_batch=False),
    )

    section("3) Module-level convenience function APIs")
    subsection("3.1 Analytical European (Black-Scholes)")
    show(
        "price_european_analytical_bs(single)",
        ql.price_european_analytical_bs(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_analytical_bs_batch",
        ql.price_european_analytical_bs_batch(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            parallel=True,
        ),
    )
    show_vector(
        "price_european_analytical_bs_batch_fast",
        ql.price_european_analytical_bs_batch_fast(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            is_call=is_call,
            qtys=qtys,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            parallel=False,
        ),
    )
    show_vector(
        "price_european_analytical_bs_batch_per_model",
        ql.price_european_analytical_bs_batch_per_model(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigmas=[0.18, 0.19, 0.20, 0.21],
            rates=[0.03, 0.031, 0.032, 0.033],
            dividends=[0.01, 0.01, 0.01, 0.01],
            parallel=True,
        ),
    )

    subsection("3.2 Analytical European (Black, forward quoted)")
    show(
        "price_european_analytical_black(single)",
        ql.price_european_analytical_black(
            forward=101.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="put",
            qty=2,
            sigma=sigma_black,
            r=r_black,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_analytical_black_batch",
        ql.price_european_analytical_black_batch(
            forwards=forwards,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigma=sigma_black,
            r=r_black,
            parallel=True,
        ),
    )
    show_vector(
        "price_european_analytical_black_batch_fast",
        ql.price_european_analytical_black_batch_fast(
            forwards=forwards,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            is_call=is_call,
            qtys=qtys,
            sigma=sigma_black,
            r=r_black,
            parallel=False,
        ),
    )
    show_vector(
        "price_european_analytical_black_batch_per_model",
        ql.price_european_analytical_black_batch_per_model(
            forwards=forwards,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigmas=[0.20, 0.21, 0.22, 0.23],
            rates=[0.03, 0.031, 0.032, 0.033],
            parallel=True,
        ),
    )

    subsection("3.3 Monte Carlo European")
    show(
        "price_european_monte_carlo_bs(single)",
        ql.price_european_monte_carlo_bs(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel=True,
            chunk_size=mc_chunk_size,
            valuation_date=valuation_date,
        ),
    )
    show(
        "price_european_monte_carlo_heston(single)",
        ql.price_european_monte_carlo_heston(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel=True,
            chunk_size=mc_chunk_size,
            valuation_date=valuation_date,
        ),
    )

    subsection("3.4 FFT European")
    show(
        "price_european_fft_bs(single)",
        ql.price_european_fft_bs(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_fft_bs_batch",
        ql.price_european_fft_bs_batch(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
    )
    show(
        "price_european_fft_black(single)",
        ql.price_european_fft_black(
            forward=101.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="put",
            qty=1,
            sigma=sigma_black,
            r=r_black,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_fft_black_batch",
        ql.price_european_fft_black_batch(
            forwards=forwards,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigma=sigma_black,
            r=r_black,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=False,
        ),
    )
    show(
        "price_european_fft_heston(single)",
        ql.price_european_fft_heston(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_fft_heston_batch",
        ql.price_european_fft_heston_batch(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
    )
    show(
        "price_european_fft_bates(single)",
        ql.price_european_fft_bates(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            lambda_jump=0.5,
            mu_j=0.0,
            sigma_j=0.2,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_fft_bates_batch",
        ql.price_european_fft_bates_batch(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            lambda_jump=0.5,
            mu_j=0.0,
            sigma_j=0.2,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
    )
    show(
        "price_european_fft_kou(single)",
        ql.price_european_fft_kou(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            lambda_jump=0.5,
            p_up=0.4,
            eta1=5.0,
            eta2=10.0,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            valuation_date=valuation_date,
        ),
    )
    show_vector(
        "price_european_fft_kou_batch",
        ql.price_european_fft_kou_batch(
            spots=spots,
            strikes=strikes,
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            lambda_jump=0.5,
            p_up=0.4,
            eta1=5.0,
            eta2=10.0,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
    )

    subsection("3.5 Monte Carlo Asian")
    show(
        "price_asian_monte_carlo_bs(single)",
        ql.price_asian_monte_carlo_bs(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="call",
            qty=1,
            averaging_type="arithmetic",
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel=True,
            chunk_size=mc_chunk_size,
            valuation_date=valuation_date,
        ),
    )
    show(
        "price_asian_monte_carlo_heston(single)",
        ql.price_asian_monte_carlo_heston(
            spot=100.0,
            strike=100.0,
            expiry_date="2026-07-02",
            option_type="put",
            qty=1,
            averaging_type="geometric",
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel=True,
            chunk_size=mc_chunk_size,
            valuation_date=valuation_date,
        ),
    )

    section("4) NumPy batch APIs (contiguous arrays, explicit dtypes)")
    # NumPy APIs require specific dtypes and contiguous arrays.
    # This is one of the most important sections for high-throughput pricing from Python.
    spots_np = np.array(spots, dtype=np.float64)
    forwards_np = np.array(forwards, dtype=np.float64)
    strikes_np = np.array(strikes, dtype=np.float64)
    expiry_days_np = np.array([90, 180, 270, 365], dtype=np.int64)
    is_call_np = np.array(is_call, dtype=np.bool_)
    qtys_np = np.array(qtys, dtype=np.int32)

    show_numpy("spots_np", spots_np)
    show_numpy("forwards_np", forwards_np)
    show_numpy("strikes_np", strikes_np)
    show_numpy("expiry_days_np", expiry_days_np)
    show_numpy("is_call_np", is_call_np)
    show_numpy("qtys_np", qtys_np)

    numpy_results = {
        "price_european_analytical_bs_batch_numpy": ql.price_european_analytical_bs_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            parallel=True,
        ),
        "price_european_analytical_black_batch_numpy": ql.price_european_analytical_black_batch_numpy(
            forwards=forwards_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            sigma=sigma_black,
            r=r_black,
            parallel=True,
        ),
        "price_european_monte_carlo_bs_batch_numpy": ql.price_european_monte_carlo_bs_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel_paths=True,
            chunk_size=mc_chunk_size,
            parallel_batch=True,
        ),
        "price_european_monte_carlo_heston_batch_numpy": ql.price_european_monte_carlo_heston_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel_paths=True,
            chunk_size=mc_chunk_size,
            parallel_batch=False,
        ),
        "price_european_fft_heston_batch_numpy": ql.price_european_fft_heston_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
        "price_european_fft_bates_batch_numpy": ql.price_european_fft_bates_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            lambda_jump=0.5,
            mu_j=0.0,
            sigma_j=0.2,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
        "price_european_fft_kou_batch_numpy": ql.price_european_fft_kou_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            lambda_jump=0.5,
            p_up=0.4,
            eta1=5.0,
            eta2=10.0,
            n=fft_n,
            b=fft_b,
            interp=fft_interp,
            parallel=True,
        ),
        "price_asian_monte_carlo_bs_batch_numpy": ql.price_asian_monte_carlo_bs_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            averaging_type="arithmetic",
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel_paths=True,
            chunk_size=mc_chunk_size,
            parallel_batch=True,
        ),
        "price_asian_monte_carlo_heston_batch_numpy": ql.price_asian_monte_carlo_heston_batch_numpy(
            spots=spots_np,
            strikes=strikes_np,
            expiry_days=expiry_days_np,
            is_call=is_call_np,
            qtys=qtys_np,
            valuation_date=valuation_date,
            averaging_type="geometric",
            kappa=kappa,
            theta=theta,
            sigma=sigma_heston,
            rho=rho,
            v0=v0,
            r=r_heston,
            q=q_heston,
            n_paths=mc_n_paths,
            n_steps=mc_n_steps,
            seed=mc_seed,
            parallel_paths=False,
            chunk_size=mc_chunk_size,
            parallel_batch=True,
        ),
    }

    for api_name, values in numpy_results.items():
        show_numpy(api_name, values)

    section("5) Calibration APIs (parameter fitting)")
    # This section creates synthetic market data and then recovers model parameters.
    # In production, replace synthetic market values with real market quotes.

    subsection("5.1 Black calibration from option prices")
    black_forwards = []
    black_strikes = []
    black_maturities = []
    black_is_call = []
    black_market_values = []

    for maturity in [0.25, 0.5, 1.0]:
        expiry_date = (np.datetime64(valuation_date) + np.timedelta64(int(round(maturity * 365)), "D")).astype(str)
        for strike in [90.0, 100.0, 110.0]:
            for call in [True, False]:
                black_forwards.append(100.0)
                black_strikes.append(strike)
                black_maturities.append(maturity)
                black_is_call.append(call)
                black_market_values.append(
                    ql.price_european_analytical_black(
                        forward=100.0,
                        strike=strike,
                        expiry_date=expiry_date,
                        option_type="call" if call else "put",
                        qty=1,
                        sigma=sigma_black,
                        r=r_black,
                        valuation_date=valuation_date,
                    )
                )

    black_calib = ql.calibrate_black(
        forwards=black_forwards,
        strikes=black_strikes,
        maturities=black_maturities,
        is_call=black_is_call,
        market_values=black_market_values,
        target="price",
        initial_sigma=0.15,
        r=r_black,
        method="hybrid",
        objective="least_squares",
        max_generations=120,
        population_size=36,
        nm_max_iterations=180,
        tolerance=1.0e-6,
        seed=7,
    )
    show_calibration_result("Black calibration result", black_calib)

    subsection("5.2 Black-Scholes calibration from option prices")
    bs_spots = []
    bs_strikes = []
    bs_maturities = []
    bs_is_call = []
    bs_market_values = []

    for maturity in [0.25, 0.5, 1.0]:
        expiry_date = (np.datetime64(valuation_date) + np.timedelta64(int(round(maturity * 365)), "D")).astype(str)
        for strike in [90.0, 100.0, 110.0]:
            for call in [True, False]:
                bs_spots.append(100.0)
                bs_strikes.append(strike)
                bs_maturities.append(maturity)
                bs_is_call.append(call)
                bs_market_values.append(
                    ql.price_european_analytical_bs(
                        spot=100.0,
                        strike=strike,
                        expiry_date=expiry_date,
                        option_type="call" if call else "put",
                        qty=1,
                        sigma=sigma_bs,
                        r=r_bs,
                        q=q_bs,
                        valuation_date=valuation_date,
                    )
                )

    bs_calib = ql.calibrate_black_scholes(
        spots=bs_spots,
        strikes=bs_strikes,
        maturities=bs_maturities,
        is_call=bs_is_call,
        market_values=bs_market_values,
        target="price",
        initial_sigma=0.15,
        r=r_bs,
        q=q_bs,
        method="hybrid",
        objective="least_squares",
        max_generations=140,
        population_size=40,
        nm_max_iterations=200,
        tolerance=1.0e-6,
        seed=11,
    )
    show_calibration_result("Black-Scholes calibration result", bs_calib)

    subsection("5.3 SABR calibration from implied volatilities")
    sabr_alpha = 0.35
    sabr_beta = 0.60
    sabr_rho = -0.30
    sabr_nu = 0.80
    sabr_forward = 100.0

    sabr_forwards = []
    sabr_strikes = []
    sabr_maturities = []
    sabr_market_vols = []

    for maturity in [0.5, 1.0, 2.0]:
        for strike in [80.0, 90.0, 100.0, 110.0, 120.0]:
            sabr_forwards.append(sabr_forward)
            sabr_strikes.append(strike)
            sabr_maturities.append(maturity)
            sabr_market_vols.append(
                sabr_hagan_implied_vol(
                    forward=sabr_forward,
                    strike=strike,
                    maturity=maturity,
                    alpha=sabr_alpha,
                    beta=sabr_beta,
                    rho=sabr_rho,
                    nu=sabr_nu,
                )
            )

    sabr_calib = ql.calibrate_sabr(
        forwards=sabr_forwards,
        strikes=sabr_strikes,
        maturities=sabr_maturities,
        market_vols=sabr_market_vols,
        r=0.01,
        initial_alpha=0.20,
        initial_beta=0.60,
        initial_rho=-0.10,
        initial_nu=0.50,
        beta_bounds=(0.60, 0.60),  # fixing beta is common in practice
        method="hybrid",
        objective="relative_least_squares",
        max_generations=180,
        population_size=42,
        nm_max_iterations=220,
        tolerance=1.0e-6,
        seed=19,
    )
    show_calibration_result("SABR calibration result", sabr_calib)

    subsection("5.4 Heston calibration from FFT-priced market surface")
    # We use a compact quote set and fix a few parameters to keep runtime low in a walkthrough.
    heston_spots = []
    heston_strikes = []
    heston_maturities = []
    heston_is_call = []
    heston_market_prices = []

    for maturity in [0.5, 1.0]:
        expiry_date = (np.datetime64(valuation_date) + np.timedelta64(int(round(maturity * 365)), "D")).astype(str)
        for strike in [90.0, 100.0, 110.0]:
            for call in [True, False]:
                heston_spots.append(100.0)
                heston_strikes.append(strike)
                heston_maturities.append(maturity)
                heston_is_call.append(call)
                heston_market_prices.append(
                    ql.price_european_fft_heston(
                        spot=100.0,
                        strike=strike,
                        expiry_date=expiry_date,
                        option_type="call" if call else "put",
                        qty=1,
                        kappa=kappa,
                        theta=theta,
                        sigma=sigma_heston,
                        rho=rho,
                        v0=v0,
                        r=r_heston,
                        q=q_heston,
                        n=512,
                        b=120.0,
                        interp="cubic",
                        valuation_date=valuation_date,
                    )
                )

    heston_calib = ql.calibrate_heston(
        spots=heston_spots,
        strikes=heston_strikes,
        maturities=heston_maturities,
        is_call=heston_is_call,
        market_prices=heston_market_prices,
        initial_kappa=kappa,
        initial_theta=theta,
        initial_sigma=0.70,
        initial_rho=-0.20,
        initial_v0=0.08,
        r=r_heston,
        q=q_heston,
        kappa_bounds=(kappa, kappa),  # fixed here for a faster/stable demo
        theta_bounds=(theta, theta),  # fixed here for a faster/stable demo
        method="nelder_mead",
        objective="least_squares",
        fft_n=512,
        fft_b=120.0,
        fft_interp="cubic",
        nm_max_iterations=140,
        tolerance=1.0e-6,
        seed=23,
    )
    show_calibration_result("Heston calibration result", heston_calib)

    section("6) Optional validation demo: error message on mismatched input lengths")
    try:
        # Intentionally broken: strikes has length 3 while spots has length 4.
        # This demonstrates the explicit validation error users should expect.
        ql.price_european_analytical_bs_batch(
            spots=spots,
            strikes=strikes[:3],
            expiry_dates=expiry_dates,
            valuation_date=valuation_date,
            option_types=option_types,
            qtys=qtys,
            sigma=sigma_bs,
            r=r_bs,
            q=q_bs,
            parallel=True,
        )
    except Exception as exc:
        show("Expected validation error", exc)

    section("Done")
    print("Walkthrough completed successfully.")
    print("You can now tweak inputs above (vols, rates, maturities, paths, FFT grid) and re-run.")
    print("Summary of NumPy API outputs:")
    pprint.pprint({name: values.tolist() for name, values in numpy_results.items()})


if __name__ == "__main__":
    main()
