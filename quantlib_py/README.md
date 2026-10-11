# quantlib_py

PyO3 bindings for QuantRustLib.

This crate exposes a Python module (`quantlib_py`) that wraps the Rust workspace:

- Core types from `quantlib_core`
- Products from `quantlib_products`
- Models from `quantlib_models`
- Engines and batch helpers from `quantlib_engines`

For the overall project documentation, start at the workspace README: [`quantlib-rs/README.md`](../README.md).

## Build locally

```bash
cd quantlib-rs/quantlib_py
maturin develop
```

## Run Python Parallelization Tests

Dedicated tests for single-thread vs parallel behavior (analytical and Monte Carlo, including
step-heavy workloads) live in `quantlib_py/tests/test_parallelization.py`.

Run from repo root with your virtualenv active:

```bash
python -m unittest discover -s quantlib_py/tests -p "test_parallelization.py" -v
```

## Run Python Calibration Tests

Calibration API coverage for Black, Black-Scholes, SABR, and Heston lives in:
`quantlib_py/tests/test_calibration_bindings.py`.

```bash
python -m unittest discover -s quantlib_py/tests -p "test_calibration_bindings.py" -v
```

## Recent Additions

### Bates Model
A stochastic volatility model combining Heston's mean-reverting variance with an additional log-normal jump component. Ideal for capturing both volatility clustering and sudden price jumps in markets.

**Parameters**: `kappa` (mean reversion speed), `theta` (mean variance), `sigma` (volatility of volatility), `rho` (correlation between asset and variance), `v0` (initial variance), `r` (risk-free rate), `q` (dividend yield), `lambda_jump` (jump intensity), `mu_j` (jump mean), `sigma_j` (jump std deviation)

**Available Through**: FFT pricing method

### Kou Model
A pure jump-diffusion model with double exponential jump distribution. Excellent for capturing asymmetric jump behavior commonly seen in equity markets and index options.

**Parameters**: `sigma` (volatility), `r` (risk-free rate), `q` (dividend yield), `lambda_jump` (jump intensity), `p_up` (probability of upward jump), `eta1` (exponential factor for up jumps), `eta2` (exponential factor for down jumps)

**Available Through**: FFT pricing method

## Comparison Notebook

- Notebook: `quantlib-rs/quantlib_py/notebooks/compare_quantlibpy_vs_quantlib_rs.ipynb`
- Purpose: compares pricing parity and performance between Rust bindings (`quantlib_py`) and the original Python implementation (`QuantLibPY/src`).
- Run from repo root with your virtualenv active:

```bash
cd quantlib-rs
jupyter lab
```

## Performance guidance

- Prefer batch pricing methods to amortize Python<->Rust overhead.
- Reuse `EuropeanOptionSpot` and `BlackScholesModel` objects when pricing repeatedly.
- Heavy pricing methods release the Python GIL while Rust computes.
- For Monte Carlo batch calls, `parallel_batch=True` by default for better throughput.
- Use the NumPy batch APIs when possible. They avoid Python list element conversion and return NumPy arrays directly.

## When to Use Each Pricing Method

### Analytical (Black-Scholes/Black)
- **Best for**: European options with constant volatility
- **Advantages**: Instant computation, zero simulation error
- **Limitations**: Only works with Black-Scholes or Black forward-quoted models

### Monte Carlo
- **Best for**: Path-dependent options (Asian), complex payoffs, flexible model calibration
- **Advantages**: Very flexible, scales to any payoff
- **Limitations**: Statistical error, slower for simple European options

### FFT (Fast Fourier Transform)
- **Best for**: European options with advanced models (Heston, Bates, Kou)
- **Advantages**: Speed and accuracy for complex jump-diffusion and stochastic volatility models
- **Limitations**: European options only; best with FFT-friendly characteristic functions

---

## API snapshot

### Models
Supported stochastic asset price models:

- **Black-Scholes**: Constant volatility model for European options
  - `BlackScholesModel(sigma, r, q)`
  
- **Black**: Forward-quoted constant volatility model
  - `BlackModel(sigma, r)`
  
- **Heston**: Stochastic volatility model with mean-reverting variance
  - `HestonModel(kappa, theta, sigma, rho, v0, r, q)`
  
- **Bates**: Stochastic volatility model with mean-reverting variance + jump component
  - `BatesModel(kappa, theta, sigma, rho, v0, r, q, lambda_jump, mu_j, sigma_j)`
  
- **Kou**: Jump-diffusion model with double exponential jump distribution
  - `KouModel(sigma, r, q, lambda_jump, p_up, eta1, eta2)`

### Products
Supported option types:

- **European Options**: `EuropeanOptionSpot`, `EuropeanOptionForward`
- **Digital Options**: `DigitalOptionSpot`, `DigitalOptionForward`
- **Asian Options**: `AsianOptionSpot`

### Analytical Pricing (Black-Scholes, spot quoted)
Closed-form solutions for European and Digital options:

- `price_european_analytical_bs(...)`
- `price_european_analytical_bs_batch(...)`
- `price_european_analytical_bs_batch_fast(...)`
- `price_european_analytical_bs_batch_per_model(...)`
- `price_digital_analytical_bs(...)`

### Analytical Pricing (Black, forward quoted)
Closed-form solutions for forward-quoted European and Digital options:

- `price_european_analytical_black(...)`
- `price_european_analytical_black_batch(...)`
- `price_european_analytical_black_batch_fast(...)`
- `price_european_analytical_black_batch_per_model(...)`
- `price_digital_analytical_black(...)`

### Monte Carlo Simulation
Pricing via path simulation (supports Heston stochastic volatility):

- **European Options**:
  - `price_european_monte_carlo_bs(...)`
  - `price_european_monte_carlo_heston(...)`
  
- **Digital Options**:
  - `price_digital_monte_carlo_bs(...)`
  - `price_digital_monte_carlo_heston(...)`
  
- **Asian Options**:
  - `price_asian_monte_carlo_bs(...)`
  - `price_asian_monte_carlo_heston(...)`

### FFT Pricing
Fast Fourier Transform-based pricing (supports multiple jump-diffusion and stochastic volatility models):

- **Black-Scholes**:
  - `price_european_fft_bs(...)`
  - `price_european_fft_bs_batch(...)`
  
- **Black (forward-quoted)**:
  - `price_european_fft_black(...)`
  - `price_european_fft_black_batch(...)`
  
- **Heston** (stochastic volatility):
  - `price_european_fft_heston(...)`
  - `price_european_fft_heston_batch(...)`
  
- **Bates** (stochastic volatility + jumps):
  - `price_european_fft_bates(...)`
  - `price_european_fft_bates_batch(...)`
  
- **Kou** (jump-diffusion with double exponential jumps):
  - `price_european_fft_kou(...)`
  - `price_european_fft_kou_batch(...)`

### NumPy Batch APIs
High-performance NumPy-based batch pricing with zero-copy array handling:

- **Analytical Pricing**:
  - `price_european_analytical_bs_batch_numpy(...)`
  - `price_european_analytical_black_batch_numpy(...)`
  
- **Monte Carlo Simulation**:
  - `price_european_monte_carlo_bs_batch_numpy(...)`
  - `price_european_monte_carlo_heston_batch_numpy(...)`
  - `price_asian_monte_carlo_bs_batch_numpy(...)`
  - `price_asian_monte_carlo_heston_batch_numpy(...)`
  
- **FFT Pricing**:
  - `price_european_fft_heston_batch_numpy(...)`
  - `price_european_fft_bates_batch_numpy(...)`
  - `price_european_fft_kou_batch_numpy(...)`

### Engine Classes
Object-oriented API for advanced use cases:

- **`AnalyticalEngine`**: Closed-form European and Digital option pricing
  - Methods: `price_european_*`, `price_digital_*`
  
- **`FFTEngine`**: FFT-based European option pricing
  - Methods: `price_european_bs*`, `price_european_black*`, `price_european_heston*`, `price_european_bates*`, `price_european_kou*`

- **`CosEngine(n=256, l=12.0)`**: Fang–Oosterlee COS pricing, much faster than FFT for single strikes
  - Methods: same as `FFTEngine`
  
- **`MonteCarloEngine`**: Path simulation-based pricing
  - `heston_scheme="qe"` (default, Andersen QE) or `"full_truncation"` for Heston paths
  - Methods: `price_european_*`, `price_digital_*`, `price_asian_*`

### Calibration APIs

Model calibration functions with flexible optimizer/objective controls:

- `calibrate_black(...)`
- `calibrate_black_scholes(...)`
- `calibrate_sabr(...)`
- `calibrate_heston(...)`

All calibration functions return a dictionary with:

- `success`, `message`, `iterations`
- `objective_value`, `rmse`, `mape`, `max_error`
- `calibrated_params`
- `errors`, `market_values`, `model_values`
- `model` (calibrated model parameter dict)

## NumPy API technical details

- Input contract:
  - Arrays must be 1D and contiguous (C-order).
  - Dtypes must match exactly:
    - `spots`, `strikes`, `forwards`: `float64`
    - `expiry_days`: `int64`
    - `is_call`: `bool`
    - `qtys`: `int32`
- Lengths:
  - All input arrays in a call must have the same length.
- Expiry representation:
  - NumPy APIs use `expiry_days` relative to `valuation_date`.
  - Each option expiry is computed in Rust as `valuation_date + expiry_days[i]`.
- Output:
  - Returned values are `numpy.ndarray[float64]`.
  - Output array is created from Rust-owned memory and transferred to NumPy.
- Parallel controls for Monte Carlo NumPy APIs:
  - `parallel_paths`: parallelism inside each Monte Carlo price (path simulation).
  - `parallel_batch`: parallelism across instruments in the batch.
  - Defaults favor batch-level parallelism for better end-to-end performance.

## NumPy usage example

### Monte Carlo pricing with Heston model
```python
import numpy as np
import quantlib_py as ql

spots = np.array([100.0, 101.0, 98.0], dtype=np.float64)
strikes = np.array([100.0, 100.0, 100.0], dtype=np.float64)
expiry_days = np.array([365, 365, 365], dtype=np.int64)
is_call = np.array([True, False, True], dtype=np.bool_)
qtys = np.array([1, 1, 2], dtype=np.int32)

prices = ql.price_european_monte_carlo_bs_batch_numpy(
    spots=spots,
    strikes=strikes,
    expiry_days=expiry_days,
    is_call=is_call,
    qtys=qtys,
    valuation_date="2026-02-11",
    sigma=0.2,
    r=0.03,
    q=0.0,
    n_paths=200_000,
    n_steps=252,
    seed=42,
    parallel_paths=True,
    chunk_size=4096,
    parallel_batch=True,
)
```

### FFT pricing with Bates model (stochastic volatility + jumps)
```python
import quantlib_py as ql

# Create Bates model (Heston with log-normal jumps)
bates_model = ql.BatesModel(
    kappa=2.0,           # mean reversion speed
    theta=0.04,          # long-term variance
    sigma=0.3,           # volatility of volatility
    rho=-0.7,            # correlation
    v0=0.04,             # initial variance
    r=0.05,              # risk-free rate
    q=0.02,              # dividend yield
    lambda_jump=0.5,     # jump intensity
    mu_j=0.0,            # jump mean (log return)
    sigma_j=0.2          # jump volatility
)

price = ql.price_european_fft_bates(
    spot=100.0,
    strike=100.0,
    expiry_date="2027-02-11",
    option_type="call",
    qty=1,
    model=bates_model,
    valuation_date="2026-02-11"
)
```

### FFT pricing with Kou model (double exponential jumps)
```python
import quantlib_py as ql

# Create Kou model (pure jump-diffusion)
kou_model = ql.KouModel(
    sigma=0.2,           # volatility
    r=0.05,              # risk-free rate
    q=0.02,              # dividend yield
    lambda_jump=0.5,     # jump intensity
    p_up=0.4,            # probability of upward jump
    eta1=5.0,            # exponential factor for up jumps
    eta2=10.0            # exponential factor for down jumps
)

price = ql.price_european_fft_kou(
    spot=100.0,
    strike=100.0,
    expiry_date="2027-02-11",
    option_type="call",
    qty=1,
    model=kou_model,
    valuation_date="2026-02-11"
)
```

## Related documentation (Rust crates)

- [`quantlib_core`](../quantlib_core/README.md)
- [`quantlib_products`](../quantlib_products/README.md)
- [`quantlib_models`](../quantlib_models/README.md)
- [`quantlib_engines`](../quantlib_engines/README.md)
- [`quantlib_examples`](../quantlib_examples/README.md)
