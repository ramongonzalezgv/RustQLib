# Binomial engine

Cox–Ross–Rubinstein (1979) recombining tree for European and American vanillas under Black–Scholes ([black-scholes.md](../models/black-scholes.md)). Code: `quantlib_engines/src/binomial_engine.rs`.

## Algorithm

With $\Delta = T/N$:

$$u = e^{\sigma\sqrt\Delta},\qquad d = 1/u,\qquad p = \frac{e^{(r-q)\Delta} - d}{u - d},\qquad D = e^{-r\Delta}$$

Node $(i, j)$, with $j$ down-moves after $i$ steps, has spot $S\,u^{i-j}d^{\,j}$.
- Terminal layer: $V_{N,j} = (\omega(S_{N,j} - K))^+$.
- Backward induction: $V_{i,j} = D\big(pV_{i+1,j} + (1-p)V_{i+1,j+1}\big)$.
- American exercise: $V_{i,j} \leftarrow \max\big(V_{i,j},\ (\omega(S_{i,j} - K))^+\big)$ at every node, including the root.
- Price: $\text{qty}\cdot V_{0,0}$. The tree is unsigned, so a short position is exactly the negated long price.

Special cases:
- $T \le 0$: $\text{qty}\cdot$ intrinsic value, with $u = d = 1$.
- $|u - d| < 10^{-14}$ (σ → 0): $u = d = e^{(r-q)\Delta}$ and $p = 1$, a deterministic forward.
- $p \notin [0, 1]$, i.e. $\sigma\sqrt\Delta$ is too small against $|r - q|\Delta$: `Engine` error.

CRR converges at $O(1/N)$, and the error oscillates with the parity of $N$.

## Implementation map

| Item | Role |
|---|---|
| `BinomialEngine { n_steps }` | `new(n)` clamps to ≥ 1; a literal `n_steps = 0` (or `Some(0)` override) is a `Validation` error. |
| `PricingEngine<EuropeanOption \| AmericanOption, BlackScholesModel>` | Spot-quoted products only (`require_spot`). |
| `price_and_levels_european` / `price_and_levels_american(product, model, n_steps: Option<usize>)` | Price plus `BinomialLevels`. `None` uses `self.n_steps`. |
| `BinomialLevels { price, up_factor, down_factor }` | `qty`-scaled price, $u$, $d$. |
| `price_and_levels`, `terminal_payoffs`, `intrinsic_value` | Tree. |

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `n_steps` | `new(n)`: `max(n, 1)`. `Default`: 100 | Tree steps $N$; 0 is rejected. |

No tree type option: CRR only.

## Design choices

- `qty` scales the price after the tree, so American exercise is always the holder's `max` regardless of sign (#51).
- No cache. An earlier cache keyed on too few inputs (exercise style, valuation date, calendar) and returned stale prices. Regression tests guard against its return.

## Validation

`quantlib_engines/tests/binomial_engine.rs`:

| Test | Checks |
|---|---|
| `european_converges_to_black_scholes` | K ∈ {90 call, 100 call, 110 put}: N = 2000 within 5e-3 of BS; error envelope max(err(800), err(801)) < ¼ of max(err(50), err(51)). |
| `american_is_worth_at_least_european` | K ∈ {80, 100, 120}, calls and puts. |
| `american_call_without_dividends_equals_european` | q = 0, within 1e-10 (Merton). |
| `american_put_dominates_intrinsic_and_has_early_exercise_premium` | K ∈ {90, 100, 130}, r = 8%, N = 500: ≥ intrinsic, premium over European > 1e-3. |
| `american_put_matches_published_reference` | Hull American put S = K = 50, r = 10%, σ = 40%, about 5 months: 4.2842 ± 1e-2 at N = 2000. |
| `one_engine_prices_european_and_american_independently`, `one_engine_reprices_options_that_differ_only_by_valuation_date_or_calendar` | No-cache regressions (#2). |

Unit tests in `src/binomial_engine.rs`: `binomial_european_converges_to_analytical`, `american_put_not_cheaper_than_european_put`, and `short_position_negates_price_and_zero_steps_rejected` (American put, qty = −1 equals −(qty = 1); `n_steps = 0` errors; #51).

## Limitations and open issues

- **#12**:
  - O(N²) `powf` calls: the American exercise check computes the node spot with two `powf` per node.
  - One `Vec` allocation per level.
  - CRR only, with an oscillating $O(1/N)$ error that rules out Richardson extrapolation.
  - Planned: precomputed spot ladder, in-place rollback, Leisen–Reimer tree (odd $N$, $O(1/N^2)$) with Richardson, and Leisen–Reimer as the default for Americans.
- Black–Scholes only. No discrete dividends, no Bermudan schedules.
- Fast American pricing without a tree: #13. Tree Greeks are not exposed: #18.
