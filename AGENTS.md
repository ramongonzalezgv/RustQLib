# Agent guidelines

## Write the least code possible

AI-written code tends to be verbose, and verbose codebases are hard to maintain. Every line must earn its place.

- Solve the problem with the fewest lines that stay clear. Prefer deleting code to adding it.
- No speculative generality: no options, traits, builders or parameters without a current caller.
- No wrappers that only forward, no getters for public fields, no one-use helpers that hide two lines.
- No duplicated logic: reuse existing types and functions (`SimpleDate`, `MarketCalendar`, `QuantLibError`, the MC driver) before writing new ones.
- Comments say why, not what. No comments restating the code, no banner comments, no commented-out code.
- Doc comments: one line unless a formula or invariant needs more.
- Tests: few, sharp, and statistical (`k * std_error`) for Monte Carlo. One test per behaviour, no copy-pasted variants; table-drive similar cases.
- Error handling: validate once at construction; do not re-validate in hot paths.
- Match the surrounding style; do not reformat unrelated code.

## Before declaring work done

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo bench --workspace --no-run
```

Python bindings changed: build with `maturin develop --release -m quantlib_py/Cargo.toml` and run `pytest -q quantlib_py/tests`.

## Process for non-trivial issues

Write, review and test with independent agents: the reviewer and tester did not write the code and judge it only against the issue's acceptance criteria and this file.
