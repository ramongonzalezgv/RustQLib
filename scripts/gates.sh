#!/usr/bin/env bash
# AGENTS.md checks. Prints only failures so agents spend no tokens on passing output.
set -u
cd "$(dirname "$0")/.."
fail=0
run() {
  if ! out=$("$@" 2>&1); then
    echo "FAIL: $*"
    echo "$out" | grep -E '^(error|warning)|Diff in|panicked|^---- |test result: FAILED' | head -40
    fail=1
  fi
}
run cargo fmt --all -- --check
run cargo clippy --workspace --all-targets -- -D warnings
run cargo test --workspace
run env RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
run cargo bench --workspace --no-run
if ! git diff --quiet main -- quantlib_py; then
  run maturin develop --release -m quantlib_py/Cargo.toml
  run pytest -q quantlib_py/tests
fi
[ "$fail" = 0 ] && echo "gates: ok"
exit "$fail"
