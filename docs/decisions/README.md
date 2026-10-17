# Decision records

Context, decision, alternatives and consequences of each design choice. Never edited once accepted; superseded by a new record.

- [0001](0001-property-based-tests-and-ci.md) Property-based and reference tests; CI on three OSes plus Python (#22)
- [0002](0002-portable-norm-cdf.md) One portable, tail-accurate normal CDF (#23)
- [0003](0003-wasm-web-playground.md) Web playground: Rust compiled to WASM, thin TypeScript UI (#24)
- [0004](0004-sabr-hagan-2002.md) SABR implied vol: Hagan et al. (2002) eq. (2.17a) exactly (#25)
- [0005](0005-fourier-model-trait.md) FourierModel trait and one generic FFT engine (#26)
- [0006](0006-cos-engine.md) COS (Fang–Oosterlee) engine for few-strike pricing (#26)
- [0007](0007-heston-calibration-strips.md) Heston calibration at exact maturities, one FFT strip per maturity (#27)
- [0008](0008-remove-pricing-cache.md) Remove the pricing cache instead of fixing it (#28)
- [0009](0009-heston-mc-qe-full-truncation.md) Heston MC: Andersen QE (default) and full truncation, no absorption (#29)
- [0010](0010-per-chunk-rng-streams.md) Per-chunk RNG streams: Xoshiro256++ seeded by SplitMix64 (#30)
- [0011](0011-streaming-path-functionals.md) Streaming path functionals and one step kernel per model (#30)
- [0012](0012-single-monte-carlo-engine.md) One Monte Carlo engine (#30)
- [0013](0013-mc-standard-errors-variance-reduction.md) MC standard errors, antithetic and control variates (#30)
- [0014](0014-control-variate-martingale-schemes.md) Control variates only under martingale-preserving discretisations (#30)
- [0015](0015-agents-process.md) AGENTS.md: least code, independent write / review / test (#30)
- [0016](0016-schedules-holiday-calendars.md) Schedules and holiday calendars, split out of #17 (#47)
- [0017](0017-docs-as-code.md) Documentation as part of done
- [0018](0018-control-variate-both-heston-schemes.md) Control variates under both Heston schemes (#50)
- [0019](0019-tiered-agent-pipeline.md) Tiered agent pipeline with blind spec tests
- [0020](0020-calibration-fixed-rates.md) Calibration: fixed rates, no flat-vol target, unsorted DE, measured FFT default (#52)
