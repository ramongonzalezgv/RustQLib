# `quantlib_utils`

Utilities and supporting components for QuantRustLib.

This crate is currently a **scaffold**. It exists to hold functionality that supports pricing but is not itself a product/model/engine—for example:

- yield curves and discounting helpers
- bootstrapping and interpolation utilities
- calibration helpers and results processing
- market calendars beyond the simple calendars in `quantlib_core`

At the moment, the crate mainly serves as a placeholder to preserve the intended modular architecture described in the workspace documentation.

## Testing

```bash
cargo test -p quantlib_utils
```

