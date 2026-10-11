//! WebAssembly entry points for the web pricing playground (`web/`).
//!
//! [`api`] holds all logic as plain Rust so it is tested natively; the
//! `bindings` module only exists on `wasm32` and converts to and from JS.

pub mod api;

#[cfg(target_arch = "wasm32")]
mod bindings;
