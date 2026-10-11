fn main() {
    // Python extension modules must not link libpython; on macOS this adds
    // `-undefined dynamic_lookup` so plain `cargo build` links like maturin does.
    pyo3_build_config::add_extension_module_link_args();
}
