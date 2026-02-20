mod bindings;
mod functions;

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
fn quantlib_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register_classes(m)?;
    functions::register_functions(m)?;
    Ok(())
}
