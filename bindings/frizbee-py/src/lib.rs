//! Python bindings for frizbee, SIMD fuzzy string matching.

use pyo3::prelude::*;

mod config;
mod haystacks;
mod matcher;
mod pattern;

#[pymodule(name = "frizbee")]
fn frizbee_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<matcher::PyMatcher>()?;
    m.add_class::<haystacks::PyHaystacks>()?;
    m.add_class::<pattern::PyPattern>()?;
    m.add_class::<matcher::PyMatch>()?;
    m.add_class::<matcher::PyMatchIndices>()?;
    m.add_class::<config::PyScoring>()?;
    m.add_function(wrap_pyfunction!(config::max_needle_len, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
