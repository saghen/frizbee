//! Python bindings for frizbee, SIMD fuzzy string matching.
//!
//! The module is named `frizbee` (see `module-name` in `pyproject.toml`) and
//! only translates types at the boundary: matching always happens in the core
//! crate.
//!
//! Haystacks either live in a Rust-owned [`haystacks::PyHaystacks`] arena — the
//! primary path for repeated matching, with the GIL released while matching —
//! or are extracted per call as zero-copy borrowed `&str`s under the GIL
//! (CPython caches the UTF-8 representation on each `str` object, so repeat
//! calls are pure pointer reads).

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
    m.add_function(wrap_pyfunction!(pattern::parse_query, m)?)?;
    m.add_function(wrap_pyfunction!(config::max_needle_len, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
