//! Python bindings for frizbee, SIMD fuzzy string matching.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

mod config;
mod haystacks;
mod matcher;
mod pattern;

fn add_literal(m: &Bound<'_, PyModule>, name: &str, values: &[&str]) -> PyResult<()> {
    let literal = m.py().import("typing")?.getattr("Literal")?;
    let values = PyTuple::new(m.py(), values)?;
    m.add(name, literal.get_item(values)?)
}

#[pymodule(name = "frizbee")]
fn frizbee_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<matcher::PyMatcher>()?;
    m.add_class::<haystacks::PyHaystacks>()?;
    m.add_class::<pattern::PyPattern>()?;
    m.add_class::<matcher::PyMatch>()?;
    m.add_class::<matcher::PyMatchIndices>()?;
    m.add_class::<config::PyScoring>()?;
    m.add_function(wrap_pyfunction!(pattern::parse_query, m)?)?;
    add_literal(m, "Casing", &["ignore", "smart", "respect"])?;
    add_literal(m, "Unicode", &["ignore", "smart", "always"])?;
    add_literal(
        m,
        "Matching",
        &["fuzzy", "exact", "prefix", "suffix", "substring"],
    )?;
    add_literal(
        m,
        "Sort",
        &[
            "score_then_index_asc",
            "score_then_index_desc",
            "index_asc",
            "index_desc",
        ],
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
