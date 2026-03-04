use pyo3::prelude::*;

mod binding;
use binding::{PyConfig, PyMatch, PyMatchIndices, PyMatcher, PyScoring};

#[pymodule(gil_used = false)]
fn frizbee_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatch>()?;
    m.add_class::<PyMatchIndices>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyScoring>()?;
    m.add_class::<PyMatcher>()?;
    m.add_function(wrap_pyfunction!(binding::py_match_list, m)?)?;
    m.add_function(wrap_pyfunction!(binding::py_match_list_indices, m)?)?;
    m.add_function(wrap_pyfunction!(binding::py_match_list_parallel, m)?)?;
    Ok(())
}
