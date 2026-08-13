//! Per call zero-copy borrowed `&str`s under the GIL
//! (`collect_haystacks`/`with_haystacks`), or copied once into the Rust-owned
//! `Haystacks` arena and matched with the GIL released.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

/// Collects the haystack `str` objects as strong references (`Bound` owns a
/// reference). `PyList`/`PyTuple` have fast paths, with any other iterable
/// being treated generically
pub(crate) fn collect_haystacks<'py>(
    haystacks: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyString>>> {
    fn as_string<'py>(item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyString>> {
        item.cast_into::<PyString>().map_err(|err| {
            PyTypeError::new_err(format!(
                "haystacks must be str, got {}",
                err.into_inner().get_type()
            ))
        })
    }

    if let Ok(list) = haystacks.cast::<PyList>() {
        list.iter().map(as_string).collect()
    } else if let Ok(tuple) = haystacks.cast::<PyTuple>() {
        tuple.iter().map(as_string).collect()
    } else {
        haystacks.try_iter()?.map(|item| as_string(item?)).collect()
    }
}

/// Extracts the haystacks as zero-copy borrowed `&str`s and calls `f` with
/// them. The `Bound` handles are kept alive for the duration of `f` so the
/// borrowed strings cannot be deallocated
pub(crate) fn with_haystacks<R>(
    haystacks: &Bound<'_, PyAny>,
    f: impl FnOnce(&[&str]) -> PyResult<R>,
) -> PyResult<R> {
    let owners = collect_haystacks(haystacks)?;
    let strs = owners
        .iter()
        .map(|s| s.to_str())
        .collect::<PyResult<Vec<&str>>>()?;
    f(&strs)
}

/// Incrementally updatable Haystack copied into Rust memory to avoid per-call
/// overhead and locking the GIL.
/// Match indices refer to this list's order
#[pyclass(name = "Haystacks", module = "frizbee")]
pub(crate) struct PyHaystacks {
    /// Concatenated utf-8 bytes of every haystack
    bytes: Vec<u8>,
    /// Haystack `i` spans `bytes[offsets[i]..offsets[i + 1]]`; always `len + 1`
    /// entries, starting at 0
    offsets: Vec<usize>,
}

impl PyHaystacks {
    fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            offsets: vec![0],
        }
    }

    pub(crate) fn from_iterable(items: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut haystacks = Self::empty();
        haystacks.extend(items)?;
        Ok(haystacks)
    }

    /// Views of every haystack; the arena is Rust-owned, so the views can be
    /// used with the GIL released
    pub(crate) fn as_strs(&self) -> Vec<&str> {
        self.offsets
            .windows(2)
            // SAFETY: the arena only ever holds bytes copied from `&str`s
            .map(|span| unsafe { core::str::from_utf8_unchecked(&self.bytes[span[0]..span[1]]) })
            .collect()
    }
}

#[pymethods]
impl PyHaystacks {
    #[new]
    #[pyo3(signature = (items = None))]
    fn new(items: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        match items {
            Some(items) => Self::from_iterable(items),
            None => Ok(Self::empty()),
        }
    }

    /// Appends a single haystack
    fn append(&mut self, item: &str) {
        self.bytes.extend_from_slice(item.as_bytes());
        self.offsets.push(self.bytes.len());
    }

    /// Appends every haystack in the iterable (list/tuple are the fast paths)
    fn extend(&mut self, items: &Bound<'_, PyAny>) -> PyResult<()> {
        for item in collect_haystacks(items)? {
            self.append(item.to_str()?);
        }
        Ok(())
    }

    /// Empties the list, keeping the allocation for reuse
    fn clear(&mut self) {
        self.bytes.clear();
        self.offsets.truncate(1);
    }

    fn __len__(&self) -> usize {
        self.offsets.len() - 1
    }

    fn __repr__(&self) -> String {
        format!("Haystacks(len={})", self.offsets.len() - 1)
    }
}
