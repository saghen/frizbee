//! The `Matcher` class and the `Match`/`MatchIndices` result types it returns.

use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::config::{
    PyScoring, build_config, casing_to_str, matching_to_str, sort_to_str, unicode_to_str,
};
use crate::haystacks::{PyHaystacks, with_haystacks};
use crate::pattern::{PatternArg, PyPattern};

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// Result of a fuzzy match, containing the score and index in the haystack
#[pyclass(
    name = "Match",
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "frizbee"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyMatch {
    /// Score of the match, higher is better
    score: u16,
    /// Index of the match in the original list of haystacks
    index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    exact: bool,
}

#[pymethods]
impl PyMatch {
    fn __repr__(&self) -> String {
        format!(
            "Match(score={}, index={}, exact={})",
            self.score,
            self.index,
            py_bool(self.exact)
        )
    }
}

impl From<frizbee::Match> for PyMatch {
    fn from(m: frizbee::Match) -> Self {
        Self {
            score: m.score,
            index: m.index,
            exact: m.exact,
        }
    }
}

/// Like [`PyMatch`] but includes the indices of the chars in the haystack that
/// matched the needle in reverse order
#[pyclass(
    name = "MatchIndices",
    frozen,
    get_all,
    eq,
    hash,
    skip_from_py_object,
    module = "frizbee"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyMatchIndices {
    /// Score of the match, higher is better
    score: u16,
    /// Index of the match in the original list of haystacks
    index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    exact: bool,
    /// Indices of the chars in the haystack that matched the needle in reverse order
    indices: Vec<u32>,
}

#[pymethods]
impl PyMatchIndices {
    fn __repr__(&self) -> String {
        format!(
            "MatchIndices(score={}, index={}, exact={}, indices={:?})",
            self.score,
            self.index,
            py_bool(self.exact),
            self.indices
        )
    }
}

impl From<frizbee::MatchIndices> for PyMatchIndices {
    fn from(m: frizbee::MatchIndices) -> Self {
        Self {
            score: m.score,
            index: m.index,
            exact: m.exact,
            indices: m.indices,
        }
    }
}

/// Primary entrypoint for fuzzy matching. Compiles the pattern(s) once, allocates
/// memory for the Smith Waterman matrix, and reuses the selected SIMD backend.
/// Ideally, only construct these at most once per list: they're cheap to
/// construct, but end up being expensive if you construct them for each item in
/// your list
#[pyclass(name = "Matcher", module = "frizbee")]
pub(crate) struct PyMatcher {
    // Boxed: the compiled SIMD backends require 64-byte alignment (AVX-512
    // vectors), which CPython's object allocator does not guarantee for the
    // pyclass payload itself
    inner: Box<frizbee::Matcher>,
    /// Truncates the `match_list` results after sorting; `None` means unlimited
    max_items: Option<u32>,
}

impl PyMatcher {
    fn truncate<T>(&self, matches: &mut Vec<T>) {
        if let Some(max_items) = self.max_items {
            matches.truncate(max_items as usize);
        }
    }
}

#[pymethods]
impl PyMatcher {
    // The config kwargs are repeated verbatim on `__init__`, `from_query`,
    // `from_patterns` and `set_config` (pyo3 disallows macro-generated methods in
    // `#[pymethods]`). `max_typos` defaults to the core `Config::default()` value
    // (0); passing `None` means unlimited. `max_items` defaults to `None` (unlimited)

    /// Creates a matcher from a single pattern (str or Pattern). Strings match
    /// literally; use `from_query` for query syntax and `from_patterns` for
    /// multi-pattern queries
    #[new]
    #[pyo3(signature = (needle, *, max_typos = frizbee::Config::default().max_typos,
        max_items = None, casing = None, unicode = None, matching = None, sort = None,
        scoring = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        needle: PatternArg,
        max_typos: Option<u16>,
        max_items: Option<u32>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<Self> {
        let config = build_config(max_typos, casing, unicode, matching, sort, scoring)?;
        Ok(Self {
            inner: Box::new(frizbee::Matcher::new(
                frizbee::Pattern::from(needle),
                &config,
            )),
            max_items,
        })
    }

    /// Shorthand for calling `from_patterns` with the parsed query (see
    /// `parse_query`), where special syntax changes each atom's matching mode:
    /// `foo` (fuzzy), `^foo` (prefix), `foo$` (suffix), `'foo` (substring),
    /// `^foo$` (exact) and `!foo` (negated, substring unless combined with the
    /// syntax above). A haystack matches when all of the atoms match, summing
    /// each atom's score.
    ///
    /// Any special character can be escaped with a backslash, e.g. `\!foo` or
    /// `foo\$` match the literal leading/trailing character, and `foo\ bar`
    /// matches the literal space
    #[classmethod]
    #[pyo3(signature = (query, *, max_typos = frizbee::Config::default().max_typos,
        max_items = None, casing = None, unicode = None, matching = None, sort = None,
        scoring = None))]
    #[allow(clippy::too_many_arguments)]
    fn from_query(
        _cls: &Bound<'_, PyType>,
        query: &str,
        max_typos: Option<u16>,
        max_items: Option<u32>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<Self> {
        let config = build_config(max_typos, casing, unicode, matching, sort, scoring)?;
        Ok(Self {
            inner: Box::new(frizbee::Matcher::from_query(query, &config)),
            max_items,
        })
    }

    /// Creates a matcher from a list of patterns (str or Pattern), matched
    /// independently. A haystack matches when all of the patterns match, where the
    /// score is the sum of each pattern's score
    #[classmethod]
    #[pyo3(signature = (patterns, *, max_typos = frizbee::Config::default().max_typos,
        max_items = None, casing = None, unicode = None, matching = None, sort = None,
        scoring = None))]
    #[allow(clippy::too_many_arguments)]
    fn from_patterns(
        _cls: &Bound<'_, PyType>,
        patterns: Vec<PatternArg>,
        max_typos: Option<u16>,
        max_items: Option<u32>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<Self> {
        let config = build_config(max_typos, casing, unicode, matching, sort, scoring)?;
        let patterns: Vec<frizbee::Pattern> = patterns.into_iter().map(Into::into).collect();
        Ok(Self {
            inner: Box::new(frizbee::Matcher::from_patterns(&patterns, &config)),
            max_items,
        })
    }

    /// Matches a list of haystacks, returning the matches ordered by the sort
    /// strategy. This API provides the most performant path when matching on lists.
    /// Pass a [`PyHaystacks`] to release the GIL while matching; plain iterables
    /// are borrowed zero-copy under the GIL
    fn match_list(
        &mut self,
        py: Python<'_>,
        haystacks: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyMatch>> {
        let mut matches = if let Ok(haystacks) = haystacks.cast::<PyHaystacks>() {
            let haystacks = haystacks.borrow();
            let strs = haystacks.as_strs();
            let inner = &mut self.inner;
            py.detach(move || inner.match_list(&strs))
        } else {
            with_haystacks(haystacks, |strs| Ok(self.inner.match_list(strs)))?
        };
        self.truncate(&mut matches);
        Ok(matches.into_iter().map(Into::into).collect())
    }

    /// Like `match_list` but matched in parallel on multiple real threads
    /// (0 = available CPU cores - 2). Threads work on 2048 item chunks, and the
    /// final result is identical to `match_list`. The GIL is released while
    /// matching; plain iterables are first copied into a temporary arena under the
    /// GIL, which passing a [`PyHaystacks`] skips
    fn match_list_parallel(
        &mut self,
        py: Python<'_>,
        haystacks: &Bound<'_, PyAny>,
        threads: usize,
    ) -> PyResult<Vec<PyMatch>> {
        // Both arms borrow `strs` from an arena that outlives `detach`
        let owned;
        let temp;
        let strs = if let Ok(haystacks) = haystacks.cast::<PyHaystacks>() {
            owned = haystacks.borrow();
            owned.as_strs()
        } else {
            temp = PyHaystacks::from_iterable(haystacks)?;
            temp.as_strs()
        };
        let inner = &mut self.inner;
        let mut matches = py.detach(move || inner.match_list_parallel(&strs, threads));
        self.truncate(&mut matches);
        Ok(matches.into_iter().map(Into::into).collect())
    }

    /// Like `match_list` but each match includes the indices of the chars in the
    /// haystack that matched the needle. This API has not been optimized for
    /// performance, and should only be used on small lists, e.g. the visible
    /// portion of the results. Useful for displaying matched indices in the UI
    fn match_list_indices(
        &mut self,
        haystacks: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<PyMatchIndices>> {
        let mut matches = if let Ok(haystacks) = haystacks.cast::<PyHaystacks>() {
            self.inner.match_list_indices(&haystacks.borrow().as_strs())
        } else {
            with_haystacks(haystacks, |strs| Ok(self.inner.match_list_indices(strs)))?
        };
        self.truncate(&mut matches);
        Ok(matches.into_iter().map(Into::into).collect())
    }

    /// Matches a single haystack, returning its match (with `index` echoed back) if
    /// it passes. This API performs ~10% slower than the `match_list` API. Consider
    /// using `match_list` if you have more than one haystack to match, as it
    /// performs significantly better
    fn match_one(&mut self, haystack: &str, index: u32) -> Option<PyMatch> {
        self.inner.match_one(haystack, index).map(Into::into)
    }

    /// Like `match_one` but includes the indices of the chars in the haystack that
    /// matched the needle. Useful for displaying matched indices in the UI
    fn match_one_indices(&mut self, haystack: &str, index: u32) -> Option<PyMatchIndices> {
        self.inner
            .match_one_indices(haystack, index)
            .map(Into::into)
    }

    /// Updates the pattern (str or Pattern), keeping the config. Skipped if the
    /// pattern is the same as the previous one
    fn set_pattern(&mut self, pattern: PatternArg) {
        self.inner.set_pattern(frizbee::Pattern::from(pattern));
    }

    /// Updates the patterns (str or Pattern items), keeping the config. Skipped if
    /// the patterns are the same as the previous ones
    fn set_patterns(&mut self, patterns: Vec<PatternArg>) {
        let patterns: Vec<frizbee::Pattern> = patterns.into_iter().map(Into::into).collect();
        self.inner.set_patterns(&patterns);
    }

    /// Updates the config, rebuilding it from the kwargs; omitted kwargs reset to
    /// their defaults (this is not a merge with the current config). Skipped if the
    /// config is the same as the previous one
    #[pyo3(signature = (*, max_typos = frizbee::Config::default().max_typos, max_items = None,
        casing = None, unicode = None, matching = None, sort = None, scoring = None))]
    #[allow(clippy::too_many_arguments)]
    fn set_config(
        &mut self,
        max_typos: Option<u16>,
        max_items: Option<u32>,
        casing: Option<&str>,
        unicode: Option<&str>,
        matching: Option<&str>,
        sort: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<()> {
        let config = build_config(max_typos, casing, unicode, matching, sort, scoring)?;
        self.inner.set_config(config);
        self.max_items = max_items;
        Ok(())
    }

    /// The current patterns
    #[getter]
    fn patterns(&self) -> Vec<PyPattern> {
        self.inner
            .patterns()
            .iter()
            .map(|pattern| PyPattern {
                inner: pattern.clone(),
            })
            .collect()
    }

    #[getter]
    fn max_typos(&self) -> Option<u16> {
        self.inner.config().max_typos
    }

    #[getter]
    fn max_items(&self) -> Option<u32> {
        self.max_items
    }

    #[getter]
    fn casing(&self) -> &'static str {
        casing_to_str(self.inner.config().casing)
    }

    #[getter]
    fn unicode(&self) -> &'static str {
        unicode_to_str(self.inner.config().unicode)
    }

    #[getter]
    fn matching(&self) -> &'static str {
        matching_to_str(self.inner.config().matching)
    }

    #[getter]
    fn sort(&self) -> &'static str {
        sort_to_str(self.inner.config().sort)
    }

    #[getter]
    fn scoring(&self) -> PyScoring {
        self.inner.config().scoring.clone().into()
    }
}
