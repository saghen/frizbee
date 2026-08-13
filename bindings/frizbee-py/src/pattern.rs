//! The `Pattern` class with its per-pattern config overrides, plus query
//! parsing.

use pyo3::prelude::*;

use crate::config::{
    PyScoring, casing_to_str, matching_to_str, parse_casing, parse_matching, parse_unicode,
    unicode_to_str,
};

/// A single pattern to match, optionally overriding parts of the matcher
/// config. Every override defaults to `None` = inherit from the matcher config
/// — including `max_typos`, where an unlimited per-pattern override is
/// inexpressible in core (set `max_typos=None` on the matcher config for
/// unlimited typos instead)
#[pyclass(name = "Pattern", frozen, eq, from_py_object, module = "frizbee")]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct PyPattern {
    pub(crate) inner: frizbee::Pattern,
}

#[pymethods]
impl PyPattern {
    #[new]
    #[pyo3(signature = (needle, *, negated = false, matching = None, max_typos = None,
        casing = None, unicode = None, scoring = None))]
    fn new(
        needle: &str,
        negated: bool,
        matching: Option<&str>,
        max_typos: Option<u16>,
        casing: Option<&str>,
        unicode: Option<&str>,
        scoring: Option<PyScoring>,
    ) -> PyResult<Self> {
        let config = frizbee::PatternConfig {
            max_typos,
            casing: casing.map(parse_casing).transpose()?,
            unicode: unicode.map(parse_unicode).transpose()?,
            matching: matching.map(parse_matching).transpose()?,
            scoring: scoring.map(Into::into),
        };
        Ok(Self {
            inner: frizbee::Pattern::new(needle, config).negated(negated),
        })
    }

    /// Raw atom text, e.g. `!^foo` when parsed from a query
    #[getter]
    fn pattern(&self) -> &str {
        &self.inner.pattern
    }

    /// Text to match with the syntax stripped, e.g. `foo`
    #[getter]
    fn needle(&self) -> &str {
        &self.inner.needle
    }

    /// Haystacks matching this atom are excluded
    #[getter]
    fn negated(&self) -> bool {
        self.inner.negated
    }

    /// Per-pattern override for `matching`; `None` inherits it
    #[getter]
    fn matching(&self) -> Option<&'static str> {
        self.inner.config.matching.map(matching_to_str)
    }

    /// Per-pattern override for `max_typos`; `None` inherits it. Because the
    /// config's `max_typos` is itself optional, there is no way to request
    /// unlimited typos for a single pattern while the matcher's config sets a
    /// limit
    #[getter]
    fn max_typos(&self) -> Option<u16> {
        self.inner.config.max_typos
    }

    /// Per-pattern override for `casing`; `None` inherits it
    #[getter]
    fn casing(&self) -> Option<&'static str> {
        self.inner.config.casing.map(casing_to_str)
    }

    /// Per-pattern override for `unicode`; `None` inherits it
    #[getter]
    fn unicode(&self) -> Option<&'static str> {
        self.inner.config.unicode.map(unicode_to_str)
    }

    /// Per-pattern override for `scoring`; `None` inherits it
    #[getter]
    fn scoring(&self) -> Option<PyScoring> {
        self.inner.config.scoring.clone().map(Into::into)
    }

    fn __repr__(&self) -> String {
        let mut repr = format!("Pattern({:?}", self.inner.needle);
        if self.inner.negated {
            repr.push_str(", negated=True");
        }
        let config = &self.inner.config;
        if let Some(matching) = config.matching {
            repr.push_str(&format!(", matching={:?}", matching_to_str(matching)));
        }
        if let Some(max_typos) = config.max_typos {
            repr.push_str(&format!(", max_typos={max_typos}"));
        }
        if let Some(casing) = config.casing {
            repr.push_str(&format!(", casing={:?}", casing_to_str(casing)));
        }
        if let Some(unicode) = config.unicode {
            repr.push_str(&format!(", unicode={:?}", unicode_to_str(unicode)));
        }
        if config.scoring.is_some() {
            repr.push_str(", scoring=...");
        }
        repr.push(')');
        repr
    }
}

/// Accepts either a `str` (matched literally, like the core `impl
/// Into<Pattern>`) or a `Pattern`
#[derive(FromPyObject)]
pub(crate) enum PatternArg {
    Pattern(PyPattern),
    Text(String),
}

impl From<PatternArg> for frizbee::Pattern {
    fn from(arg: PatternArg) -> Self {
        match arg {
            PatternArg::Pattern(pattern) => pattern.inner,
            PatternArg::Text(needle) => needle.into(),
        }
    }
}

/// Parses a query of whitespace separated atoms (see `Matcher.from_query` for
/// the atom syntax), e.g. `foo !^bar` matches haystacks that fuzzy match `foo`
/// and don't start with `bar`. Escape a literal space with a backslash, e.g.
/// `foo\ bar` is a single atom. Atoms with an empty needle, e.g. `!` or `^$`,
/// are dropped.
///
/// The returned patterns carry only the matching mode derived from the syntax.
/// Any other per-pattern override is left as `None` and inherits the matcher's
/// config; rebuild patterns to override per-pattern before passing them to
/// `Matcher.from_patterns`
#[pyfunction]
pub(crate) fn parse_query(query: &str) -> Vec<PyPattern> {
    frizbee::Pattern::parse_query(query)
        .into_iter()
        .map(|inner| PyPattern { inner })
        .collect()
}
