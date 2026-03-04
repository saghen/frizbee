use pyo3::prelude::*;

// ── Scoring ──────────────────────────────────────────────────────────

#[pyclass(name = "Scoring", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyScoring {
    #[pyo3(get)]
    pub match_score: u16,
    #[pyo3(get)]
    pub mismatch_penalty: u16,
    #[pyo3(get)]
    pub gap_open_penalty: u16,
    #[pyo3(get)]
    pub gap_extend_penalty: u16,
    #[pyo3(get)]
    pub prefix_bonus: u16,
    #[pyo3(get)]
    pub capitalization_bonus: u16,
    #[pyo3(get)]
    pub matching_case_bonus: u16,
    #[pyo3(get)]
    pub exact_match_bonus: u16,
    #[pyo3(get)]
    pub delimiter_bonus: u16,
}

#[pymethods]
impl PyScoring {
    #[new]
    #[pyo3(signature = (
        match_score = None,
        mismatch_penalty = None,
        gap_open_penalty = None,
        gap_extend_penalty = None,
        prefix_bonus = None,
        capitalization_bonus = None,
        matching_case_bonus = None,
        exact_match_bonus = None,
        delimiter_bonus = None,
    ))]
    fn new(
        match_score: Option<u16>,
        mismatch_penalty: Option<u16>,
        gap_open_penalty: Option<u16>,
        gap_extend_penalty: Option<u16>,
        prefix_bonus: Option<u16>,
        capitalization_bonus: Option<u16>,
        matching_case_bonus: Option<u16>,
        exact_match_bonus: Option<u16>,
        delimiter_bonus: Option<u16>,
    ) -> Self {
        let defaults = frizbee::Scoring::default();
        PyScoring {
            match_score: match_score.unwrap_or(defaults.match_score),
            mismatch_penalty: mismatch_penalty.unwrap_or(defaults.mismatch_penalty),
            gap_open_penalty: gap_open_penalty.unwrap_or(defaults.gap_open_penalty),
            gap_extend_penalty: gap_extend_penalty.unwrap_or(defaults.gap_extend_penalty),
            prefix_bonus: prefix_bonus.unwrap_or(defaults.prefix_bonus),
            capitalization_bonus: capitalization_bonus.unwrap_or(defaults.capitalization_bonus),
            matching_case_bonus: matching_case_bonus.unwrap_or(defaults.matching_case_bonus),
            exact_match_bonus: exact_match_bonus.unwrap_or(defaults.exact_match_bonus),
            delimiter_bonus: delimiter_bonus.unwrap_or(defaults.delimiter_bonus),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Scoring(match_score={}, mismatch_penalty={}, gap_open_penalty={}, gap_extend_penalty={}, \
             prefix_bonus={}, capitalization_bonus={}, matching_case_bonus={}, exact_match_bonus={}, \
             delimiter_bonus={})",
            self.match_score,
            self.mismatch_penalty,
            self.gap_open_penalty,
            self.gap_extend_penalty,
            self.prefix_bonus,
            self.capitalization_bonus,
            self.matching_case_bonus,
            self.exact_match_bonus,
            self.delimiter_bonus,
        )
    }
}

impl From<&PyScoring> for frizbee::Scoring {
    fn from(s: &PyScoring) -> Self {
        frizbee::Scoring {
            match_score: s.match_score,
            mismatch_penalty: s.mismatch_penalty,
            gap_open_penalty: s.gap_open_penalty,
            gap_extend_penalty: s.gap_extend_penalty,
            prefix_bonus: s.prefix_bonus,
            capitalization_bonus: s.capitalization_bonus,
            matching_case_bonus: s.matching_case_bonus,
            exact_match_bonus: s.exact_match_bonus,
            delimiter_bonus: s.delimiter_bonus,
        }
    }
}

// ── Config ───────────────────────────────────────────────────────────

#[pyclass(name = "Config", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyConfig {
    #[pyo3(get)]
    pub max_typos: Option<u16>,
    #[pyo3(get)]
    pub sort: bool,
    #[pyo3(get)]
    pub scoring: PyScoring,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (max_typos = Some(0), sort = true, scoring = None))]
    fn new(max_typos: Option<u16>, sort: bool, scoring: Option<PyScoring>) -> Self {
        PyConfig {
            max_typos,
            sort,
            scoring: scoring.unwrap_or_else(|| {
                PyScoring::new(None, None, None, None, None, None, None, None, None)
            }),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Config(max_typos={}, sort={})",
            self.max_typos
                .map(|t| t.to_string())
                .unwrap_or("None".to_string()),
            if self.sort { "True" } else { "False" },
        )
    }
}

impl From<&PyConfig> for frizbee::Config {
    fn from(c: &PyConfig) -> Self {
        frizbee::Config {
            max_typos: c.max_typos,
            sort: c.sort,
            scoring: (&c.scoring).into(),
        }
    }
}

// ── Match ────────────────────────────────────────────────────────────

#[pyclass(name = "Match", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMatch {
    #[pyo3(get)]
    pub score: u16,
    #[pyo3(get)]
    pub index: u32,
    #[pyo3(get)]
    pub exact: bool,
}

#[pymethods]
impl PyMatch {
    fn __repr__(&self) -> String {
        format!(
            "Match(score={}, index={}, exact={})",
            self.score,
            self.index,
            if self.exact { "True" } else { "False" },
        )
    }
}

impl From<frizbee::Match> for PyMatch {
    fn from(m: frizbee::Match) -> Self {
        PyMatch {
            score: m.score,
            index: m.index,
            exact: m.exact,
        }
    }
}

// ── MatchIndices ─────────────────────────────────────────────────────

#[pyclass(name = "MatchIndices", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMatchIndices {
    #[pyo3(get)]
    pub score: u16,
    #[pyo3(get)]
    pub index: u32,
    #[pyo3(get)]
    pub exact: bool,
    #[pyo3(get)]
    pub indices: Vec<usize>,
}

#[pymethods]
impl PyMatchIndices {
    fn __repr__(&self) -> String {
        format!(
            "MatchIndices(score={}, index={}, exact={}, indices={:?})",
            self.score,
            self.index,
            if self.exact { "True" } else { "False" },
            self.indices,
        )
    }
}

impl From<frizbee::MatchIndices> for PyMatchIndices {
    fn from(m: frizbee::MatchIndices) -> Self {
        PyMatchIndices {
            score: m.score,
            index: m.index,
            exact: m.exact,
            indices: m.indices,
        }
    }
}

// ── Matcher (stateful) ───────────────────────────────────────────────

#[pyclass(name = "Matcher")]
pub struct PyMatcher {
    inner: frizbee::Matcher,
}

#[pymethods]
impl PyMatcher {
    #[new]
    #[pyo3(signature = (needle, config = None))]
    fn new(needle: &str, config: Option<&PyConfig>) -> Self {
        let cfg = config.map(|c| c.into()).unwrap_or_default();
        PyMatcher {
            inner: frizbee::Matcher::new(needle, &cfg),
        }
    }

    fn set_needle(&mut self, needle: &str) {
        self.inner.set_needle(needle);
    }

    fn set_config(&mut self, config: &PyConfig) {
        let cfg: frizbee::Config = config.into();
        self.inner.set_config(&cfg);
    }

    #[pyo3(signature = (haystacks))]
    fn match_list(&mut self, haystacks: Vec<String>) -> Vec<PyMatch> {
        self.inner
            .match_list(&haystacks)
            .into_iter()
            .map(PyMatch::from)
            .collect()
    }

    #[pyo3(signature = (haystacks))]
    fn match_list_indices(&mut self, haystacks: Vec<String>) -> Vec<PyMatchIndices> {
        self.inner
            .match_list_indices(&haystacks)
            .into_iter()
            .map(PyMatchIndices::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Matcher(needle={:?})", self.inner.needle)
    }
}

// ── Free functions ───────────────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "match_list", signature = (needle, haystacks, config = None))]
pub fn py_match_list(
    needle: &str,
    haystacks: Vec<String>,
    config: Option<&PyConfig>,
) -> Vec<PyMatch> {
    let cfg = config.map(|c| c.into()).unwrap_or_default();
    frizbee::match_list(needle, &haystacks, &cfg)
        .into_iter()
        .map(PyMatch::from)
        .collect()
}

#[pyfunction]
#[pyo3(name = "match_list_indices", signature = (needle, haystacks, config = None))]
pub fn py_match_list_indices(
    needle: &str,
    haystacks: Vec<String>,
    config: Option<&PyConfig>,
) -> Vec<PyMatchIndices> {
    let cfg = config.map(|c| c.into()).unwrap_or_default();
    frizbee::match_list_indices(needle, &haystacks, &cfg)
        .into_iter()
        .map(PyMatchIndices::from)
        .collect()
}

#[pyfunction]
#[pyo3(name = "match_list_parallel", signature = (needle, haystacks, config = None, threads = None))]
pub fn py_match_list_parallel(
    needle: &str,
    haystacks: Vec<String>,
    config: Option<&PyConfig>,
    threads: Option<usize>,
) -> Vec<PyMatch> {
    let cfg = config.map(|c| c.into()).unwrap_or_default();
    let num_threads = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    frizbee::match_list_parallel(needle, &haystacks, &cfg, num_threads)
        .into_iter()
        .map(PyMatch::from)
        .collect()
}
