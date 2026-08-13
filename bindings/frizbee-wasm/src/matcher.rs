//! The `Matcher` class, the match result marshalling into plain JS objects, and
//! the hand-written TS declarations for the result types — including the
//! interface merge that lets `matchList`/`matchListIndices` accept an owned
//! `Haystacks`.

use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

use crate::config::{ConfigObject, config_from_js};
use crate::haystacks::Haystacks;
use crate::pattern::{PatternArray, PatternLike, pattern_from_js, patterns_from_js};

#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_TYPES: &'static str = r#"
/** Result of a fuzzy match, containing the score and index in the haystack */
export interface Match {
  /** Score of the match, higher is better */
  score: number;
  /** Index of the match in the original list of haystacks */
  index: number;
  /** Matched the needle exactly (e.g. "foo" on "foo") */
  exact: boolean;
}

/**
 * Like `Match` but includes the indices of the chars in the haystack that matched
 * the needle in reverse order
 */
export interface MatchIndices extends Match {
  /** Indices of the chars in the haystack that matched the needle in reverse order */
  indices: number[];
}

/* Merges into the generated Matcher class: matchList/matchListIndices also accept an
 * owned Haystacks (the package entry forwards it to matchHaystacks) */
export interface Matcher {
  /**
   * Matches a list of haystacks, returning the matches ordered by the config's sort
   * strategy. Pass an owned `Haystacks` for the primary path: no strings cross the
   * boundary per call
   */
  matchList(haystacks: string[] | Haystacks): Match[];
  /**
   * Like `matchList`, but each match includes the indices of the chars in the
   * haystack that matched the needle, in reverse order. This API has not been
   * optimized for performance, and should only be used on small lists, e.g. the
   * items in view. Useful for displaying matched indices in the UI
   */
  matchListIndices(haystacks: string[] | Haystacks): MatchIndices[];
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Match")]
    pub type MatchObject;
    #[wasm_bindgen(typescript_type = "Match[]")]
    pub type MatchArray;
    #[wasm_bindgen(typescript_type = "MatchIndices")]
    pub type MatchIndicesObject;
    #[wasm_bindgen(typescript_type = "MatchIndices[]")]
    pub type MatchIndicesArray;
}

/// `obj[key] = value`. Setting a property on a fresh `Object` cannot fail
pub(crate) fn set(obj: &Object, key: &str, value: impl Into<JsValue>) {
    Reflect::set(obj, &JsValue::from_str(key), &value.into()).unwrap_throw();
}

fn match_to_js(m: &frizbee::Match) -> JsValue {
    let obj = Object::new();
    set(&obj, "score", m.score);
    set(&obj, "index", m.index);
    set(&obj, "exact", m.exact);
    obj.into()
}

fn match_indices_to_js(m: &frizbee::MatchIndices) -> JsValue {
    let obj = Object::new();
    set(&obj, "score", m.score);
    set(&obj, "index", m.index);
    set(&obj, "exact", m.exact);
    let indices = m
        .indices
        .iter()
        .map(|&i| JsValue::from(i))
        .collect::<Array>();
    set(&obj, "indices", indices);
    obj.into()
}

fn matches_to_js(matches: &[frizbee::Match]) -> MatchArray {
    let arr = Array::new_with_length(matches.len() as u32);
    for (i, m) in matches.iter().enumerate() {
        arr.set(i as u32, match_to_js(m));
    }
    arr.unchecked_into()
}

fn matches_indices_to_js(matches: &[frizbee::MatchIndices]) -> MatchIndicesArray {
    let arr = Array::new_with_length(matches.len() as u32);
    for (i, m) in matches.iter().enumerate() {
        arr.set(i as u32, match_indices_to_js(m));
    }
    arr.unchecked_into()
}

/// Primary entrypoint for fuzzy matching. Compiles the pattern once, allocates
/// memory for the Smith Waterman matrix, and reuses the selected SIMD backend.
/// Ideally, only construct these at most once per list: they're cheap to
/// construct, but end up being expensive if you construct them for each item in
/// your list.
///
/// Instances hold memory in the wasm heap: call `.free()` when done, or rely on
/// `FinalizationRegistry` to eventually collect it.
#[wasm_bindgen]
pub struct Matcher {
    inner: frizbee::Matcher,
    /// Truncates the `matchList` results after sorting; `None` means unlimited
    max_items: Option<u32>,
}

impl Matcher {
    fn truncate<T>(&self, matches: &mut Vec<T>) {
        if let Some(max_items) = self.max_items {
            matches.truncate(max_items as usize);
        }
    }
}

#[wasm_bindgen]
impl Matcher {
    /// Creates a matcher from a single needle, matched literally (no query
    /// syntax); use `fromQuery` or `fromPatterns` for multi-pattern queries
    #[wasm_bindgen(constructor)]
    pub fn new(needle: &str, config: Option<ConfigObject>) -> Result<Matcher, JsError> {
        let (config, max_items) = config_from_js(config)?;
        let inner = frizbee::Matcher::new(needle, &config);
        Ok(Matcher { inner, max_items })
    }

    /// Shorthand for calling `fromPatterns` with the parsed query (see
    /// `parseQuery`), e.g. `foo !^bar` matches haystacks that fuzzy match
    /// `foo` and don't start with `bar`
    #[wasm_bindgen(js_name = fromQuery)]
    pub fn from_query(query: &str, config: Option<ConfigObject>) -> Result<Matcher, JsError> {
        let (config, max_items) = config_from_js(config)?;
        let inner = frizbee::Matcher::from_query(query, &config);
        Ok(Matcher { inner, max_items })
    }

    /// Creates a matcher from a list of patterns, matched independently. A
    /// haystack matches when all of the patterns match, where the score is
    /// the sum of each pattern's score
    #[wasm_bindgen(js_name = fromPatterns)]
    pub fn from_patterns(
        patterns: PatternArray,
        config: Option<ConfigObject>,
    ) -> Result<Matcher, JsError> {
        let (config, max_items) = config_from_js(config)?;
        let patterns = patterns_from_js(patterns)?;
        let inner = frizbee::Matcher::from_patterns(&patterns, &config);
        Ok(Matcher { inner, max_items })
    }

    /// Matches a list of haystacks, returning the matches ordered by the
    /// config's sort strategy. Through the package entry, also accepts a
    /// [`Haystacks`] list (forwarding to `matchHaystacks`) — the primary
    /// path for keystroke loops
    #[wasm_bindgen(js_name = matchList, skip_typescript)]
    pub fn match_list(&mut self, haystacks: Vec<String>) -> MatchArray {
        // borrow as &str so the core monomorphizes once for H = &str (shared with the
        // Haystacks path), halving the largest family of functions in the binary
        let haystacks: Vec<&str> = haystacks.iter().map(String::as_str).collect();
        let mut matches = self.inner.match_list(&haystacks);
        self.truncate(&mut matches);
        matches_to_js(&matches)
    }

    /// Like `matchList`, but each match includes the indices of the chars in
    /// the haystack that matched the needle, in reverse order. This API has
    /// not been optimized for performance, and should only be used on small
    /// lists, e.g. the items in view. Useful for displaying matched indices
    /// in the UI
    #[wasm_bindgen(js_name = matchListIndices, skip_typescript)]
    pub fn match_list_indices(&mut self, haystacks: Vec<String>) -> MatchIndicesArray {
        let haystacks: Vec<&str> = haystacks.iter().map(String::as_str).collect();
        let mut matches = self.inner.match_list_indices(&haystacks);
        self.truncate(&mut matches);
        matches_indices_to_js(&matches)
    }

    /// Like `matchList`, but matches an owned [`Haystacks`] list: no strings
    /// cross the boundary per call, making this the most performant path
    /// when matching the same list repeatedly. Through the package entry,
    /// `matchList` accepts a `Haystacks` directly and forwards here
    #[wasm_bindgen(js_name = matchHaystacks)]
    pub fn match_haystacks(&mut self, haystacks: &Haystacks) -> MatchArray {
        let mut matches = self.inner.match_list(&haystacks.as_strs());
        self.truncate(&mut matches);
        matches_to_js(&matches)
    }

    /// Like `matchListIndices`, but matches an owned [`Haystacks`] list (see
    /// `matchHaystacks`)
    #[wasm_bindgen(js_name = matchHaystacksIndices)]
    pub fn match_haystacks_indices(&mut self, haystacks: &Haystacks) -> MatchIndicesArray {
        let mut matches = self.inner.match_list_indices(&haystacks.as_strs());
        self.truncate(&mut matches);
        matches_indices_to_js(&matches)
    }

    /// Matches a single haystack, returning its match (with the given `index`)
    /// if it passes. Consider using `matchList` if you have more than one
    /// haystack to match, as it performs significantly better
    #[wasm_bindgen(js_name = matchOne)]
    pub fn match_one(&mut self, haystack: &str, index: u32) -> Option<MatchObject> {
        self.inner
            .match_one(haystack, index)
            .map(|m| match_to_js(&m).unchecked_into())
    }

    /// Like `matchOne`, but includes the indices of the chars in the haystack
    /// that matched the needle, in reverse order. Useful for displaying
    /// matched indices in the UI
    #[wasm_bindgen(js_name = matchOneIndices)]
    pub fn match_one_indices(&mut self, haystack: &str, index: u32) -> Option<MatchIndicesObject> {
        self.inner
            .match_one_indices(haystack, index)
            .map(|m| match_indices_to_js(&m).unchecked_into())
    }

    /// Updates the pattern, keeping the config. Strings match literally (no
    /// query syntax). Skipped if the pattern is the same as the previous
    /// one
    #[wasm_bindgen(js_name = setPattern)]
    pub fn set_pattern(&mut self, pattern: PatternLike) -> Result<(), JsError> {
        let pattern = pattern_from_js(pattern.into())?;
        self.inner.set_pattern(pattern);
        Ok(())
    }

    /// Updates the patterns, keeping the config. Skipped if the patterns are
    /// the same as the previous ones
    #[wasm_bindgen(js_name = setPatterns)]
    pub fn set_patterns(&mut self, patterns: PatternArray) -> Result<(), JsError> {
        let patterns = patterns_from_js(patterns)?;
        self.inner.set_patterns(&patterns);
        Ok(())
    }

    /// Updates the config, keeping the patterns. Skipped if the config is the
    /// same as the previous one
    #[wasm_bindgen(js_name = setConfig)]
    pub fn set_config(&mut self, config: ConfigObject) -> Result<(), JsError> {
        let (config, max_items) = config_from_js(Some(config))?;
        self.inner.set_config(config);
        self.max_items = max_items;
        Ok(())
    }
}
