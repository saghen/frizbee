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
   * Matches a list of haystacks, returning a list of `Match` values. This API
   * provides the most performant path when matching on lists.
   *
   * Pass an owned `Haystacks` so that no strings cross the boundary per call
   */
  matchList(haystacks: string[] | Haystacks): Match[];
  /**
   * Matches a list of haystacks, returning a list of `MatchIndices` which are
   * equivalent to `Match` except they include the indices of the matched
   * characters in the haystack.
   *
   * This API has not been optimized for performance, and should only be used on
   * small lists or after matching a list of haystacks with `matchList`. Useful
   * for displaying matched indices in the UI
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

/// Primary entrypoint for fuzzy matching
///
/// `Matcher` compiles the pattern once, allocates memory for the Smith Waterman
/// matrix, and reuses the selected SIMD backend.
///
/// Ideally, only construct these at most once per list. They're cheap to
/// construct, but end up being expensive if you construct them for each item in
/// your list.
///
/// Instances hold memory in the wasm heap: call `.free()` when done, or rely on
/// `FinalizationRegistry` to eventually collect it.
#[wasm_bindgen]
pub struct Matcher {
    inner: frizbee::Matcher,
    /// Truncates the `matchList` results after sorting; `None` means unlimited
    limit: Option<u32>,
}

impl Matcher {
    fn truncate<T>(&self, matches: &mut Vec<T>) {
        if let Some(limit) = self.limit {
            matches.truncate(limit as usize);
        }
    }
}

#[wasm_bindgen]
impl Matcher {
    /// Creates a matcher from a single needle, matched literally. Use
    /// `fromQuery` for query syntax and `fromPatterns` for multi-pattern
    /// queries
    #[wasm_bindgen(constructor)]
    pub fn new(needle: &str, config: Option<ConfigObject>) -> Result<Matcher, JsError> {
        let (config, limit) = config_from_js(config)?;
        let inner = frizbee::Matcher::new(needle, &config);
        Ok(Matcher { inner, limit })
    }

    /// Parses a single query atom, where special syntax changes the matching
    /// mode:
    ///
    /// - `foo` - defers to matching Config, which defaults to fuzzy
    /// - `^foo` - prefix
    /// - `foo$` - suffix
    /// - `'foo` - substring
    /// - `^foo$` - exact
    /// - `!foo` - negated, substring unless combined with the syntax above
    ///
    /// Any special character can be escaped with a backslash, e.g. `\!foo`,
    /// `\^foo`, `foo\$` or `\'foo` match the literal leading/trailing
    /// character, and `foo\ bar` matches the literal space.
    #[wasm_bindgen(js_name = fromQuery)]
    pub fn from_query(query: &str, config: Option<ConfigObject>) -> Result<Matcher, JsError> {
        let (config, limit) = config_from_js(config)?;
        let inner = frizbee::Matcher::from_query(query, &config);
        Ok(Matcher { inner, limit })
    }

    /// Creates a matcher from a list of patterns, matched independently. A
    /// haystack matches when all of the patterns match, where the score is
    /// the sum of each pattern's score
    #[wasm_bindgen(js_name = fromPatterns)]
    pub fn from_patterns(
        patterns: PatternArray,
        config: Option<ConfigObject>,
    ) -> Result<Matcher, JsError> {
        let (config, limit) = config_from_js(config)?;
        let patterns = patterns_from_js(patterns)?;
        let inner = frizbee::Matcher::from_patterns(&patterns, &config);
        Ok(Matcher { inner, limit })
    }

    /// Matches a list of haystacks, returning a list of `Match` values. This
    /// API provides the most performant path when matching on lists.
    ///
    /// Through the package entry, also accepts a [`Haystacks`] list (forwarded
    /// to `matchHaystacks`) so that no strings cross the boundary per call
    #[wasm_bindgen(js_name = matchList, skip_typescript)]
    pub fn match_list(&mut self, haystacks: Vec<String>) -> MatchArray {
        // borrow as &str so the core monomorphizes once for H = &str (shared with the
        // Haystacks path), halving the largest family of functions in the binary
        let haystacks: Vec<&str> = haystacks.iter().map(String::as_str).collect();
        let mut matches = self.inner.match_list(&haystacks);
        self.truncate(&mut matches);
        matches_to_js(&matches)
    }

    /// Matches a list of haystacks, returning a list of `MatchIndices` which
    /// are equivalent to `Match` except they include the indices of the matched
    /// characters in the haystack.
    ///
    /// This API has not been optimized for performance, and should only be used
    /// on small lists or after matching a list of haystacks with `matchList`.
    /// Useful for displaying matched indices in the UI
    #[wasm_bindgen(js_name = matchListIndices, skip_typescript)]
    pub fn match_list_indices(&mut self, haystacks: Vec<String>) -> MatchIndicesArray {
        let haystacks: Vec<&str> = haystacks.iter().map(String::as_str).collect();
        let mut matches = self.inner.match_list_indices(&haystacks);
        self.truncate(&mut matches);
        matches_indices_to_js(&matches)
    }

    /// Like `matchList` but matches an owned [`Haystacks`] list. No strings
    /// cross the boundary per call, making this the most performant path when
    /// matching the same list repeatedly. Through the package entry,
    /// `matchList` accepts a `Haystacks` directly and forwards here
    #[wasm_bindgen(js_name = matchHaystacks)]
    pub fn match_haystacks(&mut self, haystacks: &Haystacks) -> MatchArray {
        let mut matches = self.inner.match_list(&haystacks.as_strs());
        self.truncate(&mut matches);
        matches_to_js(&matches)
    }

    /// Like `matchListIndices` but matches an owned [`Haystacks`] list (see
    /// `matchHaystacks`)
    #[wasm_bindgen(js_name = matchHaystacksIndices)]
    pub fn match_haystacks_indices(&mut self, haystacks: &Haystacks) -> MatchIndicesArray {
        let mut matches = self.inner.match_list_indices(&haystacks.as_strs());
        self.truncate(&mut matches);
        matches_indices_to_js(&matches)
    }

    /// Matches a single haystack, returning its `Match` if it passes. This
    /// API performs much slower than the `matchList` API with `Haystacks`
    #[wasm_bindgen(js_name = matchOne)]
    pub fn match_one(&mut self, haystack: &str, index: u32) -> Option<MatchObject> {
        self.inner
            .match_one(haystack, index)
            .map(|m| match_to_js(&m).unchecked_into())
    }

    /// Like `matchOne` but includes the indices of the chars in the haystack
    /// that matched the needle. Useful for displaying matched indices in the
    /// UI
    #[wasm_bindgen(js_name = matchOneIndices)]
    pub fn match_one_indices(&mut self, haystack: &str, index: u32) -> Option<MatchIndicesObject> {
        self.inner
            .match_one_indices(haystack, index)
            .map(|m| match_indices_to_js(&m).unchecked_into())
    }

    /// Updates the pattern (string or `Pattern`), keeping the config. Skipped
    /// if the pattern is the same as the previous one
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

    /// Updates the config, keeping the patterns. Omitted fields reset to their
    /// defaults (this is not a merge with the current config). Skipped if the
    /// config is the same as the previous one
    #[wasm_bindgen(js_name = setConfig)]
    pub fn set_config(&mut self, config: ConfigObject) -> Result<(), JsError> {
        let (config, limit) = config_from_js(Some(config))?;
        self.inner.set_config(config);
        self.limit = limit;
        Ok(())
    }
}
