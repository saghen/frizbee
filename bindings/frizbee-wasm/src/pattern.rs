//! Mirror of the core `Pattern` plus query parsing, crossing the boundary as
//! plain JS objects via `js_sys::Reflect` (see the note in [`crate::config`] on
//! why not serde), with its hand-written TS declaration kept next to the struct
//! it describes.

use js_sys::{Array, Object};
use wasm_bindgen::prelude::*;

use crate::config::{
    CaseMatching, Matching, Scoring, UnicodeMatching, get_opt, opt_bool, opt_f64, opt_object,
    parse_max_typos,
};
use crate::matcher::set;

#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_TYPES: &'static str = r#"
export interface Pattern {
  needle: string;
  /** Haystacks matching this atom are excluded */
  negated?: boolean;
  /**
   * Per-pattern override; `undefined` inherits from `Config`. Because the config's
   * `maxTypos` is itself optional, there is no way to request unlimited typos for a
   * single pattern (`Infinity` is rejected) while the matcher's config sets a limit
   */
  maxTypos?: number;
  /** Per-pattern override; `undefined` inherits from `Config` */
  casing?: CaseMatching;
  /** Per-pattern override; `undefined` inherits from `Config` */
  unicode?: UnicodeMatching;
  /** Per-pattern override; `undefined` inherits from `Config` */
  matching?: Matching;
  /** Per-pattern override; `undefined` inherits from `Config` */
  scoring?: Scoring;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Pattern")]
    pub type PatternObject;
    #[wasm_bindgen(typescript_type = "string | Pattern")]
    pub type PatternLike;
    #[wasm_bindgen(typescript_type = "Pattern[]")]
    pub type PatternArray;
}

/// Core's `PatternConfig` cannot override `max_typos` back to unlimited (`None`
/// means inherit), so `Infinity` is rejected at the pattern level
fn parse_pattern_max_typos(value: f64) -> Result<u16, JsError> {
    parse_max_typos(value)?.ok_or_else(|| {
        JsError::new(
            "per-pattern maxTypos cannot be Infinity, leave it undefined to inherit the config's value",
        )
    })
}

/// Mirror of [`frizbee::Pattern`] accepted as a plain JS object. Every field
/// besides `needle` is optional; missing overrides inherit from the matcher's
/// [`Config`](crate::config::Config)
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub needle: String,
    pub negated: bool,
    pub max_typos: Option<f64>,
    pub casing: Option<CaseMatching>,
    pub unicode: Option<UnicodeMatching>,
    pub matching: Option<Matching>,
    pub scoring: Option<Scoring>,
}

impl Pattern {
    fn from_js(pattern: &JsValue) -> Result<Self, JsError> {
        if !pattern.is_object() {
            return Err(JsError::new("invalid pattern: expected an object"));
        }
        let needle = get_opt(pattern, "needle")
            .and_then(|needle| needle.as_string())
            .ok_or_else(|| JsError::new("invalid pattern: needle must be a string"))?;
        Ok(Self {
            needle,
            negated: opt_bool(pattern, "negated")?.unwrap_or(false),
            max_typos: opt_f64(pattern, "maxTypos")?,
            casing: CaseMatching::opt_from_js(pattern, "casing")?,
            unicode: UnicodeMatching::opt_from_js(pattern, "unicode")?,
            matching: Matching::opt_from_js(pattern, "matching")?,
            scoring: opt_object(pattern, "scoring")?
                .map(|scoring| Scoring::from_js(&scoring))
                .transpose()?,
        })
    }

    fn to_js(&self) -> JsValue {
        let obj = Object::new();
        set(&obj, "needle", self.needle.as_str());
        set(&obj, "negated", self.negated);
        if let Some(max_typos) = self.max_typos {
            set(&obj, "maxTypos", max_typos);
        }
        if let Some(casing) = self.casing {
            set(&obj, "casing", casing.as_str());
        }
        if let Some(unicode) = self.unicode {
            set(&obj, "unicode", unicode.as_str());
        }
        if let Some(matching) = self.matching {
            set(&obj, "matching", matching.as_str());
        }
        if let Some(scoring) = &self.scoring {
            let scoring_obj = Object::new();
            for (key, value) in [
                ("matchScore", scoring.match_score),
                ("mismatchPenalty", scoring.mismatch_penalty),
                ("gapOpenPenalty", scoring.gap_open_penalty),
                ("gapExtendPenalty", scoring.gap_extend_penalty),
                ("prefixBonus", scoring.prefix_bonus),
                ("capitalizationBonus", scoring.capitalization_bonus),
                ("matchingCaseBonus", scoring.matching_case_bonus),
                ("exactMatchBonus", scoring.exact_match_bonus),
                ("delimiterBonus", scoring.delimiter_bonus),
            ] {
                if let Some(value) = value {
                    set(&scoring_obj, key, value);
                }
            }
            set(&obj, "scoring", scoring_obj);
        }
        obj.into()
    }

    fn to_core(&self) -> Result<frizbee::Pattern, JsError> {
        let config = frizbee::PatternConfig::default()
            .max_typos(self.max_typos.map(parse_pattern_max_typos).transpose()?)
            .casing(self.casing.map(Into::into))
            .unicode(self.unicode.map(Into::into))
            .matching(self.matching.map(Into::into))
            .scoring(self.scoring.as_ref().map(Scoring::to_core));
        Ok(frizbee::Pattern::new(&self.needle, config).negated(self.negated))
    }

    fn from_core(pattern: &frizbee::Pattern) -> Self {
        Self {
            needle: pattern.needle.clone(),
            negated: pattern.negated,
            max_typos: pattern.config.max_typos.map(f64::from),
            casing: pattern.config.casing.map(Into::into),
            unicode: pattern.config.unicode.map(Into::into),
            matching: pattern.config.matching.map(Into::into),
            scoring: pattern.config.scoring.as_ref().map(Scoring::from_core),
        }
    }
}

pub(crate) fn pattern_from_js(pattern: JsValue) -> Result<frizbee::Pattern, JsError> {
    if let Some(needle) = pattern.as_string() {
        return Ok(needle.into());
    }
    Pattern::from_js(&pattern)?.to_core()
}

pub(crate) fn patterns_from_js(patterns: PatternArray) -> Result<Vec<frizbee::Pattern>, JsError> {
    let patterns: JsValue = patterns.into();
    if !Array::is_array(&patterns) {
        return Err(JsError::new("invalid patterns: expected an array"));
    }
    Array::from(&patterns)
        .iter()
        .map(|pattern| Pattern::from_js(&pattern)?.to_core())
        .collect()
}

/// Parses a query of whitespace separated atoms into patterns, where special
/// syntax changes the matching mode: `foo` (fuzzy), `^foo` (prefix), `foo$`
/// (suffix), `'foo` (substring), `^foo$` (exact) and `!foo` (negated, substring
/// unless combined with the syntax above). Any special character can be escaped
/// with a backslash, e.g. `\!foo` or `foo\$` match the literal leading/trailing
/// character, and `foo\ bar` matches the literal space. Atoms with an empty
/// needle, e.g. `!` or `^$`, are dropped.
///
/// The returned patterns carry only the matching mode derived from the syntax.
/// Any other per-pattern override is left undefined and inherits the matcher's
/// config; adjust the returned patterns before passing them to
/// `Matcher.fromPatterns` to override config per-pattern
#[wasm_bindgen(js_name = parseQuery)]
pub fn parse_query(query: &str) -> Result<PatternArray, JsError> {
    let patterns = frizbee::Pattern::parse_query(query)
        .iter()
        .map(|pattern| Pattern::from_core(pattern).to_js())
        .collect::<Array>();
    Ok(patterns.unchecked_into())
}
