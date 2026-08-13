//! Browser WASM bindings for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy
//! string matching.
//!
//! Config and Pattern cross the boundary as plain JS objects (camelCase keys,
//! enums as strings) parsed via `js_sys::Reflect` mirror structs. Haystacks
//! cross as `string[]` through wasm-bindgen's per-element string glue, or live
//! in a wasm-owned [`Haystacks`] arena filled by the wrapper with a single
//! `TextEncoder` pass — the primary path for keystroke loops.

mod config;
mod haystacks;
mod matcher;
mod pattern;

pub use config::{
    CaseMatching, ConfigObject, Matching, Scoring, ScoringObject, SortStrategy, UnicodeMatching,
    max_needle_len,
};
pub use haystacks::Haystacks;
pub use matcher::{MatchArray, MatchIndicesArray, MatchIndicesObject, MatchObject, Matcher};
pub use pattern::{Pattern, PatternArray, PatternLike, PatternObject, parse_query};

// Talc over Rust's default dlmalloc: ~4 KB smaller and faster
#[cfg(all(not(target_feature = "atomics"), target_family = "wasm"))]
#[global_allocator]
static TALC: talc::wasm::WasmDynamicTalc = talc::wasm::new_wasm_dynamic_allocator();
