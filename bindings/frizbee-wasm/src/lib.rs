//! WASM bindings for frizbee, SIMD fuzzy string matching.

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
