//! Minimal E2E checks that the binding wires through to the core correctly.
//! Matching behavior itself is covered by the core crate's test suite; these only
//! assert the boundary broadly works: construction, matching, ordering, the owned
//! haystacks path and query syntax

#![cfg(target_arch = "wasm32")]

use frizbee_wasm::{Haystacks, Matcher, parse_query};
use serde::Deserialize;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

#[derive(Debug, Deserialize, PartialEq)]
struct Match {
    score: u16,
    index: u32,
    exact: bool,
}

#[derive(Debug, Deserialize)]
struct MatchIndices {
    index: u32,
    indices: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct Pattern {
    needle: String,
    negated: bool,
}

fn from_js<T: serde::de::DeserializeOwned>(value: JsValue) -> T {
    serde_wasm_bindgen::from_value(value).unwrap()
}

fn haystacks(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[wasm_bindgen_test]
fn match_list() {
    let mut matcher = Matcher::new("foo", None).unwrap();
    let matches: Vec<Match> = from_js(
        matcher
            .match_list(haystacks(&["foo", "prelude", "xfoo", "foobar"]))
            .into(),
    );
    let mut indices: Vec<u32> = matches.iter().map(|m| m.index).collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 2, 3]);
    assert_eq!(matches[0].index, 0);
    assert!(matches[0].exact);
    assert!(matches.windows(2).all(|w| w[0].score >= w[1].score));
}

#[wasm_bindgen_test]
fn owned_haystacks_equal_match_list() {
    let mut matcher = Matcher::new("foo", None).unwrap();
    let list: Vec<Match> = from_js(
        matcher
            .match_list(haystacks(&["foo", "xfoo", "zzz"]))
            .into(),
    );

    let mut owned = Haystacks::new(Some(haystacks(&["foo", "xfoo"])));
    owned.push(haystacks(&["zzz"]));
    assert_eq!(owned.length(), 3);
    let matches: Vec<Match> = from_js(matcher.match_haystacks(&owned).into());
    assert_eq!(matches, list);

    owned.clear();
    assert_eq!(owned.length(), 0);
    let matches: Vec<Match> = from_js(matcher.match_haystacks(&owned).into());
    assert!(matches.is_empty());
}

#[wasm_bindgen_test]
fn match_one_and_indices() {
    let mut matcher = Matcher::new("foo", None).unwrap();
    assert!(matcher.match_one("zzz", 0).is_none());
    let m: Match = from_js(matcher.match_one("foobar", 5).unwrap().into());
    assert_eq!(m.index, 5);

    let m: MatchIndices = from_js(matcher.match_one_indices("foobar", 0).unwrap().into());
    assert_eq!(m.index, 0);
    let mut indices = m.indices;
    indices.sort_unstable();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[wasm_bindgen_test]
fn from_query_and_parse_query() {
    let mut matcher = Matcher::from_query("foo !^bar", None).unwrap();
    let matches: Vec<Match> = from_js(
        matcher
            .match_list(haystacks(&["foo", "barfoo", "foobar"]))
            .into(),
    );
    let mut indices: Vec<u32> = matches.iter().map(|m| m.index).collect();
    indices.sort_unstable();
    // "barfoo" starts with "bar"
    assert_eq!(indices, vec![0, 2]);

    let patterns: Vec<Pattern> = from_js(parse_query("foo !^bar").unwrap().into());
    assert_eq!(patterns.len(), 2);
    assert_eq!(patterns[1].needle, "bar");
    assert!(patterns[1].negated);
}
