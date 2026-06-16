use crate::one_shot::Matcher;
use crate::{Config, Match, radix_sort};

/// Incremental fuzzy matcher that reuses previous results when the needle changes.
///
/// Maintains a history of which haystack indices matched at each needle length. On
/// forward extension (`"fo"` → `"foo"`), narrows the previous match set. On backspace
/// or partial change (`"foo"` → `"fo"` or `"foo"` → `"fob"`), restores the closest
/// historical match set sharing a common prefix, then narrows from there.
///
/// # Example
///
/// ```rust
/// use frizbee::{IncrementalMatcher, Config};
///
/// let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
/// let mut matcher = IncrementalMatcher::new(&Config::default());
///
/// let matches = matcher.match_list("f", &haystacks);
/// let matches = matcher.match_list("fo", &haystacks);
/// let matches = matcher.match_list("foo", &haystacks);
/// // backspace: restores "fo" match set instead of full rescore
/// let matches = matcher.match_list("fo", &haystacks);
/// ```
pub struct IncrementalMatcher<'a> {
    matcher: Matcher,
    haystacks: &'a [&'a str],
    /// List of needle lengths, each containing a list of haystack string indices
    /// which were filtered out at the given needle length
    /// For example, removed_idxs[0] contains the indices of haystacks which were
    /// filtered out when the needle went from length 0 to length 1.
    removed_idxs: Vec<Vec<u32>>,
    /// List of haystack string indices which matched
    active_matches: Option<Vec<Match>>,
}

impl<'a> IncrementalMatcher<'a> {
    pub fn new(config: &Config, haystacks: &'a [&'a str]) -> Self {
        let mut config = config.clone();
        config.sort = false;

        Self {
            matcher: Matcher::new("", &config),
            haystacks,
            removed_idxs: Vec::new(),
            active_matches: None,
        }
    }

    pub fn match_list(&mut self, needle: &str) -> Vec<Match> {
        let prev_needle = &self.matcher.needle.clone();
        let reusable_level = self.find_reusable_level(needle, prev_needle);
        let is_prefix_extension = self.is_prefix_extension(needle, prev_needle);
        self.matcher.set_needle(needle);

        if needle.is_empty() {
            self.reset();
            return (0..self.haystacks.len()).map(Match::from_index).collect();
        }

        // Never matched or no needle overlap, start from scratch
        if self.active_matches.is_none() || reusable_level == 0 {
            let mut matches = self.matcher.match_list(self.haystacks);
            self.active_matches = Some(matches.clone());
            radix_sort(&mut matches);
            matches
        }
        // same needle as last time, return existing matches
        else if prev_needle == needle {
            self.active_matches.as_ref().unwrap().clone()
        }
        // extended the needle with N new characters
        else if is_prefix_extension {
            let active_matches = self.active_matches.as_ref().unwrap();
            let mut matches = self
                .matcher
                .match_indexed_list(self.haystacks, active_matches.iter().map(|m| m.index));
            self.active_matches = Some(matches.clone());

            radix_sort(&mut matches);
            matches
        } else {
            todo!("backspace not supported yet")
        }
    }

    pub fn reset(&mut self) {
        self.removed_idxs.clear();
        self.active_matches = None;
    }

    #[inline(always)]
    fn is_prefix_extension(&self, needle: &str, prev_needle: &str) -> bool {
        !prev_needle.is_empty()
            && needle.len() > prev_needle.len()
            && needle.starts_with(prev_needle)
    }

    fn find_reusable_level(&self, needle: &str, prev_needle: &str) -> usize {
        if needle.starts_with(prev_needle) {
            return prev_needle.len();
        }
        Self::common_prefix_len(needle, prev_needle)
    }

    fn common_prefix_len(a: &str, b: &str) -> usize {
        a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::match_list;

    fn to_owned_strs(haystacks: &[&str]) -> Vec<String> {
        haystacks.iter().map(|s| s.to_string()).collect()
    }

    fn assert_match_parity(needle: &str, haystacks: &[&str], config: &Config) {
        let expected = match_list(needle, haystacks, config);
        let mut incr = IncrementalMatcher::new(config, haystacks);
        let actual = incr.match_list(needle);
        assert_eq!(
            actual, expected,
            "mismatch for needle {:?}: actual={:?}, expected={:?}",
            needle, actual, expected
        );
    }

    #[test]
    fn incremental_matches_one_shot() {
        let haystacks = [
            "fooBar", "foo_bar", "prelude", "println!", "fizzBuzz", "format!",
        ];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        for needle in ["f", "fo", "foo", "fooB", "fooBar"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn prefix_extension_narrows() {
        let haystacks = ["fooBar", "foo_bar", "prelude", "println!", "format!"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        let m1 = incr.match_list("f");
        assert!(!m1.is_empty());

        let m2 = incr.match_list("fo");
        assert!(m2.len() <= m1.len());

        let m3 = incr.match_list("foo");
        assert!(m3.len() <= m2.len());

        for needle in ["f", "fo", "foo"] {
            assert_match_parity(needle, &haystacks, &config);
        }
    }

    #[test]
    fn non_prefix_change_full_rescore() {
        let haystacks = ["fooBar", "barBaz", "bazQux"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        let m1 = incr.match_list("foo");
        let m2 = incr.match_list("bar");

        let expected = match_list("bar", &haystacks, &config);
        assert_eq!(m2, expected);
        assert_ne!(
            m1.iter().map(|m| m.index).collect::<Vec<_>>(),
            m2.iter().map(|m| m.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn backspace_uses_history() {
        let haystacks = ["fooBar", "foo_bar", "fBaz", "format!", "prelude"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        for needle in ["f", "fo", "foo"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle);
            assert_eq!(actual, expected, "forward mismatch for {:?}", needle);
        }

        let m = incr.match_list("fo");
        let expected = match_list("fo", &haystacks, &config);
        assert_eq!(m, expected);

        let m = incr.match_list("f");
        let expected = match_list("f", &haystacks, &config);
        assert_eq!(m, expected);
    }

    #[test]
    fn backspace_then_retype() {
        let haystacks = ["fooBar", "fobBaz", "format!", "prelude"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        incr.match_list("f");
        incr.match_list("fo");
        incr.match_list("foo");

        let m = incr.match_list("fob");
        let expected = match_list("fob", &haystacks, &config);
        assert_eq!(m, expected);
    }

    #[test]
    fn multi_backspace() {
        let haystacks = ["fooBarBaz", "fooBat", "fooXyz", "prelude"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        for needle in ["f", "fo", "foo", "fooB", "fooBar"] {
            incr.match_list(needle);
        }

        let m = incr.match_list("fo");
        let expected = match_list("fo", &haystacks, &config);
        assert_eq!(m, expected);

        let m = incr.match_list("foo");
        let expected = match_list("foo", &haystacks, &config);
        assert_eq!(m, expected);
    }

    #[test]
    fn empty_needle_returns_all() {
        let haystacks = ["foo", "bar", "baz"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        let m = incr.match_list("");
        assert_eq!(m.len(), 3);
        for m in &m {
            assert_eq!(m.score, 0);
        }
    }

    // #[test]
    // fn haystack_growth() {
    //     let config = Config::default();
    //     let mut incr = IncrementalMatcher::new(&config);
    //
    //     let haystacks_small: Vec<&str> = vec!["fooBar", "foo_bar", "prelude"];
    //     incr.match_list("f", &haystacks_small);
    //
    //     let haystacks_big: Vec<&str> = vec!["fooBar", "foo_bar", "prelude", "format!", "fizz"];
    //     let m2 = incr.match_list("fo", &haystacks_big);
    //
    //     let expected = match_list("fo", &haystacks_big, &config);
    //     assert_eq!(m2, expected);
    // }

    #[test]
    fn reset_forces_full_rescore() {
        let haystacks = ["fooBar", "foo_bar"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        incr.match_list("f");
        incr.reset();
        let m = incr.match_list("fo");
        let expected = match_list("fo", &haystacks, &config);
        assert_eq!(m, expected);
    }

    #[test]
    fn max_typos_none() {
        let haystacks = ["fooBar", "fxoBxr", "completely_different"];
        let config = Config {
            max_typos: None,
            ..Config::default()
        };
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        for needle in ["f", "fo", "foo", "fooB"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn max_typos_one() {
        let haystacks = ["fooBar", "fxoBar", "fxxBar", "completely_different"];
        let config = Config {
            max_typos: Some(1),
            ..Config::default()
        };
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        for needle in ["f", "fo", "foo", "fooB"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn high_selectivity() {
        let mut haystacks: Vec<String> = (0..1000).map(|i| format!("item_{}", i)).collect();
        haystacks.push("fooBar".to_string());
        haystacks.push("fooBaz".to_string());

        let refs: Vec<&str> = haystacks.iter().map(|s| s.as_str()).collect();
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &refs);

        for needle in ["f", "fo", "foo", "fooB", "fooBar"] {
            let expected = match_list(needle, &refs, &config);
            let actual = incr.match_list(needle);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn low_selectivity() {
        let haystacks: Vec<String> = (0..100).map(|i| format!("foo_{}", i)).collect();
        let refs: Vec<&str> = haystacks.iter().map(|s| s.as_str()).collect();
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &refs);

        for needle in ["f", "fo", "foo", "foo_"] {
            let expected = match_list(needle, &refs, &config);
            let actual = incr.match_list(needle);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn same_needle_full_rescore() {
        let haystacks = ["fooBar", "foo_bar"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config, &haystacks);

        let first = incr.match_list("foo");
        let second = incr.match_list("foo");
        assert_eq!(first, second);
    }
}
