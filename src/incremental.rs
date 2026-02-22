use itertools::Itertools;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::one_shot::Matcher;
use crate::{Config, Match, MatchIndices};

/// Incremental fuzzy matcher that reuses previous results when the needle is extended.
///
/// When a user types a query character by character (e.g. `"f"` → `"fo"` → `"foo"`),
/// extending a needle can only eliminate matches, never create new ones. This matcher
/// stores which haystack indices matched previously and only rescores that subset when
/// the new needle is a prefix extension of the previous one.
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
/// ```
pub struct IncrementalMatcher {
    matcher: Matcher,
    prev_needle: String,
    matched_indices: Vec<u32>,
    prev_haystack_count: usize,
}

impl IncrementalMatcher {
    pub fn new(config: &Config) -> Self {
        Self {
            matcher: Matcher::new("", config),
            prev_needle: String::new(),
            matched_indices: Vec::new(),
            prev_haystack_count: 0,
        }
    }

    /// Match the needle against the haystacks, reusing previous results when possible.
    pub fn match_list<S: AsRef<str>>(&mut self, needle: &str, haystacks: &[S]) -> Vec<Match> {
        let is_prefix_extension = self.is_prefix_extension(needle);
        let haystack_count = haystacks.len();

        self.matcher.set_needle(needle);

        if needle.is_empty() {
            self.prev_needle.clear();
            self.matched_indices.clear();
            self.prev_haystack_count = haystack_count;
            return (0..haystack_count).map(Match::from_index).collect();
        }

        if is_prefix_extension && haystack_count == self.prev_haystack_count {
            let mut matches = self.match_narrowed_unsorted(haystacks);
            if self.matcher.config.sort {
                matches.sort_unstable();
            }
            self.set_prev(needle, haystack_count);
            matches
        } else if is_prefix_extension && haystack_count > self.prev_haystack_count {
            let matches = self.match_narrowed_with_growth(haystacks);
            self.set_prev(needle, haystack_count);
            matches
        } else {
            self.full_rescore(haystacks, needle, haystack_count)
        }
    }

    /// Match the needle against the haystacks with character match indices.
    pub fn match_list_indices<S: AsRef<str>>(
        &mut self,
        needle: &str,
        haystacks: &[S],
    ) -> Vec<MatchIndices> {
        let is_prefix_extension = self.is_prefix_extension(needle);
        let haystack_count = haystacks.len();

        self.matcher.set_needle(needle);

        if needle.is_empty() {
            self.prev_needle.clear();
            self.matched_indices.clear();
            self.prev_haystack_count = haystack_count;
            return (0..haystack_count).map(MatchIndices::from_index).collect();
        }

        if is_prefix_extension && haystack_count == self.prev_haystack_count {
            let mut matches = self.match_narrowed_indices_unsorted(haystacks);
            if self.matcher.config.sort {
                matches.sort_unstable();
            }
            self.set_prev(needle, haystack_count);
            matches
        } else if is_prefix_extension && haystack_count > self.prev_haystack_count {
            let matches = self.match_narrowed_indices_with_growth(haystacks);
            self.set_prev(needle, haystack_count);
            matches
        } else {
            self.full_rescore_indices(haystacks, needle, haystack_count)
        }
    }

    /// Match the needle against the haystacks in parallel.
    pub fn match_list_parallel<S: AsRef<str> + Sync>(
        &mut self,
        needle: &str,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        let is_prefix_extension = self.is_prefix_extension(needle);
        let haystack_count = haystacks.len();

        self.matcher.set_needle(needle);

        if needle.is_empty() {
            self.prev_needle.clear();
            self.matched_indices.clear();
            self.prev_haystack_count = haystack_count;
            return (0..haystack_count).map(Match::from_index).collect();
        }

        if is_prefix_extension && haystack_count >= self.prev_haystack_count {
            let matches = self.match_narrowed_parallel(haystacks, threads);
            self.set_prev(needle, haystack_count);
            matches
        } else {
            let matches = self.full_rescore_parallel(haystacks, threads);
            self.update_state_from_matches(needle, haystack_count, matches.iter().map(|m| m.index));
            matches
        }
    }

    /// Reset the incremental state, forcing a full rescore on the next call.
    pub fn reset(&mut self) {
        self.prev_needle.clear();
        self.matched_indices.clear();
        self.prev_haystack_count = 0;
    }

    pub fn matcher(&self) -> &Matcher {
        &self.matcher
    }

    #[inline(always)]
    fn is_prefix_extension(&self, needle: &str) -> bool {
        !self.prev_needle.is_empty()
            && needle.len() > self.prev_needle.len()
            && needle.starts_with(&self.prev_needle)
    }

    #[inline]
    fn set_prev(&mut self, needle: &str, haystack_count: usize) {
        self.prev_needle.clear();
        self.prev_needle.push_str(needle);
        self.prev_haystack_count = haystack_count;
    }

    #[inline]
    fn update_state_from_matches(
        &mut self,
        needle: &str,
        haystack_count: usize,
        indices: impl Iterator<Item = u32>,
    ) {
        self.prev_needle.clear();
        self.prev_needle.push_str(needle);
        self.matched_indices.clear();
        self.matched_indices.extend(indices);
        self.matched_indices.sort_unstable();
        self.prev_haystack_count = haystack_count;
    }

    #[inline]
    fn full_rescore<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        needle: &str,
        haystack_count: usize,
    ) -> Vec<Match> {
        let mut matches = Vec::new();
        self.matcher.match_list_into(haystacks, 0, &mut matches);

        // Extract indices while in insertion order (ascending)
        self.matched_indices.clear();
        self.matched_indices.extend(matches.iter().map(|m| m.index));

        if self.matcher.config.sort {
            matches.sort_unstable();
        }

        self.set_prev(needle, haystack_count);
        matches
    }

    #[inline]
    fn full_rescore_indices<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        needle: &str,
        haystack_count: usize,
    ) -> Vec<MatchIndices> {
        let mut matches = Vec::new();
        self.matcher
            .match_list_indices_into(haystacks, 0, &mut matches);

        self.matched_indices.clear();
        self.matched_indices.extend(matches.iter().map(|m| m.index));

        if self.matcher.config.sort {
            matches.sort_unstable();
        }

        self.set_prev(needle, haystack_count);
        matches
    }

    #[inline]
    fn match_narrowed_unsorted<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        let mut matches = Vec::with_capacity(self.matched_indices.len());
        let max_typos = self.matcher.config.max_typos;
        let needle_len = self.matcher.needle.len();
        let min_haystack_len = max_typos
            .map(|max| needle_len.saturating_sub(max as usize))
            .unwrap_or(0);

        let mut write = 0usize;
        for read in 0..self.matched_indices.len() {
            let idx = self.matched_indices[read];
            let haystack = haystacks[idx as usize].as_ref().as_bytes();

            if haystack.len() < min_haystack_len {
                continue;
            }

            let (matched, skipped_chunks) = match max_typos {
                Some(max) => self.matcher.prefilter.match_haystack(haystack, max),
                None => (true, 0),
            };
            if !matched {
                continue;
            }

            let trimmed = &haystack[skipped_chunks * 16..];
            if let Some(m) =
                self.matcher
                    .smith_waterman_one(trimmed, idx, skipped_chunks == 0)
            {
                self.matched_indices[write] = idx;
                write += 1;
                matches.push(m);
            }
        }
        self.matched_indices.truncate(write);

        matches
    }

    fn match_narrowed_with_growth<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        let mut matches = self.match_narrowed_unsorted(haystacks);

        let prev_count = self.prev_haystack_count;
        let matches_before_tail = matches.len();
        self.matcher
            .match_list_into(&haystacks[prev_count..], prev_count as u32, &mut matches);
        self.matched_indices
            .extend(matches[matches_before_tail..].iter().map(|m| m.index));

        if self.matcher.config.sort {
            matches.sort_unstable();
        }

        matches
    }

    #[inline]
    fn match_narrowed_indices_unsorted<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
    ) -> Vec<MatchIndices> {
        let mut matches = Vec::with_capacity(self.matched_indices.len());
        let max_typos = self.matcher.config.max_typos;
        let needle_len = self.matcher.needle.len();
        let min_haystack_len = max_typos
            .map(|max| needle_len.saturating_sub(max as usize))
            .unwrap_or(0);

        let mut write = 0usize;
        for read in 0..self.matched_indices.len() {
            let idx = self.matched_indices[read];
            let haystack = haystacks[idx as usize].as_ref().as_bytes();

            if haystack.len() < min_haystack_len {
                continue;
            }

            let (matched, skipped_chunks) = match max_typos {
                Some(max) => self.matcher.prefilter.match_haystack(haystack, max),
                None => (true, 0),
            };
            if !matched {
                continue;
            }

            let trimmed = &haystack[skipped_chunks * 16..];
            if let Some(m) = self.matcher.smith_waterman_indices_one(
                trimmed,
                skipped_chunks,
                idx,
                skipped_chunks == 0,
            ) {
                self.matched_indices[write] = idx;
                write += 1;
                matches.push(m);
            }
        }
        self.matched_indices.truncate(write);

        matches
    }

    fn match_narrowed_indices_with_growth<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
    ) -> Vec<MatchIndices> {
        let mut matches = self.match_narrowed_indices_unsorted(haystacks);

        let prev_count = self.prev_haystack_count;
        let matches_before_tail = matches.len();
        self.matcher
            .match_list_indices_into(&haystacks[prev_count..], prev_count as u32, &mut matches);
        self.matched_indices
            .extend(matches[matches_before_tail..].iter().map(|m| m.index));

        if self.matcher.config.sort {
            matches.sort_unstable();
        }

        matches
    }

    fn full_rescore_parallel<S: AsRef<str> + Sync>(
        &self,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        if haystacks.is_empty() {
            return vec![];
        }

        let chunk_size = 512;
        let num_chunks = haystacks.len().div_ceil(chunk_size);
        let next_chunk = AtomicUsize::new(0);
        let matcher = &self.matcher;
        let config = &matcher.config;

        thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut local_matches = Vec::new();
                        let mut thread_matcher = matcher.clone();

                        loop {
                            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if chunk_idx >= num_chunks {
                                break;
                            }

                            let start = chunk_idx * chunk_size;
                            let end = (start + chunk_size).min(haystacks.len());

                            thread_matcher.match_list_into(
                                &haystacks[start..end],
                                start as u32,
                                &mut local_matches,
                            );
                        }

                        if config.sort {
                            local_matches.sort_unstable();
                        }

                        local_matches
                    })
                })
                .collect();

            if config.sort {
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .kmerge()
                    .collect()
            } else {
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            }
        })
    }

    fn match_narrowed_parallel<S: AsRef<str> + Sync>(
        &mut self,
        haystacks: &[S],
        threads: usize,
    ) -> Vec<Match> {
        let mut new_tail_matches = Vec::new();
        let mut new_tail_indices = Vec::new();
        if haystacks.len() > self.prev_haystack_count {
            let prev_count = self.prev_haystack_count;
            self.matcher.match_list_into(
                &haystacks[prev_count..],
                prev_count as u32,
                &mut new_tail_matches,
            );
            new_tail_indices.extend(new_tail_matches.iter().map(|m| m.index));
        }

        if self.matched_indices.is_empty() {
            self.matched_indices = new_tail_indices;
            if self.matcher.config.sort {
                new_tail_matches.sort_unstable();
            }
            return new_tail_matches;
        }

        let chunk_size = 512;
        let num_chunks = self.matched_indices.len().div_ceil(chunk_size);
        let next_chunk = AtomicUsize::new(0);

        let matched_indices = &self.matched_indices;
        let matcher = &self.matcher;
        let config = &matcher.config;
        let max_typos = config.max_typos;
        let needle_len = matcher.needle.len();
        let min_haystack_len = max_typos
            .map(|max| needle_len.saturating_sub(max as usize))
            .unwrap_or(0);

        let (thread_matches, new_indices) = thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut local_matches = Vec::new();
                        let mut local_indices = Vec::new();
                        let mut thread_matcher = matcher.clone();

                        loop {
                            let chunk_idx = next_chunk.fetch_add(1, Ordering::Relaxed);
                            if chunk_idx >= num_chunks {
                                break;
                            }

                            let start = chunk_idx * chunk_size;
                            let end = (start + chunk_size).min(matched_indices.len());

                            for &idx in &matched_indices[start..end] {
                                let haystack = haystacks[idx as usize].as_ref().as_bytes();

                                if haystack.len() < min_haystack_len {
                                    continue;
                                }

                                let (matched, skipped_chunks) = match max_typos {
                                    Some(max) => {
                                        thread_matcher.prefilter.match_haystack(haystack, max)
                                    }
                                    None => (true, 0),
                                };
                                if !matched {
                                    continue;
                                }

                                let trimmed = &haystack[skipped_chunks * 16..];
                                if let Some(m) = thread_matcher.smith_waterman_one(
                                    trimmed,
                                    idx,
                                    skipped_chunks == 0,
                                ) {
                                    local_matches.push(m);
                                    local_indices.push(idx);
                                }
                            }
                        }

                        if config.sort {
                            local_matches.sort_unstable();
                        }

                        (local_matches, local_indices)
                    })
                })
                .collect();

            let mut all_indices = Vec::new();
            let thread_matches = if config.sort {
                let mut match_vecs = Vec::with_capacity(handles.len());
                for h in handles {
                    let (matches, indices) = h.join().unwrap();
                    all_indices.extend(indices);
                    match_vecs.push(matches);
                }
                match_vecs.into_iter().kmerge().collect::<Vec<Match>>()
            } else {
                let mut all_matches = Vec::new();
                for h in handles {
                    let (matches, indices) = h.join().unwrap();
                    all_indices.extend(indices);
                    all_matches.extend(matches);
                }
                all_matches
            };
            (thread_matches, all_indices)
        });

        self.matched_indices = new_indices;
        self.matched_indices.sort_unstable();
        self.matched_indices.extend(new_tail_indices);

        if new_tail_matches.is_empty() {
            thread_matches
        } else if config.sort {
            thread_matches
                .into_iter()
                .merge(new_tail_matches)
                .collect()
        } else {
            let mut result = thread_matches;
            result.extend(new_tail_matches);
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::match_list;

    fn assert_match_parity(needle: &str, haystacks: &[&str], config: &Config) {
        let expected = match_list(needle, haystacks, config);
        let mut incr = IncrementalMatcher::new(config);
        let actual = incr.match_list(needle, haystacks);
        assert_eq!(
            actual, expected,
            "mismatch for needle {:?}: actual={:?}, expected={:?}",
            needle, actual, expected
        );
    }

    #[test]
    fn incremental_matches_one_shot() {
        let haystacks = [
            "fooBar",
            "foo_bar",
            "prelude",
            "println!",
            "fizzBuzz",
            "format!",
        ];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "fooB", "fooBar"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle, &haystacks);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn prefix_extension_narrows() {
        let haystacks = ["fooBar", "foo_bar", "prelude", "println!", "format!"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        let m1 = incr.match_list("f", &haystacks);
        assert!(!m1.is_empty());

        let m2 = incr.match_list("fo", &haystacks);
        assert!(m2.len() <= m1.len());

        let m3 = incr.match_list("foo", &haystacks);
        assert!(m3.len() <= m2.len());

        for needle in ["f", "fo", "foo"] {
            assert_match_parity(needle, &haystacks, &config);
        }
    }

    #[test]
    fn non_prefix_change_full_rescore() {
        let haystacks = ["fooBar", "barBaz", "bazQux"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        let m1 = incr.match_list("foo", &haystacks);
        let m2 = incr.match_list("bar", &haystacks);

        let expected = match_list("bar", &haystacks, &config);
        assert_eq!(m2, expected);
        assert_ne!(
            m1.iter().map(|m| m.index).collect::<Vec<_>>(),
            m2.iter().map(|m| m.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn deletion_full_rescore() {
        let haystacks = ["fooBar", "foo_bar", "fBaz"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        incr.match_list("foo", &haystacks);
        let m = incr.match_list("fo", &haystacks);
        let expected = match_list("fo", &haystacks, &config);
        assert_eq!(m, expected);
    }

    #[test]
    fn empty_needle_returns_all() {
        let haystacks = ["foo", "bar", "baz"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        let m = incr.match_list("", &haystacks);
        assert_eq!(m.len(), 3);
        for m in &m {
            assert_eq!(m.score, 0);
        }
    }

    #[test]
    fn haystack_growth() {
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        let haystacks_small: Vec<&str> = vec!["fooBar", "foo_bar", "prelude"];
        incr.match_list("f", &haystacks_small);

        let haystacks_big: Vec<&str> = vec!["fooBar", "foo_bar", "prelude", "format!", "fizz"];
        let m2 = incr.match_list("fo", &haystacks_big);

        let expected = match_list("fo", &haystacks_big, &config);
        assert_eq!(m2, expected);
    }

    #[test]
    fn reset_forces_full_rescore() {
        let haystacks = ["fooBar", "foo_bar"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        incr.match_list("f", &haystacks);
        incr.reset();
        let m = incr.match_list("fo", &haystacks);
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
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "fooB"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle, &haystacks);
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
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "fooB"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list(needle, &haystacks);
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
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "fooB", "fooBar"] {
            let expected = match_list(needle, &refs, &config);
            let actual = incr.match_list(needle, &refs);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn low_selectivity() {
        let haystacks: Vec<String> = (0..100).map(|i| format!("foo_{}", i)).collect();
        let refs: Vec<&str> = haystacks.iter().map(|s| s.as_str()).collect();
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "foo_"] {
            let expected = match_list(needle, &refs, &config);
            let actual = incr.match_list(needle, &refs);
            assert_eq!(actual, expected, "mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn match_list_indices_parity() {
        let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "fooB"] {
            let expected = crate::match_list_indices(needle, &haystacks, &config);
            let actual = incr.match_list_indices(needle, &haystacks);
            assert_eq!(
                actual.len(),
                expected.len(),
                "length mismatch for needle {:?}",
                needle
            );
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert_eq!(a.index, e.index, "index mismatch for needle {:?}", needle);
                assert_eq!(a.score, e.score, "score mismatch for needle {:?}", needle);
                assert_eq!(a.exact, e.exact, "exact mismatch for needle {:?}", needle);
                assert_eq!(
                    a.indices, e.indices,
                    "indices mismatch for needle {:?}",
                    needle
                );
            }
        }
    }

    #[test]
    fn parallel_parity() {
        let haystacks = [
            "fooBar", "foo_bar", "prelude", "println!", "format!", "fizzBuzz",
        ];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        for needle in ["f", "fo", "foo", "fooB"] {
            let expected = match_list(needle, &haystacks, &config);
            let actual = incr.match_list_parallel(needle, &haystacks, 2);
            assert_eq!(actual, expected, "parallel mismatch for needle {:?}", needle);
        }
    }

    #[test]
    fn same_needle_full_rescore() {
        let haystacks = ["fooBar", "foo_bar"];
        let config = Config::default();
        let mut incr = IncrementalMatcher::new(&config);

        let first = incr.match_list("foo", &haystacks);
        let second = incr.match_list("foo", &haystacks);
        assert_eq!(first, second);
    }
}
