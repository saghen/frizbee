use crate::prefilter::Prefilter;
use crate::smith_waterman::AlignmentPathIter;
use crate::smith_waterman::simd::SmithWatermanMatcher;
use crate::sort::radix_sort_matches;
use crate::{Config, Match, MatchIndices, Matchable, MatchableChunked};

#[derive(Debug, Clone)]
pub struct Matcher {
    pub needle: String,
    pub config: Config,
    pub prefilter: Prefilter,
    pub smith_waterman: SmithWatermanMatcher,
}

impl Matcher {
    pub fn new(needle: &str, config: &Config) -> Self {
        let matcher = Self {
            needle: needle.to_string(),
            config: *config,
            prefilter: Prefilter::new(needle.as_bytes()),
            smith_waterman: SmithWatermanMatcher::new(needle.as_bytes(), &config.scoring),
        };
        matcher.guard_against_score_overflow();
        matcher
    }

    pub fn set_needle(&mut self, needle: &str) {
        self.needle = needle.to_string();
        self.prefilter = Prefilter::new(needle.as_bytes());
        self.smith_waterman = SmithWatermanMatcher::new(needle.as_bytes(), &self.config.scoring);
        self.guard_against_score_overflow();
    }

    pub fn set_config(&mut self, config: &Config) {
        self.config = *config;
        self.smith_waterman =
            SmithWatermanMatcher::new(self.needle.as_bytes(), &self.config.scoring);
        self.guard_against_score_overflow();
    }

    pub fn match_list<S: Matchable>(&mut self, haystacks: &[S]) -> Vec<Match> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        if self.needle.is_empty() {
            return haystacks
                .iter()
                .enumerate()
                .filter(|(_, item)| item.match_str().is_some())
                .map(|(index, _)| Match {
                    index: index as u32,
                    score: 0,
                    exact: false,
                    #[cfg(feature = "match_end_col")]
                    end_col: 0,
                })
                .collect();
        }

        let mut matches = vec![];
        self.match_list_into(haystacks, 0, &mut matches);

        if self.config.sort {
            radix_sort_matches(&mut matches);
        }

        matches
    }

    pub fn match_list_indices<S: Matchable>(&mut self, haystacks: &[S]) -> Vec<MatchIndices> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        if self.needle.is_empty() {
            return haystacks
                .iter()
                .enumerate()
                .filter(|(_, item)| item.match_str().is_some())
                .map(|(i, _)| MatchIndices::from_index(i))
                .collect();
        }

        let mut matches = vec![];
        self.match_list_indices_into(haystacks, 0, &mut matches);

        if self.config.sort {
            matches.sort_unstable();
        }

        matches
    }

    pub fn match_list_into<S: Matchable>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        Matcher::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);

        if self.needle.is_empty() {
            for (i, item) in haystacks.iter().enumerate() {
                if item.match_str().is_some() {
                    matches.push(Match::from_index(i + haystack_index_offset as usize));
                }
            }
            return;
        }

        let needle = self.needle.as_bytes();
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);

        for (index, haystack_item) in haystacks.iter().enumerate() {
            let Some(haystack_str) = haystack_item.match_str() else {
                continue;
            };
            let haystack = haystack_str.as_bytes();
            if haystack.len() < min_haystack_len {
                continue;
            }

            let (matched, skipped_chunks) = self.config.max_typos.map_or((true, 0), |max_typos| {
                self.prefilter.match_haystack(haystack, max_typos)
            });
            if !matched {
                continue;
            }

            let haystack = &haystack[skipped_chunks * 16..];
            if let Some(match_) = self.smith_waterman_one(
                haystack,
                (index as u32) + haystack_index_offset,
                skipped_chunks == 0,
            ) {
                matches.push(match_);
            }
        }
    }

    pub fn match_list_indices_into<S: Matchable>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<MatchIndices>,
    ) {
        Matcher::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);

        if self.needle.is_empty() {
            for (i, item) in haystacks.iter().enumerate() {
                if item.match_str().is_some() {
                    matches.push(MatchIndices::from_index(i + haystack_index_offset as usize));
                }
            }
            return;
        }

        let needle = self.needle.as_bytes();
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);

        for (index, haystack_item) in haystacks.iter().enumerate() {
            let Some(haystack_str) = haystack_item.match_str() else {
                continue;
            };
            let haystack = haystack_str.as_bytes();
            if haystack.len() < min_haystack_len {
                continue;
            }

            let (matched, skipped_chunks) = self.config.max_typos.map_or((true, 0), |max_typos| {
                self.prefilter.match_haystack(haystack, max_typos)
            });
            if !matched {
                continue;
            }

            let haystack = &haystack[skipped_chunks * 16..];
            if let Some(match_) = self.smith_waterman_indices_one(
                haystack,
                skipped_chunks,
                (index as u32) + haystack_index_offset,
                skipped_chunks == 0,
            ) {
                matches.push(match_);
            }
        }
    }

    /// Returns an unsorted iterator over the matches in the haystacks.
    /// The needle must not be empty
    ///
    /// ```rust
    /// use neo_frizbee::{Config, Match, Matcher};
    ///
    /// fn match_list(needle: &str, haystacks: &[&str]) -> Vec<Match> {
    ///     // Must guard against empty needles
    ///     if needle.is_empty() {
    ///         return (0..haystacks.len()).map(Match::from_index).collect()
    ///     }
    ///
    ///     let mut matcher = Matcher::new(needle, &Config::default());
    ///     let mut matches = matcher
    ///         .match_iter(haystacks)
    ///         .map(|match_| {
    ///             // apply transformations here
    ///             match_
    ///         })
    ///         .collect::<Vec<_>>();
    ///     matches.sort_unstable();
    ///     matches
    /// }
    /// ```
    pub fn match_iter<S: Matchable>(&mut self, haystacks: &[S]) -> impl Iterator<Item = Match> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        self.prefilter_iter(haystacks)
            .filter_map(|(index, haystack, skipped_chunks)| {
                self.smith_waterman_one(haystack, index as u32, skipped_chunks == 0)
            })
    }

    /// Returns an unsorted iterator over the matches in the haystacks with indices.
    /// The needle must not be empty
    ///
    /// ```rust
    /// use neo_frizbee::{Config, Matcher, MatchIndices};
    ///
    /// fn match_list_indices(needle: &str, haystacks: &[&str]) -> Vec<MatchIndices> {
    ///     // Must guard against empty needles
    ///     if needle.is_empty() {
    ///         return (0..haystacks.len()).map(MatchIndices::from_index).collect()
    ///     }
    ///
    ///     let mut matcher = Matcher::new(needle, &Config::default());
    ///     let mut matches = matcher
    ///         .match_iter_indices(haystacks)
    ///         .map(|match_| {
    ///             // apply transformations here
    ///             match_
    ///         })
    ///         .collect::<Vec<_>>();
    ///     matches.sort_unstable();
    ///     matches
    /// }
    /// ```
    pub fn match_iter_indices<S: Matchable>(
        &mut self,
        haystacks: &[S],
    ) -> impl Iterator<Item = MatchIndices> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        self.prefilter_iter(haystacks)
            .filter_map(|(index, haystack, skipped_chunks)| {
                self.smith_waterman_indices_one(
                    haystack,
                    skipped_chunks,
                    index as u32,
                    skipped_chunks == 0,
                )
            })
    }

    #[inline(always)]
    pub fn smith_waterman_one(
        &mut self,
        haystack: &[u8],
        index: u32,
        include_exact: bool,
    ) -> Option<Match> {
        #[cfg(feature = "match_end_col")]
        let (mut score, end_col) = self
            .smith_waterman
            .match_haystack_with_end_col(haystack, self.config.max_typos)?;

        #[cfg(not(feature = "match_end_col"))]
        let mut score = self
            .smith_waterman
            .match_haystack(haystack, self.config.max_typos)?;

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        Some(Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col,
        })
    }

    #[inline(always)]
    pub fn smith_waterman_one_chunked(
        &mut self,
        chunk_ptrs: &[*const u8],
        byte_len: u16,
        index: u32,
    ) -> Option<Match> {
        #[cfg(feature = "match_end_col")]
        let (mut score, end_col) = self.smith_waterman.match_haystack_chunked_with_end_col(
            chunk_ptrs,
            byte_len,
            self.config.max_typos,
        )?;

        #[cfg(not(feature = "match_end_col"))]
        let mut score = self.smith_waterman.match_haystack_chunked(
            chunk_ptrs,
            byte_len,
            self.config.max_typos,
        )?;

        let exact = chunk_ptrs.len() == 1 && (byte_len as usize) == self.needle.len() && {
            let haystack = unsafe { core::slice::from_raw_parts(chunk_ptrs[0], byte_len as usize) };
            self.needle.as_bytes() == haystack
        };
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        Some(Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col,
        })
    }

    #[inline(always)]
    pub fn smith_waterman_indices_one(
        &mut self,
        haystack: &[u8],
        skipped_chunks: usize,
        index: u32,
        include_exact: bool,
    ) -> Option<MatchIndices> {
        // Haystack too large, fallback to greedy matching
        let (mut score, indices) = self.smith_waterman.match_haystack_indices(
            haystack,
            skipped_chunks,
            self.config.max_typos,
        )?;

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        Some(MatchIndices {
            index,
            score,
            exact,
            indices,
        })
    }

    #[inline(always)]
    pub fn prefilter_iter<'a, S: Matchable>(
        &self,
        haystacks: &'a [S],
    ) -> impl Iterator<Item = (usize, &'a [u8], usize)> + use<'a, S> {
        let needle = self.needle.as_bytes();
        assert!(!needle.is_empty(), "needle must not be empty");

        // If max_typos is set, we can ignore any haystacks that are shorter than the needle
        // minus the max typos, since it's impossible for them to match
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);
        let config = self.config;
        let prefilter = self.prefilter.clone();

        haystacks
            .iter()
            .enumerate()
            .filter_map(|(i, item)| item.match_str().map(|s| (i, s.as_bytes())))
            .filter(move |(_, h)| h.len() >= min_haystack_len)
            // Prefiltering
            .filter_map(move |(i, haystack)| {
                let (matched, skipped_chunks) = config.max_typos.map_or((true, 0), |max_typos| {
                    prefilter.match_haystack(haystack, max_typos)
                });
                // Skip any chunks where we know the needle doesn't match
                matched.then(|| (i, &haystack[skipped_chunks * 16..], skipped_chunks))
            })
    }

    #[inline(always)]
    pub fn iter_alignment_path(&self, skipped_chunks: usize, score: u16) -> AlignmentPathIter<'_> {
        self.smith_waterman
            .iter_alignment_path(skipped_chunks, score, self.config.max_typos)
    }

    pub fn match_list_chunked_into<C: MatchableChunked>(
        &mut self,
        haystacks: &[C],
        ctx: &C::Ctx,
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        Matcher::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);

        if self.needle.is_empty() {
            for (i, item) in haystacks.iter().enumerate() {
                if item.haystack_info(ctx).is_some() {
                    matches.push(Match::from_index(i + haystack_index_offset as usize));
                }
            }
            return;
        }

        let needle = self.needle.as_bytes();
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);

        for (index, haystack_item) in haystacks.iter().enumerate() {
            let Some((chunk_count, byte_len)) = haystack_item.haystack_info(ctx) else {
                continue;
            };

            let total_len = byte_len as usize;
            if total_len < min_haystack_len {
                continue;
            }

            let mut ptrs_buf = [core::ptr::null::<u8>(); 32];
            for (i, slot) in ptrs_buf.iter_mut().enumerate().take(chunk_count) {
                *slot = haystack_item.load_chunk(ctx, i).as_ptr();
            }
            let chunk_ptrs = &ptrs_buf[..chunk_count];

            let (prefilter_passed, skipped_chunks) =
                self.config.max_typos.map_or((true, 0), |max_typos| {
                    self.prefilter
                        .match_haystack_chunked(chunk_ptrs, byte_len, max_typos)
                });
            if !prefilter_passed {
                continue;
            }

            let chunk_ptrs = &chunk_ptrs[skipped_chunks..];
            let byte_len = byte_len - (skipped_chunks as u16 * 16);
            if let Some(match_) = self.smith_waterman_one_chunked(
                chunk_ptrs,
                byte_len,
                (index as u32) + haystack_index_offset,
            ) {
                matches.push(match_);
            }
        }
    }

    /// Match items using a caller-provided resolver callback.
    ///
    /// For each item, `resolve` is called with a stack buffer. It should fill
    /// the buffer with chunk pointers and return `Some((ptrs_slice, byte_len))`
    /// or `None` to skip the item (e.g. deleted files).
    ///
    /// This avoids the lifetime issue of `MatchableChunked` — the resolver
    /// writes into a buffer owned by the caller, not by the item.
    pub fn match_list_resolved_into<T, F>(
        &mut self,
        items: &[T],
        item_index_offset: u32,
        resolve: &F,
        matches: &mut Vec<Match>,
    ) where
        F: Fn(&T, &mut [*const u8; 32]) -> Option<(usize, u16)>, // (chunk_count, byte_len)
    {
        Matcher::guard_against_haystack_overflow(items.len(), item_index_offset);

        if self.needle.is_empty() {
            let mut ptrs_buf = [core::ptr::null::<u8>(); 32];
            for (i, item) in items.iter().enumerate() {
                if resolve(item, &mut ptrs_buf).is_some() {
                    matches.push(Match::from_index(i + item_index_offset as usize));
                }
            }
            return;
        }

        let needle = self.needle.as_bytes();
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);

        for (index, item) in items.iter().enumerate() {
            let mut ptrs_buf = [core::ptr::null::<u8>(); 32];
            let Some((chunk_count, byte_len)) = resolve(item, &mut ptrs_buf) else {
                continue;
            };

            let total_len = byte_len as usize;
            if total_len < min_haystack_len {
                continue;
            }

            let chunk_ptrs = &ptrs_buf[..chunk_count];

            let (prefilter_passed, skipped_chunks) =
                self.config.max_typos.map_or((true, 0), |max_typos| {
                    self.prefilter
                        .match_haystack_chunked(chunk_ptrs, byte_len, max_typos)
                });
            if !prefilter_passed {
                continue;
            }

            let chunk_ptrs = &chunk_ptrs[skipped_chunks..];
            let byte_len = byte_len - (skipped_chunks as u16 * 16);
            if let Some(match_) = self.smith_waterman_one_chunked(
                chunk_ptrs,
                byte_len,
                (index as u32) + item_index_offset,
            ) {
                matches.push(match_);
            }
        }
    }

    #[inline(always)]
    pub fn guard_against_score_overflow(&self) {
        let scoring = &self.config.scoring;
        let max_per_char_score = scoring.match_score
            + scoring.capitalization_bonus / 2
            + scoring.delimiter_bonus / 2
            + scoring.matching_case_bonus;
        let max_needle_len =
            (u16::MAX - scoring.prefix_bonus - scoring.exact_match_bonus) / max_per_char_score;
        assert!(
            self.needle.len() <= max_needle_len as usize,
            "needle too long and could overflow the u16 score: {} > {}",
            self.needle.len(),
            max_needle_len
        );
    }

    #[inline(always)]
    pub fn guard_against_haystack_overflow(haystack_len: usize, haystack_index_offset: u32) {
        assert!(
            (haystack_len.saturating_add(haystack_index_offset as usize)) <= (u32::MAX as usize),
            "too many haystack which will overflow the u32 index: {} > {} (index offset: {})",
            haystack_len,
            u32::MAX,
            haystack_index_offset
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::match_list;
    use super::*;

    #[test]
    fn test_basic() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let config = Config {
            max_typos: None,
            ..Config::default()
        };
        let matches = match_list(needle, &haystack, &config);

        println!("{:?}", matches);
        assert_eq!(matches.len(), 4);
        assert_eq!(matches[0].index, 3);
        assert_eq!(matches[1].index, 0);
        assert_eq!(matches[2].index, 2);
        assert_eq!(matches[3].index, 1);
    }

    #[test]
    fn test_no_typos() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let matches = match_list(
            needle,
            &haystack,
            &Config {
                max_typos: Some(0),
                ..Config::default()
            },
        );
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_exact_match() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let matches = match_list(needle, &haystack, &Config::default());

        let exact_matches = matches.iter().filter(|m| m.exact).collect::<Vec<&Match>>();
        assert_eq!(exact_matches.len(), 1);
        assert_eq!(exact_matches[0].index, 3);
        for m in &exact_matches {
            assert_eq!(haystack[m.index as usize], needle)
        }
    }

    #[test]
    fn test_exact_matches() {
        let needle = "deadbe";
        let haystack = vec![
            "deadbe",
            "deadbeef",
            "deadbe",
            "deadbf",
            "deadbe",
            "deadbeefg",
            "deadbe",
        ];

        let matches = match_list(needle, &haystack, &Config::default());

        let exact_matches = matches.iter().filter(|m| m.exact).collect::<Vec<&Match>>();
        assert_eq!(exact_matches.len(), 4);
        for m in &exact_matches {
            assert_eq!(haystack[m.index as usize], needle)
        }
    }
    #[test]
    fn test_small_needle() {
        // max_typos longer than needle
        let config = Config {
            max_typos: Some(2),
            ..Config::default()
        };
        let matches = match_list("1", &["1"], &config);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 0);
        assert!(matches[0].exact);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_match_end_col_through_match_list() {
        let config = Config {
            max_typos: None,
            sort: false,
            ..Config::default()
        };
        let matches = match_list("abc", &["xabcx", "abcdef", "xxabc"], &config);
        assert_eq!(matches.len(), 3);
        // "abc" in "xabcx" ends at byte position 3
        assert_eq!(matches[0].end_col, 3);
        // "abc" in "abcdef" ends at byte position 2
        assert_eq!(matches[1].end_col, 2);
        // "abc" in "xxabc" ends at byte position 4
        assert_eq!(matches[2].end_col, 4);
    }

    /// Pad a string into 16-byte aligned chunks on the heap, returning
    /// (chunk_ptrs, chunk_count, byte_len). The backing memory is leaked
    /// so the pointers remain valid for the test lifetime.
    fn string_to_chunks(s: &str) -> (Vec<*const u8>, usize, u16) {
        let bytes = s.as_bytes();
        let n_chunks = if bytes.is_empty() {
            0
        } else {
            bytes.len().div_ceil(16)
        };
        let mut arena = vec![[0u8; 16]; n_chunks];
        for (i, chunk) in arena.iter_mut().enumerate() {
            let start = i * 16;
            let take = 16.min(bytes.len() - start);
            chunk[..take].copy_from_slice(&bytes[start..start + take]);
        }
        let ptrs: Vec<*const u8> = arena.iter().map(|c| c.as_ptr()).collect();
        std::mem::forget(arena);
        (ptrs, n_chunks, bytes.len() as u16)
    }

    /// Resolved matching must produce the same set of matched indices and
    /// scores as contiguous matching for arbitrary needle/haystack pairs.
    #[test]
    fn test_resolved_matches_contiguous_parity() {
        use proptest::prelude::*;
        use proptest::test_runner::{Config as PropConfig, TestRunner};

        let mut runner = TestRunner::new(PropConfig {
            cases: 2000,
            ..PropConfig::default()
        });

        let strategy = (
            "[a-z]{2,12}",                                        // needle
            proptest::collection::vec("[a-z/_\\.]{5,80}", 1..30), // haystacks
            (0u16..=8u16),                                        // max_typos
        );

        runner
            .run(&strategy, |(needle, haystacks, max_typos)| {
                let config = Config {
                    max_typos: Some(max_typos),
                    sort: false,
                    ..Config::default()
                };

                // Contiguous path
                let haystack_refs: Vec<&str> = haystacks.iter().map(String::as_str).collect();
                let contiguous = match_list(&needle, &haystack_refs, &config);

                // Build chunk data for each haystack
                let chunk_data: Vec<(Vec<*const u8>, usize, u16)> =
                    haystacks.iter().map(|s| string_to_chunks(s)).collect();

                // Resolved path
                let resolve =
                    |item: &(Vec<*const u8>, usize, u16),
                     ptrs_buf: &mut [*const u8; 32]|
                     -> Option<(usize, u16)> {
                        let (ptrs, count, byte_len) = item;
                        for (i, &p) in ptrs.iter().enumerate() {
                            ptrs_buf[i] = p;
                        }

                        Some((*count, *byte_len))
                    };

                let mut matcher = Matcher::new(&needle, &config);
                let mut resolved = Vec::new();
                matcher.match_list_resolved_into(&chunk_data, 0, &resolve, &mut resolved);

                // Compare: same indices matched
                let mut contiguous_indices: Vec<u32> =
                    contiguous.iter().map(|m| m.index).collect();
                let mut resolved_indices: Vec<u32> =
                    resolved.iter().map(|m| m.index).collect();
                contiguous_indices.sort();
                resolved_indices.sort();

                prop_assert_eq!(
                    &contiguous_indices,
                    &resolved_indices,
                    "needle={:?} max_typos={} contiguous matched {:?} but resolved matched {:?}",
                    needle,
                    max_typos,
                    contiguous_indices,
                    resolved_indices,
                );

                // Compare: same scores for each matched index
                for cm in &contiguous {
                    if let Some(rm) = resolved.iter().find(|r| r.index == cm.index) {
                        prop_assert_eq!(
                            cm.score,
                            rm.score,
                            "needle={:?} max_typos={} index={} score mismatch: contiguous={} resolved={}",
                            needle,
                            max_typos,
                            cm.index,
                            cm.score,
                            rm.score,
                        );
                    }
                }

                Ok(())
            })
            .unwrap();
    }
}
