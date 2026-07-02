use crate::prefilter::{Kernel as PrefilterKernel, Window};
use crate::smith_waterman::Kernel as SmithWatermanKernel;
use crate::{Config, Match, MatchIndices};

/// Magic numbers for `TYPOS` specialization keys beyond the literal counts 0/1/2
pub(super) const MANY_TYPOS: u16 = u16::MAX;
pub(super) const NO_PREFILTER: u16 = u16::MAX - 1;

/// Fully inlined per-backend implementations, specialized for each configuration
/// (0 typos, 1 typo, unicode variants, ...) and built with the backend's `#[target_feature]`.
///
/// # Safety
/// The backend's required CPU features must be available
pub(crate) trait Specialized: Sized {
    unsafe fn build(needle: &str, config: &Config) -> Self;

    unsafe fn match_list<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    );

    unsafe fn match_list_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
    ) -> Vec<MatchIndices>;
}

#[derive(Debug, Clone)]
pub struct MatcherImpl<P: PrefilterKernel, S: SmithWatermanKernel> {
    needle: String,
    config: Config,
    prefilter: P,
    smith_waterman: S,
}

impl<P, S> MatcherImpl<P, S>
where
    P: PrefilterKernel,
    S: SmithWatermanKernel,
{
    #[inline(always)]
    pub fn new(needle: &str, config: &Config) -> Self {
        let case_sensitive = config.casing.respects_case_for(needle);
        let matcher = Self {
            needle: needle.to_string(),
            config: config.clone(),
            prefilter: P::new(needle, case_sensitive),
            smith_waterman: S::new(needle, &config.scoring, case_sensitive),
        };
        matcher.guard_against_score_overflow();
        matcher
    }

    pub fn is_available() -> bool {
        P::is_available() && S::is_available()
    }

    #[inline(always)]
    pub(super) fn match_list_into_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        let min_haystack_len = self.min_haystack_len();
        let max_typos = self.max_typos_runtime::<TYPOS>();
        for (index, haystack_str) in (haystack_index_offset..).zip(haystacks.iter()) {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    matches.push(self.smith_waterman_one::<UNICODE>(
                        trimmed,
                        index,
                        start_pos,
                        include_exact,
                    ));
                }
            }
        }
    }

    #[inline(always)]
    fn prefilter_haystack<const TYPOS: u16, const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> Window {
        match TYPOS {
            NO_PREFILTER => (true, 0, haystack.len()),
            0 if UNICODE => self.prefilter.match_haystack_unicode(haystack),
            0 => self.prefilter.match_haystack(haystack),
            1 if UNICODE => self.prefilter.match_haystack_unicode_1_typo(haystack),
            1 => self.prefilter.match_haystack_1_typo(haystack),
            2 if UNICODE => self.prefilter.match_haystack_unicode_2_typos(haystack),
            2 => self.prefilter.match_haystack_2_typos(haystack),
            MANY_TYPOS if UNICODE => self
                .prefilter
                .match_haystack_unicode_many_typos(haystack, max_typos),
            MANY_TYPOS => self
                .prefilter
                .match_haystack_many_typos(haystack, max_typos),
            _ => unreachable!("unsupported typo count specialization"),
        }
    }

    #[inline(always)]
    pub(super) fn match_list_indices_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
    ) -> Vec<MatchIndices> {
        let min_haystack_len = self.min_haystack_len();
        let max_typos = self.max_typos_runtime::<TYPOS>();
        let max_typos_opt = if TYPOS == NO_PREFILTER {
            None
        } else {
            Some(max_typos)
        };
        let mut matches = vec![];
        for (index, haystack_str) in haystacks.iter().enumerate() {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    if let Some(match_) = self.smith_waterman_indices_one::<UNICODE>(
                        trimmed,
                        start_pos,
                        index as u32,
                        include_exact,
                        max_typos_opt,
                    ) {
                        matches.push(match_);
                    }
                }
            }
        }
        matches
    }

    #[inline(always)]
    fn smith_waterman_one<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        index: u32,
        haystack_start_pos: usize,
        include_exact: bool,
    ) -> Match {
        let mut score = if UNICODE {
            self.smith_waterman.score_haystack_unicode(haystack)
        } else {
            self.smith_waterman.score_haystack(haystack)
        };

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        #[cfg(not(feature = "match_end_col"))]
        let _ = haystack_start_pos;

        Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col: self
                .smith_waterman
                .match_end_col(haystack)
                .saturating_add(haystack_start_pos.min(u16::MAX as usize) as u16),
        }
    }

    #[inline(always)]
    fn smith_waterman_indices_one<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        skipped_chars: usize,
        index: u32,
        include_exact: bool,
        max_typos: Option<u16>,
    ) -> Option<MatchIndices> {
        let (mut score, indices) = if UNICODE {
            self.smith_waterman.score_haystack_unicode_indices(
                haystack,
                skipped_chars,
                max_typos,
            )?
        } else {
            self.smith_waterman
                .score_haystack_indices(haystack, skipped_chars, max_typos)?
        };

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

    /// The runtime typo budget for a `TYPOS` specialization: 0/1/2 encode the
    /// count directly, `MANY_TYPOS` reads it from the config, and
    /// `NO_PREFILTER` never uses it.
    #[inline(always)]
    fn max_typos_runtime<const TYPOS: u16>(&self) -> u16 {
        if TYPOS == MANY_TYPOS {
            self.config.max_typos.unwrap_or(0)
        } else {
            TYPOS
        }
    }

    #[inline(always)]
    fn min_haystack_len(&self) -> usize {
        self.config
            .max_typos
            .map(|max| self.needle.chars().count().saturating_sub(max as usize))
            .unwrap_or(0)
    }

    #[inline(always)]
    fn guard_against_score_overflow(&self) {
        let scoring = &self.config.scoring;
        let max_per_char_score = scoring.match_score
            + scoring
                .capitalization_bonus
                .max(scoring.delimiter_bonus)
                .saturating_sub(scoring.gap_open_penalty)
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
}

#[cfg(test)]
mod tests {
    use crate::{Config, match_list, match_list_indices};

    #[test]
    fn unsorted_output_preserves_candidate_order() {
        let haystacks = ["foo", "nomatch", "xfoo", "f_o_o", "bar"];
        let config = Config {
            sort: false,
            ..Config::default()
        };

        let matches = match_list("foo", &haystacks, &config);
        assert_eq!(
            matches
                .iter()
                .map(|match_| match_.index)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn match_list_indices_reports_expected_public_indices() {
        let haystacks = ["xabcx", "a_b_c", "nomatch"];
        let config = Config {
            sort: false,
            ..Config::default()
        };

        let matches = match_list_indices("abc", &haystacks, &config);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[0].indices, vec![3, 2, 1]);
        assert_eq!(matches[1].index, 1);
        assert_eq!(matches[1].indices, vec![4, 2, 0]);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn filtered_match_end_col_uses_original_haystack_offsets() {
        let config = Config {
            sort: false,
            ..Config::default()
        };

        let matches = match_list("abc", &["xxabcxx"], &config);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].end_col, 4);
    }
}
