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

    unsafe fn match_one<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<Match>;

    unsafe fn match_one_indices<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<MatchIndices>;
}

#[derive(Debug, Clone)]
pub struct MatcherImpl<P: PrefilterKernel, S: SmithWatermanKernel> {
    needle: String,
    config: Config,
    min_haystack_len: usize,
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
            min_haystack_len: config
                .max_typos
                .map(|max| needle.chars().count().saturating_sub(max as usize))
                .unwrap_or(0),
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
        let max_typos = self.max_typos_runtime::<TYPOS>();
        for (index, haystack_str) in (haystack_index_offset..).zip(haystacks.iter()) {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= self.min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let (trimmed, start_pos) = trim_haystack(haystack, start_pos, end_pos);
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

    /// Single-haystack path for `Matcher::match_iter`, which branches on the
    /// typo/unicode configuration at runtime rather than expanding a hot loop
    /// per configuration, so it monomorphizes once per backend. Each kernel
    /// call crosses the `#[target_feature]` boundary instead of inlining into
    /// a shared loop.
    #[inline(always)]
    pub(super) fn match_one_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<Match> {
        let haystack = haystack.as_ref().as_bytes();
        let max_typos = self.max_typos_runtime::<TYPOS>();
        let original_len = haystack.len();
        if original_len < self.min_haystack_len {
            return None;
        }

        let (matched, start_pos, end_pos) =
            self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
        if !matched {
            return None;
        }

        let (trimmed, start_pos) = trim_haystack(haystack, start_pos, end_pos);
        let include_exact = start_pos == 0 && end_pos == original_len;
        Some(self.smith_waterman_one::<UNICODE>(trimmed, index, start_pos, include_exact))
    }

    /// Single-haystack path for `Matcher::match_iter_indices`, mirroring
    /// `match_one_impl` but returning the matched character indices. Like the
    /// list variant it branches on the typo/unicode configuration at runtime so
    /// it monomorphizes once per backend.
    #[inline(always)]
    pub(super) fn match_one_indices_impl<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystack: H,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_ref().as_bytes();
        let max_typos = self.max_typos_runtime::<TYPOS>();
        let max_typos_opt = if TYPOS == NO_PREFILTER {
            None
        } else {
            Some(max_typos)
        };
        let original_len = haystack.len();
        if original_len < self.min_haystack_len {
            return None;
        }

        let (matched, start_pos, end_pos) =
            self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
        if !matched {
            return None;
        }

        let (trimmed, start_pos) = trim_haystack(haystack, start_pos, end_pos);
        let include_exact = start_pos == 0 && end_pos == original_len;
        self.smith_waterman_indices_one::<UNICODE>(
            trimmed,
            start_pos,
            index,
            include_exact,
            max_typos_opt,
        )
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
            if original_len >= self.min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let (trimmed, start_pos) = trim_haystack(haystack, start_pos, end_pos);
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
            self.smith_waterman
                .score_haystack_unicode(haystack, haystack_start_pos)
        } else {
            self.smith_waterman
                .score_haystack(haystack, haystack_start_pos)
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
        haystack_start_pos: usize,
        index: u32,
        include_exact: bool,
        max_typos: Option<u16>,
    ) -> Option<MatchIndices> {
        let (mut score, indices) = if UNICODE {
            self.smith_waterman.score_haystack_unicode_indices(
                haystack,
                haystack_start_pos,
                max_typos,
            )?
        } else {
            self.smith_waterman
                .score_haystack_indices(haystack, haystack_start_pos, max_typos)?
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
    fn guard_against_score_overflow(&self) {
        // The largest bonus a matched character can earn on top of `match_score`: a gap-adjusted
        // delimiter or half a capitalization bonus (whichever is larger), plus the case bonus.
        let scoring = &self.config.scoring;
        let max_bonus_per_char = scoring
            .delimiter_bonus
            .saturating_sub(scoring.gap_open_penalty)
            .max(scoring.capitalization_bonus.div_ceil(2))
            .saturating_add(scoring.matching_case_bonus);
        scoring.guard_against_score_overflow(self.needle.len(), max_bonus_per_char);
    }
}

#[inline(always)]
fn trim_haystack(haystack: &[u8], start_pos: usize, end_pos: usize) -> (&[u8], usize) {
    // substract 1 so that we add the delimiter bonus from the first char
    // otherwise, we would never see it in the smith waterman
    let start_pos = start_pos.saturating_sub(1);
    (&haystack[start_pos..end_pos], start_pos)
}

#[cfg(test)]
mod tests {
    use crate::{Config, Matcher, Scoring, SortStrategy};

    #[test]
    fn all_zero_scoring_does_not_divide_by_zero() {
        let config = Config::default().scoring(Scoring {
            match_score: 0,
            mismatch_penalty: 0,
            gap_open_penalty: 0,
            gap_extend_penalty: 0,
            prefix_bonus: 0,
            capitalization_bonus: 0,
            matching_case_bonus: 0,
            exact_match_bonus: 0,
            delimiter_bonus: 0,
        });
        Matcher::new("foo", &config).match_list(&["foobar"]);
    }

    #[test]
    fn gap_open_below_gap_extend_does_not_underflow() {
        let config = Config::default().scoring(Scoring {
            gap_open_penalty: 1,
            gap_extend_penalty: 5,
            ..Scoring::default()
        });
        Matcher::new("foo", &config).match_list(&["foobar", "fabco"]);
    }

    #[test]
    #[should_panic(expected = "needle too long")]
    fn huge_bonuses_report_descriptive_overflow_error() {
        let config = Config::default().scoring(Scoring {
            capitalization_bonus: 60000,
            matching_case_bonus: 40000,
            ..Scoring::default()
        });
        Matcher::new("f", &config);
    }

    #[test]
    fn unsorted_output_preserves_candidate_order() {
        let haystacks = ["foo", "nomatch", "xfoo", "f_o_o", "bar"];
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("foo", &config).match_list(&haystacks);
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
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("abc", &config).match_list_indices(&haystacks);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[0].indices, vec![3, 2, 1]);
        assert_eq!(matches[1].index, 1);
        assert_eq!(matches[1].indices, vec![4, 2, 0]);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn filtered_match_end_col_uses_original_haystack_offsets() {
        let config = Config::default().sort(SortStrategy::IndexAsc);

        let matches = Matcher::new("abc", &config).match_list(&["xxabcxx"]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].end_col, 4);
    }
}
