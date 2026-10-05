use crate::prefilter::{Kernel as PrefilterKernel, Window};
use crate::smith_waterman::Kernel as SmithWatermanKernel;
use crate::{Config, Match, MatchIndices};
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

/// Magic numbers for `TYPOS` specialization keys beyond 0: any typo budget
/// (read at runtime) and no prefilter
pub(super) const MANY_TYPOS: u16 = u16::MAX;
pub(super) const NO_PREFILTER: u16 = u16::MAX - 1;

/// Fully inlined per-backend implementations, specialized for each
/// configuration (0 typos, 1 typo, unicode variants, ...) and built with the
/// backend's `#[target_feature]`.
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

    /// Off the hot path, so the typo count is read at runtime rather than
    /// specialized
    unsafe fn match_one_indices<const UNICODE: bool>(
        &mut self,
        haystack: &str,
        index: u32,
    ) -> Option<MatchIndices>;
}

/// The Smith Waterman kernels, outlined (`inline(never)`) so each is built
/// once per backend rather than once per `TYPOS` specialization
///
/// # Safety
/// The backend's required CPU features must be available
pub(crate) trait Scorer {
    unsafe fn score<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
    ) -> u16;

    /// Scores two ASCII haystacks at once
    unsafe fn score_pair(&mut self, haystacks: [&[u8]; 2], start_pos: [usize; 2]) -> [u16; 2];
}

#[derive(Debug, Clone)]
pub struct MatcherImpl<P: PrefilterKernel, S: SmithWatermanKernel> {
    needle: String,
    config: Config,
    min_haystack_len: usize,
    pub(super) prefilter: P,
    pub(super) smith_waterman: S,
}

impl<P, S> MatcherImpl<P, S>
where
    P: PrefilterKernel,
    S: SmithWatermanKernel,
    Self: Scorer,
{
    #[inline(always)]
    pub fn new(needle: &str, config: &Config) -> Self {
        let case_sensitive = config.casing.respects_case_for(needle);
        let unicode = config.unicode.respects_unicode_for(needle);
        Self {
            needle: needle.to_string(),
            config: config.clone(),
            min_haystack_len: config
                .max_typos
                .map(|max| needle.chars().count().saturating_sub(max as usize))
                .unwrap_or(0),
            prefilter: P::new(needle, case_sensitive),
            smith_waterman: S::new(needle, &config.scoring, case_sensitive, unicode),
        }
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
        // `MANY_TYPOS` reads the budget at runtime, 0 is specialized, and
        // `NO_PREFILTER` doesn't use it
        let max_typos = if TYPOS == MANY_TYPOS {
            self.config.max_typos.unwrap_or(0)
        } else {
            TYPOS
        };
        let mut pending = None;
        for (index, haystack_str) in (haystack_index_offset..).zip(haystacks.iter()) {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= self.min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let candidate = trim_haystack(index, haystack, start_pos, end_pos);
                    self.score_candidate::<TYPOS, UNICODE>(candidate, &mut pending, matches);
                }
            }
        }
        self.score_pending::<TYPOS, UNICODE>(pending, matches);
    }

    /// Scores a prefiltered candidate. ASCII ones are scored in pairs, so the
    /// kernel's dependency chains overlap: the first one waits in `pending`
    /// for its partner. Not on wasm, for binary size, nor with the end column,
    /// which reads back the matrices of the last scored haystack
    #[inline(always)]
    fn score_candidate<'a, const TYPOS: u16, const UNICODE: bool>(
        &mut self,
        candidate: Candidate<'a>,
        pending: &mut Option<Candidate<'a>>,
        matches: &mut Vec<Match>,
    ) {
        if UNICODE || cfg!(target_arch = "wasm32") || cfg!(feature = "match_end_col") {
            matches.push(self.smith_waterman_one::<TYPOS, UNICODE>(candidate));
        } else if let Some(first) = pending.take() {
            let [score_a, score_b] = unsafe {
                self.score_pair(
                    [first.haystack, candidate.haystack],
                    [first.start_pos, candidate.start_pos],
                )
            };
            matches.push(self.finish_match::<UNICODE>(first, score_a));
            matches.push(self.finish_match::<UNICODE>(candidate, score_b));
        } else {
            *pending = Some(candidate);
        }
    }

    /// Scores the candidate left waiting for a partner, if any
    #[inline(always)]
    fn score_pending<const TYPOS: u16, const UNICODE: bool>(
        &mut self,
        pending: Option<Candidate<'_>>,
        matches: &mut Vec<Match>,
    ) {
        if let Some(last) = pending {
            matches.push(self.smith_waterman_one::<TYPOS, UNICODE>(last));
        }
    }

    #[inline(always)]
    pub(super) fn match_one_indices_impl<const UNICODE: bool>(
        &mut self,
        haystack: &str,
        index: u32,
    ) -> Option<MatchIndices> {
        let haystack = haystack.as_bytes();
        let max_typos = self.config.max_typos;
        if haystack.len() < self.min_haystack_len {
            return None;
        }

        let (matched, start_pos, end_pos) = match max_typos {
            None => (true, 0, haystack.len()),
            Some(max_typos) => self
                .prefilter
                .match_haystack_typos::<UNICODE>(haystack, max_typos),
        };
        if !matched {
            return None;
        }

        let candidate = trim_haystack(index, haystack, start_pos, end_pos);
        let (score, indices) =
            self.score_indices_inline::<UNICODE>(candidate.haystack, candidate.start_pos);
        let (score, exact) = self.exact_bonus(candidate, score);
        Some(MatchIndices {
            index,
            score,
            exact,
            indices,
        })
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
            _ => self
                .prefilter
                .match_haystack_typos::<UNICODE>(haystack, max_typos),
        }
    }

    #[inline(always)]
    fn smith_waterman_one<const TYPOS: u16, const UNICODE: bool>(
        &mut self,
        candidate: Candidate<'_>,
    ) -> Match {
        // With no prefilter every haystack is scored, so the kernel stays
        // inline with the loop; otherwise it's outlined and shared
        let score = if TYPOS == NO_PREFILTER {
            self.score_inline::<UNICODE>(candidate.haystack, candidate.start_pos)
        } else {
            unsafe { self.score::<UNICODE>(candidate.haystack, candidate.start_pos) }
        };
        self.finish_match::<UNICODE>(candidate, score)
    }

    /// Builds the match for a scored candidate. The end column reads back the
    /// matrices of the last scored haystack
    #[inline(always)]
    fn finish_match<const UNICODE: bool>(
        &mut self,
        candidate: Candidate<'_>,
        raw_score: u16,
    ) -> Match {
        let (score, exact) = self.exact_bonus(candidate, raw_score);
        Match {
            index: candidate.index,
            score,
            exact,
            // no cell scores in a window without a needle char
            #[cfg(feature = "match_end_col")]
            end_col: if raw_score == 0 {
                0
            } else {
                self.smith_waterman
                    .match_end_col(candidate.haystack, UNICODE, raw_score)
                    .saturating_add(candidate.start_pos.min(u16::MAX as usize) as u16)
            },
        }
    }

    /// Adds the exact match bonus when the candidate is the whole needle
    #[inline(always)]
    fn exact_bonus(&self, candidate: Candidate<'_>, score: u16) -> (u16, bool) {
        let exact = candidate.include_exact && self.needle.as_bytes() == candidate.haystack;
        let bonus = if exact {
            self.config.scoring.exact_match_bonus
        } else {
            0
        };
        (score.saturating_add(bonus), exact)
    }

    #[inline(always)]
    pub(super) fn score_inline<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
    ) -> u16 {
        if UNICODE {
            self.smith_waterman
                .score_haystack_unicode(haystack, haystack_start_pos)
        } else {
            self.smith_waterman
                .score_haystack(haystack, haystack_start_pos)
        }
    }

    #[inline(always)]
    pub(super) fn score_indices_inline<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
    ) -> (u16, Vec<u32>) {
        if UNICODE {
            self.smith_waterman
                .score_haystack_unicode_indices(haystack, haystack_start_pos)
        } else {
            self.smith_waterman
                .score_haystack_indices(haystack, haystack_start_pos)
        }
    }
}

/// A prefiltered haystack awaiting scoring
#[derive(Clone, Copy)]
struct Candidate<'a> {
    index: u32,
    /// The haystack trimmed to the prefilter's window
    haystack: &'a [u8],
    /// Start of the window in the original haystack
    start_pos: usize,
    /// Whether the window covers the full haystack (making it eligible for the
    /// exact match bonus)
    include_exact: bool,
}

/// Trims the haystack to the prefilter's window
#[inline(always)]
fn trim_haystack(index: u32, haystack: &[u8], start_pos: usize, end_pos: usize) -> Candidate<'_> {
    // substract 1 so that we add the delimiter bonus from the first char
    // otherwise, we would never see it in the smith waterman
    let start_pos = start_pos.saturating_sub(1);
    Candidate {
        index,
        haystack: &haystack[start_pos..end_pos],
        start_pos,
        include_exact: start_pos == 0 && end_pos == haystack.len(),
    }
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
    fn huge_bonuses_saturate_instead_of_panicking() {
        let config = Config::default().scoring(Scoring {
            capitalization_bonus: 60000,
            matching_case_bonus: 40000,
            ..Scoring::default()
        });
        let matches = Matcher::new("f", &config).match_list(&["f", "z"]);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn unicode_needles_score_per_char_row() {
        // the unicode path uses one score row per char (8), not per byte (24)
        let needle = "一二三四五六七八";
        let config = Config::default().scoring(Scoring {
            capitalization_bonus: 4000,
            ..Scoring::default()
        });
        let matches = Matcher::new(needle, &config).match_list(&[needle]);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn greedy_fallback_membership_agrees_between_match_list_and_indices() {
        // the match window exceeds MAX_HAYSTACK_LEN so the greedy fallback runs, and it
        // can't find 'c': both APIs must still return the prefiltered haystack, with
        // score 0 and no indices
        let haystack = format!("a{}b", "z".repeat(1100));
        let config = Config::default().max_typos(Some(1));
        let matches = Matcher::new("abc", &config).match_list(&[haystack.as_str()]);
        let indices = Matcher::new("abc", &config).match_list_indices(&[haystack.as_str()]);
        assert_eq!(matches.len(), 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(matches[0].score, indices[0].score);
        assert!(indices[0].indices.is_empty());
    }

    #[test]
    fn penalty_above_u8_range_is_not_truncated() {
        // mismatch_penalty 260 previously wrapped to 4 in the u8 backend's splat,
        // so raising the penalty raised the score
        let score = |mismatch_penalty| {
            let config = Config::default().max_typos(Some(1)).scoring(Scoring {
                mismatch_penalty,
                ..Scoring::default()
            });
            Matcher::new("abc", &config).match_list(&["aXc"])[0].score
        };
        assert!(score(260) <= score(255));
    }

    #[test]
    fn zero_gap_capitalization_scores_do_not_saturate_u8() {
        // with no gap penalties, every matched char can earn the full capitalization
        // bonus, so the u8 backend's 255 ceiling must not be selected
        let config = Config::default().scoring(Scoring {
            match_score: 40,
            capitalization_bonus: 40,
            mismatch_penalty: 0,
            gap_open_penalty: 0,
            gap_extend_penalty: 0,
            prefix_bonus: 0,
            matching_case_bonus: 0,
            exact_match_bonus: 0,
            delimiter_bonus: 0,
        });
        let matches = Matcher::new("BBBB", &config).match_list(&["aBaBaBaB"]);
        assert_eq!(matches[0].score, 4 * (40 + 40));
    }

    #[test]
    fn grouped_haystacks_score_as_alone() {
        // Every backend scores ASCII haystacks in pairs and Unicode ones
        // alone, with leftovers, haystacks past the kernels' limit and scores
        // above 255, as it does one at a time
        use super::super::backend::*;
        use super::{MANY_TYPOS, NO_PREFILTER, Specialized};
        use crate::Match;
        use crate::smith_waterman::reference::Rng;
        use alloc::string::String;

        fn check<M: Specialized, const TYPOS: u16, const UNICODE: bool>(
            needle: &str,
            haystacks: &[String],
            max_typos: Option<u16>,
        ) {
            let config = Config::default().max_typos(max_typos);
            let mut matcher = unsafe { M::build(needle, &config) };
            for n in [1, 2, 3, 5, 6, 7, 9, haystacks.len()] {
                let haystacks = &haystacks[..n];
                let mut alone = Vec::new();
                for (i, haystack) in (0..).zip(haystacks) {
                    unsafe { matcher.match_list::<TYPOS, UNICODE, _>(&[haystack], i, &mut alone) };
                }
                let mut matches: Vec<Match> = Vec::new();
                unsafe { matcher.match_list::<TYPOS, UNICODE, _>(haystacks, 0, &mut matches) };
                assert_eq!(matches, alone, "{needle} {max_typos:?} {n}");
                if needle.starts_with('/') && n > 9 {
                    assert!(alone.iter().any(|m| m.score > 255), "{needle}");
                }
            }
        }

        let ascii = [
            "components",
            "/",
            "policy",
            "Policy",
            "compnents",
            "polcy",
            "_",
            "x/",
        ];
        let unicode = [
            "니다",
            "합니다",
            "하는",
            "다",
            "가나다라마바사",
            "a니b다",
            "😀다",
            " ",
        ];
        let needles = [
            "components/policy",
            "policy",
            "/a/b/c/d/e/f/g",
            "니다",
            "/a/b/c/d/e/f/다",
        ];
        let mut rng = Rng(12345);
        let mut haystacks: Vec<String> = Vec::new();
        for _ in 0..if cfg!(miri) { 12 } else { 1000 } {
            let words = if rng.below(2) == 0 { &ascii } else { &unicode };
            let mut haystack = match rng.below(8) {
                0 => rng.pick(&needles).into(),
                1 => "x".repeat(1100),
                _ => String::new(),
            };
            for _ in 0..rng.below(24) {
                haystack.push_str(rng.pick(words));
            }
            haystacks.push(haystack);
        }

        macro_rules! check {
            ($m:ty) => {
                check!($m, needles)
            };
            ($m:ty, $needles:expr) => {
                if <$m>::is_available() {
                    for needle in $needles {
                        if needle.is_ascii() {
                            check::<$m, 0, false>(needle, &haystacks, Some(0));
                            check::<$m, MANY_TYPOS, false>(needle, &haystacks, Some(1));
                            check::<$m, NO_PREFILTER, false>(needle, &haystacks, None);
                        } else {
                            check::<$m, 0, true>(needle, &haystacks, Some(0));
                            check::<$m, MANY_TYPOS, true>(needle, &haystacks, Some(1));
                            check::<$m, NO_PREFILTER, true>(needle, &haystacks, None);
                        }
                    }
                }
            };
        }
        #[cfg(target_arch = "x86_64")]
        {
            check!(MatcherAVX512);
            // (needles whose u8 scores can't saturate, ASCII within its scan's ramp)
            check!(MatcherAVX512U8, ["policy", "니다"]);
            check!(MatcherAVX);
            check!(MatcherSSE);
        }
        #[cfg(target_arch = "aarch64")]
        check!(MatcherNEON);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        check!(MatcherWasm);
        #[cfg(scalar_fallback)]
        check!(MatcherScalar);
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

    /// Upstream issue #96: the alignment skips the needle's second space (the
    /// haystack's only one is far away), which the traceback counted as a typo
    /// past the budget and stopped, leaving only "real" highlighted
    #[test]
    fn match_list_indices_trace_past_skipped_needle_chars() {
        let haystack = "is thisreal/components//components/ real";
        for max_typos in [Some(0), Some(1), None] {
            let config = Config::default().max_typos(max_typos);
            let matches = Matcher::new("is this real", &config).match_list_indices(&[haystack]);
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].indices, (0..=10).rev().collect::<Vec<_>>());
        }
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
