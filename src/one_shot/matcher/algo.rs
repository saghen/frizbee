use super::Matcher;
use crate::prefilter::{Kernel as PrefilterKernel, Window};
use crate::smith_waterman::Kernel as SmithWatermanKernel;
use crate::sort::radix_sort;
use crate::{Config, Match, MatchIndices};

const MANY_TYPOS: u16 = u16::MAX;

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
    pub fn new_impl(needle: &str, config: &Config) -> Self {
        let case_sensitive = config.casing.respects_case_for(needle);
        let matcher = Self {
            needle: needle.to_string(),
            config: config.clone(),
            prefilter: P::new(needle, case_sensitive),
            smith_waterman: S::new(needle.as_bytes(), &config.scoring, case_sensitive),
        };
        matcher.guard_against_score_overflow();
        matcher
    }

    pub fn is_available() -> bool {
        P::is_available() && S::is_available()
    }

    #[inline(always)]
    pub fn match_list_impl<H: AsRef<str>>(&mut self, haystacks: &[H]) -> Vec<Match> {
        let mut matches = vec![];
        self.match_list_into_impl(haystacks, 0, &mut matches);

        if !self.needle.is_empty() && self.config.sort {
            radix_sort(&mut matches);
        }

        matches
    }

    #[inline(always)]
    pub fn match_list_into_impl<H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        Matcher::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);
        if self.needle.is_empty() {
            Matcher::empty_match_list_into(haystacks, haystack_index_offset, matches);
            return;
        }

        let needs_unicode = !self.needle.is_ascii();
        let min_haystack_len = self.min_haystack_len();
        match (self.config.max_typos, needs_unicode) {
            (None, _) => self.match_list_unfiltered_into(haystacks, haystack_index_offset, matches),
            (Some(0), false) => self.match_list_prefiltered_into::<0, false, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                0,
                matches,
            ),
            (Some(0), true) => self.match_list_prefiltered_into::<0, true, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                0,
                matches,
            ),
            (Some(1), false) => self.match_list_prefiltered_into::<1, false, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                1,
                matches,
            ),
            (Some(1), true) => self.match_list_prefiltered_into::<1, true, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                1,
                matches,
            ),
            (Some(2), false) => self.match_list_prefiltered_into::<2, false, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                2,
                matches,
            ),
            (Some(2), true) => self.match_list_prefiltered_into::<2, true, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                2,
                matches,
            ),
            (Some(max_typos), false) => self.match_list_prefiltered_into::<MANY_TYPOS, false, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                max_typos,
                matches,
            ),
            (Some(max_typos), true) => self.match_list_prefiltered_into::<MANY_TYPOS, true, H>(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                max_typos,
                matches,
            ),
        }
    }

    #[inline(always)]
    pub fn match_list_indices_impl<H: AsRef<str>>(&mut self, haystacks: &[H]) -> Vec<MatchIndices> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);
        if self.needle.is_empty() {
            return Matcher::empty_match_list_indices(haystacks);
        }

        let needs_unicode = !self.needle.is_ascii();
        let min_haystack_len = self.min_haystack_len();
        let mut matches = match (self.config.max_typos, needs_unicode) {
            (None, _) => self.match_list_indices_unfiltered(haystacks),
            (Some(0), false) => {
                self.match_list_indices_prefiltered::<0, false, H>(haystacks, min_haystack_len, 0)
            }
            (Some(0), true) => {
                self.match_list_indices_prefiltered::<0, true, H>(haystacks, min_haystack_len, 0)
            }
            (Some(1), false) => {
                self.match_list_indices_prefiltered::<1, false, H>(haystacks, min_haystack_len, 1)
            }
            (Some(1), true) => {
                self.match_list_indices_prefiltered::<1, true, H>(haystacks, min_haystack_len, 1)
            }
            (Some(2), false) => {
                self.match_list_indices_prefiltered::<2, false, H>(haystacks, min_haystack_len, 2)
            }
            (Some(2), true) => {
                self.match_list_indices_prefiltered::<2, true, H>(haystacks, min_haystack_len, 2)
            }
            (Some(max_typos), false) => self
                .match_list_indices_prefiltered::<MANY_TYPOS, false, H>(
                    haystacks,
                    min_haystack_len,
                    max_typos,
                ),
            (Some(max_typos), true) => self.match_list_indices_prefiltered::<MANY_TYPOS, true, H>(
                haystacks,
                min_haystack_len,
                max_typos,
            ),
        };

        if self.config.sort {
            matches.sort_unstable();
        }

        matches
    }

    #[inline(always)]
    fn match_list_unfiltered_into<H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            matches.push(self.smith_waterman_one(haystack_str.as_ref().as_bytes(), index, 0, true));
            index += 1;
        }
    }

    #[inline(always)]
    fn match_list_prefiltered_into<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        haystack_index_offset: u32,
        min_haystack_len: usize,
        max_typos: u16,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) =
                    self.prefilter_haystack::<TYPOS, UNICODE>(haystack, max_typos);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    matches.push(self.smith_waterman_one(
                        trimmed,
                        index as u32,
                        start_pos,
                        include_exact,
                    ));
                }
            }
            index += 1;
        }
    }

    #[inline(always)]
    fn prefilter_haystack<const TYPOS: u16, const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        max_typos: u16,
    ) -> Window {
        match TYPOS {
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
    fn match_list_indices_unfiltered<H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
    ) -> Vec<MatchIndices> {
        let mut matches = vec![];
        for (index, haystack_str) in haystacks.iter().enumerate() {
            let haystack = haystack_str.as_ref().as_bytes();
            if let Some(match_) =
                self.smith_waterman_indices_one(haystack, 0, index as u32, true, None)
            {
                matches.push(match_);
            }
        }
        matches
    }

    #[inline(always)]
    fn match_list_indices_prefiltered<const TYPOS: u16, const UNICODE: bool, H: AsRef<str>>(
        &mut self,
        haystacks: &[H],
        min_haystack_len: usize,
        max_typos: u16,
    ) -> Vec<MatchIndices> {
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
                    if let Some(match_) = self.smith_waterman_indices_one(
                        trimmed,
                        start_pos,
                        index as u32,
                        include_exact,
                        Some(max_typos),
                    ) {
                        matches.push(match_);
                    }
                }
            }
        }
        matches
    }

    #[inline(always)]
    fn smith_waterman_one(
        &mut self,
        haystack: &[u8],
        index: u32,
        haystack_start_pos: usize,
        include_exact: bool,
    ) -> Match {
        let mut score = self.smith_waterman.score_haystack(haystack);

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
    fn smith_waterman_indices_one(
        &mut self,
        haystack: &[u8],
        skipped_chars: usize,
        index: u32,
        include_exact: bool,
        max_typos: Option<u16>,
    ) -> Option<MatchIndices> {
        let (mut score, indices) =
            self.smith_waterman
                .match_haystack_indices(haystack, skipped_chars, max_typos)?;

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
