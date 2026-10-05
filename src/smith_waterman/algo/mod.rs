use crate::{
    Scoring,
    prefilter::{case_needle, case_needle_unicode},
    smith_waterman::greedy::match_greedy,
};
use alloc::{string::ToString, vec, vec::Vec};

use super::backend::{Backend, BytesVec, ScoreVec};
use super::{ScoringVecs, SmithWaterman};

mod ascii;
pub(crate) mod ascii_gap;
mod unicode;
pub(crate) mod unicode_gap;

pub(crate) const MAX_HAYSTACK_LEN: usize = 1024;

impl<B: Backend> SmithWaterman<B> {
    pub(crate) fn new(needle: &str, scoring: &Scoring, case_sensitive: bool) -> Self {
        let needle_simd = case_needle(needle.as_bytes(), case_sensitive)
            .iter()
            .map(|(c1, c2)| unsafe { (B::Bytes::splat(*c1), B::Bytes::splat(*c2)) })
            .collect();
        let needle_unicode = case_needle_unicode(needle, case_sensitive);
        let needle_len = needle.len();
        let needle_unicode_simd = needle_unicode
            .iter()
            .map(|c| unsafe {
                (
                    c.chars.map(|b| B::Bytes::splat(b)),
                    c.flipped_chars.map(|b| B::Bytes::splat(b)),
                )
            })
            .collect();
        let scoring_vecs = unsafe {
            ScoringVecs {
                gap_extend_penalty: B::Score::splat(scoring.gap_extend_penalty),
                gap_open_penalty: B::Score::splat(
                    scoring
                        .gap_open_penalty
                        .saturating_sub(scoring.gap_extend_penalty),
                ),
                match_score: B::Score::splat(
                    scoring.match_score.saturating_add(scoring.mismatch_penalty),
                ),
                mismatch_penalty: B::Score::splat(scoring.mismatch_penalty),
                matching_case_bonus: B::Score::splat(scoring.matching_case_bonus),
                capitalization_bonus: B::Score::splat(scoring.capitalization_bonus),
                delimiter_bonus: B::Score::splat(scoring.delimiter_bonus),
                gap_steps: ascii_gap::gap_steps::<B>(scoring.gap_extend_penalty),
            }
        };
        Self {
            needle: needle.to_string(),
            needle_simd,
            needle_unicode,
            needle_unicode_simd,
            case_sensitive,
            scoring: scoring.clone(),
            scoring_vecs,
            chunk_rows: vec![unsafe { (B::Score::zero(), B::Score::zero()) }; 2 * needle_len],
        }
    }

    #[inline(always)]
    pub(crate) fn score_inline<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
    ) -> u16 {
        if UNICODE {
            self.score_haystack_unicode(haystack, haystack_start_pos == 0)
        } else {
            self.score_haystack(haystack, haystack_start_pos == 0)
        }
    }

    #[inline(always)]
    pub(crate) fn score_indices_inline<const UNICODE: bool>(
        &mut self,
        haystack: &[u8],
        haystack_start_pos: usize,
        max_typos: Option<u16>,
    ) -> (u16, Vec<u32>) {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(
                self.needle.as_bytes(),
                haystack,
                &self.scoring,
                self.case_sensitive,
                haystack_start_pos == 0,
            )
            .map(|(score, mut indices)| {
                for index in &mut indices {
                    *index = u32::try_from(*index as usize + haystack_start_pos)
                        .expect("haystack byte index will overflow u32");
                }
                (score, indices)
            })
            .unwrap_or_else(|| (0, vec![]));
        }

        let score = self.score_inline::<UNICODE>(haystack, haystack_start_pos);
        if score == 0 {
            return (score, Vec::new());
        }

        let indices = self.alignment_indices(haystack, haystack_start_pos, UNICODE, max_typos);
        (score, indices)
    }
}
