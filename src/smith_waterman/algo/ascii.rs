use super::MAX_HAYSTACK_LEN;
use crate::smith_waterman::{
    ScoringVecs, SmithWaterman,
    backend::{Backend, BytesVec, MaskVec, ScoreVec},
    greedy::match_greedy,
};

impl<B: Backend> SmithWaterman<B> {
    /// Scores the haystack without filling the matrices, unless they're
    /// needed for the match's end column
    #[inline(always)]
    pub(crate) fn score_haystack(&mut self, haystack: &[u8], include_prefix: bool) -> u16 {
        self.score_haystack_impl::<{ cfg!(feature = "match_end_col") }>(haystack, include_prefix)
    }

    /// Scores the haystack and fills the score matrix for the
    /// alignment path
    #[inline(always)]
    pub(crate) fn score_haystack_matrix(&mut self, haystack: &[u8], include_prefix: bool) -> u16 {
        self.score_haystack_impl::<true>(haystack, include_prefix)
    }

    #[inline(always)]
    fn score_haystack_impl<const STORE: bool>(
        &mut self,
        haystack: &[u8],
        include_prefix: bool,
    ) -> u16 {
        if haystack.len() > MAX_HAYSTACK_LEN {
            return match_greedy(
                self.needle.as_bytes(),
                haystack,
                &self.scoring,
                self.case_sensitive,
                include_prefix,
            )
            .map(|(score, _)| score)
            .unwrap_or(0);
        }

        let scoring = &self.scoring;
        let haystack_chunks = haystack.len().div_ceil(B::LANES) + 1;
        self.haystack_chunks = haystack_chunks;

        // Matrix stride is fixed at construction. Row 0 and column 0 are
        // always zero (never written by the inner loop), so no re-zeroing is
        // needed between calls
        let score_matrix = &mut self.score_matrix;
        let match_masks = &mut self.match_masks;
        let chunk_rows = self.chunk_rows.as_mut_ptr();
        let needle_len = self.needle.len();

        unsafe {
            let ScoringVecs {
                gap_extend_penalty,
                gap_open_penalty,
                match_score,
                mismatch_penalty,
                matching_case_bonus,
                capitalization_bonus,
                delimiter_bonus,
                gap_steps,
            } = self.scoring_vecs;

            // State
            // TODO: have prefix bonus scale based on distance
            let mut prefix_bonus_masked = if include_prefix {
                B::Score::first_lane(scoring.prefix_bonus)
            } else {
                B::Score::zero()
            };
            let mut prev_chunk_char_is_delimiter_mask = B::Mask::zero();
            let mut prev_chunk_is_lower_mask = B::Mask::zero();
            let mut max_scores = B::Score::zero();

            for col_idx in 1..haystack_chunks {
                let haystack_chunk = B::Bytes::load_partial(
                    haystack.as_ptr(),
                    (col_idx - 1) * B::LANES,
                    haystack.len(),
                );
                // Bonus for matching a capital letter after a lowercase letter
                let is_upper_mask = haystack_chunk
                    .lt(B::Bytes::splat(b'Z' + 1))
                    .and(haystack_chunk.gt(B::Bytes::splat(b'A' - 1)));
                let is_lower_mask = haystack_chunk
                    .lt(B::Bytes::splat(b'z' + 1))
                    .and(haystack_chunk.gt(B::Bytes::splat(b'a' - 1)));
                let is_letter_mask = is_upper_mask.or(is_lower_mask);

                // Bonus when an uppercase byte follows a lowercase byte
                let capitalization_mask = B::widen_mask(
                    is_upper_mask.and(is_lower_mask.shift_right_padded_1(prev_chunk_is_lower_mask)),
                );
                let capitalization_bonus_masked = capitalization_mask.and(capitalization_bonus);
                prev_chunk_is_lower_mask = is_lower_mask;

                // Bonus for matching after a delimiter character. We consider
                // anything that isn't a digit or a letter and is within ASCII
                // range to be a delimiter.
                let is_digit_mask = haystack_chunk
                    .gt(B::Bytes::splat(b'0' - 1))
                    .and(haystack_chunk.lt(B::Bytes::splat(b'9' + 1)));
                let char_is_delimiter_mask = is_letter_mask
                    .or(is_digit_mask)
                    .or(haystack_chunk.gt(B::Bytes::splat(127)))
                    .not();
                let prev_char_is_delimiter_mask =
                    char_is_delimiter_mask.shift_right_padded_1(prev_chunk_char_is_delimiter_mask);
                let delimiter_mask =
                    B::widen_mask(prev_char_is_delimiter_mask.and(char_is_delimiter_mask.not()));
                let delimiter_bonus_masked = delimiter_mask.and(delimiter_bonus);
                prev_chunk_char_is_delimiter_mask = char_is_delimiter_mask;

                // Match-conditional bonuses (delimiter, capitalization, prefix)
                let match_and_masked_bonuses = delimiter_bonus_masked
                    .add(capitalization_bonus_masked)
                    .add(prefix_bonus_masked)
                    .add(match_score);

                // First chunk reads the zeroed half, so no reset per haystack
                let prev_chunk_rows = if col_idx == 1 {
                    chunk_rows.add(needle_len)
                } else {
                    chunk_rows
                };
                // Up gaps pay the open penalty below a match
                let mut up_penalty = gap_extend_penalty;
                // The previous chunk's row above, shifted in by the diagonal
                let mut prev_chunk_prev_row_scores = B::Score::zero();
                let mut prev_row_scores = B::Score::zero();

                for (row_idx, (needle_char, flipped_case_needle_char)) in
                    self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                {
                    // Match needle char against the chunk (case insensitive)
                    let exact_case_match_mask = needle_char.eq(haystack_chunk);
                    let flipped_case_match_mask = flipped_case_needle_char.eq(haystack_chunk);
                    let match_mask =
                        B::widen_mask(exact_case_match_mask.or(flipped_case_match_mask));
                    let match_bonuses_masked = match_mask
                        .and(match_and_masked_bonuses)
                        .add(B::widen_mask(exact_case_match_mask).and(matching_case_bonus));

                    // Diagonal - typical match/mismatch, advancing one cell
                    let diag_scores = prev_row_scores
                        .shift_right_padded::<1>(prev_chunk_prev_row_scores)
                        .add(match_bonuses_masked)
                        // Always pay the mismatch penalty
                        .subs(mismatch_penalty);

                    // Up - skipping a char in the needle
                    let up_scores = prev_row_scores.subs(up_penalty);

                    // Left - skipping chars in the haystack. Gaps opened from a
                    // match pay the open penalty at their source
                    let gap_open_penalty_masked = match_mask.and(gap_open_penalty);
                    let mut row_scores = diag_scores.max(up_scores);
                    // Gaps can't raise the last row's max: skip it there
                    // unless the matrix is stored
                    if STORE || row_idx != needle_len {
                        let gap_source_scores = row_scores.subs(gap_open_penalty_masked);
                        let (prev_chunk_row_scores, prev_chunk_gap_scores) =
                            prev_chunk_rows.add(row_idx - 1).read();
                        let gap_scores;
                        (gap_scores, row_scores) = B::propagate_horizontal_gaps(
                            gap_source_scores,
                            prev_chunk_gap_scores,
                            &gap_steps,
                            row_scores,
                        );
                        chunk_rows.add(row_idx - 1).write((row_scores, gap_scores));
                        prev_chunk_prev_row_scores = prev_chunk_row_scores;

                        if STORE {
                            score_matrix.set(row_idx, col_idx, row_scores);
                            match_masks.set(row_idx, col_idx, match_mask);
                        }
                    }
                    prev_row_scores = row_scores;
                    up_penalty = gap_extend_penalty.add(gap_open_penalty_masked);
                }

                // Last row of the matrix at this chunk
                max_scores = max_scores.max(prev_row_scores);
                prefix_bonus_masked = B::Score::zero();
            }

            max_scores.horizontal_max()
        }
    }
}
