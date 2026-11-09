use super::MAX_HAYSTACK_LEN;
use crate::smith_waterman::{
    SmithWaterman,
    backend::{Backend, BytesVec, MaskVec, ScoreVec},
    greedy::match_greedy,
};

// Below this needle length the column-major sweep already fits the reorder window
const ROW_MAJOR_MIN_NEEDLE_LEN: usize = 5;

impl<B: Backend> SmithWaterman<B> {
    #[inline(always)]
    pub(crate) fn score_haystack(&mut self, haystack: &[u8], include_prefix: bool) -> u16 {
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
        let needle_len = self.needle_simd.len();

        unsafe {
            // Constants
            let gap_extend_penalty = B::Score::splat(scoring.gap_extend_penalty);
            let gap_open_penalty = B::Score::splat(
                scoring
                    .gap_open_penalty
                    .saturating_sub(scoring.gap_extend_penalty),
            );
            let match_score =
                B::Score::splat(scoring.match_score.saturating_add(scoring.mismatch_penalty));
            let mismatch_penalty = B::Score::splat(scoring.mismatch_penalty);
            let matching_case_bonus = B::Score::splat(scoring.matching_case_bonus);
            let capitalization_bonus = B::Score::splat(scoring.capitalization_bonus);
            let delimiter_bonus = B::Score::splat(scoring.delimiter_bonus);

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

            // Match-conditional bonuses (delimiter, capitalization, prefix) of one chunk
            let mut column_bonuses = |haystack_chunk: B::Bytes| {
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

                let match_and_masked_bonuses = delimiter_bonus_masked
                    .add(capitalization_bonus_masked)
                    .add(prefix_bonus_masked)
                    .add(match_score);
                prefix_bonus_masked = B::Score::zero();
                match_and_masked_bonuses
            };

            // One matrix cell: returns the row scores and match mask of this chunk
            let cell = |needle_char: B::Bytes,
                        flipped_case_needle_char: B::Bytes,
                        haystack_chunk: B::Bytes,
                        match_and_masked_bonuses: B::Score,
                        prev_row_scores: B::Score,
                        up_gap_mask: B::Score,
                        diag_prev_scores: B::Score,
                        left_scores: B::Score,
                        left_mask: B::Score| {
                // Match needle char against the chunk (case insensitive)
                let exact_case_match_mask = needle_char.eq(haystack_chunk);
                let flipped_case_match_mask = flipped_case_needle_char.eq(haystack_chunk);
                let match_mask = B::widen_mask(exact_case_match_mask.or(flipped_case_match_mask));
                let exact_case_match_mask = B::widen_mask(exact_case_match_mask);

                // Diagonal - typical match/mismatch, advancing one cell
                let diag_scores = {
                    let diag = prev_row_scores.shift_right_padded::<1>(diag_prev_scores);
                    // Add bonuses for matches
                    let diag = diag.add(match_mask.and(match_and_masked_bonuses));
                    // Always pay the mismatch penalty
                    let diag = diag.subs(mismatch_penalty);
                    // Reward matching the needle's exact case
                    diag.add(exact_case_match_mask.and(matching_case_bonus))
                };

                // Up - skipping a char in the needle
                let up_scores = {
                    let after_extend = prev_row_scores.subs(gap_extend_penalty);
                    after_extend.subs(up_gap_mask.and(gap_open_penalty))
                };

                // Max of diagonal, up, and left (after gap extension)
                let row_scores = B::propagate_horizontal_gaps(
                    diag_scores.max(up_scores),
                    left_scores,
                    match_mask,
                    left_mask,
                    gap_open_penalty,
                    gap_extend_penalty,
                );
                (row_scores, match_mask)
            };

            if needle_len < ROW_MAJOR_MIN_NEEDLE_LEN {
                for col_idx in 1..haystack_chunks {
                    let haystack_chunk = B::Bytes::load_partial(
                        haystack.as_ptr(),
                        (col_idx - 1) * B::LANES,
                        haystack.len(),
                    );
                    let match_and_masked_bonuses = column_bonuses(haystack_chunk);

                    let mut up_gap_mask = B::Score::zero();
                    let mut prev_row_scores = B::Score::zero();
                    for (row_idx, (needle_char, flipped_case_needle_char)) in
                        self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                    {
                        let (row_scores, match_mask) = cell(
                            *needle_char,
                            *flipped_case_needle_char,
                            haystack_chunk,
                            match_and_masked_bonuses,
                            prev_row_scores,
                            up_gap_mask,
                            score_matrix.get(row_idx - 1, col_idx - 1),
                            score_matrix.get(row_idx, col_idx - 1),
                            match_masks.get(row_idx, col_idx - 1),
                        );
                        score_matrix.set(row_idx, col_idx, row_scores);
                        match_masks.set(row_idx, col_idx, match_mask);
                        prev_row_scores = row_scores;
                        up_gap_mask = match_mask;
                    }

                    // Last row of the matrix at this chunk
                    max_scores = max_scores.max(prev_row_scores);
                }
                return max_scores.horizontal_max();
            }

            for col_idx in 1..haystack_chunks {
                let haystack_chunk = B::Bytes::load_partial(
                    haystack.as_ptr(),
                    (col_idx - 1) * B::LANES,
                    haystack.len(),
                );
                *self.col_bytes.get_unchecked_mut(col_idx) = haystack_chunk;
                *self.col_bonuses.get_unchecked_mut(col_idx) = column_bonuses(haystack_chunk);
            }

            // A column-major sweep is one serial chain through gap propagation. Sweeping
            // rows instead lets consecutive rows overlap in the out-of-order window.
            for (row_idx, (needle_char, flipped_case_needle_char)) in
                self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
            {
                let mut left_scores = B::Score::zero();
                let mut left_mask = B::Score::zero();
                let mut diag_prev_scores = B::Score::zero();
                for col_idx in 1..haystack_chunks {
                    let prev_row_scores = score_matrix.get(row_idx - 1, col_idx);
                    let (row_scores, match_mask) = cell(
                        *needle_char,
                        *flipped_case_needle_char,
                        *self.col_bytes.get_unchecked(col_idx),
                        *self.col_bonuses.get_unchecked(col_idx),
                        prev_row_scores,
                        match_masks.get(row_idx - 1, col_idx),
                        diag_prev_scores,
                        left_scores,
                        left_mask,
                    );
                    score_matrix.set(row_idx, col_idx, row_scores);
                    match_masks.set(row_idx, col_idx, match_mask);
                    left_scores = row_scores;
                    left_mask = match_mask;
                    diag_prev_scores = prev_row_scores;
                    if row_idx == needle_len {
                        max_scores = max_scores.max(row_scores);
                    }
                }
            }

            max_scores.horizontal_max()
        }
    }
}
