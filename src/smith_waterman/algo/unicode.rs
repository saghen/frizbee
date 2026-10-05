use super::{MAX_HAYSTACK_LEN, ascii_gap::GAP_STEPS};
use crate::smith_waterman::{
    ScoringVecs, SmithWaterman,
    backend::{Backend, BytesVec, MaskVec, ScoreVec},
    greedy::match_greedy,
};

impl<B: Backend> SmithWaterman<B> {
    /// Scores the haystack without filling a matrix
    #[inline(always)]
    pub(crate) fn score_haystack_unicode(&mut self, haystack: &[u8], include_prefix: bool) -> u16 {
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

        if self.needle_unicode.is_empty() {
            return 0;
        }

        let scoring = &self.scoring;
        let haystack_chunks = haystack.len().div_ceil(B::LANES) + 1;

        let chunk_rows = self.chunk_rows.as_mut_ptr();

        unsafe {
            let ScoringVecs {
                gap_extend_penalty,
                gap_open_penalty,
                match_score,
                mismatch_penalty,
                matching_case_bonus,
                capitalization_bonus,
                delimiter_bonus,
                ..
            } = self.scoring_vecs;

            let final_row_idx = self.needle_unicode.len();
            let mut max_scores = B::Score::zero();

            // Like the ASCII scorer, Unicode scoring is haystack-major: each
            // chunk computes its haystack-side bonuses once, then walks the
            // needle rows, UTF-8 continuation bytes are transport lanes so a
            // scalar's score is not penalized by its byte length
            // State
            // TODO: have prefix bonus scale based on distance
            let mut prefix_bonus_masked = if include_prefix {
                B::Score::first_lane(scoring.prefix_bonus)
            } else {
                B::Score::zero()
            };
            let mut prev_chunk_char_is_delimiter_mask = B::Mask::zero();
            let mut prev_chunk_is_lower_mask = B::Mask::zero();
            let mut prev_chunk_continuation_score_mask = B::Score::zero();
            let mut prev_chunk_gap_steps = [B::Score::zero(); GAP_STEPS];

            for col_idx in 1..haystack_chunks {
                let chunk_start = (col_idx - 1) * B::LANES;
                let haystack_byte_chunks = [
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 3, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 2, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 1, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start, haystack.len()),
                ];
                let haystack_chunk = haystack_byte_chunks[3];
                let (continuation_mask, not_scalar_start_mask) =
                    unicode_scalar_masks::<B>(haystack_chunk, haystack.len(), chunk_start);

                // Lanes that receive a transported score from 1 and 2 lanes
                // back
                let continuation_score_mask = B::widen_mask(continuation_mask);
                let continuation_2_mask = continuation_score_mask.and(
                    continuation_score_mask
                        .shift_right_padded::<1>(prev_chunk_continuation_score_mask),
                );
                prev_chunk_continuation_score_mask = continuation_score_mask;

                // Lanes that don't start a scalar saturate the diagonal and up
                // penalties, so their scores are zero
                let not_scalar_start_score_mask = B::widen_mask(not_scalar_start_mask);
                let scalar_mismatch_penalty = mismatch_penalty.add(not_scalar_start_score_mask);
                let scalar_gap_extend_penalty = gap_extend_penalty.add(not_scalar_start_score_mask);

                let gap_steps = B::unicode_gap_steps(
                    B::widen_mask(continuation_mask.not()).and(gap_extend_penalty),
                    &prev_chunk_gap_steps,
                );
                prev_chunk_gap_steps = gap_steps;

                // Bonus for matching a capital letter after a lowercase ASCII
                // letter
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
                prefix_bonus_masked = B::Score::zero();

                // First chunk reads the zeroed half, so no reset per haystack
                let prev_chunk_rows = if col_idx == 1 {
                    chunk_rows.add(self.needle.len())
                } else {
                    chunk_rows
                };
                // Up gaps pay the open penalty below a match
                let mut up_penalty = scalar_gap_extend_penalty;
                // The previous chunk's row above, shifted in by the diagonal
                let mut prev_chunk_prev_row_scores = B::Score::zero();
                let mut prev_row_scores = B::Score::zero();

                for (row_idx, (needle_char, (chars, flipped_chars))) in self
                    .needle_unicode
                    .iter()
                    .zip(&self.needle_unicode_simd)
                    .enumerate()
                    .map(|(i, c)| (i + 1, c))
                {
                    // Match needle char against the chunk (case insensitive)
                    // Match the whole UTF-8 needle scalar against haystack byte
                    // windows starting at each lane, only the first byte of a
                    // multi-byte Unicode scalar can be marked as a match
                    let exact_case_match_mask =
                        unicode_char_match_mask::<B>(&haystack_byte_chunks, needle_char.len, chars);
                    // Caseless scripts have no flipped variant to check
                    let match_mask = B::widen_mask(if needle_char.has_flip {
                        exact_case_match_mask.or(unicode_char_match_mask::<B>(
                            &haystack_byte_chunks,
                            needle_char.len,
                            flipped_chars,
                        ))
                    } else {
                        exact_case_match_mask
                    });
                    let match_bonuses_masked = match_mask
                        .and(match_and_masked_bonuses)
                        .add(B::widen_mask(exact_case_match_mask).and(matching_case_bonus));

                    // Diagonal - typical match/mismatch, advancing one cell
                    let diag_scores = prev_row_scores
                        .shift_right_padded::<1>(prev_chunk_prev_row_scores)
                        .add(match_bonuses_masked)
                        // Always pay the mismatch penalty
                        .subs(scalar_mismatch_penalty);

                    // Up - skipping a char in the needle
                    let up_scores = prev_row_scores.subs(up_penalty);

                    // Left - skipping scalars in the haystack, with the gap
                    // open penalty paid at the source (see the ASCII row)
                    let gap_open_penalty_masked = match_mask.and(gap_open_penalty);
                    let mut row_scores = diag_scores.max(up_scores);
                    // Transport and gaps can't raise the last row's max:
                    // skip them there
                    if row_idx != final_row_idx {
                        let gap_source_scores = row_scores.subs(gap_open_penalty_masked);
                        let (prev_chunk_row_scores, prev_chunk_gap_scores) =
                            prev_chunk_rows.add(row_idx - 1).read();

                        // Scores cross continuation bytes for free
                        let scores = row_scores.max(
                            row_scores
                                .shift_right_padded::<1>(prev_chunk_row_scores)
                                .and(continuation_score_mask),
                        );
                        let scores = scores.max(
                            scores
                                .shift_right_padded::<2>(prev_chunk_row_scores)
                                .and(continuation_2_mask),
                        );

                        let gap_scores;
                        (gap_scores, row_scores) = B::propagate_horizontal_gaps(
                            gap_source_scores,
                            prev_chunk_gap_scores,
                            &gap_steps,
                            scores,
                        );
                        chunk_rows.add(row_idx - 1).write((row_scores, gap_scores));
                        prev_chunk_prev_row_scores = prev_chunk_row_scores;
                    }
                    prev_row_scores = row_scores;
                    up_penalty = scalar_gap_extend_penalty.add(gap_open_penalty_masked);
                }

                // Last row of the matrix at this chunk
                max_scores = max_scores.max(prev_row_scores);
            }

            max_scores.horizontal_max()
        }
    }
}

/// Lanes where the needle scalar's UTF-8 bytes start. A scalar's lead byte is
/// never a continuation byte, so this only matches at scalar starts.
#[inline(always)]
unsafe fn unicode_char_match_mask<B: Backend>(
    haystack_chunks: &[B::Bytes; 4],
    char_len: usize,
    chars: &[B::Bytes; 4],
) -> B::Mask {
    unsafe {
        // No early exit, the branch mispredicts
        let mut mask = haystack_chunks[3].eq(chars[0]);
        for byte_idx in 1..char_len {
            mask = mask.and(haystack_chunks[3 - byte_idx].eq(chars[byte_idx]));
        }
        mask
    }
}

/// Continuation byte lanes, and lanes that don't start a haystack scalar
#[inline(always)]
unsafe fn unicode_scalar_masks<B: Backend>(
    current_bytes: B::Bytes,
    haystack_len: usize,
    start: usize,
) -> (B::Mask, B::Mask) {
    unsafe {
        // Zeroed lanes past the haystack's end aren't continuation bytes
        let continuation_mask = current_bytes
            .gt(B::Bytes::splat(0x7f))
            .and(current_bytes.lt(B::Bytes::splat(0xc0)));
        let not_scalar_start_mask =
            continuation_mask.or(valid_haystack_lanes::<B>(haystack_len, start).not());
        (continuation_mask, not_scalar_start_mask)
    }
}

#[inline(always)]
pub(super) unsafe fn valid_haystack_lanes<B: Backend>(
    haystack_len: usize,
    start: usize,
) -> B::Mask {
    const INDICES: [u8; 64] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ];
    unsafe {
        let valid_lanes = haystack_len.saturating_sub(start).min(B::LANES);
        let indices = B::Bytes::load_partial(INDICES.as_ptr(), 0, B::LANES);
        indices.lt(B::Bytes::splat(valid_lanes as u8))
    }
}
