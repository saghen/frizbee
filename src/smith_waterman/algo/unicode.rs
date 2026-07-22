use super::MAX_HAYSTACK_LEN;
use crate::smith_waterman::{
    SmithWaterman,
    backend::{Backend, BytesVec, MaskVec, ScoreVec},
    greedy::match_greedy,
};

impl<B: Backend> SmithWaterman<B> {
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
        self.haystack_chunks = haystack_chunks;

        // Matrix stride is fixed at construction. Row 0 and column 0 are
        // always zero (never written by the inner loop), so no re-zeroing is
        // needed between calls
        let score_matrix = &mut self.score_matrix;
        let match_masks = &mut self.match_masks;
        let unicode_pending_gap_open_masks = &mut self.unicode_pending_gap_open_masks;

        unsafe {
            // Constants
            let gap_extend_penalty = B::Score::splat(scoring.gap_extend_penalty);
            let gap_open_penalty = B::Score::splat(
                scoring
                    .gap_open_penalty
                    .saturating_sub(scoring.gap_extend_penalty),
            );
            let match_score = B::Score::splat(scoring.match_score + scoring.mismatch_penalty);
            let mismatch_penalty = B::Score::splat(scoring.mismatch_penalty);
            let matching_case_bonus = B::Score::splat(scoring.matching_case_bonus);
            let capitalization_bonus = B::Score::splat(scoring.capitalization_bonus);
            let delimiter_bonus = B::Score::splat(scoring.delimiter_bonus);

            let final_row_idx = self.needle_unicode.len();
            let mut max_scores = B::Score::zero();

            for pending_gap_open_mask in unicode_pending_gap_open_masks
                .iter_mut()
                .take(final_row_idx + 1)
            {
                *pending_gap_open_mask = B::Score::zero();
            }

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
            let mut prev_chunk_continuation_gap_extend_penalty = B::Score::zero();
            let mut prev_chunk_scalar_start_mask = B::Score::zero();

            for col_idx in 1..haystack_chunks {
                let chunk_start = (col_idx - 1) * B::LANES;
                let haystack_byte_chunks = [
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 3, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 2, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 1, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start, haystack.len()),
                ];
                let haystack_chunk = haystack_byte_chunks[3];
                let (continuation_mask, scalar_start_mask) =
                    unicode_scalar_masks::<B>(haystack_chunk, haystack.len(), chunk_start);
                let scalar_start_score_mask = B::widen_mask(scalar_start_mask);
                let continuation_gap_extend_penalty =
                    B::widen_mask(continuation_mask).and(gap_extend_penalty);

                // Bonus for matching a capital letter after a lowercase ASCII letter
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

                let mut up_gap_mask = B::Score::zero();
                let mut prev_row_scores = B::Score::zero();
                let mut row_scores = B::Score::zero();

                for (row_idx, needle_char) in self
                    .needle_unicode
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i + 1, c))
                {
                    // Match needle char against the chunk (case insensitive)
                    // Match the whole UTF-8 needle scalar against haystack byte
                    // windows starting at each lane, only the first byte of a
                    // multi-byte Unicode scalar can be marked as a match
                    let exact_case_match_mask = unicode_char_match_mask::<B>(
                        haystack_byte_chunks,
                        scalar_start_mask,
                        needle_char.len,
                        needle_char.chars,
                    );
                    let flipped_case_match_mask = unicode_char_match_mask::<B>(
                        haystack_byte_chunks,
                        scalar_start_mask,
                        needle_char.len,
                        needle_char.flipped_chars,
                    );
                    let match_mask =
                        B::widen_mask(exact_case_match_mask.or(flipped_case_match_mask));
                    let exact_case_match_mask = B::widen_mask(exact_case_match_mask);

                    // Diagonal - typical match/mismatch, advancing one cell
                    let diag_scores = {
                        let diag = prev_row_scores
                            .shift_right_padded::<1>(score_matrix.get(row_idx - 1, col_idx - 1));
                        // Add bonuses for matches
                        let diag = diag.add(match_mask.and(match_and_masked_bonuses));
                        // Always pay the mismatch penalty
                        let diag = diag.subs(mismatch_penalty);
                        // Reward matching the needle's exact case
                        let diag = diag.add(exact_case_match_mask.and(matching_case_bonus));
                        diag.and(scalar_start_score_mask)
                    };

                    // Up - skipping a char in the needle
                    let up_scores = {
                        let after_extend = prev_row_scores.subs(gap_extend_penalty);
                        let up = after_extend.subs(up_gap_mask.and(gap_open_penalty));
                        up.and(scalar_start_score_mask)
                    };

                    // Max of diagonal, up, and left (after gap extension)
                    // Left skips bytes in the haystack
                    let (next_row_scores, pending_gap_open_mask) =
                        B::propagate_horizontal_unicode_gaps(
                            diag_scores.max(up_scores),
                            score_matrix.get(row_idx, col_idx - 1),
                            match_mask,
                            *unicode_pending_gap_open_masks.get_unchecked(row_idx),
                            continuation_gap_extend_penalty,
                            prev_chunk_continuation_gap_extend_penalty,
                            scalar_start_score_mask,
                            prev_chunk_scalar_start_mask,
                            gap_open_penalty,
                            gap_extend_penalty,
                        );

                    score_matrix.set(row_idx, col_idx, next_row_scores);
                    match_masks.set(row_idx, col_idx, match_mask);
                    *unicode_pending_gap_open_masks.get_unchecked_mut(row_idx) =
                        pending_gap_open_mask;
                    prev_row_scores = next_row_scores;
                    row_scores = next_row_scores;
                    up_gap_mask = match_mask;
                }

                // Last row of the matrix at this chunk
                max_scores = max_scores.max(row_scores);
                prev_chunk_continuation_gap_extend_penalty = continuation_gap_extend_penalty;
                prev_chunk_scalar_start_mask = scalar_start_score_mask;
            }

            max_scores.horizontal_max()
        }
    }
}

#[inline(always)]
unsafe fn unicode_char_match_mask<B: Backend>(
    haystack_chunks: [B::Bytes; 4],
    scalar_start_mask: B::Mask,
    char_len: usize,
    chars: [u8; 4],
) -> B::Mask {
    unsafe {
        let mut mask = haystack_chunks[4 - char_len]
            .eq(B::Bytes::splat(chars[char_len - 1]))
            .and(scalar_start_mask);

        if char_len > 1 && !mask.is_zero() {
            for byte_idx in 0..(char_len - 1) {
                let haystack_bytes = haystack_chunks[3 - byte_idx];
                mask = mask.and(haystack_bytes.eq(B::Bytes::splat(chars[byte_idx])));
            }
        }

        mask
    }
}

#[inline(always)]
unsafe fn unicode_scalar_masks<B: Backend>(
    current_bytes: B::Bytes,
    haystack_len: usize,
    start: usize,
) -> (B::Mask, B::Mask) {
    unsafe {
        let valid_mask = valid_haystack_lanes::<B>(haystack_len, start);

        let continuation_mask = current_bytes
            .gt(B::Bytes::splat(0x7f))
            .and(current_bytes.lt(B::Bytes::splat(0xc0)))
            .and(valid_mask);
        let scalar_start_mask = continuation_mask.not().and(valid_mask);

        (continuation_mask, scalar_start_mask)
    }
}

#[inline(always)]
unsafe fn valid_haystack_lanes<B: Backend>(haystack_len: usize, start: usize) -> B::Mask {
    unsafe {
        let valid_lanes = haystack_len.saturating_sub(start).min(B::LANES);
        let mut bytes = [0u8; 64];
        for byte in bytes.iter_mut().take(valid_lanes) {
            *byte = u8::MAX;
        }
        let valid_bytes = B::Bytes::load_partial(bytes.as_ptr(), 0, B::LANES);
        valid_bytes.gt(B::Bytes::splat(0))
    }
}
