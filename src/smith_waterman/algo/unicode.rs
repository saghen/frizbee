use super::{ChunkBonuses, MAX_HAYSTACK_LEN};
use crate::smith_waterman::{
    ScoringVecs, SmithWaterman,
    backend::{Backend, BytesVec, MaskVec, ScoreVec},
    greedy::match_greedy,
};

impl<B: Backend> SmithWaterman<B> {
    /// Scores the haystack without filling the matrices, unless they're
    /// needed for the match's end column
    #[inline(always)]
    pub(crate) fn score_haystack_unicode(&mut self, haystack: &[u8], include_prefix: bool) -> u16 {
        self.score_haystack_unicode_impl::<{ cfg!(feature = "match_end_col") }>(
            haystack,
            include_prefix,
        )
    }

    /// Scores the haystack and fills the score matrix for the
    /// alignment path
    #[inline(always)]
    pub(crate) fn score_haystack_unicode_matrix(
        &mut self,
        haystack: &[u8],
        include_prefix: bool,
    ) -> u16 {
        self.score_haystack_unicode_impl::<true>(haystack, include_prefix)
    }

    #[inline(always)]
    fn score_haystack_unicode_impl<const STORE: bool>(
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

        if self.needle_unicode.is_empty() {
            return 0;
        }

        let haystack_chunks = haystack.len().div_ceil(B::LANES) + 1;
        self.haystack_chunks = haystack_chunks;

        unsafe {
            let mut chunk_bonuses = ChunkBonuses::<B>::new(
                include_prefix,
                self.scoring.prefix_bonus,
                &self.scoring_vecs,
            );
            let mut prev_chunk_continuation = B::Score::zero();
            let mut prev_chunk_gap_steps = [B::Score::zero(); 10];
            let mut max_scores = B::Score::zero();

            // Like the ASCII scorer, Unicode scoring is haystack-major: each
            // chunk computes its haystack-side bonuses once, then walks the
            // needle rows, UTF-8 continuation bytes are transport lanes so a
            // scalar's score is not penalized by its byte length
            for col_idx in 1..haystack_chunks {
                let chunk_start = (col_idx - 1) * B::LANES;
                let haystack_byte_chunks = [
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 3, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 2, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start + 1, haystack.len()),
                    B::Bytes::load_partial(haystack.as_ptr(), chunk_start, haystack.len()),
                ];
                let haystack_chunk = haystack_byte_chunks[3];
                let (continuation_mask, not_start_mask) =
                    unicode_scalar_masks::<B>(haystack_chunk, haystack.len(), chunk_start);

                // Lanes that receive a transported score from 1 and 2 lanes back
                let continuation = B::widen_mask(continuation_mask);
                let transport_2 =
                    continuation.and(continuation.shift_right_padded::<1>(prev_chunk_continuation));
                prev_chunk_continuation = continuation;

                let gap_steps = B::unicode_gap_steps(
                    B::widen_mask(continuation_mask.not())
                        .and(self.scoring_vecs.gap_extend_penalty),
                    &prev_chunk_gap_steps,
                );
                prev_chunk_gap_steps = gap_steps;

                let chunk = UnicodeChunk::<B> {
                    bytes: haystack_byte_chunks,
                    not_start: B::widen_mask(not_start_mask),
                    continuation,
                    transport_2,
                    gap_steps,
                    bonuses: chunk_bonuses.next(haystack_chunk, &self.scoring_vecs),
                };
                let final_row = self.unicode_rows::<STORE>(&chunk, col_idx);
                max_scores = max_scores.max(final_row);
            }

            max_scores.horizontal_max()
        }
    }

    /// The needle rows of one chunk, returning the final row
    #[inline(always)]
    unsafe fn unicode_rows<const STORE: bool>(
        &mut self,
        chunk: &UnicodeChunk<B>,
        col_idx: usize,
    ) -> B::Score {
        let score_matrix = &mut self.score_matrix;
        let match_masks = &mut self.match_masks;
        // The first chunk reads the zeroed second half of the row carry, so it
        // needs no reset between haystacks
        let row_carry = self.row_carry.as_mut_ptr();
        let carry_in = if col_idx == 1 {
            unsafe { row_carry.add(self.needle.len()) }
        } else {
            row_carry
        };
        let final_row_idx = self.needle_unicode.len();

        unsafe {
            let ScoringVecs {
                gap_extend_penalty,
                gap_open_penalty,
                mismatch_penalty,
                matching_case_bonus,
                ..
            } = self.scoring_vecs;

            // Lanes that don't start a scalar saturate the diagonal and up
            // penalties, so their scores are zero
            let mismatch = mismatch_penalty.add(chunk.not_start);
            let gap_extend = gap_extend_penalty.add(chunk.not_start);

            // The up penalty, plus the gap open penalty below a match
            let mut up_penalty = gap_extend;
            let mut prev_row_scores = B::Score::zero();
            // The previous chunk's row above, shifted in by the diagonal
            let mut prev_chunk_row_above = B::Score::zero();

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
                let len = needle_char.len;
                let exact_case_match_mask = unicode_char_match_mask::<B>(&chunk.bytes, len, chars);
                // Caseless scripts have no flipped variant to check
                let match_mask = B::widen_mask(if needle_char.has_flip {
                    exact_case_match_mask.or(unicode_char_match_mask::<B>(
                        &chunk.bytes,
                        len,
                        flipped_chars,
                    ))
                } else {
                    exact_case_match_mask
                });
                // Bonuses for matches, including the needle's exact case
                let bonuses = match_mask
                    .and(chunk.bonuses)
                    .add(B::widen_mask(exact_case_match_mask).and(matching_case_bonus));

                // Diagonal - typical match/mismatch, advancing one cell
                let diag_scores = prev_row_scores
                    .shift_right_padded::<1>(prev_chunk_row_above)
                    .add(bonuses);

                // Up - skipping a char in the needle
                let up_scores = prev_row_scores.subs(up_penalty);

                // Left - skipping scalars in the haystack, with the gap
                // open penalty paid at the source (see the ASCII row)
                let open = match_mask.and(gap_open_penalty);
                let mut row_scores = diag_scores.subs(mismatch).max(up_scores);
                // Transport and the scan only add (decayed) copies of
                // scores from this or earlier chunks of the row, so they
                // can't raise the last row's max: skip them and the carry
                // there unless the matrix is stored
                if STORE || row_idx != final_row_idx {
                    // Saturating subtraction commutes with max
                    let sources = row_scores.subs(open);

                    // Scores ride across the scalar's continuation bytes,
                    // so the next row's diagonal sees them at the next
                    // scalar
                    let (prev_chunk_row, prev_chunk_scan) = carry_in.add(row_idx - 1).read();
                    let scores = row_scores.max(
                        row_scores
                            .shift_right_padded::<1>(prev_chunk_row)
                            .and(chunk.continuation),
                    );
                    let scores = scores.max(
                        scores
                            .shift_right_padded::<2>(prev_chunk_row)
                            .and(chunk.transport_2),
                    );

                    let scan;
                    (scan, row_scores) =
                        B::horizontal_scan(sources, prev_chunk_scan, &chunk.gap_steps, scores);
                    row_carry.add(row_idx - 1).write((row_scores, scan));
                    prev_chunk_row_above = prev_chunk_row;

                    if STORE {
                        score_matrix.set(row_idx, col_idx, row_scores);
                        match_masks.set(row_idx, col_idx, match_mask);
                    }
                }
                prev_row_scores = row_scores;
                up_penalty = gap_extend.add(open);
            }

            prev_row_scores
        }
    }
}

/// The haystack-dependent inputs of one chunk's rows, shared by every row
struct UnicodeChunk<B: Backend> {
    /// Haystack bytes at lane offsets 3, 2, 1 and 0
    bytes: [B::Bytes; 4],
    /// Saturated in lanes that don't start a scalar
    not_start: B::Score,
    /// Lanes that receive a transported score from 1 and 2 lanes back
    continuation: B::Score,
    transport_2: B::Score,
    /// Decays of the horizontal scan (see [`super::unicode_gap`])
    gap_steps: [B::Score; 10],
    /// The match score plus the match-conditional bonuses
    bonuses: B::Score,
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
        // Always verify every byte, rather than branching on the first's
        // matches, to avoid branch misprediction
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
        let not_start_mask =
            continuation_mask.or(valid_haystack_lanes::<B>(haystack_len, start).not());
        (continuation_mask, not_start_mask)
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
