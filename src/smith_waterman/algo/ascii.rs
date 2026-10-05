use super::{ChunkBonuses, MAX_HAYSTACK_LEN, unicode::valid_haystack_lanes};
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

        let haystack_chunks = haystack.len().div_ceil(B::LANES) + 1;

        let needle_len = self.needle.len();
        let score_matrix = &mut self.score_matrix;
        // The haystack carries through `row_carry[..needle_len]`; the first
        // chunk reads the zeroed second half
        let row_carry = self.row_carry.as_mut_ptr();
        let zero_carry = unsafe { row_carry.add(needle_len) };

        unsafe {
            let ScoringVecs {
                gap_extend_penalty,
                gap_open_penalty,
                mismatch_penalty,
                matching_case_bonus,
                gap_steps,
                ..
            } = self.scoring_vecs;

            let mut chunk_bonuses = ChunkBonuses::<B>::new(
                include_prefix,
                self.scoring.prefix_bonus,
                &self.scoring_vecs,
            );
            let mut max_scores = B::Score::zero();
            // The latest row: at a chunk's end, the last row of the matrix
            // there
            let mut prev_row_scores = B::Score::zero();

            for col_idx in 1..haystack_chunks {
                let haystack_chunk = B::Bytes::load_partial(
                    haystack.as_ptr(),
                    (col_idx - 1) * B::LANES,
                    haystack.len(),
                );
                let match_and_masked_bonuses =
                    chunk_bonuses.next(haystack_chunk, &self.scoring_vecs);
                // The previous chunk wasn't the haystack's last: its last row
                // joins the max unmasked
                max_scores = max_scores.max(prev_row_scores);
                prev_row_scores = B::Score::zero();

                // Up gaps pay the open penalty below a match
                let mut up_penalty = gap_extend_penalty;
                // The previous chunk's row above, shifted in by the diagonal
                let mut prev_chunk_row_above = B::Score::zero();

                for (row_idx, (needle_char, flipped_case_needle_char)) in
                    self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                {
                    let carry_out = row_carry.add(row_idx - 1);
                    let carry_in = if col_idx == 1 {
                        zero_carry.add(row_idx - 1)
                    } else {
                        carry_out
                    };

                    // Match needle char against the chunk (case insensitive)
                    let exact_case_match_mask = needle_char.eq(haystack_chunk);
                    let flipped_case_match_mask = flipped_case_needle_char.eq(haystack_chunk);
                    let match_mask =
                        B::widen_mask(exact_case_match_mask.or(flipped_case_match_mask));
                    // Bonuses for matches, including the needle's exact case
                    let bonuses = match_mask
                        .and(match_and_masked_bonuses)
                        .add(B::widen_mask(exact_case_match_mask).and(matching_case_bonus));

                    // Diagonal - typical match/mismatch, advancing one cell. The
                    // mismatch penalty is always paid (and undone by the match
                    // score)
                    let diag_scores = prev_row_scores
                        .shift_right_padded::<1>(prev_chunk_row_above)
                        .add(bonuses);

                    // Up - skipping a char in the needle
                    let up_scores = prev_row_scores.subs(up_penalty);

                    // Left - skipping chars in the haystack. Gaps opened from a
                    // match pay the open penalty at their source
                    let open = match_mask.and(gap_open_penalty);
                    let mut row_scores = diag_scores.subs(mismatch_penalty).max(up_scores);
                    // The scan can't raise the last row's max: skip it there
                    // unless the matrix is stored
                    if STORE || row_idx != needle_len {
                        let sources = row_scores.subs(open);
                        let (prev_chunk_row, prev_chunk_scan) = carry_in.read();
                        let scan;
                        (scan, row_scores) =
                            B::horizontal_scan(sources, prev_chunk_scan, &gap_steps, row_scores);
                        carry_out.write((row_scores, scan));
                        prev_chunk_row_above = prev_chunk_row;

                        if STORE {
                            score_matrix.set(row_idx, col_idx, row_scores);
                        }
                    }
                    prev_row_scores = row_scores;
                    up_penalty = gap_extend_penalty.add(open);
                }
            }

            // Alignments must not end in the zeroed lanes past the haystack's
            // end, which never feed a lane inside it: masking the last row of
            // the last chunk is enough
            let last_chunk_start = haystack.len().saturating_sub(1) / B::LANES * B::LANES;
            let valid = B::widen_mask(valid_haystack_lanes::<B>(haystack.len(), last_chunk_start));
            max_scores.max(prev_row_scores.and(valid)).horizontal_max()
        }
    }
}
