use super::{ChunkBonuses, MAX_HAYSTACK_LEN, unicode::valid_haystack_lanes};
use crate::smith_waterman::{
    ScoringVecs, SmithWaterman,
    backend::{Backend, BytesVec, MaskVec, ScoreVec},
    greedy::match_greedy,
    u8_scan_ramp_fits,
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

    /// Scores two haystacks at once, interleaving their row loops so their
    /// dependency chains overlap. Never stores the matrices.
    #[inline(always)]
    pub(crate) fn score_haystack_pair(
        &mut self,
        [a, b]: [&[u8]; 2],
        [prefix_a, prefix_b]: [bool; 2],
    ) -> [u16; 2] {
        if a.len() > MAX_HAYSTACK_LEN || b.len() > MAX_HAYSTACK_LEN {
            return [
                self.score_haystack_impl::<false>(a, prefix_a),
                self.score_haystack_impl::<false>(b, prefix_b),
            ];
        }
        self.score_chunks::<2, false>([a, b], [prefix_a, prefix_b])
    }

    #[inline(always)]
    pub(super) fn score_haystack_impl<const STORE: bool>(
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
        self.score_chunks::<1, STORE>([haystack], [include_prefix])[0]
    }

    /// Scores `N` haystacks chunk by chunk, with one fused row loop per chunk
    #[inline(always)]
    fn score_chunks<const N: usize, const STORE: bool>(
        &mut self,
        haystacks: [&[u8]; N],
        include_prefix: [bool; N],
    ) -> [u16; N] {
        const { assert!(N == 1 || !STORE) };
        // (the ramped scan needs the headroom, see `Backend::ASCII_SCAN_RAMP`)
        debug_assert!(
            B::ASCII_SCAN_RAMP == 0
                || u8_scan_ramp_fits(self.needle.len(), &self.scoring, B::ASCII_SCAN_RAMP)
        );
        let mut max_len = 0;
        for haystack in haystacks {
            max_len = max_len.max(haystack.len());
        }
        let haystack_chunks = max_len.div_ceil(B::LANES) + 1;

        let needle_len = self.needle.len();
        let score_matrix = &mut self.score_matrix;
        // Haystack `s` carries through `row_carry[2 * s * needle_len..]`; the
        // first chunk reads the zeroed region in between
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

            let mut chunk_bonuses = [ChunkBonuses::<B>::new(false, 0, &self.scoring_vecs); N];
            for s in 0..N {
                chunk_bonuses[s] = ChunkBonuses::new(
                    include_prefix[s],
                    self.scoring.prefix_bonus,
                    &self.scoring_vecs,
                );
            }
            let mut max_scores = [B::Score::zero(); N];
            // Each haystack's latest row: at a chunk's end, the last row of
            // the matrix there
            let mut prev_row_scores = [B::Score::zero(); N];

            for col_idx in 1..haystack_chunks {
                let mut haystack_chunk = [B::Bytes::splat(0); N];
                let mut match_and_masked_bonuses = [B::Score::zero(); N];
                for s in 0..N {
                    let start = (col_idx - 1) * B::LANES;
                    haystack_chunk[s] =
                        B::Bytes::load_partial(haystacks[s].as_ptr(), start, haystacks[s].len());
                    match_and_masked_bonuses[s] =
                        chunk_bonuses[s].next(haystack_chunk[s], &self.scoring_vecs);
                    // The previous chunk wasn't the haystack's last: its last
                    // row joins the max unmasked
                    if N == 1 || start < haystacks[s].len() {
                        max_scores[s] = max_scores[s].max(prev_row_scores[s]);
                        prev_row_scores[s] = B::Score::zero();
                    }
                }

                // Up gaps pay the open penalty below a match
                let mut up_penalty = [gap_extend_penalty; N];
                // The previous chunk's row above, shifted in by the diagonal
                let mut prev_chunk_row_above = [B::Score::zero(); N];

                for (row_idx, (needle_char, flipped_case_needle_char)) in
                    self.needle_simd.iter().enumerate().map(|(i, c)| (i + 1, c))
                {
                    for s in 0..N {
                        // The shorter haystack sits out the longer one's tail
                        if N > 1 && (col_idx - 1) * B::LANES >= haystacks[s].len() {
                            continue;
                        }
                        let carry_out = row_carry.add(2 * s * needle_len + row_idx - 1);
                        let carry_in = if col_idx == 1 {
                            zero_carry.add(row_idx - 1)
                        } else {
                            carry_out
                        };

                        // Match needle char against the chunk (case insensitive)
                        let exact_case_match_mask = needle_char.eq(haystack_chunk[s]);
                        let flipped_case_match_mask =
                            flipped_case_needle_char.eq(haystack_chunk[s]);
                        let match_mask =
                            B::widen_mask(exact_case_match_mask.or(flipped_case_match_mask));
                        // Bonuses for matches, including the needle's exact case
                        let bonuses = match_mask
                            .and(match_and_masked_bonuses[s])
                            .add(B::widen_mask(exact_case_match_mask).and(matching_case_bonus));

                        // Diagonal - typical match/mismatch, advancing one cell. The
                        // mismatch penalty is always paid (and undone by the match
                        // score)
                        let diag_scores = prev_row_scores[s]
                            .shift_right_padded::<1>(prev_chunk_row_above[s])
                            .add(bonuses);

                        // Up - skipping a char in the needle
                        let up_scores = prev_row_scores[s].subs(up_penalty[s]);

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
                                B::ascii_scan(sources, prev_chunk_scan, &gap_steps, row_scores);
                            carry_out.write((row_scores, scan));
                            prev_chunk_row_above[s] = prev_chunk_row;

                            if STORE {
                                score_matrix.set(row_idx, col_idx, row_scores);
                            }
                        }
                        prev_row_scores[s] = row_scores;
                        up_penalty[s] = gap_extend_penalty.add(open);
                    }
                }
            }

            // Alignments must not end in the zeroed lanes past a haystack's
            // end, which never feed a lane inside it: masking the last row of
            // the last chunk is enough
            let mut scores = [0; N];
            for s in 0..N {
                let len = haystacks[s].len();
                let last_chunk_start = len.saturating_sub(1) / B::LANES * B::LANES;
                let valid = B::widen_mask(valid_haystack_lanes::<B>(len, last_chunk_start));
                scores[s] = max_scores[s]
                    .max(prev_row_scores[s].and(valid))
                    .horizontal_max();
            }
            scores
        }
    }
}
