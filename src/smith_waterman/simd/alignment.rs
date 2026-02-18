use crate::{
    simd::{Vector128Expansion, Vector256},
    smith_waterman::simd::{AlignmentPathIter, algo::SmithWatermanMatcherInternal},
};

impl<Simd128: Vector128Expansion<Simd256>, Simd256: Vector256>
    SmithWatermanMatcherInternal<Simd128, Simd256>
{
    #[inline(always)]
    pub fn iter_alignment_path(
        &self,
        haystack_len: usize,
        skipped_chunks: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'_> {
        AlignmentPathIter::new(
            &self.score_matrix,
            self.needle.len(),
            haystack_len,
            skipped_chunks,
            score,
            max_typos,
        )
    }

    #[inline(always)]
    pub fn has_alignment_path(&self, haystack_len: usize, score: u16, max_typos: u16) -> bool {
        let iter = AlignmentPathIter::new(
            &self.score_matrix,
            self.needle.len(),
            haystack_len,
            0,
            score,
            Some(max_typos),
        );
        for pos in iter {
            if pos.is_none() {
                return false;
            }
        }
        true
    }
}
