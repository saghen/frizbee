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
        skipped_chunks: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'_> {
        AlignmentPathIter::new(
            &self.score_matrix,
            &self.match_masks,
            self.needle.len(),
            self.haystack_chunks,
            skipped_chunks,
            score,
            max_typos,
        )
    }

    #[inline(always)]
    pub fn has_alignment_path(&self, score: u16, max_typos: u16) -> bool {
        if score == 0 {
            return self.needle.len() <= max_typos as usize;
        }
        let iter = AlignmentPathIter::new(
            &self.score_matrix,
            &self.match_masks,
            self.needle.len(),
            self.haystack_chunks,
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
