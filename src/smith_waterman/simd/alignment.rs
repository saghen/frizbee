use crate::{
    simd::Backend,
    smith_waterman::simd::{AlignmentPathIter, algo::SmithWatermanMatcherInternal},
};

impl<B: Backend> SmithWatermanMatcherInternal<B> {
    #[inline(always)]
    pub fn iter_alignment_path(
        &self,
        skipped_chunks: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'_> {
        AlignmentPathIter::new::<B>(
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
        let iter = AlignmentPathIter::new::<B>(
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
