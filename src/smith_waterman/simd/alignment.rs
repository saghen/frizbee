use crate::smith_waterman::simd::{AlignmentPathIter, SmithWaterman};

use super::backend::Backend;

impl<B: Backend> SmithWaterman<B> {
    #[inline(always)]
    pub fn iter_alignment_path(
        &self,
        skipped_chars: usize,
        score: u16,
        max_typos: Option<u16>,
    ) -> AlignmentPathIter<'_> {
        AlignmentPathIter::new::<B>(
            &self.score_matrix,
            &self.match_masks,
            self.needle.len(),
            self.haystack_chunks,
            skipped_chars,
            score,
            max_typos,
        )
    }

    #[cfg(test)]
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
