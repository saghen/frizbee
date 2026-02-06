use std::cmp::Ordering;

use bumpalo::collections::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::simd::AVXVector;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IncrementalMatch<'a> {
    pub score: u16,
    /** Index of the match in the original list of haystacks */
    pub index: u32,
    /** Matched the needle exactly (i.e. "foo" on "foo") */
    pub exact: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) score_matrix: Vec<'a, AVXVector>,
}

// TODO: drop when switching to a thread safe bump allocator
unsafe impl<'a> Send for IncrementalMatch<'a> {}

impl<'a> PartialOrd for IncrementalMatch<'a> {
    fn partial_cmp(&self, other: &IncrementalMatch) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}
impl<'a> Ord for IncrementalMatch<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.score as u64)
            .cmp(&(other.score as u64))
            .reverse()
            .then_with(|| self.index.cmp(&other.index))
    }
}
impl<'a> PartialEq for IncrementalMatch<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.index == other.index
    }
}
impl<'a> Eq for IncrementalMatch<'a> {}

impl<'a> IncrementalMatch<'a> {
    pub fn extend(&mut self, len: usize, haystack_len: usize) {
        let haystack_chunk = haystack_len.div_ceil(16) + 1;
        let capacity = len * haystack_chunk;
        self.score_matrix.reserve(capacity);
        unsafe {
            std::ptr::write_bytes(
                self.score_matrix
                    .as_mut_ptr()
                    .add(self.score_matrix.len() - capacity),
                0,
                capacity,
            );
            self.score_matrix.set_len(capacity);
        }
    }

    pub fn truncate(&mut self, len: usize, haystack_len: usize) {
        let haystack_chunk = haystack_len.div_ceil(16) + 1;
        self.score_matrix
            .truncate((self.score_matrix.len() - len + 1) * haystack_chunk);
    }
}
