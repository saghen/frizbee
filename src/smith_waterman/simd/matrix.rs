use super::backend::{Backend, ScoreVec};

#[derive(Debug, Clone)]
pub struct Matrix<B: Backend> {
    pub matrix: Vec<B::Score>,
    pub needle_len: usize,
    /// Number of LANES-wide chunks per row, including the leading zero column.
    pub haystack_chunks: usize,
}

impl<B: Backend> Matrix<B> {
    #[inline(always)]
    pub fn new(needle_len: usize, haystack_len: usize) -> Self {
        let haystack_chunks = haystack_len.div_ceil(B::LANES) + 1;
        let matrix = (0..((needle_len + 1) * haystack_chunks))
            .map(|_| unsafe { B::Score::zero() })
            .collect();
        Self {
            matrix,
            needle_len,
            haystack_chunks,
        }
    }

    #[inline(always)]
    pub fn get(&self, needle_idx: usize, haystack_idx: usize) -> B::Score {
        unsafe {
            *self
                .matrix
                .get_unchecked(needle_idx * self.haystack_chunks + haystack_idx)
        }
    }

    #[inline(always)]
    pub fn set(&mut self, needle_idx: usize, haystack_idx: usize, value: B::Score) {
        unsafe {
            *self
                .matrix
                .get_unchecked_mut(needle_idx * self.haystack_chunks + haystack_idx) = value;
        }
    }

    /// View the matrix as a flat byte slice.
    ///
    /// Each lane occupies `B::LANE_BYTES` bytes. The byte offset for cell
    /// `(row, lane_index)` is
    ///   `(row * haystack_chunks * LANES + lane_index) * LANE_BYTES`,
    /// where `lane_index = chunk_idx * LANES + lane_within_chunk`. This lets
    /// the alignment iterator read either u8 (LANE_BYTES = 1) or u16
    /// (LANE_BYTES = 2) cells from the same backing storage.
    #[inline(always)]
    pub fn as_byte_slice(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self.matrix.as_ptr() as *const u8,
                self.matrix.len() * B::LANES * B::LANE_BYTES,
            )
        }
    }
}
