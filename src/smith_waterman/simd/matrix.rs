use crate::simd::Vector256;

#[derive(Debug, Clone)]
pub struct Matrix<Simd256: Vector256> {
    pub matrix: Vec<Simd256>,
    pub needle_len: usize,
    pub haystack_chunks: usize,
}

impl<Simd256: Vector256> Matrix<Simd256> {
    #[inline(always)]
    pub fn new(needle_len: usize, haystack_len: usize) -> Self {
        let haystack_chunks = haystack_len.div_ceil(16) + 1;
        let matrix = (0..((needle_len + 1) * haystack_chunks))
            .map(|_| unsafe { Simd256::splat_u16(0) })
            .collect();
        Self {
            matrix,
            needle_len,
            haystack_chunks,
        }
    }

    #[inline]
    pub fn reserve_haystack_len(&mut self, haystack_len: usize) {
        let haystack_chunks = haystack_len.div_ceil(16) + 1;
        if haystack_chunks <= self.haystack_chunks {
            return;
        }

        self.matrix = (0..((self.needle_len + 1) * haystack_chunks))
            .map(|_| unsafe { Simd256::splat_u16(0) })
            .collect();
        self.haystack_chunks = haystack_chunks;
    }

    #[inline(always)]
    pub fn get(&self, needle_idx: usize, haystack_idx: usize) -> Simd256 {
        unsafe {
            *self
                .matrix
                .get_unchecked(haystack_idx * (self.needle_len + 1) + needle_idx)
        }
    }

    #[inline(always)]
    pub fn set(&mut self, needle_idx: usize, haystack_idx: usize, value: Simd256) {
        unsafe {
            *self
                .matrix
                .get_unchecked_mut(haystack_idx * (self.needle_len + 1) + needle_idx) = value;
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[[u16; 16]] {
        unsafe { std::mem::transmute::<&[Simd256], &[[u16; 16]]>(&self.matrix) }
    }
}
