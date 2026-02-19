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

    #[inline(always)]
    pub fn zero(&mut self) {
        unsafe {
            std::ptr::write_bytes(
                self.matrix.as_mut_ptr(),
                0,
                (self.haystack_chunks + 1) * (self.needle_len + 1),
            );
        }
    }

    #[inline(always)]
    pub fn set_haystack_chunks(&mut self, haystack_chunks: usize) {
        self.haystack_chunks = haystack_chunks;
    }

    #[inline(always)]
    pub fn get(&self, needle_idx: usize, haystack_idx: usize) -> Simd256 {
        self.matrix[needle_idx * self.haystack_chunks + haystack_idx]
    }

    #[inline(always)]
    pub fn set(&mut self, needle_idx: usize, haystack_idx: usize, value: Simd256) {
        self.matrix[needle_idx * self.haystack_chunks + haystack_idx] = value;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[[u16; 16]] {
        unsafe { std::mem::transmute::<&[Simd256], &[[u16; 16]]>(&self.matrix) }
    }
}
