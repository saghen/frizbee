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
    pub fn get(&self, needle_idx: usize, haystack_idx: usize) -> Simd256 {
        unsafe {
            *self
                .matrix
                .get_unchecked(needle_idx * self.haystack_chunks + haystack_idx)
        }
    }

    #[inline(always)]
    pub fn set(&mut self, needle_idx: usize, haystack_idx: usize, value: Simd256) {
        unsafe {
            *self
                .matrix
                .get_unchecked_mut(needle_idx * self.haystack_chunks + haystack_idx) = value;
        }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[[u16; 16]] {
        unsafe { std::mem::transmute::<&[Simd256], &[[u16; 16]]>(&self.matrix) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves that `zero()` must not write more elements than were allocated.
    ///
    /// The bug (before the fix): `zero()` used `(haystack_chunks + 1) * (needle_len + 1)`
    /// but `new()` only allocates `haystack_chunks * (needle_len + 1)`, causing a heap
    /// buffer overflow of `(needle_len + 1)` elements.
    #[test]
    fn test_zero_does_not_overflow_allocation() {
        for needle_len in [1, 3, 5, 10, 20, 50] {
            for haystack_len in [16usize, 32, 64, 128, 256, 512] {
                let haystack_chunks = haystack_len.div_ceil(16) + 1;
                let allocated = (needle_len + 1) * haystack_chunks;

                // zero() must write at most `allocated` elements
                let zeroed = haystack_chunks * (needle_len + 1);
                assert!(
                    zeroed <= allocated,
                    "zero() writes {zeroed} elements but only {allocated} were allocated \
                     (needle_len={needle_len}, haystack_len={haystack_len})"
                );

                // The buggy version: (haystack_chunks + 1) * (needle_len + 1)
                let buggy_zeroed = (haystack_chunks + 1) * (needle_len + 1);
                assert!(
                    buggy_zeroed > allocated,
                    "expected buggy formula to overflow \
                     (needle_len={needle_len}, haystack_len={haystack_len})"
                );
            }
        }
    }

    /// Actually call zero() on a real Matrix and verify no out-of-bounds write.
    /// We append a canary value after the matrix and check it survives zero().
    #[test]
    fn test_zero_preserves_canary() {
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            use crate::simd::{AVXVector, Vector};
            let needle_len = 10;
            let haystack_len = 512; // MAX_HAYSTACK_LEN in algo.rs

            let mut matrix: Matrix<AVXVector> = Matrix::new(needle_len, haystack_len);

            // Push a canary element at the end of the vec
            let canary = unsafe { AVXVector::splat_u16(0xBEEF) };
            matrix.matrix.push(canary);

            // Zero the matrix — this must not touch the canary
            matrix.zero();

            // Verify the canary is untouched
            let last = matrix.matrix.last().unwrap();
            let canary_slice: &[u16; 16] =
                unsafe { std::mem::transmute::<&AVXVector, &[u16; 16]>(last) };
            for &val in canary_slice {
                assert_eq!(
                    val, 0xBEEF,
                    "canary was overwritten — zero() wrote past the allocation"
                );
            }
        }
    }
}
