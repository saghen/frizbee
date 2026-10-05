use core::arch::x86_64::*;

use super::{Backend, ChunkBlock, eq_block_chunks, load_block_chunks};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVXBackend;

impl Backend for PrefilterAVXBackend {
    const LANES: usize = 32;

    type Chunk = __m256i;
    type Mask = u32;
    type Block = u128;
    type Chunks = ChunkBlock<__m256i, 4>;

    fn is_available() -> bool {
        crate::cpuid::detect().avx2
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe { _mm256_set1_epi8(c as i8) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) as u32 }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm256_loadu_si256(ptr as *const __m256i) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let mask = _mm256_or_si256(
                _mm256_cmpeq_epi8(needle.0, chunk),
                _mm256_cmpeq_epi8(needle.1, chunk),
            );
            _mm256_movemask_epi8(mask) as u32
        }
    }

    #[inline(always)]
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block) {
        unsafe { load_block_chunks::<Self, 4>(haystack) }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        unsafe {
            for chunk in chunks.chunks.iter_mut().take(chunks.count) {
                let shifted = _mm256_sub_epi8(*chunk, _mm256_set1_epi8(b'A' as i8));
                let upper =
                    _mm256_cmpeq_epi8(_mm256_min_epu8(shifted, _mm256_set1_epi8(25)), shifted);
                *chunk = _mm256_or_si256(*chunk, _mm256_and_si256(upper, _mm256_set1_epi8(0x20)));
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe { eq_block_chunks::<Self, 4>(chunks, needle) }
    }
}
