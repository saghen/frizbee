use core::arch::x86_64::*;

use super::{Backend, ChunkBlock, eq_block_chunks, load_block_chunks};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterSSEBackend;

impl Backend for PrefilterSSEBackend {
    const LANES: usize = 16;

    type Chunk = __m128i;
    type Mask = u16;
    type Block = u64;
    type Chunks = ChunkBlock<__m128i, 4>;

    fn is_available() -> bool {
        crate::cpuid::detect().sse2
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> __m128i {
        unsafe { _mm_set1_epi8(c as i8) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) as u16 }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm_loadu_si128(ptr as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe {
            let mask = _mm_or_si128(
                _mm_cmpeq_epi8(needle.0, chunk),
                _mm_cmpeq_epi8(needle.1, chunk),
            );
            _mm_movemask_epi8(mask) as u16
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
                let shifted = _mm_sub_epi8(*chunk, _mm_set1_epi8(b'A' as i8));
                let upper = _mm_cmpeq_epi8(_mm_min_epu8(shifted, _mm_set1_epi8(25)), shifted);
                *chunk = _mm_or_si128(*chunk, _mm_and_si128(upper, _mm_set1_epi8(0x20)));
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe { eq_block_chunks::<Self, 4>(chunks, needle) }
    }
}
