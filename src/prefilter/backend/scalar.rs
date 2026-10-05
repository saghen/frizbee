use super::{Backend, ChunkBlock, eq_block_chunks, load_block_chunks};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterScalarBackend;

impl Backend for PrefilterScalarBackend {
    const LANES: usize = 16;

    type Chunk = [u8; 16];
    type Mask = u16;
    type Block = u64;
    type Chunks = ChunkBlock<[u8; 16], 4>;

    fn is_available() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        [c; 16]
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sse2::movemask(sse2::eq(a, b))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut mask = 0u16;
            for (idx, &byte) in a.iter().enumerate() {
                if byte == b[idx] {
                    mask |= 1u16 << idx;
                }
            }
            mask
        }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe {
            let mut chunk = [0u8; 16];
            core::ptr::copy_nonoverlapping(ptr, chunk.as_mut_ptr(), 16);
            chunk
        }
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const u8, remaining: usize, _mask: Self::Mask) -> Self::Chunk {
        unsafe {
            let mut chunk = [0u8; 16];
            core::ptr::copy_nonoverlapping(ptr, chunk.as_mut_ptr(), remaining);
            chunk
        }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            sse2::movemask(sse2::or(
                sse2::eq(chunk, needle.0),
                sse2::eq(chunk, needle.1),
            ))
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut mask = 0u16;
            for (idx, &byte) in chunk.iter().enumerate() {
                // use `|` because `||` prevents auto-vectorization
                if (byte == needle.0[idx]) | (byte == needle.1[idx]) {
                    mask |= 1u16 << idx;
                }
            }
            mask
        }
    }

    #[inline(always)]
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block) {
        unsafe { load_block_chunks::<Self, 4>(haystack) }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        for chunk in chunks.chunks.iter_mut().take(chunks.count) {
            for byte in chunk.iter_mut() {
                *byte = byte.to_ascii_lowercase();
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe { eq_block_chunks::<Self, 4>(chunks, needle) }
    }
}

/// SSE2 is part of x86-64, so the compares there are a compare and a move
/// mask: LLVM turns the per-byte loops into long vector shuffle sequences,
/// repeated at every compare the walks inline
#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::*;

    #[inline(always)]
    pub(super) unsafe fn eq(a: [u8; 16], b: [u8; 16]) -> __m128i {
        unsafe {
            _mm_cmpeq_epi8(
                _mm_loadu_si128(a.as_ptr().cast()),
                _mm_loadu_si128(b.as_ptr().cast()),
            )
        }
    }

    #[inline(always)]
    pub(super) unsafe fn or(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_or_si128(a, b) }
    }

    #[inline(always)]
    pub(super) unsafe fn movemask(eq: __m128i) -> u16 {
        unsafe { _mm_movemask_epi8(eq) as u16 }
    }
}
