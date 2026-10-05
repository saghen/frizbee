use core::arch::wasm32::*;

use super::{Backend, ChunkBlock, eq_block_chunks, load_block_chunks};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterWasmBackend;

impl Backend for PrefilterWasmBackend {
    const LANES: usize = 16;

    type Chunk = v128;
    type Mask = u16;
    type Block = u64;
    type Chunks = ChunkBlock<v128, 4>;

    fn is_available() -> bool {
        // simd128 must be enabled at compile-time, it cannot be runtime-detected
        true
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> v128 {
        u8x16_splat(c)
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        u8x16_bitmask(u8x16_eq(a, b))
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { v128_load(ptr as *const v128) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        let mask = v128_or(u8x16_eq(needle.0, chunk), u8x16_eq(needle.1, chunk));
        u8x16_bitmask(mask)
    }

    #[inline(always)]
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block) {
        unsafe { load_block_chunks::<Self, 4>(haystack) }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        unsafe {
            for chunk in chunks.chunks.iter_mut().take(chunks.count) {
                let upper = u8x16_lt(u8x16_sub(*chunk, u8x16_splat(b'A')), u8x16_splat(26));
                *chunk = v128_or(*chunk, v128_and(upper, u8x16_splat(0x20)));
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe { eq_block_chunks::<Self, 4>(chunks, needle) }
    }
}
