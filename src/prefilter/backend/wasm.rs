use core::arch::wasm32::*;

use super::Backend;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterWasmBackend;

impl Backend for PrefilterWasmBackend {
    const LANES: usize = 16;

    type Chunk = v128;
    type Mask = u16;

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
}
