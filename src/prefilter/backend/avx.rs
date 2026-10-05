use core::arch::x86_64::*;

use super::{Backend, BitMaskOps, copy_partial};
use crate::prefilter::algo::can_overread;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVXBackend;

impl Backend for PrefilterAVXBackend {
    const LANES: usize = 32;

    type Chunk = __m256i;
    type Mask = u32;
    type Block = u128;
    type Chunks = [__m256i; 4];
    const PADDED_BLOCKS: bool = true;

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
        // Plain loads of the whole block with the padding OR'd in, copied
        // only when the block crosses a page
        unsafe {
            let remaining = haystack.len().min(128);
            let copy;
            let ptr = if can_overread(haystack.as_ptr(), 128) {
                haystack.as_ptr()
            } else {
                copy = copy_partial::<128>(haystack.as_ptr(), remaining);
                copy.as_ptr()
            };
            // (no closures: target features)
            let mut chunks = [
                Self::load(ptr),
                Self::load(ptr.add(32)),
                Self::load(ptr.add(64)),
                Self::load(ptr.add(96)),
            ];
            let last = _mm256_set1_epi8((remaining - 1) as i8);
            for (c, chunk) in chunks.iter_mut().enumerate() {
                let iota = _mm256_add_epi8(
                    _mm256_setr_epi8(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    ),
                    _mm256_set1_epi8((c * 32) as i8),
                );
                *chunk = _mm256_or_si256(*chunk, _mm256_cmpgt_epi8(iota, last));
            }
            (chunks, u128::first_n(remaining))
        }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        unsafe {
            for chunk in chunks {
                let shifted = _mm256_sub_epi8(*chunk, _mm256_set1_epi8(b'A' as i8));
                let upper =
                    _mm256_cmpeq_epi8(_mm256_min_epu8(shifted, _mm256_set1_epi8(25)), shifted);
                *chunk = _mm256_or_si256(*chunk, _mm256_and_si256(upper, _mm256_set1_epi8(0x20)));
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe {
            // Not `chunks.map(..)`: see `load_block`
            let [c0, c1, c2, c3] = [
                Self::eq(chunks[0], needle) as u64,
                Self::eq(chunks[1], needle) as u64,
                Self::eq(chunks[2], needle) as u64,
                Self::eq(chunks[3], needle) as u64,
            ];
            (((c3 << 32) | c2) as u128) << 64 | ((c1 << 32) | c0) as u128
        }
    }
}
