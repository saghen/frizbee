use core::arch::x86_64::*;

use super::{BLOCK_PAD, Backend};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefilterAVX512Backend;

impl Backend for PrefilterAVX512Backend {
    const LANES: usize = 64;

    type Chunk = __m512i;
    type Mask = u64;
    type Block = u128;
    type Chunks = [__m512i; 2];
    const PADDED_BLOCKS: bool = true;

    fn is_available() -> bool {
        let features = crate::cpuid::detect();
        features.avx512f && features.avx512bw && features.bmi1 && features.bmi2
    }

    #[inline(always)]
    unsafe fn splat(c: u8) -> Self::Chunk {
        unsafe { _mm512_set1_epi8(c as i8) }
    }

    #[inline(always)]
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask {
        unsafe { _mm512_cmpeq_epi8_mask(a, b) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const u8) -> Self::Chunk {
        unsafe { _mm512_loadu_si512(ptr as *const __m512i) }
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const u8, _remaining: usize, mask: Self::Mask) -> Self::Chunk {
        // AVX-512 has native masked byte loads, so loads at page boundaries
        // don't need a copy based fallback
        unsafe { _mm512_maskz_loadu_epi8(mask, ptr as *const i8) }
    }

    #[inline(always)]
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask {
        unsafe { _mm512_cmpeq_epi8_mask(needle.0, chunk) | _mm512_cmpeq_epi8_mask(needle.1, chunk) }
    }

    #[inline(always)]
    unsafe fn first_hit_pos(hit: Self::Mask) -> usize {
        unsafe { _tzcnt_u64(hit) as usize }
    }

    #[inline(always)]
    unsafe fn load_block(haystack: &[u8]) -> (Self::Chunks, Self::Block) {
        // Two masked loads merging into the padding, unconditionally: a
        // masked-off byte is never read, so neither the page past the
        // haystack nor a zero second mask needs a branch
        unsafe {
            let (ptr, remaining) = (haystack.as_ptr(), haystack.len());
            let pad = _mm512_set1_epi8(BLOCK_PAD as i8);
            let m0 = _bzhi_u64(u64::MAX, remaining.min(64) as u32);
            let m1 = _bzhi_u64(u64::MAX, remaining.saturating_sub(64).min(64) as u32);
            let c0 = _mm512_mask_loadu_epi8(pad, m0, ptr as *const i8);
            let c1 = _mm512_mask_loadu_epi8(pad, m1, ptr.wrapping_add(64) as *const i8);
            ([c0, c1], ((m1 as u128) << 64) | m0 as u128)
        }
    }

    #[inline(always)]
    unsafe fn fold_block(chunks: &mut Self::Chunks) {
        unsafe {
            for chunk in chunks.iter_mut() {
                // `A..=Z` as one unsigned range check; adding 0x20 there is
                // the lowercase bit
                let upper = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(*chunk, _mm512_set1_epi8(b'A' as i8)),
                    _mm512_set1_epi8(26),
                );
                *chunk = _mm512_mask_add_epi8(*chunk, upper, *chunk, _mm512_set1_epi8(0x20));
            }
        }
    }

    #[inline(always)]
    unsafe fn eq_block(chunks: &Self::Chunks, needle: Self::Chunk) -> Self::Block {
        unsafe {
            ((Self::eq(chunks[1], needle) as u128) << 64) | Self::eq(chunks[0], needle) as u128
        }
    }
}
