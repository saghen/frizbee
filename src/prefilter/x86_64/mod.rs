use std::arch::x86_64::*;

mod avx2;
mod sse;

pub use avx2::*;
pub use sse::*;

/// Loads a chunk of 16 bytes from the haystack, with overlap when remaining bytes < 16,
/// since it's dramatically faster than a memcpy.
///
/// If the remaining bytes in the haystack is < 16, but the total length is > 16,
/// the last 16 bytes are loaded from the end of the haystack. (start: 16, len: 24, loads: 8-24)
///
/// If the haystack is < 16 bytes, we load the first 8 bytes from the haystack, and the last 8
/// bytes, and combine them into a single vector.
///
/// # Safety
/// Caller must ensure that haystack length >= 8
#[inline(always)]
pub unsafe fn overlapping_load(haystack: &[u8], start: usize, len: usize) -> __m128i {
    unsafe {
        match len {
            0..=7 => unreachable!(),
            8 => _mm_loadl_epi64(haystack.as_ptr() as *const __m128i),
            // Loads 8 bytes from the start of the haystack, and 8 bytes from the end of the haystack
            // and combines them into a single vector. Much faster than a memcpy
            9..=15 => {
                let low = _mm_loadl_epi64(haystack.as_ptr() as *const __m128i);
                let high_start = len - 8;
                let high = _mm_loadl_epi64(haystack[high_start..].as_ptr() as *const __m128i);
                _mm_unpacklo_epi64(low, high)
            }
            16 => _mm_loadu_si128(haystack.as_ptr() as *const __m128i),
            // Avoid reading past the end, instead re-read the last 16 bytes
            _ => _mm_loadu_si128(haystack[start.min(len - 16)..].as_ptr() as *const __m128i),
        }
    }
}
