use super::super::overlapping_load;
use std::arch::x86_64::*;

/// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
/// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
/// the order does matter across 16 byte boundaries. The needle chars must include both the
/// uppercase and lowercase variants of the character.
///
/// Fastest with SSE2, AVX, and AVX2, but still very fast with just SSE2. Use a function with
/// `#[target_feature(enable = "sse2,avx,avx2")]` or `#[target_feature(enable = "sse2")]`
///
/// # Safety
/// When W > 16, the caller must ensure that the minimum length of the haystack is >= 16.
/// When W <= 16, the caller must ensure that the minimum length of the haystack is >= 8.
/// In all cases, the caller must ensure the needle.len() > 0 and that SSE2 is available.
#[inline(always)]
pub unsafe fn match_haystack_unordered_insensitive(needle: &[(u8, u8)], haystack: &[u8]) -> bool {
    let len = haystack.len();

    let mut needle_iter = needle
        .iter()
        .map(|&(c1, c2)| unsafe { (_mm_set1_epi8(c1 as i8), _mm_set1_epi8(c2 as i8)) });
    let mut needle_char = needle_iter.next().unwrap();

    for start in (0..len).step_by(16) {
        let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

        loop {
            if unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(needle_char.0, haystack_chunk)) } == 0
                && unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(needle_char.1, haystack_chunk)) } == 0
            {
                // No match, advance to next chunk
                break;
            }

            // Progress to next needle char, if available
            if let Some(next_needle_char) = needle_iter.next() {
                needle_char = next_needle_char;
            } else {
                return true;
            }
        }
    }

    false
}

/// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
/// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
/// the order does matter across 16 byte boundaries. The needle chars must include both the
/// uppercase and lowercase variants of the character.
///
/// Fastest with SSE2, AVX, and AVX2, but still very fast with just SSE2. Use a function with
/// `#[target_feature(enable = "sse2,avx,avx2")]` or `#[target_feature(enable = "sse2")]`
///
/// # Safety
/// When W > 16, the caller must ensure that the minimum length of the haystack is >= 16.
/// When W <= 16, the caller must ensure that the minimum length of the haystack is >= 8.
/// In all cases, the caller must ensure the needle.len() > 0 and that SSE2 is available.
#[inline(always)]
pub unsafe fn match_haystack_unordered_insensitive_with_chunks(
    needle: &[(u8, u8)],
    haystack: &[u8],
) -> (bool, usize) {
    let len = haystack.len();

    let mut can_skip_chunks = true;
    let mut skipped_chars = 0;

    let mut needle_iter = needle
        .iter()
        .map(|&(c1, c2)| unsafe { (_mm_set1_epi8(c1 as i8), _mm_set1_epi8(c2 as i8)) });
    let mut needle_char = needle_iter.next().unwrap();

    for start in (0..len).step_by(16) {
        let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

        loop {
            let mask = unsafe {
                _mm_movemask_epi8(_mm_or_si128(
                    _mm_cmpeq_epi8(needle_char.1, haystack_chunk),
                    _mm_cmpeq_epi8(needle_char.0, haystack_chunk),
                ))
            };
            if mask == 0 {
                // No match, advance to next chunk
                if can_skip_chunks {
                    skipped_chars += 1;
                }
                break;
            }

            // Progress to next needle char, if available
            if let Some(next_needle_char) = needle_iter.next() {
                can_skip_chunks = false;
                needle_char = next_needle_char;
            } else {
                return (true, skipped_chars);
            }
        }
    }

    (false, skipped_chars)
}

/// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
/// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
/// the order does matter across 16 byte boundaries. The needle chars must include both the
/// uppercase and lowercase variants of the character.
///
/// Use a function with `#[target_feature(enable = "sse2,avx,avx2")]`
///
/// # Safety
/// When W > 16, the caller must ensure that the minimum length of the haystack is >= 16.
/// When W <= 16, the caller must ensure that the minimum length of the haystack is >= 8.
/// In all cases, the caller must ensure the needle.len() > 0 and that SSE2 and AVX2 are available.
#[inline(always)]
pub unsafe fn match_haystack_unordered_insensitive_avx2(
    needle_simd: &[__m256i],
    haystack: &[u8],
) -> bool {
    let len = haystack.len();

    let mut needle_iter = needle_simd.iter();
    let mut needle_char = *needle_iter.next().unwrap();

    for start in (0..haystack.len()).step_by(16) {
        let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };
        let haystack_chunk = unsafe { _mm256_broadcastsi128_si256(haystack_chunk) };

        // For AVX2, we store the uppercase in the first 16 bytes, and the lowercase in the
        // last 16 bytes. This allows us to compare the uppercase and lowercase versions of
        // the needle char in the same comparison.
        loop {
            if unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(needle_char, haystack_chunk)) } == 0
            {
                // No match, advance to next chunk
                break;
            }

            // Progress to next needle char, if available
            if let Some(next_needle_char) = needle_iter.next() {
                needle_char = *next_needle_char;
            } else {
                return true;
            }
        }
    }

    false
}
