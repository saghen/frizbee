use super::overlapping_load;
use std::arch::x86_64::*;

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
pub unsafe fn match_haystack_unordered_insensitive_typos(
    needle: &[__m256i],
    haystack: &[u8],
    max_typos: u16,
) -> (bool, usize) {
    let len = haystack.len();

    let mut needle_iter = needle.iter();
    let mut needle_char = *needle_iter.next().unwrap();

    let mut typos = 0;
    loop {
        let mut skipped_chunks = 0;
        let mut can_skip_chunks = true;

        if typos > max_typos as usize {
            return (false, 0);
        }

        // TODO: this is slightly incorrect, because if we match on the third chunk,
        // we would only scan from the third chunk onwards for the next needle. Technically,
        // we should scan from the beginning of the haystack instead, but I believe the
        // previous memchr implementation had the same bug.
        for start in (0..len).step_by(16) {
            let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };
            let haystack_chunk = unsafe { _mm256_broadcastsi128_si256(haystack_chunk) };

            // For AVX2, we store the uppercase in the first 16 bytes, and the lowercase in the
            // last 16 bytes. This allows us to compare the uppercase and lowercase versions of
            // the needle char in the same comparison.
            loop {
                if unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(needle_char, haystack_chunk)) }
                    == 0
                {
                    // No match, advance to next chunk
                    break;
                }

                // Progress to next needle char, if available
                if let Some(next_needle_char) = needle_iter.next() {
                    if can_skip_chunks {
                        skipped_chunks = start / 16;
                    }
                    can_skip_chunks = false;

                    needle_char = *next_needle_char;
                } else {
                    return (true, skipped_chunks);
                }
            }
        }

        typos += 1;
        if typos > max_typos as usize {
            return (false, 0);
        }

        if let Some(next_needle_char) = needle_iter.next() {
            needle_char = *next_needle_char;
        } else {
            return (true, 0);
        }
    }
}
