use super::overlapping_load;
use std::arch::x86_64::*;

/// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
/// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
/// the order does matter across 16 byte boundaries. The needle chars must include both the
/// uppercase and lowercase variants of the character.
///
/// # Safety
/// The caller must ensure that the minimum length of the haystack is >= 8.
/// The caller must ensure the needle.len() > 0 and that SSE2 is available.
#[inline(always)]
pub unsafe fn match_haystack_unordered_insensitive(
    needle: &[__m256i],
    haystack: &[u8],
) -> (bool, usize) {
    let len = haystack.len();

    let mut skipped_chunks = 0;
    let mut can_skip_chunks = true;

    let mut needle_iter = needle.iter();
    let mut needle_char = *needle_iter.next().unwrap();

    for start in (0..len).step_by(16) {
        let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };
        let haystack_chunk = unsafe { _mm256_broadcastsi128_si256(haystack_chunk) };
        loop {
            if unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(needle_char, haystack_chunk)) } == 0
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

    (false, skipped_chunks)
}
