use super::overlapping_load;
use std::arch::x86_64::*;

/// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
/// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
/// the order does matter across 16 byte boundaries.
///
/// # Safety
/// The caller must ensure that the minimum length of the haystack is >= 8.
/// The caller must ensure the needle.len() > 0 and that SSE2 is available.
#[inline(always)]
pub unsafe fn match_haystack_unordered(needle: &[u8], haystack: &[u8]) -> (bool, usize) {
    let len = haystack.len();

    let mut can_skip_chunks = true;
    let mut skipped_chunks = 0;

    let mut needle_iter = needle.iter().map(|&c| unsafe { _mm_set1_epi8(c as i8) });
    let mut needle_char = needle_iter.next().unwrap();

    for start in (0..len).step_by(16) {
        let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

        loop {
            // No match, advance to next chunk
            if unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(needle_char, haystack_chunk)) } == 0 {
                if can_skip_chunks {
                    skipped_chunks += 1;
                }
                break;
            }

            // Progress to next needle char, if available
            if let Some(next_needle_char) = needle_iter.next() {
                can_skip_chunks = false;
                needle_char = next_needle_char;
            } else {
                return (true, skipped_chunks);
            }
        }
    }

    (false, skipped_chunks)
}
