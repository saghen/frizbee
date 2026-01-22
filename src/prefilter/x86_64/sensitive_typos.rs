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
pub unsafe fn match_haystack_unordered_typos(
    needle: &[u8],
    haystack: &[u8],
    max_typos: u16,
) -> (bool, usize) {
    let len = haystack.len();

    // TODO: skipped chunks calculation

    let mut needle_iter = needle.iter().map(|&c| unsafe { _mm_set1_epi8(c as i8) });
    let mut needle_char = needle_iter.next().unwrap();

    let mut typos = 0;
    loop {
        // TODO: this is slightly incorrect, because if we match on the third chunk,
        // we would only scan from the third chunk onwards for the next needle. Technically,
        // we should scan from the beginning of the haystack instead, but I believe the
        // previous memchr implementation had the same bug.
        for start in (0..len).step_by(16) {
            let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

            loop {
                // Compare each byte (0xFF if equal, 0x00 if not)
                let cmp = unsafe { _mm_cmpeq_epi8(needle_char, haystack_chunk) };
                // No match, advance to next chunk
                if unsafe { _mm_movemask_epi8(cmp) } == 0 {
                    break;
                }

                // Progress to next needle char, if available
                if let Some(next_needle_char) = needle_iter.next() {
                    needle_char = next_needle_char;
                } else {
                    return (true, 0);
                }
            }
        }

        typos += 1;
        if typos > max_typos as usize {
            return (false, 0);
        }

        if let Some(next_needle_char) = needle_iter.next() {
            needle_char = next_needle_char;
        } else {
            return (true, 0);
        }
    }
}
