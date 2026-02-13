use std::arch::aarch64::*;

use crate::prefilter::{case_needle, scalar};

/// Loads a chunk of 16 bytes from the haystack, with overlap when remaining bytes < 16,
/// since it's dramatically faster than a memcpy.
///
/// If the haystack the number of remaining bytes is < 16, and the total length is > 16,
/// the last 16 bytes are loaded from the end of the haystack.
///
/// If the haystack is < 16 bytes, we load the first 8 bytes from the haystack, and the last 8
/// bytes, and combine them into a single vector.
///
/// # Safety
/// Caller must ensure that haystack length >= 8
#[inline(always)]
unsafe fn overlapping_load(haystack: &[u8], start: usize, len: usize) -> uint8x16_t {
    unsafe {
        match len {
            0..=7 => unreachable!(),
            8 => {
                // Load 8 bytes and zero-extend to 128-bit vector
                let low = vld1_u8(haystack.as_ptr());
                vcombine_u8(low, vdup_n_u8(0))
            }
            // Loads 8 bytes from the start of the haystack, and 8 bytes from the end of the haystack
            // and combines them into a single vector. Much faster than a memcpy
            9..=15 => {
                let low = vld1_u8(haystack.as_ptr());
                let high_start = len - 8;
                let high = vld1_u8(haystack[high_start..].as_ptr());
                vcombine_u8(low, high)
            }
            16 => vld1q_u8(haystack.as_ptr()),
            // Avoid reading past the end, instead re-read the last 16 bytes
            _ => vld1q_u8(haystack[start.min(len - 16)..].as_ptr()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefilterNEON {
    needle: Vec<(u8, u8)>,
}

impl PrefilterNEON {
    #[inline]
    pub fn new(needle: &[u8]) -> Self {
        Self {
            needle: case_needle(needle),
        }
    }

    /// Checks if the needle is wholly contained in the haystack, ignoring the exact order of the
    /// bytes. For example, if the needle is "test", the haystack "tset" will return true. However,
    /// the order does matter across 16 byte boundaries. The needle chars must include both the
    /// uppercase and lowercase variants of the character.
    ///
    /// # Safety
    /// The caller must ensure that NEON is available.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn match_haystack(&self, haystack: &[u8]) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (scalar::match_haystack(&self.needle, haystack), 0);
            }
            _ => {}
        };

        let mut can_skip_chunks = true;
        let mut skipped_chunks = 0;

        let mut needle_iter = self
            .needle
            .iter()
            .map(|&(c1, c2)| (vdupq_n_u8(c1), vdupq_n_u8(c2)));
        let mut needle_char = needle_iter.next().unwrap();

        for start in (0..len).step_by(16) {
            let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

            loop {
                let mask = vmaxvq_u8(vorrq_u8(
                    vceqq_u8(needle_char.1, haystack_chunk),
                    vceqq_u8(needle_char.0, haystack_chunk),
                ));
                if mask == 0 {
                    // No match, advance to next chunk
                    break;
                }

                // Progress to next needle char, if available
                if let Some(next_needle_char) = needle_iter.next() {
                    if can_skip_chunks {
                        skipped_chunks = start / 16;
                    }
                    can_skip_chunks = false;
                    needle_char = next_needle_char;
                } else {
                    return (true, skipped_chunks);
                }
            }
        }

        (false, skipped_chunks)
    }

    /// # Safety
    /// The caller must ensure that the minimum length of the haystack is >= 8.
    /// The caller must ensure the needle.len() > 0 and that SSE2 is available.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn match_haystack_typos(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        let len = haystack.len();

        match len {
            0 => return (true, 0),
            1..=7 => {
                return (
                    scalar::match_haystack_typos(&self.needle, haystack, max_typos),
                    0,
                );
            }
            _ => {}
        };

        let mut needle_iter = self
            .needle
            .iter()
            .map(|&(c1, c2)| (vdupq_n_u8(c1), vdupq_n_u8(c2)));
        let mut needle_char = needle_iter.next().unwrap();

        let mut typos = 0;
        loop {
            let mut skipped_chunks = 0;
            let mut can_skip_chunks = true;

            // TODO: this is slightly incorrect, because if we match on the third chunk,
            // we would only scan from the third chunk onwards for the next needle. Technically,
            // we should scan from the beginning of the haystack instead, but I believe the
            // previous memchr implementation had the same bug.
            for start in (0..len).step_by(16) {
                let haystack_chunk = unsafe { overlapping_load(haystack, start, len) };

                loop {
                    let mask = vmaxvq_u8(vorrq_u8(
                        vceqq_u8(needle_char.1, haystack_chunk),
                        vceqq_u8(needle_char.0, haystack_chunk),
                    ));
                    if mask == 0 {
                        // No match, advance to next chunk
                        break;
                    }

                    // Progress to next needle char, if available
                    if let Some(next_needle_char) = needle_iter.next() {
                        if can_skip_chunks {
                            skipped_chunks = start / 16;
                        }
                        can_skip_chunks = false;

                        needle_char = next_needle_char;
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
                needle_char = next_needle_char;
            } else {
                return (true, skipped_chunks);
            }
        }
    }
}
