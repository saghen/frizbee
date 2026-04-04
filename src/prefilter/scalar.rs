use super::case_needle;

#[inline(always)]
pub fn match_haystack(needle: &[(u8, u8)], haystack: &[u8]) -> bool {
    let mut haystack_idx = 0;
    for needle in needle.iter() {
        loop {
            if haystack_idx == haystack.len() {
                return false;
            }

            if needle.0 == haystack[haystack_idx] || needle.1 == haystack[haystack_idx] {
                haystack_idx += 1;
                break;
            }
            haystack_idx += 1;
        }
    }

    true
}

#[inline(always)]
pub fn match_haystack_typos(needle: &[(u8, u8)], haystack: &[u8], max_typos: u16) -> bool {
    let mut haystack_idx = 0;
    let mut typos = 0;
    for needle in needle.iter() {
        loop {
            if haystack_idx == haystack.len() {
                typos += 1;
                if typos > max_typos as usize {
                    return false;
                }

                haystack_idx = 0;
                break;
            }

            if needle.0 == haystack[haystack_idx] || needle.1 == haystack[haystack_idx] {
                haystack_idx += 1;
                break;
            }
            haystack_idx += 1;
        }
    }

    true
}

/// Replicates the SIMD prefilter behavior in scalar for unsupported platforms.
/// TODO: Replace this with an ordered prefilter following a memchr style approach
#[derive(Debug, Clone)]
pub struct PrefilterScalar {
    needle: Vec<(u8, u8)>,
}

impl PrefilterScalar {
    pub fn new(needle: &[u8]) -> Self {
        Self {
            needle: case_needle(needle),
        }
    }

    /// Determines the byte range to check for a given chunk, replicating SIMD
    /// overlapping load behavior where when reaching the last chunk, the last
    /// 16 bytes are loaded (overlapping with the previous chunk).
    #[inline(always)]
    fn overlapping_load(start: usize, len: usize) -> (usize, usize) {
        if len <= 16 {
            (0, len)
        } else if start + 16 <= len {
            (start, start + 16)
        } else {
            (len - 16, len)
        }
    }

    /// Unordered prefilter matching SIMD prefilter behavior: checks each needle
    /// char exists somewhere in the current 16-byte chunk (unordered, with
    /// replacement). Matches across chunks must appear in increasing chunk order.
    pub fn match_haystack(&self, haystack: &[u8]) -> (bool, usize) {
        let len = haystack.len();
        match len {
            0 => return (true, 0),
            1..=7 => return (match_haystack(&self.needle, haystack), 0),
            _ => {}
        }

        let mut can_skip_chunks = true;
        let mut skipped_chunks = 0;
        let mut needle_idx = 0;

        for start in (0..len).step_by(16) {
            let (chunk_start, chunk_end) = PrefilterScalar::overlapping_load(start, len);
            let chunk = &haystack[chunk_start..chunk_end];

            loop {
                if needle_idx >= self.needle.len() {
                    return (true, skipped_chunks);
                }

                let (c1, c2) = self.needle[needle_idx];
                if chunk.iter().any(|&b| b == c1 || b == c2) {
                    if can_skip_chunks {
                        skipped_chunks = start / 16;
                    }
                    can_skip_chunks = false;
                    needle_idx += 1;
                } else {
                    break;
                }
            }
        }

        (needle_idx >= self.needle.len(), skipped_chunks)
    }

    /// Unordered prefilter with typo support, matching SIMD prefilter behavior.
    pub fn match_haystack_typos(&self, haystack: &[u8], max_typos: u16) -> (bool, usize) {
        let len = haystack.len();
        match len {
            0 => return (true, 0),
            1..=7 => return (match_haystack_typos(&self.needle, haystack, max_typos), 0),
            _ => {}
        }

        if max_typos >= 3 {
            return (true, 0);
        }

        let mut needle_idx = 0;
        let mut typos = 0;

        loop {
            for start in (0..len).step_by(16) {
                let (chunk_start, chunk_end) = PrefilterScalar::overlapping_load(start, len);
                let chunk = &haystack[chunk_start..chunk_end];

                loop {
                    if needle_idx >= self.needle.len() {
                        return (true, 0);
                    }

                    let (c1, c2) = self.needle[needle_idx];
                    if chunk.iter().any(|&b| b == c1 || b == c2) {
                        needle_idx += 1;
                    } else {
                        break;
                    }
                }
            }

            typos += 1;
            if typos > max_typos as usize {
                return (false, 0);
            }

            needle_idx += 1;
            if needle_idx >= self.needle.len() {
                return (true, 0);
            }
        }
    }
}
