//! Kept for reference, but no longer used in the codebase due to poor performance without AVX512

use std::simd::{Select, Simd, cmp::SimdPartialOrd, num::SimdUint};

const LANES: usize = 8;

/// Converts a string to a u64 where each bit represents the existence of a character in the ASCII
/// range `[33, 90]`. To tell if two strings are likely to match, we perform a bitwise XOR between
/// the two bitmasks. The number of 1s in the resulting bitmask indicates the number of characters
/// in the needle which are not in the haystack and the number of characters in the haystack which
/// are not in the needle.
///
/// TODO: Only fast on AVX512
pub fn string_to_bitmask(s: &[u8]) -> u64 {
    let mut mask: u64 = 0;

    let zero = Simd::splat(0);
    let zero_wide = Simd::splat(0);
    let one = Simd::splat(1);
    let to_uppercase = Simd::splat(0x20);

    let mut i = 0;
    while i < s.len() {
        let simd_chunk = Simd::<u8, LANES>::load_or_default(&s[i..(i + LANES).min(s.len())]);

        // Convert to uppercase
        let is_lower =
            simd_chunk.simd_ge(Simd::splat(b'a')) & simd_chunk.simd_le(Simd::splat(b'z'));
        let simd_chunk = simd_chunk - is_lower.select(to_uppercase, zero);

        // Check if characters are in the valid range [33, 90]
        let in_range =
            simd_chunk.simd_ge(Simd::splat(32u8)) & simd_chunk.simd_le(Simd::splat(90u8));

        // Compute indices
        let indices = in_range.cast::<i64>().select(
            one << (simd_chunk - Simd::splat(32u8)).cast::<u64>(),
            zero_wide,
        );

        mask |= indices.reduce_or();

        i += LANES;
    }

    mask
}

pub fn string_to_bitmask_scalar(s: &[u8]) -> u64 {
    let mut mask: u64 = 0;

    for byte in s.iter() {
        if byte.is_ascii_lowercase() {
            mask |= 1u64 << (byte - 64);
        } else if (32..=90).contains(byte) {
            mask |= 1u64 << (byte - 32);
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::string_to_bitmask;

    #[test]
    fn test_case_insensitive() {
        assert_eq!(
            string_to_bitmask("ABC".as_bytes()),
            string_to_bitmask("abc".as_bytes())
        );
    }

    #[test]
    fn test_letters() {
        assert_eq!(
            string_to_bitmask("abC".as_bytes()),
            0b0000000000000000000000000000111000000000000000000000000000000000
        );
    }

    #[test]
    fn test_numbers() {
        assert_eq!(
            string_to_bitmask("123".as_bytes()),
            0b00000000000000000000000000000000000000000011100000000000000000
        );
    }

    #[test]
    fn test_symbols() {
        assert_eq!(
            string_to_bitmask("!\"#$%&'()*+,-./".as_bytes()),
            0b00000000000000000000000000000000000000000000001111111111111110
        );
    }
}
