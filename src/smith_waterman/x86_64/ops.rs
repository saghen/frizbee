use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn get_prefix_mask() -> __m256i {
    _mm256_setr_epi8(
        -1, -1, // -1 = 0xFF as i8
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
}

#[inline(always)]
unsafe fn load_partial_safe(ptr: *const u8, len: usize) -> __m128i {
    debug_assert!(len < 8);

    let val: u64 = match len {
        0 => 0,
        1 => *ptr as u64,
        2 => (ptr as *const u16).read_unaligned() as u64,
        3 => {
            let lo = (ptr as *const u16).read_unaligned() as u64;
            let hi = *ptr.add(2) as u64;
            lo | (hi << 16)
        }
        4 => (ptr as *const u32).read_unaligned() as u64,
        5 => {
            let lo = (ptr as *const u32).read_unaligned() as u64;
            let hi = *ptr.add(4) as u64;
            lo | (hi << 32)
        }
        6 => {
            let lo = (ptr as *const u32).read_unaligned() as u64;
            let hi = (ptr.add(4) as *const u16).read_unaligned() as u64;
            lo | (hi << 32)
        }
        7 => {
            let lo = (ptr as *const u32).read_unaligned() as u64;
            let hi = (ptr.add(4) as *const u32).read_unaligned() as u64;
            // Mask hi to only 3 bytes (24 bits)
            lo | ((hi & 0xFFFFFF) << 32)
        }
        _ => std::hint::unreachable_unchecked(),
    };

    _mm_cvtsi64_si128(val as i64)
}

#[inline(always)]
fn can_overread_8(ptr: *const u8) -> bool {
    // Safe if not in last 7 bytes of a page
    (ptr as usize & 0xFFF) <= (4096 - 8)
}

pub unsafe fn _mm_loadu(haystack: &[u8], start: usize, len: usize) -> __m128i {
    unsafe {
        match len {
            0 => _mm_setzero_si128(),
            8 => _mm_loadl_epi64(haystack.as_ptr() as *const __m128i),
            16 => _mm_loadu_si128(haystack.as_ptr() as *const __m128i),

            1..=7 if can_overread_8(haystack.as_ptr()) => {
                let low = _mm_loadl_epi64(haystack.as_ptr() as *const __m128i);
                let mask = _mm_set_epi64x(0, (1i64 << (len * 8)) - 1);
                _mm_and_si128(low, mask)
            }
            1..=7 => load_partial_safe(haystack.as_ptr(), len),
            9..=15 => {
                let lo = _mm_loadl_epi64(haystack.as_ptr() as *const __m128i);

                let high_start = len - 8;
                let high = _mm_loadl_epi64(haystack[high_start..].as_ptr() as *const __m128i);
                let mask = _mm_set_epi64x(0, (1i64 << ((len - 8) * 8)) - 1);
                let high = _mm_and_si128(high, mask);

                _mm_unpacklo_epi64(lo, high)
            }

            _ if start + 16 <= len => _mm_loadu_si128(haystack[start..].as_ptr() as *const __m128i),
            _ => {
                let overlap = start + 16 - len; // bytes we need to shift out
                let data = _mm_loadu_si128(haystack[len - 16..].as_ptr() as *const __m128i);

                // Shift left by 'overlap' bytes to align data to start position
                // This zeros out the rightmost 'overlap' bytes and shifts content left
                match overlap {
                    1 => _mm_slli_si128(data, 1),
                    2 => _mm_slli_si128(data, 2),
                    3 => _mm_slli_si128(data, 3),
                    4 => _mm_slli_si128(data, 4),
                    5 => _mm_slli_si128(data, 5),
                    6 => _mm_slli_si128(data, 6),
                    7 => _mm_slli_si128(data, 7),
                    8 => _mm_slli_si128(data, 8),
                    9 => _mm_slli_si128(data, 9),
                    10 => _mm_slli_si128(data, 10),
                    11 => _mm_slli_si128(data, 11),
                    12 => _mm_slli_si128(data, 12),
                    13 => _mm_slli_si128(data, 13),
                    14 => _mm_slli_si128(data, 14),
                    15 => _mm_slli_si128(data, 15),
                    _ => _mm_setzero_si128(),
                }
            }
        }
    }
}

#[inline(always)]
pub unsafe fn _mm256_not_epi16(v: __m256i) -> __m256i {
    _mm256_xor_si256(v, _mm256_set1_epi16(-1))
}

#[inline(always)]
pub unsafe fn _mm256_cmpneq_epi16(a: __m256i, b: __m256i) -> __m256i {
    let eq = _mm256_cmpeq_epi16(a, b);
    _mm256_xor_si256(eq, _mm256_set1_epi16(-1)) // not
}

/// Returns the maximum value in the vector as a scalar
#[inline(always)]
pub unsafe fn _mm256_smax_epu16(v: __m256i) -> u16 {
    let high = _mm256_extracti128_si256(v, 1);
    let low = _mm256_castsi256_si128(v);

    let max128 = _mm_max_epu16(low, high);
    let max64 = _mm_max_epu16(max128, _mm_srli_si128(max128, 8));
    let max32 = _mm_max_epu16(max64, _mm_srli_si128(max64, 4));
    let max16 = _mm_max_epu16(max32, _mm_srli_si128(max32, 2));

    _mm_extract_epi16(max16, 0) as u16
}

/// Returns the index of the value in the vector.
/// Return `8` if the element was not found
#[inline(always)]
pub unsafe fn _mm256_idx_epu16(v: __m256i, search: u16) -> usize {
    // compare all elements with max, get mask
    let cmp = _mm256_cmpeq_epi16(v, _mm256_set1_epi16(search as i16));
    let mask = _mm256_movemask_epi8(cmp) as u32;

    // find first set bit
    // divide by 2 to get element index (since u16 = 2 bytes)
    mask.trailing_zeros() as usize / 2
}

#[inline(always)]
pub unsafe fn _mm256_shift_right_padded_epi16(v: __m256i, padding: __m256i) -> __m256i {
    // Permute: [padding_low, padding_high] + [low, high] -> [padding_high, low]
    let shifted_lanes = _mm256_permute2x128_si256(padding, v, 0x21);

    // alignr to shift right by 2 bytes within each 128-bit lane pair
    // alignr(b, combined, 2) does:
    //   low lane:  takes from [a_high : b_low] and shifts right by 2
    //   high lane: takes from [b_low : b_high] and shifts right by 2
    _mm256_alignr_epi8(v, shifted_lanes, 14)
}

#[inline(always)]
pub unsafe fn _mm256_shift_right_two_padded_epi16(v: __m256i, padding: __m256i) -> __m256i {
    // Permute: [padding_low, padding_high] + [low, high] -> [padding_high, low]
    let shifted_lanes = _mm256_permute2x128_si256(padding, v, 0x21);

    // alignr to shift right by 4 bytes within each 128-bit lane pair
    // alignr(b, combined, 4) does:
    //   low lane:  takes from [a_high : b_low] and shifts right by 4
    //   high lane: takes from [b_low : b_high] and shifts right by 4
    _mm256_alignr_epi8(v, shifted_lanes, 12)
}

#[inline(always)]
pub unsafe fn _mm256_shift_right_four_padded_epi16(v: __m256i, padding: __m256i) -> __m256i {
    // Permute: [padding_low, padding_high] + [low, high] -> [padding_high, low]
    let shifted_lanes = _mm256_permute2x128_si256(padding, v, 0x21);

    // alignr to shift right by 8 bytes within each 128-bit lane pair
    // alignr(b, combined, 8) does:
    //   low lane:  takes from [a_high : b_low] and shifts right by 8
    //   high lane: takes from [b_low : b_high] and shifts right by 8
    _mm256_alignr_epi8(v, shifted_lanes, 8)
}

#[inline(always)]
pub unsafe fn _mm256_shift_right_epi16(v: __m256i) -> __m256i {
    // Permute: [low, high] -> [zeros, low]
    let shifted_lanes = _mm256_permute2x128_si256(v, v, 0x08);

    // alignr shifts within 128-bit lanes
    // We need to shift by 2 bytes (one u16) to the right
    // alignr(a, b, n) = (a:b) >> (n*8) for each 128-bit lane
    _mm256_alignr_epi8(v, shifted_lanes, 14)
}

#[inline(always)]
pub unsafe fn _mm256_shift_right_two_epi16(v: __m256i) -> __m256i {
    // Permute: [low, high] -> [zeros, low]
    let shifted_lanes = _mm256_permute2x128_si256(v, v, 0x08);

    // alignr shifts within 128-bit lanes
    // We need to shift by 2 bytes (one u16) to the right
    // alignr(a, b, n) = (a:b) >> (n*8) for each 128-bit lane
    _mm256_alignr_epi8(v, shifted_lanes, 12)
}

#[inline(always)]
pub unsafe fn _mm256_shift_right_four_epi16(v: __m256i) -> __m256i {
    // Permute: [low, high] -> [zeros, low]
    let shifted_lanes = _mm256_permute2x128_si256(v, v, 0x08);

    // alignr shifts within 128-bit lanes
    // We need to shift by 2 bytes (one u16) to the right
    // alignr(a, b, n) = (a:b) >> (n*8) for each 128-bit lane
    _mm256_alignr_epi8(v, shifted_lanes, 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_epu16() {
        let v = unsafe { _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 11, 12, 13, 14, 15) };
        let result = unsafe { _mm256_idx_epu16(v, _mm256_smax_epu16(v)) };
        assert_eq!(result, 10);
    }

    #[test]
    fn test_shift_right_padded_epi16() {
        let v =
            unsafe { _mm256_setr_epi16(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) };
        let padding = unsafe { _mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) };
        let result = unsafe { _mm256_shift_right_two_padded_epi16(v, padding) };
        let mut result_arr: [i16; 16] = [0; 16];
        unsafe { _mm256_storeu_si256(result_arr.as_mut_ptr() as *mut __m256i, result) };
        assert_eq!(
            result_arr,
            [1, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
    }

    #[test]
    fn test_overlapping_load_7_bytes() {
        let value = [10, 9, 8, 7, 6, 5, 4];
        assert!(can_overread_8(value.as_ptr()));

        let vector = unsafe { _mm_loadu(&value, 0, value.len()) };
        let mut data = [0u8; 16];
        unsafe {
            _mm_storeu_si128(data.as_mut_ptr() as *mut __m128i, vector);
        }
        assert_eq!(data[0..7], [10, 9, 8, 7, 6, 5, 4]);
    }

    #[test]
    fn test_overlapping_load_7_bytes_safe() {
        let value = [10, 9, 8, 7, 6, 5, 4];
        let vector = unsafe { load_partial_safe(value.as_ptr(), value.len()) };
        let mut data = [0u8; 16];
        unsafe {
            _mm_storeu_si128(data.as_mut_ptr() as *mut __m128i, vector);
        }
        assert_eq!(data[0..7], [10, 9, 8, 7, 6, 5, 4]);
    }
}
