use std::arch::x86_64::*;

#[inline(always)]
pub unsafe fn _mm256_cmpneq_epi16(a: __m256i, b: __m256i) -> __m256i {
    let eq = _mm256_cmpeq_epi16(a, b);
    _mm256_xor_si256(eq, _mm256_set1_epi16(-1)) // not
}

#[inline(always)]
pub unsafe fn argmax_epu16(v: __m256i) -> usize {
    // Step 1: Find the maximum value using horizontal reduction
    // Compare high and low 128-bit lanes
    let high = _mm256_extracti128_si256(v, 1);
    let low = _mm256_castsi256_si128(v);
    let max128 = _mm_max_epu16(low, high);

    // Reduce 128-bit to find max (shift and compare)
    let max64 = _mm_max_epu16(max128, _mm_srli_si128(max128, 8));
    let max32 = _mm_max_epu16(max64, _mm_srli_si128(max64, 4));
    let max16 = _mm_max_epu16(max32, _mm_srli_si128(max32, 2));

    // Broadcast max value to all lanes
    let max_val = _mm256_set1_epi16(_mm_extract_epi16(max16, 0) as i16);

    // Step 2: Compare all elements with max, get mask
    let cmp = _mm256_cmpeq_epi16(v, max_val);
    let mask = _mm256_movemask_epi8(cmp) as u32;

    // Step 3: Find first set bit (each u16 produces 2 bits in mask)
    // Divide by 2 to get element index
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
        let result = unsafe { argmax_epu16(v) };
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
}
