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
pub unsafe fn _mm256_shift_left_epi16(v: __m256i) -> __m256i {
    // Permute: [low, high] -> [high, zeros]
    let shifted_lanes = _mm256_permute2x128_si256(v, v, 0x81);

    // alignr with 2 bytes shifts left by one u16
    // alignr(a, b, n) concatenates and shifts right, so we swap operand order
    _mm256_alignr_epi8(shifted_lanes, v, 2)
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
