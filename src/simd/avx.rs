use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy)]
pub struct AVXVector(pub __m256i);

impl super::Vector for AVXVector {
    fn is_available() -> bool {
        raw_cpuid::CpuId::new()
            .get_extended_feature_info()
            .is_some_and(|info| info.has_avx2())
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm256_setzero_si256()) }
    }

    #[inline(always)]
    unsafe fn splat_u8(value: u8) -> Self {
        unsafe { Self(_mm256_set1_epi8(value as i8)) }
    }

    #[inline(always)]
    unsafe fn splat_u16(value: u16) -> Self {
        unsafe { Self(_mm256_set1_epi16(value as i16)) }
    }

    #[inline(always)]
    unsafe fn eq_u8(self, other: Self) -> Self {
        unsafe { Self(_mm256_cmpeq_epi8(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn gt_u8(self, other: Self) -> Self {
        unsafe {
            let sign_bit = _mm256_set1_epi8(-128i8);
            let a_flipped = _mm256_xor_si256(self.0, sign_bit);
            let b_flipped = _mm256_xor_si256(other.0, sign_bit);
            Self(_mm256_cmpgt_epi8(a_flipped, b_flipped))
        }
    }

    #[inline(always)]
    unsafe fn lt_u8(self, other: Self) -> Self {
        unsafe {
            let sign_bit = _mm256_set1_epi8(-128i8);
            let a_flipped = _mm256_xor_si256(self.0, sign_bit);
            let b_flipped = _mm256_xor_si256(other.0, sign_bit);
            Self(_mm256_cmpgt_epi8(b_flipped, a_flipped))
        }
    }

    #[inline(always)]
    unsafe fn max_u16(self, other: Self) -> Self {
        unsafe { Self(_mm256_max_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn smax_u16(self) -> u16 {
        unsafe {
            let high = _mm256_extracti128_si256(self.0, 1);
            let low = _mm256_castsi256_si128(self.0);

            let max128 = _mm_max_epu16(low, high);
            let max64 = _mm_max_epu16(max128, _mm_srli_si128(max128, 8));
            let max32 = _mm_max_epu16(max64, _mm_srli_si128(max64, 4));
            let max16 = _mm_max_epu16(max32, _mm_srli_si128(max32, 2));

            _mm_extract_epi16(max16, 0) as u16
        }
    }

    #[inline(always)]
    unsafe fn add_u16(self, other: Self) -> Self {
        unsafe { Self(_mm256_add_epi16(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn subs_u16(self, other: Self) -> Self {
        unsafe { Self(_mm256_subs_epu16(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm256_and_si256(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        unsafe { Self(_mm256_or_si256(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }

    #[inline(always)]
    unsafe fn shift_right_padded_u16<const N: i32>(self, other: Self) -> Self {
        unsafe {
            assert!(N >= 0 && N <= 8);

            // Permute: [padding_low, padding_high] + [low, high] -> [padding_high, low]
            let shifted_lanes = _mm256_permute2x128_si256(other.0, self.0, 0x21);

            // alignr to shift right by 2 bytes within each 128-bit lane pair
            // alignr(b, combined, 2) does:
            //   low lane:  takes from [a_high : b_low] and shifts right by 2
            //   high lane: takes from [b_low : b_high] and shifts right by 2
            Self(match N {
                1 => _mm256_alignr_epi8(self.0, shifted_lanes, 14),
                2 => _mm256_alignr_epi8(self.0, shifted_lanes, 12),
                3 => _mm256_alignr_epi8(self.0, shifted_lanes, 10),
                4 => _mm256_alignr_epi8(self.0, shifted_lanes, 8),
                5 => _mm256_alignr_epi8(self.0, shifted_lanes, 6),
                6 => _mm256_alignr_epi8(self.0, shifted_lanes, 4),
                7 => _mm256_alignr_epi8(self.0, shifted_lanes, 2),
                8 => shifted_lanes,
                _ => unreachable!(),
            })
        }
    }

    #[cfg(test)]
    fn from_array(arr: [u8; 16]) -> Self {
        Self(unsafe {
            _mm256_broadcastsi128_si256(_mm_loadu_si128(arr.as_ptr() as *const __m128i))
        })
    }
    #[cfg(test)]
    fn to_array(self) -> [u8; 16] {
        let mut arr = [0u8; 32];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.0) };
        arr[0..16].try_into().unwrap()
    }
    #[cfg(test)]
    fn from_array_u16(arr: [u16; 8]) -> Self {
        Self(unsafe {
            _mm256_broadcastsi128_si256(_mm_loadu_si128(arr.as_ptr() as *const __m128i))
        })
    }
    #[cfg(test)]
    fn to_array_u16(self) -> [u16; 8] {
        let mut arr = [0u16; 16];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.0) };
        arr[0..8].try_into().unwrap()
    }
}

impl super::Vector256 for AVXVector {
    #[cfg(test)]
    fn from_array_256_u16(arr: [u16; 16]) -> Self {
        Self(unsafe { _mm256_loadu_si256(arr.as_ptr() as *const __m256i) })
    }
    #[cfg(test)]
    fn to_array_256_u16(self) -> [u16; 16] {
        let mut arr = [0u16; 16];
        unsafe { _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, self.0) };
        arr
    }

    #[inline(always)]
    unsafe fn load_unaligned(data: [u8; 32]) -> Self {
        Self(unsafe { _mm256_loadu_si256(data.as_ptr() as *const __m256i) })
    }

    #[inline(always)]
    unsafe fn idx_u16(self, search: u16) -> usize {
        unsafe {
            // compare all elements with max, get mask
            let cmp = _mm256_cmpeq_epi16(self.0, _mm256_set1_epi16(search as i16));
            let mask = _mm256_movemask_epi8(cmp) as u32;

            // find first set bit
            // divide by 2 to get element index (since u16 = 2 bytes)
            mask.trailing_zeros() as usize / 2
        }
    }
}
