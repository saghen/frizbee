use core::arch::x86_64::*;

use crate::prefilter::algo::can_overread;

use super::{Backend, BytesVec, MaskVec, ScoreVec, keep_first};

/// Loads the first `n` (1..64) bytes at `ptr`, zeroing the rest: a masked load
/// (which stalls on cache misses) only when a full one would cross a page. The
/// keep mask comes from [`keep_first`] so LLVM can't fold it into a masked load
#[inline(always)]
unsafe fn load_first(ptr: *const u8, n: usize) -> __m512i {
    unsafe {
        if can_overread(ptr, 64) {
            let keep = keep_first(n);
            _mm512_and_si512(
                _mm512_loadu_si512(ptr as *const __m512i),
                _mm512_loadu_si512(keep as *const __m512i),
            )
        } else {
            _mm512_maskz_loadu_epi8((1u64 << n) - 1, ptr as *const i8)
        }
    }
}

/// [`load_first`] for `n` in 1..=32 bytes, in the low half of the vector
#[inline(always)]
unsafe fn load_first_256(ptr: *const u8, n: usize) -> __m512i {
    unsafe {
        if can_overread(ptr, 32) {
            let keep = keep_first(n);
            _mm512_zextsi256_si512(_mm256_and_si256(
                _mm256_loadu_si256(ptr as *const __m256i),
                _mm256_loadu_si256(keep as *const __m256i),
            ))
        } else {
            _mm512_maskz_loadu_epi8((1u64 << n) - 1, ptr as *const i8)
        }
    }
}

/// 32-lane u16 scoring (512-bit __m512i), 32-lane u8 input (low half of
/// __m512i).
#[derive(Debug, Clone, Copy)]
pub struct BackendAVX512;

#[derive(Debug, Clone, Copy)]
pub struct Avx512Bytes(__m512i);

/// Native AVX-512 32-bit predicate mask. Bit `i` corresponds to lane `i`
/// (i.e. lane 0 is the LSB).
#[derive(Debug, Clone, Copy)]
pub struct Avx512Mask(__mmask32);

#[derive(Debug, Clone, Copy)]
pub struct Avx512Score(__m512i);

/// 64-lane u8 scoring (512-bit __m512i), 64-lane u8 input (512-bit __m512i).
#[derive(Debug, Clone, Copy)]
pub struct BackendAVX512U8;

#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Bytes(__m512i);

/// Native AVX-512 64-bit predicate mask. Bit `i` corresponds to lane `i`
/// (i.e. lane 0 is the LSB).
#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Mask(__mmask64);

#[derive(Debug, Clone, Copy)]
pub struct Avx512U8Score(__m512i);

impl Backend for BackendAVX512 {
    const LANES: usize = 32;
    const LANE_BYTES: usize = 2;
    type Bytes = Avx512Bytes;
    type Mask = Avx512Mask;
    type Score = Avx512Score;

    fn is_available() -> bool {
        let features = crate::cpuid::detect();
        features.avx512f && features.avx512bw
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        // movm_epi16 spreads each mask bit to a full word (0xFFFF / 0x0000).
        unsafe { Avx512Score(_mm512_movm_epi16(m.0)) }
    }

    gap_dispatch!(32);
}

impl Backend for BackendAVX512U8 {
    const LANES: usize = 64;
    const LANE_BYTES: usize = 1;
    type Bytes = Avx512U8Bytes;
    type Mask = Avx512U8Mask;
    type Score = Avx512U8Score;

    fn is_available() -> bool {
        let features = crate::cpuid::detect();
        features.avx512f && features.avx512bw && features.avx512vbmi
    }

    #[inline(always)]
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score {
        // movm_epi8 spreads each mask bit to a full byte (0xFF / 0x00).
        unsafe { Avx512U8Score(_mm512_movm_epi8(m.0)) }
    }

    gap_dispatch!(64);

    const ASCII_SCAN_RAMP: usize = 63;

    /// The ramp `gap_extend * j`, then `gap_extend` (the previous chunk's
    /// step)
    #[inline(always)]
    unsafe fn ascii_gap_steps(gap_extend: u16) -> [Self::Score; 10] {
        let mut ramp = [0u8; 64];
        for (j, lane) in ramp.iter_mut().enumerate() {
            *lane = gap_extend.saturating_mul(j as u16).min(255) as u8;
        }
        unsafe {
            let mut steps = [Avx512U8Score::splat(gap_extend); 10];
            steps[0] = Avx512U8Score(_mm512_loadu_si512(ramp.as_ptr() as *const __m512i));
            steps
        }
    }

    /// The scan decays by `gap_extend` per lane, so it's the plain prefix max
    /// of `v` plus the ramp `gap_extend * j`, minus the ramp (see
    /// `u8_scan_ramp_fits`): within each 128-bit segment, then from the last
    /// lane of every earlier segment and of the previous chunk
    #[inline(always)]
    unsafe fn ascii_scan(
        v: Self::Score,
        prev: Self::Score,
        steps: &[Self::Score; 10],
        scores: Self::Score,
    ) -> (Self::Score, Self::Score) {
        unsafe {
            let ramp = steps[0];
            let carry = Avx512U8Score(_mm512_permutexvar_epi8(_mm512_set1_epi8(63), prev.0));
            let carry = carry.subs(steps[1]);
            let v = v.add(ramp);
            let v = v
                .max(carry)
                .max(in_segment::<1>(v))
                .max(in_segment::<2>(v).max(in_segment::<3>(v)));
            let v = v
                .max(in_segment::<4>(v))
                .max(in_segment::<8>(v).max(in_segment::<12>(v)));
            let v = v
                .max(segment_end(v, 0))
                .max(segment_end(v, 1).max(segment_end(v, 2)));
            let scan = v.subs(ramp);
            (scan, scan.max(scores))
        }
    }
}

impl BytesVec for Avx512Bytes {
    type Mask = Avx512Mask;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(_mm512_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        unsafe { Avx512Mask((_mm512_cmpeq_epi8_mask(self.0, other.0) as u32) as __mmask32) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        unsafe { Avx512Mask((_mm512_cmpgt_epu8_mask(self.0, other.0) as u32) as __mmask32) }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe { Avx512Mask((_mm512_cmplt_epu8_mask(self.0, other.0) as u32) as __mmask32) }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start).min(32);
            if remaining == 0 {
                return Self(_mm512_setzero_si512());
            }
            Self(load_first_256(data.add(start), remaining))
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 32);
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(values);
        Self(unsafe { _mm512_loadu_si512(buf.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 64];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf[..32].to_vec()
    }
}

impl BytesVec for Avx512U8Bytes {
    type Mask = Avx512U8Mask;

    #[inline(always)]
    unsafe fn splat(value: u8) -> Self {
        unsafe { Self(_mm512_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn eq(self, other: Self) -> Self::Mask {
        unsafe { Avx512U8Mask(_mm512_cmpeq_epi8_mask(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn gt(self, other: Self) -> Self::Mask {
        unsafe { Avx512U8Mask(_mm512_cmpgt_epu8_mask(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn lt(self, other: Self) -> Self::Mask {
        unsafe { Avx512U8Mask(_mm512_cmplt_epu8_mask(self.0, other.0)) }
    }

    #[inline(always)]
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self {
        unsafe {
            let remaining = len.saturating_sub(start);
            if remaining == 0 {
                return Self(_mm512_setzero_si512());
            }
            let ptr = data.add(start);
            Self(match remaining {
                0 => unreachable!(),
                1..64 => load_first(ptr, remaining),
                64.. => _mm512_loadu_si512(ptr as *const __m512i),
            })
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self {
        assert_eq!(values.len(), 64);
        Self(unsafe { _mm512_loadu_si512(values.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8> {
        let mut buf = [0u8; 64];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf.to_vec()
    }
}

impl MaskVec for Avx512Mask {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(0)
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(!self.0)
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // Lane i = bit i. shift_right_padded_1 places prev's highest lane
        // (bit 31) into lane 0 of the result, and shifts every other lane up
        // by one (lane i -> lane i+1).
        Self((self.0 << 1) | (prev.0 >> 31))
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 32);
        let mut m: u32 = 0;
        for (i, &v) in values.iter().enumerate() {
            if v {
                m |= 1u32 << i;
            }
        }
        Self(m)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        (0..32).map(|i| (self.0 >> i) & 1 == 1).collect()
    }
}

impl MaskVec for Avx512U8Mask {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self(0)
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
    #[inline(always)]
    unsafe fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(!self.0)
    }
    #[inline(always)]
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self {
        // Lane i = bit i. shift_right_padded_1 places prev's highest lane
        // (bit 63) into lane 0 of the result, and shifts every other lane up
        // by one (lane i -> lane i+1). In bit terms:
        //   new = (self << 1) | (prev >> 63)
        Self((self.0 << 1) | (prev.0 >> 63))
    }

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self {
        assert_eq!(values.len(), 64);
        let mut m: u64 = 0;
        for (i, &v) in values.iter().enumerate() {
            if v {
                m |= 1u64 << i;
            }
        }
        Self(m)
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool> {
        (0..64).map(|i| (self.0 >> i) & 1 == 1).collect()
    }
}

impl ScoreVec for Avx512Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm512_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe {
            let lo = _mm_cvtsi32_si128(value as i32);
            Self(_mm512_zextsi128_si512(lo))
        }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm512_max_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            // The quarters are extracted in parallel, a shorter chain than
            // halving twice
            let (q0, q1, q2, q3) = quarters(self.0);
            let m = _mm_max_epu16(_mm_max_epu16(q0, q1), _mm_max_epu16(q2, q3));
            super::sse::horizontal_max_epu16(m)
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm512_adds_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm512_subs_epu16(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe { Self(shift_right_u16_lanes::<L>(prev.0, self.0)) }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let target = _mm512_set1_epi16(search as i16);
            let mask = _mm512_cmpeq_epi16_mask(self.0, target);
            (mask.trailing_zeros() as usize).min(32)
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 32);
        Self(unsafe { _mm512_loadu_si512(values.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u16; 32];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf.to_vec()
    }
}

impl ScoreVec for Avx512U8Score {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(_mm512_setzero_si512()) }
    }
    #[inline(always)]
    unsafe fn splat(value: u16) -> Self {
        unsafe { Self(_mm512_set1_epi8(value as i8)) }
    }
    #[inline(always)]
    unsafe fn first_lane(value: u16) -> Self {
        unsafe {
            let lo = _mm_cvtsi32_si128((value & 0xFF) as i32);
            Self(_mm512_zextsi128_si512(lo))
        }
    }
    #[inline(always)]
    unsafe fn max(self, other: Self) -> Self {
        unsafe { Self(_mm512_max_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn horizontal_max(self) -> u16 {
        unsafe {
            let (q0, q1, q2, q3) = quarters(self.0);
            let m = _mm_max_epu8(_mm_max_epu8(q0, q1), _mm_max_epu8(q2, q3));
            super::sse::horizontal_max_epu8(m)
        }
    }
    #[inline(always)]
    unsafe fn add(self, other: Self) -> Self {
        unsafe { Self(_mm512_adds_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn subs(self, other: Self) -> Self {
        unsafe { Self(_mm512_subs_epu8(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn and(self, other: Self) -> Self {
        unsafe { Self(_mm512_and_si512(self.0, other.0)) }
    }
    #[inline(always)]
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self {
        unsafe { Self(shift_right_lanes::<L>(prev.0, self.0)) }
    }
    #[inline(always)]
    unsafe fn find_lane(self, search: u16) -> usize {
        unsafe {
            let target = _mm512_set1_epi8(search as i8);
            let mask = _mm512_cmpeq_epi8_mask(self.0, target);
            mask.trailing_zeros() as usize
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self {
        assert_eq!(values.len(), 64);
        let mut buf = [0u8; 64];
        for i in 0..64 {
            buf[i] = values[i] as u8;
        }
        Self(unsafe { _mm512_loadu_si512(buf.as_ptr() as *const __m512i) })
    }
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16> {
        let mut buf = [0u8; 64];
        unsafe { _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, self.0) };
        buf.iter().map(|&v| v as u16).collect()
    }
}

/// The four 128-bit quarters of `v`
#[inline(always)]
unsafe fn quarters(v: __m512i) -> (__m128i, __m128i, __m128i, __m128i) {
    unsafe {
        (
            _mm512_castsi512_si128(v),
            _mm512_extracti32x4_epi32::<1>(v),
            _mm512_extracti32x4_epi32::<2>(v),
            _mm512_extracti32x4_epi32::<3>(v),
        )
    }
}

/// Zero-filled shift within 128-bit segments by `L` lanes
#[inline(always)]
unsafe fn in_segment<const L: i32>(v: Avx512U8Score) -> Avx512U8Score {
    unsafe { Avx512U8Score(_mm512_bslli_epi128::<L>(v.0)) }
}

/// The last lane of the segment `k + 1` segments back in every lane of a
/// segment. Segments without one take their own lanes, which a decay (or the
/// ramp) keeps below the scan
#[inline(always)]
unsafe fn segment_end(v: Avx512U8Score, k: usize) -> Avx512U8Score {
    static SEGMENT_ENDS: [[u8; 64]; 3] = {
        let mut arr = [[0u8; 64]; 3];
        let mut i = 0;
        while i < 3 * 64 {
            let (k, j) = (i / 64 + 1, i % 64);
            arr[k - 1][j] = if j / 16 >= k {
                ((j / 16 - k) * 16 + 15) as u8
            } else {
                j as u8
            };
            i += 1;
        }
        arr
    };
    unsafe {
        let idx = _mm512_loadu_si512(SEGMENT_ENDS[k].as_ptr() as *const __m512i);
        Avx512U8Score(_mm512_permutexvar_epi8(idx, v.0))
    }
}

#[inline(always)]
unsafe fn shift_right_u16_lanes<const L: i32>(prev: __m512i, cur: __m512i) -> __m512i {
    const { assert!(L >= 0 && L <= 32) };
    match L {
        0 => cur,
        32 => prev,

        _ if cfg!(miri) => unsafe {
            // Miri doesn't support _mm512_permutex2var_epi16, so use a fallback
            // implementation TODO: add support to Miri
            let mut cur_arr = [0u16; 32];
            _mm512_storeu_si512(cur_arr.as_mut_ptr() as *mut __m512i, cur);

            let mut prev_arr = [0u16; 32];
            _mm512_storeu_si512(prev_arr.as_mut_ptr() as *mut __m512i, prev);

            let mut shifted = [0u16; 32];
            core::ptr::copy_nonoverlapping(
                cur_arr.as_ptr(),
                shifted.as_mut_ptr().add(L as usize),
                32 - L as usize,
            );
            core::ptr::copy_nonoverlapping(
                prev_arr.as_ptr().add(32 - L as usize),
                shifted.as_mut_ptr(),
                L as usize,
            );
            _mm512_loadu_si512(shifted.as_ptr() as *const __m512i)
        },

        _ => {
            let idx_arr: [u16; 32] = const {
                let mut arr = [0u16; 32];
                let mut i = 0usize;
                while i < 32 {
                    arr[i] = ((32 - L) + i as i32) as u16;
                    i += 1;
                }
                arr
            };
            let idx = unsafe { _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i) };
            unsafe { _mm512_permutex2var_epi16(prev, idx, cur) }
        }
    }
}

/// Cross-lane byte shift used by both BytesVec and ScoreVec.
#[inline(always)]
unsafe fn shift_right_lanes<const L: i32>(prev: __m512i, cur: __m512i) -> __m512i {
    const { assert!(L >= 0 && L < 64) };
    match L {
        0 => cur,

        // use native cross-lane byte shifts when available
        4 => unsafe { _mm512_alignr_epi32::<15>(cur, prev) },
        12 => unsafe { _mm512_alignr_epi32::<13>(cur, prev) },
        48 => unsafe { _mm512_alignr_epi32::<4>(cur, prev) },
        8 => unsafe { _mm512_alignr_epi64::<7>(cur, prev) },
        16 => unsafe { _mm512_alignr_epi64::<6>(cur, prev) },
        32 => unsafe { _mm512_alignr_epi64::<4>(cur, prev) },

        _ if cfg!(miri) => unsafe {
            // Miri doesn't support _mm512_permutex2var_epi16, so use a fallback
            // implementation TODO: add support to Miri
            let mut cur_arr = [0u8; 64];
            _mm512_storeu_si512(cur_arr.as_mut_ptr() as *mut __m512i, cur);

            let mut prev_arr = [0u8; 64];
            _mm512_storeu_si512(prev_arr.as_mut_ptr() as *mut __m512i, prev);

            let mut shifted = [0u8; 64];
            core::ptr::copy_nonoverlapping(
                cur_arr.as_ptr(),
                shifted.as_mut_ptr().add(L as usize),
                64 - L as usize,
            );
            core::ptr::copy_nonoverlapping(
                prev_arr.as_ptr().add(64 - L as usize),
                shifted.as_mut_ptr(),
                L as usize,
            );
            _mm512_loadu_si512(shifted.as_ptr() as *const __m512i)
        },

        // fallback to byte permute
        _ => {
            let idx_arr: [u8; 64] = const {
                let mut arr = [0u8; 64];
                let mut i = 0usize;
                while i < 64 {
                    arr[i] = ((64 - L) + i as i32) as u8;
                    i += 1;
                }
                arr
            };
            let idx = unsafe { _mm512_loadu_si512(idx_arr.as_ptr() as *const __m512i) };
            unsafe { _mm512_permutex2var_epi8(prev, idx, cur) }
        }
    }
}
