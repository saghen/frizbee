//! SIMD backend abstraction for the Smith-Waterman kernel.
//!
//! Each backend exposes its native vector width via [`Backend::LANES`] and
//! three associated vector types: [`BytesVec`] for haystack bytes (LANES bytes
//! wide), [`MaskVec`] for boolean comparison results, and [`ScoreVec`] for the
//! score matrix (LANES × u16 or LANES × u8). Keeping masks as a distinct type
//! lets AVX-512 carry them as native `__mmask*` bitmasks rather than
//! 64-byte vectors, while other backends can alias [`MaskVec`] to their
//! [`BytesVec`] type with no loss.
//!
//! The kernel itself is generic over [`Backend`] and the lane count threads
//! through chunk arithmetic, the matrix stride, the traceback,
//! and the horizontal-gap propagation unroll.
//!
//! Available backends:
//!   - AVX-512: LANES = 32/64 (scoring u16 x 32 = 512-bit or u8 x 64 = 512-bit)
//!   - AVX2:    LANES = 16/32 (scoring u16 x 16 = 256-bit or u8 x 32 = 256-bit)
//!   - SSE:     LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - NEON:    LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - WASM:    LANES = 8/16  (scoring u16 x 8 = 128-bit or u8 x 16 = 128-bit)
//!   - Scalar:  LANES = 8/16 (fallback for non-SIMD systems)

/// Implements a backend's gap propagation entry points by forwarding to the
/// helpers specialized for its lane count
macro_rules! gap_dispatch {
    (8) => { gap_dispatch!(@impl scan_8_lane, unicode_steps_8_lane); };
    (16) => { gap_dispatch!(@impl scan_16_lane, unicode_steps_16_lane); };
    (32) => { gap_dispatch!(@impl scan_32_lane, unicode_steps_32_lane); };
    (64) => { gap_dispatch!(@impl scan_64_lane, unicode_steps_64_lane); };
    (@impl $scan:ident, $unicode_steps:ident) => {
        #[inline(always)]
        unsafe fn horizontal_scan(
            values: Self::Score,
            prev: Self::Score,
            steps: &[Self::Score; 10],
            scores: Self::Score,
        ) -> (Self::Score, Self::Score) {
            unsafe {
                crate::smith_waterman::algo::ascii_gap::$scan::<Self>(values, prev, steps, scores)
            }
        }

        #[inline(always)]
        unsafe fn unicode_gap_steps(
            d1: Self::Score,
            prev: &[Self::Score; 10],
        ) -> [Self::Score; 10] {
            unsafe { crate::smith_waterman::algo::unicode_gap::$unicode_steps::<Self>(d1, prev) }
        }
    };
}

/// A mask of `n` (0..=64) 0xFF bytes then zeros, for the partial loads: ANDed
/// with a full load that stays within the page (see
/// [`can_overread`](crate::prefilter::algo::can_overread))
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[inline(always)]
pub(crate) fn keep_first(n: usize) -> *const u8 {
    #[repr(C, align(64))]
    struct KeepFirst([u8; 128]);
    static KEEP_FIRST: KeepFirst = {
        let mut bytes = [0; 128];
        let mut i = 0;
        while i < 64 {
            bytes[i] = 0xFF;
            i += 1;
        }
        KeepFirst(bytes)
    };
    debug_assert!(n <= 64);
    unsafe { KEEP_FIRST.0.as_ptr().add(64 - n) }
}

/// The `remaining` bytes at `ptr`, zero padded. Only the copy is outlined, so
/// the caller loads it with its target features
///
/// # Safety
/// `ptr` must point to `remaining <= N` readable bytes.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[cold]
#[inline(never)]
pub(crate) unsafe fn copy_partial<const N: usize>(ptr: *const u8, remaining: usize) -> [u8; N] {
    let mut data = [0u8; N];
    unsafe { core::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), remaining) };
    data
}

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(test, scalar_fallback))]
mod scalar;
#[cfg(target_arch = "x86_64")]
mod sse;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

#[cfg(target_arch = "x86_64")]
pub use avx::{BackendAVX, BackendAVXU8};
#[cfg(target_arch = "x86_64")]
pub use avx512::{BackendAVX512, BackendAVX512U8};
#[cfg(target_arch = "aarch64")]
pub use neon::{BackendNEON, BackendNEONU8};
#[cfg(any(test, scalar_fallback))]
pub use scalar::{BackendScalar8, BackendScalar16U8};
#[cfg(target_arch = "x86_64")]
pub use sse::{BackendSSE, BackendSSEU8};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm::{BackendWasm, BackendWasmU8};

/// A SIMD backend for Smith-Waterman matching that supports variable-width
/// lanes (8, 16, 32, 64) and score primitives (u8, u16).
///
/// `LANES` is the number of cells per chunk in the score matrix. The matching
/// [`BytesVec`] holds `LANES` meaningful bytes (one per lane); comparisons
/// produce a [`MaskVec`] which can be widened element-wise to the [`ScoreVec`]
/// via [`Backend::widen_mask`].
pub trait Backend: Sized + core::fmt::Debug + Clone + 'static {
    const LANES: usize;
    /// Size of a single score lane in bytes. 1 for u8-element backends,
    /// 2 for u16-element backends. Used by the traceback to
    /// read individual cells from the matrix's raw byte view.
    const LANE_BYTES: usize;
    type Bytes: BytesVec<Mask = Self::Mask>;
    type Mask: MaskVec;
    type Score: ScoreVec;

    /// Whether this backend's required CPU features are available at runtime
    fn is_available() -> bool;

    /// Widen a comparison [`MaskVec`] into a [`ScoreVec`] with each lane set
    /// to `0xFF` (u8 score) or `0xFFFF` (u16 score) where the mask is true and
    /// zero elsewhere.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn widen_mask(m: Self::Mask) -> Self::Score;

    /// The steps [`Backend::horizontal_scan`] takes in the ASCII kernel, for a
    /// gap extend penalty
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    #[inline(always)]
    unsafe fn ascii_gap_steps(gap_extend: u16) -> [Self::Score; 10] {
        use crate::smith_waterman::algo::ascii_gap::GAP_SHIFTS;
        // (no closures: target features)
        unsafe {
            let mut steps = [Self::Score::zero(); 10];
            for (i, step) in steps.iter_mut().enumerate() {
                let shift = GAP_SHIFTS.get(i).copied().unwrap_or(Self::LANES as u16);
                *step = Self::Score::splat(gap_extend.saturating_mul(shift));
            }
            steps
        }
    }

    /// Decayed prefix max of a score row, carrying the previous chunk's scan
    /// `prev` (see [`crate::smith_waterman::algo::ascii_gap`]). Returns the
    /// scan and the scan folded with the row's `scores`.
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn horizontal_scan(
        values: Self::Score,
        prev: Self::Score,
        steps: &[Self::Score; 10],
        scores: Self::Score,
    ) -> (Self::Score, Self::Score);

    /// Per-chunk decays of the Unicode horizontal scan from `d1`, the gap
    /// extend penalty in the chunk's non-continuation lanes, and the previous
    /// chunk's decays `prev` (see [`crate::smith_waterman::algo::unicode_gap`])
    ///
    /// # Safety
    /// The backend's required target features must be enabled at the call site.
    unsafe fn unicode_gap_steps(d1: Self::Score, prev: &[Self::Score; 10]) -> [Self::Score; 10];
}

/// A byte-wide vector with `LANES` meaningful bytes
pub trait BytesVec: Copy + core::fmt::Debug {
    /// The backend's mask type, produced by comparison operations.
    type Mask: MaskVec;

    /// Broadcast `value` into every byte lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u8) -> Self;

    /// Per-lane equality, producing a true bit in each lane where equal.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn eq(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned greater-than, producing a true bit where
    /// `self > other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn gt(self, other: Self) -> Self::Mask;

    /// Per-lane unsigned less-than, producing a true bit where `self < other`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn lt(self, other: Self) -> Self::Mask;

    /// Load up to `LANES` bytes from `data + start`. `len` is the total length
    /// of the buffer pointed to by `data`.
    ///
    /// # Safety
    /// Caller must guarantee `data` points to at least `len` bytes and the
    /// backend's target features are enabled.
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self;

    /// Whether each byte is an ASCII uppercase or lowercase letter, or a
    /// delimiter: any ASCII byte that isn't a letter or a digit
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    #[inline(always)]
    unsafe fn classes(self) -> (Self::Mask, Self::Mask, Self::Mask) {
        unsafe {
            let is_upper = self
                .lt(Self::splat(b'Z' + 1))
                .and(self.gt(Self::splat(b'A' - 1)));
            let is_lower = self
                .lt(Self::splat(b'z' + 1))
                .and(self.gt(Self::splat(b'a' - 1)));
            let is_digit = self
                .gt(Self::splat(b'0' - 1))
                .and(self.lt(Self::splat(b'9' + 1)));
            let is_delimiter = is_upper
                .or(is_lower)
                .or(is_digit)
                .or(self.gt(Self::splat(127)))
                .not();
            (is_upper, is_lower, is_delimiter)
        }
    }

    #[cfg(test)]
    fn from_lanes(values: &[u8]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u8>;
}

/// A boolean mask over `LANES` lanes, produced by [`BytesVec`] comparisons.
///
/// Each lane holds a logical true/false. Non-AVX-512 backends typically alias
/// this to their [`BytesVec`] type and represent the mask as 0xFF/0x00 per
/// byte. AVX-512 uses a native `__mmask64` bitmask.
pub trait MaskVec: Copy + core::fmt::Debug {
    /// False in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Per-lane logical AND.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Per-lane logical OR.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn or(self, other: Self) -> Self;

    /// Per-lane logical NOT.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn not(self) -> Self;

    /// Shift right by 1 lane, filling lane 0 with the highest meaningful lane
    /// of `prev`.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded_1(self, prev: Self) -> Self;

    #[cfg(test)]
    fn from_lanes(values: &[bool]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<bool>;
}

/// A score-wide vector holding `LANES` u8 or u16 elements.
pub trait ScoreVec: Copy + core::fmt::Debug {
    /// Zero in every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn zero() -> Self;

    /// Broadcast `value` into every lane.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn splat(value: u16) -> Self;

    /// `value` in lane 0, zero in all other lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn first_lane(value: u16) -> Self;

    /// Per-lane unsigned max
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn max(self, other: Self) -> Self;

    /// Horizontal unsigned max across all lanes
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn horizontal_max(self) -> u16;

    /// Per-lane saturating add
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn add(self, other: Self) -> Self;

    /// Per-lane saturating subtract
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn subs(self, other: Self) -> Self;

    /// Bitwise AND
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn and(self, other: Self) -> Self;

    /// Shift right by `L` lanes, filling the low `L` lanes with the high `L`
    /// lanes of `prev`. `L` must be in `0..=LANES`. `L == LANES` returns
    /// `prev` unmodified.
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn shift_right_padded<const L: i32>(self, prev: Self) -> Self;

    /// Index of the lowest lane equal to `search`, or `LANES` if absent
    ///
    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn find_lane(self, search: u16) -> usize;

    // ----- Test-only helpers ---------------------------------------------

    #[cfg(test)]
    fn from_lanes(values: &[u16]) -> Self;
    #[cfg(test)]
    fn to_lanes(self) -> Vec<u16>;
}

#[cfg(test)]
mod tests;
