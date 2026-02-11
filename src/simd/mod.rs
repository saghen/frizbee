mod avx;
mod sse;

pub use avx::AVXVector;
pub use sse::SSEVector;

#[repr(C, align(32))]
pub struct Aligned32<T>(pub T);

pub trait Vector: Copy + core::fmt::Debug {
    /// Create a vector with zeros in all lanes.
    unsafe fn zero() -> Self;
    /// Create a vector with 8-bit lanes with the given byte repeated into each
    /// lane.
    unsafe fn splat_u8(value: u8) -> Self;
    /// Create a vector with 16-bit lanes with the given byte repeated into each
    /// lane.
    unsafe fn splat_u16(value: u16) -> Self;

    /// Read a vector-size number of bytes from the given pointer. The pointer
    /// must be aligned to the size of the vector.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that at least `BYTES` bytes are readable from
    /// `data` and that `data` is aligned to a `BYTES` boundary.
    unsafe fn load_aligned(data: *const u8) -> Self;

    /// Read a vector-size number of bytes from the given pointer. The pointer
    /// does not need to be aligned.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that at least `BYTES` bytes are readable from
    /// `data`.
    unsafe fn load_unaligned(data: *const u8) -> Self;

    unsafe fn eq_u8(self, other: Self) -> Self;
    unsafe fn gt_u8(self, other: Self) -> Self;
    unsafe fn lt_u8(self, other: Self) -> Self;

    unsafe fn max_u16(self, other: Self) -> Self;
    /// Get the maximum value in the vector as a scalar
    unsafe fn smax_u16(self) -> u16;

    unsafe fn add_u16(self, other: Self) -> Self;
    unsafe fn subs_u16(self, other: Self) -> Self;

    unsafe fn and(self, other: Self) -> Self;
    unsafe fn or(self, other: Self) -> Self;
    unsafe fn xor(self, other: Self) -> Self;
    unsafe fn not(self) -> Self;

    unsafe fn shift_right_padded_u16<const N: i32>(self, other: Self) -> Self;
}

pub trait Vector128: Vector {
    /// The 256-bit vector type that this 128-bit vector expands to
    type Expanded: Vector256;

    /// Loads from the given pointer, where the number of remaining bytes may be less than
    /// the vector size. The pointer does not need to be aligned.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that the pointer contains `len` bytes and `start < len`.
    unsafe fn load_partial(data: *const u8, start: usize, len: usize) -> Self;

    /// Expands the vector from 128-bit to 256-bit by expanding each byte
    unsafe fn cast_i8_to_i16(self) -> Self::Expanded;

    /// Shift `self` right by `L` bytes, filling in the low bytes with the right most values in
    /// `other`
    unsafe fn shift_right_padded_u8<const L: i32>(self, other: Self) -> Self;
}

pub trait Vector256: Vector {
    /// Extract the value at the given index from the vector
    unsafe fn idx_u16(self, search: u16) -> usize;
    /// Load the vector via transmute since alinment is guaranteed
    unsafe fn from_aligned(data: Aligned32<[u8; 32]>) -> Self;
    /// Uses a mask to blend the values of `self` and `other` where `00` means `self` and `FF` means
    /// `other`
    unsafe fn blendv(self, other: Self, mask: Self) -> Self;
    /// Shift `self` right by `N` bytes, filling in the low bytes with zeros
    unsafe fn shift_right_u16<const N: i32>(self) -> Self;
}
