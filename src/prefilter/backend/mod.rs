use std::fmt::Debug;

use super::algo::Prefilter;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod sse;

#[cfg(target_arch = "x86_64")]
pub use avx::PrefilterAVX;

#[cfg(target_arch = "x86_64")]
pub type PrefilterAVX512 = Prefilter<avx512::PrefilterAVX512Backend>;
#[cfg(target_arch = "aarch64")]
pub type PrefilterNEON = Prefilter<neon::PrefilterNEONBackend>;
pub type PrefilterScalar = Prefilter<scalar::PrefilterScalarBackend>;
#[cfg(target_arch = "x86_64")]
pub type PrefilterSSE = Prefilter<sse::PrefilterSSEBackend>;

pub(crate) trait Mask:
    Copy + Debug + PartialOrd + BitMaskOps + Send + Sync + 'static
{
}

impl<T> Mask for T where T: Copy + Debug + PartialOrd + BitMaskOps + Send + Sync + 'static {}

pub(crate) trait BitMaskOps {
    fn zero() -> Self;
    fn all() -> Self;
    fn first_n(n: usize) -> Self;
    fn is_zero(self) -> bool;
    fn trailing_zeros(self) -> usize;
    fn leading_zeros(self) -> usize;
    fn or(self, other: Self) -> Self;
    fn and(self, other: Self) -> Self;
    fn clear_through_lowest(self, hit: Self) -> Self;
}

macro_rules! impl_mask {
    ($ty:ty) => {
        impl BitMaskOps for $ty {
            #[inline(always)]
            fn zero() -> Self {
                0
            }

            #[inline(always)]
            fn all() -> Self {
                <$ty>::MAX
            }

            #[inline(always)]
            fn first_n(n: usize) -> Self {
                if n >= <$ty>::BITS as usize {
                    <$ty>::MAX
                } else {
                    ((1 as $ty) << n) - 1
                }
            }

            #[inline(always)]
            fn is_zero(self) -> bool {
                self == 0
            }

            #[inline(always)]
            fn trailing_zeros(self) -> usize {
                <$ty>::trailing_zeros(self) as usize
            }

            #[inline(always)]
            fn leading_zeros(self) -> usize {
                <$ty>::leading_zeros(self) as usize
            }

            #[inline(always)]
            fn or(self, other: Self) -> Self {
                self | other
            }

            #[inline(always)]
            fn and(self, other: Self) -> Self {
                self & other
            }

            #[inline(always)]
            fn clear_through_lowest(self, hit: Self) -> Self {
                self & !(hit ^ hit.wrapping_sub(1))
            }
        }
    };
}

impl_mask!(u16);
impl_mask!(u32);
impl_mask!(u64);

pub(crate) trait Backend: Sized + Debug + Clone + 'static {
    const LANES: usize;

    type Chunk: Copy + Debug;
    type Mask: Mask;

    fn is_available() -> bool;

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn splat(c: u8) -> Self::Chunk;

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn eq(a: Self::Chunk, b: Self::Chunk) -> Self::Mask;

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn broadcast(c: (u8, u8)) -> (Self::Chunk, Self::Chunk);

    /// # Safety
    /// `ptr` must point to at least `LANES` readable bytes, and the backend's
    /// target features must be enabled.
    unsafe fn load(ptr: *const u8) -> Self::Chunk;

    /// # Safety
    /// `ptr` must point to `remaining` readable bytes, `remaining < LANES`, and
    /// the backend's target features must be enabled.
    #[inline(always)]
    unsafe fn load_partial(ptr: *const u8, remaining: usize, _mask: Self::Mask) -> Self::Chunk {
        unsafe { load_partial_copy::<Self>(ptr, remaining) }
    }

    /// # Safety
    /// The backend's target features must be enabled.
    unsafe fn occ(chunk: Self::Chunk, needle: (Self::Chunk, Self::Chunk)) -> Self::Mask;

    /// # Safety
    /// The backend's target features must be enabled and `hit` must be nonzero.
    #[inline(always)]
    unsafe fn first_hit_pos(hit: Self::Mask) -> usize {
        hit.trailing_zeros()
    }

    /// # Safety
    /// The backend's target features must be enabled and `hit` must be nonzero.
    #[inline(always)]
    unsafe fn clear_through_lowest(mask: Self::Mask, hit: Self::Mask) -> Self::Mask {
        mask.clear_through_lowest(hit)
    }
}

#[cold]
#[inline(never)]
unsafe fn load_partial_copy<B: Backend>(ptr: *const u8, remaining: usize) -> B::Chunk {
    unsafe {
        let mut data = [0u8; 64];
        std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), remaining);
        B::load(data.as_ptr())
    }
}
