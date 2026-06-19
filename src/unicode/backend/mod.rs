use std::fmt::Debug;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) use avx::Utf8ToUtf32AVX2;
#[cfg(target_arch = "x86_64")]
pub(crate) use avx512::Utf8ToUtf32AVX512;
pub(crate) use scalar::Utf8ToUtf32Scalar;

pub(crate) trait Backend: Sized + Debug + Clone + 'static {
    fn is_available() -> bool;

    /// # Safety
    /// The backend's target features must be enabled at the call site.
    unsafe fn convert_into(input: &str, out: &mut Vec<u32>);
}

#[cfg(all(feature = "benching", target_arch = "x86_64"))]
#[inline(always)]
pub fn avx512_is_available() -> bool {
    Utf8ToUtf32AVX512::is_available()
}

#[cfg(all(feature = "benching", target_arch = "x86_64"))]
#[inline(always)]
pub fn avx2_is_available() -> bool {
    Utf8ToUtf32AVX2::is_available()
}

#[cfg(all(feature = "benching", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn convert_avx512_into(input: &str, out: &mut Vec<u32>) {
    unsafe {
        Utf8ToUtf32AVX512::convert_into(input, out);
    }
}

#[cfg(all(feature = "benching", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn convert_avx2_into(input: &str, out: &mut Vec<u32>) {
    unsafe {
        Utf8ToUtf32AVX2::convert_into(input, out);
    }
}

#[cfg(feature = "benching")]
#[inline(always)]
pub fn convert_scalar_into(input: &str, out: &mut Vec<u32>) {
    unsafe {
        Utf8ToUtf32Scalar::convert_into(input, out);
    }
}
