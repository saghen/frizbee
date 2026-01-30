use std::simd::Simd;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512;
pub mod generic;

#[inline(never)]
pub fn interleave<const W: usize, const L: usize>(strs: [&str; L]) -> [Simd<u16, L>; W] {
    match L {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        32 if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx2") =>
        unsafe {
            let strs_ptr = &strs as *const [&str; L] as *const [&str; 32];
            let result = avx512::interleave::<W>(*strs_ptr);
            let result_ptr = &result as *const [Simd<u16, 32>; W] as *const [Simd<u16, L>; W];
            std::ptr::read(result_ptr)
        },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        16 if is_x86_feature_detected!("avx2") => unsafe {
            let strs_ptr = &strs as *const [&str; L] as *const [&str; 16];
            let result = avx2::interleave::<W>(*strs_ptr);
            let result_ptr = &result as *const [Simd<u16, 16>; W] as *const [Simd<u16, L>; W];
            std::ptr::read(result_ptr)
        },
        _ => generic::interleave::<W, L>(strs),
    }
}
