//! Runtime CPU feature detection for x86-64

use core::arch::asm;
use core::arch::x86_64::{__cpuid, __cpuid_count, CpuidResult};
use core::sync::atomic::{AtomicU32, Ordering};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Features {
    pub sse2: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vbmi: bool,
    pub bmi1: bool,
    pub bmi2: bool,
}

const SSE2: u32 = 1 << 0;
const SSSE3: u32 = 1 << 1;
const SSE41: u32 = 1 << 2;
const AVX: u32 = 1 << 3;
const AVX2: u32 = 1 << 4;
const FMA: u32 = 1 << 5;
const AVX512F: u32 = 1 << 6;
const AVX512BW: u32 = 1 << 7;
const AVX512VBMI: u32 = 1 << 8;
const BMI1: u32 = 1 << 9;
const BMI2: u32 = 1 << 10;
/// Marks the cache as populated, so "no features" differs from "not yet run"
const INIT: u32 = 1 << 31;

static CACHE: AtomicU32 = AtomicU32::new(0);

pub fn detect() -> Features {
    // racing doesn't matter, bits are deterministic
    let mut bits = CACHE.load(Ordering::Relaxed);
    if bits == 0 {
        bits = detect_bits() | INIT;
        CACHE.store(bits, Ordering::Relaxed);
    }
    Features {
        sse2: bits & SSE2 != 0,
        ssse3: bits & SSSE3 != 0,
        sse41: bits & SSE41 != 0,
        avx: bits & AVX != 0,
        avx2: bits & AVX2 != 0,
        fma: bits & FMA != 0,
        avx512f: bits & AVX512F != 0,
        avx512bw: bits & AVX512BW != 0,
        avx512vbmi: bits & AVX512VBMI != 0,
        bmi1: bits & BMI1 != 0,
        bmi2: bits & BMI2 != 0,
    }
}

const fn flag(cond: bool, bit: u32) -> u32 {
    if cond { bit } else { 0 }
}

// `__cpuid`/`__cpuid_count` became safe in Rust 1.94 but we support >= 1.89
#[allow(unused_unsafe)]
fn detect_bits() -> u32 {
    // miri cannot execute `cpuid`/`xgetbv`, so fall back to compile time detection
    if cfg!(miri) {
        return flag(cfg!(target_feature = "sse2"), SSE2)
            | flag(cfg!(target_feature = "ssse3"), SSSE3)
            | flag(cfg!(target_feature = "sse4.1"), SSE41)
            | flag(cfg!(target_feature = "avx"), AVX)
            | flag(cfg!(target_feature = "avx2"), AVX2)
            | flag(cfg!(target_feature = "fma"), FMA)
            | flag(cfg!(target_feature = "avx512f"), AVX512F)
            | flag(cfg!(target_feature = "avx512bw"), AVX512BW)
            | flag(cfg!(target_feature = "avx512vbmi"), AVX512VBMI)
            | flag(cfg!(target_feature = "bmi1"), BMI1)
            | flag(cfg!(target_feature = "bmi2"), BMI2);
    }

    let max_leaf = unsafe { __cpuid(0) }.eax;
    let leaf1 = unsafe { __cpuid_count(1, 0) }; // mandatory on x86_64
    // optional based on leaf 0
    let leaf7 = if max_leaf >= 7 {
        unsafe { __cpuid_count(7, 0) }
    } else {
        CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        }
    };

    // OSXSAVE (leaf1 ecx bit 27) means XCR0 is readable via xgetbv
    let osxsave = leaf1.ecx & (1 << 27) != 0;
    let xcr0 = if osxsave {
        let lo: u32;
        let hi: u32;
        // Caller observed OSXSAVE, which implies both that `xgetbv` exists
        // and that CR4.OSXSAVE is set, so it cannot fault.
        //
        // Raw asm rather than `_xgetbv` intrinsic to avoid the compiler hoisting this
        // out of the branch guard, even though in practice, it shouldn't. We're just
        // being extra safe here.
        unsafe {
            asm!(
                "xgetbv",
                in("ecx") 0u32,
                out("eax") lo,
                out("edx") hi,
                options(nomem, nostack, preserves_flags),
            );
        }
        ((hi as u64) << 32) | (lo as u64)
    } else {
        0
    };

    // XCR0: bit 1 = XMM, 2 = YMM, 5 = opmask, 6 = ZMM_Hi256, 7 = Hi16_ZMM
    let avx_os = xcr0 & 0x06 == 0x06;
    let avx512_os = xcr0 & 0xE6 == 0xE6; // includes XMM+YMM, not just ZMM state

    let avx = leaf1.ecx & (1 << 28) != 0 && avx_os;
    let avx2 = avx && leaf7.ebx & (1 << 5) != 0;
    // require AVX2 for AVX512F to simplify consumers
    let avx512f = avx2 && avx512_os && leaf7.ebx & (1 << 16) != 0;

    flag(leaf1.edx & (1 << 26) != 0, SSE2)
        | flag(leaf1.ecx & (1 << 9) != 0, SSSE3)
        | flag(leaf1.ecx & (1 << 19) != 0, SSE41)
        | flag(avx, AVX)
        | flag(avx2, AVX2)
        | flag(avx && leaf1.ecx & (1 << 12) != 0, FMA)
        | flag(avx512f, AVX512F)
        | flag(avx512f && leaf7.ebx & (1 << 30) != 0, AVX512BW)
        | flag(avx512f && leaf7.ecx & (1 << 1) != 0, AVX512VBMI)
        | flag(leaf7.ebx & (1 << 3) != 0, BMI1)
        | flag(leaf7.ebx & (1 << 8) != 0, BMI2)
}

#[cfg(test)]
mod tests {
    use super::detect;
    use std::arch::is_x86_feature_detected;

    /// Every flag must agree with std's runtime detection. Under miri, both
    /// sides fall back to compile time features.
    #[test]
    fn matches_std_detection() {
        let features = detect();
        assert_eq!(features.sse2, is_x86_feature_detected!("sse2"), "sse2");
        assert_eq!(features.ssse3, is_x86_feature_detected!("ssse3"), "ssse3");
        assert_eq!(features.sse41, is_x86_feature_detected!("sse4.1"), "sse4.1");
        assert_eq!(features.avx, is_x86_feature_detected!("avx"), "avx");
        assert_eq!(features.avx2, is_x86_feature_detected!("avx2"), "avx2");
        assert_eq!(features.fma, is_x86_feature_detected!("fma"), "fma");
        assert_eq!(
            features.avx512f,
            is_x86_feature_detected!("avx512f"),
            "avx512f"
        );
        assert_eq!(
            features.avx512bw,
            is_x86_feature_detected!("avx512bw"),
            "avx512bw"
        );
        assert_eq!(
            features.avx512vbmi,
            is_x86_feature_detected!("avx512vbmi"),
            "avx512vbmi"
        );
        assert_eq!(features.bmi1, is_x86_feature_detected!("bmi1"), "bmi1");
        assert_eq!(features.bmi2, is_x86_feature_detected!("bmi2"), "bmi2");
    }
}
