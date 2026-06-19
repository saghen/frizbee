//! UTF-8 preprocessing for future Unicode-aware matching.
//!
//! The matcher still operates on bytes today. This module only prepares the
//! UTF-32 representation that a future Unicode prefilter and Smith-Waterman
//! path can consume.

#![allow(dead_code)]

#[cfg(feature = "benching")]
pub mod backend;
#[cfg(not(feature = "benching"))]
pub(crate) mod backend;

use backend::{Backend, Utf8ToUtf32Scalar};

#[cfg(target_arch = "x86_64")]
use backend::{Utf8ToUtf32AVX2, Utf8ToUtf32AVX512};

/// Reusable UTF-8 preprocessing state.
///
/// Construction performs runtime backend detection once. Each call to
/// [`Unicode::prepare`] then reuses the owned UTF-32 buffer, replacing its
/// previous contents with the input's codepoints.
#[derive(Debug, Clone)]
pub struct Unicode {
    backend: UnicodeBackend,
    codepoints: Vec<u32>,
}

impl Default for Unicode {
    fn default() -> Self {
        Self::new()
    }
}

impl Unicode {
    pub fn new() -> Self {
        Self {
            backend: UnicodeBackend::detect(),
            codepoints: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            backend: UnicodeBackend::detect(),
            codepoints: Vec::with_capacity(capacity),
        }
    }

    #[cfg(any(test, feature = "fuzzing", feature = "benching"))]
    pub fn new_scalar() -> Self {
        Self {
            backend: UnicodeBackend::Scalar,
            codepoints: Vec::new(),
        }
    }

    #[cfg(any(test, feature = "fuzzing", feature = "benching"))]
    pub fn new_scalar_with_capacity(capacity: usize) -> Self {
        Self {
            backend: UnicodeBackend::Scalar,
            codepoints: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn needs_unicode(&self, needle: &str) -> bool {
        !needle.is_ascii()
    }

    #[inline(always)]
    pub fn prepare(&mut self, input: &str) -> &[u32] {
        unsafe {
            self.backend.convert_into(input, &mut self.codepoints);
        }
        &self.codepoints
    }
}

#[derive(Debug, Clone, Copy)]
enum UnicodeBackend {
    #[cfg(target_arch = "x86_64")]
    AVX512,
    #[cfg(target_arch = "x86_64")]
    AVX2,
    Scalar,
}

impl UnicodeBackend {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if Utf8ToUtf32AVX512::is_available() {
                return Self::AVX512;
            }
            if Utf8ToUtf32AVX2::is_available() {
                return Self::AVX2;
            }
        }

        Self::Scalar
    }

    #[inline(always)]
    unsafe fn convert_into(self, input: &str, out: &mut Vec<u32>) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX512 => unsafe {
                Utf8ToUtf32AVX512::convert_into(input, out);
            },
            #[cfg(target_arch = "x86_64")]
            Self::AVX2 => unsafe {
                Utf8ToUtf32AVX2::convert_into(input, out);
            },
            Self::Scalar => unsafe {
                Utf8ToUtf32Scalar::convert_into(input, out);
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bolero::check;

    #[test]
    fn unicode_detection_only_depends_on_non_ascii() {
        let unicode = Unicode::new();
        assert!(!unicode.needs_unicode(""));
        assert!(!unicode.needs_unicode("abcXYZ012/_-"));
        assert!(unicode.needs_unicode("cafe\u{301}"));
        assert!(unicode.needs_unicode("\u{6771}\u{4eac}"));
        assert!(unicode.needs_unicode("hello \u{1f600}"));
    }

    #[test]
    fn converts_manual_cases() {
        for input in [
            "",
            "abc",
            "01234567890123456789012345678901",
            "012345678901234567890123456789012",
            "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}",
            "\u{416}\u{416}\u{416}\u{416}\u{416}\u{416}\u{416}\u{416}",
            "\u{6771}\u{4eac}\u{6771}\u{4eac}\u{6771}\u{4eac}\u{6771}\u{4eac}",
            "\u{1f600}\u{1f600}\u{1f600}\u{1f600}",
            "a\u{e9}b\u{416}c\u{6771}d\u{1f600}e",
            "0123456789012345678901234567890\u{e9}",
            "\u{e9}0123456789012345678901234567890",
            "abc\u{e9}\u{416}\u{6771}\u{4eac}\u{1f600}xyz",
        ] {
            assert_converts(input);
        }
    }

    #[test]
    fn reusable_buffer_replaces_previous_contents() {
        let mut unicode = Unicode::new();
        assert_eq!(unicode.prepare("abcdef"), &[97, 98, 99, 100, 101, 102]);
        assert_eq!(
            unicode.prepare("\u{e9}\u{6771}\u{1f600}"),
            &[0xe9, 0x6771, 0x1f600]
        );
        assert!(unicode.prepare("").is_empty());
    }

    #[test]
    fn randomized_valid_utf8_matches_chars() {
        check!()
            .with_iterations(if cfg!(miri) { 4 } else { 256 })
            .with_max_len(if cfg!(miri) { 128 } else { 4096 })
            .for_each(|input: &[u8]| {
                let text = generated_string(input, 256);
                assert_converts(&text);
            });
    }

    fn assert_converts(input: &str) {
        let want = input.chars().map(|c| c as u32).collect::<Vec<_>>();

        let mut unicode = Unicode::new();
        let got = unicode.prepare(input);
        assert_eq!(
            got,
            want.as_slice(),
            "runtime conversion mismatch for {input:?}"
        );

        let mut scalar = Unicode::new_scalar();
        let scalar = scalar.prepare(input);
        assert_eq!(
            scalar,
            want.as_slice(),
            "scalar conversion mismatch for {input:?}"
        );

        #[cfg(target_arch = "x86_64")]
        if Utf8ToUtf32AVX512::is_available() {
            let mut avx512 = Vec::new();
            unsafe {
                Utf8ToUtf32AVX512::convert_into(input, &mut avx512);
            }
            assert_eq!(avx512, want, "AVX-512 conversion mismatch for {input:?}");
        }

        #[cfg(target_arch = "x86_64")]
        if Utf8ToUtf32AVX2::is_available() {
            let mut avx2 = Vec::new();
            unsafe {
                Utf8ToUtf32AVX2::convert_into(input, &mut avx2);
            }
            assert_eq!(avx2, want, "AVX2 conversion mismatch for {input:?}");
        }
    }

    fn generated_string(input: &[u8], max_chars: usize) -> String {
        let mut cursor = ByteCursor::new(input);
        let len = cursor.len(
            max_chars,
            &[0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127],
        );
        (0..len).map(|_| cursor.char()).collect()
    }

    struct ByteCursor<'a> {
        input: &'a [u8],
        pos: usize,
    }

    impl<'a> ByteCursor<'a> {
        fn new(input: &'a [u8]) -> Self {
            Self { input, pos: 0 }
        }

        fn next(&mut self) -> u8 {
            let byte = if self.input.is_empty() {
                (self.pos as u8).wrapping_mul(31).wrapping_add(17)
            } else {
                self.input[self.pos % self.input.len()]
                    .wrapping_add(((self.pos / self.input.len()) as u8).wrapping_mul(23))
            };
            self.pos += 1;
            byte
        }

        fn usize(&mut self) -> usize {
            let mut value = 0usize;
            for shift in (0..usize::BITS as usize).step_by(8) {
                value |= (self.next() as usize) << shift;
            }
            value
        }

        fn len(&mut self, max: usize, boundaries: &[usize]) -> usize {
            if self.next() % 4 == 0 {
                boundaries[(self.next() as usize) % boundaries.len()].min(max)
            } else {
                self.usize() % (max + 1)
            }
        }

        fn char(&mut self) -> char {
            match self.next() % 24 {
                0 => '\0',
                1 => ' ',
                2 => '/',
                3 => '.',
                4 => '_',
                5 => '-',
                6..=9 => (b'a' + (self.next() % 26)) as char,
                10..=12 => (b'A' + (self.next() % 26)) as char,
                13..=14 => (b'0' + (self.next() % 10)) as char,
                15 => '\u{e9}',
                16 => '\u{df}',
                17 => '\u{416}',
                18 => '\u{3a9}',
                19 => '\u{4e2d}',
                20 => '\u{6771}',
                21 => '\u{301}',
                22 => '\u{1f600}',
                _ => '\u{1f680}',
            }
        }
    }
}
