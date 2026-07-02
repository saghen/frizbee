//! Checks that every available backend produces the same score and the same
//! alignment-path indices as the scalar reference

use super::super::Backend;
use super::super::{BackendScalar8, BackendScalar16U8};
use super::generator::{ByteCursor, run_generated_inputs, test_bound};
use crate::Scoring;
use crate::smith_waterman::{Kernel, SmithWaterman, score_fits_in_u8};

/// randomized matcher invocation: which needle, haystack, and config to run
#[derive(Debug, Clone)]
struct BackendCase {
    needle: String,
    haystack: Vec<u8>,
    max_typos: Option<u16>,
    case_sensitive: bool,
}

impl BackendCase {
    fn from_bytes(input: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(input);
        // Bias lengths toward SIMD chunk boundaries (8/16/32/64) and the
        // greedy-fallback boundary (512), where lane-boundary bugs hide
        let needle_len = cursor
            .len(test_bound(96, 32), &[1, 7, 8, 15, 16, 31, 32, 63, 64])
            .max(1);
        let haystack_len = cursor.len(
            test_bound(768, 128),
            &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 511, 512, 513],
        );
        let max_typos = match cursor.next() % 5 {
            0 => None,
            byte => Some((byte as u16 - 1) % 17),
        };
        let case_sensitive = cursor.bool();

        Self {
            needle: String::from_utf8_lossy(&cursor.bytes(needle_len)).to_string(),
            haystack: cursor.bytes(haystack_len),
            max_typos,
            case_sensitive,
        }
    }
}

/// A small fixed corpus run in place of proptest under miri
fn miri_inputs() -> &'static [&'static [u8]] {
    &[
        b"",
        b"abcABC012 /.,_-:",
        b"lane-boundary-8-16-32-64",
        b"greedy-511-512-513",
    ]
}

/// The full backend matrix, declared once. For each backend it emits one call
/// to a caller-supplied `$callback!` macro with these trailing fields:
///
/// `$cfg, $label, $backend, $reference, $is_u8`
///
/// where `$cfg` is the `cfg(..)` predicate gating the backend's existence
/// (`all()` for the always-present scalar backends), `$reference` is the
/// lane-matched scalar oracle used by the randomized test, and `$is_u8` marks
/// backends whose score must fit in u8 (so the caller can gate them on
/// [`score_fits_in_u8`]).
macro_rules! for_each_backend {
    ($callback:ident !( $($ctx:tt)* )) => {
        $callback!($($ctx)* ; all(), "Scalar-u16", BackendScalar8, BackendScalar8, false);
        $callback!($($ctx)* ; all(), "Scalar-u8", BackendScalar16U8, BackendScalar16U8, true);

        $callback!($($ctx)* ; target_arch = "x86_64", "SSE-u16",
            super::super::BackendSSE, BackendScalar8, false);
        $callback!($($ctx)* ; target_arch = "x86_64", "AVX-u16",
            super::super::BackendAVX, super::super::scalar::TestScalar16, false);
        $callback!($($ctx)* ; target_arch = "x86_64", "AVX-512-u16",
            super::super::BackendAVX512, super::super::scalar::TestScalar32, false);
        $callback!($($ctx)* ; target_arch = "x86_64", "SSE-u8",
            super::super::BackendSSEU8, BackendScalar16U8, true);
        $callback!($($ctx)* ; target_arch = "x86_64", "AVX-u8",
            super::super::BackendAVXU8, super::super::scalar::TestScalar32U8, true);
        $callback!($($ctx)* ; target_arch = "x86_64", "AVX-512-u8",
            super::super::BackendAVX512U8, super::super::scalar::TestScalar64U8, true);

        $callback!($($ctx)* ; target_arch = "aarch64", "NEON-u16",
            super::super::BackendNEON, BackendScalar8, false);
        $callback!($($ctx)* ; target_arch = "aarch64", "NEON-u8",
            super::super::BackendNEONU8, BackendScalar16U8, true);
    };
}

// ---------------------------------------------------------------------------
// Hand-picked boundary cases: every backend must match the scalar-u16 oracle.
// ---------------------------------------------------------------------------

fn cases() -> Vec<(&'static str, &'static str)> {
    vec![
        // short
        ("a", "abc"),
        ("abc", "abc"),
        ("foo", "fooBar"),
        // crossing 8-byte chunk boundary (SSE u16 LANES = 8)
        ("foo", "012345foo"),
        ("foo", "01234567foo"),
        ("foo", "0123456789foo"),
        // crossing 16-byte boundary (AVX u16, SSE u8 LANES = 16)
        ("foo", "0123456789012345foo"),
        // crossing 32-byte boundary (AVX u8 LANES = 32)
        ("foo", "0123456789012345678901234567foo"),
        // ranges that cross multiple chunks for all widths
        ("test", "Utooooeoooosoooot"),
        ("test", "Utooooooeoooooosoooooot"),
        // typos
        ("foo", "Ufooo"),
        ("foo", "Ufo"),
        // delimiter / capitalization
        ("hw", "hello_world"),
        ("fBr", "fooBar"),
        ("D", "FOR_DIST"),
        // long needles (some short enough for u8, some not)
        ("needle", "____________needle____________"),
        ("abcdefghij", "abcdefghij"),
        ("abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
    ]
}

fn score_with<B: Backend>(needle: &str, haystack: &str) -> u16 {
    let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), false);
    matcher.score_haystack(haystack.as_bytes())
}

fn indices_with<B: Backend>(needle: &str, haystack: &str) -> Option<Vec<usize>> {
    let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), false);
    matcher
        .score_haystack_indices(haystack.as_bytes(), 0, None)
        .map(|(_, indices)| indices)
}

fn assert_score_backend<B: Backend>(label: &str, needle: &str, haystack: &str, want: u16) {
    if B::is_available() {
        let got = score_with::<B>(needle, haystack);
        assert_eq!(
            got, want,
            "{label} score mismatch for needle={needle:?} haystack={haystack:?}"
        );
    }
}

fn assert_indices_backend<B: Backend>(
    label: &str,
    needle: &str,
    haystack: &str,
    want: Option<Vec<usize>>,
) {
    if B::is_available() {
        let got = indices_with::<B>(needle, haystack);
        assert_eq!(
            got, want,
            "{label} indices mismatch for needle={needle:?} haystack={haystack:?}"
        );
    }
}

/// Per-row body for [`cross_backend_parity_score`].
macro_rules! assert_score_row {
    ($needle:expr, $haystack:expr, $want:expr ;
     $cfg:meta, $label:literal, $backend:ty, $reference:ty, $is_u8:literal) => {
        #[cfg($cfg)]
        {
            if !$is_u8 || score_fits_in_u8($needle.len(), &Scoring::default()) {
                assert_score_backend::<$backend>($label, $needle, $haystack, $want);
            }
        }
    };
}

/// Per-row body for [`cross_backend_parity_indices`].
macro_rules! assert_indices_row {
    ($needle:expr, $haystack:expr, $want:expr ;
     $cfg:meta, $label:literal, $backend:ty, $reference:ty, $is_u8:literal) => {
        #[cfg($cfg)]
        {
            if !$is_u8 || score_fits_in_u8($needle.len(), &Scoring::default()) {
                assert_indices_backend::<$backend>($label, $needle, $haystack, $want.clone());
            }
        }
    };
}

#[test]
fn cross_backend_parity_score() {
    for (needle, haystack) in cases() {
        let want = score_with::<BackendScalar8>(needle, haystack);
        for_each_backend!(assert_score_row!(needle, haystack, want));
    }
}

#[test]
fn cross_backend_parity_indices() {
    for (needle, haystack) in cases() {
        let want = indices_with::<BackendScalar8>(needle, haystack);
        for_each_backend!(assert_indices_row!(needle, haystack, want));
    }
}

// ---------------------------------------------------------------------------
// Randomized: every backend must match its lane-matched scalar reference over
// arbitrary inputs. Runs under proptest normally, and over a small fixed
// corpus under miri.
// ---------------------------------------------------------------------------

fn score_bytes_with<B: Backend>(needle: &str, haystack: &[u8], case_sensitive: bool) -> u16 {
    let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), case_sensitive);
    matcher.score_haystack(haystack)
}

fn indices_bytes_with<B: Backend>(
    needle: &str,
    haystack: &[u8],
    max_typos: Option<u16>,
    case_sensitive: bool,
) -> Option<(u16, Vec<usize>)> {
    let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), case_sensitive);
    matcher.score_haystack_indices(haystack, 0, max_typos)
}

fn assert_backend<B: Backend>(
    label: &str,
    needle: &str,
    haystack: &[u8],
    max_typos: Option<u16>,
    case_sensitive: bool,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    if B::is_available() {
        assert_eq!(
            score_bytes_with::<B>(needle, haystack, case_sensitive),
            want_score,
            "{label} score mismatch for needle={needle:?} haystack_len={}",
            haystack.len()
        );
        let indices = indices_bytes_with::<B>(needle, haystack, max_typos, case_sensitive);
        assert_eq!(
            indices.as_ref().map(|(score, _)| *score),
            want_indices_score,
            "{label} indexed score mismatch for needle={needle:?} haystack_len={}",
            haystack.len()
        );
        if let Some((_, indices)) = indices {
            assert_indices_valid(label, needle, haystack, &indices);
        }
    }
}

fn assert_backend_matches_reference<B: Backend, R: Backend>(
    label: &str,
    needle: &str,
    haystack: &[u8],
    max_typos: Option<u16>,
    case_sensitive: bool,
) {
    if B::is_available() {
        let want_score = score_bytes_with::<R>(needle, haystack, case_sensitive);
        let want_indices_score =
            indices_bytes_with::<R>(needle, haystack, max_typos, case_sensitive)
                .as_ref()
                .map(|(score, _)| *score);
        assert_backend::<B>(
            label,
            needle,
            haystack,
            max_typos,
            case_sensitive,
            want_score,
            want_indices_score,
        );
    }
}

/// Structural invariants that hold for any backend's returned indices.
fn assert_indices_valid(label: &str, needle: &str, haystack: &[u8], indices: &[usize]) {
    assert!(
        indices.windows(2).all(|window| window[0] > window[1]),
        "{} indices are not in reverse order: {:?}",
        label,
        indices
    );
    assert!(
        indices.len() <= needle.len(),
        "{} indices contain more positions than needle bytes: indices={:?} needle_len={}",
        label,
        indices,
        needle.len()
    );
    for &index in indices {
        assert!(
            index < haystack.len(),
            "{} index {} is out of bounds for haystack_len={}",
            label,
            index,
            haystack.len()
        );
    }
}

/// Per-row body for [`assert_backend_case`], using each backend's lane-matched
/// reference oracle.
macro_rules! assert_case_row {
    ($case:expr ;
     $cfg:meta, $label:literal, $backend:ty, $reference:ty, $is_u8:literal) => {
        #[cfg($cfg)]
        {
            if !$is_u8 || score_fits_in_u8($case.needle.len(), &Scoring::default()) {
                assert_backend_matches_reference::<$backend, $reference>(
                    $label,
                    &$case.needle,
                    &$case.haystack,
                    $case.max_typos,
                    $case.case_sensitive,
                );
            }
        }
    };
}

fn assert_backend_case(case: &BackendCase) {
    for_each_backend!(assert_case_row!(case));
}

#[test]
fn randomized_cross_backend_parity() {
    if cfg!(miri) {
        for input in miri_inputs() {
            assert_backend_case(&BackendCase::from_bytes(input));
        }
        return;
    }

    run_generated_inputs(192, test_bound(2048, 384), |input| {
        assert_backend_case(&BackendCase::from_bytes(input));
    });
}
