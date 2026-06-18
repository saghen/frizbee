use std::collections::BTreeMap;

use crate::prefilter::{Kernel as PrefilterKernel, Window, backend::PrefilterScalar};
use crate::smith_waterman::{
    Kernel as SmithWatermanKernel, SmithWaterman,
    backend::{Backend, BackendScalar8, BackendScalar16U8},
    score_fits_in_u8,
};
use crate::{
    CaseMatching, Config, Match, MatchIndices, Matcher, Scoring, match_list, match_list_indices,
    match_list_parallel,
};

#[derive(Debug, Clone)]
pub struct ApiCase {
    needle: String,
    haystacks: Vec<String>,
    config: Config,
}

impl ApiCase {
    pub fn from_bytes(input: &[u8]) -> Self {
        Self::from_bytes_with_limits(input, 32, 32, 96)
    }

    pub fn from_long_bytes(input: &[u8]) -> Self {
        Self::from_bytes_with_limits(input, 96, 8, 768)
    }

    fn from_bytes_with_limits(
        input: &[u8],
        max_needle_len: usize,
        max_haystack_count: usize,
        max_haystack_len: usize,
    ) -> Self {
        let mut cursor = ByteCursor::new(input);
        let needle_len = cursor.len(
            max_needle_len,
            &[
                0,
                1,
                2,
                7,
                8,
                15,
                16,
                31,
                32,
                63.min(max_needle_len),
                max_needle_len,
            ],
        );
        let haystack_count = cursor.len(max_haystack_count, &[0, 1, 2, max_haystack_count]);
        let haystacks = (0..haystack_count)
            .map(|_| {
                let len = cursor.len(
                    max_haystack_len,
                    &[
                        0,
                        1,
                        7,
                        8,
                        15,
                        16,
                        31,
                        32,
                        63,
                        64,
                        95,
                        96,
                        511.min(max_haystack_len),
                        512.min(max_haystack_len),
                        513.min(max_haystack_len),
                        max_haystack_len,
                    ],
                );
                cursor.string(len)
            })
            .collect();

        let max_typos = match cursor.next() % 5 {
            0 => None,
            1 => Some(0),
            2 => Some(1),
            3 => Some(2),
            _ => Some((cursor.next() as u16) % 8),
        };
        let casing = match cursor.next() % 3 {
            0 => CaseMatching::Ignore,
            1 => CaseMatching::Smart,
            _ => CaseMatching::Respect,
        };

        Self {
            needle: cursor.string(needle_len),
            haystacks,
            config: Config {
                max_typos,
                casing,
                sort: cursor.bool(),
                ..Config::default()
            },
        }
    }
}

pub fn assert_public_api(input: &[u8]) {
    assert_public_api_case(&ApiCase::from_bytes(input));
}

pub fn assert_public_api_long_inputs(input: &[u8]) {
    assert_public_api_case(&ApiCase::from_long_bytes(input));
}

pub fn assert_public_api_case(case: &ApiCase) {
    let one_shot = match_list(&case.needle, &case.haystacks, &case.config);
    let mut matcher = Matcher::new(&case.needle, &case.config);
    let reusable = matcher.match_list(&case.haystacks);
    assert_match_views_eq("Matcher::match_list", &reusable, &one_shot);

    let one_shot_indices = match_list_indices(&case.needle, &case.haystacks, &case.config);
    let mut matcher = Matcher::new(&case.needle, &case.config);
    let reusable_indices = matcher.match_list_indices(&case.haystacks);
    assert_eq!(
        indices_views(&reusable_indices),
        indices_views(&one_shot_indices),
        "Matcher::match_list_indices mismatch for {case:?}"
    );

    let parallel_one = match_list_parallel(&case.needle, &case.haystacks, &case.config, 1);
    assert_match_views_eq("parallel threads=1", &parallel_one, &one_shot);

    for threads in [2, 3, 8] {
        let parallel = match_list_parallel(&case.needle, &case.haystacks, &case.config, threads);
        if case.config.sort {
            assert_match_views_eq("parallel sorted", &parallel, &one_shot);
        } else {
            assert_eq!(
                sorted_match_views(&parallel),
                sorted_match_views(&one_shot),
                "parallel unsorted multiset mismatch for threads={threads} case={case:?}"
            );
        }
    }

    assert_indices_contract(case, &one_shot, &one_shot_indices);
}

fn assert_indices_contract(case: &ApiCase, matches: &[Match], indices: &[MatchIndices]) {
    let match_set = matches
        .iter()
        .map(|match_| (match_.index, (match_.score, match_.exact)))
        .collect::<BTreeMap<_, _>>();
    let indices_set = indices
        .iter()
        .map(|match_| (match_.index, (match_.score, match_.exact)))
        .collect::<BTreeMap<_, _>>();

    for match_ in indices {
        assert!(
            (match_.index as usize) < case.haystacks.len(),
            "index {} is out of bounds for {case:?}",
            match_.index
        );
        assert_eq!(
            match_set.get(&match_.index),
            Some(&(match_.score, match_.exact)),
            "indices result is not present in match_list for {case:?}"
        );

        let haystack = case.haystacks[match_.index as usize].as_bytes();
        assert!(
            match_
                .indices
                .windows(2)
                .all(|window| window[0] > window[1]),
            "indices are not reverse ordered for {case:?}: {:?}",
            match_.indices
        );
        assert!(
            match_.indices.len() <= case.needle.len(),
            "too many indices for {case:?}: {:?}",
            match_.indices
        );
        for &index in &match_.indices {
            assert!(
                index < haystack.len(),
                "index {index} out of bounds for haystack len {} in {case:?}",
                haystack.len()
            );
        }
    }

    if case.config.max_typos.is_none()
        && case.haystacks.iter().all(|haystack| haystack.len() <= 512)
    {
        assert_eq!(
            indices_set, match_set,
            "indices and matches should agree exactly without typo filtering for {case:?}"
        );
    }
}

#[cfg(not(feature = "match_end_col"))]
type MatchView = (u16, u32, bool);

#[cfg(feature = "match_end_col")]
type MatchView = (u16, u32, bool, u16);

#[cfg(not(feature = "match_end_col"))]
fn match_view(match_: &Match) -> MatchView {
    (match_.score, match_.index, match_.exact)
}

#[cfg(feature = "match_end_col")]
fn match_view(match_: &Match) -> MatchView {
    (match_.score, match_.index, match_.exact, match_.end_col)
}

fn match_views(matches: &[Match]) -> Vec<MatchView> {
    matches.iter().map(match_view).collect()
}

fn sorted_match_views(matches: &[Match]) -> Vec<MatchView> {
    let mut views = match_views(matches);
    views.sort();
    views
}

fn assert_match_views_eq(label: &str, got: &[Match], want: &[Match]) {
    assert_eq!(
        match_views(got),
        match_views(want),
        "{label} mismatch: got={got:?} want={want:?}"
    );
}

fn indices_views(matches: &[MatchIndices]) -> Vec<(u16, u32, bool, Vec<usize>)> {
    matches
        .iter()
        .map(|match_| {
            (
                match_.score,
                match_.index,
                match_.exact,
                match_.indices.clone(),
            )
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct PrefilterCase {
    needle: Vec<u8>,
    haystack: Vec<u8>,
    max_typos: u16,
    case_sensitive: bool,
}

impl PrefilterCase {
    pub fn from_bytes(input: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(input);
        let needle_len = cursor.len(96, &[1, 7, 8, 15, 16, 31, 32, 63, 64]).max(1);
        let haystack_len = cursor.len(768, &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 511, 512, 513]);
        let max_typos = (cursor.next() as u16) % 17;
        let case_sensitive = cursor.bool();

        Self {
            needle: cursor.bytes(needle_len),
            haystack: cursor.bytes(haystack_len),
            max_typos,
            case_sensitive,
        }
    }
}

pub fn assert_prefilter(input: &[u8]) {
    assert_prefilter_case(&PrefilterCase::from_bytes(input));
}

pub fn assert_prefilter_case(case: &PrefilterCase) {
    let scalar_result = prefilter_result::<PrefilterScalar>(
        &case.needle,
        &case.haystack,
        case.max_typos,
        case.case_sensitive,
    );
    let oracle = reference_matches_by_deleting_needle_bytes(
        &case.needle,
        &case.haystack,
        case.max_typos,
        case.case_sensitive,
    );
    assert_eq!(
        scalar_result.0, oracle,
        "scalar/oracle mismatch for {case:?}"
    );
    assert_valid_window(scalar_result, &case.haystack, "Scalar", case);

    #[cfg(target_arch = "x86_64")]
    {
        use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};

        if PrefilterSSE::is_available() {
            let result = prefilter_result::<PrefilterSSE>(
                &case.needle,
                &case.haystack,
                case.max_typos,
                case.case_sensitive,
            );
            assert_same_prefilter_result(result, scalar_result, "SSE", case);
            assert_valid_window(result, &case.haystack, "SSE", case);
        }

        if PrefilterAVX::is_available() {
            let result = prefilter_result::<PrefilterAVX>(
                &case.needle,
                &case.haystack,
                case.max_typos,
                case.case_sensitive,
            );
            assert_same_prefilter_result(result, scalar_result, "AVX2", case);
            assert_valid_window(result, &case.haystack, "AVX2", case);
        }

        if PrefilterAVX512::is_available() {
            let result = prefilter_result::<PrefilterAVX512>(
                &case.needle,
                &case.haystack,
                case.max_typos,
                case.case_sensitive,
            );
            assert_same_prefilter_result(result, scalar_result, "AVX-512", case);
            assert_valid_window(result, &case.haystack, "AVX-512", case);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::prefilter::backend::PrefilterNEON;

        if PrefilterNEON::is_available() {
            let result = prefilter_result::<PrefilterNEON>(
                &case.needle,
                &case.haystack,
                case.max_typos,
                case.case_sensitive,
            );
            assert_same_prefilter_result(result, scalar_result, "NEON", case);
            assert_valid_window(result, &case.haystack, "NEON", case);
        }
    }
}

fn prefilter_result<P: PrefilterKernel>(
    needle: &[u8],
    haystack: &[u8],
    max_typos: u16,
    case_sensitive: bool,
) -> Window {
    let mut prefilter = P::new(needle, case_sensitive);
    match max_typos {
        0 => prefilter.match_haystack(haystack),
        1 => prefilter.match_haystack_1_typo(haystack),
        2 => prefilter.match_haystack_2_typos(haystack),
        _ => prefilter.match_haystack_many_typos(haystack, max_typos),
    }
}

fn reference_matches_by_deleting_needle_bytes(
    needle: &[u8],
    haystack: &[u8],
    max_typos: u16,
    case_sensitive: bool,
) -> bool {
    if max_typos as usize >= needle.len() {
        return true;
    }

    longest_common_subsequence_len(needle, haystack, case_sensitive) + max_typos as usize
        >= needle.len()
}

fn longest_common_subsequence_len(needle: &[u8], haystack: &[u8], case_sensitive: bool) -> usize {
    let mut previous = vec![0usize; haystack.len() + 1];
    let mut current = vec![0usize; haystack.len() + 1];
    for &needle_byte in needle {
        current[0] = 0;
        for (idx, &haystack_byte) in haystack.iter().enumerate() {
            current[idx + 1] = if bytes_match(needle_byte, haystack_byte, case_sensitive) {
                previous[idx] + 1
            } else {
                previous[idx + 1].max(current[idx])
            };
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[haystack.len()]
}

fn bytes_match(needle: u8, haystack: u8, case_sensitive: bool) -> bool {
    needle == haystack || (!case_sensitive && needle.eq_ignore_ascii_case(&haystack))
}

fn assert_valid_window(result: Window, haystack: &[u8], context: &str, case: &PrefilterCase) {
    if !result.0 {
        return;
    }

    assert!(
        result.1 <= result.2 && result.2 <= haystack.len(),
        "{} returned invalid window {:?} for haystack_len={} case={:?}",
        context,
        result,
        haystack.len(),
        case
    );
}

fn assert_same_prefilter_result(got: Window, want: Window, context: &str, case: &PrefilterCase) {
    if want.0 {
        assert_eq!(got, want, "{context} mismatch for {case:?}");
    } else {
        assert_eq!(got.0, want.0, "{context} mismatch for {case:?}");
    }
}

#[derive(Debug, Clone)]
pub struct SmithWatermanCase {
    needle: Vec<u8>,
    haystack: Vec<u8>,
    max_typos: Option<u16>,
    case_sensitive: bool,
}

impl SmithWatermanCase {
    pub fn from_bytes(input: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(input);
        let needle_len = cursor.len(96, &[1, 7, 8, 15, 16, 31, 32, 63, 64]).max(1);
        let haystack_len = cursor.len(768, &[0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 511, 512, 513]);
        let max_typos = match cursor.next() % 5 {
            0 => None,
            byte => Some((byte as u16 - 1) % 17),
        };
        let case_sensitive = cursor.bool();

        Self {
            needle: cursor.ascii_bytes(needle_len),
            haystack: cursor.ascii_bytes(haystack_len),
            max_typos,
            case_sensitive,
        }
    }
}

pub fn assert_smith_waterman(input: &[u8]) {
    assert_smith_waterman_case(&SmithWatermanCase::from_bytes(input));
}

pub fn assert_smith_waterman_case(case: &SmithWatermanCase) {
    let needle = &case.needle;
    let haystack = &case.haystack;

    let scalar_score = score_with::<BackendScalar8>(needle, haystack, case.case_sensitive);
    let scalar_indices_score =
        indices_with::<BackendScalar8>(needle, haystack, case.max_typos, case.case_sensitive)
            .as_ref()
            .map(|(score, _)| *score);

    assert_smith_waterman_backend_matches::<BackendScalar8>(
        "scalar-u16",
        case,
        scalar_score,
        scalar_indices_score,
    );

    if score_fits_in_u8(needle.len(), &Scoring::default()) {
        let scalar_u8_score =
            score_with::<BackendScalar16U8>(needle, haystack, case.case_sensitive);
        let scalar_u8_indices_score = indices_with::<BackendScalar16U8>(
            needle,
            haystack,
            case.max_typos,
            case.case_sensitive,
        )
        .as_ref()
        .map(|(score, _)| *score);
        assert_smith_waterman_backend_matches::<BackendScalar16U8>(
            "scalar-u8",
            case,
            scalar_u8_score,
            scalar_u8_indices_score,
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        use crate::smith_waterman::backend::{
            BackendAVX, BackendAVX512, BackendAVX512U8, BackendAVXU8, BackendSSE, BackendSSEU8,
        };

        if BackendSSE::is_available() {
            unsafe {
                assert_sse_backend(case, scalar_score, scalar_indices_score);
            }
        }
        if BackendAVX::is_available() {
            unsafe {
                assert_avx_backend(case, scalar_score, scalar_indices_score);
            }
        }
        if BackendAVX512::is_available() {
            unsafe {
                assert_avx512_backend(case, scalar_score, scalar_indices_score);
            }
        }

        if score_fits_in_u8(needle.len(), &Scoring::default()) {
            let scalar_u8_score =
                score_with::<BackendScalar16U8>(needle, haystack, case.case_sensitive);
            let scalar_u8_indices_score = indices_with::<BackendScalar16U8>(
                needle,
                haystack,
                case.max_typos,
                case.case_sensitive,
            )
            .as_ref()
            .map(|(score, _)| *score);

            if BackendSSEU8::is_available() {
                unsafe {
                    assert_sse_u8_backend(case, scalar_u8_score, scalar_u8_indices_score);
                }
            }
            if BackendAVXU8::is_available() {
                unsafe {
                    assert_avx_u8_backend(case, scalar_u8_score, scalar_u8_indices_score);
                }
            }
            if BackendAVX512U8::is_available() {
                unsafe {
                    assert_avx512_u8_backend(case, scalar_u8_score, scalar_u8_indices_score);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::smith_waterman::backend::{BackendNEON, BackendNEONU8};

        if BackendNEON::is_available() {
            unsafe {
                assert_neon_backend(case, scalar_score, scalar_indices_score);
            }
        }

        if score_fits_in_u8(needle.len(), &Scoring::default()) {
            let scalar_u8_score =
                score_with::<BackendScalar16U8>(needle, haystack, case.case_sensitive);
            let scalar_u8_indices_score = indices_with::<BackendScalar16U8>(
                needle,
                haystack,
                case.max_typos,
                case.case_sensitive,
            )
            .as_ref()
            .map(|(score, _)| *score);

            if BackendNEONU8::is_available() {
                unsafe {
                    assert_neon_u8_backend(case, scalar_u8_score, scalar_u8_indices_score);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2,ssse3,sse4.1")]
unsafe fn assert_sse_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendSSE;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendSSE>("SSE-u16", case);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2,ssse3,sse4.1")]
unsafe fn assert_sse_u8_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendSSEU8;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendSSEU8>("SSE-u8", case);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn assert_avx_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendAVX;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendAVX>("AVX-u16", case);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn assert_avx_u8_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendAVXU8;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendAVXU8>("AVX-u8", case);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,bmi1,bmi2")]
unsafe fn assert_avx512_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendAVX512;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendAVX512>("AVX-512-u16", case);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,bmi1,bmi2")]
unsafe fn assert_avx512_u8_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendAVX512U8;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendAVX512U8>("AVX-512-u8", case);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn assert_neon_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendNEON;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendNEON>("NEON-u16", case);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn assert_neon_u8_backend(
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    use crate::smith_waterman::backend::BackendNEONU8;

    let _ = (want_score, want_indices_score);
    assert_smith_waterman_backend_valid::<BackendNEONU8>("NEON-u8", case);
}

#[inline(always)]
fn assert_smith_waterman_backend_matches<B: Backend>(
    label: &str,
    case: &SmithWatermanCase,
    want_score: u16,
    want_indices_score: Option<u16>,
) {
    if !B::is_available() {
        return;
    }

    let score = score_with::<B>(&case.needle, &case.haystack, case.case_sensitive);
    assert_eq!(score, want_score, "{label} score mismatch for {case:?}");

    let indices = indices_with::<B>(
        &case.needle,
        &case.haystack,
        case.max_typos,
        case.case_sensitive,
    );
    assert_eq!(
        indices.as_ref().map(|(score, _)| *score),
        want_indices_score,
        "{label} indexed score mismatch for {case:?}"
    );
    if let Some((_, indices)) = indices {
        assert_indices_valid(label, &case.needle, &case.haystack, &indices);
    }
}

#[inline(always)]
fn assert_smith_waterman_backend_valid<B: Backend>(label: &str, case: &SmithWatermanCase) {
    if !B::is_available() {
        return;
    }

    let score = score_with::<B>(&case.needle, &case.haystack, case.case_sensitive);
    let indices = indices_with::<B>(
        &case.needle,
        &case.haystack,
        case.max_typos,
        case.case_sensitive,
    );

    if let Some((indices_score, indices)) = indices {
        assert_eq!(
            indices_score, score,
            "{label} score/indexed score mismatch for {case:?}"
        );
        assert_indices_valid(label, &case.needle, &case.haystack, &indices);
    }
}

#[inline(always)]
fn score_with<B: Backend>(needle: &[u8], haystack: &[u8], case_sensitive: bool) -> u16 {
    let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), case_sensitive);
    matcher.score_haystack(haystack)
}

#[inline(always)]
fn indices_with<B: Backend>(
    needle: &[u8],
    haystack: &[u8],
    max_typos: Option<u16>,
    case_sensitive: bool,
) -> Option<(u16, Vec<usize>)> {
    let mut matcher = SmithWaterman::<B>::new(needle, &Scoring::default(), case_sensitive);
    matcher.match_haystack_indices(haystack, 0, max_typos)
}

fn assert_indices_valid(label: &str, needle: &[u8], haystack: &[u8], indices: &[usize]) {
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
            (self.pos as u8).wrapping_mul(31).wrapping_add(13)
        } else {
            self.input[self.pos % self.input.len()]
                .wrapping_add(((self.pos / self.input.len()) as u8).wrapping_mul(23))
        };
        self.pos += 1;
        byte
    }

    fn bool(&mut self) -> bool {
        self.next() & 1 == 1
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

    fn string(&mut self, len: usize) -> String {
        (0..len).map(|_| self.char()).collect()
    }

    fn char(&mut self) -> char {
        let byte = self.next();
        match byte % 18 {
            0 => 'a',
            1 => ' ',
            2 => '/',
            3 => '.',
            4 => ',',
            5 => '_',
            6 => '-',
            7 => ':',
            8..=11 => (b'a' + (byte % 26)) as char,
            12..=15 => (b'A' + (byte % 26)) as char,
            _ => (b'0' + (byte % 10)) as char,
        }
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.byte()).collect()
    }

    fn ascii_bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.ascii_byte()).collect()
    }

    fn ascii_byte(&mut self) -> u8 {
        let byte = self.next();
        match byte % 16 {
            0 => b'a',
            1 => b' ',
            2 => b'/',
            3 => b'.',
            4 => b',',
            5 => b'_',
            6 => b'-',
            7 => b':',
            8..=10 => b'a' + (byte % 26),
            11..=13 => b'A' + (byte % 26),
            _ => b'0' + (byte % 10),
        }
    }

    fn byte(&mut self) -> u8 {
        let byte = self.next();
        match byte % 18 {
            0 => 0,
            1 => b' ',
            2 => b'/',
            3 => b'.',
            4 => b',',
            5 => b'_',
            6 => b'-',
            7 => b':',
            8..=10 => b'a' + (byte % 26),
            11..=13 => b'A' + (byte % 26),
            14..=15 => b'0' + (byte % 10),
            16 => 0x80 | (byte & 0x3f),
            _ => byte,
        }
    }
}
