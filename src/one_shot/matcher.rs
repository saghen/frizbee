use crate::prefilter::Kernel as PrefilterKernel;
#[cfg(target_arch = "aarch64")]
use crate::prefilter::backend::PrefilterNEON;
use crate::prefilter::backend::PrefilterScalar;
#[cfg(target_arch = "x86_64")]
use crate::prefilter::backend::{PrefilterAVX, PrefilterAVX512, PrefilterSSE};
use crate::smith_waterman::AlignmentPathIter;
use crate::smith_waterman::simd::{
    Kernel as SmithWatermanKernel, SmithWatermanMatcherScalar, SmithWatermanMatcherScalarU8,
    score_fits_in_u8,
};
#[cfg(target_arch = "x86_64")]
use crate::smith_waterman::simd::{
    SmithWatermanMatcherAVX2, SmithWatermanMatcherAVX2U8, SmithWatermanMatcherAVX512,
    SmithWatermanMatcherAVX512U8, SmithWatermanMatcherSSE, SmithWatermanMatcherSSEU8,
};
#[cfg(target_arch = "aarch64")]
use crate::smith_waterman::simd::{SmithWatermanMatcherNEON, SmithWatermanMatcherNEONU8};
use crate::sort::radix_sort_matches;
use crate::{Config, Match, MatchIndices};
use std::marker::PhantomData;

trait MatcherBackend: Clone + std::fmt::Debug + 'static {
    type Prefilter: PrefilterKernel;
    type SmithWaterman: SmithWatermanKernel;

    fn is_available(needle: &[u8], config: &Config) -> bool;
}

macro_rules! define_backend {
    ($name:ident, $prefilter:ty, $smith_waterman:ty, $available:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name;

        impl MatcherBackend for $name {
            type Prefilter = $prefilter;
            type SmithWaterman = $smith_waterman;

            #[inline]
            fn is_available(needle: &[u8], config: &Config) -> bool {
                $available(needle, config)
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
define_backend!(
    Avx512U8,
    PrefilterAVX512,
    SmithWatermanMatcherAVX512U8,
    |needle: &[u8], config: &Config| {
        PrefilterAVX512::is_available()
            && SmithWatermanMatcherAVX512U8::is_available()
            && score_fits_in_u8(needle.len(), &config.scoring)
    }
);

#[cfg(target_arch = "x86_64")]
define_backend!(
    Avx512,
    PrefilterAVX512,
    SmithWatermanMatcherAVX512,
    |_needle: &[u8], _config: &Config| {
        PrefilterAVX512::is_available() && SmithWatermanMatcherAVX512::is_available()
    }
);

#[cfg(target_arch = "x86_64")]
define_backend!(
    Avx2U8,
    PrefilterAVX,
    SmithWatermanMatcherAVX2U8,
    |needle: &[u8], config: &Config| {
        PrefilterAVX::is_available()
            && SmithWatermanMatcherAVX2U8::is_available()
            && score_fits_in_u8(needle.len(), &config.scoring)
    }
);

#[cfg(target_arch = "x86_64")]
define_backend!(
    Avx2,
    PrefilterAVX,
    SmithWatermanMatcherAVX2,
    |_needle: &[u8], _config: &Config| {
        PrefilterAVX::is_available() && SmithWatermanMatcherAVX2::is_available()
    }
);

#[cfg(target_arch = "x86_64")]
define_backend!(
    SseU8,
    PrefilterSSE,
    SmithWatermanMatcherSSEU8,
    |needle: &[u8], config: &Config| {
        PrefilterSSE::is_available()
            && SmithWatermanMatcherSSEU8::is_available()
            && score_fits_in_u8(needle.len(), &config.scoring)
    }
);

#[cfg(target_arch = "x86_64")]
define_backend!(
    Sse,
    PrefilterSSE,
    SmithWatermanMatcherSSE,
    |_needle: &[u8], _config: &Config| {
        PrefilterSSE::is_available() && SmithWatermanMatcherSSE::is_available()
    }
);

#[cfg(target_arch = "aarch64")]
define_backend!(
    NeonU8,
    PrefilterNEON,
    SmithWatermanMatcherNEONU8,
    |needle: &[u8], config: &Config| {
        PrefilterNEON::is_available()
            && SmithWatermanMatcherNEONU8::is_available()
            && score_fits_in_u8(needle.len(), &config.scoring)
    }
);

#[cfg(target_arch = "aarch64")]
define_backend!(
    Neon,
    PrefilterNEON,
    SmithWatermanMatcherNEON,
    |_needle: &[u8], _config: &Config| {
        PrefilterNEON::is_available() && SmithWatermanMatcherNEON::is_available()
    }
);

define_backend!(
    ScalarU8,
    PrefilterScalar,
    SmithWatermanMatcherScalarU8,
    |needle: &[u8], config: &Config| score_fits_in_u8(needle.len(), &config.scoring)
);

define_backend!(
    Scalar,
    PrefilterScalar,
    SmithWatermanMatcherScalar,
    |_needle: &[u8], _config: &Config| true
);

#[derive(Debug, Clone)]
struct MatcherCore<B: MatcherBackend> {
    needle: String,
    config: Config,
    prefilter: B::Prefilter,
    smith_waterman: B::SmithWaterman,
    _backend: PhantomData<B>,
}

#[allow(private_bounds)]
#[derive(Debug, Clone)]
pub struct BackendMatcher<B: MatcherBackend> {
    core: MatcherCore<B>,
}

#[derive(Debug, Clone)]
pub enum Matcher {
    #[cfg(target_arch = "x86_64")]
    AVX512U8(BackendMatcher<Avx512U8>),
    #[cfg(target_arch = "x86_64")]
    AVX512(BackendMatcher<Avx512>),
    #[cfg(target_arch = "x86_64")]
    AVX2U8(BackendMatcher<Avx2U8>),
    #[cfg(target_arch = "x86_64")]
    AVX2(BackendMatcher<Avx2>),
    #[cfg(target_arch = "x86_64")]
    SSEU8(BackendMatcher<SseU8>),
    #[cfg(target_arch = "x86_64")]
    SSE(BackendMatcher<Sse>),
    #[cfg(target_arch = "aarch64")]
    NEONU8(BackendMatcher<NeonU8>),
    #[cfg(target_arch = "aarch64")]
    NEON(BackendMatcher<Neon>),
    ScalarU8(BackendMatcher<ScalarU8>),
    Scalar(BackendMatcher<Scalar>),
}

macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            #[cfg(target_arch = "x86_64")]
            Self::AVX512U8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            Self::AVX512($m) => $body,
            #[cfg(target_arch = "x86_64")]
            Self::AVX2U8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            Self::AVX2($m) => $body,
            #[cfg(target_arch = "x86_64")]
            Self::SSEU8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            Self::SSE($m) => $body,
            #[cfg(target_arch = "aarch64")]
            Self::NEONU8($m) => $body,
            #[cfg(target_arch = "aarch64")]
            Self::NEON($m) => $body,
            Self::ScalarU8($m) => $body,
            Self::Scalar($m) => $body,
        }
    };
}

#[allow(private_bounds)]
impl<B: MatcherBackend> BackendMatcher<B> {
    fn new(needle: &str, config: &Config) -> Self {
        Self {
            core: MatcherCore::new(needle, config),
        }
    }
}

impl<B: MatcherBackend> MatcherCore<B> {
    fn new(needle: &str, config: &Config) -> Self {
        let matcher = Self {
            needle: needle.to_string(),
            config: config.clone(),
            prefilter: B::Prefilter::new(needle.as_bytes()),
            smith_waterman: B::SmithWaterman::new(needle.as_bytes(), &config.scoring),
            _backend: PhantomData,
        };
        matcher.guard_against_score_overflow();
        matcher
    }

    #[inline(always)]
    fn needle(&self) -> &str {
        &self.needle
    }

    #[inline(always)]
    fn config(&self) -> &Config {
        &self.config
    }

    fn match_list<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        if self.needle.is_empty() {
            return (0..haystacks.len())
                .map(|index| Match {
                    index: index as u32,
                    score: 0,
                    exact: false,
                    #[cfg(feature = "match_end_col")]
                    end_col: 0,
                })
                .collect();
        }

        let mut matches = vec![];
        self.match_list_into(haystacks, 0, &mut matches);

        if self.config.sort {
            radix_sort_matches(&mut matches);
        }

        matches
    }

    fn match_list_indices<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<MatchIndices> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        if self.needle.is_empty() {
            return (0..haystacks.len()).map(MatchIndices::from_index).collect();
        }

        let mut matches = vec![];
        self.match_list_indices_into(haystacks, 0, &mut matches);

        if self.config.sort {
            matches.sort_unstable();
        }

        matches
    }

    fn match_list_into<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        Matcher::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);

        if self.needle.is_empty() {
            for index in (0..haystacks.len()).map(|i| i + haystack_index_offset as usize) {
                matches.push(Match::from_index(index));
            }
            return;
        }

        let min_haystack_len = self.min_haystack_len();
        match self.config.max_typos {
            None => self.match_list_into_unfiltered(haystacks, haystack_index_offset, matches),
            Some(0) => self.match_list_into_prefiltered_0(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                matches,
            ),
            Some(1) => self.match_list_into_prefiltered_1(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                matches,
            ),
            Some(2) => self.match_list_into_prefiltered_2(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                matches,
            ),
            Some(max_typos) => self.match_list_into_prefiltered_many(
                haystacks,
                haystack_index_offset,
                min_haystack_len,
                max_typos,
                matches,
            ),
        }
    }

    #[inline(always)]
    fn match_list_into_unfiltered<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            matches.push(self.smith_waterman_one(haystack_str.as_ref().as_bytes(), index, true));
            index += 1;
        }
    }

    #[inline(always)]
    fn match_list_into_prefiltered_0<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        min_haystack_len: usize,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) = self.prefilter.match_0(haystack);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    matches.push(self.smith_waterman_one(trimmed, index, include_exact));
                }
            }
            index += 1;
        }
    }

    #[inline(always)]
    fn match_list_into_prefiltered_1<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        min_haystack_len: usize,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) = self.prefilter.match_1(haystack);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    matches.push(self.smith_waterman_one(trimmed, index, include_exact));
                }
            }
            index += 1;
        }
    }

    #[inline(always)]
    fn match_list_into_prefiltered_2<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        min_haystack_len: usize,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) = self.prefilter.match_2(haystack);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    matches.push(self.smith_waterman_one(trimmed, index, include_exact));
                }
            }
            index += 1;
        }
    }

    #[inline(always)]
    fn match_list_into_prefiltered_many<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        min_haystack_len: usize,
        max_typos: u16,
        matches: &mut Vec<Match>,
    ) {
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) = self.prefilter.match_many(haystack, max_typos);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    matches.push(self.smith_waterman_one(trimmed, index, include_exact));
                }
            }
            index += 1;
        }
    }

    fn match_list_indices_into<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<MatchIndices>,
    ) {
        Matcher::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);

        if self.needle.is_empty() {
            for index in (0..haystacks.len()).map(|i| i + haystack_index_offset as usize) {
                matches.push(MatchIndices::from_index(index));
            }
            return;
        }

        let min_haystack_len = self.min_haystack_len();
        let mut index = haystack_index_offset;
        for haystack_str in haystacks {
            let haystack = haystack_str.as_ref().as_bytes();
            let original_len = haystack.len();
            if original_len >= min_haystack_len {
                let (matched, start_pos, end_pos) = self.prefilter_window(haystack);
                if matched {
                    let trimmed = &haystack[start_pos..end_pos];
                    let include_exact = start_pos == 0 && end_pos == original_len;
                    if let Some(match_) =
                        self.smith_waterman_indices_one(trimmed, start_pos, index, include_exact)
                    {
                        matches.push(match_);
                    }
                }
            }
            index += 1;
        }
    }

    fn match_iter<'a, S: AsRef<str> + 'a>(
        &'a mut self,
        haystacks: &'a [S],
    ) -> impl Iterator<Item = Match> + 'a {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        self.prefilter_iter(haystacks)
            .map(|(index, haystack, _skipped_chars, is_full_haystack)| {
                self.smith_waterman_one(haystack, index as u32, is_full_haystack)
            })
    }

    fn match_iter_indices<'a, S: AsRef<str> + 'a>(
        &'a mut self,
        haystacks: &'a [S],
    ) -> impl Iterator<Item = MatchIndices> + 'a {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        self.prefilter_iter(haystacks).filter_map(
            |(index, haystack, skipped_chars, is_full_haystack)| {
                self.smith_waterman_indices_one(
                    haystack,
                    skipped_chars,
                    index as u32,
                    is_full_haystack,
                )
            },
        )
    }

    #[inline(always)]
    fn smith_waterman_one(&mut self, haystack: &[u8], index: u32, include_exact: bool) -> Match {
        let mut score = self.smith_waterman.score_haystack(haystack);

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        Match {
            index,
            score,
            exact,
            #[cfg(feature = "match_end_col")]
            end_col: self.smith_waterman.match_end_col(haystack),
        }
    }

    #[inline(always)]
    fn smith_waterman_indices_one(
        &mut self,
        haystack: &[u8],
        skipped_chars: usize,
        index: u32,
        include_exact: bool,
    ) -> Option<MatchIndices> {
        let (mut score, indices) = self.smith_waterman.match_haystack_indices(
            haystack,
            skipped_chars,
            self.config.max_typos,
        )?;

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        Some(MatchIndices {
            index,
            score,
            exact,
            indices,
        })
    }

    fn prefilter_iter<'a, S: AsRef<str> + 'a>(
        &self,
        haystacks: &'a [S],
    ) -> impl Iterator<Item = (usize, &'a [u8], usize, bool)> + use<'a, S, B> {
        assert!(!self.needle.is_empty(), "needle must not be empty");

        let min_haystack_len = self.min_haystack_len();
        let max_typos = self.config.max_typos;
        let mut prefilter = self.prefilter.clone();

        haystacks
            .iter()
            .map(|h| h.as_ref().as_bytes())
            .enumerate()
            .filter(move |(_, h)| h.len() >= min_haystack_len)
            .filter_map(move |(i, haystack)| {
                let original_len = haystack.len();
                let (matched, skipped_chars, end_pos) = max_typos
                    .map_or((true, 0, original_len), |max_typos| {
                        prefilter.match_haystack(haystack, max_typos)
                    });
                let is_full_haystack = skipped_chars == 0 && end_pos == original_len;
                matched.then(|| {
                    (
                        i,
                        &haystack[skipped_chars..end_pos],
                        skipped_chars,
                        is_full_haystack,
                    )
                })
            })
    }

    #[inline(always)]
    fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        self.smith_waterman.score_haystack(haystack)
    }

    #[inline(always)]
    fn iter_alignment_path(&self, skipped_chars: usize, score: u16) -> AlignmentPathIter<'_> {
        self.smith_waterman
            .iter_alignment_path(skipped_chars, score, self.config.max_typos)
    }

    #[inline(always)]
    fn min_haystack_len(&self) -> usize {
        self.config
            .max_typos
            .map(|max| self.needle.len().saturating_sub(max as usize))
            .unwrap_or(0)
    }

    #[inline(always)]
    fn prefilter_window(&mut self, haystack: &[u8]) -> (bool, usize, usize) {
        let original_len = haystack.len();
        self.config
            .max_typos
            .map_or((true, 0, original_len), |max_typos| {
                self.prefilter.match_haystack(haystack, max_typos)
            })
    }

    #[inline(always)]
    fn guard_against_score_overflow(&self) {
        let scoring = &self.config.scoring;
        let max_per_char_score = scoring.match_score
            + scoring
                .capitalization_bonus
                .max(scoring.delimiter_bonus)
                .saturating_sub(scoring.gap_open_penalty)
            + scoring.matching_case_bonus;
        let max_needle_len =
            (u16::MAX - scoring.prefix_bonus - scoring.exact_match_bonus) / max_per_char_score;
        assert!(
            self.needle.len() <= max_needle_len as usize,
            "needle too long and could overflow the u16 score: {} > {}",
            self.needle.len(),
            max_needle_len
        );
    }
}

impl Matcher {
    pub fn new(needle: &str, config: &Config) -> Self {
        let needle_bytes = needle.as_bytes();
        let use_u8 = score_fits_in_u8(needle_bytes.len(), &config.scoring);

        #[cfg(target_arch = "x86_64")]
        {
            if use_u8 {
                if Avx512U8::is_available(needle_bytes, config) {
                    return Self::AVX512U8(BackendMatcher::new(needle, config));
                }
                if Avx2U8::is_available(needle_bytes, config) {
                    return Self::AVX2U8(BackendMatcher::new(needle, config));
                }
                if SseU8::is_available(needle_bytes, config) {
                    return Self::SSEU8(BackendMatcher::new(needle, config));
                }
            } else {
                if Avx512::is_available(needle_bytes, config) {
                    return Self::AVX512(BackendMatcher::new(needle, config));
                }
                if Avx2::is_available(needle_bytes, config) {
                    return Self::AVX2(BackendMatcher::new(needle, config));
                }
                if Sse::is_available(needle_bytes, config) {
                    return Self::SSE(BackendMatcher::new(needle, config));
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if use_u8 {
                if NeonU8::is_available(needle_bytes, config) {
                    return Self::NEONU8(BackendMatcher::new(needle, config));
                }
            } else if Neon::is_available(needle_bytes, config) {
                return Self::NEON(BackendMatcher::new(needle, config));
            }
        }

        if use_u8 {
            Self::ScalarU8(BackendMatcher::new(needle, config))
        } else {
            Self::Scalar(BackendMatcher::new(needle, config))
        }
    }

    pub fn set_needle(&mut self, needle: &str) {
        let config = self.config().clone();
        *self = Self::new(needle, &config);
    }

    pub fn set_config(&mut self, config: &Config) {
        let needle = self.needle().to_string();
        *self = Self::new(&needle, config);
    }

    #[inline(always)]
    pub fn needle(&self) -> &str {
        dispatch!(self, matcher => matcher.core.needle())
    }

    #[inline(always)]
    pub fn config(&self) -> &Config {
        dispatch!(self, matcher => matcher.core.config())
    }

    pub fn match_list<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        dispatch!(self, matcher => matcher.core.match_list(haystacks))
    }

    pub fn match_list_indices<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<MatchIndices> {
        dispatch!(self, matcher => matcher.core.match_list_indices(haystacks))
    }

    pub fn match_list_into<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        dispatch!(self, matcher => {
            matcher
                .core
                .match_list_into(haystacks, haystack_index_offset, matches)
        })
    }

    pub fn match_list_indices_into<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<MatchIndices>,
    ) {
        dispatch!(self, matcher => {
            matcher
                .core
                .match_list_indices_into(haystacks, haystack_index_offset, matches)
        })
    }

    /// Returns an unsorted iterator over the matches in the haystacks.
    /// The needle must not be empty.
    pub fn match_iter<'a, S: AsRef<str> + 'a>(
        &'a mut self,
        haystacks: &'a [S],
    ) -> Box<dyn Iterator<Item = Match> + 'a> {
        dispatch!(self, matcher => Box::new(matcher.core.match_iter(haystacks)))
    }

    /// Returns an unsorted iterator over the matches in the haystacks with indices.
    /// The needle must not be empty.
    pub fn match_iter_indices<'a, S: AsRef<str> + 'a>(
        &'a mut self,
        haystacks: &'a [S],
    ) -> Box<dyn Iterator<Item = MatchIndices> + 'a> {
        dispatch!(self, matcher => Box::new(matcher.core.match_iter_indices(haystacks)))
    }

    #[inline(always)]
    pub fn smith_waterman_one(
        &mut self,
        haystack: &[u8],
        index: u32,
        include_exact: bool,
    ) -> Option<Match> {
        Some(dispatch!(self, matcher => {
            matcher
                .core
                .smith_waterman_one(haystack, index, include_exact)
        }))
    }

    #[inline(always)]
    pub fn smith_waterman_indices_one(
        &mut self,
        haystack: &[u8],
        skipped_chars: usize,
        index: u32,
        include_exact: bool,
    ) -> Option<MatchIndices> {
        dispatch!(self, matcher => {
            matcher.core.smith_waterman_indices_one(
                haystack,
                skipped_chars,
                index,
                include_exact,
            )
        })
    }

    /// Yields `(index, slice, skipped_chars, is_full_haystack)` for each haystack
    /// that survives the prefilter.
    pub fn prefilter_iter<'a, S: AsRef<str> + 'a>(
        &self,
        haystacks: &'a [S],
    ) -> Box<dyn Iterator<Item = (usize, &'a [u8], usize, bool)> + 'a> {
        dispatch!(self, matcher => Box::new(matcher.core.prefilter_iter(haystacks)))
    }

    #[inline(always)]
    pub fn score_haystack(&mut self, haystack: &[u8]) -> u16 {
        dispatch!(self, matcher => matcher.core.score_haystack(haystack))
    }

    #[inline(always)]
    pub fn iter_alignment_path(&self, skipped_chars: usize, score: u16) -> AlignmentPathIter<'_> {
        dispatch!(self, matcher => matcher.core.iter_alignment_path(skipped_chars, score))
    }

    #[inline(always)]
    pub fn guard_against_score_overflow(&self) {
        dispatch!(self, matcher => matcher.core.guard_against_score_overflow())
    }

    #[inline(always)]
    pub fn guard_against_haystack_overflow(haystack_len: usize, haystack_index_offset: u32) {
        assert!(
            (haystack_len.saturating_add(haystack_index_offset as usize)) <= (u32::MAX as usize),
            "too many items in haystack, will overflow the u32 index: {} > {} (index offset: {})",
            haystack_len,
            u32::MAX,
            haystack_index_offset
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::match_list;
    use super::*;

    #[test]
    fn test_basic() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let config = Config {
            max_typos: None,
            ..Config::default()
        };
        let matches = match_list(needle, &haystack, &config);

        println!("{:?}", matches);
        assert_eq!(matches.len(), 4);
        assert_eq!(matches[0].index, 3);
        assert_eq!(matches[1].index, 0);
        assert_eq!(matches[2].index, 2);
        assert_eq!(matches[3].index, 1);
    }

    #[test]
    fn test_no_typos() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let matches = match_list(
            needle,
            &haystack,
            &Config {
                max_typos: Some(0),
                ..Config::default()
            },
        );
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_exact_match() {
        let needle = "deadbe";
        let haystack = vec!["deadbeef", "deadbf", "deadbeefg", "deadbe"];

        let matches = match_list(needle, &haystack, &Config::default());

        let exact_matches = matches.iter().filter(|m| m.exact).collect::<Vec<&Match>>();
        assert_eq!(exact_matches.len(), 1);
        assert_eq!(exact_matches[0].index, 3);
        for m in &exact_matches {
            assert_eq!(haystack[m.index as usize], needle)
        }
    }

    #[test]
    fn test_exact_matches() {
        let needle = "deadbe";
        let haystack = vec![
            "deadbe",
            "deadbeef",
            "deadbe",
            "deadbf",
            "deadbe",
            "deadbeefg",
            "deadbe",
        ];

        let matches = match_list(needle, &haystack, &Config::default());

        let exact_matches = matches.iter().filter(|m| m.exact).collect::<Vec<&Match>>();
        assert_eq!(exact_matches.len(), 4);
        for m in &exact_matches {
            assert_eq!(haystack[m.index as usize], needle)
        }
    }

    #[test]
    fn test_small_needle() {
        let config = Config {
            max_typos: Some(2),
            ..Config::default()
        };
        let matches = match_list("1", &["1"], &config);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 0);
        assert!(matches[0].exact);
    }

    #[test]
    #[cfg(feature = "match_end_col")]
    fn test_match_end_col_through_match_list() {
        let config = Config {
            max_typos: None,
            sort: false,
            ..Config::default()
        };
        let matches = match_list("abc", &["xabcx", "abcdef", "xxabc"], &config);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].end_col, 3);
        assert_eq!(matches[1].end_col, 2);
        assert_eq!(matches[2].end_col, 4);
    }
}
