use crate::smith_waterman::score_fits_in_u8;
use crate::sort::radix_sort;
use crate::{Config, Match, MatchIndices};

mod algo;
mod backend;
use algo::{MANY_TYPOS, NO_PREFILTER, Specialized};
use backend::*;

#[derive(Debug, Clone)]
pub struct Matcher {
    pub needle: String,
    pub config: Config,
    backend: MatcherBackend,
}

/// Many variants so we use a macro to expand to the correct impl
macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512U8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVXU8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::SSEU8($m) => $body,
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::SSE($m) => $body,
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEONU8($m) => $body,
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEON($m) => $body,
            MatcherBackend::ScalarU8($m) => $body,
            MatcherBackend::Scalar($m) => $body,
        }
    };
}

/// Each of these receives its own inline(always) hot loop, so a single function that branches
/// would be enormous (thousands of symbols). So instead, we dispatch to the correct implementation
/// which contains the `#[target_feature]` and hot loop (in backend.rs)
#[rustfmt::skip]
macro_rules! dispatch_typos {
    ($max_typos:expr, $needs_unicode:expr, |$typos:ident, $unicode:ident| $body:expr) => {
        match ($max_typos, $needs_unicode) {
            (None, false)     => { const $typos: u16 = NO_PREFILTER; const $unicode: bool = false; $body }
            (None, true)      => { const $typos: u16 = NO_PREFILTER; const $unicode: bool = true;  $body }
            (Some(0), false)  => { const $typos: u16 = 0;            const $unicode: bool = false; $body }
            (Some(0), true)   => { const $typos: u16 = 0;            const $unicode: bool = true;  $body }
            (Some(1), false)  => { const $typos: u16 = 1;            const $unicode: bool = false; $body }
            (Some(1), true)   => { const $typos: u16 = 1;            const $unicode: bool = true;  $body }
            (Some(2), false)  => { const $typos: u16 = 2;            const $unicode: bool = false; $body }
            (Some(2), true)   => { const $typos: u16 = 2;            const $unicode: bool = true;  $body }
            (Some(_), false)  => { const $typos: u16 = MANY_TYPOS;   const $unicode: bool = false; $body }
            (Some(_), true)   => { const $typos: u16 = MANY_TYPOS;   const $unicode: bool = true;  $body }
        }
    };
}

impl Matcher {
    pub fn new(needle: &str, config: &Config) -> Self {
        Self {
            backend: Self::get_backend(needle, config),
            needle: needle.to_string(),
            config: config.clone(),
        }
    }

    pub fn set_config(&mut self, config: Config) {
        if self.config == config {
            return;
        }
        self.config = config;
        self.backend = Self::get_backend(&self.needle, &self.config);
    }

    pub fn set_needle(&mut self, needle: &str) {
        if self.needle == needle {
            return;
        }
        self.needle = needle.to_string();
        self.backend = Self::get_backend(&self.needle, &self.config);
    }

    pub fn match_list<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        let mut matches = vec![];
        self.match_list_into(haystacks, 0, &mut matches);
        if !self.needle.is_empty() && self.config.sort {
            radix_sort(&mut matches);
        }
        matches
    }

    pub fn match_list_indices<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<MatchIndices> {
        Self::guard_against_haystack_overflow(haystacks.len(), 0);
        if self.needle.is_empty() {
            return (0..haystacks.len()).map(MatchIndices::from_index).collect();
        }

        let needs_unicode = self.config.unicode.respects_unicode_for(&self.needle);
        let mut matches = dispatch!(&mut self.backend, matcher => {
            dispatch_typos!(self.config.max_typos, needs_unicode, |TYPOS, UNICODE| {
                // SAFETY: the backend was only constructed after its
                // `is_available()` check passed (see `get_backend`), so its
                // target features are available here.
                unsafe { matcher.match_list_indices::<TYPOS, UNICODE, S>(haystacks) }
            })
        });

        if self.config.sort {
            matches.sort_unstable();
        }
        matches
    }

    pub(super) fn match_list_into<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        haystack_index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        Self::guard_against_haystack_overflow(haystacks.len(), haystack_index_offset);
        if self.needle.is_empty() {
            let indices = (0..haystacks.len()).map(|i| i + haystack_index_offset as usize);
            matches.extend(indices.map(Match::from_index));
            return;
        }

        let needs_unicode = self.config.unicode.respects_unicode_for(&self.needle);
        dispatch!(&mut self.backend, matcher => {
            dispatch_typos!(self.config.max_typos, needs_unicode, |TYPOS, UNICODE| {
                unsafe {
                    matcher.match_list::<TYPOS, UNICODE, S>(
                        haystacks,
                        haystack_index_offset,
                        matches,
                    )
                }
            })
        })
    }

    #[inline(always)]
    fn guard_against_haystack_overflow(haystack_len: usize, haystack_index_offset: u32) {
        assert!(
            haystack_len.saturating_add(haystack_index_offset as usize) <= (u32::MAX as usize),
            "too many items in haystack, will overflow the u32 index: {} > {} (index offset: {})",
            haystack_len,
            u32::MAX,
            haystack_index_offset
        );
    }

    fn get_backend(needle: &str, config: &Config) -> MatcherBackend {
        let use_u8 = score_fits_in_u8(needle.len(), &config.scoring);

        #[cfg(target_arch = "x86_64")]
        {
            if use_u8 {
                if MatcherAVX512U8::is_available() {
                    return MatcherBackend::AVX512U8(unsafe {
                        MatcherAVX512U8::build(needle, config)
                    });
                }
                if MatcherAVXU8::is_available() {
                    return MatcherBackend::AVXU8(unsafe { MatcherAVXU8::build(needle, config) });
                }
                if MatcherSSEU8::is_available() {
                    return MatcherBackend::SSEU8(unsafe { MatcherSSEU8::build(needle, config) });
                }
            } else {
                if MatcherAVX512::is_available() {
                    return MatcherBackend::AVX512(unsafe { MatcherAVX512::build(needle, config) });
                }
                if MatcherAVX::is_available() {
                    return MatcherBackend::AVX(unsafe { MatcherAVX::build(needle, config) });
                }
                if MatcherSSE::is_available() {
                    return MatcherBackend::SSE(unsafe { MatcherSSE::build(needle, config) });
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if use_u8 {
                if MatcherNEONU8::is_available() {
                    return MatcherBackend::NEONU8(unsafe { MatcherNEONU8::build(needle, config) });
                }
            } else if MatcherNEON::is_available() {
                return MatcherBackend::NEON(unsafe { MatcherNEON::build(needle, config) });
            }
        }

        if use_u8 {
            MatcherBackend::ScalarU8(unsafe { MatcherScalarU8::build(needle, config) })
        } else {
            MatcherBackend::Scalar(unsafe { MatcherScalar::build(needle, config) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::match_list;
    use super::*;
    use crate::CaseMatching;

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
    fn test_case_sensitive_matching() {
        let haystack = ["foo", "FOO", "fOo", "xxfooxx"];
        let config = Config {
            sort: false,
            ..Config::default()
        };

        let matches = match_list("foo", &haystack, &config);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );

        let config = Config {
            casing: CaseMatching::Respect,
            sort: false,
            ..Config::default()
        };

        let matches = match_list("foo", &haystack, &config);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 3]
        );

        let indices = Matcher::new("foo", &config).match_list_indices(&haystack);
        assert_eq!(
            indices.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![0, 3]
        );

        let config = Config {
            casing: CaseMatching::Smart,
            sort: false,
            ..Config::default()
        };

        let matches = match_list("FoO", &["foo", "FOO", "FoO", "xxFoOxx"], &config);
        assert_eq!(
            matches.iter().map(|m| m.index).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn test_empty_needle() {
        let haystack = ["foo", "bar"];
        let mut matcher = Matcher::new("", &Config::default());
        let is_u8 = match &matcher.backend {
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512U8(_) | MatcherBackend::AVXU8(_) | MatcherBackend::SSEU8(_) => {
                true
            }
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEONU8(_) => true,
            MatcherBackend::ScalarU8(_) => true,
            _ => false,
        };
        assert!(is_u8);

        let matches = matcher.match_list(&haystack);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[1].index, 1);

        let indices = matcher.match_list_indices(&haystack);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].index, 0);
        assert_eq!(indices[1].index, 1);
    }

    #[test]
    fn u8_path_selected_for_short_needle() {
        let matcher = Matcher::new("abc", &Config::default());
        let is_u8 = match &matcher.backend {
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512U8(_) | MatcherBackend::AVXU8(_) | MatcherBackend::SSEU8(_) => {
                true
            }
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEONU8(_) => true,
            MatcherBackend::ScalarU8(_) => true,
            _ => false,
        };
        assert!(is_u8);
    }

    #[test]
    fn u16_path_selected_for_long_needle() {
        let matcher = Matcher::new("abcdefghijklmnopqrst", &Config::default());
        let is_u16 = match &matcher.backend {
            #[cfg(target_arch = "x86_64")]
            MatcherBackend::AVX512(_) | MatcherBackend::AVX(_) | MatcherBackend::SSE(_) => true,
            #[cfg(target_arch = "aarch64")]
            MatcherBackend::NEON(_) => true,
            MatcherBackend::Scalar(_) => true,
            _ => false,
        };
        assert!(is_u16);
    }

    #[test]
    fn reuse_handles_state_changes() {
        let long_needle = "abcdefghijklmnopqrst";
        let first_haystacks = [
            "xxabcdefghijklmnopqrstxx".to_string(),
            "abcdefghijklmnopqrst".to_string(),
            "no-match".to_string(),
        ];
        let first_config = Config {
            max_typos: None,
            sort: false,
            ..Config::default()
        };
        let mut matcher = Matcher::new(long_needle, &first_config);

        let first = matcher.match_list(&first_haystacks);
        assert_eq!(
            &first,
            &match_list(long_needle, &first_haystacks, &first_config)
        );

        let second_haystacks = [
            "fooBar".to_string(),
            "foo_bar".to_string(),
            "fbr".to_string(),
            "bar".to_string(),
        ];
        matcher.set_needle("fB");
        let second_config = Config {
            casing: CaseMatching::Smart,
            sort: false,
            ..Config::default()
        };
        matcher.set_config(second_config.clone());
        let second = matcher.match_list(&second_haystacks);
        assert_eq!(
            &second,
            &match_list("fB", &second_haystacks, &second_config)
        );

        let unicode_haystacks = [
            "é다😀".to_string(),
            "xxé__다__😀yy".to_string(),
            "é다".to_string(),
            "plain ascii".to_string(),
        ];
        matcher.set_needle("é다😀");
        let unicode_config = Config {
            max_typos: Some(0),
            sort: false,
            ..Config::default()
        };
        matcher.set_config(unicode_config.clone());
        let unicode = matcher.match_list(&unicode_haystacks);
        assert_eq!(
            &unicode,
            &match_list("é다😀", &unicode_haystacks, &unicode_config)
        );

        matcher.set_needle("fB");
        let third_config = Config {
            casing: CaseMatching::Ignore,
            max_typos: Some(1),
            sort: true,
            ..Config::default()
        };
        matcher.set_config(third_config.clone());
        let third = matcher.match_list(&first_haystacks);
        assert_eq!(&third, &match_list("fB", &first_haystacks, &third_config));
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
