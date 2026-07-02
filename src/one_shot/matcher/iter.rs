use super::Matcher;
use crate::{Config, Match, MatchIndices};

/// Extension trait adding fuzzy matching functions to any iterator whose items
/// are strings. Results are yielded lazily in iteration order (not sorted by score).
/// Items that don't match are skipped.
///
/// # Example
///
/// ```
/// use frizbee::{Config, iter::FuzzyMatchExt};
///
/// let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
/// let matches: Vec<_> = haystacks
///     .iter()
///     .fuzzy_match("fBr", &Config::default())
///     .collect();
/// ```
pub trait FuzzyMatchExt: Iterator + Sized {
    /// Fuzzy matches each item against `needle`, yielding a [`Match`] for every
    /// item that passes. See [`Matcher::match_iter`].
    fn fuzzy_match(self, needle: &str, config: &Config) -> FuzzyMatch<Self>
    where
        Self::Item: AsRef<str>,
    {
        let matcher = Matcher::new(needle, config);
        let empty_needle = matcher.needle.is_empty();
        let needs_unicode = matcher.config.unicode.respects_unicode_for(&matcher.needle);
        FuzzyMatch {
            matcher,
            iter: self,
            index: 0,
            empty_needle,
            needs_unicode,
        }
    }

    /// Fuzzy matches each item against `needle`, yielding a [`MatchIndices`]
    /// (includes the matched character indices) for every item that passes.
    /// See [`Matcher::match_iter_indices`].
    fn fuzzy_match_indices(self, needle: &str, config: &Config) -> FuzzyMatchIndices<Self>
    where
        Self::Item: AsRef<str>,
    {
        let matcher = Matcher::new(needle, config);
        let empty_needle = matcher.needle.is_empty();
        let needs_unicode = matcher.config.unicode.respects_unicode_for(&matcher.needle);
        FuzzyMatchIndices {
            matcher,
            iter: self,
            index: 0,
            empty_needle,
            needs_unicode,
        }
    }
}

impl<I: Iterator> FuzzyMatchExt for I {}

/// Iterator adapter created by [`FuzzyMatchExt::fuzzy_match`]
#[derive(Debug, Clone)]
pub struct FuzzyMatch<I> {
    matcher: Matcher,
    iter: I,
    index: usize,
    empty_needle: bool,
    needs_unicode: bool,
}

impl<I> Iterator for FuzzyMatch<I>
where
    I: Iterator,
    I::Item: AsRef<str>,
{
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        loop {
            let haystack = self.iter.next()?;
            let index = u32::try_from(self.index)
                .expect("too many items in haystack, will overflow the u32 index");
            self.index += 1;
            if let Some(m) =
                self.matcher
                    .match_one(haystack, index, self.empty_needle, self.needs_unicode)
            {
                return Some(m);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Every item is filtered independently, so we can't know how many pass
        (0, self.iter.size_hint().1)
    }
}

/// Iterator adapter created by [`FuzzyMatchExt::fuzzy_match_indices`]
#[derive(Debug, Clone)]
pub struct FuzzyMatchIndices<I> {
    matcher: Matcher,
    iter: I,
    index: usize,
    empty_needle: bool,
    needs_unicode: bool,
}

impl<I> Iterator for FuzzyMatchIndices<I>
where
    I: Iterator,
    I::Item: AsRef<str>,
{
    type Item = MatchIndices;

    fn next(&mut self) -> Option<MatchIndices> {
        loop {
            let haystack = self.iter.next()?;
            let index = u32::try_from(self.index)
                .expect("too many items in haystack, will overflow the u32 index");
            self.index += 1;
            if let Some(m) = self.matcher.match_one_indices(
                haystack,
                index,
                self.empty_needle,
                self.needs_unicode,
            ) {
                return Some(m);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.iter.size_hint().1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HAYSTACKS: [&str; 7] = [
        "deadbeef",
        "deadbf",
        "deadbeefg",
        "deadbe",
        "no-match",
        "DeAdBe",
        "é다😀dead__be",
    ];

    #[test]
    fn fuzzy_match_matches_match_iter() {
        for needle in ["deadbe", "é다😀"] {
            for max_typos in [None, Some(0), Some(1), Some(2), Some(3)] {
                let config = Config {
                    max_typos,
                    sort: false,
                    ..Config::default()
                };
                let from_ext = HAYSTACKS
                    .iter()
                    .fuzzy_match(needle, &config)
                    .collect::<Vec<_>>();
                let from_iter = Matcher::new(needle, &config)
                    .match_iter(HAYSTACKS.iter())
                    .collect::<Vec<_>>();
                assert_eq!(
                    from_ext, from_iter,
                    "needle: {needle:?}, max_typos: {max_typos:?}"
                );
            }
        }
    }

    #[test]
    fn fuzzy_match_indices_matches_match_iter_indices() {
        for needle in ["deadbe", "é다😀"] {
            for max_typos in [None, Some(0), Some(1), Some(2), Some(3)] {
                let config = Config {
                    max_typos,
                    sort: false,
                    ..Config::default()
                };
                let from_ext = HAYSTACKS
                    .iter()
                    .fuzzy_match_indices(needle, &config)
                    .collect::<Vec<_>>();
                let from_iter = Matcher::new(needle, &config)
                    .match_iter_indices(HAYSTACKS.iter())
                    .collect::<Vec<_>>();
                assert_eq!(
                    from_ext, from_iter,
                    "needle: {needle:?}, max_typos: {max_typos:?}"
                );
            }
        }
    }

    #[test]
    fn fuzzy_match_empty_needle_yields_all() {
        let matches = ["foo", "bar"]
            .iter()
            .fuzzy_match("", &Config::default())
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[1].index, 1);
    }

    #[test]
    fn fuzzy_match_indices_empty_needle_yields_all() {
        let matches = ["foo", "bar"]
            .iter()
            .fuzzy_match_indices("", &Config::default())
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[1].index, 1);
    }

    #[test]
    fn fuzzy_match_chains_with_other_adapters() {
        // Confirms the adapter composes like any other iterator adapter.
        let scores = HAYSTACKS
            .iter()
            .filter(|h| !h.starts_with("no"))
            .fuzzy_match(
                "deadbe",
                &Config {
                    max_typos: Some(0),
                    sort: false,
                    ..Config::default()
                },
            )
            .map(|m| m.index)
            .collect::<Vec<_>>();
        assert!(!scores.is_empty());
    }
}
