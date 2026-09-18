use super::Matcher;
use crate::{Config, Match, MatchIndices, Pattern};
use alloc::vec::Vec;

/// Haystacks pulled from the source iterator per block (see [`Blocks`])
const BLOCK: usize = 256;

/// Runs an iterator of haystacks through the list path a block at a time,
/// so haystacks are prefiltered and scored with the same pair kernel as
/// `match_list`, and yields the matches in iteration order.
pub(crate) struct Blocks<I: Iterator> {
    iter: I,
    /// The block being matched; empty between calls, so cloning needs no
    /// `I::Item: Clone`
    haystacks: Vec<I::Item>,
    matches: Vec<Match>,
    /// Next match of `matches` to yield
    next: usize,
    /// Index of the first haystack of the next block
    index: usize,
}

impl<I> Blocks<I>
where
    I: Iterator,
    I::Item: AsRef<str>,
{
    pub(crate) fn new(iter: I) -> Self {
        Self {
            iter,
            haystacks: Vec::new(),
            matches: Vec::new(),
            next: 0,
            index: 0,
        }
    }

    pub(crate) fn next_match(&mut self, matcher: &mut Matcher) -> Option<Match> {
        loop {
            if let Some(m) = self.matches.get(self.next) {
                self.next += 1;
                return Some(*m);
            }
            self.haystacks.clear();
            self.haystacks.extend(self.iter.by_ref().take(BLOCK));
            if self.haystacks.is_empty() {
                return None;
            }
            let offset = u32::try_from(self.index)
                .expect("too many items in haystack, will overflow the u32 index");
            self.index += self.haystacks.len();
            self.matches.clear();
            self.next = 0;
            matcher.match_list_into(&self.haystacks, offset, &mut self.matches);
            self.haystacks.clear();
        }
    }

    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        // Every item is filtered independently, so we can't know how many pass
        let buffered = self.matches.len() - self.next;
        (buffered, self.iter.size_hint().1.map(|n| n + buffered))
    }
}

impl<I: Iterator + Clone> Clone for Blocks<I> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            haystacks: Vec::new(),
            matches: self.matches.clone(),
            next: self.next,
            index: self.index,
        }
    }
}

impl<I: Iterator + core::fmt::Debug> core::fmt::Debug for Blocks<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Blocks")
            .field("iter", &self.iter)
            .field("matches", &self.matches)
            .field("next", &self.next)
            .field("index", &self.index)
            .finish()
    }
}

/// Extension trait adding fuzzy matching functions to any iterator whose items
/// are strings. Results are yielded lazily in iteration order (not sorted by
/// score). Items that don't match are skipped.
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
    /// item that passes. Items are pulled in blocks and matched through the
    /// same path as [`Matcher::match_list`].
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
    fn fuzzy_match(self, pattern: impl Into<Pattern>, config: &Config) -> FuzzyMatch<Self>
    where
        Self::Item: AsRef<str>,
    {
        FuzzyMatch {
            matcher: Matcher::new(pattern, config),
            blocks: Blocks::new(self),
        }
    }

    /// Fuzzy matches each item against `needle`, yielding a [`MatchIndices`],
    /// which are equivalent to [`Match`] except they include the indices of
    /// the matched characters in the haystack.
    ///
    /// This API has not been optimized for performance, and should only be used
    /// on small lists or after matching a list of haystacks with
    /// [`FuzzyMatchExt::fuzzy_match`]. Useful for displaying
    /// matched indices in the UI.
    ///
    /// ```
    /// use frizbee::{Config, iter::FuzzyMatchExt};
    ///
    /// let haystacks = ["fooBar", "foo_bar", "prelude", "println!"];
    /// let matches: Vec<_> = haystacks
    ///     .iter()
    ///     .fuzzy_match_indices("fBr", &Config::default())
    ///     .collect();
    /// ```
    fn fuzzy_match_indices(
        self,
        pattern: impl Into<Pattern>,
        config: &Config,
    ) -> FuzzyMatchIndices<Self>
    where
        Self::Item: AsRef<str>,
    {
        FuzzyMatchIndices {
            matcher: Matcher::new(pattern, config),
            iter: self,
            index: 0,
        }
    }
}

impl<I: Iterator> FuzzyMatchExt for I {}

/// Iterator adapter created by [`FuzzyMatchExt::fuzzy_match`]
#[derive(Debug, Clone)]
pub struct FuzzyMatch<I: Iterator> {
    matcher: Matcher,
    blocks: Blocks<I>,
}

impl<I> Iterator for FuzzyMatch<I>
where
    I: Iterator,
    I::Item: AsRef<str>,
{
    type Item = Match;

    fn next(&mut self) -> Option<Match> {
        self.blocks.next_match(&mut self.matcher)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.blocks.size_hint()
    }
}

/// Iterator adapter created by [`FuzzyMatchExt::fuzzy_match_indices`]
#[derive(Debug, Clone)]
pub struct FuzzyMatchIndices<I> {
    matcher: Matcher,
    iter: I,
    index: usize,
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
            if let Some(m) = self.matcher.match_one_indices(haystack, index) {
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
    use crate::SortStrategy;

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
                let config = Config::default()
                    .max_typos(max_typos)
                    .sort(SortStrategy::IndexAsc);
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
                let config = Config::default()
                    .max_typos(max_typos)
                    .sort(SortStrategy::IndexAsc);
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
                &Config::default()
                    .max_typos(Some(0))
                    .sort(SortStrategy::IndexAsc),
            )
            .map(|m| m.index)
            .collect::<Vec<_>>();
        assert!(!scores.is_empty());
    }
}
