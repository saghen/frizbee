use crate::prefilter::Prefilter;
use crate::smith_waterman::AlignmentPathIter;
use crate::smith_waterman::simd::SmithWatermanMatcher;
use crate::{Config, Match, MatchIndices};

#[derive(Debug, Clone)]
pub struct Matcher {
    pub needle: String,
    pub config: Config,
    pub prefilter: Prefilter,
    pub smith_waterman: SmithWatermanMatcher,
}

impl Matcher {
    pub fn new(needle: &str, config: &Config) -> Self {
        let matcher = Self {
            needle: needle.to_string(),
            config: config.clone(),
            prefilter: Prefilter::new(needle.as_bytes()),
            smith_waterman: SmithWatermanMatcher::new(needle.as_bytes(), &config.scoring),
        };
        matcher.guard_against_score_overflow();
        matcher
    }

    pub fn set_needle(&mut self, needle: &str) {
        self.needle = needle.to_string();
        self.prefilter = Prefilter::new(needle.as_bytes());
        self.smith_waterman = SmithWatermanMatcher::new(needle.as_bytes(), &self.config.scoring);
        self.guard_against_score_overflow();
    }

    pub fn set_config(&mut self, config: &Config) {
        self.config = config.clone();
        self.smith_waterman =
            SmithWatermanMatcher::new(self.needle.as_bytes(), &self.config.scoring);
        self.guard_against_score_overflow();
    }

    pub fn match_list<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<Match> {
        Matcher::guard_against_haystack_overflow(haystacks.len(), 0);

        if self.needle.is_empty() {
            return (0..haystacks.len())
                .map(|index| Match {
                    index: index as u32,
                    score: 0,
                    exact: false,
                })
                .collect();
        }

        let mut matches = vec![];
        self.match_list_into(haystacks, 0, &mut matches);

        if self.config.sort {
            matches.sort_unstable();
        }

        matches
    }

    pub fn match_list_indices<S: AsRef<str>>(&mut self, haystacks: &[S]) -> Vec<MatchIndices> {
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

    pub fn match_list_into<S: AsRef<str>>(
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

        for match_ in
            self.prefilter_iter(haystacks)
                .filter_map(|(index, haystack, skipped_chunks)| {
                    self.smith_waterman_one(
                        haystack,
                        (index as u32) + haystack_index_offset,
                        skipped_chunks == 0,
                    )
                })
        {
            matches.push(match_);
        }
    }

    pub fn match_list_indices_into<S: AsRef<str>>(
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

        for match_ in
            self.prefilter_iter(haystacks)
                .filter_map(|(index, haystack, skipped_chunks)| {
                    self.smith_waterman_indices_one(
                        haystack,
                        skipped_chunks,
                        (index as u32) + haystack_index_offset,
                        skipped_chunks == 0,
                    )
                })
        {
            matches.push(match_);
        }
    }

    #[inline(always)]
    pub fn smith_waterman_one(
        &mut self,
        haystack: &[u8],
        index: u32,
        include_exact: bool,
    ) -> Option<Match> {
        // Haystack too large, fallback to greedy matching
        let mut score = self
            .smith_waterman
            .match_haystack(haystack, self.config.max_typos)?;

        let exact = include_exact && self.needle.as_bytes() == haystack;
        if exact {
            score += self.config.scoring.exact_match_bonus;
        }

        Some(Match {
            index,
            score,
            exact,
        })
    }

    #[inline(always)]
    pub fn smith_waterman_indices_one(
        &mut self,
        haystack: &[u8],
        skipped_chunks: usize,
        index: u32,
        include_exact: bool,
    ) -> Option<MatchIndices> {
        // Haystack too large, fallback to greedy matching
        let (mut score, indices) = self.smith_waterman.match_haystack_indices(
            haystack,
            skipped_chunks,
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

    #[inline(always)]
    pub fn prefilter_iter<'a, S: AsRef<str>>(
        &self,
        haystacks: &'a [S],
    ) -> impl Iterator<Item = (usize, &'a [u8], usize)> + use<'a, S> {
        let needle = self.needle.as_bytes();
        assert!(!needle.is_empty(), "needle must not be empty");

        // If max_typos is set, we can ignore any haystacks that are shorter than the needle
        // minus the max typos, since it's impossible for them to match
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);
        let config = self.config.clone();
        let prefilter = self.prefilter.clone();

        haystacks
            .iter()
            .map(|h| h.as_ref().as_bytes())
            .enumerate()
            .filter(move |(_, h)| h.len() >= min_haystack_len)
            // Prefiltering
            .filter_map(move |(i, haystack)| {
                let (matched, skipped_chunks) = config.max_typos.map_or((true, 0), |max_typos| {
                    prefilter.match_haystack(haystack, max_typos)
                });
                // Skip any chunks where we know the needle doesn't match
                matched.then(|| (i, &haystack[skipped_chunks * 16..], skipped_chunks))
            })
    }

    #[inline(always)]
    pub fn iter_alignment_path(
        &self,
        haystack_len: usize,
        skipped_chunks: usize,
        score: u16,
    ) -> AlignmentPathIter<'_> {
        self.smith_waterman.iter_alignment_path(
            haystack_len,
            skipped_chunks,
            score,
            self.config.max_typos,
        )
    }

    #[inline(always)]
    pub fn guard_against_score_overflow(&self) {
        let scoring = &self.config.scoring;
        let max_per_char_score = scoring.match_score
            + scoring.capitalization_bonus / 2
            + scoring.delimiter_bonus / 2
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

    #[inline(always)]
    pub fn guard_against_haystack_overflow(haystack_len: usize, haystack_index_offset: u32) {
        assert!(
            (haystack_len.saturating_add(haystack_index_offset as usize)) <= (u32::MAX as usize),
            "too many haystack which will overflow the u32 index: {} > {} (index offset: {})",
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
        // max_typos longer than needle
        let config = Config {
            max_typos: Some(2),
            ..Config::default()
        };
        let matches = match_list("1", &["1"], &config);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 0);
        assert!(matches[0].exact);
    }
}
