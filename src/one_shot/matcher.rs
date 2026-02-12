use crate::prefilter::Prefilter;
use crate::smith_waterman::greedy::match_greedy;
use crate::smith_waterman::simd::SmithWatermanMatcher;
use crate::{Config, Match};

#[derive(Debug, Clone)]
pub(crate) struct Matcher {
    needle: String,
    config: Config,
    prefilter: Prefilter,
    smith_waterman: SmithWatermanMatcher,
}

impl Matcher {
    pub fn new(needle: &str, config: &Config) -> Self {
        Self {
            needle: needle.to_string(),
            config: config.clone(),
            prefilter: Prefilter::new(needle.as_bytes()),
            smith_waterman: SmithWatermanMatcher::new(needle.as_bytes(), &config.scoring),
        }
    }

    pub fn match_list_impl<S: AsRef<str>>(
        &mut self,
        haystacks: &[S],
        index_offset: u32,
        matches: &mut Vec<Match>,
    ) {
        // Guard against empty needle or empty haystacks
        if self.needle.is_empty() {
            for index in (0..haystacks.len()).map(|i| i as u32 + index_offset) {
                matches.push(Match {
                    index,
                    score: 0,
                    exact: false,
                });
            }
            return;
        }
        if haystacks.is_empty() {
            return;
        }

        let needle = self.needle.as_bytes();

        // If max_typos is set, we can ignore any haystacks that are shorter than the needle
        // minus the max typos, since it's impossible for them to match
        let min_haystack_len = self
            .config
            .max_typos
            .map(|max| needle.len().saturating_sub(max as usize))
            .unwrap_or(0);

        for (i, haystack, skipped_chunks) in haystacks
            .iter()
            .map(|h| h.as_ref().as_bytes())
            .enumerate()
            .filter(|(_, h)| h.len() >= min_haystack_len)
            // Prefiltering
            .filter_map(|(i, haystack)| {
                let (matched, skipped_chunks) =
                    self.config.max_typos.map_or((true, 0), |max_typos| {
                        self.prefilter.match_haystack(haystack, max_typos)
                    });
                // Skip any chunks where we know the needle doesn't match
                matched.then(|| (i, &haystack[skipped_chunks * 16..], skipped_chunks))
            })
        {
            // Haystack too large, fallback to greedy matching
            let match_score = if haystack.len() > 512 {
                match_greedy(needle, haystack, &self.config.scoring).map(|(score, _)| score)
            }
            // Smith waterman matching
            else {
                self.smith_waterman
                    .match_haystack(haystack, self.config.max_typos)
            };

            if let Some(mut score) = match_score {
                // Exact match bonus
                let exact = skipped_chunks == 0 && needle == haystack;
                if exact {
                    score += self.config.scoring.exact_match_bonus;
                }

                matches.push(Match {
                    index: i as u32 + index_offset,
                    score,
                    exact,
                });
            }
        }
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
        let mut config = Config::default();
        config.max_typos = Some(2);  // max_typos longer than needle
        let matches = match_list("1", &["1"], &config);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 0);
        assert_eq!(matches[0].exact, true);
    }
}
