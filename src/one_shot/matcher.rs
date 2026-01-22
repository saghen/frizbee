use super::Appendable;

use crate::prefilter::Prefilter;
use crate::smith_waterman::greedy::match_greedy;
use crate::smith_waterman::x86_64;
use crate::{Config, Match};

/// Computes the Smith-Waterman score with affine gaps for a needle against a list of haystacks.
///
/// You should call this function with as many haystacks as you have available as it will
/// automatically chunk the haystacks based on string length to avoid unnecessary computation
/// due to SIMD
pub fn match_list<S1: AsRef<str>, S2: AsRef<str>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
) -> Vec<Match> {
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    // Guard against empty needle or empty haystacks
    if needle.as_ref().is_empty() {
        return (0..haystacks.len())
            .map(|i| Match {
                index: i as u32,
                score: 0,
                exact: false,
            })
            .collect();
    }
    if haystacks.is_empty() {
        return vec![];
    }

    // Matching
    let mut matches = if config.max_typos.is_none() {
        Vec::with_capacity(haystacks.len())
    } else {
        vec![]
    };
    match_list_impl(needle, haystacks, config, &mut matches);

    // Sorting
    if config.sort {
        #[cfg(feature = "parallel_sort")]
        {
            use rayon::prelude::*;
            matches.par_sort();
        }
        #[cfg(not(feature = "parallel_sort"))]
        matches.sort_unstable();
    }

    matches
}

pub(crate) fn match_list_impl<S1: AsRef<str>, S2: AsRef<str>, M: Appendable<Match>>(
    needle: S1,
    haystacks: &[S2],
    config: &Config,
    matches: &mut M,
) {
    let prefilter = Prefilter::new(needle.as_ref(), config.max_typos.unwrap_or(0));
    let needle = needle.as_ref().as_bytes();

    // If max_typos is set, we can ignore any haystacks that are shorter than the needle
    // minus the max typos, since it's impossible for them to match
    let min_haystack_len = config
        .max_typos
        .map(|max| needle.len().saturating_sub(max as usize))
        .unwrap_or(0);

    // TODO: test if building the full matrix is faster than calculating the max length
    let max_haystack_len = haystacks
        .iter()
        .map(|h| h.as_ref().len())
        .max()
        .unwrap()
        .min(512);

    let mut score_matrix = x86_64::generate_score_matrix(needle.len(), max_haystack_len);

    for (i, original_haystack, haystack) in haystacks
        .iter()
        .map(|h| h.as_ref().as_bytes())
        .enumerate()
        .filter(|(_, h)| h.len() >= min_haystack_len)
        // Prefiltering
        .filter_map(|(i, haystack)| {
            let (matched, skipped_chunks) = config.max_typos.map_or((true, 0), |_| {
                prefilter.match_haystack_insensitive(haystack)
            });
            // Skip any chunks where we know the needle doesn't match
            matched.then(|| (i, haystack, haystack[skipped_chunks * 16..].as_ref()))
        })
    {
        // Haystack too large, fallback to greedy matching
        let mut score = if haystack.len() > 512 {
            let (score, _) = match_greedy(needle, haystack, &config.scoring);
            if score == 0 {
                continue;
            }
            score
        }
        // Smith waterman matching
        else {
            let score = x86_64::smith_waterman(
                &prefilter.needle_cased,
                haystack,
                &config.scoring,
                &mut score_matrix,
            );

            if let Some(max_typos) = config.max_typos
                && x86_64::typos_from_score_matrix(&score_matrix, score, max_typos, haystack.len())
                    > max_typos
            {
                continue;
            }

            score
        };

        // Exact match bonus
        let exact = needle == original_haystack;
        if exact {
            score += config.scoring.exact_match_bonus;
        }

        matches.append(Match {
            index: i as u32,
            score,
            exact,
        });
    }
}

#[cfg(test)]
mod tests {
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
