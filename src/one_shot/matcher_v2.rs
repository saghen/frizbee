use super::Appendable;

use crate::prefilter::Prefilter;
use crate::prefilter::scalar::match_haystack_insensitive;
use crate::prefilter::x86_64::{
    match_haystack_unordered_insensitive_with_chunks, overlapping_load,
};
use crate::smith_waterman::greedy::match_greedy;
use crate::smith_waterman::v2;
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
    let mut matches = if config.max_typos.is_none() {
        Vec::with_capacity(haystacks.len())
    } else {
        vec![]
    };

    match_list_impl(needle, haystacks, config, &mut matches);

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
    assert!(
        haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    let needle = needle.as_ref();
    let needle_cased = Prefilter::case_needle(needle);
    let needle_struct = v2::Needle::new(needle);

    // nothing to match, return empty matches
    if needle.is_empty() {
        for (i, _) in haystacks.iter().enumerate() {
            matches.append(Match {
                index: (i as u32),
                score: 0,
                exact: false,
            });
        }
        return;
    }

    // If max_typos is set, we can ignore any haystacks that are shorter than the needle
    // minus the max typos, since it's impossible for them to match
    let min_haystack_len = config
        .max_typos
        .map(|max| needle.len() - (max as usize))
        .unwrap_or(0);
    let max_haystack_len = haystacks
        .iter()
        .map(|h| h.as_ref().len())
        .max()
        .unwrap()
        .min(512);

    let mut score_matrix = v2::generate_score_matrix(needle.len(), max_haystack_len);

    for (i, haystack) in haystacks
        .iter()
        .map(|h| h.as_ref())
        .enumerate()
        .filter(|(_, h)| h.len() >= min_haystack_len)
    {
        let (prefilter, skipped_chunks) = match (config.max_typos, haystack.len()) {
            (None, _) => (true, 0),
            (_, 0) => (true, 0),
            (_, 1..8) => (
                match_haystack_insensitive(&needle_cased, haystack.as_bytes()),
                0,
            ),
            _ => unsafe {
                match_haystack_unordered_insensitive_with_chunks(&needle_cased, haystack.as_bytes())
            },
        };
        if !prefilter {
            continue;
        }

        // haystack too large, fallback to greedy matching
        let (score, exact) = if haystack.len() > 512 {
            let (score, _, exact) = match_greedy(needle, haystack, &config.scoring);
            if score == 0 {
                continue;
            }
            (score, exact)
        }
        // regular smith waterman matching
        else {
            let mut score = v2::smith_waterman(
                &needle_struct,
                &haystack.as_bytes()[(skipped_chunks * 16)..],
                &config.scoring,
                &mut score_matrix,
            );

            if let Some(max_typos) = config.max_typos
                && unsafe {
                    v2::typos_from_score_matrix(&score_matrix, score, max_typos, haystack.len())
                } > max_typos
            {
                continue;
            }

            let exact = needle == haystack;
            if exact {
                score += config.scoring.exact_match_bonus;
            }

            (score, exact)
        };

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
}

