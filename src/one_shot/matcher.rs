use super::Appendable;
#[cfg(feature = "simd")]
use super::bucket::FixedWidthBucket;

use crate::one_shot::match_too_large;
use crate::prefilter::Prefilter;
use crate::smith_waterman::greedy::match_greedy;
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

    match_list_impl(needle, haystacks, 0, config, &mut matches);

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
    index_offset: u32,
    config: &Config,
    matches: &mut M,
) {
    assert!(
        (index_offset as usize) + haystacks.len() < (u32::MAX as usize),
        "haystack index overflow"
    );

    let needle = needle.as_ref();
    if needle.is_empty() {
        for (i, _) in haystacks.iter().enumerate() {
            matches.append(Match {
                index: (i as u32) + index_offset,
                score: 0,
                exact: false,
            });
        }
        return;
    }

    let prefilter = Prefilter::new(needle, config.max_typos.unwrap_or(0));

    #[cfg(feature = "simd")]
    let mut bucket_size_4 = FixedWidthBucket::<4, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_8 = FixedWidthBucket::<8, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_12 = FixedWidthBucket::<12, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_16 = FixedWidthBucket::<16, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_20 = FixedWidthBucket::<20, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_24 = FixedWidthBucket::<24, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_32 = FixedWidthBucket::<32, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_48 = FixedWidthBucket::<48, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_64 = FixedWidthBucket::<64, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_96 = FixedWidthBucket::<96, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_128 = FixedWidthBucket::<128, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_160 = FixedWidthBucket::<160, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_192 = FixedWidthBucket::<192, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_224 = FixedWidthBucket::<224, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_256 = FixedWidthBucket::<256, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_384 = FixedWidthBucket::<384, M>::new(needle, &config);
    #[cfg(feature = "simd")]
    let mut bucket_size_512 = FixedWidthBucket::<512, M>::new(needle, &config);

    // If max_typos is set, we can ignore any haystacks that are shorter than the needle
    // minus the max typos, since it's impossible for them to match
    let min_haystack_len = config
        .max_typos
        .map(|max| needle.len() - (max as usize))
        .unwrap_or(0);

    for (i, haystack) in haystacks
        .iter()
        .map(|h| h.as_ref())
        .enumerate()
        .filter(|(_, h)| h.len() >= min_haystack_len)
        .filter(|(_, h)| {
            config.max_typos.is_none()
                || prefilter.match_haystack_unordered_insensitive(h.as_bytes())
        })
    {
        let i = i as u32 + index_offset;

        // fallback to greedy matching
        if match_too_large(needle, haystack) {
            let (score, _, exact) = match_greedy(needle, haystack, &config.scoring);
            matches.append(Match {
                index: i,
                score,
                exact,
            });
            continue;
        }

        // Pick the bucket to insert into based on the length of the haystack
        match haystack.len() {
            #[cfg(feature = "simd")]
            0..=4 => bucket_size_4.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            5..=8 => bucket_size_8.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            9..=12 => bucket_size_12.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            13..=16 => bucket_size_16.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            17..=20 => bucket_size_20.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            21..=24 => bucket_size_24.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            25..=32 => bucket_size_32.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            33..=48 => bucket_size_48.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            49..=64 => bucket_size_64.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            65..=96 => bucket_size_96.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            97..=128 => bucket_size_128.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            129..=160 => bucket_size_160.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            161..=192 => bucket_size_192.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            193..=224 => bucket_size_224.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            225..=256 => bucket_size_256.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            257..=384 => bucket_size_384.add_haystack(matches, haystack, i),
            #[cfg(feature = "simd")]
            385..=512 => bucket_size_512.add_haystack(matches, haystack, i),

            // fallback to greedy matching
            _ => {
                let (score, _, exact) = match_greedy(needle, haystack, &config.scoring);
                matches.append(Match {
                    index: i,
                    score,
                    exact,
                });
                continue;
            }
        };
    }

    // Run processing on remaining haystacks in the buckets
    #[cfg(feature = "simd")]
    bucket_size_4.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_8.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_12.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_16.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_20.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_24.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_32.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_48.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_64.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_96.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_128.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_160.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_192.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_224.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_256.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_384.finalize(matches);
    #[cfg(feature = "simd")]
    bucket_size_512.finalize(matches);
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
