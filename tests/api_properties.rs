use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use bolero::check;
use frizbee::{
    CaseMatching, Config, Match, MatchIndices, Matcher, Scoring, match_list, match_list_indices,
    match_list_parallel,
};

#[derive(Debug, Clone)]
struct ApiCase {
    needle: String,
    haystacks: Vec<String>,
    config: Config,
}

impl ApiCase {
    fn from_bytes(input: &[u8]) -> Self {
        let mut cursor = ByteCursor::new(input);
        let needle_len = cursor.len(32, &[0, 1, 2, 7, 8, 15, 16, 31, 32]);
        let haystack_count = cursor.len(32, &[0, 1, 2, 7, 8, 15, 16, 31, 32]);
        let haystacks = (0..haystack_count)
            .map(|_| {
                let len = cursor.len(96, &[0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96]);
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
        if byte & 0x0f == 0 {
            return match (byte >> 4) & 3 {
                0 => 'é',
                1 => 'ن',
                2 => '다',
                _ => '😀',
            };
        }

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
}

fn test_iterations(default: usize) -> usize {
    if cfg!(miri) { default.min(4) } else { default }
}

#[test]
fn generated_public_api_properties() {
    check!()
        .with_iterations(test_iterations(192))
        .with_max_len(if cfg!(miri) { 384 } else { 4096 })
        .for_each(|input: &[u8]| {
            let case = ApiCase::from_bytes(input);
            assert_public_api_case(&case);
        });
}

fn assert_public_api_case(case: &ApiCase) {
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

    if case.config.max_typos.is_none() {
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

#[test]
fn empty_needle_matches_everything() {
    let haystacks = ["foo", "bar"];
    let config = Config::default();

    let matches = match_list("", &haystacks, &config);
    assert_eq!(
        match_views(&matches),
        match_views(&[Match::from_index(0), Match::from_index(1)])
    );

    let indices = match_list_indices("", &haystacks, &config);
    assert_eq!(
        indices_views(&indices),
        indices_views(&[MatchIndices::from_index(0), MatchIndices::from_index(1)])
    );
}

#[test]
fn exact_match_flag_tracks_full_haystack_match() {
    let haystacks = ["deadbe", "deadbeef", "deadbe", "deadbf", "xxdeadbexx"];
    let matches = match_list("deadbe", &haystacks, &Config::default());

    let exact_by_index = matches
        .iter()
        .map(|match_| (match_.index, match_.exact))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(exact_by_index.get(&0), Some(&true));
    assert_eq!(exact_by_index.get(&1), Some(&false));
    assert_eq!(exact_by_index.get(&2), Some(&true));
    assert_eq!(exact_by_index.get(&4), Some(&false));
}

#[test]
fn unicode_matcher_zero_typo_uses_byte_offsets_and_exact_flags() {
    let haystacks = vec![
        "xxإنyy".to_string(),
        "إن".to_string(),
        "\u{06e5}\u{0606}".to_string(),
        "nomatch".to_string(),
        "x".repeat(65),
    ];
    let config = Config {
        max_typos: Some(0),
        sort: false,
        ..Config::default()
    };

    let matches = match_list("إن", &haystacks, &config);
    let mut matcher = Matcher::new("إن", &config);
    let matcher_matches = matcher.match_list(&haystacks);
    assert_match_views_eq("unicode Matcher::match_list", &matcher_matches, &matches);
    assert_eq!(
        matches
            .iter()
            .map(|match_| (match_.index, match_.exact))
            .collect::<Vec<_>>(),
        vec![(0, false), (1, true)]
    );

    let indices = match_list_indices("إن", &haystacks, &config);
    let mut matcher = Matcher::new("إن", &config);
    let matcher_indices = matcher.match_list_indices(&haystacks);
    assert_eq!(indices_views(&matcher_indices), indices_views(&indices));
    assert_eq!(indices.len(), 2);
    assert_eq!((indices[0].index, indices[0].exact), (0, false));
    assert_eq!(indices[0].indices, vec![5, 4, 3, 2]);
    assert_eq!((indices[1].index, indices[1].exact), (1, true));
    assert_eq!(indices[1].indices, vec![3, 2, 1, 0]);
}

#[test]
fn unicode_matcher_indices_cover_gaps_and_chunk_boundaries() {
    let config = Config {
        max_typos: None,
        sort: false,
        ..Config::default()
    };

    let gap_haystacks = ["é😀x"];
    let gap_matches = match_list("éx", &gap_haystacks, &config);
    let gap_indices = match_list_indices("éx", &gap_haystacks, &config);
    assert_eq!(gap_indices.len(), 1);
    assert_eq!(gap_indices[0].indices, vec![6, 1, 0]);
    assert_eq!(
        (gap_indices[0].score, gap_indices[0].exact),
        (gap_matches[0].score, gap_matches[0].exact)
    );

    let boundary_haystacks = ["_______😀x"];
    let boundary_matches = match_list("😀x", &boundary_haystacks, &config);
    let boundary_indices = match_list_indices("😀x", &boundary_haystacks, &config);
    assert_eq!(boundary_indices.len(), 1);
    assert_eq!(boundary_indices[0].indices, vec![11, 10, 9, 8, 7]);
    assert_eq!(
        (boundary_indices[0].score, boundary_indices[0].exact),
        (boundary_matches[0].score, boundary_matches[0].exact)
    );
}

#[test]
fn unicode_matcher_typo_prefilter_counts_scalar_values() {
    let haystacks = ["ن", "😀", "x"];
    let config = Config {
        max_typos: Some(1),
        sort: false,
        ..Config::default()
    };

    let matches = match_list("إن", &haystacks, &config);
    assert_eq!(
        matches
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![0]
    );

    let many_typo_config = Config {
        max_typos: Some(2),
        sort: false,
        ..Config::default()
    };
    let matches = match_list("éन😀", &haystacks, &many_typo_config);
    assert_eq!(
        matches
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn case_matching_modes_apply_to_matches_and_indices() {
    let haystacks = ["foo", "FOO", "fOo", "xxfooxx"];
    let config = Config {
        sort: false,
        ..Config::default()
    };
    assert_eq!(
        match_list("foo", &haystacks, &config)
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );

    let config = Config {
        casing: CaseMatching::Respect,
        sort: false,
        ..Config::default()
    };
    assert_eq!(
        match_list("foo", &haystacks, &config)
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![0, 3]
    );
    assert_eq!(
        match_list_indices("foo", &haystacks, &config)
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![0, 3]
    );

    let config = Config {
        casing: CaseMatching::Smart,
        sort: false,
        ..Config::default()
    };
    assert_eq!(
        match_list("FoO", &["foo", "FOO", "FoO", "xxFoOxx"], &config)
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn score_overflow_guard_panics() {
    let long_needle = "a".repeat(5000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = Matcher::new(&long_needle, &Config::default());
    }));
    assert!(result.is_err());
}

#[test]
fn zero_parallel_threads_panics() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = match_list_parallel("a", &["a"], &Config::default(), 0);
    }));
    assert!(result.is_err());
}

#[test]
fn parallel_chunk_boundaries_match_sequential() {
    let mut haystacks = (0..4101)
        .map(|index| format!("nomatch-{index}"))
        .collect::<Vec<_>>();
    for (index, value) in [
        (0, "abc"),
        (2047, "xabc"),
        (2048, "abxc"),
        (2049, "alpha/beta/abc"),
        (4095, "ABC"),
        (4096, "a_b_c"),
        (4100, "zabc"),
    ] {
        haystacks[index] = value.to_string();
    }

    let sorted = Config {
        sort: true,
        ..Config::default()
    };
    let sequential = match_list("abc", &haystacks, &sorted);
    let parallel_one = match_list_parallel("abc", &haystacks, &sorted, 1);
    assert_match_views_eq("single-thread chunk boundary", &parallel_one, &sequential);

    let parallel = match_list_parallel("abc", &haystacks, &sorted, 8);
    assert_match_views_eq("sorted chunk boundary", &parallel, &sequential);

    let unsorted = Config {
        sort: false,
        ..Config::default()
    };
    let sequential = match_list("abc", &haystacks, &unsorted);
    let parallel = match_list_parallel("abc", &haystacks, &unsorted, 8);
    assert_eq!(
        sorted_match_views(&parallel),
        sorted_match_views(&sequential)
    );
}

#[test]
fn sorted_parallel_equal_scores_use_index_tiebreaking_across_chunks() {
    let mut haystacks = (0..4097)
        .map(|index| format!("nomatch-{index}"))
        .collect::<Vec<_>>();
    haystacks[2047] = "abc".to_string();
    haystacks[2048] = "abc".to_string();
    haystacks[4096] = "abc".to_string();

    let config = Config {
        sort: true,
        ..Config::default()
    };
    let sequential = match_list("abc", &haystacks, &config);
    assert_eq!(
        sequential
            .iter()
            .map(|match_| match_.index)
            .collect::<Vec<_>>(),
        vec![2047, 2048, 4096]
    );

    let parallel = match_list_parallel("abc", &haystacks, &config, 8);
    assert_match_views_eq("equal-score chunk tie", &parallel, &sequential);
}

#[test]
fn public_indices_are_reverse_byte_offsets() {
    let haystacks = ["xabcx", "a_b_c", "nomatch"];
    let config = Config {
        sort: false,
        ..Config::default()
    };

    let matches = match_list_indices("abc", &haystacks, &config);
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].index, 0);
    assert_eq!(matches[0].indices, vec![3, 2, 1]);
    assert_eq!(matches[1].index, 1);
    assert_eq!(matches[1].indices, vec![4, 2, 0]);
}

#[test]
#[cfg(feature = "match_end_col")]
fn match_end_col_survives_prefilter_offsets() {
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

#[test]
fn custom_scoring_stays_within_overflow_guard() {
    let config = Config {
        scoring: Scoring {
            match_score: 8,
            matching_case_bonus: 1,
            ..Scoring::default()
        },
        ..Config::default()
    };
    let matches = match_list("abc", &["abc", "a_b_c"], &config);
    assert_eq!(matches.len(), 2);
}
