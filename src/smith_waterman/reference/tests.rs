//! Every available kernel, and the matcher, against the reference DP: scores,
//! indices (an optimal alignment, see [`indices_score`])

use super::{
    Rng, indices_score, opposite_case, optimal_indices, score, score_fits_in_u8, score_unicode,
    score_with_exact_bonus,
};
use crate::smith_waterman::algo::MAX_HAYSTACK_LEN;
use crate::smith_waterman::backend::{Backend, BackendScalar8, BackendScalar16U8};
use crate::smith_waterman::{Kernel, ScoreWidth, SmithWaterman};
use crate::{CaseMatching, Config, Matcher, Scoring, SortStrategy, UnicodeMatching};
use std::boxed::Box;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;

#[derive(Debug, Clone)]
struct Case {
    needle: String,
    haystack: String,
    unicode: bool,
    case_sensitive: bool,
    include_prefix: bool,
}

impl Case {
    fn generate(rng: &mut Rng) -> Self {
        // (needle alphabet, haystack alphabet, unicode)
        const ALPHABETS: [(&str, &str, bool); 6] = [
            ("ab", "abx", false),
            ("aAbB_", "aAbB_/x", false),
            ("abc/", "abcABC/._-x1", false),
            ("aéÉنب다😀_", "aéÉنب다😀_x/Aß", true),
            ("ab", "abxé다", true),
            // Combining marks (2 bytes), Hangul jamo and syllables (3 bytes)
            ("e\u{301}ᄒ하😀", "e\u{301}\u{308}ᄒ하한😀😁 x", true),
        ];
        let (needle_chars, haystack_chars, unicode) = rng.pick(&ALPHABETS);
        let needle_chars: Vec<char> = needle_chars.chars().collect();
        let haystack_chars: Vec<char> = haystack_chars.chars().collect();
        let max_len = rng.pick(&[24, 8, 8, 8]);
        let needle_len = 1 + rng.below(max_len);
        let mut case = Self {
            needle: rng.string(&needle_chars, needle_len),
            haystack: random_haystack(rng, &haystack_chars),
            unicode,
            case_sensitive: rng.below(4) == 0,
            include_prefix: rng.below(4) != 0,
        };
        // Half the time the needle is a subsequence of the haystack (some
        // chars case flipped), so most of its chars can match
        let chars: Vec<char> = case.haystack.chars().collect();
        if rng.below(2) == 0 && !chars.is_empty() {
            let len = 1 + rng.below(chars.len().min(16));
            let mut picks: Vec<usize> = (0..len).map(|_| rng.below(chars.len())).collect();
            picks.sort_unstable();
            case.needle = picks
                .into_iter()
                .map(|i| match rng.below(8) {
                    0 => opposite_case(chars[i]),
                    _ => chars[i],
                })
                .collect();
        }
        case
    }

    fn reference(&self, haystack: &str, scoring: &Scoring) -> u16 {
        let (cs, prefix) = (self.case_sensitive, self.include_prefix);
        if self.unicode {
            score_unicode(&self.needle, haystack, scoring, cs, prefix)
        } else {
            score(&self.needle, haystack.as_bytes(), scoring, cs, prefix)
        }
    }
}

/// A haystack whose length (in chars) is biased toward the chunk boundaries
fn random_haystack(rng: &mut Rng, alphabet: &[char]) -> String {
    let len = match rng.below(4) {
        0 => rng.pick(&[
            7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1009, 1023, 1024,
        ]),
        1 => rng.below(400),
        _ => rng.below(80),
    };
    let mut haystack = rng.string(alphabet, len);
    while haystack.len() > MAX_HAYSTACK_LEN {
        haystack.pop();
    }
    haystack
}

/// Default scoring a third of the time, otherwise random values from tiny to
/// extreme (gaps free or costlier than any match), sometimes with bonuses
/// dominating the per-char score
fn random_scoring(rng: &mut Rng) -> Scoring {
    if rng.below(3) == 0 {
        return Scoring::default();
    }
    let values: &[u16] = match rng.below(4) {
        0 => &[0, 1, 2, 3, 4, 5, 6, 8, 12],
        1 => &[0, 1, 2, 4, 8, 16, 24, 32, 48],
        2 => &[0, 1, 3, 12, 24, 40, 64, 100, 128, 200, 255],
        _ => &[0, 2, 24, 100, 255, 256, 1000, 20000, 30000],
    };
    let mut s = Scoring {
        match_score: rng.pick(values),
        mismatch_penalty: rng.pick(values),
        gap_open_penalty: rng.pick(values),
        gap_extend_penalty: rng.pick(values),
        prefix_bonus: rng.pick(values),
        capitalization_bonus: rng.pick(values),
        matching_case_bonus: rng.pick(values),
        exact_match_bonus: rng.pick(values),
        delimiter_bonus: rng.pick(values),
    };
    if rng.below(3) == 0 {
        s.match_score = rng.below(4) as u16;
        s.mismatch_penalty = rng.below(3) as u16;
        s.matching_case_bonus = rng.below(3) as u16;
        s.gap_extend_penalty = rng.below(3) as u16;
        s.gap_open_penalty = rng.pick(&[0, 8, 30, 60, 120, 125]);
        s.delimiter_bonus = rng.pick(&[0, 20, 40, 60, 90, 120]);
        s.capitalization_bonus = rng.pick(&[0, 20, 40, 60, 90, 120]);
    }
    s
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// A kernel under test, built for the ASCII and the Unicode kernels
struct Both<K> {
    ascii: K,
    unicode: K,
}

/// What the tests read from a kernel, without its type
trait Scorer {
    /// The score, and the indices traced through the stored matrix (relative
    /// to the haystack)
    fn run(&mut self, haystack: &[u8], unicode: bool, start_pos: usize) -> (u16, (u16, Vec<u32>));
}

impl<K: Kernel> Scorer for Both<K> {
    fn run(&mut self, haystack: &[u8], unicode: bool, start_pos: usize) -> (u16, (u16, Vec<u32>)) {
        let (score, (indices_score, indices)) = if unicode {
            (
                self.unicode.score_haystack_unicode(haystack, start_pos),
                self.unicode
                    .score_haystack_unicode_indices(haystack, start_pos),
            )
        } else {
            (
                self.ascii.score_haystack(haystack, start_pos),
                self.ascii.score_haystack_indices(haystack, start_pos),
            )
        };
        let relative = indices.into_iter().map(|i| i - start_pos as u32);
        (score, (indices_score, relative.collect()))
    }
}

/// Every kernel available on this machine for the needle and scoring, labeled
/// `<tier>-<lanes>x<bits>`, or `<tier>-auto` for the kernel picking the width
fn kernels(needle: &str, cs: bool, s: &Scoring) -> Vec<(&'static str, Box<dyn Scorer>)> {
    fn both<K: Kernel>(needle: &str, cs: bool, s: &Scoring) -> Box<dyn Scorer> {
        Box::new(Both {
            ascii: K::new(needle, s, cs, false),
            unicode: K::new(needle, s, cs, true),
        })
    }
    let mut all: Vec<(&'static str, Box<dyn Scorer>)> = Vec::new();
    macro_rules! push {
        ($label:literal, $b:ty, u8) => {
            if <$b>::is_available() && score_fits_in_u8(needle.len(), s) {
                all.push(($label, both::<SmithWaterman<$b>>(needle, cs, s)));
            }
        };
        ($label:literal, $b:ty, u16) => {
            if <$b>::is_available() {
                all.push(($label, both::<SmithWaterman<$b>>(needle, cs, s)));
            }
        };
        ($label:literal, $u8:ty, $u16:ty) => {
            if <$u16>::is_available() {
                let k = both::<ScoreWidth<SmithWaterman<$u8>, SmithWaterman<$u16>>>(needle, cs, s);
                all.push(($label, k));
            }
        };
    }
    push!("scalar-8x16", BackendScalar8, u16);
    push!("scalar-16x8", BackendScalar16U8, u8);
    push!("scalar-auto", BackendScalar16U8, BackendScalar8);
    #[cfg(target_arch = "x86_64")]
    {
        use crate::smith_waterman::backend::*;
        push!("sse-8x16", BackendSSE, u16);
        push!("sse-16x8", BackendSSEU8, u8);
        push!("sse-auto", BackendSSEU8, BackendSSE);
        push!("avx2-16x16", BackendAVX, u16);
        push!("avx2-32x8", BackendAVXU8, u8);
        push!("avx2-auto", BackendAVXU8, BackendAVX);
        push!("avx512-32x16", BackendAVX512, u16);
        push!("avx512-64x8", BackendAVX512U8, u8);
    }
    #[cfg(target_arch = "aarch64")]
    {
        use crate::smith_waterman::backend::*;
        push!("neon-8x16", BackendNEON, u16);
        push!("neon-16x8", BackendNEONU8, u8);
        push!("neon-auto", BackendNEONU8, BackendNEON);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use crate::smith_waterman::backend::*;
        push!("wasm-8x16", BackendWasm, u16);
        push!("wasm-16x8", BackendWasmU8, u8);
        push!("wasm-auto", BackendWasmU8, BackendWasm);
    }
    all
}

/// Why the indices aren't an optimal alignment of the case, if they aren't:
/// the score must be the reference's and re-score to it along the indices
fn check_indices(case: &Case, s: &Scoring, (score, indices): &(u16, Vec<u32>)) -> Option<String> {
    let (needle, haystack) = (case.needle.as_str(), case.haystack.as_bytes());
    let (cs, prefix, unicode) = (case.case_sensitive, case.include_prefix, case.unicode);
    let (want, _) = optimal_indices(needle, haystack, s, cs, prefix, unicode);
    if *score != want {
        return Some(format!("score {score}, reference {want}"));
    }
    match indices_score(needle, haystack, s, cs, prefix, unicode, indices) {
        Ok(path) if path == want as i64 => None,
        Ok(path) => Some(format!("indices {indices:?} score {path} along their path")),
        Err(err) => Some(err),
    }
}

/// Checks every kernel on the case: its score and indices against the
/// reference. Returns the number of kernels checked
fn check(case: &Case, s: &Scoring) -> usize {
    let want = case.reference(&case.haystack, s);
    let (needle, haystack) = (case.needle.as_str(), case.haystack.as_bytes());
    let (cs, unicode) = (case.case_sensitive, case.unicode);
    let msg = |label: &str| format!("{label} {case:?} {s:?}");
    let start = if case.include_prefix { 0 } else { 1 };
    let mut all = kernels(needle, cs, s);
    for (label, kernel) in &mut all {
        let (score, indices) = kernel.run(haystack, unicode, start);
        assert_eq!(score, want, "{}", msg(label));
        if let Some(err) = check_indices(case, s, &indices) {
            panic!("{}: {err}", msg(label));
        }
    }
    all.len()
}

#[test]
fn kernels_match_reference() {
    // (the unoptimized kernels are slow)
    let iterations = match () {
        _ if cfg!(miri) => 8,
        _ if cfg!(debug_assertions) => 2_000,
        _ => 10_000,
    };
    let mut rng = Rng(0x5eed_0f_2026);
    for _ in 0..iterations {
        let case = Case::generate(&mut rng);
        let scoring = random_scoring(&mut rng);
        // Longer needles may saturate u16 scores, where the reference doesn't
        if case.needle.len() > scoring.max_needle_len() {
            continue;
        }
        check(&case, &scoring);

        // The reference's own traceback is an optimal alignment
        let (needle, haystack) = (case.needle.as_str(), case.haystack.as_bytes());
        let (cs, prefix, unicode) = (case.case_sensitive, case.include_prefix, case.unicode);
        let path = optimal_indices(needle, haystack, &scoring, cs, prefix, unicode);
        assert_eq!(path.0, case.reference(&case.haystack, &scoring));
        assert_eq!(check_indices(&case, &scoring, &path), None);
    }
}

/// Haystacks of tokens earning the prefix, delimiter and capitalization
/// bonuses back to back (e.g. "/aC/aC"), with needles from windows of them
/// near the u8 kernels' limit: [`Scoring::max_score`] bounds the score, and
/// the kernels picking the width match the reference on both sides of it.
///
/// Why [`Scoring::max_score`] bounds every cell:
///
/// A cell is the score of its best local alignment. Matching a needle char
/// gains at most `match_score + matching_case_bonus` plus the bonus of its
/// haystack column, every other step only subtracts, and an alignment uses
/// each row at most once. So a cell is at most `rows` per-char gains plus the
/// bonuses of the matched columns minus the penalties paid between them.
///
/// A column's bonus is one of: `prefix_bonus` (first column only, which has no
/// previous char), `delimiter_bonus` ("D", a non-delimiter after a
/// delimiter), `capitalization_bonus` ("C", an uppercase letter after a
/// lowercase letter) or none. They constrain each other on adjacent columns:
/// a D column isn't a delimiter, so the next column can't be D (but can be C,
/// e.g. "/aC"), and a C column is uppercase, so the next column has no bonus.
/// So the matched columns with a D or C bonus form blocks of adjacent columns,
/// each "D", "C" or "DC" (at most `single = max(D, C)` or `pair = D + C`).
/// Two consecutive blocks are separated by at least one row without a bonus
/// (a matched char without one, a mismatch or a skipped needle char), or by
/// skipping haystack columns within one row: a left gap out of the block's
/// matched cell, which pays at least `gap = max(gap_open, gap_extend)` (the
/// kernels charge `gap_open - gap_extend`, clamped at zero, on top of the
/// extend penalty).
///
/// Pairing each block with the separator after it, every pair earns at most
/// `rate` bonus per row, the best of `pair / 3` (DC and a row), `(pair - gap) /
/// 2` (DC and a gap), `single / 2` and `single - gap`. The last block needs no
/// separator, so with `rows` rows the D and C bonuses sum to at most
/// `blocks(rows) = max(rate * rows, single + rate * (rows - 1), pair + rate *
/// (rows - 2))`, rounded down as scores are integers. A matched first column
/// adds the prefix bonus and uses a row, and may directly precede a block, so
/// the bonuses sum to at most `max(blocks(rows), prefix + blocks(rows - 1))`.
#[test]
fn width_matches_reference() {
    let iterations = if cfg!(miri) { 4 } else { 6_000 };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let check_case = |needle: &str, haystack: &str, s: &Scoring, unicode, cs| {
        let case = Case {
            needle: needle.to_string(),
            haystack: haystack.to_string(),
            unicode,
            case_sensitive: cs,
            include_prefix: true,
        };
        let rows = if unicode {
            needle.chars().count()
        } else {
            needle.len()
        };
        let want = case.reference(haystack, s);
        assert!(s.max_score(rows) >= want as u64, "{case:?} {s:?}");
        if needle.len() <= s.max_needle_len() {
            check(&case, s);
        }
    };

    // Scores 256, which the u8 kernel returns as 249
    let default = Scoring::default();
    check_case("_a/d_bab_aaa_d", "_a/d_bab_aaa_d", &default, false, false);
    // Adjacent delimiter and capitalization bonuses: 13 rows score 252, and a
    // 14th row skipped by an up gap leaves 247, both past the u8 kernels
    for (needle, haystack) in [
        ("/aC/aC/aC/aC/", "/aC/aC/aC/aC/"),
        ("/aC/aC/aC/aC/", "x/aC/aC/aC/aC/"),
        ("/aC/aC/aC/aC/z", "/aC/aC/aC/aC/"),
        ("/aC/aC/aC/aC/a", "x/aC/aC/aC/aC/a"),
        ("/ac/ac/ac/ac/a", "/aC/aC/aC/aC/a_"),
    ] {
        check_case(needle, haystack, &default, false, false);
        check_case(needle, haystack, &default, true, false);
    }

    for i in 0..iterations {
        let unicode = rng.below(4) == 0;
        let scoring = if rng.below(8) == 0 {
            default.clone()
        } else {
            random_scoring(&mut rng)
        };
        // Half of the needles are as long as the u8 kernels allow, or longer
        let u8_rows = (1..64)
            .take_while(|&rows| score_fits_in_u8(rows, &scoring))
            .last();
        let len = match u8_rows {
            Some(rows) if rng.below(2) == 0 => rows + rng.below(3),
            _ => 1 + rng.below(24),
        };
        let haystack = if i % 2 == 0 {
            // Copies of a random needle with edits, so most rows score highly
            let alphabet: Vec<char> = if unicode { "aé/다B_" } else { "aAbB/_c" }
                .chars()
                .collect();
            let needle = rng.string(&alphabet, len);
            let mut haystack = String::new();
            for _ in 0..1 + rng.below(3) {
                for c in needle.chars() {
                    match rng.below(8) {
                        0 => {}
                        1 => haystack.push(rng.pick(&['x', 'Y', '/', '-', 'é'])),
                        _ => haystack.push(c),
                    }
                    if rng.below(6) == 0 {
                        haystack.extend(c.to_uppercase());
                    }
                }
            }
            if !unicode {
                haystack = haystack.replace('é', "e");
            }
            haystack
        } else {
            // Repeating a few tokens reaches the bound more often than mixing
            let tokens: &[&str] = if unicode {
                &["/aC", "/a", "aB", "/é", "é", "다", "_", "x", "/", "aC"]
            } else {
                &["/aC", "/a", "aC", "aB", "a", "/", "_", "x", "A", "-b"]
            };
            let tokens = [rng.pick(tokens), rng.pick(tokens), rng.pick(tokens)];
            let mut haystack = String::new();
            while haystack.chars().count() < len + rng.below(len + 1) {
                haystack.push_str(rng.pick(&tokens));
            }
            haystack
        };
        // A window of the haystack, some chars dropped, replaced or case
        // flipped
        let chars: Vec<char> = haystack.chars().collect();
        let start = rng.below(chars.len().saturating_sub(len) + 1);
        let mutation_percent = rng.pick(&[0, 0, 3, 10]);
        let mut needle: String = chars[start..]
            .iter()
            .take(len)
            .filter_map(|&c| match rng.below(100) < mutation_percent {
                false => Some(c),
                true => [None, Some('z'), Some(opposite_case(c))][rng.below(3)],
            })
            .collect();
        if needle.is_empty() {
            needle.push('a');
        }
        let cs = rng.below(4) == 0;
        check_case(&needle, &haystack, &scoring, unicode, cs);
    }
}

/// Indices are only checked when the matcher scored the whole haystack (a
/// prefilter may trim it to a window)
#[test]
fn matcher_matches_reference() {
    let iterations = if cfg!(miri) { 4 } else { 6_000 };
    let mut rng = Rng(0x7ace_0096);
    let mut checked = 0;
    for _ in 0..iterations {
        let mut case = Case::generate(&mut rng);
        let scoring = random_scoring(&mut rng);
        if case.needle.len() > scoring.max_needle_len() {
            continue;
        }
        let casing = rng.pick(&[
            CaseMatching::Ignore,
            CaseMatching::Smart,
            CaseMatching::Respect,
        ]);
        case.case_sensitive = casing.respects_case_for(&case.needle);
        case.include_prefix = true;
        let unicode = if case.unicode {
            UnicodeMatching::Always
        } else {
            UnicodeMatching::Ignore
        };
        let config = Config::default()
            .casing(casing)
            .unicode(unicode)
            .scoring(scoring.clone())
            .sort(SortStrategy::IndexAsc);

        // The haystack, its exact needle, and sub-ranges of it
        let mut haystacks = Vec::from([case.haystack.clone(), case.needle.clone()]);
        let chars: Vec<usize> = case.haystack.char_indices().map(|(i, _)| i).collect();
        for _ in 0..4 {
            let a = chars.get(rng.below(chars.len() + 1)).copied().unwrap_or(0);
            let b = chars.get(rng.below(chars.len() + 1)).copied();
            let b = b.unwrap_or(case.haystack.len()).max(a);
            haystacks.push(case.haystack[a..b].to_string());
        }
        let mut matcher = Matcher::new(&case.needle, &config.clone().max_typos(None));
        let matches = matcher.match_list(&haystacks);
        assert_eq!(matches.len(), haystacks.len());
        for m in matches {
            let haystack = &haystacks[m.index as usize];
            let (cs, unicode) = (case.case_sensitive, case.unicode);
            let want = score_with_exact_bonus(&case.needle, haystack, &scoring, cs, unicode);
            assert_eq!(m.score, want, "{case:?} haystack={haystack:?} {scoring:?}");
            assert_eq!(matcher.match_one(haystack, m.index), Some(m));
        }

        let max_typos = rng.pick(&[None, Some(0), Some(1), Some(2)]);
        let Some(m) = Matcher::new(case.needle.as_str(), &config.max_typos(max_typos))
            .match_list_indices(&[case.haystack.as_str()])
            .pop()
        else {
            continue;
        };
        let exact_bonus = if m.exact {
            scoring.exact_match_bonus
        } else {
            0
        };
        let got = (m.score - exact_bonus, m.indices);
        if got.0 != case.reference(&case.haystack, &scoring) {
            continue;
        }
        checked += 1;
        if let Some(err) = check_indices(&case, &scoring, &got) {
            panic!("{case:?} max_typos={max_typos:?} {scoring:?}: {err}");
        }
    }
    assert!(
        cfg!(miri) || checked > iterations / 2,
        "{checked} matches checked"
    );
}

#[test]
fn reference_fixed_cases() {
    let s = Scoring::default();
    let char_score = s.match_score + s.matching_case_bonus;
    let ascii = |needle: &str, haystack: &str, s: &Scoring| {
        score(needle, haystack.as_bytes(), s, false, true)
    };
    assert_eq!(ascii("abc", "abc", &s), 3 * char_score + s.prefix_bonus);
    assert_eq!(ascii("b", "a-b", &s), char_score + s.delimiter_bonus);
    assert_eq!(
        ascii("D", "forDist", &s),
        char_score + s.capitalization_bonus
    );
    assert_eq!(ascii("a", "A", &s), s.match_score + s.prefix_bonus);
    // One gap from a match pays the open penalty once, then extends
    assert_eq!(
        ascii("test", "Uteost", &s),
        4 * char_score - s.gap_open_penalty
    );
    assert_eq!(
        ascii("test", "Uteoost", &s),
        4 * char_score - s.gap_open_penalty - s.gap_extend_penalty
    );
    // A skipped needle char below a match
    assert_eq!(ascii("foo", "Ufo", &s), 2 * char_score - s.gap_open_penalty);
    // Scalars count once, whatever their UTF-8 length
    let unicode = |needle: &str, haystack: &str| score_unicode(needle, haystack, &s, false, true);
    assert_eq!(unicode("é", "é"), char_score + s.prefix_bonus);
    assert_eq!(unicode("ab", "aé😀b"), unicode("ab", "axyb"));
    assert_eq!(opposite_case('É'), 'é');
    assert_eq!(opposite_case('ß'), 'ß');
    assert_eq!(
        score_with_exact_bonus("foo", "foo", &s, false, false),
        3 * char_score + s.prefix_bonus + s.exact_match_bonus
    );

    // The kernels charge max(open, extend) for a gap's first step
    let s = Scoring {
        gap_open_penalty: 1,
        gap_extend_penalty: 5,
        ..Scoring::default()
    };
    for (needle, haystack, want) in [
        ("ab", "axb", 2 * char_score + s.prefix_bonus - 5),
        ("ab", "axxb", 2 * char_score + s.prefix_bonus - 10),
        ("axb", "ab", 2 * char_score + s.prefix_bonus - 5),
    ] {
        assert_eq!(ascii(needle, haystack, &s), want);
        let mut sw = <SmithWaterman<BackendScalar8> as Kernel>::new(needle, &s, false, false);
        assert_eq!(sw.score_haystack(haystack.as_bytes(), true), want);
    }

    // A delimiter bonus ("a" after "/") directly followed by a capitalization
    // bonus ("C" after "a"), with no gap in between
    let s = Scoring {
        match_score: 24,
        mismatch_penalty: 3,
        gap_open_penalty: 24,
        gap_extend_penalty: 2,
        prefix_bonus: 12,
        capitalization_bonus: 100,
        matching_case_bonus: 5,
        exact_match_bonus: 0,
        delimiter_bonus: 100,
    };
    assert_eq!(ascii("ac", "/aC", &s), 253);
    assert!(s.max_score(2) >= 253 && !score_fits_in_u8(2, &s));
    // Delimiter, capitalization, none, delimiter: no gap anywhere
    let s = Scoring {
        match_score: 0,
        mismatch_penalty: 0,
        gap_open_penalty: 125,
        prefix_bonus: 0,
        matching_case_bonus: 0,
        ..s
    };
    assert_eq!(ascii("aC/a", "/aC/a", &s), 300);
    assert!(!score_fits_in_u8(4, &s));
    // "/aC" repeated earns the prefix bonus and four delimiter and
    // capitalization pairs in 13 rows, which with the diagonal's transient
    // mismatch penalty exceeds 255
    let default = Scoring::default();
    assert_eq!(ascii("/aC/aC/aC/aC/", "/aC/aC/aC/aC/", &default), 252);
    assert_eq!(default.max_score(13), 254);
    assert!(score_fits_in_u8(12, &default) && !score_fits_in_u8(13, &default));
}
