#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Matching;

/// A single pattern to match, parsed from syntax like `!^foo`
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pattern {
    /// Matching mode from the atom syntax, where `None` defers to [`crate::Config::matching`]
    /// when building the matcher
    pub matching: Option<Matching>,
    /// Raw atom text, e.g. `!^foo`
    pub pattern: String,
    /// Haystacks matching this atom are excluded
    pub negated: bool,
    /// Text to match with the syntax stripped, e.g. `foo`
    pub needle: String,
}

/// Matches the needle literally, without parsing any syntax, using the matching
/// mode from [`crate::Config::matching`]
impl From<&str> for Pattern {
    fn from(needle: &str) -> Self {
        Pattern::new(needle, None, false)
    }
}

impl From<String> for Pattern {
    fn from(needle: String) -> Self {
        needle.as_str().into()
    }
}

impl From<&String> for Pattern {
    fn from(needle: &String) -> Self {
        needle.as_str().into()
    }
}

impl Pattern {
    /// Creates a pattern that matches the needle literally, without parsing any syntax
    pub fn new(needle: &str, matching: Option<Matching>, negated: bool) -> Self {
        Self {
            matching,
            pattern: needle.to_string(),
            negated,
            needle: needle.to_string(),
        }
    }

    /// Parses a single query atom, where special syntax changes the matching mode:
    ///
    /// `foo` - `None` (defers to [`crate::Config::matching`])
    /// `^foo` - [`Matching::Prefix`]
    /// `foo$` - [`Matching::Suffix`]
    /// `'foo` - [`Matching::Substring`]
    /// `^foo$` - [`Matching::Exact`]
    /// `!foo` - negated, [`Matching::Substring`] unless combined with the syntax above
    ///
    /// Any special character can be escaped with a backslash, e.g. `\!foo`, `\^foo`,
    /// `foo\$` or `\'foo` match the literal leading/trailing character, and `foo\ bar`
    /// matches the literal space.
    pub fn parse(atom: &str) -> Self {
        let mut needle = atom;
        let mut negated = false;
        let mut prefix = false;
        let mut suffix = false;
        let mut substring = false;

        // Negation
        if let Some(rest) = needle.strip_prefix('!') {
            needle = rest;
            negated = true;
        }
        if needle.starts_with("\\!") {
            needle = &needle[1..]; // drop the backslash, keep the literal char
        }

        // Leading
        if let Some(rest) = needle.strip_prefix('^') {
            needle = rest;
            prefix = true;
        } else if let Some(rest) = needle.strip_prefix('\'') {
            needle = rest;
            substring = true;
        } else if needle.starts_with("\\^") || needle.starts_with("\\'") {
            needle = &needle[1..]; // drop the backslash, keep the literal char
        }

        // Trailing
        let needle = if let Some(head) = needle.strip_suffix("\\$") {
            format!("{head}$") // drop the backslash, keep the literal `$`
        } else if let Some(head) = needle.strip_suffix('$') {
            suffix = true;
            head.to_string()
        } else {
            needle.to_string()
        };
        let needle = needle.replace("\\ ", " ");

        let matching = match (prefix, suffix, substring) {
            (true, true, _) => Some(Matching::Exact),
            (true, false, _) => Some(Matching::Prefix),
            (false, true, _) => Some(Matching::Suffix),
            (false, false, true) => Some(Matching::Substring),
            // Negating a fuzzy match excludes far more than users typically expect,
            // so bare negated atoms match substrings, like fzf and nucleo
            _ if negated => Some(Matching::Substring),
            _ => None,
        };

        Self {
            matching,
            pattern: atom.to_string(),
            negated,
            needle,
        }
    }

    /// Parses a query of whitespace separated atoms (see [`Pattern::parse`]), e.g.
    /// `foo !^bar` matches haystacks that fuzzy match `foo` and don't start with `bar`.
    /// Escape a literal space with a backslash, e.g. `foo\ bar` is a single atom.
    /// Atoms with an empty needle, e.g. `!` or `^$`, are dropped.
    pub fn parse_query(query: &str) -> Vec<Pattern> {
        let mut patterns = vec![];
        let mut start: Option<usize> = None;
        let mut escaped = false;

        let mut push = |atom: &str| {
            let pattern = Self::parse(atom);
            if !pattern.needle.is_empty() {
                patterns.push(pattern);
            }
        };

        for (i, c) in query.char_indices() {
            if escaped {
                // Previous char was an escaping backslash; keep this char in the atom
                escaped = false;
            } else if c == '\\' {
                start.get_or_insert(i);
                escaped = true;
            } else if c.is_whitespace() {
                if let Some(s) = start.take() {
                    push(&query[s..i]);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            push(&query[s..]);
        }

        patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_parse(atom: &str, needle: &str, matching: Option<Matching>, negated: bool) {
        let pattern = Pattern::parse(atom);
        assert_eq!(pattern.pattern, atom);
        assert_eq!(pattern.needle, needle, "atom: {atom:?}");
        assert_eq!(pattern.matching, matching, "atom: {atom:?}");
        assert_eq!(pattern.negated, negated, "atom: {atom:?}");
    }

    #[test]
    fn parse_selects_matching_mode() {
        assert_parse("foo", "foo", None, false);
        assert_parse("^foo", "foo", Some(Matching::Prefix), false);
        assert_parse("foo$", "foo", Some(Matching::Suffix), false);
        assert_parse("'foo", "foo", Some(Matching::Substring), false);
        assert_parse("^foo$", "foo", Some(Matching::Exact), false);
    }

    #[test]
    fn parse_negation() {
        // Bare negated atoms match substrings, like fzf and nucleo
        assert_parse("!foo", "foo", Some(Matching::Substring), true);
        assert_parse("!^foo", "foo", Some(Matching::Prefix), true);
        assert_parse("!foo$", "foo", Some(Matching::Suffix), true);
        assert_parse("!'foo", "foo", Some(Matching::Substring), true);
        assert_parse("!^foo$", "foo", Some(Matching::Exact), true);
    }

    #[test]
    fn parse_escapes_special_syntax() {
        assert_parse("\\^foo", "^foo", None, false);
        assert_parse("foo\\$", "foo$", None, false);
        assert_parse("\\'foo", "'foo", None, false);
        assert_parse("\\!foo", "!foo", None, false);
        assert_parse("foo\\ bar", "foo bar", None, false);
        assert_parse("!\\^foo", "^foo", Some(Matching::Substring), true);
        assert_parse("!\\!foo", "!foo", Some(Matching::Substring), true);
    }

    #[test]
    fn parse_query_splits_atoms() {
        let patterns = Pattern::parse_query("foo !^bar");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0], Pattern::parse("foo"));
        assert_eq!(patterns[1], Pattern::parse("!^bar"));

        let patterns = Pattern::parse_query("  foo \t bar  ");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].needle, "foo");
        assert_eq!(patterns[1].needle, "bar");
    }

    #[test]
    fn parse_query_escaped_space() {
        let patterns = Pattern::parse_query("foo\\ bar baz");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].needle, "foo bar");
        assert_eq!(patterns[1].needle, "baz");
    }

    #[test]
    fn parse_query_escaped_backslash_before_space_splits() {
        // The escaped backslash consumes the escape, so the space still separates atoms
        let patterns = Pattern::parse_query("foo\\\\ bar");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].needle, "foo\\\\");
        assert_eq!(patterns[1].needle, "bar");
    }

    #[test]
    fn parse_query_drops_empty_atoms() {
        assert!(Pattern::parse_query("").is_empty());
        assert!(Pattern::parse_query("   ").is_empty());
        assert!(Pattern::parse_query("! ^$ '").is_empty());
    }
}
