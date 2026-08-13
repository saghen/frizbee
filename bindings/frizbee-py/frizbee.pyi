"""SIMD fuzzy string matching: typo-resistant Smith-Waterman, similar to FZF.

These stubs are the API reference for the compiled `frizbee` extension module.

Config kwargs
    Accepted by `Matcher`, `Matcher.from_query`, `Matcher.from_patterns` and
    `Matcher.set_config`:

    - ``max_typos``: max characters missing from the needle before a haystack is
      filtered out. Defaults to ``0`` (mirroring the core default); ``None`` means
      unlimited typos.
    - ``max_items``: max number of matches returned from the ``match_list`` APIs,
      applied after sorting (so with the default sort, the best ``max_items``
      matches are returned). Defaults to ``None`` (unlimited).
    - ``casing``, ``unicode``, ``matching``, ``sort``: string literals (see the
      ``Casing``/``Unicode``/``Matching``/``Sort`` aliases below). Invalid strings
      raise ``ValueError`` listing the valid values.
    - ``scoring``: a `Scoring` instance (not a dict).

Haystacks
    The ``match_list*`` methods accept any iterable of ``str``, or a `Haystacks`
    — a Rust-owned copy of the list that skips the per-call boundary cost and
    releases the GIL while matching. Prefer `Haystacks` when matching the same
    list repeatedly (e.g. per keystroke).

Errors
    Invalid config strings (``casing``, ``unicode``, ``matching``, ``sort``) raise
    ``ValueError`` listing the valid values; non-``str`` haystacks raise
    ``TypeError``.
"""

from collections.abc import Iterable, Sequence
from typing import Literal, final

__version__: str

Casing = Literal["ignore", "smart", "respect"]
Unicode = Literal["ignore", "smart", "always"]
Matching = Literal["fuzzy", "exact", "prefix", "suffix", "substring"]
Sort = Literal[
    "score_then_index_asc", "score_then_index_desc", "index_asc", "index_desc"
]

@final
class Haystacks:
    """Owns the haystack list as a packed UTF-8 arena in Rust memory.

    The primary path for matching the same list repeatedly (e.g. per keystroke):
    matching pays no per-call boundary cost and releases the GIL. Construct once
    (copying each string once, under the GIL), ``append``/``extend`` to add
    items, and rebuild (or ``clear``) for anything else — match indices refer to
    this list's order.
    """

    def __init__(self, items: Iterable[str] | None = None) -> None: ...
    def append(self, item: str) -> None:
        """Appends a single haystack."""

    def extend(self, items: Iterable[str]) -> None:
        """Appends every haystack in the iterable (list/tuple are the fast paths)."""

    def clear(self) -> None:
        """Empties the list, keeping the allocation for reuse."""

    def __len__(self) -> int: ...

@final
class Scoring:
    """Controls the scoring used by the smith waterman algorithm.

    Defaults mirror the core crate. You may tweak these but pay close attention
    to the documentation for each property, as small changes can lead to poor
    matching.
    """

    def __init__(
        self,
        *,
        match_score: int = 12,
        mismatch_penalty: int = 6,
        gap_open_penalty: int = 5,
        gap_extend_penalty: int = 1,
        prefix_bonus: int = 12,
        capitalization_bonus: int = 4,
        matching_case_bonus: int = 4,
        exact_match_bonus: int = 8,
        delimiter_bonus: int = 4,
    ) -> None: ...
    @property
    def match_score(self) -> int:
        """Score for a matching character between needle and haystack."""

    @property
    def mismatch_penalty(self) -> int:
        """Penalty for a mismatch (substitution)."""

    @property
    def gap_open_penalty(self) -> int:
        """Penalty for opening a gap (deletion/insertion)."""

    @property
    def gap_extend_penalty(self) -> int:
        """Penalty for extending a gap (deletion/insertion)."""

    @property
    def prefix_bonus(self) -> int:
        """Bonus for matching the first character of the haystack (e.g. "h" on "hello_world")."""

    @property
    def capitalization_bonus(self) -> int:
        """Bonus for matching a capital letter after a lowercase letter.

        E.g. "b" on "fooBar" will receive a bonus on "B".
        """

    @property
    def matching_case_bonus(self) -> int:
        """Bonus for matching the case of the needle.

        E.g. "WorLd" on "WoRld" will receive a bonus on "W", "o", "d".
        """

    @property
    def exact_match_bonus(self) -> int:
        """Bonus for matching the exact needle (e.g. "foo" on "foo" will receive the bonus)."""

    @property
    def delimiter_bonus(self) -> int:
        """Bonus for matching *after* a delimiter character.

        E.g. "hw" on "hello_world" will give a bonus on "w".
        """

    def __eq__(self, other: object) -> bool: ...

@final
class Match:
    """Result of a fuzzy match, containing the score and index in the haystack.

    Frozen; supports ``==`` and ``hash()``.
    """

    @property
    def score(self) -> int:
        """Score of the match, higher is better."""

    @property
    def index(self) -> int:
        """Index of the match in the original list of haystacks."""

    @property
    def exact(self) -> bool:
        """Matched the needle exactly (e.g. "foo" on "foo")."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class MatchIndices:
    """Like `Match` but includes the indices of the chars in the haystack that
    matched the needle in reverse order."""

    @property
    def score(self) -> int:
        """Score of the match, higher is better."""

    @property
    def index(self) -> int:
        """Index of the match in the original list of haystacks."""

    @property
    def exact(self) -> bool:
        """Matched the needle exactly (e.g. "foo" on "foo")."""

    @property
    def indices(self) -> list[int]:
        """Indices of the chars in the haystack that matched the needle in reverse order."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Pattern:
    """A single pattern to match, optionally overriding parts of the matcher config.

    Frozen. The needle is matched literally (no query syntax); use `parse_query`
    for query syntax. Every per-pattern override defaults to ``None`` = "inherit
    from the matcher config" — including ``max_typos``: an unlimited per-pattern
    override cannot be expressed, so for unlimited typos set ``max_typos=None`` on
    the matcher config instead.
    """

    def __init__(
        self,
        needle: str,
        *,
        negated: bool = False,
        matching: Matching | None = None,
        max_typos: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        scoring: Scoring | None = None,
    ) -> None: ...
    @property
    def pattern(self) -> str:
        """Raw atom text, e.g. ``!^foo`` when parsed from a query."""

    @property
    def needle(self) -> str:
        """Text to match with the syntax stripped, e.g. ``foo``."""

    @property
    def negated(self) -> bool:
        """Haystacks matching this atom are excluded."""

    @property
    def matching(self) -> Matching | None:
        """Per-pattern override for ``matching``; ``None`` inherits the config."""

    @property
    def max_typos(self) -> int | None:
        """Per-pattern override for ``max_typos``; ``None`` inherits the config.

        Because the config's ``max_typos`` is itself optional, there is no way to
        request unlimited typos for a single pattern while the matcher's config
        sets a limit.
        """

    @property
    def casing(self) -> Casing | None:
        """Per-pattern override for ``casing``; ``None`` inherits the config."""

    @property
    def unicode(self) -> Unicode | None:
        """Per-pattern override for ``unicode``; ``None`` inherits the config."""

    @property
    def scoring(self) -> Scoring | None:
        """Per-pattern override for ``scoring``; ``None`` inherits the config."""

    def __eq__(self, other: object) -> bool: ...

@final
class Matcher:
    """Primary entrypoint for fuzzy matching.

    Compiles the pattern(s) once, allocates memory for the Smith Waterman matrix
    and reuses the selected SIMD backend. Ideally, only construct these at most
    once per list: they're cheap to construct, but end up being expensive if you
    construct them for each item in your list.

    ``match_list``/``match_list_indices``/``match_one`` hold the GIL and borrow
    the haystack strings zero-copy (CPython caches the UTF-8 representation on
    each ``str``, so only the first call on a given string pays for the cache
    fill). ``match_list_parallel`` borrows the haystacks the same way, then
    releases the GIL while matching on real threads.

    Any ``Sequence[str]`` (or iterable of ``str``) is accepted for haystacks, but
    ``list`` (and ``tuple``) are the fast paths.
    """

    def __init__(
        self,
        needle: str | Pattern,
        *,
        max_typos: int | None = 0,
        max_items: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> None:
        """Creates a matcher from a single pattern (str or Pattern).

        Strings match literally; use `from_query` for query syntax and
        `from_patterns` for multi-pattern queries.
        """

    @classmethod
    def from_query(
        cls,
        query: str,
        *,
        max_typos: int | None = 0,
        max_items: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> Matcher:
        """Shorthand for `from_patterns` with the parsed query (see `parse_query`).

        Special syntax changes each atom's matching mode: ``foo`` (fuzzy),
        ``^foo`` (prefix), ``foo$`` (suffix), ``'foo`` (substring), ``^foo$``
        (exact) and ``!foo`` (negated, substring unless combined with the syntax
        above). Any special character can be escaped with a backslash, e.g.
        ``\\!foo`` or ``foo\\$`` match the literal leading/trailing character.
        """

    @classmethod
    def from_patterns(
        cls,
        patterns: Sequence[str | Pattern],
        *,
        max_typos: int | None = 0,
        max_items: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> Matcher:
        """Creates a matcher from patterns (str or Pattern), matched independently.

        A haystack matches when all of the patterns match, where the score is the
        sum of each pattern's score. Plain strings are matched literally.
        """

    def match_list(self, haystacks: Sequence[str] | Haystacks) -> list[Match]:
        """Matches a list of haystacks, ordered by the ``sort`` strategy.

        This API provides the most performant path when matching on lists. Pass a
        `Haystacks` to release the GIL while matching; plain iterables are
        borrowed zero-copy under the GIL.
        """

    def match_list_parallel(
        self, haystacks: Sequence[str] | Haystacks, threads: int
    ) -> list[Match]:
        """Like `match_list` but matched in parallel on ``threads`` real threads.

        ``threads=0`` uses the available CPU cores - 2. Threads work on 2048 item
        chunks, and the final result is identical to `match_list`. The GIL is
        released while matching; plain iterables are first copied into a
        temporary arena under the GIL, which passing a `Haystacks` skips.
        """

    def match_list_indices(
        self, haystacks: Sequence[str] | Haystacks
    ) -> list[MatchIndices]:
        """Like `match_list` but each match includes the indices of the chars in
        the haystack that matched the needle.

        This API has not been optimized for performance, and should only be used
        on small lists, e.g. the visible portion of the results. Useful for
        displaying matched indices in the UI.
        """

    def match_one(self, haystack: str, index: int) -> Match | None:
        """Matches a single haystack, returning its match if it passes.

        ``index`` is echoed back on the returned match. This API performs ~10%
        slower than the `match_list` API. Consider using `match_list` if you
        have more than one haystack to match, as it performs significantly
        better.
        """

    def match_one_indices(self, haystack: str, index: int) -> MatchIndices | None:
        """Like `match_one` but includes the indices of the chars in the haystack
        that matched the needle."""

    def set_pattern(self, pattern: str | Pattern) -> None:
        """Updates the pattern, keeping the config. Strings are matched literally.

        Skipped if the pattern is the same as the previous one.
        """

    def set_patterns(self, patterns: Sequence[str | Pattern]) -> None:
        """Updates the patterns, keeping the config.

        Skipped if the patterns are the same as the previous ones.
        """

    def set_config(
        self,
        *,
        max_typos: int | None = 0,
        max_items: int | None = None,
        casing: Casing | None = None,
        unicode: Unicode | None = None,
        matching: Matching | None = None,
        sort: Sort | None = None,
        scoring: Scoring | None = None,
    ) -> None:
        """Updates the config, rebuilding it from the kwargs.

        Omitted kwargs reset to their defaults; this is not a merge with the
        current config. Skipped if the config is the same as the previous one.
        """

    @property
    def patterns(self) -> list[Pattern]:
        """The current patterns."""

    @property
    def max_typos(self) -> int | None:
        """Max characters missing from the needle; ``None`` means unlimited."""

    @property
    def max_items(self) -> int | None:
        """Max matches returned from the ``match_list`` APIs; ``None`` means unlimited."""

    @property
    def casing(self) -> Casing:
        """How case sensitivity is handled while matching."""

    @property
    def unicode(self) -> Unicode:
        """How unicode is handled while matching."""

    @property
    def matching(self) -> Matching:
        """The matching algorithm: fuzzy (Smith-Waterman) or a literal mode."""

    @property
    def sort(self) -> Sort:
        """How results are ordered."""

    @property
    def scoring(self) -> Scoring:
        """The Smith-Waterman scoring parameters."""

def parse_query(query: str) -> list[Pattern]:
    """Parses a query of whitespace separated atoms into patterns.

    See `Matcher.from_query` for the atom syntax. E.g. ``foo !^bar`` matches
    haystacks that fuzzy match ``foo`` and don't start with ``bar``. Escape a
    literal space with a backslash, e.g. ``foo\\ bar`` is a single atom. Atoms
    with an empty needle, e.g. ``!`` or ``^$``, are dropped.

    The returned patterns carry only the ``matching`` mode derived from the
    syntax. Any other per-pattern override is left as ``None`` and inherits the
    matcher's config; rebuild patterns to override per-pattern before passing
    them to `Matcher.from_patterns`.
    """

def max_needle_len(scoring: Scoring | None = None) -> int:
    """Needle length up to which scores are guaranteed to fit within the 16-bit
    score.

    Uses the default `Scoring` when omitted. Longer needles still match, but
    their scores may saturate at ``0xFFFF``.
    """
