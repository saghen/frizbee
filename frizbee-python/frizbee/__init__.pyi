class Scoring:
    """Scoring parameters for the Smith-Waterman fuzzy matching algorithm."""

    match_score: int
    """Score for a matching character between needle and haystack."""
    mismatch_penalty: int
    """Penalty for a mismatch (substitution)."""
    gap_open_penalty: int
    """Penalty for opening a gap (deletion/insertion)."""
    gap_extend_penalty: int
    """Penalty for extending a gap (deletion/insertion)."""
    prefix_bonus: int
    """Bonus for matching the first character of the haystack."""
    capitalization_bonus: int
    """Bonus for matching a capital letter after a lowercase letter."""
    matching_case_bonus: int
    """Bonus for matching the case of the needle."""
    exact_match_bonus: int
    """Bonus for matching the exact needle."""
    delimiter_bonus: int
    """Bonus for matching after a delimiter character."""

    def __init__(
        self,
        match_score: int | None = None,
        mismatch_penalty: int | None = None,
        gap_open_penalty: int | None = None,
        gap_extend_penalty: int | None = None,
        prefix_bonus: int | None = None,
        capitalization_bonus: int | None = None,
        matching_case_bonus: int | None = None,
        exact_match_bonus: int | None = None,
        delimiter_bonus: int | None = None,
    ) -> None:
        """Create a Scoring config. All parameters default to frizbee's defaults."""
        ...

class Config:
    """Configuration for fuzzy matching."""

    max_typos: int | None
    """Maximum number of typos (missing chars) before filtering out. None = unlimited."""
    sort: bool
    """Whether to sort results by score (descending)."""
    scoring: Scoring
    """Scoring parameters for the algorithm."""

    def __init__(
        self,
        max_typos: int | None = 0,
        sort: bool = True,
        scoring: Scoring | None = None,
    ) -> None:
        """Create a Config. Defaults match frizbee's defaults (max_typos=0, sort=True)."""
        ...

class Match:
    """A single fuzzy match result."""

    score: int
    """Match score (higher is better)."""
    index: int
    """Index of the matched item in the original haystack list."""
    exact: bool
    """Whether the needle matched the haystack exactly."""

class MatchIndices:
    """A fuzzy match result with character match indices."""

    score: int
    """Match score (higher is better)."""
    index: int
    """Index of the matched item in the original haystack list."""
    exact: bool
    """Whether the needle matched the haystack exactly."""
    indices: list[int]
    """Indices of matched characters in the haystack (in reverse order)."""

class Matcher:
    """Stateful fuzzy matcher. Reuse for matching one needle against many haystacks."""

    def __init__(self, needle: str, config: Config | None = None) -> None:
        """Create a new Matcher for the given needle."""
        ...
    def set_needle(self, needle: str) -> None:
        """Update the needle to match against."""
        ...
    def set_config(self, config: Config) -> None:
        """Update the matching configuration."""
        ...
    def match_list(self, haystacks: list[str]) -> list[Match]:
        """Match the needle against a list of haystacks.

        Returns a list of Match objects for items that matched.
        Results are sorted by score (descending) if config.sort is True.
        """
        ...
    def match_list_indices(self, haystacks: list[str]) -> list[MatchIndices]:
        """Match the needle against a list of haystacks, returning match indices.

        Returns a list of MatchIndices objects with the positions of matched characters.
        Results are sorted by score (descending) if config.sort is True.
        """
        ...

def match_list(
    needle: str,
    haystacks: list[str],
    config: Config | None = None,
) -> list[Match]:
    """Fuzzy match a needle against a list of haystacks.

    Returns a list of Match objects for items that matched, sorted by score
    (descending) by default.

    Args:
        needle: The search string.
        haystacks: List of strings to match against.
        config: Optional matching configuration.
    """
    ...

def match_list_indices(
    needle: str,
    haystacks: list[str],
    config: Config | None = None,
) -> list[MatchIndices]:
    """Fuzzy match a needle against a list of haystacks, returning match indices.

    Returns a list of MatchIndices objects with the positions of matched characters.

    Args:
        needle: The search string.
        haystacks: List of strings to match against.
        config: Optional matching configuration.
    """
    ...

def match_list_parallel(
    needle: str,
    haystacks: list[str],
    config: Config | None = None,
    threads: int | None = None,
) -> list[Match]:
    """Fuzzy match a needle against a list of haystacks using multiple threads.

    Like match_list but parallelized for large haystack lists.

    Args:
        needle: The search string.
        haystacks: List of strings to match against.
        config: Optional matching configuration.
        threads: Number of threads to use. Defaults to available parallelism.
    """
    ...
