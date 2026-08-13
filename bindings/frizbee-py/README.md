# Frizbee (Python bindings)

Python bindings for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy string matching. Supports CPython 3.10+ on linux x64/arm64 (glibc + musl), macOS x64/arm64 and windows x64.

```sh
pip install frizbee
```

## Usage

```python
import frizbee

matcher = frizbee.Matcher("fBr", max_typos=1, casing="smart")
matches = matcher.match_list(["fooBar", "foo_bar", "barfoo", "prelude"])
# [Match(score=53, index=0, exact=False), ...]

# Owned haystacks (recommended): copy to Rust once for much faster matching
haystacks = frizbee.Haystacks(["fooBar", "foo_bar", "barfoo", "prelude"])
matches = matcher.match_list(haystacks)

# Perform multi-pattern matching (whitespace separated) with syntax for controlling
# the matching mode:
# fuzzy  substring  prefix    suffix    exact    negated (combines with others)
# foo    'foo       ^foo      foo$      ^foo$    !foo
matcher = frizbee.Matcher.from_query("foo !^bar")

# Per-pattern config overrides
matcher = frizbee.Matcher.from_patterns([
    frizbee.Pattern("foo", max_typos=1),
    frizbee.Pattern("bar", negated=True, matching="prefix"),
])
```

## Development

<!-- venv + `maturin develop` + `python -m unittest discover tests`, bench.py -->
