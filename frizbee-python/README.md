# frizbee-rs

Python bindings for [frizbee](https://github.com/saghen/frizbee), a fast SIMD fuzzy string matcher written in Rust.

Frizbee uses Smith-Waterman with affine gaps for typo-resistant fuzzy matching, similar to FZF/FZY but faster. In benchmarks it outperforms [nucleo](https://github.com/helix-editor/nucleo) by ~1.7x and [fzf](https://github.com/junegunn/fzf) by ~2.1x.

## Installation

```bash
pip install frizbee
```

## Quick Start

```python
import frizbee_rs

results = frizbee.match_list("fBr", ["fooBar", "foo_bar", "prelude", "println!"])
for m in results:
    print(f"index={m.index}, score={m.score}, exact={m.exact}")
# index=0, score=53, exact=False
# index=1, score=48, exact=False
```

## API

### Functions

#### `match_list(needle, haystacks, config=None) -> list[Match]`

Fuzzy match a needle against a list of haystacks. Returns matches sorted by score (descending) by default.

```python
results = frizbee.match_list("foo", ["foobar", "baz", "foo"])
# [Match(score=68, index=2, exact=True), Match(score=60, index=0, exact=False)]
```

#### `match_list_indices(needle, haystacks, config=None) -> list[MatchIndices]`

Like `match_list` but also returns the indices of matched characters in each haystack.

```python
results = frizbee.match_list_indices("fb", ["foobar"])
# [MatchIndices(score=29, index=0, exact=False, indices=[3, 0])]
```

#### `match_list_parallel(needle, haystacks, config=None, threads=None) -> list[Match]`

Multithreaded version of `match_list`. Defaults to all available cores.

```python
results = frizbee.match_list_parallel("query", big_list, threads=8)
```

### Classes

#### `Matcher(needle, config=None)`

Stateful matcher for reusing a needle against multiple haystack lists.

```python
m = frizbee.Matcher("foo")
results1 = m.match_list(["foobar", "baz"])
results2 = m.match_list(["food", "drink"])

m.set_needle("bar")  # change needle
m.set_config(frizbee.Config(max_typos=1))  # change config
```

#### `Config(max_typos=0, sort=True, scoring=None)`

| Parameter   | Default | Description                                                     |
|-------------|---------|-----------------------------------------------------------------|
| `max_typos` | `0`     | Max missing chars before filtering out. `None` = unlimited.     |
| `sort`      | `True`  | Sort results by score (descending).                             |
| `scoring`   | default | `Scoring` instance for fine-grained control.                    |

```python
# Allow typos
cfg = frizbee_rs.Config(max_typos=2)

# Disable sorting
cfg = frizbee_rs.Config(sort=False)
```

#### `Scoring(...)`

All parameters are optional and default to frizbee's built-in values.

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `match_score`          | 12      | Score for a matching character.                |
| `mismatch_penalty`     | 6       | Penalty for a substitution.                    |
| `gap_open_penalty`     | 5       | Penalty for opening a gap.                     |
| `gap_extend_penalty`   | 1       | Penalty for extending a gap.                   |
| `prefix_bonus`         | 12      | Bonus for matching the first character.        |
| `capitalization_bonus` | 4       | Bonus for matching camelCase boundaries.       |
| `matching_case_bonus`  | 4       | Bonus for matching the case of the needle.     |
| `exact_match_bonus`    | 8       | Bonus for an exact match.                      |
| `delimiter_bonus`      | 4       | Bonus for matching after a delimiter (`_` etc).|

#### `Match`

| Attribute | Type   | Description                                 |
|-----------|--------|---------------------------------------------|
| `score`   | `int`  | Match score (higher is better).             |
| `index`   | `int`  | Index in the original haystack list.        |
| `exact`   | `bool` | Whether the needle matched exactly.         |

#### `MatchIndices`

Same as `Match` plus:

| Attribute | Type        | Description                                      |
|-----------|-------------|--------------------------------------------------|
| `indices` | `list[int]` | Positions of matched characters in the haystack.  |

## License

MIT
