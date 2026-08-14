# Frizbee (WASM)

WASM bindings for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy string matching. Requires [SIMD128 support](https://caniuse.com/wasm-simd) (Chrome 91+, Firefox 89+, Safari 16.4+, Node 16.4+).

```sh
npm install frizbee
```

## Usage

Call `await init()` once before use. In the browser, the WASM is fetched next to the JS module. In Node, it's read from disk. Explore the full API via the Typescript definitions.

```js
import init, { Matcher, Haystacks, parseQuery } from 'frizbee'
await init()

let matcher = new Matcher('fBr', { maxTypos: 1, casing: 'smart' })
let matches = matcher.matchList(['fooBar', 'foo_bar', 'barfoo', 'prelude'])
// [{ score: 53, index: 0, exact: false }, ...]

// Owned haystacks (recommended): copy to WASM memory once for much faster matching
const haystacks = new Haystacks(['fooBar', 'foo_bar', 'barfoo', 'prelude'])
matches = matcher.matchList(haystacks)

// Perform multi-pattern matching (whitespace separated) with syntax for controlling
// the matching mode:
// fuzzy  substring  prefix    suffix    exact    negated (combines with others)
// foo    'foo       ^foo      foo$      ^foo$    !foo
matcher = Matcher.fromQuery('foo !^bar')

// Per-pattern config overrides
const patterns = parseQuery('foo !^bar')
for (const pattern of patterns) {
  pattern.maxTypos = Math.floor(pattern.needle.length / 4)
}
matcher = Matcher.fromPatterns(patterns)
```

## Performance

Matching runs on WASM SIMD128 (16 byte lanes), roughly 60% slower than the native crate's AVX-512 throughput. The `string[]` path re-encodes every string on every call, so prefer an owned `Haystacks` when matching the same list repeatedly (e.g. per keystroke). Setting `limit` is essential for performance, as creating 100k `Match` instances costs >100ms. On the Chromium file list (1.4M haystacks, needle "linux", limit 1000):

```
new Haystacks(string[]):    156 ms (one-time copy)
matchList(string[]):        144 ms/iter
matchList(Haystacks):        42 ms/iter
```

`Matcher` and `Haystacks` instances live in the WASM heap. Call `.free()` when done, or rely on `FinalizationRegistry` to eventually collect them. A `Haystacks` holds a full copy of the haystack text. Match results are plain JS objects and need no freeing.

## Development

```sh
# use nix dev shell to get all clis/pkgs
nix develop

just build-wasm
just test-wasm
just bench-wasm
```
