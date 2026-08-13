# Frizbee (WASM)

WASM bindings for [frizbee](https://github.com/saghen/frizbee), SIMD fuzzy string matching. Requires [SIMD128 support](https://caniuse.com/wasm-simd) (Chrome 91+, Firefox 89+, Safari 16.4+, Node 16.4+).

```sh
npm install frizbee
```

## Usage

<!-- owner: initialization notes per bundler: `--target web` output requires calling
`await init()` once before use; vite needs `?url` or `vite-plugin-wasm` notes if
applicable. Show the basic matchList flow and point at the .d.ts for the full API -->

```js
import init, { Haystacks, Matcher } from 'frizbee'
await init()

const matcher = new Matcher('fBr', { maxTypos: 0, maxItems: 1000 })
const matches = matcher.matchList(['fooBar', 'foo_bar', 'prelude'])
// [{ score: 53, index: 0, exact: false }]

// Owned haystacks (recommended): copy to wasm memory once for ~1.5x faster matching
const haystacks = new Haystacks(items) // encoded into wasm memory once
const matcher = new Matcher('fBr', { maxItems: 1000 })
const matches = matcher.matchList(haystacks) // no boundary cost per call

haystacks.push(['more', 'items'])
```

## Performance

<!-- owner: report bench.mjs numbers (matchList(string[]) vs matchList(Haystacks) on
the Chromium case), note that wasm simd128 uses 16-byte lanes vs AVX2/AVX-512
natively, and that the boundary (string re-encoding) dominates the string[] path -->

<!-- owner: Matcher and Haystacks instances live in the wasm heap; call `.free()` (or
use `using`/explicit resource management) when done, otherwise FinalizationRegistry
cleans up eventually. A Haystacks holds a full copy of the haystack text. Note match
results are plain JS objects and need no freeing -->

## Development

<!-- owner: build/test/bench commands, e.g.:
  just build-wasm   (wasm-pack build --target web --release)
  just test-wasm
  just bench-wasm
The binding directory is the npm package root: package.json is committed, the
wrappers sit beside it, and wasm-pack output lands in ./pkg -->
