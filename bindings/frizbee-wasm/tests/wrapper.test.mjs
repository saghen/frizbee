// Asserts wrapper.mjs's fast paths — the encode-into-arena Haystacks fill and the
// matchList/matchListIndices dispatch — return exactly what the native wasm-bindgen
// glue paths return.
//
// Requires the web build: wasm-pack build --target web --release
// Run: node --test tests/wrapper.test.mjs
import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

if (!existsSync(new URL('../pkg/frizbee_wasm.js', import.meta.url))) {
  console.error('missing pkg/ — run: wasm-pack build --target web --release')
  process.exit(1)
}

// import the glue directly and capture the native paths' results before the
// wrapper patches the prototypes (same module instance once wrapper.mjs loads it)
const native = await import('../pkg/frizbee_wasm.js')
await native.default({
  module_or_path: await readFile(new URL('../pkg/frizbee_wasm_bg.wasm', import.meta.url)),
})

// empty strings and unicode exercise the fill path; a lone surrogate encodes to
// U+FFFD on both paths
const haystacks = ['', 'foo', 'fooBar', '🚀 foo', '日本語のfooテキスト', 'lone\uD800foo', 'bar']

const matcher = new native.Matcher('foo', { maxTypos: 1 })
const nativeMatches = matcher.matchList(haystacks)
const nativeIndices = matcher.matchListIndices(haystacks)

const wrapper = await import('../wrapper.mjs')

// spy on matchHaystacks to prove the wrapper's fast path is actually taken
let haystacksCalls = 0
for (const method of ['matchHaystacks', 'matchHaystacksIndices']) {
  const original = native.Matcher.prototype[method]
  native.Matcher.prototype[method] = function (...args) {
    haystacksCalls += 1
    return original.apply(this, args)
  }
}

test('wrapper matchList/matchListIndices equal the native results', () => {
  assert.deepStrictEqual(matcher.matchList(haystacks), nativeMatches)
  assert.deepStrictEqual(matcher.matchListIndices(haystacks), nativeIndices)
  assert.ok(haystacksCalls > 0, 'wrapper did not route through matchHaystacks')
})

test('owned Haystacks equal the native results', () => {
  // the wrapper constructor encodes straight into the arena; growth across pushes
  // exercises reserve/commit on a non-empty arena
  const owned = new wrapper.Haystacks(haystacks.slice(0, 3))
  owned.push(haystacks.slice(3))
  assert.equal(owned.length, haystacks.length)

  assert.deepStrictEqual(matcher.matchList(owned), nativeMatches)
  assert.deepStrictEqual(matcher.matchHaystacks(owned), nativeMatches)
  assert.deepStrictEqual(matcher.matchListIndices(owned), nativeIndices)

  owned.clear()
  assert.equal(owned.length, 0)
  assert.deepStrictEqual(matcher.matchList(owned), [])

  owned.free()
  matcher.free()
})
