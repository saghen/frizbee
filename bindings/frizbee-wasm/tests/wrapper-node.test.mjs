// Proves the published package works in plain Node: the `node` exports condition
// routes to wrapper-node.mjs, whose no-argument `init()` must load the wasm from
// disk (the web-target glue's default is a fetch(file:// URL) that Node rejects).
//
// Requires the web build: wasm-pack build --target web --release
// Run: node --test tests/wrapper-node.test.mjs
import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

if (!existsSync(new URL('../pkg/frizbee_wasm_bg.wasm', import.meta.url))) {
  console.error('missing pkg/ — run: wasm-pack build --target web --release')
  process.exit(1)
}

test('package.json routes node imports to wrapper-node.mjs', async () => {
  const manifest = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))
  assert.equal(manifest.name, 'frizbee')
  assert.equal(manifest.exports['.'].node, './wrapper-node.mjs')
  assert.equal(manifest.exports['.'].default, './wrapper.mjs')
  assert.equal(manifest.exports['./frizbee_wasm_bg.wasm'], './pkg/frizbee_wasm_bg.wasm')
  const shipped = [
    'pkg/frizbee_wasm_bg.wasm',
    'pkg/frizbee_wasm.js',
    'pkg/frizbee_wasm.d.ts',
    'wrapper.mjs',
    'wrapper-node.mjs',
  ]
  for (const file of shipped) assert.ok(manifest.files.includes(file), `files must include ${file}`)
  assert.ok(manifest.sideEffects.includes('./wrapper-node.mjs'))
})

test('no-arg init() loads the wasm from disk; wrapper surface works', async () => {
  const pkg = await import('../wrapper-node.mjs')
  await pkg.default()

  const matcher = new pkg.Matcher('foo', { maxTypos: 0 })
  const matches = matcher.matchList(['foo', 'fooBar', 'bar'])
  assert.deepStrictEqual(matches.map((m) => m.index).sort(), [0, 1])

  const owned = new pkg.Haystacks(['foo', 'bar'])
  assert.deepStrictEqual(matcher.matchList(owned).map((m) => m.index), [0])
  owned.free()
  matcher.free()
})
