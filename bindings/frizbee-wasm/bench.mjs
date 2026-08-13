import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))

const dataPath = join(here, '../../benches/data/chromium.txt')
if (!existsSync(dataPath)) {
  console.log('run benches/data/download.sh')
  process.exit(0)
}

if (!existsSync(join(here, 'pkg/frizbee_wasm.js'))) {
  console.log('run: wasm-pack build --target web --release')
  process.exit(1)
}

// the node entry inits the web build from disk
const { default: init, Matcher, Haystacks } = await import('./wrapper-node.mjs')
await init()

// drop the empty string after the trailing newline
const haystacks = readFileSync(dataPath, 'utf8').split('\n')
if (haystacks.at(-1) === '') haystacks.pop()

const WARMUP = 20
const ITERS = 100
const MAX_ITEMS = 1000

const pad = (name) => `${name}:`.padEnd(36)
const fmt = (ms) => ms.toFixed(2).padStart(8)

function bench(name, fn) {
  for (let i = 0; i < WARMUP; i++) fn()
  const times = []
  for (let i = 0; i < ITERS; i++) {
    const start = performance.now()
    fn()
    times.push(performance.now() - start)
  }
  const mean = times.reduce((a, b) => a + b, 0) / times.length
  console.log(`${pad(name)}${fmt(mean)} ms/iter (min ${fmt(Math.min(...times))} ms)`)
}

const matcher = new Matcher('linux', { maxItems: MAX_ITEMS })
// report match count
const matches = matcher.matchList(haystacks)
console.log(
  `chromium: ${haystacks.length} haystacks -> ${matches.length} matches, ` +
    `needle "linux", max_items = ${MAX_ITEMS}`,
)

const start = performance.now()
const owned = new Haystacks(haystacks)
console.log(`${pad('new Haystacks(string[])')}${fmt(performance.now() - start)} ms`)

bench('match_list (string[])', () => matcher.matchList(haystacks))
bench('match_list (Haystacks)', () => matcher.matchList(owned))

owned.free()
matcher.free()
