// Transparent fast paths for the wasm-pack output:
// - `Haystacks` construction/push encode each string directly into the wasm-owned
//   arena with a single TextEncoder pass (see `fill`), skipping wasm-bindgen's
//   per-string glue (encodeInto + wasm malloc per string).
// - `matchList`/`matchListIndices` accept a `Haystacks` (forwarded to
//   `matchHaystacks`) or a `string[]` (packed into a reused scratch `Haystacks`).
// TextEncoder only ever emits valid utf-8, satisfying `commit`'s contract. Same
// arguments, same results as the native exports.
//
// Hand-written package entry (package.json routes `default` imports here); the
// generated wasm-pack output lives in ./pkg. Everything else re-exports from the
// generated module unchanged.
export * from './pkg/frizbee_wasm.js'
export { default } from './pkg/frizbee_wasm.js'
import { Matcher, Haystacks as NativeHaystacks } from './pkg/frizbee_wasm.js'

const encoder = new TextEncoder()

/** Encodes `strings` straight into the arena's reserved spare capacity — one
 * encodeInto per string, no intermediate buffer. No wasm calls may happen between
 * taking the views and `commit`: wasm memory growth would detach them, which is
 * why `reserve` takes the utf-8 worst case (3 bytes per UTF-16 code unit) up front */
function fill(haystacks, strings) {
  let worstCase = 0
  for (const s of strings) worstCase += s.length * 3
  haystacks.reserve(worstCase, strings.length)
  const bytes = haystacks.bytesView()
  const offsets = haystacks.offsetsView()
  let pos = 0
  for (let i = 0; i < strings.length; i++) {
    pos += encoder.encodeInto(strings[i], bytes.subarray(pos)).written
    offsets[i] = pos
  }
  haystacks.commit(pos, strings.length)
  return haystacks
}

export class Haystacks extends NativeHaystacks {
  constructor(items) {
    super()
    if (items?.length) fill(this, items)
  }

  push(items) {
    fill(this, items)
  }
}

// Scratch arena for the plain string[] paths, reused across calls (and matchers):
// a keystroke loop allocates nothing after the first call. Retained for the life
// of the module, sized by the largest list seen
let scratch
function packed(strings) {
  scratch ??= new Haystacks()
  scratch.clear()
  return fill(scratch, strings)
}

Matcher.prototype.matchList = function matchList(haystacks) {
  return this.matchHaystacks(
    haystacks instanceof NativeHaystacks ? haystacks : packed(haystacks),
  )
}

Matcher.prototype.matchListIndices = function matchListIndices(haystacks) {
  return this.matchHaystacksIndices(
    haystacks instanceof NativeHaystacks ? haystacks : packed(haystacks),
  )
}
