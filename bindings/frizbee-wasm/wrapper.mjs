export * from './pkg/frizbee_wasm.js'
export { default } from './pkg/frizbee_wasm.js'
import { Matcher, Haystacks as NativeHaystacks } from './pkg/frizbee_wasm.js'

const encoder = new TextEncoder()

/** Encode strings directly into WASM memory */
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

// Scratch arena for the plain string[] paths, reused across calls (and matchers).
// Retained for the life of the module, sized by the largest list seen
let scratch
function packed(strings) {
  scratch ??= new Haystacks()
  scratch.clear()
  return fill(scratch, strings)
}

Matcher.prototype.matchList = function matchList(haystacks) {
  return this.matchHaystacks(haystacks instanceof NativeHaystacks ? haystacks : packed(haystacks))
}

Matcher.prototype.matchListIndices = function matchListIndices(haystacks) {
  return this.matchHaystacksIndices(
    haystacks instanceof NativeHaystacks ? haystacks : packed(haystacks),
  )
}
