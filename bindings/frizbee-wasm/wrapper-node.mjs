// Node entry for the published package, selected by the `node` exports condition
// (see package.json). Same surface as wrapper.mjs, except a no-argument `init()`
// reads the wasm from disk: the web-target glue defaults to
// fetch(new URL('frizbee_wasm_bg.wasm', import.meta.url)), and Node's fetch
// rejects file:// URLs. The entry is chosen at resolution time, so bundlers
// targeting the web never see this file or its node:fs import.
export * from './wrapper.mjs'
import initNative from './wrapper.mjs'
import { readFile } from 'node:fs/promises'

export default async function init(module_or_path) {
  if (module_or_path === undefined) {
    module_or_path = {
      module_or_path: await readFile(new URL('./pkg/frizbee_wasm_bg.wasm', import.meta.url)),
    }
  }
  return initNative(module_or_path)
}
