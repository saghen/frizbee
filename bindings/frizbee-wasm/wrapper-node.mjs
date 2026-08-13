// Special entry for node to ensure wasm loading works
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
