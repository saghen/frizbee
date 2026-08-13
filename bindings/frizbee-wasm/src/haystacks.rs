//! A wasm-owned packed utf-8 arena, filled through wasm-bindgen's per-string
//! glue (`new`/`push`) or the wrapper's single `TextEncoder` pass
//! (`reserve`/`bytesView`/`offsetsView`/`commit`).

use js_sys::{Uint8Array, Uint32Array};
use wasm_bindgen::prelude::*;

/// Incrementally updatable Haystack copied into wasm memory to avoid per-call
/// overhead.
/// Match indices refer to this list's order
///
/// Instances hold memory in the wasm heap: call `.free()` when done, or rely on
/// `FinalizationRegistry` to eventually collect it.
#[wasm_bindgen]
pub struct Haystacks {
    /// Concatenated utf-8 bytes of every haystack
    bytes: Vec<u8>,
    /// Haystack `i` spans `bytes[offsets[i]..offsets[i + 1]]`; always `len + 1`
    /// entries, starting at 0
    offsets: Vec<u32>,
}

impl Haystacks {
    fn push_str(&mut self, item: &str) {
        self.bytes.extend_from_slice(item.as_bytes());
        self.offsets.push(self.bytes.len() as u32);
    }

    /// Views of every haystack, borrowed from the arena
    pub(crate) fn as_strs(&self) -> Vec<&str> {
        self.offsets
            .windows(2)
            // SAFETY: the arena only ever holds bytes copied from `&str`s or written
            // by `TextEncoder` (see `commit`), both valid utf-8
            .map(|span| unsafe {
                core::str::from_utf8_unchecked(&self.bytes[span[0] as usize..span[1] as usize])
            })
            .collect()
    }
}

#[wasm_bindgen]
impl Haystacks {
    /// Creates the list, optionally from an initial `string[]`
    #[wasm_bindgen(constructor)]
    pub fn new(items: Option<Vec<String>>) -> Haystacks {
        let mut haystacks = Haystacks {
            bytes: Vec::new(),
            offsets: vec![0],
        };
        if let Some(items) = items {
            haystacks.push(items);
        }
        haystacks
    }

    /// Appends haystacks. Through the package entry this is a single
    /// `TextEncoder` pass straight into the arena, while calling the native
    /// export directly pays wasm-bindgen's per-string glue instead
    pub fn push(&mut self, items: Vec<String>) {
        for item in &items {
            self.push_str(item);
        }
    }

    /// Empties the list, keeping the allocation for reuse
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.offsets.truncate(1);
    }

    /// Number of haystacks
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Low-level fill protocol, used by the package entry: reserves spare
    /// capacity for at least `bytes` more utf-8 bytes and `items` more
    /// haystacks, to be filled via `bytesView`/`offsetsView` and committed
    /// with `commit`. Prefer `push` unless you are writing glue code
    pub fn reserve(&mut self, bytes: usize, items: usize) {
        self.bytes.reserve(bytes);
        self.offsets.reserve(items);
    }

    /// View over the reserved spare byte capacity, to be filled with
    /// concatenated utf-8 haystacks. Detached by ANY call into the wasm
    /// module (memory may grow and move): take it after `reserve`, write,
    /// `commit`, and never touch it again
    #[wasm_bindgen(js_name = bytesView)]
    pub fn bytes_view(&mut self) -> Uint8Array {
        let len = self.bytes.len();
        let spare = self.bytes.capacity() - len;
        unsafe { Uint8Array::view_mut_raw(self.bytes.as_mut_ptr().add(len), spare) }
    }

    /// View over the reserved spare offset capacity, to be filled with each new
    /// haystack's END byte offset, relative to the start of the newly written
    /// bytes. Same detachment rules as `bytesView`
    #[wasm_bindgen(js_name = offsetsView)]
    pub fn offsets_view(&mut self) -> Uint32Array {
        let len = self.offsets.len();
        let spare = self.offsets.capacity() - len;
        unsafe { Uint32Array::view_mut_raw(self.offsets.as_mut_ptr().add(len), spare) }
    }

    /// Commits `bytes_written` bytes and `items` end-offsets written into the
    /// views. Offsets are validated (monotonically increasing, within
    /// `bytes_written`), so matching cannot read out of bounds. The bytes are
    /// NOT validated and MUST be valid utf-8 (e.g. written by `TextEncoder`,
    /// which only emits valid utf-8). Feeding invalid utf-8 is undefined
    /// behavior
    pub fn commit(&mut self, bytes_written: usize, items: usize) -> Result<(), JsError> {
        if bytes_written > self.bytes.capacity() - self.bytes.len()
            || items > self.offsets.capacity() - self.offsets.len()
        {
            return Err(JsError::new("commit exceeds the reserved capacity"));
        }

        let base = self.bytes.len() as u32;
        let start = self.offsets.len();
        // SAFETY: the caller wrote `items` offsets into the spare capacity via
        // `offsetsView`; the bounds were checked above
        unsafe { self.offsets.set_len(start + items) };

        // Validate before adjusting, leaving the arena untouched on error
        let mut prev = 0u32;
        let mut invalid = None;
        for (i, &offset) in self.offsets[start..].iter().enumerate() {
            if offset < prev || offset as usize > bytes_written {
                invalid = Some((i, offset));
                break;
            }
            prev = offset;
        }
        if let Some((i, offset)) = invalid {
            self.offsets.truncate(start);
            return Err(JsError::new(&format!(
                "offset {i} ({offset}) must be monotonically increasing and within \
                 {bytes_written} bytes"
            )));
        }

        for offset in &mut self.offsets[start..] {
            *offset += base;
        }
        // The arena ends at the last haystack's end; `bytes_written` only bounds the
        // offsets. SAFETY: the caller wrote at least `prev <= bytes_written` bytes
        unsafe { self.bytes.set_len((base + prev) as usize) };
        Ok(())
    }
}
