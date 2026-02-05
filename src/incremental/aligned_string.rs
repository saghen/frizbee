use std::alloc::{Layout, alloc, dealloc};

/// Byte slice aligned to 16 bytes allowing us to use aligned SIMD loads
#[derive(Debug)]
pub(crate) struct AlignedBytes {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

impl AlignedBytes {
    pub fn new(s: &[u8]) -> Self {
        let len = s.len();
        let layout = Layout::from_size_align(len.max(1), 16).expect("Invalid layout");

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            std::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len);

            AlignedBytes { ptr, len, layout }
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}
