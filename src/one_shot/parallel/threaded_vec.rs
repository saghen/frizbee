use std::{
    alloc::{Layout, alloc, dealloc},
    cell::RefCell,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::one_shot::Appendable;

thread_local!(static THREAD_IDX: RefCell<Option<usize>> = const { RefCell::new(None) });

/// Thread-safe append-only expandable vec for parallel matching in cases where there won't be
/// too many writes and the number of elements cannot be known ahead of time.
/// !!! Only one of these can exist at a time per thread !!!
#[derive(Debug)]
pub(crate) struct ThreadedVec<T> {
    data: Mutex<Vec<T>>,
    thread_batches: Vec<ThreadBatch<T>>,
    thread_idx: AtomicUsize,
    thread_count: usize,
}

unsafe impl<T: Send> Send for ThreadedVec<T> {}
unsafe impl<T: Sync> Sync for ThreadedVec<T> {}

impl<T> ThreadedVec<T> {
    pub fn new(batch_size: usize, thread_count: usize) -> Self {
        let mut batches = Vec::with_capacity(thread_count);
        for _ in 0..thread_count {
            batches.push(ThreadBatch::new(batch_size));
        }
        ThreadedVec {
            data: Mutex::new(vec![]),
            thread_batches: batches,
            thread_idx: AtomicUsize::new(0),
            thread_count,
        }
    }

    fn get_batch_idx(&mut self) -> usize {
        THREAD_IDX.with(|idx| {
            let mut thread_idx = idx.borrow_mut();

            match thread_idx.as_ref() {
                Some(idx) => *idx,
                None => {
                    loop {
                        let current_thread_idx = self.thread_idx.load(Ordering::Relaxed);
                        assert!(
                            current_thread_idx < self.thread_count,
                            "too many threads attempting to use expandable batched vec"
                        );

                        match self.thread_idx.compare_exchange(
                            current_thread_idx,
                            current_thread_idx + 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => {
                                thread_idx.replace(current_thread_idx);
                                return current_thread_idx;
                            }
                            // Contention: another thread succeeded. Retry loop.
                            Err(_) => continue,
                        }
                    }
                }
            }
        })
    }

    /// Pushes a single element to the current batch, creating a new thread-local batch if needed
    pub fn push(&mut self, value: T) {
        // Get the batch
        let batch_idx = self.get_batch_idx();
        let batches_ptr = self.thread_batches.as_ptr();
        let batch = unsafe { &mut *(batches_ptr.add(batch_idx) as *mut ThreadBatch<T>) };

        // Push the element
        unsafe { batch.data.add(batch.pos).write(value) };
        batch.pos += 1;

        // Write to the main vector if the batch is full
        if batch.is_full() {
            self.expand_by_batch(batch);
            batch.pos = 0;
        }
    }

    /// Expands the queue by the provided batch
    fn expand_by_batch(&self, batch: &ThreadBatch<T>) {
        // Ensure there's something to copy
        let count = batch.pos;
        if count == 0 {
            return;
        }

        // Reserve space for the new elements
        let mut data = self.data.lock().unwrap();
        data.reserve(count);

        // Copy new elements into the vector
        let old_len = data.len();
        unsafe {
            data.set_len(old_len + count);
            std::ptr::copy_nonoverlapping(batch.data, data.as_mut_ptr().add(old_len), count);
        }
    }

    /// Consumes the vector and converts it into `Vec<T>`. We must clean up the dangling
    /// commited slots with no elements
    pub fn into_vec(self) -> Vec<T> {
        for batch in self.thread_batches.iter() {
            self.expand_by_batch(batch);
        }

        self.data.into_inner().unwrap()
    }
}

impl<T> Appendable<T> for Arc<ThreadedVec<T>> {
    fn append(&mut self, value: T) {
        #[cfg(feature = "simd")]
        unsafe {
            Arc::get_mut_unchecked(self).push(value)
        };
        #[cfg(not(feature = "simd"))]
        Arc::get_mut(self)
            .expect("Failed to acquire the threaded vec")
            .push(value);
    }
}

#[derive(Clone, Debug)]
struct ThreadBatch<T> {
    pub data: *mut T,
    pub pos: usize,
    pub len: usize,
}

impl<T> ThreadBatch<T> {
    pub fn new(len: usize) -> Self {
        let layout = Layout::array::<T>(len).unwrap();
        assert!(
            layout.size() > 0,
            "Batch size and elements must be greater than 0"
        );

        let data = unsafe { alloc(Layout::array::<T>(len).unwrap()) };
        assert!(!data.is_null(), "Failed to allocate memory for batch");

        ThreadBatch {
            data: data as *mut T,
            pos: 0,
            len,
        }
    }

    pub fn is_full(&self) -> bool {
        self.pos == self.len
    }
}

impl<T> Drop for ThreadBatch<T> {
    fn drop(&mut self) {
        if self.data.is_null() {
            return;
        }

        unsafe {
            // Drop the initialized elements
            let slice_ptr = std::slice::from_raw_parts_mut(self.data, self.pos);
            std::ptr::drop_in_place(slice_ptr);

            // Deallocate the memory
            dealloc(self.data as *mut u8, Layout::array::<T>(self.len).unwrap())
        };

        // Avoid double-free on panic during drop_in_place etc.
        self.data = std::ptr::null_mut();
    }
}
