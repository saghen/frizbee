pub mod bucket;
mod indices;
mod matcher;
pub mod matcher_v2;
mod parallel;

pub use indices::match_indices;
pub use matcher::match_list;
pub use parallel::match_list_parallel;

pub trait Appendable<T> {
    fn append(&mut self, value: T);
}

impl<T> Appendable<T> for Vec<T> {
    fn append(&mut self, value: T) {
        self.push(value);
    }
}

const MAX_MATRIX_BYTES: usize = 32 * 1024; // 32 KB
#[inline(always)]
pub(crate) fn match_too_large(needle: &str, haystack: &str) -> bool {
    let max_haystack_len = MAX_MATRIX_BYTES / needle.len().max(1) / 2; // divide by 2 since we use u16
    haystack.len() > max_haystack_len
}
