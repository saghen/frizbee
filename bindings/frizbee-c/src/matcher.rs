//! The matcher handle, the match result types and every matching `extern "C"` fn,
//! including the `*_free` functions releasing the result buffers.

use std::ptr;

use frizbee::Matcher;

use crate::config::{config_to_core, frizbee_config_t, frizbee_str_t};

/// Opaque matcher handle, created by `frizbee_matcher_new` or
/// `frizbee_matcher_from_query` and destroyed by `frizbee_matcher_free`.
///
/// Compiles the pattern once, allocates memory for the Smith Waterman matrix, and
/// reuses the selected SIMD backend across calls. Ideally, only construct these at
/// most once per list: they're cheap to construct, but end up being expensive if
/// you construct them for each item in your list.
///
/// Not thread-safe: use one matcher per thread or use external locking
pub struct frizbee_matcher_t {
    inner: Matcher,
}

/// Result of a fuzzy match, containing the score and index in the haystack
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct frizbee_match_t {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
}

/// A list of matches owned by frizbee
///
/// Release with `frizbee_matches_free`
#[repr(C)]
#[derive(Debug)]
pub struct frizbee_matches_t {
    pub items: *mut frizbee_match_t,
    pub len: usize,
}

/// Like `frizbee_match_t` but includes the indices of the chars in the haystack
/// that matched the needle in reverse order. Match `i` of
/// `frizbee_match_indices_list_t` owns
/// `indices[items[i].indices_start .. items[i].indices_start + items[i].indices_len]`
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct frizbee_match_indices_t {
    pub score: u16,
    /// Index of the match in the original list of haystacks
    pub index: u32,
    /// Matched the needle exactly (e.g. "foo" on "foo")
    pub exact: bool,
    /// Start of this match's slice of the shared `indices` buffer
    pub indices_start: u32,
    /// Length of this match's slice of the shared `indices` buffer
    pub indices_len: u32,
}

/// A list of matches with indices, owned by frizbee. All matches share the one
/// `indices` buffer.
///
/// Release with `frizbee_match_indices_list_free`
#[repr(C)]
#[derive(Debug)]
pub struct frizbee_match_indices_list_t {
    pub items: *mut frizbee_match_indices_t,
    pub len: usize,
    pub indices: *mut u32,
    pub indices_len: usize,
}

/// SAFETY: the caller promises `s.ptr` points to `s.len` bytes of valid UTF-8
unsafe fn str_from_c<'a>(s: frizbee_str_t) -> &'a str {
    if s.len == 0 {
        return "";
    }
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(s.ptr.cast(), s.len)) }
}

/// `frizbee_str_t` usable as a core-matcher haystack. `repr(transparent)` so the
/// caller's `frizbee_str_t` array can be reinterpreted as `&[haystack_str_t]` and
/// passed to the core matcher without a per-call copy
#[repr(transparent)]
struct haystack_str_t(frizbee_str_t);

impl AsRef<str> for haystack_str_t {
    fn as_ref(&self) -> &str {
        // SAFETY: the caller promised the bytes are valid UTF-8 (see `frizbee_str_t`)
        unsafe { str_from_c(self.0) }
    }
}

// SAFETY: threads only read the caller's bytes, which must stay valid and
// unmodified for the duration of the call
unsafe impl Sync for haystack_str_t {}

/// SAFETY: the caller promises `haystacks` points to `len` valid `frizbee_str_t`
unsafe fn haystacks_from_c<'a>(haystacks: *const frizbee_str_t, len: usize) -> &'a [haystack_str_t] {
    if len == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(haystacks.cast(), len) }
}

/// The buffers cross the ABI as boxed slices so the free functions can rebuild
/// them from pointer + length alone, without carrying capacities in the structs
fn box_into_raw_parts<T>(slice: Box<[T]>) -> (*mut T, usize) {
    let len = slice.len();
    (Box::into_raw(slice) as *mut T, len)
}

fn match_to_c(m: &frizbee::Match) -> frizbee_match_t {
    frizbee_match_t {
        score: m.score,
        index: m.index,
        exact: m.exact,
    }
}

fn matches_to_c(matches: Vec<frizbee::Match>) -> frizbee_matches_t {
    let items = matches.iter().map(match_to_c).collect::<Box<[_]>>();
    let (items, len) = box_into_raw_parts(items);
    frizbee_matches_t { items, len }
}

/// Creates a matcher that matches `needle` literally (no query syntax); use
/// `frizbee_matcher_from_query` for query syntax and multi-pattern queries.
/// Never returns NULL. `config` must not be NULL. Destroy the matcher with
/// `frizbee_matcher_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_new(
    needle: frizbee_str_t,
    config: *const frizbee_config_t,
) -> *mut frizbee_matcher_t {
    let needle = unsafe { str_from_c(needle) };
    let config = config_to_core(unsafe { &*config });
    let inner = Matcher::new(needle, &config);
    Box::into_raw(Box::new(frizbee_matcher_t { inner }))
}

/// Creates a matcher from a query of whitespace separated atoms, where special
/// syntax changes the matching mode: `foo` (fuzzy), `^foo` (prefix), `foo$`
/// (suffix), `'foo` (substring), `^foo$` (exact) and `!foo` (negated, substring
/// unless combined with the syntax above). A haystack matches when all of the
/// atoms match, summing each atom's score.
///
/// Any special character can be escaped with a backslash, e.g. `\!foo` or
/// `foo\$` match the literal leading/trailing character, and `foo\ bar` matches
/// the literal space. Atoms with an empty needle, e.g. `!` or `^$`, are dropped.
/// Otherwise identical to `frizbee_matcher_new`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_from_query(
    query: frizbee_str_t,
    config: *const frizbee_config_t,
) -> *mut frizbee_matcher_t {
    let query = unsafe { str_from_c(query) };
    let config = config_to_core(unsafe { &*config });
    let inner = Matcher::from_query(query, &config);
    Box::into_raw(Box::new(frizbee_matcher_t { inner }))
}

/// Destroys a matcher. NULL is a no-op
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_free(matcher: *mut frizbee_matcher_t) {
    if !matcher.is_null() {
        drop(unsafe { Box::from_raw(matcher) });
    }
}

/// Matches `haystacks_len` haystacks against the matcher's pattern, returning
/// the matches ordered by the config's sort strategy. This API provides the most
/// performant path when matching on lists. The result is owned by frizbee:
/// release it with `frizbee_matches_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_match_list(
    matcher: *mut frizbee_matcher_t,
    haystacks: *const frizbee_str_t,
    haystacks_len: usize,
) -> frizbee_matches_t {
    let matcher = unsafe { &mut *matcher };
    let haystacks = unsafe { haystacks_from_c(haystacks, haystacks_len) };
    matches_to_c(matcher.inner.match_list(haystacks))
}

/// Like `frizbee_match_list`, matching in parallel on `threads` real threads
/// (`0` = available CPU cores - 2). Threads work on 2048 item chunks, and the
/// final result is identical to the sequential version
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_match_list_parallel(
    matcher: *mut frizbee_matcher_t,
    haystacks: *const frizbee_str_t,
    haystacks_len: usize,
    threads: usize,
) -> frizbee_matches_t {
    let matcher = unsafe { &mut *matcher };
    let haystacks = unsafe { haystacks_from_c(haystacks, haystacks_len) };
    matches_to_c(matcher.inner.match_list_parallel(haystacks, threads))
}

/// Releases a match list returned by `frizbee_match_list` or
/// `frizbee_match_list_parallel` and zeroes it, so a second free is a no-op.
/// NULL is a no-op
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matches_free(matches: *mut frizbee_matches_t) {
    if matches.is_null() {
        return;
    }
    let matches = unsafe { &mut *matches };
    if !matches.items.is_null() {
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(matches.items, matches.len)) });
    }
    matches.items = ptr::null_mut();
    matches.len = 0;
}

/// Matches a single haystack, returning whether it matched and writing the
/// match to `out` only when it did (`*out` is untouched otherwise).
///
/// This API performs ~10% slower than the `frizbee_match_list` API. Consider using
/// `frizbee_match_list` if you have more than one haystack to match, as it performs
/// significantly better.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_match_one(
    matcher: *mut frizbee_matcher_t,
    haystack: frizbee_str_t,
    index: u32,
    out: *mut frizbee_match_t,
) -> bool {
    let matcher = unsafe { &mut *matcher };
    let haystack = unsafe { str_from_c(haystack) };
    match matcher.inner.match_one(haystack, index) {
        Some(m) => {
            unsafe { out.write(match_to_c(&m)) };
            true
        }
        None => false,
    }
}

/// Like `frizbee_match_list`, but each match includes the indices of the chars
/// in the haystack that matched the needle (see `frizbee_match_indices_t` for
/// the layout). This API has not been optimized for performance, and should only
/// be used on small lists, e.g. the visible portion of results. Useful for
/// displaying matched indices in the UI. The result is owned by frizbee:
/// release it with `frizbee_match_indices_list_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_match_list_indices(
    matcher: *mut frizbee_matcher_t,
    haystacks: *const frizbee_str_t,
    haystacks_len: usize,
) -> frizbee_match_indices_list_t {
    let matcher = unsafe { &mut *matcher };
    let haystacks = unsafe { haystacks_from_c(haystacks, haystacks_len) };
    let matches = matcher.inner.match_list_indices(haystacks);

    // Flatten the per-match indices into one shared buffer
    let total = matches.iter().map(|m| m.indices.len()).sum();
    let mut items = Vec::with_capacity(matches.len());
    let mut indices = Vec::<u32>::with_capacity(total);
    for m in &matches {
        items.push(frizbee_match_indices_t {
            score: m.score,
            index: m.index,
            exact: m.exact,
            indices_start: indices.len() as u32,
            indices_len: m.indices.len() as u32,
        });
        indices.extend_from_slice(&m.indices);
    }

    let (items, len) = box_into_raw_parts(items.into_boxed_slice());
    let (indices, indices_len) = box_into_raw_parts(indices.into_boxed_slice());
    frizbee_match_indices_list_t {
        items,
        len,
        indices,
        indices_len,
    }
}

/// Releases a list returned by `frizbee_match_list_indices` and zeroes it, so a
/// second free is a no-op. NULL is a no-op
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_match_indices_list_free(
    matches: *mut frizbee_match_indices_list_t,
) {
    if matches.is_null() {
        return;
    }
    let matches = unsafe { &mut *matches };
    if !matches.items.is_null() {
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(matches.items, matches.len)) });
    }
    if !matches.indices.is_null() {
        drop(unsafe {
            Box::from_raw(ptr::slice_from_raw_parts_mut(
                matches.indices,
                matches.indices_len,
            ))
        });
    }
    matches.items = ptr::null_mut();
    matches.len = 0;
    matches.indices = ptr::null_mut();
    matches.indices_len = 0;
}
