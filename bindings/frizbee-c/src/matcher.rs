//! The matcher handle, the match result types and every matching `extern "C"`
//! fn, including the `*_free` functions releasing the result buffers.

use std::ptr;

use frizbee::Matcher;

use crate::config::{config_or_default, frizbee_config_t, frizbee_str_t};

/// Primary entrypoint for fuzzy matching
///
/// Opaque matcher handle, created by `frizbee_matcher_new` or
/// `frizbee_matcher_from_query` and destroyed by `frizbee_matcher_free`.
///
/// This compiles the pattern once, allocates memory for the Smith Waterman
/// matrix, and reuses the selected SIMD backend.
///
/// Ideally, only construct these at most once per list. They're cheap to
/// construct, but end up being expensive if you construct them for each item in
/// your list.
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

impl From<Vec<frizbee::Match>> for frizbee_matches_t {
    fn from(matches: Vec<frizbee::Match>) -> Self {
        let items = matches
            .into_iter()
            .map(frizbee_match_t::from)
            .collect::<Box<[_]>>();
        let (items, len) = box_into_raw_parts(items);
        frizbee_matches_t { items, len }
    }
}

/// SAFETY: the caller promises `s.ptr` points to `s.len` bytes of valid UTF-8
unsafe fn str_from_c<'a>(s: frizbee_str_t) -> &'a str {
    if s.len == 0 {
        return "";
    }
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(s.ptr.cast(), s.len)) }
}

/// `frizbee_str_t` usable as a core-matcher haystack. `repr(transparent)` so
/// the caller's `frizbee_str_t` array can be reinterpreted as
/// `&[haystack_str_t]` and passed to the core matcher without a per-call copy
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

/// SAFETY: the caller promises `haystacks` points to `len` valid
/// `frizbee_str_t`
unsafe fn haystacks_from_c<'a>(
    haystacks: *const frizbee_str_t,
    len: usize,
) -> &'a [haystack_str_t] {
    if len == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(haystacks.cast(), len) }
}

/// SAFETY: `matcher` must point to a live `frizbee_matcher_t`
unsafe fn matcher_from_c<'a>(matcher: *mut frizbee_matcher_t) -> &'a mut Matcher {
    assert!(!matcher.is_null(), "matcher must not be NULL");
    unsafe { &mut (*matcher).inner }
}

/// The buffers cross the ABI as boxed slices so the free functions can rebuild
/// them from pointer + length alone, without carrying capacities in the
/// structs. Returns (null, 0) for empty slices to avoid exposing dangling
/// pointers to C.
fn box_into_raw_parts<T>(slice: Box<[T]>) -> (*mut T, usize) {
    let len = slice.len();
    if len == 0 {
        (ptr::null_mut(), 0)
    } else {
        (Box::into_raw(slice) as *mut T, len)
    }
}

impl From<frizbee::Match> for frizbee_match_t {
    fn from(m: frizbee::Match) -> Self {
        frizbee_match_t {
            score: m.score,
            index: m.index,
            exact: m.exact,
        }
    }
}

impl From<&frizbee::MatchIndices> for frizbee_match_t {
    fn from(m: &frizbee::MatchIndices) -> Self {
        frizbee_match_t {
            score: m.score,
            index: m.index,
            exact: m.exact,
        }
    }
}

/// Creates a matcher that matches `needle` literally (no query syntax); use
/// `frizbee_matcher_from_query` for query syntax and multi-pattern queries.
/// Never returns NULL. Pass NULL for `config` to use default configuration.
/// Destroy the matcher with `frizbee_matcher_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_new(
    needle: frizbee_str_t,
    config: *const frizbee_config_t,
) -> *mut frizbee_matcher_t {
    let needle = unsafe { str_from_c(needle) };
    let config = unsafe { config_or_default(config) };
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
/// the literal space. Atoms with an empty needle, e.g. `!` or `^$`, are
/// dropped. Pass NULL for `config` to use default configuration. Otherwise
/// identical to `frizbee_matcher_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_from_query(
    query: frizbee_str_t,
    config: *const frizbee_config_t,
) -> *mut frizbee_matcher_t {
    let query = unsafe { str_from_c(query) };
    let config = unsafe { config_or_default(config) };
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
/// the matches ordered by the config's sort strategy. This API provides the
/// most performant path when matching on lists. The result is owned by frizbee:
/// release it with `frizbee_matches_free`. `matcher` must not be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_match_list(
    matcher: *mut frizbee_matcher_t,
    haystacks: *const frizbee_str_t,
    haystacks_len: usize,
) -> frizbee_matches_t {
    let matcher = unsafe { matcher_from_c(matcher) };
    let haystacks = unsafe { haystacks_from_c(haystacks, haystacks_len) };
    matcher.match_list(haystacks).into()
}

/// Like `frizbee_matcher_match_list`, matching in parallel on `threads` real
/// threads (`0` = available CPU cores - 2). Threads work on 2048 item chunks,
/// and the final result is identical to the sequential version. `matcher` must
/// not be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_match_list_parallel(
    matcher: *mut frizbee_matcher_t,
    haystacks: *const frizbee_str_t,
    haystacks_len: usize,
    threads: usize,
) -> frizbee_matches_t {
    let matcher = unsafe { matcher_from_c(matcher) };
    let haystacks = unsafe { haystacks_from_c(haystacks, haystacks_len) };
    matcher.match_list_parallel(haystacks, threads).into()
}

/// Releases a match list returned by `frizbee_matcher_match_list` or
/// `frizbee_matcher_match_list_parallel` and zeroes it, so a second free is a
/// no-op. NULL is a no-op
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matches_free(matches: *mut frizbee_matches_t) {
    if matches.is_null() {
        return;
    }
    let matches = unsafe { &mut *matches };
    if !matches.items.is_null() && matches.len > 0 {
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(matches.items, matches.len)) });
    }
    matches.items = ptr::null_mut();
    matches.len = 0;
}

/// Matches a single haystack, returning whether it matched and writing the
/// match to `out` only when it did (`*out` is untouched otherwise). `out` may
/// be NULL if only the boolean result is needed. `matcher` must not be NULL.
///
/// This API performs ~10% slower than the `frizbee_matcher_match_list` API.
/// Consider using `frizbee_matcher_match_list` if you have more than one
/// haystack to match, as it performs significantly better.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_match_one(
    matcher: *mut frizbee_matcher_t,
    haystack: frizbee_str_t,
    index: u32,
    out: *mut frizbee_match_t,
) -> bool {
    let matcher = unsafe { matcher_from_c(matcher) };
    let haystack = unsafe { str_from_c(haystack) };
    match matcher.match_one(haystack, index) {
        Some(m) => {
            if !out.is_null() {
                unsafe { out.write(frizbee_match_t::from(m)) };
            }
            true
        }
        None => false,
    }
}

/// Matches a single haystack with indices, returning whether it matched.
///
/// If `out_match` is non-NULL and a match occurred, writes the match metadata.
/// If `out_indices` is non-NULL and `indices_capacity > 0`, copies up to
/// `indices_capacity` matched byte offsets into `out_indices`.
/// If `out_indices_len` is non-NULL, writes the total number of matched
/// indices. `matcher` must not be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn frizbee_matcher_match_one_indices(
    matcher: *mut frizbee_matcher_t,
    haystack: frizbee_str_t,
    index: u32,
    out_match: *mut frizbee_match_t,
    out_indices: *mut u32,
    indices_capacity: usize,
    out_indices_len: *mut usize,
) -> bool {
    let matcher = unsafe { matcher_from_c(matcher) };
    let haystack = unsafe { str_from_c(haystack) };
    match matcher.match_one_indices(haystack, index) {
        Some(m) => {
            if !out_match.is_null() {
                unsafe { out_match.write(frizbee_match_t::from(&m)) };
            }
            if !out_indices_len.is_null() {
                unsafe { out_indices_len.write(m.indices.len()) };
            }
            if !out_indices.is_null() && indices_capacity > 0 {
                let to_copy = m.indices.len().min(indices_capacity);
                unsafe {
                    ptr::copy_nonoverlapping(m.indices.as_ptr(), out_indices, to_copy);
                }
            }
            true
        }
        None => false,
    }
}
