//! The [Smith Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) performs local sequence alignment ([explanation](https://kaell.se/bibook/pairwise/waterman.html)), originally designed to find similar sequences between two DNA strings. Guaranteed to find the optimal alignment and supports typos.
//!
//! The algorithm's time and space complexity of O(nm) led to plenty of research on parallelization. Each cell in the matrix has a data dependency on the cell to the left, up, and left-up diagonal. For biology, DNA sequences are typically quite large (m > 1000), so most of the parallelization approaches focused on large matrices ([see this paper for common parallelization techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822)).
//!
//! As a fuzzy matcher, the matrices in Frizbee are typically much smaller than those in DNA alignment (m < 128). Frizbee uses an approach similar to [sequential layout](https://pmc.ncbi.nlm.nih.gov/articles/PMC8419822/#Sec11), except the horizontal (vertical in the paper, but flipped in frizbee) data dependency [is applied immediately](src/smith_waterman/simd/gaps.rs). This approach supports [affine gaps](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Affine).
//!
//! ```text
//! needle: "foo"
//! haystack: "some/long/foo/path"
//!
//! // assuming 4 lane SIMD for simplicity
//! // in reality, we use 16 SIMD lanes (16 bytes each, 256 bit)
//!
//! score_matrix:
//!    [s   o   m   e]   [/   l   o   n]   [g   /   f   o]   [o   /   p   a]   [t   h   _   _]
//! f  [0   0   0   0]   [0   0   0   0]   [0   0   16  11]  [10  9   8   7]   [6   5   4   3]
//! o  [0   16  11  10]  [9   8   16  11]  [10  9   8   32]  [27  26  25  24]  [23  22  21  20]
//! o  [0   16  11  10]  [9   8   24  19]  [18  17  16  24]  [48  43  42  41]  [40  39  38  37]
//!
//! // for the SIMD register at row 2, col 1, we would start with
//!
//! needle:      [o   o   o   o]
//! haystack:    [/   l   o   n]
//! match mask:  [f   f   t   f]
//!
//! diagonal:    [10  9   8   16]
//! up:          [9   8   16  11]
//! current:     [8   7   24  9]
//!
//! // now we propagate the left data dependency
//!
//! left:        [0   16  11  10]
//! // shift current right by 1 element, filling in with right most element from left
//! shifted:     [10  8   7   24]
//! // decay by gap extend penalty (1)
//! // last element decayed by 5 (gap open penalty) instead of 1 (gap extend penalty)
//! // because the previous element matched (affine gaps)
//! decayed:     [9   7   6   19]
//! // max with current
//! current:     [9   7   24  19]
//! // repeat for shifting by 2 elements
//! shifted:     [11  10  9   7]
//! decayed:     [9   8   7   5] // gap extend penalty * 2 or gap open penalty + extend penalty
//! current:     [9   8   24  19]
//!
//! final:       [8   7   24  19]
//! ```
//!
//! Frizbee previously used inter-sequence parallelism (one needle, $LANES haystacks) but this performed about the same as sequential layout due to requiring interleaving the haystacks and bucketing based on haystack length, while performing worse in parallel due to the required bucketing.

mod greedy;
pub(crate) mod simd;

pub use greedy::match_greedy;
pub use simd::{Alignment, AlignmentPathIter, SmithWatermanMatcher};
