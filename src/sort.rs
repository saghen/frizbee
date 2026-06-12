use crate::Match;

/// Sorts a slice of [`Match`] values in-place by descending `score` using a
/// stable radix sort. This assumes that the matches are already sorted by index.
#[inline]
pub fn radix_sort_matches(matches: &mut [Match]) {
    // pass 1
    let mut histogram = [0u32; 256];
    for m in matches.iter() {
        let radix = m.score & 0xFF;
        histogram[radix as usize] += 1;
    }
    let mut offsets = [0u32; 256];
    for idx in (1..256).rev() {
        offsets[idx - 1] = offsets[idx] + histogram[idx];
    }

    let mut matches_b = vec![
        Match {
            score: 0,
            index: 0,
            exact: false,
            #[cfg(feature = "match_end_col")]
            end_col: 0,
        };
        matches.len()
    ];
    for m in matches.iter() {
        let radix = m.score & 0xFF;
        matches_b[offsets[radix as usize] as usize] = *m;
        offsets[radix as usize] += 1;
    }

    // pass 2
    let mut histogram = [0u32; 256];
    for m in matches_b.iter() {
        let radix = (m.score >> 8) & 0xFF;
        histogram[radix as usize] += 1;
    }
    offsets[255] = 0;
    for idx in (1..256).rev() {
        offsets[idx - 1] = offsets[idx] + histogram[idx];
    }
    for m in matches_b {
        let radix = (m.score >> 8) & 0xFF;
        matches[offsets[radix as usize] as usize] = m;
        offsets[radix as usize] += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_sorted() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut matches = (0u32..1 << 24)
            .map(|index| Match {
                score: rng.random::<u16>(),
                index,
                exact: rng.random_bool(0.5),
                #[cfg(feature = "match_end_col")]
                end_col: 0,
            })
            .collect::<Vec<_>>();

        radix_sort_matches(&mut matches);

        assert!(matches.is_sorted());
    }
}
