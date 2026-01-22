use std::marker::PhantomData;

use crate::smith_waterman::simd::{smith_waterman, typos_from_score_matrix};
use crate::{Config, Match, Scoring};

use super::Appendable;

#[derive(Debug)]
pub(crate) struct FixedWidthBucket<'a, const W: usize, M: Appendable<Match>> {
    // has_avx512: bool,
    has_avx2: bool,

    length: usize,
    needle: &'a str,
    haystacks: [&'a str; 32],
    idxs: [u32; 32],

    max_typos: Option<u16>,
    scoring: Scoring,

    _phantom: PhantomData<M>,
}

impl<'a, const W: usize, M: Appendable<Match>> FixedWidthBucket<'a, W, M> {
    pub fn new(needle: &'a str, config: &Config) -> Self {
        FixedWidthBucket {
            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // has_avx512: is_x86_feature_detected!("avx512f")
            //     && is_x86_feature_detected!("avx512bitalg"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            has_avx2: is_x86_feature_detected!("avx2"),

            // #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            // has_avx512: false,
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            has_avx2: false,

            length: 0,
            needle,
            haystacks: [""; 32],
            idxs: [0; 32],

            max_typos: config.max_typos,
            scoring: config.scoring.clone(),

            _phantom: PhantomData,
        }
    }

    pub fn add_haystack(&mut self, matches: &mut M, haystack: &'a str, idx: u32) {
        self.haystacks[self.length] = haystack;
        self.idxs[self.length] = idx;
        self.length += 1;

        match self.length {
            // 32 if self.has_avx512 => self._finalize::<32>(matches),
            16 if self.has_avx2 => self._finalize::<16>(matches),
            8 if !self.has_avx2 => self._finalize::<8>(matches),
            _ => {}
        }
    }

    pub fn finalize(&mut self, matches: &mut M) {
        match self.length {
            // 17.. => self._finalize::<32>(matches),
            9.. => self._finalize::<16>(matches),
            0.. => self._finalize::<8>(matches),
        }
    }

    fn _finalize<const L: usize>(&mut self, matches: &mut M) {
        if self.length == 0 {
            return;
        }

        let (scores, score_matrix, exact_matches) = smith_waterman::<W, L>(
            self.needle,
            &self.haystacks.get(0..L).unwrap().try_into().unwrap(),
            self.max_typos,
            &self.scoring,
        );

        let typos = self
            .max_typos
            .map(|max_typos| typos_from_score_matrix::<W, L>(&score_matrix, max_typos));

        for idx in 0..self.length {
            if let Some(max_typos) = self.max_typos
                && typos.is_some_and(|typos| typos[idx] > max_typos)
            {
                continue;
            }

            let score_idx = self.idxs[idx];
            matches.append(Match {
                index: score_idx,
                score: scores[idx],
                exact: exact_matches[idx],
            });
        }

        self.length = 0;
    }
}
