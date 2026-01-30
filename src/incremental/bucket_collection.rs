use super::bucket::{IncrementalBucket, IncrementalBucketTrait};

pub(crate) struct IncrementalBucketCollection<'a, const W: usize, const L: usize> {
    length: usize,
    haystacks: [&'a str; L],
    idxs: [u32; L],
}

impl<'a, const W: usize, const L: usize> IncrementalBucketCollection<'a, W, L> {
    pub fn new() -> Self {
        Self {
            length: 0,
            haystacks: [""; L],
            idxs: [0; L],
        }
    }

    fn build_bucket(&self) -> Box<IncrementalBucket<W, L>> {
        Box::new(IncrementalBucket::<W, L>::new(
            &self.haystacks,
            self.idxs,
            self.length,
        ))
    }

    pub fn add_haystack(
        &mut self,
        haystack: &'a str,
        idx: u32,
        buckets: &mut Vec<Box<dyn IncrementalBucketTrait>>,
    ) {
        self.haystacks[self.length] = haystack;
        self.idxs[self.length] = idx;
        self.length += 1;

        if self.length == L {
            buckets.push(self.build_bucket());
            self.idxs = [0; L];
            self.length = 0;
        }
    }

    pub fn finalize(&mut self, buckets: &mut Vec<Box<dyn IncrementalBucketTrait>>) {
        if self.length > 0 {
            buckets.push(self.build_bucket());
        }
    }
}
