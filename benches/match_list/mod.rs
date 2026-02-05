use criterion::BenchmarkId;
use frizbee::IncrementalMatcher;
use std::hint::black_box;

use nucleo::{
    Config as NucleoConfig, Matcher as NucleoMatcher,
    pattern::{Atom, AtomKind, CaseMatching, Normalization},
};

mod generate;
use generate::{HaystackGenerationOptions, generate_haystack};

const SEED: u64 = 12345;

pub fn match_list_generated_bench(
    c: &mut criterion::Criterion,
    name: &str,
    match_percentage: f64,
    partial_match_percentage: f64,
) {
    let needle = "deadbeef";

    for median_length in [16, 32, 64, 128] {
        // Generate haystacks
        let options = HaystackGenerationOptions {
            seed: SEED,
            partial_match_percentage,
            match_percentage,
            median_length,
            std_dev_length: median_length / 4,
            num_samples: 100_000,
        };
        let haystack_owned = generate_haystack(needle, options);
        let haystack = &haystack_owned
            .iter()
            .map(|x| x.as_str())
            .collect::<Vec<_>>();

        match_list_bench(c, name, needle, haystack);
    }
}

pub fn match_list_bench(c: &mut criterion::Criterion, name: &str, needle: &str, haystack: &[&str]) {
    let mut group = c.benchmark_group(name);

    let size = haystack.iter().map(|x| x.len()).sum::<usize>();
    let median_length = size / haystack.len();
    group.throughput(criterion::Throughput::Bytes(size as u64));

    // Sequential
    group.bench_with_input(
        BenchmarkId::new("Nucleo", median_length),
        haystack,
        |b, haystack| {
            let mut matcher = NucleoMatcher::new(NucleoConfig::DEFAULT);
            let atom = Atom::new(
                needle,
                CaseMatching::Ignore,
                Normalization::Never,
                AtomKind::Fuzzy,
                false,
            );
            b.iter(|| atom.match_list(black_box(haystack.iter()), &mut matcher))
        },
    );
    group.bench_with_input(
        BenchmarkId::new("Frizbee", median_length),
        haystack,
        |b, haystack| b.iter(|| match_list(needle, haystack, Some(0))),
    );
    group.bench_with_input(
        BenchmarkId::new("Frizbee: Incremental", median_length),
        haystack,
        |b, haystack| {
            let mut matcher = IncrementalMatcher::new(Default::default());
            matcher.add_haystacks(haystack);
            b.iter(|| {
                let matches = matcher.match_list(black_box(needle));
                drop(matches);
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("Frizbee: Parallel", median_length),
        haystack,
        |b, haystack| b.iter(|| match_list_parallel(needle, haystack, Some(0))),
    );
    group.bench_with_input(
        BenchmarkId::new("Frizbee: All Scores", median_length),
        haystack,
        |b, haystack| b.iter(|| match_list(needle, haystack, None)),
    );
    group.bench_with_input(
        BenchmarkId::new("Frizbee: 1 Typo", median_length),
        haystack,
        |b, haystack| b.iter(|| match_list(needle, haystack, Some(1))),
    );
}

fn match_list(needle: &str, haystack: &[&str], max_typos: Option<u16>) -> Vec<frizbee::Match> {
    frizbee::match_list(
        black_box(needle),
        black_box(haystack),
        black_box(&frizbee::Config {
            max_typos,
            ..Default::default()
        }),
    )
}

fn match_list_parallel(
    needle: &str,
    haystack: &[&str],
    max_typos: Option<u16>,
) -> Vec<frizbee::Match> {
    frizbee::match_list_parallel(
        black_box(needle),
        black_box(haystack),
        black_box(&frizbee::Config {
            max_typos,
            ..Default::default()
        }),
        8,
    )
}
