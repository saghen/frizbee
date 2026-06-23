use criterion::BenchmarkId;
use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use nucleo::{
    Config as NucleoConfig, Matcher as NucleoMatcher, Nucleo,
    pattern::{Atom, AtomKind, CaseMatching, Normalization},
};

mod generate;
use generate::{HaystackGenerationOptions, generate_haystack};

const SEED: u64 = 12345;
const NUCLEO_RESET_PATTERN: &str = "\0";

pub fn match_list_generated_bench(
    c: &mut criterion::Criterion,
    name: &str,
    needle: &str,
    match_percentage: f64,
    partial_match_percentage: f64,
) {
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
    match_list_bench_impl(c, name, needle, haystack, BenchmarkInput::MedianLength);
}

pub fn match_list_real_bench(
    c: &mut criterion::Criterion,
    name: &str,
    needle: &str,
    haystack: &[&str],
    include_fzf: bool,
) {
    match_list_bench_impl(
        c,
        name,
        needle,
        haystack,
        BenchmarkInput::SequentialAndParallel { include_fzf },
    );
}

#[derive(Clone, Copy)]
enum BenchmarkInput {
    MedianLength,
    SequentialAndParallel { include_fzf: bool },
}

fn match_list_bench_impl(
    c: &mut criterion::Criterion,
    name: &str,
    needle: &str,
    haystack: &[&str],
    input: BenchmarkInput,
) {
    let mut group = c.benchmark_group(name);

    let size = haystack.iter().map(|x| x.len()).sum::<usize>();

    let median_length = size / haystack.len();
    let benchmark_id = |name: &str| -> BenchmarkId {
        match input {
            BenchmarkInput::MedianLength => BenchmarkId::new(name, median_length),
            BenchmarkInput::SequentialAndParallel { .. } => BenchmarkId::new(name, "Sequential"),
        }
    };

    group.throughput(criterion::Throughput::Bytes(size as u64));

    // Sequential
    group.bench_with_input(benchmark_id("Nucleo"), haystack, |b, haystack| {
        let mut matcher = NucleoMatcher::new(NucleoConfig::DEFAULT);
        let atom = Atom::new(
            needle,
            CaseMatching::Ignore,
            Normalization::Never,
            AtomKind::Fuzzy,
            false,
        );
        b.iter(|| atom.match_list(black_box(haystack.iter()), &mut matcher))
    });
    if let BenchmarkInput::SequentialAndParallel { include_fzf: true } = input {
        group.bench_function(benchmark_id("FZF"), |b| {
            b.iter(|| {
                // measured with fzf --filter linux --tiebreak index --bench 10s --threads 1 < benches/match_list/data.txt
                std::thread::sleep(Duration::from_micros(120610));
            })
        });
    }
    group.bench_with_input(benchmark_id("Frizbee"), haystack, |b, haystack| {
        b.iter(|| match_list(needle, haystack, Some(0)))
    });
    // group.bench_with_input(benchmark_id("All Scores"), haystack, |b, haystack| {
    //     b.iter(|| match_list(needle, haystack, None))
    // });
    // group.bench_with_input(benchmark_id("1 Typo"), haystack, |b, haystack| {
    //     b.iter(|| match_list(needle, haystack, Some(1)))
    // });
    // group.bench_with_input(benchmark_id("2 Typos"), haystack, |b, haystack| {
    //     b.iter(|| match_list(needle, haystack, Some(2)))
    // });
    // group.bench_with_input(benchmark_id("3 Typos"), haystack, |b, haystack| {
    //     b.iter(|| match_list(needle, haystack, Some(3)))
    // });

    if let BenchmarkInput::SequentialAndParallel { include_fzf } = input {
        group.bench_with_input(
            BenchmarkId::new("Nucleo", "Parallel (x8)"),
            haystack,
            |b, haystack| {
                b.iter_custom(|iters| {
                    let mut nucleo = nucleo_parallel_worker(haystack);
                    let mut elapsed = Duration::ZERO;

                    for _ in 0..iters {
                        let start = Instant::now();
                        nucleo_reparse(&mut nucleo, black_box(needle));
                        black_box(tick_nucleo_until_done(&mut nucleo));
                        elapsed += start.elapsed();

                        nucleo_reparse(&mut nucleo, NUCLEO_RESET_PATTERN);
                        black_box(tick_nucleo_until_done(&mut nucleo));
                    }

                    elapsed
                })
            },
        );
        if include_fzf {
            group.bench_function(BenchmarkId::new("FZF", "Parallel (x8)"), |b| {
                b.iter(|| {
                    // measured with fzf --filter linux --tiebreak index --bench 10s --threads 8 < benches/match_list/data.txt
                    std::thread::sleep(Duration::from_micros(16170));
                })
            });
        }
        group.bench_with_input(
            BenchmarkId::new("Frizbee", "Parallel (x8)"),
            haystack,
            |b, haystack| b.iter(|| match_list_parallel(needle, haystack, Some(0))),
        );
        // group.bench_with_input(
        //     BenchmarkId::new("All Scores", "Parallel (x8)"),
        //     haystack,
        //     |b, haystack| b.iter(|| match_list_parallel(needle, haystack, None)),
        // );
        // group.bench_with_input(
        //     BenchmarkId::new("1 Typo", "Parallel (x8)"),
        //     haystack,
        //     |b, haystack| b.iter(|| match_list_parallel(needle, haystack, Some(1))),
        // );
        // group.bench_with_input(
        //     BenchmarkId::new("2 Typos", "Parallel (x8)"),
        //     haystack,
        //     |b, haystack| b.iter(|| match_list_parallel(needle, haystack, Some(2))),
        // );
        // group.bench_with_input(
        //     BenchmarkId::new("3 Typos", "Parallel (x8)"),
        //     haystack,
        //     |b, haystack| b.iter(|| match_list_parallel(needle, haystack, Some(3))),
        // );
    }
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

fn nucleo_parallel_worker(haystack: &[&str]) -> Nucleo<String> {
    let mut config = NucleoConfig::DEFAULT;
    config.normalize = false;
    config.ignore_case = true;

    let mut nucleo = Nucleo::new(config, Arc::new(|| {}), Some(8), 1);
    {
        let injector = nucleo.injector();
        for item in haystack {
            injector.push((*item).to_owned(), |item, columns| {
                columns[0] = item.as_str().into();
            });
        }
    }
    nucleo_reparse(&mut nucleo, NUCLEO_RESET_PATTERN);
    tick_nucleo_until_done(&mut nucleo);
    nucleo
}

fn nucleo_reparse(nucleo: &mut Nucleo<String>, needle: &str) {
    nucleo
        .pattern
        .reparse(0, needle, CaseMatching::Ignore, Normalization::Never, false);
}

fn tick_nucleo_until_done(nucleo: &mut Nucleo<String>) -> u32 {
    loop {
        let status = nucleo.tick(1000);
        if !status.running {
            return nucleo.snapshot().matched_item_count();
        }
    }
}
