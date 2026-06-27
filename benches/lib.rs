use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;

mod match_list;

use match_list::{match_list_generated_bench, match_list_real_bench};

struct DatasetBenchmark {
    name: &'static str,
    path: &'static str,
    needle: &'static str,
    // fzf --filter needle --tiebreak index --bench 10s --threads 1 < path
    fff_sequential: Duration,
    // fzf --filter needle --tiebreak index --bench 10s --threads 8 < path
    fff_parallel: Duration,
}

const DATASET_BENCHMARKS: &[DatasetBenchmark] = &[
    DatasetBenchmark {
        name: "Chromium",
        path: "benches/data/chromium.txt",
        needle: "linux",
        fff_sequential: Duration::from_micros(120610),
        fff_parallel: Duration::from_micros(16170),
    },
    DatasetBenchmark {
        name: "Arabic",
        path: "benches/data/arabic_unicode.txt",
        needle: "إن",
        fff_sequential: Duration::from_micros(165730),
        fff_parallel: Duration::from_micros(21960),
    },
    DatasetBenchmark {
        name: "Korean",
        path: "benches/data/korean_unicode.txt",
        needle: "니다",
        fff_sequential: Duration::from_micros(114320),
        fff_parallel: Duration::from_micros(15390),
    },
];

fn criterion_benchmark(c: &mut Criterion) {
    // Bench on real data
    for dataset in DATASET_BENCHMARKS {
        let haystack_owned = read_lines(dataset.path);
        let haystack = haystack_owned
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();

        match_list_real_bench(
            c,
            dataset.name,
            dataset.needle,
            &haystack,
            dataset.fff_sequential,
            dataset.fff_parallel,
        );
    }

    // Bench on synthetic data
    for (name, (match_percentage, partial_match_percentage)) in [
        ("Partial Match", (0.05, 0.2)),
        ("All Match", (1.0, 0.0)),
        ("No Match with Partial", (0.0, 0.15)),
        ("No Match", (0.0, 0.0)),
    ] {
        match_list_generated_bench(
            c,
            name,
            "deadbeef",
            match_percentage,
            partial_match_percentage,
        );
    }
    match_list_generated_bench(c, "Copy", "", 0., 0.);
}

fn read_lines(path: &str) -> Vec<String> {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("Failed to read benchmark data from {path}: {err}"));
    let lines = data
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();

    assert!(!lines.is_empty(), "No benchmark data loaded from {path}");
    lines
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2));
    targets = criterion_benchmark
}
criterion_main!(benches);
