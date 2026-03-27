use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion};
use frizbee::Match;
use rand::RngExt;

fn bench_with_scale(c: &mut Criterion, name: &str, scale: usize, batch_size: BatchSize) {
    let mut rng = rand::rng();
    let matches = (0..16 * scale)
        .map(|index| Match {
            score: rng.random(),
            index: index as u32,
            exact: rng.random_bool(0.5),
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group(name);

    for idx in 1..=16 {
        let len = scale * idx;
        let input = Vec::from(&matches[..len]);

        group.throughput(criterion::Throughput::Elements(len as u64));

        group.bench_with_input(BenchmarkId::new("std", len), &input, |b, i| {
            b.iter_batched(
                || i.clone(),
                |mut d| {
                    d.sort_by_key(|m| m.score);
                    d
                },
                batch_size,
            )
        });

        group.bench_with_input(BenchmarkId::new("radix", len), &input, |b, i| {
            b.iter_batched(
                || i.clone(),
                |mut m| {
                    frizbee::sort::radix_sort_matches_by_score(&mut m);
                    m
                },
                batch_size,
            )
        });
    }

    group.finish();
}

fn bench(c: &mut Criterion) {
    bench_with_scale(c, "sort tiny", 10, BatchSize::SmallInput);
    bench_with_scale(c, "sort small", 100, BatchSize::SmallInput);
    bench_with_scale(c, "sort large", 10000, BatchSize::LargeInput);
}

criterion::criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500))
        .with_plots();
    targets = bench
}
criterion::criterion_main!(benches);
