#![feature(portable_simd)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};

mod interleave;
mod match_list;
mod prefilter;

use frizbee::{
    Scoring,
    smith_waterman::{simd, v2},
};
use interleave::{interleave_bench, interleave_misaligned_bench};
use match_list::{match_list_bench, match_list_generated_bench};
use prefilter::prefilter_bench;

fn criterion_benchmark(c: &mut Criterion) {
    let needle = "test";
    let haystack: [&str; 16] = std::array::repeat("~~~~~~t~est~~~~");

    let mut max_score = 0;
    let mut typo_count = 0;
    c.bench_function("intra-sequence", |b| {
        let scoring = Scoring::default();
        let mut score_matrix = v2::generate_score_matrix(needle.len(), haystack[0].len());
        let needle = v2::Needle::new(needle);
        b.iter(|| unsafe {
            for haystack in haystack.iter() {
                max_score =
                    v2::smith_waterman(&needle, haystack.as_bytes(), &scoring, &mut score_matrix);
                typo_count =
                    v2::typos_from_score_matrix(&score_matrix, max_score, 0, haystack.len());
                black_box(typo_count);
            }
        });
    });
    println!("typo_count: {}", typo_count);
    println!("max_score: {}", max_score);

    let mut max_score_val = 0;
    c.bench_function("inter-sequence", |b| {
        let scoring = Scoring::default();
        b.iter(|| {
            let (max_scores, score_matrix, _) = simd::smith_waterman::<16, 16>(
                black_box(needle),
                black_box(&haystack),
                None,
                &scoring,
            );
            max_score_val = max_scores[0];
            let typo_count = simd::typos_from_score_matrix(&score_matrix, 0);
            assert_eq!(typo_count[0], 0);
        })
    });
    println!("max_score_val: {}", max_score_val);

    prefilter_bench(c);

    interleave_bench(c);
    interleave_misaligned_bench(c);

    // Bench on real data
    let haystack_bytes = std::fs::read("benches/match_list/data.txt")
        .expect("Failed to read benchmark data. Run `wget -O benches/match_list/data.txt https://gist.github.com/ii14/637689ef8d071824e881a78044670310/raw/dc1dbc859daa38b62f4b9a69dec1fc599e4735e7/data.txt`");
    let haystack_str =
        String::from_utf8(haystack_bytes).expect("Failed to parse chromium benchmark data");
    let haystack = haystack_str.split('\n').collect::<Vec<_>>();

    match_list_bench(c, "Chromium", "linux", &haystack);

    // Bench on synthetic data
    for (name, (match_percentage, partial_match_percentage)) in [
        ("Partial Match", (0.05, 0.20)),
        ("All Match", (1.0, 0.0)),
        ("No Match", (0.0, 0.0)),
    ] {
        match_list_generated_bench(c, name, match_percentage, partial_match_percentage);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2));
    targets = criterion_benchmark
}
criterion_main!(benches);
