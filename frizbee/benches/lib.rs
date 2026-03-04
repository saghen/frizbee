use criterion::{Criterion, criterion_group, criterion_main};
use std::time::Duration;

mod match_list;

use match_list::{match_list_bench, match_list_generated_bench};

fn criterion_benchmark(c: &mut Criterion) {
    // Bench on real data
    let haystack_bytes = std::fs::read("benches/match_list/data.txt")
        .expect("Failed to read benchmark data. Run `wget -O benches/match_list/data.txt https://gist.github.com/ii14/637689ef8d071824e881a78044670310/raw/dc1dbc859daa38b62f4b9a69dec1fc599e4735e7/data.txt`");
    let haystack_str =
        String::from_utf8(haystack_bytes).expect("Failed to parse chromium benchmark data");
    let haystack = haystack_str.split('\n').collect::<Vec<_>>();

    match_list_bench(c, "Chromium", "linux", &haystack);

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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2));
    targets = criterion_benchmark
}
criterion_main!(benches);
