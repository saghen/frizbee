use std::{
    hint::black_box,
    path::{Path, PathBuf},
    time::Duration,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const DATASETS: &[(&str, &str)] = &[
    ("Arabic", "lipsum/Arabic-Lipsum.utf8.txt"),
    ("Chinese", "lipsum/Chinese-Lipsum.utf8.txt"),
    ("Emoji", "lipsum/Emoji-Lipsum.utf8.txt"),
    ("Hebrew", "lipsum/Hebrew-Lipsum.utf8.txt"),
    ("Hindi", "lipsum/Hindi-Lipsum.utf8.txt"),
    ("Japanese", "lipsum/Japanese-Lipsum.utf8.txt"),
    ("Korean", "lipsum/Korean-Lipsum.utf8.txt"),
    ("Latin", "lipsum/Latin-Lipsum.utf8.txt"),
    ("Russian", "lipsum/Russian-Lipsum.utf8.txt"),
];

fn criterion_benchmark(c: &mut Criterion) {
    let datasets = load_datasets(&PathBuf::from("benches/data/unicode"));

    let mut group = c.benchmark_group("Unicode UTF-8 to UTF-32");
    for (name, input) in &datasets {
        group.throughput(Throughput::Bytes(input.len() as u64));

        #[cfg(target_arch = "x86_64")]
        if frizbee::unicode::backend::avx512_is_available() {
            group.bench_with_input(BenchmarkId::new("AVX-512", name), input, |b, input| {
                let mut output = Vec::with_capacity(input.len());
                b.iter(|| unsafe {
                    output.clear();
                    frizbee::unicode::backend::convert_avx512_into(
                        black_box(input.as_str()),
                        &mut output,
                    );
                    black_box(output.len())
                });
            });
        }

        #[cfg(target_arch = "x86_64")]
        if frizbee::unicode::backend::avx2_is_available() {
            group.bench_with_input(BenchmarkId::new("AVX2", name), input, |b, input| {
                let mut output = Vec::with_capacity(input.len());
                b.iter(|| unsafe {
                    output.clear();
                    frizbee::unicode::backend::convert_avx2_into(
                        black_box(input.as_str()),
                        &mut output,
                    );
                    black_box(output.len())
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("simdutf", name), input, |b, input| {
            let mut output = Vec::with_capacity(input.len());
            b.iter(|| unsafe {
                let bytes = black_box(input.as_bytes());
                output.clear();
                let written = simdutf::convert_valid_utf8_to_utf32(
                    bytes.as_ptr(),
                    bytes.len(),
                    output.as_mut_ptr(),
                );
                output.set_len(written);
                black_box(output.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", name), input, |b, input| {
            let mut output = Vec::with_capacity(input.len());
            b.iter(|| {
                output.clear();
                frizbee::unicode::backend::convert_scalar_into(
                    black_box(input.as_str()),
                    &mut output,
                );
                black_box(output.len())
            });
        });
    }
}

fn load_datasets(root: &Path) -> Vec<(&'static str, String)> {
    DATASETS
        .iter()
        .map(|&(name, relative_path)| {
            let path = root.join(relative_path);
            let input = std::fs::read_to_string(&path).unwrap_or_else(|err| {
                panic!(
                    "failed to read simdutf unicode_lipsum dataset {}: {err}\n\
                     expected root: {}\n\
                     download with: ./benches/data/download.sh",
                    path.display(),
                    root.display()
                )
            });
            (name, input)
        })
        .collect()
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2));
    targets = criterion_benchmark
}
criterion_main!(benches);
