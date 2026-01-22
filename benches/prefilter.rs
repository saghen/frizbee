use std::hint::black_box;

use frizbee::prefilter::{Prefilter, scalar};

fn create_haystack(text: &str, text_pos: usize, length: usize) -> String {
    let mut haystack = String::new();
    for _ in 0..text_pos {
        haystack.push('_');
    }
    haystack.push_str(text);
    for _ in 0..(length - text.len() - text_pos) {
        haystack.push('_');
    }
    haystack
}

pub fn prefilter_bench(c: &mut criterion::Criterion) {
    let needle = "test";

    run_prefilter_bench::<64>(c, needle, &create_haystack(needle, 40, 52));
    run_prefilter_bench::<16>(c, needle, &create_haystack(needle, 10, 14));
}

fn run_prefilter_bench<const W: usize>(c: &mut criterion::Criterion, needle: &str, haystack: &str) {
    let needle_cased = &black_box(Prefilter::case_needle(needle));
    let needle = black_box(needle).as_bytes();
    let haystack = black_box(haystack).as_bytes();

    let length = haystack.len();
    let mut group = c.benchmark_group(format!("prefilter/{length}"));

    // Ordered
    group.bench_function("scalar", |b| {
        b.iter(|| scalar::match_haystack(needle, haystack))
    });

    // Ordered Insensitive
    group.bench_function("scalar/insensitive", |b| {
        b.iter(|| scalar::match_haystack_insensitive(needle_cased, haystack))
    });

    // Unordered
    #[cfg(target_arch = "x86_64")]
    group.bench_function("x86_64", |b| {
        b.iter(|| unsafe { frizbee::prefilter::x86_64::match_haystack_unordered(needle, haystack) })
    });

    // Unordered Insensitive
    #[cfg(target_arch = "x86_64")]
    group.bench_function("x86_64/insensitive", |b| {
        b.iter(|| unsafe {
            frizbee::prefilter::x86_64::match_haystack_unordered_insensitive(needle_cased, haystack)
        })
    });

    // Unordered Typos
    #[cfg(target_arch = "x86_64")]
    group.bench_function("x86_64/typos", |b| {
        b.iter(|| unsafe {
            frizbee::prefilter::x86_64::match_haystack_unordered_typos(needle, haystack, 1)
        })
    });

    // Unordered Typos Insensitive
    #[cfg(target_arch = "x86_64")]
    group.bench_function("x86_64/typos/insensitive", |b| {
        b.iter(|| unsafe {
            frizbee::prefilter::x86_64::match_haystack_unordered_typos_insensitive(
                needle_cased,
                haystack,
                1,
            )
        })
    });

    group.finish();
}
