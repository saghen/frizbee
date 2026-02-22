use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use frizbee::{Config, IncrementalMatcher, match_list};

const ITERS: u32 = 20;

fn main() {
    let config = Config::default();

    for &count in &[50_000usize, 200_000, 500_000] {
        let haystacks = gen_paths(count, 42);
        let refs: Vec<&str> = haystacks.iter().map(|s| s.as_str()).collect();

        println!("--- {} haystacks ---\n", count);

        bench_query(&refs, &["m", "mc", "mch"], &config);
        bench_query(
            &refs,
            &[
                "s", "sr", "src", "src/", "src/c", "src/co", "src/com", "src/comp",
            ],
            &config,
        );
        bench_query(&refs, &["B", "Bt", "BtL", "BtLs"], &config);
        bench_query(&refs, &["z", "zx", "zxq"], &config);
    }
}

fn bench_query(haystacks: &[&str], steps: &[&str], config: &Config) {
    let one_shot_total = time_avg(ITERS, || {
        for &n in steps {
            black_box(match_list(n, black_box(haystacks), config));
        }
    });
    let incr_total = time_avg(ITERS, || {
        let mut m = IncrementalMatcher::new(config);
        for &n in steps {
            black_box(m.match_list(n, black_box(haystacks)));
        }
    });

    let label = steps.last().unwrap();
    println!(
        "  {:?}: one-shot {:>9.2?}  incr {:>9.2?}  ({:.2}x)",
        label,
        one_shot_total,
        incr_total,
        one_shot_total.as_nanos() as f64 / incr_total.as_nanos() as f64,
    );

    println!(
        "    {:>10} {:>8} {:>10} {:>10} {:>7}",
        "needle", "matches", "one-shot", "incr", "speedup"
    );

    for (i, &needle) in steps.iter().enumerate() {
        let n_matches = match_list(needle, haystacks, config).len();

        let os = time_avg(ITERS, || {
            black_box(match_list(black_box(needle), black_box(haystacks), config));
        });

        // replay prior steps then measure just this one
        let inc = time_avg(ITERS, || {
            let mut m = IncrementalMatcher::new(config);
            for &prev in &steps[..i] {
                m.match_list(prev, haystacks);
            }
            black_box(m.match_list(black_box(needle), black_box(haystacks)));
        });
        let setup = if i > 0 {
            time_avg(ITERS, || {
                let mut m = IncrementalMatcher::new(config);
                for &prev in &steps[..i] {
                    m.match_list(prev, haystacks);
                }
            })
        } else {
            Duration::ZERO
        };
        let inc_step = inc.saturating_sub(setup);

        if inc_step.as_nanos() == 0 {
            println!(
                "    {:>10} {:>8} {:>10.2?} {:>10} {:>6}x",
                format!("{:?}", needle),
                n_matches,
                os,
                "~0",
                ">99"
            );
        } else {
            let speedup = os.as_nanos() as f64 / inc_step.as_nanos() as f64;
            println!(
                "    {:>10} {:>8} {:>10.2?} {:>10.2?} {:>5.1}x",
                format!("{:?}", needle),
                n_matches,
                os,
                inc_step,
                speedup
            );
        }
    }
    println!();
}

fn time_avg(iters: u32, mut f: impl FnMut()) -> Duration {
    f(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters
}

/// Generates paths like "src/components/ButtonList.tsx"
fn gen_paths(count: usize, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let depth_dist = Normal::new(3.0, 1.0).unwrap();

    let dirs = [
        "src", "lib", "test", "docs", "build", "config", "scripts", "assets", "public", "vendor",
        "internal", "pkg", "cmd", "api", "web",
    ];
    let subdirs = [
        "components",
        "utils",
        "hooks",
        "services",
        "models",
        "views",
        "controllers",
        "middleware",
        "helpers",
        "types",
        "store",
        "pages",
        "layouts",
        "widgets",
        "auth",
        "db",
        "cache",
    ];
    let names = [
        "Button", "Input", "Modal", "Table", "Form", "List", "Card", "Nav", "Header", "Footer",
        "Sidebar", "Menu", "Dialog", "Panel", "Search", "Filter", "Sort", "Page", "App", "User",
        "Auth", "Data", "Config", "Cache", "Handler", "Manager",
    ];
    let name_suffixes = [
        "",
        "Item",
        "List",
        "View",
        "Detail",
        "Edit",
        "Create",
        "Form",
        "Page",
        "Layout",
        "Container",
        "Provider",
        "Context",
        "Service",
        "Controller",
        "Helper",
    ];
    let exts = [
        ".rs", ".ts", ".tsx", ".js", ".py", ".go", ".java", ".css", ".json", ".toml", ".md",
    ];

    (0..count)
        .map(|_| {
            let depth = (depth_dist.sample(&mut rng) as f64)
                .round()
                .abs()
                .max(1.0)
                .min(5.0) as usize;
            let mut parts = Vec::with_capacity(depth + 1);

            parts.push(dirs[rng.random_range(0..dirs.len())]);
            for _ in 1..depth {
                parts.push(subdirs[rng.random_range(0..subdirs.len())]);
            }

            let name = names[rng.random_range(0..names.len())];
            let suffix = name_suffixes[rng.random_range(0..name_suffixes.len())];
            let ext = exts[rng.random_range(0..exts.len())];
            let num = if rng.random_ratio(1, 5) {
                rng.random_range(1..100u32).to_string()
            } else {
                String::new()
            };

            format!("{}/{name}{suffix}{num}{ext}", parts.join("/"))
        })
        .collect()
}
