//! Benchmark: IncrementalMatcher vs one-shot match_list
//!
//! Measures the real-world benefit of incremental matching when a user types
//! a query character by character. Tests multiple query patterns, dataset sizes,
//! and shows per-step breakdowns where the narrowing effect is visible.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use frizbee::{Config, IncrementalMatcher, match_list};

fn main() {
    let config = Config::default();

    // ============================================================
    // 1. File-path dataset (realistic fuzzy finder workload)
    // ============================================================
    println!("=== File-Path Dataset (simulated project tree) ===\n");

    for &count in &[50_000usize, 200_000, 500_000] {
        let haystacks = generate_file_paths(count, 42);
        let refs: Vec<&str> = haystacks.iter().map(|s| s.as_str()).collect();

        // Multiple query patterns that resemble real fuzzy finder usage
        let queries: &[(&str, &[&str])] = &[
            // Searching for a file by abbreviation (high selectivity)
            ("mch", &["m", "mc", "mch"]),
            // Typing a common path segment (low selectivity early, high later)
            ("src/comp", &["s", "sr", "src", "src/", "src/c", "src/co", "src/com", "src/comp"]),
            // CamelCase search for a component name
            ("BtLs", &["B", "Bt", "BtL", "BtLs"]),
        ];

        println!("--- {} haystacks ---\n", count);

        for &(label, steps) in queries {
            bench_query(&refs, steps, label, &config);
        }
    }

    // ============================================================
    // 2. Selectivity comparison
    // ============================================================
    println!("\n=== Selectivity Impact ===\n");
    println!("Shows how incremental speedup varies with match ratio.\n");

    let haystacks = generate_file_paths(200_000, 42);
    let refs: Vec<&str> = haystacks.iter().map(|s| s.as_str()).collect();

    // Pairs: (query, description) — ordered by expected selectivity (high match% → low match%)
    let selectivity_queries: &[(&[&str], &str)] = &[
        (&["s", "sr", "src"], "common prefix (high match%)"),
        (&["c", "co", "com", "comp"], "medium selectivity"),
        (&["B", "Bt", "BtL", "BtLs"], "CamelCase (low match%)"),
        (&["z", "zx", "zxq"], "rare chars (very low match%)"),
    ];

    println!(
        "  {:30} {:>10} {:>10} {:>7} {:>10}",
        "Query", "One-shot", "Incremental", "Speedup", "Final matches"
    );
    println!("  {}", "-".repeat(71));

    for &(steps, desc) in selectivity_queries {
        let final_needle = steps.last().unwrap();
        let final_matches = match_list(final_needle, &refs, &config).len();

        let iters = 20u32;
        let one_shot = time_avg(iters, || {
            for &n in steps {
                black_box(match_list(n, black_box(&refs), &config));
            }
        });
        let incr = time_avg(iters, || {
            let mut m = IncrementalMatcher::new(&config);
            for &n in steps {
                black_box(m.match_list(n, black_box(&refs)));
            }
        });

        println!(
            "  {:30} {:>10.2?} {:>10.2?} {:>6.2}x {:>10}",
            format!("{:?} ({})", steps.last().unwrap(), desc),
            one_shot,
            incr,
            one_shot.as_nanos() as f64 / incr.as_nanos() as f64,
            final_matches,
        );
    }
}

/// Benchmark a single query pattern: overall + per-step breakdown
fn bench_query(haystacks: &[&str], steps: &[&str], label: &str, config: &Config) {
    let iters = 20u32;

    // Overall timing
    let one_shot_total = time_avg(iters, || {
        for &n in steps {
            black_box(match_list(n, black_box(haystacks), config));
        }
    });
    let incr_total = time_avg(iters, || {
        let mut m = IncrementalMatcher::new(config);
        for &n in steps {
            black_box(m.match_list(n, black_box(haystacks)));
        }
    });

    println!("  Query {:?} ({} steps)", label, steps.len());
    println!(
        "    Overall:  one-shot {:>9.2?}  incremental {:>9.2?}  ({:.2}x)",
        one_shot_total,
        incr_total,
        one_shot_total.as_nanos() as f64 / incr_total.as_nanos() as f64,
    );

    // Per-step breakdown
    println!(
        "    {:>10} {:>8} {:>10} {:>10} {:>7}",
        "Needle", "Matches", "One-shot", "Incremental", "Speedup"
    );

    for (step_idx, &needle) in steps.iter().enumerate() {
        let matches = match_list(needle, haystacks, config).len();

        // One-shot: just measure this single needle from scratch
        let os = time_avg(iters, || {
            black_box(match_list(black_box(needle), black_box(haystacks), config));
        });

        // Incremental: set up state from prior steps, then measure this step
        let inc = time_avg(iters, || {
            let mut m = IncrementalMatcher::new(config);
            // Replay prior steps to build up state
            for &prev in &steps[..step_idx] {
                m.match_list(prev, haystacks);
            }
            // Measure just this step
            black_box(m.match_list(black_box(needle), black_box(haystacks)));
        });
        // Subtract the setup cost to isolate this step's incremental time
        let setup = if step_idx > 0 {
            time_avg(iters, || {
                let mut m = IncrementalMatcher::new(config);
                for &prev in &steps[..step_idx] {
                    m.match_list(prev, haystacks);
                }
            })
        } else {
            Duration::ZERO
        };
        let inc_step = inc.saturating_sub(setup);

        // Guard against measurement noise where setup ≈ total
        let (inc_display, speedup_display) = if inc_step.as_nanos() == 0 {
            ("~0".to_string(), ">99".to_string())
        } else {
            let speedup = os.as_nanos() as f64 / inc_step.as_nanos() as f64;
            (format!("{:.2?}", inc_step), format!("{:.1}", speedup))
        };

        println!(
            "    {:>10} {:>8} {:>10.2?} {:>10} {:>6}x",
            format!("{:?}", needle),
            matches,
            os,
            inc_display,
            speedup_display,
        );
    }
    println!();
}

fn time_avg(iters: u32, mut f: impl FnMut()) -> Duration {
    // Warmup
    f();

    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters
}

/// Generate realistic file-path-like haystacks (e.g. "src/components/ButtonList.tsx")
fn generate_file_paths(count: usize, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);

    let dirs = [
        "src", "lib", "test", "docs", "build", "config", "scripts", "assets",
        "public", "vendor", "internal", "pkg", "cmd", "api", "web",
    ];
    let subdirs = [
        "components", "utils", "hooks", "services", "models", "views",
        "controllers", "middleware", "helpers", "types", "constants",
        "store", "actions", "reducers", "selectors", "pages", "layouts",
        "widgets", "auth", "config", "db", "cache", "queue", "workers",
    ];
    let prefixes = [
        "Button", "Input", "Modal", "Table", "Form", "List", "Card", "Nav",
        "Header", "Footer", "Sidebar", "Menu", "Dialog", "Panel", "Tab",
        "Search", "Filter", "Sort", "Page", "App", "User", "Auth", "Data",
        "Event", "File", "Image", "Video", "Audio", "Chart", "Graph",
        "Config", "Cache", "Queue", "Worker", "Handler", "Manager",
    ];
    let suffixes = [
        "", "Item", "List", "View", "Detail", "Edit", "Create", "Delete",
        "Update", "Form", "Modal", "Page", "Layout", "Container", "Wrapper",
        "Provider", "Context", "Hook", "Service", "Controller", "Model",
        "Helper", "Util", "Type", "Const", "Action", "Reducer", "Selector",
    ];
    let extensions = [
        ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java",
        ".css", ".scss", ".html", ".json", ".toml", ".yaml", ".md",
    ];

    let normal = Normal::new(3.0, 1.0).unwrap();

    (0..count)
        .map(|_| {
            // Random depth 1-5
            let depth = (normal.sample(&mut rng) as f64).round().abs().max(1.0) as usize;
            let depth = depth.min(5);
            let mut parts = Vec::with_capacity(depth + 1);

            parts.push(dirs[rng.random_range(0..dirs.len())].to_string());
            for _ in 1..depth {
                parts.push(subdirs[rng.random_range(0..subdirs.len())].to_string());
            }

            let prefix = prefixes[rng.random_range(0..prefixes.len())];
            let suffix = suffixes[rng.random_range(0..suffixes.len())];
            let ext = extensions[rng.random_range(0..extensions.len())];

            // Occasionally add a numeric suffix
            let num = if rng.random_ratio(1, 5) {
                format!("{}", rng.random_range(1..100u32))
            } else {
                String::new()
            };

            parts.push(format!("{}{}{}{}", prefix, suffix, num, ext));
            parts.join("/")
        })
        .collect()
}
