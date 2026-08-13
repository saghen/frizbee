import os
import statistics
import sys
import time

import frizbee

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "benches",
    "data",
    "chromium.txt",
)

WARMUP = 20
ITERS = 100
THREADS = 8
MAX_ITEMS = 1000


def bench(name, func):
    for _ in range(WARMUP):
        func()
    times = []
    for _ in range(ITERS):
        start = time.perf_counter_ns()
        func()
        times.append((time.perf_counter_ns() - start) / 1e6)
    print(f"{name + ':':<36}{statistics.mean(times):8.2f} ms/iter")


def main():
    if not os.path.exists(DATA_PATH):
        print("run benches/data/download.sh")
        return 1

    with open(DATA_PATH, encoding="utf-8") as f:
        haystacks = f.read().splitlines()

    matcher = frizbee.Matcher("linux", max_items=MAX_ITEMS)
    # report match count and warm CPython's one-time UTF-8 cache
    matches = matcher.match_list(haystacks)
    print(
        f"chromium: {len(haystacks)} haystacks -> {len(matches)} matches, "
        f'needle "linux", max_items = {MAX_ITEMS}'
    )

    start = time.perf_counter_ns()
    owned = frizbee.Haystacks(haystacks)
    print(
        f"{'new Haystacks(string[]):':<36}{(time.perf_counter_ns() - start) / 1e6:8.2f} ms"
    )

    bench("match_list (string[])", lambda: matcher.match_list(haystacks))
    bench("match_list (Haystacks)", lambda: matcher.match_list(owned))
    bench(
        f"match_list_parallel({THREADS}) (string[])",
        lambda: matcher.match_list_parallel(haystacks, THREADS),
    )
    bench(
        f"match_list_parallel({THREADS}) (Haystacks)",
        lambda: matcher.match_list_parallel(owned, THREADS),
    )


if __name__ == "__main__":
    sys.exit(main())
