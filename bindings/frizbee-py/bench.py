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
LIMIT = 1000


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

    matcher = frizbee.Matcher("linux", limit=LIMIT)
    # report match count and warm CPython's one-time UTF-8 cache
    matches = matcher.match(haystacks)
    print(
        f"chromium: {len(haystacks)} haystacks -> {len(matches)} matches, "
        f'needle "linux", limit = {LIMIT}'
    )

    start = time.perf_counter_ns()
    owned = frizbee.Haystacks(haystacks)
    print(
        f"{'new Haystacks(string[]):':<36}{(time.perf_counter_ns() - start) / 1e6:8.2f} ms"
    )

    bench("match (string[])", lambda: matcher.match(haystacks))
    bench("match (Haystacks)", lambda: matcher.match(owned))
    bench(
        f"match(threads={THREADS}) (string[])",
        lambda: matcher.match(haystacks, threads=THREADS),
    )
    bench(
        f"match(threads={THREADS}) (Haystacks)",
        lambda: matcher.match(owned, threads=THREADS),
    )


if __name__ == "__main__":
    sys.exit(main())
