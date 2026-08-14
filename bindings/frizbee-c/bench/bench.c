// Chromium benchmark: needle "linux" vs benches/data/chromium.txt. Labels and
// output format mirror bindings/frizbee-py/bench.py and
// bindings/frizbee-wasm/bench.mjs so the boundary costs can be compared
// directly. The C API has no max_items and no Haystacks: matches come back as
// raw structs and the caller's frizbee_str_t array is already zero-copy, so
// only the string[] cases exist here.
//
// cargo build --release -p frizbee-c
// cc -O2 -std=c11 bench/bench.c -Iinclude ../../target/release/libfrizbee.a
//    -lpthread -ldl -lm -o bench-c && ./bench-c   (one line, from this dir)

// for clock_gettime under strict -std=c11
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <frizbee/frizbee.h>

#define WARMUP 20
#define ITERS 100
#define THREADS 8

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void report(const char *name, const double times[ITERS]) {
  double sum = 0;
  for (int i = 0; i < ITERS; i++) {
    sum += times[i];
  }
  char label[64];
  snprintf(label, sizeof(label), "%s:", name);
  printf("%-36s%8.2f ms/iter\n", label, sum / ITERS);
}

static char *read_file(const char *path, size_t *out_size) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;

  // ensure the file has content
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  if (size <= 0) {
    fclose(f);
    return NULL;
  }
  fseek(f, 0, SEEK_SET);

  // read the file
  char *buf = malloc((size_t)size);
  if (!buf || fread(buf, 1, (size_t)size, f) != (size_t)size) {
    fclose(f);
    free(buf);
    return NULL;
  }

  fclose(f);
  *out_size = (size_t)size;
  return buf;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "benches/data/chromium.txt";
  size_t size = 0;
  char *buf = read_file(path, &size);
  if (!buf) {
    fprintf(stderr, "missing chromium.txt, run `just download-bench-data`\n");
    return 1;
  }

  // allocate memory for haystacks
  size_t count = 0;
  for (size_t i = 0; i < size; i++)
    if (buf[i] == '\n')
      count++;
  frizbee_str_t *haystacks = malloc((count + 1) * sizeof(frizbee_str_t));
  if (!haystacks) {
    free(buf);
    return 1;
  }

  // split file by lines and populate haystacks
  size_t len = 0, start = 0;
  for (size_t i = 0; i < size; i++) {
    if (buf[i] != '\n')
      continue;
    haystacks[len].ptr = buf + start;
    haystacks[len].len = i - start;
    len++;
    start = i + 1;
  }
  if (start < size) {
    haystacks[len].ptr = buf + start;
    haystacks[len].len = size - start;
    len++;
  }

  frizbee_matcher_t *matcher = frizbee_matcher_new((frizbee_str_t){"linux", 5}, NULL);

  frizbee_matches_t matches;
  double times[ITERS];

  // untimed first call, reports the match count
  matches = frizbee_matcher_match_list(matcher, haystacks, len);
  printf("chromium: %zu haystacks -> %zu matches, needle \"linux\"\n", len, matches.len);
  frizbee_matches_free(&matches);

  // bench `frizbee_matcher_match_list`
  for (int i = 0; i < WARMUP; i++) {
    matches = frizbee_matcher_match_list(matcher, haystacks, len);
    frizbee_matches_free(&matches);
  }
  for (int i = 0; i < ITERS; i++) {
    double t0 = now_ms();
    matches = frizbee_matcher_match_list(matcher, haystacks, len);
    times[i] = now_ms() - t0;
    frizbee_matches_free(&matches);
  }
  report("match_list (string[])", times);

  // bench `frizbee_matcher_match_list_parallel`
  for (int i = 0; i < WARMUP; i++) {
    matches = frizbee_matcher_match_list_parallel(matcher, haystacks, len, THREADS);
    frizbee_matches_free(&matches);
  }
  for (int i = 0; i < ITERS; i++) {
    double t0 = now_ms();
    matches = frizbee_matcher_match_list_parallel(matcher, haystacks, len, THREADS);
    times[i] = now_ms() - t0;
    frizbee_matches_free(&matches);
  }
  char parallel_name[64];
  snprintf(parallel_name, sizeof(parallel_name), "match_list_parallel(%d) (string[])", THREADS);
  report(parallel_name, times);

  frizbee_matcher_free(matcher);
  free(haystacks);
  free(buf);
  return 0;
}
