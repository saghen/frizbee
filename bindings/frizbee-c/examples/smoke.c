// cargo build -p frizbee-c
// cc -std=c11 examples/smoke.c -Iinclude ../../target/debug/libfrizbee.a -lpthread -ldl -lm -o smoke
// ./smoke

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frizbee.h"

static frizbee_str_t str_lit(const char *s) {
  frizbee_str_t out = {s, strlen(s)};
  return out;
}

int main(void) {
  frizbee_config_t config = frizbee_config_default();

  frizbee_matcher_t *matcher = frizbee_matcher_new(str_lit("fBr"), &config);
  assert(matcher != NULL);

  frizbee_str_t haystacks[] = {
      str_lit("fooBar"),   str_lit("foo_bar"), str_lit("barfoo"),
      str_lit("prelude"), str_lit("println!"),
  };
  size_t haystacks_len = sizeof(haystacks) / sizeof(haystacks[0]);

  // match_list
  frizbee_matches_t matches = frizbee_match_list(matcher, haystacks, haystacks_len);
  assert(matches.len == 1);
  assert(matches.items[0].index == 0);
  assert(!matches.items[0].exact);
  frizbee_matches_free(&matches);
  frizbee_matches_free(&matches); // freed lists are zeroed; second free is a no-op

  // match_list_parallel
  matches = frizbee_match_list_parallel(matcher, haystacks, haystacks_len, 2);
  assert(matches.len == 1);
  frizbee_matches_free(&matches);

  // match_list_indices
  frizbee_match_indices_list_t indices =
      frizbee_match_list_indices(matcher, haystacks, haystacks_len);
  assert(indices.len == 1);
  frizbee_match_indices_t m = indices.items[0];
  assert(m.index == 0);
  assert(m.indices_len == 3); // "fBr" matched 3 bytes of "fooBar"
  printf("matched \"fooBar\" (score %u), byte offsets (reverse order):", m.score);
  for (uint32_t i = 0; i < m.indices_len; i++)
    printf(" %u", indices.indices[m.indices_start + i]);
  printf("\n");
  frizbee_match_indices_list_free(&indices);

  // match_one
  frizbee_match_t one;
  assert(frizbee_match_one(matcher, str_lit("fooBar"), 42, &one));
  assert(one.index == 42);
  assert(!frizbee_match_one(matcher, str_lit("zzz"), 0, &one));

  // threads == 0 defaults to available CPU cores - 2
  matches = frizbee_match_list_parallel(matcher, haystacks, haystacks_len, 0);
  assert(matches.len == 1);
  frizbee_matches_free(&matches);

  frizbee_matcher_free(matcher);
  frizbee_matcher_free(NULL); // no-op

  // query syntax
  matcher = frizbee_matcher_from_query(str_lit("foo !^bar"), &config);
  frizbee_str_t query_haystacks[] = {str_lit("foo"), str_lit("barfoo"), str_lit("foobar")};
  matches = frizbee_match_list(matcher, query_haystacks, 3);
  assert(matches.len == 2); // "barfoo" starts with "bar"
  frizbee_matches_free(&matches);
  frizbee_matcher_free(matcher);

  printf("smoke.c ok\n");
  return 0;
}
