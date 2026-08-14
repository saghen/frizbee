// cargo build -p frizbee-c
// cc -std=c11 examples/smoke.c -Iinclude ../../target/debug/libfrizbee.a -lpthread -ldl -lm -o smoke
// ./smoke

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <frizbee/frizbee.h>

static frizbee_str_t str(const char *s) {
  frizbee_str_t out = {s, s ? strlen(s) : 0};
  return out;
}

int main(void) {
  // matcher_new with NULL config (defaults)
  frizbee_matcher_t *matcher = frizbee_matcher_new(str("fBr"), NULL);
  assert(matcher != NULL);

  frizbee_str_t haystacks[] = {
      str("fooBar"),   str("foo_bar"),
      str("barfoo"),   str("prelude"),
      str("println!"),
  };
  size_t haystacks_len = sizeof(haystacks) / sizeof(haystacks[0]);

  // match_list
  frizbee_matches_t matches =
      frizbee_matcher_match_list(matcher, haystacks, haystacks_len);
  assert(matches.len == 1);
  assert(matches.items[0].index == 0);
  assert(!matches.items[0].exact);
  frizbee_matches_free(&matches);
  frizbee_matches_free(&matches); // freed lists are zeroed; second free is a no-op

  // empty matches return NULL items
  frizbee_str_t no_haystacks[] = {str("zzz")};
  frizbee_matches_t no_matches =
      frizbee_matcher_match_list(matcher, no_haystacks, 1);
  assert(no_matches.len == 0);
  assert(no_matches.items == NULL);
  frizbee_matches_free(&no_matches);

  // match_list_parallel
  matches =
      frizbee_matcher_match_list_parallel(matcher, haystacks, haystacks_len, 2);
  assert(matches.len == 1);
  frizbee_matches_free(&matches);

  // match_one (with out pointer and with NULL out pointer)
  frizbee_match_t one;
  assert(frizbee_matcher_match_one(matcher, str("fooBar"), 42, &one));
  assert(one.index == 42);
  assert(frizbee_matcher_match_one(matcher, str("fooBar"), 42, NULL));
  assert(!frizbee_matcher_match_one(matcher, str("zzz"), 0, &one));
  assert(!frizbee_matcher_match_one(matcher, str("zzz"), 0, NULL));

  // match_one_indices
  uint32_t one_indices[8];
  size_t one_indices_len = 0;
  assert(frizbee_matcher_match_one_indices(matcher, str("fooBar"), 42,
                                           &one, one_indices, 8,
                                           &one_indices_len));
  assert(one.index == 42);
  assert(one_indices_len == 3);
  assert(one_indices[0] == 5 && one_indices[1] == 3 && one_indices[2] == 0);

  frizbee_matcher_free(matcher);
  frizbee_matcher_free(NULL); // no-op

  // query syntax constructor with NULL config
  matcher = frizbee_matcher_from_query(str("foo !^bar"), NULL);
  frizbee_str_t query_haystacks[] = {str("foo"), str("barfoo"), str("foobar")};
  matches = frizbee_matcher_match_list(matcher, query_haystacks, 3);
  assert(matches.len == 2);
  frizbee_matches_free(&matches);
  frizbee_matcher_free(matcher);

  // custom configuration is supplied when constructing a matcher
  frizbee_config_t config = frizbee_config_default();
  config.matching = FRIZBEE_MATCHING_EXACT;
  matcher = frizbee_matcher_new(str("foo"), &config);
  matches = frizbee_matcher_match_list(matcher, query_haystacks, 3);
  assert(matches.len == 1);
  assert(matches.items[0].index == 0);
  frizbee_matches_free(&matches);
  frizbee_matcher_free(matcher);

  printf("smoke.c ok\n");
  return 0;
}
