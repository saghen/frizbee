// cargo build -p frizbee-c
// c++ -std=c++17 examples/smoke.cpp -Iinclude ../../target/debug/libfrizbee.a -lpthread -ldl -lm -o smoke-cpp
// ./smoke-cpp

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "frizbee.hpp"

int main() {
  frizbee::Matcher matcher("fBr");

  auto matches = matcher.match_list({"fooBar", "foo_bar", "barfoo", "prelude", "println!"});
  assert(matches.size() == 1);
  assert(matches[0].index == 0);
  assert(!matches[0].exact);

  // any range of string_view convertibles works
  std::vector<std::string> haystacks = {"fooBar", "foo_bar", "barfoo"};
  auto parallel = matcher.match_list_parallel(haystacks, 2);
  assert(parallel.size() == 1);
  assert(parallel[0].index == 0);

  auto indices = matcher.match_list_indices(haystacks);
  assert(indices.size() == 1);
  assert(indices[0].indices.size() == 3); // "fBr" matched 3 bytes of "fooBar"

  auto one = matcher.match_one("fooBar", 42);
  assert(one.has_value());
  assert(one->index == 42);
  assert(!matcher.match_one("zzz", 0).has_value());

  auto query = frizbee::Matcher::from_query("foo !^bar");
  auto query_matches = query.match_list({"foo", "barfoo", "foobar"});
  assert(query_matches.size() == 2); // "barfoo" starts with "bar"

  // moved-from matchers destruct safely
  frizbee::Matcher moved = std::move(matcher);
  assert(moved.match_list({"fooBar"}).size() == 1);

  std::printf("smoke.cpp ok\n");
  return 0;
}
