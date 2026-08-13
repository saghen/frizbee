// Header-only C++17 wrapper over the frizbee C API (frizbee.h). Hand-written;
// no ABI of its own — everything forwards to the extern "C" functions.
//
// Strings are NOT validated as UTF-8 (see frizbee.h); passing invalid UTF-8 is
// undefined behavior. A Matcher is not thread-safe: use one per thread or lock.

#ifndef FRIZBEE_HPP
#define FRIZBEE_HPP

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "frizbee.h"

namespace frizbee {

// Plain C structs; get defaults from `default_config()` and tweak fields
using Config = frizbee_config_t;
using Scoring = frizbee_scoring_t;
using Match = frizbee_match_t;

inline Config default_config() { return frizbee_config_default(); }

/// Needle length up to which scores are guaranteed to fit within the
/// `uint16_t` score. Longer needles still match, but their scores may
/// saturate at `UINT16_MAX`
inline std::size_t max_needle_len(const Scoring &scoring) {
  return frizbee_scoring_max_needle_len(&scoring);
}

/// Like `Match` but includes the indices of the chars in the haystack that
/// matched the needle in reverse order
struct MatchIndices {
  std::uint16_t score;
  std::uint32_t index;
  bool exact;
  std::vector<std::uint32_t> indices;
};

namespace detail {

inline frizbee_str_t to_str(std::string_view s) {
  return frizbee_str_t{s.data(), s.size()};
}

// Accepts any range of string_view-convertibles (std::string, const char*, ...).
// The range must yield lvalue references (or std::string_view by value): the
// haystack bytes are only borrowed until the C call returns, so a range that
// materializes temporary owning strings (e.g. a transform returning
// std::string) would dangle
template <typename Range>
std::vector<frizbee_str_t> to_strs(const Range &haystacks) {
  using std::begin;
  using element_t = decltype(*begin(haystacks));
  static_assert(
      std::is_lvalue_reference_v<element_t> ||
          std::is_same_v<std::remove_cv_t<std::remove_reference_t<element_t>>,
                         std::string_view>,
      "this range yields temporary owning strings whose bytes die before "
      "frizbee reads them; materialize it (e.g. into a "
      "std::vector<std::string>) first");
  std::vector<frizbee_str_t> out;
  for (const auto &haystack : haystacks)
    out.push_back(to_str(std::string_view(haystack)));
  return out;
}

// Frees the C result buffer on scope exit, so a throwing std::vector
// constructor (bad_alloc) cannot leak it
struct matches_owner {
  frizbee_matches_t &m;
  ~matches_owner() { frizbee_matches_free(&m); }
};
struct match_indices_list_owner {
  frizbee_match_indices_list_t &m;
  ~match_indices_list_owner() { frizbee_match_indices_list_free(&m); }
};

inline std::vector<Match> collect(frizbee_matches_t &matches) {
  matches_owner owner{matches};
  return std::vector<Match>(matches.items, matches.items + matches.len);
}

} // namespace detail

/// RAII wrapper over `frizbee_matcher_t`. Move-only.
///
/// Compiles the pattern once, allocates memory for the Smith Waterman matrix, and
/// reuses the selected SIMD backend across calls. Ideally, only construct these at
/// most once per list
class Matcher {
public:
  /// Matches `needle` literally, as in `frizbee_matcher_new`
  explicit Matcher(std::string_view needle, const Config &config = default_config())
      : handle_(frizbee_matcher_new(detail::to_str(needle), &config)) {}

  /// Parses a query of whitespace separated atoms (`foo`, `^foo`, `foo$`, `'foo`,
  /// `^foo$`, `!foo`), as in `frizbee_matcher_from_query`
  static Matcher from_query(std::string_view query, const Config &config = default_config()) {
    return Matcher(query_tag{}, query, config);
  }

  ~Matcher() { frizbee_matcher_free(handle_); }

  Matcher(Matcher &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Matcher &operator=(Matcher &&other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  Matcher(const Matcher &) = delete;
  Matcher &operator=(const Matcher &) = delete;

  /// Matches a range of string_view-convertibles, ordered by the config's
  /// sort strategy
  template <typename Range>
  std::vector<Match> match_list(const Range &haystacks) {
    const auto strs = detail::to_strs(haystacks);
    frizbee_matches_t out = frizbee_match_list(handle_, strs.data(), strs.size());
    return detail::collect(out);
  }
  std::vector<Match> match_list(std::initializer_list<std::string_view> haystacks) {
    return match_list<>(haystacks);
  }

  /// Like `match_list`, matching in parallel on `threads` real threads
  /// (`0` = available CPU cores - 2)
  template <typename Range>
  std::vector<Match> match_list_parallel(const Range &haystacks, std::size_t threads) {
    const auto strs = detail::to_strs(haystacks);
    frizbee_matches_t out =
        frizbee_match_list_parallel(handle_, strs.data(), strs.size(), threads);
    return detail::collect(out);
  }
  std::vector<Match> match_list_parallel(std::initializer_list<std::string_view> haystacks,
                                         std::size_t threads) {
    return match_list_parallel<>(haystacks, threads);
  }

  /// Like `match_list`, but each match includes the indices of the chars in the
  /// haystack that matched the needle. Not optimized for performance: intended
  /// for small lists, e.g. the visible portion of results
  template <typename Range>
  std::vector<MatchIndices> match_list_indices(const Range &haystacks) {
    const auto strs = detail::to_strs(haystacks);
    frizbee_match_indices_list_t out =
        frizbee_match_list_indices(handle_, strs.data(), strs.size());
    detail::match_indices_list_owner owner{out};

    std::vector<MatchIndices> result;
    result.reserve(out.len);
    for (std::size_t i = 0; i < out.len; i++) {
      const frizbee_match_indices_t &item = out.items[i];
      const std::uint32_t *start = out.indices + item.indices_start;
      result.push_back(MatchIndices{
          item.score, item.index, item.exact,
          std::vector<std::uint32_t>(start, start + item.indices_len)});
    }
    return result;
  }
  std::vector<MatchIndices>
  match_list_indices(std::initializer_list<std::string_view> haystacks) {
    return match_list_indices<>(haystacks);
  }

  /// Matches a single haystack; `index` is echoed into the match. Consider using
  /// `match_list` if you have more than one haystack to match, as it performs
  /// significantly better
  std::optional<Match> match_one(std::string_view haystack, std::uint32_t index) {
    Match out{};
    if (!frizbee_match_one(handle_, detail::to_str(haystack), index, &out))
      return std::nullopt;
    return out;
  }

private:
  struct query_tag {};
  Matcher(query_tag, std::string_view query, const Config &config)
      : handle_(frizbee_matcher_from_query(detail::to_str(query), &config)) {}

  frizbee_matcher_t *handle_ = nullptr;
};

} // namespace frizbee

#endif // FRIZBEE_HPP
