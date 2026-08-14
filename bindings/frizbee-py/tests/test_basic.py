"""Minimal E2E checks to ensure the binding calls the core correctly."""

import inspect
import unittest

from frizbee import Haystacks, Matcher, Pattern, Scoring, parse_query


class BasicTests(unittest.TestCase):
    def test_match_list(self):
        matcher = Matcher("foo")
        matches = matcher.match(["foo", "prelude", "xfoo", "foobar"])
        self.assertEqual(sorted(m.index for m in matches), [0, 2, 3])
        self.assertEqual(matches[0].index, 0)
        self.assertTrue(matches[0].exact)
        scores = [m.score for m in matches]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_match_one_and_indices(self):
        matcher = Matcher("foo")
        self.assertIsNone(matcher.match_one("zzz", 0))
        self.assertEqual(matcher.match_one("foobar", 5).index, 5)
        indices = matcher.match_one_indices("foobar", 0).indices
        self.assertEqual(indices, (0, 1, 2))

    def test_indices_are_ascending_python_character_indices(self):
        matcher = Matcher("é")
        self.assertEqual(matcher.match_one_indices("xxé", 0).indices, (2,))
        self.assertEqual(matcher.match_indices(["xxé"])[0].indices, (2,))
        self.assertEqual(matcher.match_indices(Haystacks(["xxé"]))[0].indices, (2,))

        # The emoji occupies four UTF-8 bytes but only one Python character.
        gap = Matcher("éx", max_typos=None).match_one_indices("é😀x", 0)
        self.assertEqual(gap.indices, (0, 2))

    def test_match_list_parallel_equals_sequential(self):
        haystacks = [f"item {i}" for i in range(1000)] + ["foo"]
        matcher = Matcher("foo")
        expected = matcher.match(haystacks)
        self.assertEqual(matcher.match(haystacks, threads=0), expected)

    def test_from_query(self):
        matcher = Matcher.from_query("foo !^bar")
        matches = matcher.match(["foo", "barfoo", "foobar"])
        # "barfoo" starts with "bar"
        self.assertEqual(sorted(m.index for m in matches), [0, 2])

        patterns = parse_query("foo !^bar")
        self.assertEqual([p.needle for p in patterns], ["foo", "bar"])
        self.assertTrue(patterns[1].negated)
        rebuilt = Matcher(patterns)
        self.assertEqual(rebuilt.match(["foo", "barfoo", "foobar"]), matches)

    def test_haystacks(self):
        items = ["foo", "prelude", "xfoo", "foobar"]
        matcher = Matcher("foo")
        expected = matcher.match(items)

        haystacks = Haystacks(items)
        self.assertEqual(len(haystacks), 4)
        self.assertEqual(matcher.match(haystacks), expected)
        self.assertEqual(matcher.match(haystacks, threads=0), expected)
        self.assertEqual(matcher.match_indices(haystacks), matcher.match_indices(items))

        haystacks.clear()
        self.assertEqual(len(haystacks), 0)
        self.assertEqual(matcher.match(haystacks), [])

        haystacks.extend(["foo"])
        haystacks.append("xfoo")
        self.assertEqual(len(haystacks), 2)
        self.assertEqual([m.index for m in matcher.match(haystacks)], [0, 1])

    def test_haystacks_extend_is_atomic(self):
        haystacks = Haystacks(["base"])
        with self.assertRaises(UnicodeEncodeError):
            haystacks.extend(["added", "\ud800"])
        self.assertEqual(len(haystacks), 1)

    def test_bare_str_is_not_a_haystack_collection(self):
        matcher = Matcher("o")
        with self.assertRaisesRegex(TypeError, "iterable of str, not str"):
            Haystacks("foo")
        with self.assertRaisesRegex(TypeError, "iterable of str, not str"):
            matcher.match("foo")
        with self.assertRaisesRegex(TypeError, "iterable of str, not str"):
            matcher.match("foo", threads=1)
        with self.assertRaisesRegex(TypeError, "iterable of str, not str"):
            matcher.match_indices("foo")

    def test_pattern_setters(self):
        [pattern] = parse_query("foo")
        pattern.max_typos = 1
        matches = Matcher([pattern]).match(["fio", "xyz"])
        # "fio" matches with one typo via the per-pattern override
        self.assertEqual([m.index for m in matches], [0])

        pattern.matching = "prefix"
        matches = Matcher([pattern]).match(["xfoo", "foobar"])
        # "xfoo" doesn't start with "foo"
        self.assertEqual([m.index for m in matches], [1])

        with self.assertRaises(ValueError):
            pattern.matching = "bogus"
        with self.assertRaises(TypeError):
            pattern.max_typos = 1.5
        with self.assertRaises(AttributeError):
            pattern.needle = "bar"

    def test_from_patterns_and_options(self):
        matcher = Matcher(
            ["foo", Pattern("bar", negated=True)],
            max_typos=1,
            scoring=Scoring(match_score=24),
        )
        # "fio" matches with one typo, "foobar" is excluded by the negated pattern
        matches = matcher.match(["foo", "fio", "foobar"])
        self.assertEqual([m.index for m in matches], [0, 1])
        self.assertEqual(matcher.scoring.match_score, 24)

    def test_runtime_signatures_show_public_defaults(self):
        query_signature = inspect.signature(Matcher.from_query)
        self.assertEqual(query_signature.parameters["max_typos"].default, 0)
        self.assertEqual(query_signature.parameters["limit"].default, None)
        self.assertNotIn("Ellipsis", str(query_signature))

        scoring_signature = inspect.signature(Scoring)
        self.assertEqual(scoring_signature.parameters["match_score"].default, 12)
        self.assertEqual(scoring_signature.parameters["delimiter_bonus"].default, 4)


if __name__ == "__main__":
    unittest.main()
