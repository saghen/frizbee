"""Minimal E2E checks to ensure the binding calls the core correctly."""

import unittest

from frizbee import Haystacks, Matcher, Pattern, Scoring


class BasicTests(unittest.TestCase):
    def test_match_list(self):
        matcher = Matcher("foo")
        matches = matcher.match_list(["foo", "prelude", "xfoo", "foobar"])
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
        self.assertEqual(sorted(indices), [0, 1, 2])

    def test_match_list_parallel_equals_sequential(self):
        haystacks = [f"item {i}" for i in range(1000)] + ["foo"]
        matcher = Matcher("foo")
        expected = matcher.match_list(haystacks)
        self.assertEqual(matcher.match_list_parallel(haystacks, 0), expected)

    def test_from_query(self):
        matcher = Matcher.from_query("foo !^bar")
        matches = matcher.match_list(["foo", "barfoo", "foobar"])
        # "barfoo" starts with "bar"
        self.assertEqual(sorted(m.index for m in matches), [0, 2])

        patterns = Pattern.from_query("foo !^bar")
        self.assertEqual([p.needle for p in patterns], ["foo", "bar"])
        self.assertTrue(patterns[1].negated)
        rebuilt = Matcher.from_patterns(patterns)
        self.assertEqual(rebuilt.match_list(["foo", "barfoo", "foobar"]), matches)

    def test_haystacks(self):
        items = ["foo", "prelude", "xfoo", "foobar"]
        matcher = Matcher("foo")
        expected = matcher.match_list(items)

        haystacks = Haystacks(items)
        self.assertEqual(len(haystacks), 4)
        self.assertEqual(matcher.match_list(haystacks), expected)
        self.assertEqual(matcher.match_list_parallel(haystacks, 0), expected)
        self.assertEqual(
            matcher.match_list_indices(haystacks), matcher.match_list_indices(items)
        )

        haystacks.clear()
        self.assertEqual(len(haystacks), 0)
        self.assertEqual(matcher.match_list(haystacks), [])

        haystacks.extend(["foo"])
        haystacks.append("xfoo")
        self.assertEqual(len(haystacks), 2)
        self.assertEqual([m.index for m in matcher.match_list(haystacks)], [0, 1])

    def test_pattern_setters(self):
        [pattern] = Pattern.from_query("foo")
        pattern.max_typos = 1
        matches = Matcher.from_patterns([pattern]).match_list(["fio", "xyz"])
        # "fio" matches with one typo via the per-pattern override
        self.assertEqual([m.index for m in matches], [0])

        pattern.matching = "prefix"
        matches = Matcher.from_patterns([pattern]).match_list(["xfoo", "foobar"])
        # "xfoo" doesn't start with "foo"
        self.assertEqual([m.index for m in matches], [1])

        with self.assertRaises(ValueError):
            pattern.matching = "bogus"
        with self.assertRaises(TypeError):
            pattern.max_typos = 1.5
        with self.assertRaises(AttributeError):
            pattern.needle = "bar"

    def test_from_patterns_and_config(self):
        matcher = Matcher.from_patterns(
            ["foo", Pattern("bar", negated=True)],
            max_typos=1,
            scoring=Scoring(match_score=24),
        )
        # "fio" matches with one typo, "foobar" is excluded by the negated pattern
        matches = matcher.match_list(["foo", "fio", "foobar"])
        self.assertEqual([m.index for m in matches], [0, 1])
        self.assertEqual(matcher.scoring.match_score, 24)


if __name__ == "__main__":
    unittest.main()
