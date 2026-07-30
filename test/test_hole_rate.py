"""Unit tests for multirag.eval.hole_rate.hole_rate -- pure, no API calls.

Mirrors test/test_c0_3_reliability.py's convention: stdlib unittest, manual
sys.path insertion, hand-computed synthetic fixtures.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multirag.eval.hole_rate import hole_rate  # noqa: E402


class TestHoleRate(unittest.TestCase):
    def setUp(self):
        # Q1: 4 docs ranked, only d1/d3 judged (2 holes at k=4 -> 0.5)
        # Q2: 3 docs ranked, all judged (0 holes)
        self.ranking = {
            "Q1": ["d1", "d2", "d3", "d4"],
            "Q2": ["d5", "d6", "d7"],
        }
        self.qrels = {
            "Q1": {"d1": 2, "d3": 0},
            "Q2": {"d5": 1, "d6": 0, "d7": 2},
        }

    def test_hand_computed_hole_rate_at_k(self):
        # k=2: Q1 top-2 = [d1, d2] -> 1 hole / 2 = 0.5
        #      Q2 top-2 = [d5, d6] -> 0 holes / 2 = 0.0
        # mean over queries = (0.5 + 0.0) / 2 = 0.25
        self.assertAlmostEqual(hole_rate(self.ranking, self.qrels, k=2), 0.25)

    def test_hand_computed_hole_rate_full_ranking(self):
        # k=4: Q1 top-4 = 2 holes / 4 = 0.5; Q2 top-3 (only 3 exist) = 0/3 = 0.0
        self.assertAlmostEqual(hole_rate(self.ranking, self.qrels, k=4), 0.25)

    def test_no_holes_when_all_judged(self):
        ranking = {"Q2": ["d5", "d6", "d7"]}
        self.assertAlmostEqual(hole_rate(ranking, self.qrels, k=3), 0.0)

    def test_all_holes_when_query_missing_from_qrels(self):
        ranking = {"Q99": ["x", "y", "z"]}
        self.assertAlmostEqual(hole_rate(ranking, self.qrels, k=3), 1.0)

    def test_empty_ranking_returns_zero(self):
        self.assertEqual(hole_rate({}, self.qrels, k=10), 0.0)

    def test_query_with_empty_doc_list_is_skipped_not_zero_division(self):
        ranking = {"Q1": []}
        self.assertEqual(hole_rate(ranking, self.qrels, k=10), 0.0)

    def test_k_smaller_than_ranking_length_truncates(self):
        # k=1: Q1 top-1 = [d1] -> judged -> 0 holes / 1 = 0.0
        ranking = {"Q1": ["d1", "d2"]}
        self.assertAlmostEqual(hole_rate(ranking, self.qrels, k=1), 0.0)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            hole_rate(self.ranking, self.qrels, k=0)


if __name__ == "__main__":
    unittest.main()
