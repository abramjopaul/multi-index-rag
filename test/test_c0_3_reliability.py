"""Unit tests for the per-topic / per-repeat noise aggregation in run_c0_3.py.

Covers Debug Steps 4 & 5 from the c0_2_block_k5 C0.3 reliability-run
investigation (per_topic_std == 0 for every metric despite repeat_std > 0):

- test_per_topic_std_hand_computed / test_repeat_std_hand_computed /
  test_mean_per_topic_std_hand_computed: verify RagasEvaluator.aggregate()
  and run_c0_3's two reduction paths (across-repeats-per-topic vs
  across-topics-per-repeat) against hand-computed values on synthetic
  3-topic x 2-repeat data, confirming there is no axis/transposition bug.
- TestNaNDropoutInconsistency: reproduces the actual root-cause mechanism
  found in the production run (per-repeat NaN dropout from judge timeouts
  leaves every topic with < 2 usable data points, so every topic's own std
  is trivially 0, while the run-level mean still shifts between repeats
  because a different subset of topics contributes to each repeat's mean)
  and confirms check_per_topic_std_consistency() catches it.

Uses stdlib unittest only (no new test-framework dependency).
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multirag.generation.ragas_eval import RagasEvaluator  # noqa: E402
from run_c0_3 import (  # noqa: E402
    check_per_topic_std_consistency,
    compute_mean_per_topic_std,
    compute_overall_noise_floor,
)

NAN = float("nan")
METRIC = "context_precision"


def _aggregate(per_sample: list[dict]) -> dict:
    # RagasEvaluator.aggregate() reads no `self` state (verified by inspection);
    # calling it unbound avoids constructing a real judge client (network/API key).
    return RagasEvaluator.aggregate(None, per_sample)


def _per_topic_noise(all_repeats: list[list[dict]], topic_ids: list[str]) -> dict[str, dict]:
    per_topic_noise = {}
    for idx, tid in enumerate(topic_ids):
        repeats_for_topic = [all_repeats[r][idx] for r in range(len(all_repeats))]
        per_topic_noise[tid] = _aggregate(repeats_for_topic)
    return per_topic_noise


class TestGenuineVariationHandComputed(unittest.TestCase):
    """3 topics x 2 repeats. T1, T2 constant across repeats; T3 varies
    0.5 -> 0.8. Chosen so every intermediate is exact (no rounding noise)
    except the final sqrt in repeat_std.
    """

    def setUp(self):
        self.topic_ids = ["T1", "T2", "T3"]
        self.all_repeats = [
            [{METRIC: 0.4}, {METRIC: 0.6}, {METRIC: 0.5}],  # repeat 1, mean = 0.5
            [{METRIC: 0.4}, {METRIC: 0.6}, {METRIC: 0.8}],  # repeat 2, mean = 0.6
        ]

    def test_per_topic_std_hand_computed(self):
        per_topic_noise = _per_topic_noise(self.all_repeats, self.topic_ids)
        self.assertEqual(per_topic_noise["T1"][f"{METRIC}_std"], 0.0)
        self.assertEqual(per_topic_noise["T2"][f"{METRIC}_std"], 0.0)
        # population std of [0.5, 0.8] = sqrt(((0.5-0.65)^2+(0.8-0.65)^2)/2) = 0.15
        self.assertAlmostEqual(per_topic_noise["T3"][f"{METRIC}_std"], 0.15, places=6)

    def test_mean_per_topic_std_hand_computed(self):
        per_topic_noise = _per_topic_noise(self.all_repeats, self.topic_ids)
        mean_pts = compute_mean_per_topic_std(per_topic_noise, [METRIC])
        # mean(0, 0, 0.15) = 0.05
        self.assertAlmostEqual(mean_pts[METRIC], 0.05, places=6)

    def test_repeat_std_hand_computed(self):
        run_level_aggregates = [_aggregate(self.all_repeats[r]) for r in range(2)]
        self.assertEqual(run_level_aggregates[0][f"{METRIC}_mean"], 0.5)
        self.assertEqual(run_level_aggregates[1][f"{METRIC}_mean"], 0.6)
        noise_floor = compute_overall_noise_floor(run_level_aggregates)
        # sample stdev (ddof=1) of [0.5, 0.6]: mean=0.55, sum((v-mean)^2)=0.005,
        # variance = 0.005 / (2-1) = 0.005, std = sqrt(0.005)
        self.assertAlmostEqual(noise_floor[METRIC]["repeat_std"], math.sqrt(0.005), places=4)

    def test_consistency_check_does_not_raise_on_genuine_variation(self):
        # repeat_std > 0 here is explained by T3's own std also being > 0 --
        # a real, non-degenerate case, not a per-repeat-dropout artifact.
        run_level_aggregates = [_aggregate(self.all_repeats[r]) for r in range(2)]
        noise_floor = compute_overall_noise_floor(run_level_aggregates)
        mean_pts = compute_mean_per_topic_std(
            _per_topic_noise(self.all_repeats, self.topic_ids), [METRIC]
        )
        check_per_topic_std_consistency(noise_floor, mean_pts, Path("unused.csv"))


class TestNaNDropoutInconsistency(unittest.TestCase):
    """3 topics x 2 repeats, one NaN per topic (simulating a judge-call
    timeout landing on a different topic each repeat -- the mechanism
    identified in the c0_2_block_k5 production run's timeout log).
    """

    def setUp(self):
        self.topic_ids = ["T1", "T2", "T3"]
        self.all_repeats = [
            [{METRIC: NAN}, {METRIC: 0.3}, {METRIC: 0.5}],  # repeat 1
            [{METRIC: 0.9}, {METRIC: NAN}, {METRIC: 0.5}],  # repeat 2
        ]

    def test_per_topic_std_is_trivially_zero(self):
        per_topic_noise = _per_topic_noise(self.all_repeats, self.topic_ids)
        for tid in self.topic_ids:
            self.assertEqual(per_topic_noise[tid][f"{METRIC}_std"], 0.0)

    def test_repeat_std_is_nonzero_despite_zero_per_topic_std(self):
        run_level_aggregates = [_aggregate(self.all_repeats[r]) for r in range(2)]
        # repeat 1 mean over non-NaN topics (T2, T3) = (0.3+0.5)/2 = 0.4
        # repeat 2 mean over non-NaN topics (T1, T3) = (0.9+0.5)/2 = 0.7
        self.assertAlmostEqual(run_level_aggregates[0][f"{METRIC}_mean"], 0.4, places=6)
        self.assertAlmostEqual(run_level_aggregates[1][f"{METRIC}_mean"], 0.7, places=6)
        noise_floor = compute_overall_noise_floor(run_level_aggregates)
        self.assertGreater(noise_floor[METRIC]["repeat_std"], 1e-6)

    def test_consistency_check_raises(self):
        run_level_aggregates = [_aggregate(self.all_repeats[r]) for r in range(2)]
        noise_floor = compute_overall_noise_floor(run_level_aggregates)
        mean_pts = compute_mean_per_topic_std(
            _per_topic_noise(self.all_repeats, self.topic_ids), [METRIC]
        )
        self.assertEqual(mean_pts[METRIC], 0.0)
        with self.assertRaises(RuntimeError):
            check_per_topic_std_consistency(noise_floor, mean_pts, Path("unused.csv"))


if __name__ == "__main__":
    unittest.main()
