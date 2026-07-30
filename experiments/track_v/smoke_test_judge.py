#!/usr/bin/env python3
"""Track V judge smoke test -- run before any real spend.

Pushes one hand-labelled (topic, answer) pair through judge_sync, then
through the cache-first path submit_batch/partition_cached would take on a
repeat call. Asserts:
  - label in {0,1,2,3}, parse_ok=True on the first (real) call
  - the second judge_sync call on the same pair is a cache hit with an
    identical label
  - submit_batch on the now-cached pair returns zero new batch jobs (it's a
    pure cache lookup, proving the batch path shares the same cache as sync)
  - a manifest is written

Usage:
    poetry run python experiments/track_v/smoke_test_judge.py
    poetry run python experiments/track_v/smoke_test_judge.py --config configs/eval/judge.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.judge_config import JudgeConfigManager  # noqa: E402
from multirag.config.path_configs import (
    EVAL_CONFIG_DIR,  # noqa: E402
    RESULTS_TRACK_V_DIR,
)
from multirag.eval.judge_client import JudgeClient  # noqa: E402
from multirag.eval.manifest import build_manifest, sha256_file  # noqa: E402
from multirag.eval.schema import JudgePair  # noqa: E402

configure_logging(level="INFO")

_HAND_LABELLED_PAIR = JudgePair(
    topic_id="SMOKE.1",
    answer_id="smoke-answer-1",
    question="Question: Why is $\\sqrt{2}$ irrational? Show that $\\sqrt{2}$ "
    "cannot be written as a ratio of two integers.",
    answer_text="Suppose $\\sqrt{2} = p/q$ in lowest terms. Then $2q^2 = p^2$, so "
    "$p^2$ is even, so $p$ is even, so $p = 2k$. Substituting, $2q^2 = 4k^2$, so "
    "$q^2 = 2k^2$, so $q$ is also even -- contradicting that $p/q$ was in lowest terms.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Track V judge smoke test")
    parser.add_argument(
        "--config",
        default=str(EVAL_CONFIG_DIR / "judge.yaml"),
        help="Path to judge.yaml",
    )
    args = parser.parse_args()

    config = JudgeConfigManager.from_yaml(args.config)
    client = JudgeClient(config)

    print(f"Judge: provider={config.judge.provider} model={config.judge.model}")
    print(f"Cache dir: {config.cache.dir}")
    print(f"Prompt sha256: {client.prompt_sha256}")

    # --- 1. Sync path, first call: real judge call -----------------------
    results = client.judge_sync([_HAND_LABELLED_PAIR])
    assert len(results) == 1, f"expected 1 judgement, got {len(results)}"
    first = results[0]
    print(
        f"[sync #1] label={first.label} parse_ok={first.parse_ok} source={first.source}"
    )
    assert (
        first.parse_ok
    ), f"parse_ok should be True, got response: {first.raw_response!r}"
    assert first.label in {0, 1, 2, 3}, f"label must be 0-3, got {first.label}"

    # --- 2. Sync path, second call: must be a cache hit -------------------
    results2 = client.judge_sync([_HAND_LABELLED_PAIR])
    second = results2[0]
    print(
        f"[sync #2] label={second.label} parse_ok={second.parse_ok} source={second.source}"
    )
    assert (
        second.source == "cache"
    ), f"expected cache hit on repeat call, got source={second.source!r}"
    assert (
        second.label == first.label
    ), f"cached label {second.label} != original label {first.label}"

    # --- 3. Batch path: cache-first must short-circuit the same pair ------
    handles = client.submit_batch([_HAND_LABELLED_PAIR])
    print(f"[batch] new handles submitted: {len(handles)} (expect 0 -- already cached)")
    assert handles == [], "submit_batch should not re-submit an already-cached pair"

    cached, remaining = client.partition_cached([_HAND_LABELLED_PAIR])
    assert len(cached) == 1 and not remaining, "pair should be fully served from cache"
    assert (
        cached[0].label == first.label
    ), "batch-path cache hit label must match sync path"
    print(f"[batch] cache hit label={cached[0].label} -- identical to sync path")

    # --- 4. Manifest ------------------------------------------------------
    config_sha256 = sha256_file(args.config)
    manifest = build_manifest(
        config=config,
        execution_mode="sync+batch-smoke",
        prompt_sha256=client.prompt_sha256,
        config_sha256=config_sha256,
        usage=client.usage,
    )
    RESULTS_TRACK_V_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_TRACK_V_DIR / "smoke_test_judge_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
