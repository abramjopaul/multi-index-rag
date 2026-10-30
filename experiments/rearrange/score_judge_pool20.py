#!/usr/bin/env python3
"""Score the 15 reported Task 1 runs against the ONE pooled judge qrel (2026-09-18).

Protocol (manifest §9, with the common-pool decision of 2026-09-18):
  - qrel: results/rearranged/judge_qrels_pooled/qrel_task1_2022_official_judge_pool20.tsv
          (union of the 15 runs' top-20 judge labels, 78 official topics) -- the same
          qrel, and therefore the same ideal ranking, for every configuration
  - run:  each topic's top 20 (score-descending; the depth the judge labelled), then
          unjudged answers removed -> prime variants
  - measures: nDCG'@5/10/20 (linear gain = grade) and P'@5/10/20 (grade >= 2),
          pytrec_eval, macro average over the 78 topics

Writes results/rearranged/metrics_judge_pool20/ (never overwrites):
  metrics_aggregate.csv, metrics_per_topic.csv, idcg_per_topic.csv, SHA256SUMS

Usage:
    python experiments/rearrange/score_judge_pool20.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    CONFIGS, CUTOFFS, OUT, POOL_DEPTH, POOLED_QREL_NAME, RELEVANCE_LEVEL, ranking,
    read_qrels, read_run, sha256, topic_key,
)

MEASURES = {f"ndcg_cut.{','.join(map(str, CUTOFFS))}", f"P.{','.join(map(str, CUTOFFS))}"}
COLS = [f"ndcg'@{k}" for k in CUTOFFS] + [f"precision'@{k}" for k in CUTOFFS]


def prime_top_n(run: dict, qrel: dict, depth: int) -> dict[str, dict[str, float]]:
    """Top-`depth` answers per qrel topic, then unjudged answers removed."""
    ranked = ranking(run)
    out = {}
    for t in qrel:
        top = ranked.get(t, [])[:depth]
        out[t] = {d: run[t][d] for d in top if d in qrel[t]}
    return out


def idcg(grades: list[int], k: int) -> float:
    g = sorted(grades, reverse=True)[:k]
    return sum(x / math.log2(i + 2) for i, x in enumerate(g) if x > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rearranged", type=Path, default=OUT)
    args = ap.parse_args()
    root = args.rearranged.resolve()
    out_dir = root / "metrics_judge_pool20"
    targets = [out_dir / n for n in ("metrics_aggregate.csv", "metrics_per_topic.csv",
                                     "idcg_per_topic.csv", "SHA256SUMS")]
    if any(p.exists() for p in targets):
        sys.exit(f"refusing to overwrite existing files in {out_dir} (nothing was written)")

    qrel = read_qrels(root / "judge_qrels_pooled" / POOLED_QREL_NAME)
    assert len(qrel) == 78, len(qrel)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, MEASURES, relevance_level=RELEVANCE_LEVEL)

    agg_rows, topic_rows = [], []
    for rid, label, cfg in CONFIGS:
        run = read_run(root / "runs_task1" / f"{cfg}.tsv")
        scored = prime_top_n(run, qrel, POOL_DEPTH)
        empty = [t for t, d in scored.items() if not d]
        assert not empty, f"{cfg}: no judged answer in the top {POOL_DEPTH} for {empty}"
        per_topic = evaluator.evaluate(scored)
        assert len(per_topic) == 78, f"{cfg}: scored {len(per_topic)} topics, expected 78"
        vals = {c: [] for c in COLS}
        for t in sorted(per_topic, key=topic_key):
            m = per_topic[t]
            row = {"registry_id": rid, "config_name": cfg, "topic_id": t,
                   "n_judged_in_top20": len(scored[t])}
            for k in CUTOFFS:
                row[f"ndcg'@{k}"] = m[f"ndcg_cut_{k}"]
                row[f"precision'@{k}"] = m[f"P_{k}"]
            for c in COLS:
                vals[c].append(row[c])
            topic_rows.append(row)
        agg = {"registry_id": rid, "label": label, "config_name": cfg, "n_topics": len(per_topic)}
        agg.update({c: sum(v) / len(v) for c, v in vals.items()})
        agg_rows.append(agg)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(targets[0], "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["registry_id", "label", "config_name", "n_topics"] + COLS)
        w.writeheader()
        w.writerows(agg_rows)
    with open(targets[1], "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["registry_id", "config_name", "topic_id", "n_judged_in_top20"] + COLS)
        w.writeheader()
        w.writerows(topic_rows)
    with open(targets[2], "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id"] + [f"idcg@{k}" for k in CUTOFFS])
        for t in sorted(qrel, key=topic_key):
            w.writerow([t] + [repr(idcg(list(qrel[t].values()), k)) for k in CUTOFFS])
    with open(targets[3], "x") as f:
        for p in targets[:3]:
            f.write(f"{sha256(p)}  {p.name}\n")

    print(f"{'id':6s} {'configuration':40s} " + " ".join(f"{c:>13s}" for c in COLS))
    for a in agg_rows:
        print(f"{a['registry_id']:6s} {a['label'][:40]:40s} " + " ".join(f"{a[c]:13.4f}" for c in COLS))
    print(f"\nwrote {len(targets)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
