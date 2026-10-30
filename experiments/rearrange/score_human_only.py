#!/usr/bin/env python3
"""§5.3.2: the human_only channel and hole rates for the 15 reported runs, 78 topics (2026-09-18).

The ARQMath assessors' labels alone (official 2022 Task 1 qrels, 78 topics), full 1000-deep
runs, as in the original human_only channel:
  - standard nDCG/P @5/10/20  (an unjudged answer counts as not relevant)
  - prime   nDCG'/P' @5/10/20 (unjudged answers removed first)
  - hole@5/10/20              (share of the top k with no ARQMath judgment)
nDCG gain = grade (linear); P binarises at grade >= 2. Macro average over the 78 topics.

Writes results/rearranged/metrics_human_only/ (never overwrites).

Usage:
    python experiments/rearrange/score_human_only.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    CONFIGS, CUTOFFS, OFFICIAL_QRELS, OUT, hole_rate_per_topic, judged_only, ranking, read_qrels,
    read_run, topic_key, write_checksums,
)

OUT_H = OUT / "metrics_human_only"
MEASURES = {f"ndcg_cut.{','.join(map(str, CUTOFFS))}", f"P.{','.join(map(str, CUTOFFS))}"}
COLS = ([f"ndcg@{k}" for k in CUTOFFS] + [f"precision@{k}" for k in CUTOFFS]
        + [f"ndcg'@{k}" for k in CUTOFFS] + [f"precision'@{k}" for k in CUTOFFS]
        + [f"hole@{k}" for k in CUTOFFS])


def main() -> int:
    if OUT_H.exists() and any(OUT_H.iterdir()):
        sys.exit(f"refusing to overwrite existing files in {OUT_H} (nothing was written)")
    qrel = read_qrels(OFFICIAL_QRELS)
    assert len(qrel) == 78
    ev = pytrec_eval.RelevanceEvaluator(qrel, MEASURES, relevance_level=2)

    agg_rows, topic_rows = [], []
    for rid, label, cfg in CONFIGS:
        run = {t: d for t, d in read_run(OUT / "runs_task1" / f"{cfg}.tsv").items() if t in qrel}
        assert set(run) == set(qrel), f"{cfg}: run lacks topics {set(qrel) - set(run)}"
        std = ev.evaluate(run)
        prm = ev.evaluate(judged_only(run, qrel))
        rk = ranking(run)
        holes = {k: hole_rate_per_topic(rk, qrel, k) for k in CUTOFFS}
        sums = {c: 0.0 for c in COLS}
        for t in sorted(qrel, key=topic_key):
            row = {"registry_id": rid, "config_name": cfg, "topic_id": t}
            for k in CUTOFFS:
                row[f"ndcg@{k}"] = std[t][f"ndcg_cut_{k}"]
                row[f"precision@{k}"] = std[t][f"P_{k}"]
                row[f"ndcg'@{k}"] = prm[t][f"ndcg_cut_{k}"] if t in prm else 0.0
                row[f"precision'@{k}"] = prm[t][f"P_{k}"] if t in prm else 0.0
                row[f"hole@{k}"] = holes[k][t]
            for c in COLS:
                sums[c] += row[c]
            topic_rows.append(row)
        agg_rows.append({"registry_id": rid, "label": label, "config_name": cfg, "n_topics": 78,
                         **{c: sums[c] / 78 for c in COLS}})

    OUT_H.mkdir(parents=True, exist_ok=True)
    with open(OUT_H / "metrics_aggregate.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["registry_id", "label", "config_name", "n_topics"] + COLS)
        w.writeheader()
        w.writerows(agg_rows)
    with open(OUT_H / "metrics_per_topic.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["registry_id", "config_name", "topic_id"] + COLS)
        w.writeheader()
        w.writerows(topic_rows)
    with open(OUT_H / "README.md", "x") as f:
        f.write("""# metrics_human_only — ARQMath labels alone, and hole rates (thesis §5.3.2)

Written by `experiments/rearrange/score_human_only.py`. The 15 reported runs (`../runs_task1/`, full
1000-deep rankings) scored against the official 2022 Task 1 qrels (78 topics, copy in `../qrels_official/`),
the assessors' labels only. Standard measures count an unjudged answer as not relevant; prime measures
remove unjudged answers first; `hole@k` is the share of the top k with no ARQMath judgment. nDCG gain =
grade; P binarises at grade ≥ 2. Macro average over the 78 topics.

This is the channel §5.3.2 uses to show why a judge was needed; §5.4 reads `../metrics_judge_pool20/`.
""")
    write_checksums(OUT_H)
    print(f"{'id':6s} {'configuration':32s} {'ndcg@10':>8s} {'ndcg`@10':>8s} {'P@10':>6s} {'P`@10':>6s} {'hole@10':>7s} {'hole@20':>7s}")
    for a in agg_rows:
        print(f"{a['registry_id']:6s} {a['label'][:32]:32s} {a['ndcg@10']:8.3f} {a[chr(110)+'dcg'+chr(39)+'@10']:8.3f} "
              f"{a['precision@10']:6.3f} {a['precision'+chr(39)+'@10']:6.3f} {a['hole@10']:7.3f} {a['hole@20']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
