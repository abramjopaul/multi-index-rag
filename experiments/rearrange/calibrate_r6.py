#!/usr/bin/env python3
"""R6 / §5.3.1: evaluation-pipeline calibration, as a results file (2026-09-18).

Scores the ARQMath-3 TF-IDF (Terrier) baseline run against the official 2022 Task 1
qrels (78 topics) and compares with the organisers' published figures -- Mansouri et al.,
Overview of ARQMath-3, CEUR Vol-3180 paper-01, Table 3, row "TF-IDF (Terrier)":
nDCG' 0.272, MAP' 0.064, P'@10 0.124.

Same measures and conventions as experiments/track_b/calibrate_baseline.py (which only
prints): nDCG' = ndcg'@1000 (uncut on a 1000-deep run), MAP' uncut, P'@10; prime =
unjudged removed; relevance_level = 2. Macro average over the 78 topics.

Writes results/rearranged/calibration_r6/ (never overwrites).

Usage:
    python experiments/rearrange/calibrate_r6.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from common import OFFICIAL_QRELS, OUT, SRC_R6_RUN, judged_only, read_qrels, read_run, topic_key, write_checksums  # noqa: E402

OUT_R6 = OUT / "calibration_r6"
PUBLISHED = {"ndcg'": 0.272, "map'": 0.064, "precision'@10": 0.124}
KEYS = {"ndcg'": "ndcg_cut_1000", "map'": "map", "precision'@10": "P_10"}


def main() -> int:
    if OUT_R6.exists() and any(OUT_R6.iterdir()):
        sys.exit(f"refusing to overwrite existing files in {OUT_R6} (nothing was written)")
    qrel = read_qrels(OFFICIAL_QRELS)
    assert len(qrel) == 78
    run = read_run(SRC_R6_RUN)
    ev = pytrec_eval.RelevanceEvaluator(qrel, {"ndcg_cut.1000", "map", "P.10"}, relevance_level=2)
    res = ev.evaluate(judged_only({t: d for t, d in run.items() if t in qrel}, qrel))
    per_topic = {t: {m: (res[t][k] if t in res else 0.0) for m, k in KEYS.items()} for t in qrel}
    ours = {m: sum(v[m] for v in per_topic.values()) / 78 for m in KEYS}

    OUT_R6.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_R6_RUN, OUT_R6 / SRC_R6_RUN.name)
    with open(OUT_R6 / "calibration_task1_tfidf_terrier.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "ours", "published", "diff", "matches_at_published_precision", "n_topics", "qrels"])
        for m in KEYS:
            d = ours[m] - PUBLISHED[m]
            w.writerow([m, repr(ours[m]), PUBLISHED[m], repr(d), abs(d) <= 0.0005, 78, "qrel_task1_2022_official.tsv"])
    with open(OUT_R6 / "calibration_task1_tfidf_terrier_per_topic.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id"] + list(KEYS))
        for t in sorted(per_topic, key=topic_key):
            w.writerow([t] + [repr(per_topic[t][m]) for m in KEYS])
    with open(OUT_R6 / "README.md", "x") as f:
        f.write(f"""# calibration_r6 — evaluation-pipeline calibration (R6, thesis §5.3.1, precondition P1)

Written by `experiments/rearrange/calibrate_r6.py`. The run is the TF-IDF (Terrier) baseline distributed
with ARQMath-3 Task 1 (copy of `data/runs/arqmath-2022-runs/{SRC_R6_RUN.name}`), scored against the
official 2022 Task 1 qrels (78 topics, copy in `../qrels_official/`) with the same pytrec_eval settings
as every other Task 1 score here. Published figures: Mansouri et al., *Overview of ARQMath-3*, CEUR
Vol-3180 paper-01, Table 3, row "TF-IDF (Terrier)". The baseline is TF-IDF, not BM25.

| Measure | Ours | Published |
|---|---|---|
| nDCG′ | {ours["ndcg'"]:.4f} | 0.272 |
| MAP′ | {ours["map'"]:.4f} | 0.064 |
| P′@10 | {ours["precision'@10"]:.4f} | 0.124 |
""")
    write_checksums(OUT_R6)
    for m in KEYS:
        print(f"{m:15s} ours {ours[m]:.4f}  published {PUBLISHED[m]:.3f}  diff {ours[m] - PUBLISHED[m]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
