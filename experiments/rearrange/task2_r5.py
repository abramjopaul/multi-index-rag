#!/usr/bin/env python3
"""R5 / §5.4.1: Task 2 results in one place -- TangentCFT2 reproduction + hole rates (2026-09-18).

1. Collapse TangentCFT2's submitted run (DPRL-Task2-CFT-auto-math-A.tsv, instance level:
   topic, formula_id, post_id, rank, score, run) to visually distinct formulas: walk each
   topic's list in rank order, map formula_id -> visual_id with the `visual_id` column of
   slt_representation_v3, keep the first occurrence. Judgments attach to visual ids.
2. Score it and this thesis's R5.1 run (already at visual-id level) against the official
   2022 Task 2 qrels (76 topics): nDCG' (= ndcg'@1000, uncut on a 1000-deep run), MAP'
   (uncut), P'@10 -- ARQMath's Table 4 measures -- plus nDCG'/P' @5/10/20.
   Prime = unjudged formulas removed; relevance_level = 2 (H+M) for MAP' and P'.
3. Hole rates @5/10/20/100/1000 for both runs, macro over the 76 assessed topics.

Writes results/rearranged/task2_r5/ (never overwrites). Streams ~10 GB of TSV for the
visual-id map, so expect several minutes; --cache-dir/--time-budget make that resumable.

Usage:
    python experiments/rearrange/task2_r5.py
    python experiments/rearrange/task2_r5.py --cache-dir /tmp/vidcache --time-budget 140   # repeat until exit 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    OFFICIAL_QRELS_T2, OUT, SLT_REPRESENTATION_DIR, SRC_T2_CFT, SRC_T2_OURS, hole_rate_per_topic,
    judged_only, ranking, read_qrels, read_run, sha256, topic_key, write_checksums,
)

OUT_T2 = OUT / "task2_r5"
CFT_COLLAPSED = "DPRL-Task2-CFT-auto-math-A.visual_id_collapsed.tsv"
HOLE_KS = (5, 10, 20, 100, 1000)
# ARQMath-3 overview (Mansouri et al., CEUR Vol-3180 paper-01), Table 4
PUBLISHED = {"TangentCFT2": {"ndcg'": 0.641, "map'": 0.419, "precision'@10": 0.534}}
MEASURES = {"ndcg_cut.5,10,20,1000", "P.5,10,20", "map"}
COLS = ["ndcg'@1000", "map'", "precision'@10", "ndcg'@5", "ndcg'@10", "ndcg'@20", "precision'@5", "precision'@20"]
KEYMAP = {"ndcg'@1000": "ndcg_cut_1000", "map'": "map", "precision'@10": "P_10", "ndcg'@5": "ndcg_cut_5",
          "ndcg'@10": "ndcg_cut_10", "ndcg'@20": "ndcg_cut_20", "precision'@5": "P_5", "precision'@20": "P_20"}


def read_cft(path: Path) -> list[tuple[str, str, int, float]]:
    rows = []
    with open(path, newline="") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 5:
                rows.append((p[0], p[1], int(p[3]), float(p[4])))
    return rows


def visual_id_map(needed: set[str], cache_dir: Path | None, budget_s: float) -> dict[str, str]:
    """formula_id -> visual_id for `needed`, first occurrence in shard order. With a cache
    dir, each shard's hits are saved as they are found, so a run that stops on the time
    budget (exit 75) resumes where it left off; the result is the same either way."""
    csv.field_size_limit(sys.maxsize)
    t0 = time.monotonic()
    key = hashlib.sha256("\n".join(sorted(needed)).encode()).hexdigest()[:16]
    if cache_dir:
        (cache_dir / key).mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    files = sorted(SLT_REPRESENTATION_DIR.glob("*.tsv"), key=lambda p: int(p.stem))
    for i, fp in enumerate(files, 1):
        cached = cache_dir / key / f"{fp.stem}.json" if cache_dir else None
        if cached and cached.exists():
            hits = json.loads(cached.read_text())
        else:
            if time.monotonic() - t0 > budget_s:
                print(f"time budget reached after {i - 1}/{len(files)} shards -- re-run to continue", flush=True)
                raise SystemExit(75)
            hits = {}
            with open(fp, newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                header = next(reader)
                i_id, i_vid = header.index("id"), header.index("visual_id")
                for row in reader:
                    if len(row) > i_vid and row[i_id] in needed and row[i_id] not in hits:
                        hits[row[i_id]] = row[i_vid].strip()
            if cached:
                tmp = cached.with_suffix(".tmp")
                tmp.write_text(json.dumps(hits))
                tmp.rename(cached)
        for fid, vid in hits.items():
            out.setdefault(fid, vid)
        if i % 20 == 0:
            print(f"  visual-id map: {i}/{len(files)} shards, {len(out)}/{len(needed)} resolved", flush=True)
    return out


def score(run: dict, qrel: dict) -> tuple[dict, dict]:
    ev = pytrec_eval.RelevanceEvaluator(qrel, MEASURES, relevance_level=2)
    res = ev.evaluate(judged_only({t: d for t, d in run.items() if t in qrel}, qrel))
    per_topic = {}
    for t in qrel:  # a topic with no judged formula in the run scores 0, it is not dropped
        m = res.get(t)
        per_topic[t] = {c: (m[KEYMAP[c]] if m else 0.0) for c in COLS}
    agg = {c: sum(v[c] for v in per_topic.values()) / len(per_topic) for c in COLS}
    return agg, per_topic


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="resumable cache for the visual-id scan (keep it outside the repo)")
    ap.add_argument("--time-budget", type=float, default=float("inf"),
                    help="seconds of shard scanning before exiting 75 to resume later")
    args = ap.parse_args()
    targets = [OUT_T2 / n for n in ("runs", "metrics_task2.csv", "metrics_task2_per_topic.csv",
                                    "hole_rates_task2.csv", "hole_rates_task2_per_topic.csv", "README.md")]
    if any(p.exists() for p in targets):
        sys.exit(f"refusing to overwrite existing files in {OUT_T2} (nothing was written)")

    qrel = read_qrels(OFFICIAL_QRELS_T2)
    assert len(qrel) == 76, len(qrel)

    cft = read_cft(SRC_T2_CFT)
    needed = {fid for _, fid, _, _ in cft}
    print(f"TangentCFT2 run: {len(cft)} rows, {len({t for t, *_ in cft})} topics, {len(needed)} formula ids")
    vmap = visual_id_map(needed, args.cache_dir, args.time_budget)
    unresolved = needed - set(vmap)
    empty = {f for f, v in vmap.items() if not v}
    assert not unresolved and not empty, f"{len(unresolved)} unresolved, {len(empty)} empty visual ids"

    by_topic: dict[str, list[tuple[int, str, float]]] = {}
    for t, fid, rk, s in cft:
        by_topic.setdefault(t, []).append((rk, fid, s))
    collapsed_lines, collapsed = [], {}
    for t in sorted(by_topic, key=topic_key):
        seen = set()
        for rk, fid, s in sorted(by_topic[t]):
            v = vmap[fid]
            if v in seen:
                continue
            seen.add(v)
            collapsed.setdefault(t, {})[v] = s
            collapsed_lines.append(f"{t}\tQ0\t{v}\t{len(seen)}\t{s!r}\tCFT\n")
    mean_len = sum(len(v) for v in collapsed.values()) / len(collapsed)

    ours = read_run(SRC_T2_OURS)
    runs = {"R5.1 (this thesis)": ours, "TangentCFT2": collapsed}
    results, holes = {}, {}
    for name, run in runs.items():
        results[name] = score(run, qrel)
        rk = {t: d for t, d in ranking(run).items() if t in qrel}
        holes[name] = {k: hole_rate_per_topic(rk, qrel, k) for k in HOLE_KS}
        assert all(len(holes[name][k]) == 76 for k in HOLE_KS), name

    # --- write (all computation done; nothing is written if anything above failed) ---
    (OUT_T2 / "runs").mkdir(parents=True)
    shutil.copy2(SRC_T2_OURS, OUT_T2 / "runs" / SRC_T2_OURS.name)
    shutil.copy2(SRC_T2_CFT, OUT_T2 / "runs" / SRC_T2_CFT.name)
    with open(OUT_T2 / "runs" / CFT_COLLAPSED, "x") as f:
        f.writelines(collapsed_lines)

    with open(OUT_T2 / "metrics_task2.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "n_topics"] + COLS + ["published_ndcg'", "published_map'", "published_precision'@10"])
        for name, (agg, _) in results.items():
            pub = PUBLISHED.get(name, {})
            w.writerow([name, 76] + [repr(agg[c]) for c in COLS]
                       + [pub.get("ndcg'", ""), pub.get("map'", ""), pub.get("precision'@10", "")])
    with open(OUT_T2 / "metrics_task2_per_topic.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "topic_id"] + COLS)
        for name, (_, pt) in results.items():
            for t in sorted(pt, key=topic_key):
                w.writerow([name, t] + [repr(pt[t][c]) for c in COLS])
    with open(OUT_T2 / "hole_rates_task2.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "n_topics"] + [f"hole@{k}" for k in HOLE_KS])
        for name, h in holes.items():
            w.writerow([name, 76] + [repr(sum(h[k].values()) / 76) for k in HOLE_KS])
    with open(OUT_T2 / "hole_rates_task2_per_topic.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "topic_id"] + [f"hole@{k}" for k in HOLE_KS])
        for name, h in holes.items():
            for t in sorted(qrel, key=topic_key):
                w.writerow([name, t] + [repr(h[k][t]) for k in HOLE_KS])

    cagg = results["TangentCFT2"][0]
    oagg = results["R5.1 (this thesis)"][0]
    with open(OUT_T2 / "README.md", "x") as f:
        f.write(f"""# task2_r5 — Task 2 formula retrieval (R5, thesis §5.4.1)

Written by `experiments/rearrange/task2_r5.py`. Qrels: `data/raw/qrels/qrel_task2_2022_official.tsv`
(76 assessed topics; a copy is in `../qrels_official/`). Measures follow ARQMath-3's Table 4: nDCG′ is
`ndcg'@1000` (uncut on a 1000-deep run), MAP′ uncut, P′@10; prime = unjudged formulas removed; MAP′ and P′
binarise at grade ≥ 2. Macro average over the 76 topics; a topic with no judged formula would score 0.

| File | What |
|---|---|
| `runs/{SRC_T2_OURS.name}` | R5.1, this thesis's formula block (visual-id level), copy of `data/runs/task2/` |
| `runs/{SRC_T2_CFT.name}` | TangentCFT2 as submitted (instance level), copy of `data/runs/task2/` |
| `runs/{CFT_COLLAPSED}` | TangentCFT2 collapsed to visually distinct formulas (first occurrence in rank order, `visual_id` from `slt_representation_v3`) — {len(needed):,} formula ids, all resolved; mean {mean_len:.1f} per topic over {len(collapsed)} topics |
| `metrics_task2.csv`, `metrics_task2_per_topic.csv` | Scores, with the published TangentCFT2 figures beside ours |
| `hole_rates_task2.csv`, `hole_rates_task2_per_topic.csv` | Share of the top k with no judgment, k ∈ {{5, 10, 20, 100, 1000}} |

**TangentCFT2 reproduction:** nDCG′ {cagg["ndcg'@1000"]:.4f} (published 0.641), MAP′ {cagg["map'"]:.4f} (0.419),
P′@10 {cagg["precision'@10"]:.4f} (0.534).
**R5.1:** nDCG′ {oagg["ndcg'@1000"]:.4f}, MAP′ {oagg["map'"]:.4f}, P′@10 {oagg["precision'@10"]:.4f}.
""")
    write_checksums(OUT_T2)

    print(f"collapsed TangentCFT2: mean {mean_len:.1f} visual ids per topic over {len(collapsed)} topics")
    for name, (agg, _) in results.items():
        print(f"{name:20s} " + "  ".join(f"{c}={agg[c]:.4f}" for c in COLS))
    for name, h in holes.items():
        print(f"{name:20s} " + "  ".join(f"hole@{k}={sum(h[k].values()) / 76:.3f}" for k in HOLE_KS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
