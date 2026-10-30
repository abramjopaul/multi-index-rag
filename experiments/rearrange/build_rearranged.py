#!/usr/bin/env python3
"""Build results/rearranged/ -- the RQ1 evaluation inputs in one place (2026-09-18).

Creates, and never overwrites:
  runs_task1/              the 15 reported Task 1 run files, byte-identical copies
  judge_qrels_per_run/     each run's top-20 judge labels as TREC qrels, 78 official topics
  judge_qrels_pooled/      the ONE qrel every run is scored against: the union of the
                           15 runs' top-20 judge labels (depth-20 pool), 78 official topics
  track_v/                 copies of the judge-validation files (originals untouched)
  README.md, SHA256SUMS

Sources are only read. If any output file already exists the script stops
before writing anything.

Usage:
    python experiments/rearrange/build_rearranged.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import argparse  # noqa: E402

from common import (  # noqa: E402
    CONFIGS, OFFICIAL_QRELS, OUT as DEFAULT_OUT, PER_RUN_SUFFIX, POOL_DEPTH, POOLED_QREL_NAME,
    ROOT, SRC_AUDIT_DIR, SRC_RUNS_DIR, SRC_TRACK_V_DIR, doc_key, read_qrels, sha256, topic_key,
)


def load_audit(config: str) -> list[dict]:
    path = SRC_AUDIT_DIR / f"{config}_answers_audit.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["rank"] = int(r["rank"])
        r["llm_label"] = None if r["llm_label"] in ("", "null", "None") else int(r["llm_label"])
    return rows


def write_qrel(path: Path, pairs: dict[tuple[str, str], int]) -> int:
    lines = [
        f"{t}\t0\t{d}\t{g}\n"
        for (t, d), g in sorted(pairs.items(), key=lambda kv: (topic_key(kv[0][0]), doc_key(kv[0][1])))
    ]
    with open(path, "x") as f:  # 'x' = fail rather than overwrite
        f.writelines(lines)
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output root (default results/rearranged); used by the smoke test to rebuild elsewhere")
    args = ap.parse_args()
    OUT = args.out.resolve()
    OUT_RUNS, OUT_PER_RUN = OUT / "runs_task1", OUT / "judge_qrels_per_run"
    OUT_POOLED, OUT_TRACK_V = OUT / "judge_qrels_pooled", OUT / "track_v"
    OUT_METRICS = OUT / "metrics_judge_pool20"

    official = read_qrels(OFFICIAL_QRELS)
    topics78 = set(official)
    assert len(topics78) == 78, f"expected 78 official topics, got {len(topics78)}"

    run_srcs = [SRC_RUNS_DIR / f"{c}.tsv" for _, _, c in CONFIGS]
    tv_srcs = sorted(p for p in SRC_TRACK_V_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    planned = (
        [OUT_RUNS / p.name for p in run_srcs]
        + [OUT_PER_RUN / f"{c}{PER_RUN_SUFFIX}" for _, _, c in CONFIGS]
        + [OUT_POOLED / POOLED_QREL_NAME, OUT_POOLED / "pool_per_topic.csv", OUT_POOLED / "excluded_unparsed.tsv"]
        + [OUT_TRACK_V / p.name for p in tv_srcs]
        + [OUT / "README.md", OUT / "SHA256SUMS"]
    )
    missing = [p for p in run_srcs if not p.exists()]
    if missing:
        sys.exit(f"missing run files: {missing}")
    clash = [p for p in planned if p.exists()]
    if clash:
        sys.exit(f"refusing to overwrite {len(clash)} existing file(s), e.g. {clash[0]}  (nothing was written)")
    for d in (OUT_RUNS, OUT_PER_RUN, OUT_POOLED, OUT_TRACK_V):
        d.mkdir(parents=True, exist_ok=True)

    # 1 -- run files, unchanged (the qrels decide which topics count)
    for src in run_srcs:
        shutil.copy2(src, OUT_RUNS / src.name)

    # 2 + 4 -- per-run judge qrels, 78 official topics, unparseable labels dropped
    pooled: dict[tuple[str, str], int] = {}
    pool_runs: dict[tuple[str, str], set[str]] = defaultdict(set)
    unparsed: dict[tuple[str, str], set[str]] = defaultdict(set)
    summary = []
    for rid, label, cfg in CONFIGS:
        rows = [r for r in load_audit(cfg) if r["topic_id"] in topics78]
        assert all(r["rank"] <= POOL_DEPTH for r in rows)
        pairs = {}
        for r in rows:
            key = (r["topic_id"], r["answer_id"])
            if r["llm_label"] is None:
                unparsed[key].add(cfg)
                continue
            pairs[key] = r["llm_label"]
            if key in pooled and pooled[key] != r["llm_label"]:
                sys.exit(f"conflicting judge labels for {key}: {pooled[key]} vs {r['llm_label']} ({cfg})")
            pooled[key] = r["llm_label"]
            pool_runs[key].add(cfg)
        n = write_qrel(OUT_PER_RUN / f"{cfg}{PER_RUN_SUFFIX}", pairs)
        summary.append((rid, cfg, n, sum(1 for r in rows if r["llm_label"] is None)))

    # a pair unparsed in one run but labelled in another would be judged in the pool
    still_unparsed = {k: v for k, v in unparsed.items() if k not in pooled}

    # 3 -- the common qrel
    n_pooled = write_qrel(OUT_POOLED / POOLED_QREL_NAME, pooled)

    with open(OUT_POOLED / "pool_per_topic.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_id", "n_judged", "n_grade0", "n_grade1", "n_grade2", "n_grade3",
                    "n_unparsed_excluded", "mean_runs_per_answer"])
        by_topic: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (t, d), g in pooled.items():
            by_topic[t].append((d, g))
        for t in sorted(topics78, key=topic_key):
            gs = [g for _, g in by_topic[t]]
            nr = [len(pool_runs[(t, d)]) for d, _ in by_topic[t]]
            w.writerow([t, len(gs)] + [gs.count(k) for k in range(4)]
                       + [sum(1 for k in still_unparsed if k[0] == t), f"{sum(nr) / len(nr):.3f}"])

    with open(OUT_POOLED / "excluded_unparsed.tsv", "x") as f:
        f.write("topic_id\tanswer_id\tconfigs_with_this_answer_in_top20\n")
        for (t, d), cfgs in sorted(still_unparsed.items(), key=lambda kv: (topic_key(kv[0][0]), doc_key(kv[0][1]))):
            f.write(f"{t}\t{d}\t{','.join(sorted(cfgs))}\n")

    # 5 -- Track V, copied (originals stay: rescore_run.py and the manifest point at them)
    for src in tv_srcs:
        shutil.copy2(src, OUT_TRACK_V / src.name)

    # README
    n_topics_pool = len({t for t, _ in pooled})
    readme = f"""# results/rearranged — RQ1 evaluation inputs

Built {date.today().isoformat()} by `experiments/rearrange/build_rearranged.py` from the sources named below.
Sources were only read; nothing outside this folder was changed. `SHA256SUMS` covers every file the build wrote.

## Decisions this folder encodes (2026-09-18)

- **Topics:** Task 1 is evaluated on the **78 official ARQMath-3 topics**
  (`data/raw/qrels/qrel_task1_2022_official.tsv`), not the 87 of `qrel_task1_2022_all.tsv`
  and not the 100 in the run files. Task 2 is unchanged (its own official qrels, 76 topics).
- **Configurations:** the 15 reported configurations (manifest §4e). The three heuristic and three
  filter-off runs are retired and not included.
- **One common qrel.** Every configuration is scored against the same judgments: the union of the 15
  runs' top-{POOL_DEPTH} answers (a depth-{POOL_DEPTH} pool), every answer graded by the LLM judge
  (`gemini-3.5-flash-lite`, prompt `arqmath_relevance_v1`). No ARQMath human grade is in it, and no
  answer is in it that none of the 15 runs placed in its top {POOL_DEPTH}. The ideal ranking behind
  nDCG′ is therefore identical for all 15 configurations.
- **Unparseable judge outputs** ({len(still_unparsed)} pairs over the 78 topics, listed in
  `judge_qrels_pooled/excluded_unparsed.tsv`) are left out of every qrel, so prime scoring treats them
  as unjudged.

## Layout

| Path | What | Source |
|---|---|---|
| `runs_task1/<config>.tsv` | The 15 run files, byte-identical copies (1000 deep, all 100 topics — the qrels select the 78) | `data/runs/task1/` |
| `judge_qrels_per_run/<config>{PER_RUN_SUFFIX}` | That run's top-{POOL_DEPTH} judge labels, TREC qrels (`topic 0 answer_id grade`), 78 topics | `results/per_topic/<config>_answers_audit.csv` (column `llm_label`) |
| `judge_qrels_pooled/{POOLED_QREL_NAME}` | **The qrel to score with.** Union of the 15 per-run files, deduplicated: {n_pooled} pairs, {n_topics_pool} topics | the 15 files above |
| `judge_qrels_pooled/pool_per_topic.csv` | Pool size and grade counts per topic | derived |
| `judge_qrels_pooled/excluded_unparsed.tsv` | Pairs with no usable judge label | derived |
| `metrics_judge_pool20/` | nDCG′ and P′ @5/10/20 per configuration, aggregate and per topic (written by `score_judge_pool20.py`, with its own `SHA256SUMS`) | runs + pooled qrel |
| `track_v/` | Judge-validation files (R4), copied; the originals remain in `results/track_v/` | `results/track_v/` |

Grades are 0–3 (Not / Low / Medium / High). nDCG′ uses the grade as gain (trec_eval's linear
convention); P′ binarises at grade ≥ 2 (`relevance_level = 2`).

## Scoring

```python
import pytrec_eval
# qrel: {{topic: {{answer_id: grade}}}} from judge_qrels_pooled/{POOLED_QREL_NAME}
# run:  {{topic: {{answer_id: score}}}}, cut to each topic's top {POOL_DEPTH}, then unjudged answers removed (prime)
ev = pytrec_eval.RelevanceEvaluator(qrel, {{"ndcg_cut.5,10,20", "P.5,10,20"}}, relevance_level=2)
```

`experiments/rearrange/score_judge_pool20.py` does exactly this; `smoke_test_rearranged.py` checks it.

## Per-run qrel sizes (78 topics)

| Registry | Config | Judged pairs | Unparsed dropped |
|---|---|---|---|
""" + "".join(f"| {rid} | `{cfg}` | {n} | {u} |\n" for rid, cfg, n, u in summary)
    with open(OUT / "README.md", "x") as f:
        f.write(readme)

    # checksums of everything the build wrote
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                   and OUT_METRICS not in p.parents)
    with open(OUT / "SHA256SUMS", "x") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(OUT)}\n")

    print(f"wrote {len(files) + 1} files under {OUT}")
    print(f"pooled qrel: {n_pooled} pairs over {n_topics_pool} topics; unparsed excluded: {len(still_unparsed)}")
    for rid, cfg, n, u in summary:
        print(f"  {rid:6s} {cfg:55s} {n:5d} judged  {u} unparsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
