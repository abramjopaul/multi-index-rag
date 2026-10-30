#!/usr/bin/env python3
"""File the RQ1 extras into results/rearranged/ (2026-09-18): the official qrels, the R7
outputs, and the root README/SHA256SUMS entries for the new folders.

The R6, human_only and Task 2 folders are written by their own scripts
(calibrate_r6.py, score_human_only.py, task2_r5.py). R7 is run by run_r7_resumable.py into
a scratch directory first, because it runs in steps; this script copies the finished
outputs in. Never overwrites a data file; only the root README.md (one appended section,
once) and its line in the root SHA256SUMS are updated.

Usage:
    python experiments/rearrange/finalize_rq1_extras.py --r7-78 /tmp/r7_78 --r7-87 /tmp/r7_87
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OFFICIAL_QRELS, OFFICIAL_QRELS_T2, OUT, sha256, write_checksums  # noqa: E402

R7_FILES = ["overlap_summary.csv", "overlap_by_topic.csv", "triviality_report.csv", "worked_examples.md",
            "summary.json", "schema_report.md"]
MARKER = "## Added 2026-09-18 — the rest of RQ1"


def copy_new(src: Path, dst: Path) -> None:
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r7-78", type=Path, required=True, help="run_r7_resumable.py output, 78 official topics")
    ap.add_argument("--r7-87", type=Path, required=True, help="run_r7_resumable.py output, 87 topics (validation)")
    args = ap.parse_args()
    for d in (args.r7_78, args.r7_87):
        miss = [f for f in R7_FILES if not (d / f).exists()]
        if miss:
            sys.exit(f"{d} is incomplete: {miss}")

    # qrels_official/
    q = OUT / "qrels_official"
    copy_new(OFFICIAL_QRELS, q / OFFICIAL_QRELS.name)
    copy_new(OFFICIAL_QRELS_T2, q / OFFICIAL_QRELS_T2.name)
    (q / "README.md").open("x").write(
        "# qrels_official — the ARQMath-3 (2022) assessors' judgments used here\n\n"
        "Byte-identical copies of `data/raw/qrels/`.\n\n"
        "| File | Task | Topics | Used by |\n|---|---|---|---|\n"
        f"| `{OFFICIAL_QRELS.name}` | 1 (answers) | 78 | `../calibration_r6/`, `../metrics_human_only/`, "
        "`../formula_overlap_r7/`, and the topic set of `../judge_qrels_pooled/` |\n"
        f"| `{OFFICIAL_QRELS_T2.name}` | 2 (formulas, visual ids) | 76 | `../task2_r5/` |\n\n"
        "The Task 1 judge channel (§5.4) does **not** read these grades; it reads "
        "`../judge_qrels_pooled/`. On the 78 official topics `qrel_task1_2022_all.tsv` carries exactly "
        "these labels; its extra 9 topics come from `qrel_task1_2022_additional.tsv`.\n")
    write_checksums(q)

    # formula_overlap_r7/
    r = OUT / "formula_overlap_r7"
    for f in R7_FILES:
        copy_new(args.r7_78 / f, r / f)
        copy_new(args.r7_87 / f, r / "validation_87_topics" / f)
    with open(r / "overlap_summary.csv", newline="") as f:
        s = {(row["query_strategy"], row["representation"]): row for row in csv.DictReader(f)}
    slt = s[("fanout", "slt")]
    m_all = float(slt["mean_relevant"]) - float(slt["mean_non_relevant"])
    m_max = float(slt["max_relevant"]) - float(slt["max_non_relevant"])
    rows = "".join(
        f"| {rep} | {float(s[('fanout', rep)]['mean_relevant']):.3f} | {float(s[('fanout', rep)]['mean_non_relevant']):.3f} | "
        f"{s[('fanout', rep)]['n_topics_mean_relevant_gt_non_relevant']}/{s[('fanout', rep)]['n_topics']} | "
        f"{float(s[('fanout', rep)]['max_relevant']):.3f} | {float(s[('fanout', rep)]['max_non_relevant']):.3f} | "
        f"{s[('fanout', rep)]['n_topics_max_relevant_gt_non_relevant']}/{s[('fanout', rep)]['n_topics']} |\n"
        for rep in ("slt", "opt", "slt_type"))
    (r / "README.md").open("x").write(f"""# formula_overlap_r7 — where the formula signal lives (R7, thesis §5.4.3, H1.2)

Produced by the unchanged `experiments/diagnostics/formula_overlap.py`, seed 42, on the **78 official
Task 1 topics** (`--topic-ids`; on those topics `qrel_task1_2022_all.tsv` equals the official qrels, so
the relevance labels are the assessors' official ones). It was run through
`experiments/rearrange/run_r7_resumable.py`, which only caches the script's file scans and memoises
its pure tuple extractor so the run can proceed in steps; every output file is written by the
script's own `main()`.

**Validation (`validation_87_topics/`).** The same driver on the original 87 topics reproduces
`results/formula_overlap/overlap_results.csv` on every mean, max and top-5 value exactly. Four
win counts differ by one topic (OPT rows, relevant vs non-relevant): on those topics the two values are
mathematically equal and differ only in floating-point rounding (e.g. 0.13333333333333336 vs
0.13333333333333333), so the strict `>` is decided by summation order. The 78-topic run has no such
near-ties.

## Headline — fanout, relevance = High + Medium

| Representation | mean, relevant | mean, non-relevant | topics rel > non-rel | max, relevant | max, non-relevant | topics rel > non-rel |
|---|---|---|---|---|---|---|
{rows}
H1.2 on SLT: (a) margin over all answers {m_all:+.3f}, relevant ahead on {slt['n_topics_mean_relevant_gt_non_relevant']}/78;
(b) at the best-matching answer the margin is {m_max:+.3f} and relevant is ahead on {slt['n_topics_max_relevant_gt_non_relevant']}/78.

Heuristic rows are in `overlap_summary.csv` for the registry only (manifest §4e).
""")
    write_checksums(r)

    # root README section + its checksum line
    readme = OUT / "README.md"
    text = readme.read_text()
    if MARKER not in text:
        text += f"""
{MARKER}

Each folder below has its own `README.md` and `SHA256SUMS`.

| Folder | Thesis | What | Written by |
|---|---|---|---|
| `qrels_official/` | — | Official 2022 qrels, Task 1 (78 topics) and Task 2 (76), copies | `finalize_rq1_extras.py` |
| `calibration_r6/` | §5.3.1, P1 | TF-IDF (Terrier) baseline scored with this pipeline against the published Table 3 figures | `calibrate_r6.py` |
| `metrics_human_only/` | §5.3.2 | The 15 runs under the ARQMath labels alone: standard and prime nDCG/P @5/10/20, hole@5/10/20, 78 topics | `score_human_only.py` |
| `task2_r5/` | §5.4.1, H1.1 (a) | R5.1 and TangentCFT2 (collapsed to visual ids): nDCG′, MAP′, P′@10 with the published figures; hole rates | `task2_r5.py` |
| `formula_overlap_r7/` | §5.4.3, H1.2 | Formula-overlap diagnostic on the 78 official topics, plus the 87-topic validation run | `run_r7_resumable.py` + `finalize_rq1_extras.py` |
"""
        readme.write_text(text)
        sums = OUT / "SHA256SUMS"
        lines = sums.read_text().splitlines()
        lines = [f"{sha256(readme)}  README.md" if ln.endswith("  README.md") else ln for ln in lines]
        sums.write_text("\n".join(lines) + "\n")
    print("filed qrels_official/, formula_overlap_r7/ and updated README.md + SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
