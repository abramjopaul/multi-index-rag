#!/usr/bin/env python3
"""R4.4 / thesis §4.4: judge calibration and the fuzziness values (2026-09-19).

How close may two systems be before the judge can no longer be trusted to order them?
This script answers that on the official ARQMath-3 participant runs, and draws the figure
(thesis Figure 4.2) that shows the answer.

Protocol
  - runs:   the 38 official ARQMath-3 Task 1 participant runs (data/runs/arqmath-2022-runs/)
  - labels: results/rearranged/track_v/track_v_paired_labels_2022.csv -- every 2022 judged
            pair with its human and its judge grade. Pairs whose judge output could not be
            parsed are dropped from BOTH label sets, so the two qrels cover exactly the same
            pairs and the labels are the only thing that differs between the two scorings.
  - topics: the 78 official topics (qrel_task1_2022_official.tsv) by default -- the topic set
            every Chapter 5 score is averaged over. --topics all uses the 87 judged topics.
  - scoring: each run cut to its top 20 (score-descending, stable on file order -- common.ranking,
            the depth the judge channel is scored at), unjudged answers removed (prime),
            pytrec_eval P'@5/10/20 (grade >= 2) and nDCG'@5/10/20 (grade as gain); macro
            average over the topics, a topic with nothing judged in the run's top 20 scoring 0.
            Each run is scored twice: under human labels and under judge labels.
  - fit:    least squares, judge = a * human + b, with R^2, per measure.
  - error rate at fuzziness value f (Buckley & Voorhees 2000): over the pairs of runs whose
            JUDGE-label scores differ by more than f, the share that the two label sets order
            differently (human difference and judge difference of opposite sign). The threshold
            is applied to the judge-label difference because that is the only difference a
            later chapter can see.
  - adopted fuzziness value: the smallest f in 0.00, 0.01, ..., 0.10 whose error rate is < 5%.
            The thesis applies one threshold, 0.02, to every measure and cutoff: on the 78 official
            topics it is the largest per-cell adopted value, so every cell is below 5% there. The
            figure annotates each panel with its error rate at 0.02.

Writes results/rearranged/judge_calibration_r4/ (never overwrites; --dry-run writes nothing):
  scores_per_run.csv          38 runs x 6 measures, human and judge score
  calibration.csv             per measure: a, b, R^2, error rate at every f, adopted f
  judge_calibration_2022.pdf  the figure, vector (for LaTeX)
  judge_calibration_2022.png  the same, 200 dpi
  README.md, SHA256SUMS

Usage:
    python experiments/rearrange/judge_calibration_r4.py --dry-run          # print only
    python experiments/rearrange/judge_calibration_r4.py                    # write the folder
    python experiments/rearrange/judge_calibration_r4.py --topics all --dry-run
    python experiments/rearrange/judge_calibration_r4.py --redraw           # restyle the figure only

--redraw is the one mode that touches an existing folder: it overwrites the two figure files and
SHA256SUMS, from the folder's own CSVs, and recomputes no number.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    OFFICIAL_QRELS, OUT, OUT_TRACK_V, POOL_DEPTH, RELEVANCE_LEVEL, ROOT, ranking, read_qrels,
    read_run, write_checksums,
)

RUNS_DIR = ROOT / "data" / "runs" / "arqmath-2022-runs"
PAIRED = OUT_TRACK_V / "track_v_paired_labels_2022.csv"
OUT_DIR = OUT / "judge_calibration_r4"

CUTOFFS = (5, 10, 20)
# (key, display, pytrec_eval key) -- P' first, then nDCG', the order of the figure's rows
MEASURES = [(f"P'@{k}", f"P′@{k}", f"P_{k}") for k in CUTOFFS] + \
           [(f"nDCG'@{k}", f"nDCG′@{k}", f"ndcg_cut_{k}") for k in CUTOFFS]
FUZZ = [round(0.01 * i, 2) for i in range(11)]  # 0.00 .. 0.10
TARGET = 0.05                                     # error rate must fall below this
REPORTED_FUZZINESS = 0.02                         # the one threshold the thesis applies (manifest §1c-i);
                                                  # the figure annotates every panel at this value
EPS = 1e-12


def load_labels(topics: set[str]) -> tuple[dict, dict, int]:
    """Human and judge qrels over the same pairs; returns (human, judge, n_dropped_unparsed)."""
    human: dict[str, dict[str, int]] = {}
    judge: dict[str, dict[str, int]] = {}
    dropped = 0
    with open(PAIRED, newline="") as f:
        for r in csv.DictReader(f):
            t = r["topic_id"]
            if t not in topics:
                continue
            if r["judge_label"] in ("", "None", "null"):
                dropped += 1
                continue
            human.setdefault(t, {})[r["answer_id"]] = int(r["human_label"])
            judge.setdefault(t, {})[r["answer_id"]] = int(r["judge_label"])
    return human, judge, dropped


def score(run: dict, qrel: dict, evaluator) -> dict[str, float]:
    """Top-20 -> prime -> macro average over every qrel topic (0 where nothing is judged)."""
    ranked = ranking(run)
    scored = {}
    for t in qrel:
        top = {d: run[t][d] for d in ranked.get(t, [])[:POOL_DEPTH] if d in qrel[t]}
        if top:
            scored[t] = top
    res = evaluator.evaluate(scored) if scored else {}
    return {key: sum(res[t][pk] if t in res else 0.0 for t in qrel) / len(qrel)
            for key, _, pk in MEASURES}


def fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    slope = sxy / sxx
    return slope, my - slope * mx, sxy * sxy / (sxx * syy)


def error_rates(h: list[float], j: list[float]) -> dict[float, tuple[int, int]]:
    """f -> (misordered pairs, pairs whose judge-score difference exceeds f)."""
    out = {}
    pairs = list(itertools.combinations(range(len(h)), 2))
    for f in FUZZ:
        kept = [(p, q) for p, q in pairs if abs(j[p] - j[q]) > f + EPS]
        bad = sum(1 for p, q in kept if (h[p] - h[q]) * (j[p] - j[q]) < 0)
        out[f] = (bad, len(kept))
    return out


def calibrate(topic_set: str) -> tuple[list[dict], list[dict], dict]:
    qrel_topics = set(read_qrels(OFFICIAL_QRELS)) if topic_set == "official" else None
    if qrel_topics is None:
        with open(PAIRED, newline="") as f:
            qrel_topics = {r["topic_id"] for r in csv.DictReader(f)}
    human, judge, dropped = load_labels(qrel_topics)
    assert set(human) == set(judge), "human and judge qrels must cover the same topics"
    ev_h = pytrec_eval.RelevanceEvaluator(human, {"P.5,10,20", "ndcg_cut.5,10,20"},
                                          relevance_level=RELEVANCE_LEVEL)
    ev_j = pytrec_eval.RelevanceEvaluator(judge, {"P.5,10,20", "ndcg_cut.5,10,20"},
                                          relevance_level=RELEVANCE_LEVEL)
    run_files = sorted(p for p in RUNS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    assert len(run_files) == 38, f"expected the 38 official ARQMath-3 runs, found {len(run_files)}"

    per_run = []
    for p in run_files:
        run = read_run(p)
        per_run.append({"run": p.stem, "human": score(run, human, ev_h), "judge": score(run, judge, ev_j)})

    calib = []
    for key, disp, _ in MEASURES:
        h = [r["human"][key] for r in per_run]
        j = [r["judge"][key] for r in per_run]
        a, b, r2 = fit(h, j)
        er = error_rates(h, j)
        adopted = next((f for f in FUZZ if er[f][1] and er[f][0] / er[f][1] < TARGET), None)
        calib.append({"measure": key, "display": disp, "slope_a": a, "intercept_b": b, "r2": r2,
                      "n_runs": len(h), "rates": er, "adopted": adopted})
    meta = {"topic_set": topic_set, "n_topics": len(human), "n_pairs": sum(len(v) for v in human.values()),
            "n_dropped_unparsed": dropped, "n_runs": len(per_run)}
    return per_run, calib, meta


def print_table(calib: list[dict], meta: dict) -> None:
    print(f"topics={meta['topic_set']} ({meta['n_topics']}), pairs={meta['n_pairs']}, "
          f"unparsed dropped={meta['n_dropped_unparsed']}, runs={meta['n_runs']}")
    head = "".join(f"{f:>7.2f}" for f in FUZZ[:6])
    print(f"{'measure':11s}{'a':>6s}{'b':>7s}{'R2':>7s}  {head}   adopted")
    for c in calib:
        rates = "".join(f"{100 * c['rates'][f][0] / c['rates'][f][1]:6.1f}%" for f in FUZZ[:6])
        m0 = c["rates"][0.0]
        print(f"{c['measure']:11s}{c['slope_a']:6.2f}{c['intercept_b']:+7.2f}{c['r2']:7.3f}  {rates}"
              f"   {c['adopted']}  (misordered at 0: {m0[0]}/{m0[1]})")


def draw(per_run: list[dict], calib: list[dict], meta: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, ink2, grid = "#1f1f1f", "#555555", "#e6e6e6"
    point, identity = "#3d6fa8", "#8a8a8a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b0b0b0",
                         "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2})

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.2))
    for row, lim_keys in enumerate((calib[:3], calib[3:])):
        vmax = max(max(r["human"][c["measure"]], r["judge"][c["measure"]]) for c in lim_keys for r in per_run)
        lim = min(1.0, (int(vmax * 10) + 1) / 10)
        for col, c in enumerate(lim_keys):
            ax = axes[row][col]
            key = c["measure"]
            h = [r["human"][key] for r in per_run]
            j = [r["judge"][key] for r in per_run]
            ax.plot([0, lim], [0, lim], ls=(0, (4, 3)), lw=1.0, color=identity, zorder=1)
            ax.plot([0, lim], [c["intercept_b"], c["slope_a"] * lim + c["intercept_b"]],
                    lw=1.6, color=ink, zorder=2)
            ax.scatter(h, j, s=34, color=point, edgecolor="white", linewidth=0.8, zorder=3)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.grid(color=grid, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(c["display"], color=ink, fontsize=10.5, pad=6)
            if row == 1:
                ax.set_xlabel("score under human labels")
            if col == 0:
                ax.set_ylabel("score under judge labels")
            b = c["intercept_b"]
            m0, n0 = c["rates"][0.0]
            f = REPORTED_FUZZINESS
            mf, nf = c["rates"][f]
            eq = f"fit: {c['slope_a']:.2f} \u00d7 human"
            if round(abs(b), 2) >= 0.01:
                sign = "\u2212" if b < 0 else "+"
                eq += f" {sign} {abs(b):.2f}"
            ax.text(0.97, 0.045, f"{eq}\nmisordered: {m0}/{n0} ({100 * m0 / n0:.1f}%)",
                    transform=ax.transAxes, ha="right", va="bottom", color=ink2, fontsize=8,
                    linespacing=1.45)
            ax.text(0.97, 0.045 + 0.125, f"fuzziness {f:.2f}: {100 * mf / nf:.1f}%",
                    transform=ax.transAxes, ha="right", va="bottom", color=ink, fontsize=8,
                    fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", markersize=6.5, markerfacecolor=point,
                      markeredgecolor="white", label="one participant run"),
               Line2D([], [], color=ink, lw=1.6, label="least-squares fit"),
               Line2D([], [], color=identity, lw=1.0, ls=(0, (4, 3)), label="equal score under both label sets")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def write(per_run: list[dict], calib: list[dict], meta: dict, alt: tuple | None) -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"refusing to overwrite existing files in {OUT_DIR} (nothing was written)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "scores_per_run.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "measure", "human", "judge"])
        for r in per_run:
            for key, _, _ in MEASURES:
                w.writerow([r["run"], key, repr(r["human"][key]), repr(r["judge"][key])])
    with open(OUT_DIR / "calibration.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic_set", "measure", "slope_a", "intercept_b", "r2", "n_runs"]
                   + [f"misordered_at_{x:.2f}" for x in FUZZ] + [f"pairs_at_{x:.2f}" for x in FUZZ]
                   + ["adopted_fuzziness", "error_rate_at_adopted"])
        for m, cal in ([(meta, calib)] + ([alt] if alt else [])):
            for c in cal:
                f_ = c["adopted"]
                w.writerow([m["topic_set"], c["measure"], repr(c["slope_a"]), repr(c["intercept_b"]),
                            repr(c["r2"]), c["n_runs"]] + [c["rates"][x][0] for x in FUZZ]
                           + [c["rates"][x][1] for x in FUZZ]
                           + [f_, repr(c["rates"][f_][0] / c["rates"][f_][1])])
    draw(per_run, calib, meta, OUT_DIR / "judge_calibration_2022")
    rows = "\n".join(f"| {c['display']} | {c['slope_a']:.2f} | {c['intercept_b']:+.2f} | {c['r2']:.3f} | "
                     f"{c['rates'][0.0][0]}/{c['rates'][0.0][1]} | **{c['adopted']:.2f}** |" for c in calib)
    alt_line = ""
    if alt:
        am, ac = alt
        alt_line = ("\nThe same calibration on all " + str(am["n_topics"]) + " judged topics gives adopted values "
                    + ", ".join(f"{c['display']} {c['adopted']:.2f}" for c in ac) + " (rows `topic_set = all` in "
                    "`calibration.csv`; no figure).\n")
    with open(OUT_DIR / "README.md", "x") as f:
        f.write(f"""# judge_calibration_r4 — judge calibration and fuzziness values (R4.4, thesis §4.4, Figure 4.2)

Written by `experiments/rearrange/judge_calibration_r4.py` (protocol in its docstring). The 38 official
ARQMath-3 Task 1 participant runs (`data/runs/arqmath-2022-runs/`), each cut to its top 20, scored twice with
prime measures over the {meta['n_topics']} {'official' if meta['topic_set'] == 'official' else 'judged'} topics:
under the assessors' labels and under the judge's, from `../track_v/track_v_paired_labels_2022.csv`
({meta['n_pairs']} pairs; {meta['n_dropped_unparsed']} with an unparseable judge output dropped from both label
sets, so both cover the same pairs).

Error rate at fuzziness value f: among run pairs whose judge-label scores differ by more than f, the share the
two label sets order differently (Buckley & Voorhees 2000). Adopted fuzziness value: smallest f in 0.00–0.10
with error rate below 5%. A later difference between two systems in the judge channel counts only if it
exceeds the adopted value for its measure and cutoff.

| Measure | a | b | R² | misordered at 0 | adopted fuzziness |
|---|---|---|---|---|---|
{rows}
{alt_line}
Supersedes, as the source of Figure 4.2, `../track_v/track_v_calibration_2022_fullpool.json`, which has no
generating script in the repository and does not reproduce under these scoring conventions.
""")
    write_checksums(OUT_DIR)


def redraw() -> None:
    """Re-render the figure from OUT_DIR's own CSVs and refresh SHA256SUMS. Recomputes nothing."""
    per = {}
    with open(OUT_DIR / "scores_per_run.csv", newline="") as f:
        for r in csv.DictReader(f):
            e = per.setdefault(r["run"], {"run": r["run"], "human": {}, "judge": {}})
            e["human"][r["measure"]] = float(r["human"])
            e["judge"][r["measure"]] = float(r["judge"])
    disp = {k: d for k, d, _ in MEASURES}
    calib = []
    with open(OUT_DIR / "calibration.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["topic_set"] != "official":
                continue
            rates = {x: (int(r[f"misordered_at_{x:.2f}"]), int(r[f"pairs_at_{x:.2f}"])) for x in FUZZ}
            calib.append({"measure": r["measure"], "display": disp[r["measure"]],
                          "slope_a": float(r["slope_a"]), "intercept_b": float(r["intercept_b"]),
                          "r2": float(r["r2"]), "n_runs": int(r["n_runs"]), "rates": rates,
                          "adopted": float(r["adopted_fuzziness"])})
    calib.sort(key=lambda c: [k for k, _, _ in MEASURES].index(c["measure"]))
    draw(list(per.values()), calib, {}, OUT_DIR / "judge_calibration_2022")
    from common import sha256
    files = sorted(p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with open(OUT_DIR / "SHA256SUMS", "w") as f:
        for p in files:
            f.write(f"{sha256(p)}  {p.relative_to(OUT_DIR)}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topics", choices=("official", "all"), default="official")
    ap.add_argument("--dry-run", action="store_true", help="print the calibration, write nothing")
    ap.add_argument("--redraw", action="store_true",
                    help="re-render the figure from the existing folder's CSVs (recomputes nothing)")
    args = ap.parse_args()
    if args.redraw:
        redraw()
        print(f"redrew the figure in {OUT_DIR}")
        return 0
    per_run, calib, meta = calibrate(args.topics)
    print_table(calib, meta)
    if args.dry_run:
        return 0
    other = "all" if args.topics == "official" else "official"
    _, calib_alt, meta_alt = calibrate(other)
    print_table(calib_alt, meta_alt)
    write(per_run, calib, meta, (meta_alt, calib_alt))
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
