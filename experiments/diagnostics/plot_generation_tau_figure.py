"""Figure for thesis section 6.3.3 -- the two generation measures against retrieval quality.

Reads the per-answer RAGAS scores of the v3 generation runs
(results/rearranged/track_c/per_sample_score/1.5b/, manifest §14 R9) and the judge-pool
retrieval metrics (results/rearranged/metrics_judge_pool20/metrics_aggregate.csv, R1-R3),
and writes thesis/report-tex/images/gen_tau_retrieval_quality.pdf.

Two panels sharing the x axis, one per measure, each carrying its own Kendall's tau over all
105 pairs of the fifteen reported configurations. Two panels rather than two y axes (manifest
§9b): the two measures span different ranges, so on one pair of axes the clouds overlap and
invite a comparison of heights that are measured against different scales.

Tau is (C - D)/(C + D) as §3.6 defines it, on the means before rounding. Descriptive: no
p-value, and no pair is set aside under the 0.02 rule (manifest §1c).

Palette matches the chapter's existing figures (ret_h11_add_one.pdf, res_containment_opt.pdf).
Usage:  python experiments/diagnostics/plot_generation_tau_figure.py
"""
import csv, itertools, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCORES = ROOT / "results/rearranged/track_c/per_sample_score/1.5b"
METRICS = ROOT / "results/rearranged/metrics_judge_pool20/metrics_aggregate.csv"
OUT = ROOT / "thesis/report-tex/images"

INK, MID, BAND, GRID, TEXT = "#264e8a", "#8096bc", "#eaf1fa", "#e6e7e9", "#3d4650"

# Both panels share one vertical scale (0.02-0.26, ticks every 0.05). The two measures span
# different ranges -- faithfulness 0.049-0.217, FC-P 0.148-0.241 -- so panel-fitted limits made
# the same vertical distance mean different things in the two panels. With one scale a point's
# height means the same thing wherever it sits, which is the whole reason for splitting the
# panels rather than using two y axes.
YLIM = (0.02, 0.26)
YTICKS = [0.05, 0.10, 0.15, 0.20, 0.25]

# run-directory stem (timestamp removed) -> registry id of the retrieval configuration
FLOOR_RUN = "no_rag_k0"          # the no-retrieval floor, for the reference line on the FC-P panel
ANOMALY = "R2.7"                 # SLT-TYPE: named in the figure because §6.3.2-§6.4 argue from it

RUNS = {
    "bm25_k5": "R1.1", "dense_mpnet_k5": "R1.2", "formula_v1_slt_k5": "R2.1",
    "formula_v1_opt_k5": "R2.4", "formula_v1_slt_type_k5": "R2.7",
    "fusion_bm25_dense_k5": "R3.1", "fuse_dense_slt_k5": "R3.2",
    "fuse_dense_opt_k5": "R3.3", "fuse_dense_slt_type_k5": "R3.4",
    "formula_block_k5": "R3.5", "blocked_k5": "R3.6", "flat_5way_k5": "R3.7",
    "blocked_drop_bm25_k5": "R3.8", "flat_drop_bm25_k5": "R3.9",
    "flat_drop_slt_type_k5": "R3.10",
}


def retrieval_quality():
    return {r["registry_id"]: float(r["ndcg'@10"]) for r in csv.DictReader(open(METRICS))}


def floor_fcp():
    for d in sorted(SCORES.iterdir()):
        if d.is_dir() and "_".join(d.name.split("_")[:-1]) == FLOOR_RUN:
            rows = [json.loads(line) for line in open(d / "scores_per_sample.jsonl")]
            vals = [r["factual_correctness_precision"] for r in rows
                    if r.get("factual_correctness_precision") is not None]
            return sum(vals) / len(vals)
    raise SystemExit("floor run not found")


def generation_means():
    out = {}
    for d in sorted(SCORES.iterdir()):
        if not d.is_dir():
            continue
        stem = "_".join(d.name.split("_")[:-1])
        if stem not in RUNS:
            continue                      # smoke runs, oracle, filter-off variants: not reported
        rows = [json.loads(line) for line in open(d / "scores_per_sample.jsonl")]

        def mean(field):                  # undefined scores are dropped, not counted as zero
            vals = [r[field] for r in rows if r.get(field) is not None]
            return sum(vals) / len(vals)

        out[RUNS[stem]] = (mean("faithfulness"), mean("factual_correctness_precision"))
    return out


def tau(points):
    """(C - D) / (C + D) over every pair, as Section 3.6 defines it."""
    conc = disc = 0
    for (x1, y1), (x2, y2) in itertools.combinations(points, 2):
        s = (x1 - x2) * (y1 - y2)
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    return (conc - disc) / (conc + disc), conc, disc


def main():
    ret, gen = retrieval_quality(), generation_means()
    floor = floor_fcp()
    ids = sorted(gen, key=lambda i: ret[i])
    nd = [ret[i] for i in ids]
    fa = [gen[i][0] for i in ids]
    fp = [gen[i][1] for i in ids]
    t_fa, c_fa, d_fa = tau(list(zip(nd, fa)))
    t_fp, c_fp, d_fp = tau(list(zip(nd, fp)))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8, "axes.edgecolor": TEXT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    })
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.10, hspace=0.12)

    a1.plot(nd, fa, "o", ms=6.5, color=INK, zorder=4)
    a1.set_ylabel("Faithfulness")
    a1.text(0.575, 0.034, rf"$\tau = {t_fa:.3f}$", ha="right", va="bottom",
            fontsize=10, color=INK)

    a2.plot(nd, fp, "o", ms=6.5, color=INK, zorder=4)
    a2.set_ylabel("FC-P")
    a2.text(0.575, 0.034, rf"$\tau = {t_fp:.3f}$", ha="right", va="bottom",
            fontsize=10, color=INK)
    a2.set_xlabel("nDCG$'$@10")

    # The floor is a level on FC-P only: faithfulness needs a context and the floor has none.
    a2.axhline(floor, color=TEXT, lw=1.0, ls=(0, (2, 2)), zorder=2)
    a2.text(0.578, floor - 0.010, f"no context (the floor), {floor:.3f}",
            ha="right", va="top", fontsize=8.5, color=TEXT)

    # SLT-TYPE is named in both panels: §6.3.2, §6.3.3 and §6.4 all argue from this one point.
    k = ids.index(ANOMALY)
    a1.annotate("SLT-TYPE", (nd[k], fa[k]), textcoords="offset points", xytext=(10, 0),
                ha="left", va="center", fontsize=8.5, color=TEXT)
    a2.annotate("SLT-TYPE", (nd[k], fp[k]), textcoords="offset points", xytext=(10, 4),
                ha="left", va="bottom", fontsize=8.5, color=TEXT)

    for ax in (a1, a2):
        ax.set_xlim(0.07, 0.59)
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = OUT / "gen_tau_retrieval_quality.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"{out}  ({len(ids)} configurations)")
    print(f"  faithfulness tau={t_fa:.3f}  concordant {c_fa}  discordant {d_fa}")
    print(f"  FC-P         tau={t_fp:.3f}  concordant {c_fp}  discordant {d_fp}")
    print(f"  floor FC-P={floor:.3f}   above it: {sum(1 for v in fp if v > floor)} of {len(fp)}")
    print(f"  {ANOMALY} (SLT-TYPE): ndcg={nd[k]:.3f} faith={fa[k]:.3f} fcp={fp[k]:.3f}")


if __name__ == "__main__":
    main()
