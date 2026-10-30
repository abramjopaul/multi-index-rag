"""Figure for thesis section 5.5 -- containment of a topic's formulas in its judged answers.

Reads results/rearranged/formula_overlap_r7/overlap_by_topic.csv (R7, fanout, relevance = H+M,
78 official topics) and writes thesis/report-tex/images/res_containment_opt.pdf.

Palette matches the chapter's existing figures (ret_h11_add_one.pdf, ret_h12_add_all_three.pdf).
Usage:  python experiments/diagnostics/plot_containment_figure.py [representation]
"""
import csv, collections, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "results/rearranged/formula_overlap_r7/overlap_by_topic.csv"
OUT = ROOT / "thesis/report-tex/images"

INK, MID, BAND, GRID, TEXT = "#264e8a", "#8096bc", "#eaf1fa", "#e6e7e9", "#3d4650"
STATS = ["mean_containment", "top5_mean_containment", "max_containment"]
XT = ["all of a topic's judged answers", "its five best-matching", "its single best-matching"]


def load(rep):
    rows = [r for r in csv.DictReader(open(SRC))
            if r["query_strategy"] == "fanout" and r["relevance_def"] == "high_medium"
            and r["representation"] == rep]
    d = collections.defaultdict(dict)
    for r in rows:
        d[r["topic_id"]][r["group"]] = r
    topics = sorted(d)
    return {g: np.array([np.mean([float(d[t][g][s]) for t in topics]) for s in STATS])
            for g in ("relevant", "non_relevant", "random")}, len(topics)


def main(rep="opt"):
    v, n = load(rep)
    rel, non, rnd = v["relevant"], v["non_relevant"], v["random"]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": TEXT, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))

    ax.axhline(1.0, color=MID, lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax.annotate("a complete match", (-0.36, 1.012), ha="left", va="bottom",
                fontsize=8.5, color=MID)

    ax.fill_between(x, non, rel, color=BAND, zorder=1)
    ax.plot(x, rel, "-o", color=INK, mfc=INK, ms=6, lw=2.0, zorder=4,
            label="answers judged relevant")
    ax.plot(x, non, "--s", color=INK, mfc="white", mew=1.4, ms=6, lw=1.5, zorder=4,
            label="answers judged not relevant")
    ax.plot(x, rnd, ":^", color=MID, mfc="white", mew=1.3, ms=6, lw=1.5, zorder=4,
            label="answers drawn at random")

    for i in range(3):
        ax.annotate(f"{rel[i]:.2f}", (i, rel[i]), textcoords="offset points", xytext=(0, 9),
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.annotate(f"{non[i]:.2f}", (i, non[i]), textcoords="offset points", xytext=(0, -11),
                    ha="center", va="top", fontsize=9, color=INK)
        off = (13, -4) if rnd[i] < 0.12 else (0, -11)
        ax.annotate(f"{rnd[i]:.2f}", (i, rnd[i]), textcoords="offset points", xytext=off,
                    ha="left" if rnd[i] < 0.12 else "center",
                    va="center" if rnd[i] < 0.12 else "top", fontsize=9, color=MID)

    ax.set_xticks(x); ax.set_xticklabels(XT)
    ax.set_xlim(-0.42, 2.42); ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("containment of the\ntopic's formula", linespacing=1.4)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4, borderaxespad=0.6)
    fig.tight_layout()
    out = OUT / f"res_containment_{rep}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"{out}  ({n} topics)  rel={np.round(rel,3)} non={np.round(non,3)} rnd={np.round(rnd,3)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "opt")
