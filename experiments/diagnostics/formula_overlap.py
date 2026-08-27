#!/usr/bin/env python3
"""R7: how often does a Task 1 topic's formulas actually appear in the answers
humans rated relevant?

The formula channel (SLT/OPT/SLT-TYPE trees over FastText+FAISS) scores well on
ARQMath Task 2 (formula retrieval, where structural match IS the relevance
definition) but poorly on Task 1 (answer retrieval). Hypothesis: this is a
property of Task 1's relevance definition, not an implementation defect --
a correct answer often reaches its result by a different symbolic route than
the query, so the query formula rarely appears in structurally matchable form
inside a relevant answer.

Method: for each ARQMath-3 (2022) Task 1 topic, select query formulas exactly
as the real formula retriever does -- via formula_selector.select_fanout (all
non-trivial formulas) and select_heuristic (one, title-preferred), both run
side by side. For each selected topic formula and each candidate answer
formula, compute tuple containment (fraction of the topic formula's tuples
present in the answer formula) on three representations actually used by the
retriever: SLT, OPT, SLT-TYPE (type-erased SLT). No similarity threshold is
applied anywhere -- the real retriever has none either, it's pure top-k/RRF
over embeddings, so plain containment numbers are the more faithful choice.

Compares human-judged-relevant answers against two controls: non-relevant
judged answers (same topic, qrel label=0) and random answers from the whole
collection. Reports three per-topic statistics, not just the mean: mean (the
old headline number), max (does at least one answer in the group contain the
query formula well -- what retrieval actually needs, since ranking only
needs one relevant answer to beat the crowd), and mean-of-top-5 (a less
noisy middle ground). Plain aggregates and simple directional counts only --
no significance tests.

Pure offline computation over precomputed MathML in the v3 TSVs and
answers.jsonl -- no network, no LaTeXML subprocess. Deterministic given --seed.

Usage:
    poetry run python experiments/diagnostics/formula_overlap.py
    poetry run python experiments/diagnostics/formula_overlap.py --topic-ids A.301,A.302 --out-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # experiments/ (logging_config.py)

from logging_config import configure_logging  # noqa: E402

from multirag.config.path_configs import (  # noqa: E402
    ANSWERS_JSONL,
    OPT_REPRESENTATION,
    QREL_TASK1_2022_ALL,
    RESULTS_DIAGNOSTICS_DIR,
    SLT_REPRESENTATION,
    TOPICS_FORMULAS_LATEX,
    TOPICS_FORMULAS_OPT,
    TOPICS_FORMULAS_SLT,
    TOPICS_TASK1_2022_XML,
)
from multirag.eval.wandb_logger import ExperimentLogger  # noqa: E402
from multirag.formula_search.formula_selector import (  # noqa: E402
    is_trivial,
    select_fanout,
    select_heuristic,
)
from multirag.formula_search.tuple_extraction import extract_tuples_from_mathml_direct  # noqa: E402
from multirag.preprocessing.topic_parser import TopicReader  # noqa: E402
from multirag.config.judge_config import WandbConfig  # noqa: E402

configure_logging(level="INFO")
logger = logging.getLogger(__name__)

REPRESENTATIONS = ["slt", "opt", "slt_type"]
QUERY_STRATEGIES = ["fanout", "heuristic"]


# --------------------------------------------------------------------------
# Core matching primitives
# --------------------------------------------------------------------------

def _type_erase_node_tag(tag: str) -> str:
    """Reproduces TupleTokenizer._split_node_tag's Type-mode behaviour: a
    tagged node (contains '!') collapses to its type prefix; a bare node
    (no '!', e.g. an operator symbol) is left unchanged. Verified against
    tuple_tokenizer.py's TupleTokenizationMode.Type."""
    if "!" in tag:
        return tag.split("!", 1)[0] + "!"
    return tag


def type_erase_tuple(tuple_str: str) -> str:
    parts = tuple_str.split("\t")
    if len(parts) != 4:
        return tuple_str
    n1, n2, edge, loc = parts
    return "\t".join([_type_erase_node_tag(n1), _type_erase_node_tag(n2), edge, loc])


def containment(query_tuples: frozenset[str], answer_tuples: frozenset[str]) -> float:
    """Fraction of the query formula's tuples present in the answer formula."""
    if not query_tuples:
        return 0.0
    return len(query_tuples & answer_tuples) / len(query_tuples)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Formula:
    """One formula, with the three tuple representations the real retriever
    actually uses (SLT, OPT, SLT-TYPE) precomputed once."""

    formula_id: str
    latex: str
    slt_tuples: frozenset[str]
    opt_tuples: frozenset[str]
    slt_type_tuples: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.slt_type_tuples = frozenset(type_erase_tuple(t) for t in self.slt_tuples)

    def tuples_for(self, representation: str) -> frozenset[str]:
        if representation == "slt":
            return self.slt_tuples
        if representation == "opt":
            return self.opt_tuples
        if representation == "slt_type":
            return self.slt_type_tuples
        raise ValueError(f"unknown representation {representation!r}")


@dataclass
class TrivialityStats:
    topic_total_unique: int = 0
    topic_selected_fanout: int = 0
    topic_selected_heuristic: int = 0
    answer_total: int = 0
    answer_dropped_trivial: int = 0


def best_pair_containment(
    topic_formulas: list[Formula], answer_formulas: list[Formula], representation: str
) -> float:
    """Max containment over all (topic_formula, answer_formula) pairs -- how
    well the single best-matching formula pair overlaps. This is more
    meaningful than averaging over every pair: most cross-pairs in a
    multi-formula answer are unrelated to the query even for a genuinely
    relevant answer, which would dilute a flat average. 0.0 if either side
    has no formulas to compare."""
    if not topic_formulas or not answer_formulas:
        return 0.0
    best = 0.0
    for tf in topic_formulas:
        tset = tf.tuples_for(representation)
        for af in answer_formulas:
            c = containment(tset, af.tuples_for(representation))
            if c > best:
                best = c
    return best


TOP_K = 5


def topic_containment_stats(
    topic_formulas: list[Formula], answer_formula_lists: list[list[Formula]], representation: str
) -> dict:
    """Per-topic containment stats for one group of answers, all derived from
    the same per-answer best_pair_containment values (computed once):
      - mean: fraction-of-tuples-matched averaged over every answer in the
        group -- dilutes a single great match under a long tail of misses.
      - max: does at least ONE answer in this group contain the query
        formula well -- the number retrieval actually needs, since ranking
        only needs one relevant answer to outrank the crowd.
      - top{TOP_K}_mean: mean of the TOP_K highest per-answer values (or
        fewer if the group has fewer than TOP_K answers) -- a less noisy
        middle ground than max alone.
    """
    if not answer_formula_lists:
        return {"mean": 0.0, "max": 0.0, f"top{TOP_K}_mean": 0.0}
    vals = [best_pair_containment(topic_formulas, af, representation) for af in answer_formula_lists]
    vals_desc = sorted(vals, reverse=True)
    return {
        "mean": sum(vals) / len(vals),
        "max": vals_desc[0],
        f"top{TOP_K}_mean": sum(vals_desc[:TOP_K]) / min(TOP_K, len(vals_desc)),
    }


# --------------------------------------------------------------------------
# Step 0: schema inspection
# --------------------------------------------------------------------------

def inspect_schemas(out_dir: Path) -> str:
    lines = ["# R7 Step 0 -- data schema report\n"]

    def _tsv_head(path: Path, n: int = 2) -> None:
        lines.append(f"## `{path}`\n")
        with open(path, newline="", encoding="utf-8") as f:
            csv.field_size_limit(sys.maxsize)
            reader = csv.DictReader(f, delimiter="\t")
            lines.append(f"Columns: {reader.fieldnames}\n")
            lines.append("```")
            for i, row in enumerate(reader):
                if i >= n:
                    break
                trimmed = {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v) for k, v in row.items()}
                lines.append(str(trimmed))
            lines.append("```\n")

    _tsv_head(SLT_REPRESENTATION / "1.tsv")
    _tsv_head(OPT_REPRESENTATION / "1.tsv")
    _tsv_head(TOPICS_FORMULAS_LATEX)
    _tsv_head(TOPICS_FORMULAS_SLT)
    _tsv_head(TOPICS_FORMULAS_OPT)

    lines.append(f"## `{QREL_TASK1_2022_ALL}`\n")
    lines.append("Format: whitespace-delimited, no header: `topic_id Q0 doc_id label`\n")
    with open(QREL_TASK1_2022_ALL) as f:
        sample = [next(f).strip() for _ in range(3)]
    lines.append("```\n" + "\n".join(sample) + "\n```\n")

    lines.append(f"## `{ANSWERS_JSONL}` (one JSON object per line)\n")
    with open(ANSWERS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("formulas"):
                keys = list(d.keys())
                lines.append(f"Keys: {keys}\n")
                lines.append("```")
                lines.append(str({**{k: d[k] for k in keys if k != "body_text"}, "body_text": d["body_text"][:100] + "..."}))
                lines.append("```\n")
                break

    report = "\n".join(lines)
    out_path = out_dir / "schema_report.md"
    out_path.write_text(report)
    print(report)
    return report


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, score = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def load_topic_formula_rows(path: Path) -> dict[str, list[dict]]:
    """{topic_id: [{id, thread_id, type, formula}, ...]}"""
    out: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        csv.field_size_limit(sys.maxsize)
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["topic_id"]].append(row)
    return out


def load_topic_formulas(
    topic_ids: set[str], stats: TrivialityStats
) -> dict[str, dict[str, list[Formula]]]:
    """Returns {"fanout": {topic_id: [Formula,...]}, "heuristic": {...}}.

    Query-formula selection reuses the actual production functions
    (formula_selector.select_fanout / select_heuristic) instead of a
    hand-rolled dedup/filter loop, so the diagnostic's query set is identical
    to what the real retriever would query with -- including its
    triviality-fallback behaviour (if every formula is trivial, fall back to
    the most complex one rather than returning empty)."""
    latex_rows = load_topic_formula_rows(TOPICS_FORMULAS_LATEX)
    slt_rows = load_topic_formula_rows(TOPICS_FORMULAS_SLT)
    opt_rows = load_topic_formula_rows(TOPICS_FORMULAS_OPT)

    result: dict[str, dict[str, list[Formula]]] = {"fanout": {}, "heuristic": {}}
    for topic_id in topic_ids:
        latex_by_id = {r["id"]: r["formula"] for r in latex_rows.get(topic_id, [])}
        type_by_id = {r["id"]: r["type"] for r in latex_rows.get(topic_id, [])}
        slt_by_id = {r["id"]: r["formula"] for r in slt_rows.get(topic_id, [])}
        opt_by_id = {r["id"]: r["formula"] for r in opt_rows.get(topic_id, [])}

        stats.topic_total_unique += len(set(latex_by_id.values()))

        topic_formulas_field = [
            {"latex": latex, "in_title": type_by_id.get(fid) == "title"}
            for fid, latex in latex_by_id.items()
        ]
        opt_tuple_map: dict[str, list[str]] = {
            latex: extract_tuples_from_mathml_direct(opt_by_id.get(fid, ""), tree_type="OPT")
            for fid, latex in latex_by_id.items()
        }
        topic_dict = {"topic_id": topic_id, "formulas": topic_formulas_field}

        first_id_for_latex: dict[str, str] = {}
        for fid, latex in latex_by_id.items():
            first_id_for_latex.setdefault(latex, fid)

        def _to_formulas(selected_latex: list[str]) -> list[Formula]:
            out = []
            for latex in selected_latex:
                fid = first_id_for_latex.get(latex)
                if fid is None:
                    continue
                slt_tuples = extract_tuples_from_mathml_direct(slt_by_id.get(fid, ""), tree_type="SLT")
                out.append(Formula(fid, latex, frozenset(slt_tuples), frozenset(opt_tuple_map.get(latex, []))))
            return out

        fanout_latex = select_fanout(topic_dict, opt_tuple_map, trivial_filter=True)
        heuristic_latex = select_heuristic(topic_dict, opt_tuple_map, trivial_filter=True)

        result["fanout"][topic_id] = _to_formulas(fanout_latex)
        result["heuristic"][topic_id] = _to_formulas(heuristic_latex)
        stats.topic_selected_fanout += len(fanout_latex)
        stats.topic_selected_heuristic += len(heuristic_latex)

    return result


def stream_needed_answer_records(needed_ids: set[str]) -> dict[str, dict]:
    """Single pass over answers.jsonl, keeping only records whose id is in
    needed_ids. Returns {answer_id: {"formulas": [...], "body_text": str}}."""
    out: dict[str, dict] = {}
    if not needed_ids:
        return out
    remaining = set(needed_ids)
    with open(ANSWERS_JSONL) as f:
        for line in f:
            if not remaining:
                break
            d = json.loads(line)
            aid = d["id"]
            if aid in remaining:
                out[aid] = {"formulas": d.get("formulas") or [], "body_text": d.get("body_text", "")}
                remaining.discard(aid)
    if remaining:
        logger.warning("answers.jsonl: %d requested answer ids not found", len(remaining))
    return out


def load_all_answer_ids() -> list[str]:
    ids: list[str] = []
    with open(ANSWERS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["id"])
    return ids


def build_formula_mathml_index(needed_formula_ids: set[str]) -> dict[str, dict]:
    """Single filtered pass over all SLT+OPT collection shards.
    Returns {formula_id: {"slt_mathml": str, "opt_mathml": str}}."""
    index: dict[str, dict] = {fid: {"slt_mathml": "", "opt_mathml": ""} for fid in needed_formula_ids}

    def _scan(dir_path: Path, key: str) -> None:
        remaining = {fid for fid in needed_formula_ids if not index[fid][key]}
        if not remaining:
            return
        shard_files = sorted(dir_path.glob("*.tsv"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        for shard in shard_files:
            if not remaining:
                break
            with open(shard, newline="", encoding="utf-8", errors="replace") as f:
                csv.field_size_limit(sys.maxsize)
                for row in csv.DictReader(f, delimiter="\t"):
                    fid = row.get("id")
                    if fid in remaining:
                        index[fid][key] = row.get("formula", "")
                        remaining.discard(fid)

    _scan(SLT_REPRESENTATION, "slt_mathml")
    _scan(OPT_REPRESENTATION, "opt_mathml")

    missing = [fid for fid in needed_formula_ids if not index[fid]["slt_mathml"] and not index[fid]["opt_mathml"]]
    if missing:
        logger.warning("formula ids not found in collection shards: %d (e.g. %s)", len(missing), missing[:5])
    return index


def build_answer_formulas(
    answer_record: dict, mathml_index: dict[str, dict], stats: TrivialityStats,
    formula_cache: dict[str, Formula | None],
) -> list[Formula]:
    """Answer-side formulas are filtered for triviality (is_trivial on OPT
    tuples, matching the same predicate/representation the real query-formula
    selectors use) -- a diagnostic-only step: the real indexer does not
    filter the answer corpus for triviality at all, only query-formula
    selection does. Kept here because a bare 'x' would otherwise trivially
    'match' everything and make the diagnostic meaningless."""
    out: list[Formula] = []
    for fdict in answer_record["formulas"]:
        fid = fdict["formula_id"]
        if fid in formula_cache:
            cached = formula_cache[fid]
            if cached is not None:
                out.append(cached)
            continue
        mm = mathml_index.get(fid)
        if mm is None:
            formula_cache[fid] = None
            continue
        stats.answer_total += 1
        opt_tuples = extract_tuples_from_mathml_direct(mm["opt_mathml"], tree_type="OPT") if mm["opt_mathml"] else []
        if is_trivial(opt_tuples, tree_type="OPT"):
            stats.answer_dropped_trivial += 1
            formula_cache[fid] = None
            continue
        slt_tuples = extract_tuples_from_mathml_direct(mm["slt_mathml"], tree_type="SLT") if mm["slt_mathml"] else []
        f = Formula(
            formula_id=fid,
            latex=fdict.get("latex", ""),
            slt_tuples=frozenset(slt_tuples),
            opt_tuples=frozenset(opt_tuples),
        )
        formula_cache[fid] = f
        out.append(f)
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def run_diagnostic(
    qrels: dict[str, dict[str, int]],
    topic_ids: list[str],
    seed: int,
) -> dict:
    stats = TrivialityStats()
    logger.info("Loading topic formulas for %d topics (fanout + heuristic)...", len(topic_ids))
    topic_formulas_by_strategy = load_topic_formulas(set(topic_ids), stats)

    # Determine relevant/non-relevant answer ids per topic (High=3, High+Med>=2)
    relevant_high: dict[str, list[str]] = {}
    relevant_hm: dict[str, list[str]] = {}
    nonrelevant: dict[str, list[str]] = {}
    excluded_no_topic_formula: list[str] = []
    excluded_no_relevant_high: list[str] = []
    excluded_no_relevant_hm: list[str] = []

    for tid in topic_ids:
        labels = qrels.get(tid, {})
        high = sorted([d for d, s in labels.items() if s == 3])
        hm = sorted([d for d, s in labels.items() if s >= 2])
        zero = sorted([d for d, s in labels.items() if s == 0])
        relevant_high[tid] = high
        relevant_hm[tid] = hm
        nonrelevant[tid] = zero
        if not topic_formulas_by_strategy["fanout"].get(tid):
            excluded_no_topic_formula.append(tid)
        if not high:
            excluded_no_relevant_high.append(tid)
        if not hm:
            excluded_no_relevant_hm.append(tid)

    logger.info("Loading answer-id universe from answers.jsonl for random sampling...")
    all_answer_ids = load_all_answer_ids()

    rng = random.Random(seed)
    random_controls: dict[str, list[str]] = {}
    for tid in sorted(topic_ids):
        n_needed = len(relevant_hm.get(tid, []))
        random_controls[tid] = rng.sample(all_answer_ids, min(n_needed, len(all_answer_ids))) if n_needed else []

    nonrelevant_controls: dict[str, list[str]] = {}
    nonrel_shortfall: dict[str, int] = {}
    for tid in sorted(topic_ids):
        n_needed = len(relevant_hm.get(tid, []))
        pool = nonrelevant.get(tid, [])
        if n_needed == 0:
            nonrelevant_controls[tid] = []
            continue
        if len(pool) <= n_needed:
            nonrelevant_controls[tid] = list(pool)
            if len(pool) < n_needed:
                nonrel_shortfall[tid] = n_needed - len(pool)
        else:
            nonrelevant_controls[tid] = rng.sample(pool, n_needed)

    needed_answer_ids: set[str] = set()
    for tid in topic_ids:
        needed_answer_ids.update(relevant_high[tid])
        needed_answer_ids.update(relevant_hm[tid])
        needed_answer_ids.update(nonrelevant_controls[tid])
        needed_answer_ids.update(random_controls[tid])

    logger.info("Streaming %d needed answer records from answers.jsonl...", len(needed_answer_ids))
    answer_records = stream_needed_answer_records(needed_answer_ids)

    needed_formula_ids: set[str] = set()
    for rec in answer_records.values():
        for fd in rec["formulas"]:
            needed_formula_ids.add(fd["formula_id"])

    logger.info("Scanning collection shards for %d needed formula ids...", len(needed_formula_ids))
    mathml_index = build_formula_mathml_index(needed_formula_ids)

    formula_cache: dict[str, Formula | None] = {}

    def _formulas_for(answer_id: str) -> list[Formula]:
        rec = answer_records.get(answer_id)
        if rec is None:
            return []
        return build_answer_formulas(rec, mathml_index, stats, formula_cache)

    # Cache answer-formula lookups across strategies/representations (same answer id
    # is reused many times: relevant/control, per strategy, per representation).
    afs_cache: dict[str, list[Formula]] = {}

    def _afs(ids: list[str]) -> list[list[Formula]]:
        out = []
        for aid in ids:
            if aid not in afs_cache:
                afs_cache[aid] = _formulas_for(aid)
            out.append(afs_cache[aid])
        return out

    # ---- Analysis A+B: per-topic containment stats, per strategy/representation/group ----
    overlap_by_topic_rows: list[dict] = []
    overlap_lookup: dict[tuple, dict] = {}  # (strategy, tid, rel_def, representation, group) -> {mean,max,top5_mean}

    def _stats_row(strategy, tid, rel_def, representation, group, stats, n_answers, n_formulas) -> dict:
        return {
            "query_strategy": strategy, "topic_id": tid, "relevance_def": rel_def,
            "representation": representation, "group": group,
            "mean_containment": stats["mean"], "max_containment": stats["max"],
            f"top{TOP_K}_mean_containment": stats[f"top{TOP_K}_mean"],
            "n_answers": n_answers, "n_topic_formulas_selected": n_formulas,
        }

    for strategy in QUERY_STRATEGIES:
        for tid in sorted(topic_ids):
            tformulas = topic_formulas_by_strategy[strategy].get(tid, [])

            # "high" and "high_medium": relevant group only (Analysis A)
            for rel_def, rel_ids in (("high", relevant_high[tid]), ("high_medium", relevant_hm[tid])):
                afs = _afs(rel_ids)
                for representation in REPRESENTATIONS:
                    cont_stats = topic_containment_stats(tformulas, afs, representation)
                    overlap_lookup[(strategy, tid, rel_def, representation, "relevant")] = cont_stats
                    overlap_by_topic_rows.append(
                        _stats_row(strategy, tid, rel_def, representation, "relevant", cont_stats, len(rel_ids), len(tformulas))
                    )

            # controls: sized against high_medium only (Analysis B), same as v1's convention
            if not tformulas or not relevant_hm[tid]:
                continue
            nonrel_afs = _afs(nonrelevant_controls[tid])
            rand_afs = _afs(random_controls[tid])
            for representation in REPRESENTATIONS:
                stats_nonrel = topic_containment_stats(tformulas, nonrel_afs, representation)
                stats_rand = topic_containment_stats(tformulas, rand_afs, representation)
                overlap_lookup[(strategy, tid, "high_medium", representation, "non_relevant")] = stats_nonrel
                overlap_lookup[(strategy, tid, "high_medium", representation, "random")] = stats_rand
                overlap_by_topic_rows.append(
                    _stats_row(strategy, tid, "high_medium", representation, "non_relevant",
                               stats_nonrel, len(nonrelevant_controls[tid]), len(tformulas))
                )
                overlap_by_topic_rows.append(
                    _stats_row(strategy, tid, "high_medium", representation, "random",
                               stats_rand, len(random_controls[tid]), len(tformulas))
                )

    # ---- Overlap summary: plain aggregates + directionality counts, no significance tests ----
    stat_keys = ["mean", "max", f"top{TOP_K}_mean"]
    overlap_summary_rows: list[dict] = []
    for strategy in QUERY_STRATEGIES:
        for representation in REPRESENTATIONS:
            rel_stats, nonrel_stats, rand_stats = [], [], []
            for tid in sorted(topic_ids):
                if not topic_formulas_by_strategy[strategy].get(tid) or not relevant_hm[tid]:
                    continue
                rel_stats.append(overlap_lookup[(strategy, tid, "high_medium", representation, "relevant")])
                nonrel_stats.append(overlap_lookup[(strategy, tid, "high_medium", representation, "non_relevant")])
                rand_stats.append(overlap_lookup[(strategy, tid, "high_medium", representation, "random")])
            n = len(rel_stats)
            row = {"query_strategy": strategy, "representation": representation, "n_topics": n}
            for stat_key in stat_keys:
                rel_vals = [s[stat_key] for s in rel_stats]
                nonrel_vals = [s[stat_key] for s in nonrel_stats]
                rand_vals = [s[stat_key] for s in rand_stats]
                row[f"{stat_key}_relevant"] = (sum(rel_vals) / n) if n else float("nan")
                row[f"{stat_key}_non_relevant"] = (sum(nonrel_vals) / n) if n else float("nan")
                row[f"{stat_key}_random"] = (sum(rand_vals) / n) if n else float("nan")
                row[f"n_topics_{stat_key}_relevant_gt_non_relevant"] = sum(1 for r, c in zip(rel_vals, nonrel_vals) if r > c)
                row[f"n_topics_{stat_key}_relevant_gt_random"] = sum(1 for r, c in zip(rel_vals, rand_vals) if r > c)
            overlap_summary_rows.append(row)

    # ---- Analysis C: worked examples (fanout only, High-relevant, lowest overlap) ----
    candidates: list[tuple[float, str]] = []
    for tid in sorted(topic_ids):
        tformulas = topic_formulas_by_strategy["fanout"].get(tid, [])
        high_ids = relevant_high[tid]
        if len(tformulas) < 2 or len(high_ids) < 3:
            continue
        afs = _afs(high_ids)
        avg_containment = sum(
            topic_containment_stats(tformulas, afs, representation)["mean"] for representation in REPRESENTATIONS
        ) / len(REPRESENTATIONS)
        candidates.append((avg_containment, tid))
    candidates.sort(key=lambda x: (x[0], x[1]))
    selected = [tid for _, tid in candidates[:3]]

    worked_examples_md_parts = []
    if selected:
        topics_xml_reader = TopicReader(TOPICS_TASK1_2022_XML)
        for tid in selected:
            topic_obj = topics_xml_reader.map_topics.get(tid)
            tformulas = topic_formulas_by_strategy["fanout"].get(tid, [])
            parts = [f"## Topic {tid}\n"]
            if topic_obj is not None:
                parts.append(f"**Title:** {topic_obj.title}\n")
                parts.append(f"**Question:**\n\n{topic_obj.question}\n")
            parts.append("**Topic formulas queried (fanout, LaTeX):**\n")
            for f in tformulas:
                parts.append(f"- `{f.latex}`")
            parts.append("\n**High-rated (label=3) answers:**\n")
            for aid in relevant_high[tid]:
                rec = answer_records.get(aid)
                latexes = [fd["latex"] for fd in rec["formulas"]] if rec else []
                parts.append(f"- Answer `{aid}`:")
                for lx in latexes:
                    parts.append(f"    - `{lx}`")
            worked_examples_md_parts.append("\n".join(parts))
    worked_examples_md = "\n\n---\n\n".join(worked_examples_md_parts) if worked_examples_md_parts else (
        "No topic satisfied the selection criteria (>=2 fanout-selected topic formulas, "
        ">=3 High-rated answers)."
    )

    return {
        "stats": stats,
        "overlap_by_topic_rows": overlap_by_topic_rows,
        "overlap_summary_rows": overlap_summary_rows,
        "worked_examples_md": worked_examples_md,
        "n_worked_examples": len(selected),
        "excluded_no_topic_formula": excluded_no_topic_formula,
        "excluded_no_relevant_high": excluded_no_relevant_high,
        "excluded_no_relevant_hm": excluded_no_relevant_hm,
        "nonrel_shortfall": nonrel_shortfall,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    """fieldnames is the union of all keys seen, in first-appearance order --
    rows aren't guaranteed perfectly homogeneous across call sites."""
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(result: dict) -> None:
    print("\n=== R7 formula-overlap: headline summary (relevance_def=high_medium) ===")
    for stat_key, label in (("mean", "MEAN"), ("max", "MAX"), (f"top{TOP_K}_mean", f"TOP-{TOP_K} MEAN")):
        print(f"\n--- {label} containment ---")
        print(f"{'strategy':10s} {'repr':9s} {'n':>4s} {'relevant':>9s} {'non_rel':>9s} {'random':>9s} "
              f"{'rel>nonrel':>11s} {'rel>rand':>9s}")
        for row in result["overlap_summary_rows"]:
            print(f"{row['query_strategy']:10s} {row['representation']:9s} {row['n_topics']:>4d} "
                  f"{row[f'{stat_key}_relevant']:>9.3f} {row[f'{stat_key}_non_relevant']:>9.3f} "
                  f"{row[f'{stat_key}_random']:>9.3f} "
                  f"{row[f'n_topics_{stat_key}_relevant_gt_non_relevant']:>5d}/{row['n_topics']:<5d} "
                  f"{row[f'n_topics_{stat_key}_relevant_gt_random']:>4d}/{row['n_topics']:<4d}")
    print(f"\nWorked examples selected: {result['n_worked_examples']}")
    print(f"Topics excluded (no topic formula at all): {len(result['excluded_no_topic_formula'])}")
    print(f"Topics excluded (no High-relevant answers): {len(result['excluded_no_relevant_high'])}")
    print(f"Topics excluded (no High+Medium-relevant answers): {len(result['excluded_no_relevant_hm'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="R7: formula-overlap diagnostic for ARQMath Task 1")
    parser.add_argument("--max-topics", type=int, default=None)
    parser.add_argument("--topic-ids", default=None, help="Comma-separated allowlist, e.g. A.301,A.302")
    parser.add_argument("--out-dir", default=str(RESULTS_DIAGNOSTICS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inspect_schemas(out_dir)

    qrels = load_qrels(QREL_TASK1_2022_ALL)
    topic_ids = sorted(qrels.keys())
    if args.topic_ids:
        allow = set(args.topic_ids.split(","))
        topic_ids = [t for t in topic_ids if t in allow]
    if args.max_topics:
        topic_ids = topic_ids[: args.max_topics]

    logger.info("Running diagnostic over %d topics", len(topic_ids))
    result = run_diagnostic(qrels, topic_ids, seed=args.seed)

    stats: TrivialityStats = result["stats"]
    write_csv(out_dir / "triviality_report.csv", [{
        "topic_total_unique": stats.topic_total_unique,
        "topic_selected_fanout": stats.topic_selected_fanout,
        "topic_selected_heuristic": stats.topic_selected_heuristic,
        "answer_total": stats.answer_total,
        "answer_dropped_trivial": stats.answer_dropped_trivial,
    }])
    write_csv(out_dir / "overlap_by_topic.csv", result["overlap_by_topic_rows"])
    write_csv(out_dir / "overlap_summary.csv", result["overlap_summary_rows"])
    (out_dir / "worked_examples.md").write_text(result["worked_examples_md"])

    summary = {
        "n_topics": len(topic_ids),
        "triviality": {
            "topic_total_unique": stats.topic_total_unique,
            "topic_selected_fanout": stats.topic_selected_fanout,
            "topic_selected_heuristic": stats.topic_selected_heuristic,
            "answer_total": stats.answer_total,
            "answer_dropped_trivial": stats.answer_dropped_trivial,
        },
        "excluded_no_topic_formula": result["excluded_no_topic_formula"],
        "excluded_no_relevant_high": result["excluded_no_relevant_high"],
        "excluded_no_relevant_hm": result["excluded_no_relevant_hm"],
        "n_worked_examples": result["n_worked_examples"],
        "overlap_summary": result["overlap_summary_rows"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print_summary(result)

    if not args.no_wandb:
        try:
            wandb_config = WandbConfig(group="Diagnostics : Formula Overlap", job_type="r7-formula-overlap",
                                        tags=["R7", "diagnostics", "formula", "arqmath-3-task1"])
            logger_ = ExperimentLogger(wandb_config)
            logger_.start_run(config={"seed": args.seed, "n_topics": len(topic_ids)},
                               run_name=f"r7-formula-overlap-{datetime.now(timezone.utc):%Y%m%d-%H%M}")
            logger_.log_table("overlap_by_topic", result["overlap_by_topic_rows"])
            logger_.log_table("overlap_summary", result["overlap_summary_rows"])
            for f in ["triviality_report.csv", "overlap_by_topic.csv", "overlap_summary.csv",
                      "worked_examples.md", "summary.json", "schema_report.md"]:
                logger_.save_file(out_dir / f)
            logger_.finish()
        except Exception as e:  # noqa: BLE001
            logger.warning("W&B logging failed (results already written locally): %s", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
