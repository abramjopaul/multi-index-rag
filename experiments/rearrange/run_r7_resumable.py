#!/usr/bin/env python3
"""Run R7 (experiments/diagnostics/formula_overlap.py) in steps that each fit a time limit.

formula_overlap.py is imported unchanged and its own main() produces every output. Only
four of its helpers are wrapped, each deterministic given its inputs, so the result is
identical to one uninterrupted run:

  load_all_answer_ids, stream_needed_answer_records   -- result cached to disk
  build_formula_mathml_index                          -- same first-occurrence scan of the
                                                         SLT/OPT shards, cached shard by shard
  extract_tuples_from_mathml_direct                   -- memoised (pure function)

When --time-budget is reached before an uncached step, the run saves its caches and exits
75; run the same command again to continue. Exit 0 means formula_overlap.py finished and
wrote its outputs to --out-dir.

Usage (the 78 official topics are the default):
    python experiments/rearrange/run_r7_resumable.py --out-dir /tmp/r7 --cache-dir /tmp/r7cache --time-budget 140
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T0 = time.monotonic()


class OutOfTime(SystemExit):
    def __init__(self) -> None:
        super().__init__(75)


def _key(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--time-budget", type=float, default=float("inf"))
    ap.add_argument("--qrels", type=Path, default=ROOT / "data/raw/qrels/qrel_task1_2022_official.tsv",
                    help="topic ids are taken from this file and passed to formula_overlap.py --topic-ids")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    def over_budget() -> bool:
        return time.monotonic() - T0 > args.time_budget

    spec = importlib.util.spec_from_file_location("formula_overlap", ROOT / "experiments/diagnostics/formula_overlap.py")
    fo = importlib.util.module_from_spec(spec)
    sys.modules["formula_overlap"] = fo  # dataclasses look the module up here
    spec.loader.exec_module(fo)

    # 1. answer-id universe (order matters: the random controls sample from it)
    orig_ids = fo.load_all_answer_ids

    def load_all_answer_ids():
        p = cache / "answer_ids.json"
        if p.exists():
            return json.loads(p.read_text())
        ids = orig_ids()
        p.write_text(json.dumps(ids))
        return ids

    # 2. the needed answer records
    orig_stream = fo.stream_needed_answer_records

    def stream_needed_answer_records(needed_ids):
        p = cache / f"answers_{_key(needed_ids)}.pkl"
        if p.exists():
            return pickle.loads(p.read_bytes())
        out = orig_stream(needed_ids)
        p.write_bytes(pickle.dumps(out))
        return out

    # 3. the shard scan -- same semantics as the original, one shard at a time
    def build_formula_mathml_index(needed_formula_ids):
        index = {fid: {"slt_mathml": "", "opt_mathml": ""} for fid in needed_formula_ids}
        sdir = cache / f"shards_{_key(needed_formula_ids)}"
        sdir.mkdir(exist_ok=True)
        for dir_path, key in ((fo.SLT_REPRESENTATION, "slt_mathml"), (fo.OPT_REPRESENTATION, "opt_mathml")):
            remaining = {fid for fid in needed_formula_ids if not index[fid][key]}
            shards = sorted(dir_path.glob("*.tsv"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            for shard in shards:
                if not remaining:
                    break
                cp = sdir / f"{key}_{shard.stem}.pkl"
                if cp.exists():
                    hits = pickle.loads(cp.read_bytes())
                else:
                    if over_budget():
                        print(f"[r7] time budget reached in the {key} scan at shard {shard.name}", flush=True)
                        raise OutOfTime()
                    hits = {}
                    with open(shard, newline="", encoding="utf-8", errors="replace") as f:
                        csv.field_size_limit(sys.maxsize)
                        for row in csv.DictReader(f, delimiter="\t"):
                            fid = row.get("id")
                            if fid in index and fid not in hits:
                                hits[fid] = row.get("formula", "")
                    tmp = cp.with_suffix(".tmp")
                    tmp.write_bytes(pickle.dumps(hits))
                    tmp.rename(cp)
                for fid, formula in hits.items():
                    if fid in remaining:
                        index[fid][key] = formula
                        remaining.discard(fid)
        missing = [fid for fid in needed_formula_ids if not index[fid]["slt_mathml"] and not index[fid]["opt_mathml"]]
        if missing:
            fo.logger.warning("formula ids not found in collection shards: %d (e.g. %s)", len(missing), missing[:5])
        return index

    # 4. tuple extraction, memoised
    memo_path = cache / "tuples_memo.pkl"
    memo = pickle.loads(memo_path.read_bytes()) if memo_path.exists() else {}
    n_before = len(memo)
    orig_extract = fo.extract_tuples_from_mathml_direct

    def extract_tuples_from_mathml_direct(mathml, tree_type="SLT"):
        k = (hashlib.sha1(mathml.encode("utf-8", "surrogatepass")).hexdigest(), tree_type)
        if k in memo:
            return list(memo[k])
        if over_budget():
            raise OutOfTime()
        out = orig_extract(mathml, tree_type=tree_type)
        memo[k] = tuple(out)
        return out

    fo.load_all_answer_ids = load_all_answer_ids
    fo.stream_needed_answer_records = stream_needed_answer_records
    fo.build_formula_mathml_index = build_formula_mathml_index
    fo.extract_tuples_from_mathml_direct = extract_tuples_from_mathml_direct

    topics = sorted({ln.split()[0] for ln in args.qrels.read_text().splitlines() if ln.strip()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = ["formula_overlap.py", "--topic-ids", ",".join(topics), "--out-dir", str(args.out_dir),
                "--seed", str(args.seed), "--no-wandb"]
    code = 0
    try:
        code = fo.main() or 0
    except OutOfTime:
        code = 75
    finally:
        if len(memo) > n_before:
            tmp = memo_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(memo))
            tmp.rename(memo_path)
        print(f"[r7] exit {code} after {time.monotonic() - T0:.0f}s; {len(topics)} topics; "
              f"tuple memo {n_before} -> {len(memo)}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
