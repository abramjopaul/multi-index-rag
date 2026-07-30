"""Reproducibility manifest for Track V runs.

Turns the ad hoc "_meta" row pattern used elsewhere (e.g. experiments/run_c0_1.py)
into one reusable builder, so the JSONL meta row and wandb.config never drift
apart — every Track V script calls build_manifest() once and writes/passes the
same dict to both places.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_text(Path(path).read_text())


def _package_version(name: str) -> str | None:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> tuple[str | None, bool]:
    """Returns (commit_hash, is_dirty). Both None/False if not a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def build_manifest(
    config,
    execution_mode: str,
    prompt_sha256: str,
    config_sha256: str,
    usage: dict | None = None,
) -> dict:
    """Build the reproducibility manifest shared by every Track V result file
    and mirrored verbatim into wandb.config (same dict, no drift).

    Args:
        config: a JudgeConfig (src/multirag/config/judge_config.py).
        execution_mode: "batch" | "sync" — the path actually used for this run.
        prompt_sha256: sha256 of the resolved prompt template text.
        config_sha256: sha256 of the resolved (YAML) config text.
        usage: optional usage/cost log (pairs submitted, cache hits, tokens,
            estimated cost) — merged in as-is once known.
    """
    commit, dirty = _git_commit()
    return {
        "judge_model": config.judge.model,
        "judge_provider": config.judge.provider,
        "judge_temperature": config.judge.temperature,
        "prompt_sha256": prompt_sha256,
        "execution_mode": execution_mode,
        "config_sha256": config_sha256,
        "python_version": sys.version.split()[0],
        "google_genai_version": _package_version("google-genai"),
        "instructor_version": _package_version("instructor"),
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "usage": usage or {},
    }
