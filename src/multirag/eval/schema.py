"""Data schemas for the Track V judge pipeline.

Plain dataclasses (matching RagasSample's convention in
generation/ragas_eval.py) — round-trip losslessly through JSONL via
dataclasses.asdict / **kwargs construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class JudgePair:
    topic_id: str
    answer_id: str
    question: str  # topic title + body (LaTeX verbatim)
    answer_text: str  # answer post body (LaTeX verbatim)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JudgePair":
        return cls(**d)


@dataclass
class RelevanceJudgement:
    topic_id: str
    answer_id: str
    label: int | None  # 0|1|2|3 ; None if parse failed
    parse_ok: bool
    raw_response: str
    prompt_sha256: str
    judge_model: str
    source: str  # 'batch' | 'sync' | 'cache'
    timestamp: str  # UTC ISO-8601

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RelevanceJudgement":
        return cls(**d)


@dataclass
class PairedLabel:
    """Side-by-side row: human label and judge label for the SAME (topic, answer).

    This is the raw material kappa/tau are computed from, and the primary
    human-readable audit artifact. One row per judged (topic, answer) pair.
    """

    topic_id: str
    answer_id: str
    human_label: int  # 0|1|2|3  (from qrels)
    judge_label: int | None  # 0|1|2|3  (None if parse failed)
    human_bin: int  # H+M -> 1, L+N -> 0  (relevance_level=2)
    judge_bin: int | None
    agree_graded: bool | None  # human_label == judge_label
    agree_binary: bool | None  # human_bin  == judge_bin
    delta: int | None  # judge_label - human_label  (signed error; None if parse failed)
    topic_type: str | None  # computation | concept | proof (if available)
    multi_approach: bool  # flagged topics (e.g. A.302)
    question: str  # for eyeballing disagreements
    answer_text: str
    judge_raw_response: str  # so a disagreement can be inspected without a re-run
    year: str | None = (
        None  # "2020" | "2021" | "2022", from validation_set's qrel_source
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PairedLabel":
        return cls(**d)


# Compact view of a PairedLabel for the eyeball-in-a-spreadsheet CSV.
CSV_COLUMNS = [
    "topic_id",
    "answer_id",
    "year",
    "human_label",
    "judge_label",
    "human_bin",
    "judge_bin",
    "delta",
    "agree_graded",
    "agree_binary",
]


def paired_label_to_csv_row(p: PairedLabel) -> dict:
    return {col: getattr(p, col) for col in CSV_COLUMNS}
