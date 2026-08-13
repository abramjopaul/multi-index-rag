"""JudgeClient: the single module that talks to the LLM relevance judge API.

Two execution backends behind one interface, so callers never change when
scale changes:
  - submit_batch / poll_batch / fetch_batch: Gemini Batch API (~50% cost, high
    throughput, no latency requirement) -- the default, and the only path
    used for real spend (V's ~100K pairs, Track B's later ~6K).
  - judge_sync: synchronous single-call path, smoke tests / tiny runs only.

Cache-first: pairs already labelled -- keyed on
sha256(topic_id | answer_id | prompt_sha256 | judge_model) -- are never
re-submitted; results are written to cache the instant they're fetched. This
is what lets Track B/C reuse Track V's spend as a lookup instead of a re-spend.

No model strings or endpoints are hardcoded here beyond the pydantic defaults
in judge_config.py -- everything real comes from configs/eval/judge.yaml.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from multirag.config.judge_config import JudgeConfig
from multirag.eval.manifest import sha256_text
from multirag.eval.schema import JudgePair, RelevanceJudgement

logger = logging.getLogger(__name__)

# Terminal Gemini Batch API job states (google.genai.types.JobState names).
_TERMINAL_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }
)

# Rough, approximate USD/1M-token rates for cost logging only -- NOT billing-
# accurate. Update alongside the provider's published pricing if it changes;
# unknown models fall back to the flash-tier default.
_COST_PER_MILLION_TOKENS_USD = {
    "gemini-2.0-flash": 0.15,
    "gemini-1.5-flash": 0.15,
}
_DEFAULT_COST_PER_MILLION_TOKENS_USD = 0.15


class _RelevanceOutput(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the label")
    label: int = Field(
        ge=0, le=3, description="0=not relevant, 1=low, 2=medium, 3=high"
    )


def load_prompt_template(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_model_output_failure(e: Exception) -> bool:
    """True only for a genuine "the model's output failed validation after a
    real API round-trip" outcome -- as opposed to a client-side/config/
    library/provider error (bad request, missing key, network blip, outage),
    which must never be cached (see judge_sync's comment on why).
    """
    import pydantic
    from instructor.core import InstructorRetryException

    return isinstance(e, (InstructorRetryException, pydantic.ValidationError))


def _cache_key(
    topic_id: str, answer_id: str, prompt_sha256: str, judge_model: str
) -> str:
    return sha256_text(f"{topic_id}|{answer_id}|{prompt_sha256}|{judge_model}")


@dataclass
class BatchHandle:
    job_name: str
    chunk_id: str
    pair_keys: list[str]  # cache keys in submission order, for checkpointing


@dataclass
class BatchStatus:
    job_name: str
    state: str
    done: bool


class JudgeClient:
    """The only module that talks to the judge API.

    Track B and Track C call it through submit_batch/fetch_batch -- if any
    future step needs the judge API, it goes through this client; do not add
    a second API path anywhere.
    """

    def __init__(self, config: JudgeConfig):
        self._config = config
        self._cache_dir = Path(config.cache.dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._cache_dir / "checkpoint.json"

        prompt_path = Path(config.prompt.template_path)
        self._prompt = load_prompt_template(prompt_path)
        self._prompt_sha256 = sha256_text(prompt_path.read_text())

        self._genai_client = None
        self._instructor_client = None

        self.usage = {
            "pairs_submitted": 0,
            "cache_hits": 0,
            "tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    @property
    def prompt_sha256(self) -> str:
        return self._prompt_sha256

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _cache_get(self, pair: JudgePair) -> RelevanceJudgement | None:
        key = _cache_key(
            pair.topic_id, pair.answer_id, self._prompt_sha256, self._config.judge.model
        )
        path = self._cache_path(key)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        judgement = RelevanceJudgement.from_dict(data)
        judgement.source = "cache"
        return judgement

    def _cache_put(
        self, topic_id: str, answer_id: str, judgement: RelevanceJudgement
    ) -> None:
        key = _cache_key(
            topic_id, answer_id, self._prompt_sha256, self._config.judge.model
        )
        with open(self._cache_path(key), "w") as f:
            json.dump(judgement.to_dict(), f)

    def partition_cached(
        self, pairs: list[JudgePair]
    ) -> tuple[list[RelevanceJudgement], list[JudgePair]]:
        """Split pairs into (already-cached judgements, pairs still needing labelling)."""
        cached: list[RelevanceJudgement] = []
        remaining: list[JudgePair] = []
        for pair in pairs:
            hit = self._cache_get(pair)
            if hit is not None:
                cached.append(hit)
                self.usage["cache_hits"] += 1
            else:
                remaining.append(pair)
        return cached, remaining

    def _record_tokens(self, n_tokens: int | None) -> None:
        if not n_tokens:
            return
        self.usage["tokens"] += n_tokens
        rate = _COST_PER_MILLION_TOKENS_USD.get(
            self._config.judge.model, _DEFAULT_COST_PER_MILLION_TOKENS_USD
        )
        self.usage["estimated_cost_usd"] = round(
            self.usage["estimated_cost_usd"] + (n_tokens / 1_000_000) * rate, 6
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _messages_for_instructor(self, pair: JudgePair) -> list[dict]:
        """Chat-message format for the sync/instructor path. The native
        google-genai backend expects Gemini's own role name "model" for
        assistant turns (confirmed empirically -- passing "assistant" fails
        with "Unsupported role: assistant"); the openai_compat fallback goes
        through an OpenAI-style API and expects "assistant". instructor does
        not translate between the two, so the role name must match whichever
        backend is actually in use.
        """
        assistant_role = (
            "assistant" if self._config.judge.use_openai_compat_endpoint else "model"
        )
        messages = [{"role": "system", "content": self._prompt["system_prompt"]}]
        for ex in self._prompt.get("few_shot_examples", []):
            user_msg = self._prompt["user_prompt_template"].format(
                question=ex["question"], answer_text=ex["answer"]
            )
            messages.append({"role": "user", "content": user_msg})
            messages.append(
                {
                    "role": assistant_role,
                    "content": f"{ex['reasoning']}\nLABEL: {ex['label']}",
                }
            )
        final_msg = self._prompt["user_prompt_template"].format(
            question=pair.question, answer_text=pair.answer_text
        )
        messages.append({"role": "user", "content": final_msg})
        return messages

    def _build_contents(self, pair: JudgePair):
        """google.genai Content list, for the native batch path."""
        from google.genai import types

        contents = [
            types.Content(
                role="user", parts=[types.Part(text=self._prompt["system_prompt"])]
            )
        ]
        for ex in self._prompt.get("few_shot_examples", []):
            user_msg = self._prompt["user_prompt_template"].format(
                question=ex["question"], answer_text=ex["answer"]
            )
            contents.append(
                types.Content(role="user", parts=[types.Part(text=user_msg)])
            )
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=f"{ex['reasoning']}\nLABEL: {ex['label']}")],
                )
            )
        final_msg = self._prompt["user_prompt_template"].format(
            question=pair.question, answer_text=pair.answer_text
        )
        contents.append(types.Content(role="user", parts=[types.Part(text=final_msg)]))
        return contents

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _api_key(self) -> str:
        import os

        api_key = os.environ.get(self._config.judge.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Judge API key not found. Set the {self._config.judge.api_key_env!r} "
                "environment variable."
            )
        return api_key

    def _get_genai_client(self):
        from google import genai

        if self._genai_client is None:
            self._genai_client = genai.Client(api_key=self._api_key())
        return self._genai_client

    def _get_sync_client(self):
        """Instructor-wrapped client for judge_sync -- native google-genai by
        default, or the Gemini OpenAI-compatible endpoint as a fallback
        (works around instructor#1658's safety-settings error on some
        genai/instructor version combos). Embeddings are unaffected and not
        needed by the relevance judge at all.
        """
        import instructor

        if self._instructor_client is not None:
            return self._instructor_client

        if self._config.judge.use_openai_compat_endpoint:
            from openai import OpenAI

            client = OpenAI(
                base_url=self._config.judge.openai_compat_base_url,
                api_key=self._api_key(),
            )
            self._instructor_client = instructor.from_openai(client)
        else:
            genai_client = self._get_genai_client()
            mode = getattr(instructor.Mode, "GENAI_TOOLS", instructor.Mode.TOOLS)
            self._instructor_client = instructor.from_genai(genai_client, mode=mode)
        return self._instructor_client

    # ------------------------------------------------------------------
    # Sync path (smoke tests / tiny runs only)
    # ------------------------------------------------------------------

    def judge_sync(self, pairs: list[JudgePair]) -> list[RelevanceJudgement]:
        cached, remaining = self.partition_cached(pairs)
        results = list(cached)
        client = self._get_sync_client()

        for pair in remaining:
            messages = self._messages_for_instructor(pair)
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                output, completion = client.create_with_completion(
                    model=self._config.judge.model,
                    messages=messages,
                    response_model=_RelevanceOutput,
                    temperature=self._config.judge.temperature,
                    max_tokens=self._config.judge.max_tokens,
                )
                raw_text = json.dumps(output.model_dump())
                judgement = RelevanceJudgement(
                    topic_id=pair.topic_id,
                    answer_id=pair.answer_id,
                    label=output.label,
                    parse_ok=True,
                    raw_response=raw_text,
                    prompt_sha256=self._prompt_sha256,
                    judge_model=self._config.judge.model,
                    source="sync",
                    timestamp=timestamp,
                )
                usage_meta = getattr(completion, "usage_metadata", None) or getattr(
                    completion, "usage", None
                )
                total_tokens = None
                if usage_meta is not None:
                    total_tokens = getattr(
                        usage_meta, "total_token_count", None
                    ) or getattr(usage_meta, "total_tokens", None)
                self._record_tokens(total_tokens)
            except (
                Exception
            ) as e:  # noqa: BLE001 -- any judge/parse failure -> parse_ok=False
                logger.warning(
                    f"Judge sync call failed for {pair.topic_id}/{pair.answer_id}: {e}"
                )
                judgement = RelevanceJudgement(
                    topic_id=pair.topic_id,
                    answer_id=pair.answer_id,
                    label=None,
                    parse_ok=False,
                    raw_response=str(e),
                    prompt_sha256=self._prompt_sha256,
                    judge_model=self._config.judge.model,
                    source="sync",
                    timestamp=timestamp,
                )
                # Only cache a genuine "the model's output failed validation"
                # outcome (instructor exhausted its retries on real API
                # responses) -- never cache a client-side/config/library
                # error (bad role name, missing key, network blip, provider
                # outage), or a transient bug would poison the cache forever
                # and every future run would silently replay the same
                # failure instead of retrying. See the "Unsupported role"
                # bug this caught during development.
                if _is_model_output_failure(e):
                    self._cache_put(pair.topic_id, pair.answer_id, judgement)
                self.usage["pairs_submitted"] += 1
                results.append(judgement)
                continue
            self._cache_put(pair.topic_id, pair.answer_id, judgement)
            self.usage["pairs_submitted"] += 1
            results.append(judgement)
        return results

    # ------------------------------------------------------------------
    # Checkpointing (resumability)
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> list[dict]:
        if not self._checkpoint_path.exists():
            return []
        with open(self._checkpoint_path) as f:
            return json.load(f)

    def _save_checkpoint(self, records: list[dict]) -> None:
        with open(self._checkpoint_path, "w") as f:
            json.dump(records, f, indent=2)

    def _checkpoint_submit(self, handle: BatchHandle) -> None:
        records = self._load_checkpoint()
        records.append(
            {
                "chunk_id": handle.chunk_id,
                "job_name": handle.job_name,
                "pair_keys": handle.pair_keys,
                "status": "submitted",
            }
        )
        self._save_checkpoint(records)

    def _checkpoint_mark_completed(self, chunk_id: str) -> None:
        records = self._load_checkpoint()
        for r in records:
            if r["chunk_id"] == chunk_id:
                r["status"] = "completed"
        self._save_checkpoint(records)

    def pending_batches(self) -> list[BatchHandle]:
        """Chunks submitted but not yet fetched -- lets a caller resume after
        a crash without re-submitting.
        """
        records = self._load_checkpoint()
        return [
            BatchHandle(
                job_name=r["job_name"], chunk_id=r["chunk_id"], pair_keys=r["pair_keys"]
            )
            for r in records
            if r["status"] == "submitted"
        ]

    # ------------------------------------------------------------------
    # Batch path (default; the whole point)
    # ------------------------------------------------------------------

    def submit_batch(self, pairs: list[JudgePair]) -> list[BatchHandle]:
        """Submit unlabelled pairs as one or more Gemini Batch jobs, chunked
        and capped at batch_max_in_flight concurrent jobs. Cache-first: pairs
        already labelled are dropped before submission and never re-spent.
        """
        from google.genai import types

        _, remaining = self.partition_cached(pairs)
        if not remaining:
            return []

        client = self._get_genai_client()
        max_in_flight = max(1, self._config.execution.batch_max_in_flight)
        max_chunk_size = self._config.execution.max_batch_chunk_size

        # Chunks are as large as possible (up to max_chunk_size) so a small
        # input becomes ONE job, not one job per pair. batch_max_in_flight
        # caps how many of the resulting chunks are submitted in this call;
        # any remainder is left uncached/unsubmitted for a subsequent call
        # (run_judge_over_validation.py's resumable design already expects a
        # ~100K job to span multiple submissions).
        all_chunks = [
            remaining[i : i + max_chunk_size]
            for i in range(0, len(remaining), max_chunk_size)
        ]
        if len(all_chunks) > max_in_flight:
            logger.info(
                f"{len(all_chunks)} chunk(s) needed but batch_max_in_flight={max_in_flight}; "
                f"submitting {max_in_flight} now, remainder picked up on a later run"
            )
            all_chunks = all_chunks[:max_in_flight]

        handles: list[BatchHandle] = []
        for chunk in all_chunks:
            requests = []
            pair_keys = []
            for pair in chunk:
                key = _cache_key(
                    pair.topic_id,
                    pair.answer_id,
                    self._prompt_sha256,
                    self._config.judge.model,
                )
                pair_keys.append(key)
                requests.append(
                    types.InlinedRequest(
                        contents=self._build_contents(pair),
                        config=types.GenerateContentConfig(
                            temperature=self._config.judge.temperature,
                            top_p=self._config.judge.top_p,
                            max_output_tokens=self._config.judge.max_tokens,
                            response_mime_type="application/json",
                            response_schema=_RelevanceOutput,
                        ),
                        metadata={
                            "topic_id": pair.topic_id,
                            "answer_id": pair.answer_id,
                        },
                    )
                )
            job = client.batches.create(model=self._config.judge.model, src=requests)
            handle = BatchHandle(
                job_name=job.name, chunk_id=str(uuid.uuid4()), pair_keys=pair_keys
            )
            self._checkpoint_submit(handle)
            self.usage["pairs_submitted"] += len(chunk)
            handles.append(handle)
        return handles

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        client = self._get_genai_client()
        job = client.batches.get(name=handle.job_name)
        state_name = job.state.name if hasattr(job.state, "name") else str(job.state)
        return BatchStatus(
            job_name=handle.job_name,
            state=state_name,
            done=state_name in _TERMINAL_STATES,
        )

    def fetch_batch(self, handle: BatchHandle) -> list[RelevanceJudgement]:
        """Fetch a completed batch job's results, write each to cache
        immediately, and mark the chunk completed in the checkpoint so a
        re-run never re-submits it.
        """
        client = self._get_genai_client()
        job = client.batches.get(name=handle.job_name)
        inlined = (job.dest.inlined_responses if job.dest else None) or []

        results: list[RelevanceJudgement] = []
        for resp in inlined:
            meta = resp.metadata or {}
            topic_id = meta.get("topic_id", "")
            answer_id = meta.get("answer_id", "")
            timestamp = datetime.now(timezone.utc).isoformat()

            if resp.error is not None:
                judgement = RelevanceJudgement(
                    topic_id=topic_id,
                    answer_id=answer_id,
                    label=None,
                    parse_ok=False,
                    raw_response=str(resp.error),
                    prompt_sha256=self._prompt_sha256,
                    judge_model=self._config.judge.model,
                    source="batch",
                    timestamp=timestamp,
                )
            else:
                raw_text = resp.response.text if resp.response else ""
                usage_meta = getattr(resp.response, "usage_metadata", None)
                total_tokens = (
                    getattr(usage_meta, "total_token_count", None)
                    if usage_meta
                    else None
                )
                self._record_tokens(total_tokens)
                try:
                    parsed = _RelevanceOutput.model_validate_json(raw_text)
                    judgement = RelevanceJudgement(
                        topic_id=topic_id,
                        answer_id=answer_id,
                        label=parsed.label,
                        parse_ok=True,
                        raw_response=raw_text,
                        prompt_sha256=self._prompt_sha256,
                        judge_model=self._config.judge.model,
                        source="batch",
                        timestamp=timestamp,
                    )
                except (
                    Exception
                ) as e:  # noqa: BLE001 -- malformed model output -> parse_ok=False
                    judgement = RelevanceJudgement(
                        topic_id=topic_id,
                        answer_id=answer_id,
                        label=None,
                        parse_ok=False,
                        raw_response=raw_text or str(e),
                        prompt_sha256=self._prompt_sha256,
                        judge_model=self._config.judge.model,
                        source="batch",
                        timestamp=timestamp,
                    )
            self._cache_put(topic_id, answer_id, judgement)
            results.append(judgement)

        self._checkpoint_mark_completed(handle.chunk_id)
        return results
