"""ExperimentLogger: the single W&B entry point for Track V.

Mirrors how the judge API lives in one JudgeClient -- experiment scripts call
ExperimentLogger, never wandb.* directly. This is the one exception to the
repo's existing pattern of direct wandb.init/log/finish calls scattered
across each script (experiments/run_c0_1.py, run_task1_experiment.py, ...);
it does not touch or refactor those.

W&B is logging only, never a source of truth: results still land in
results/track_v/; if wandb is down, or wandb.mode == "disabled", the pipeline
must still complete and write local files, unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path

from multirag.config.judge_config import WandbConfig

logger = logging.getLogger(__name__)


class ExperimentLogger:
    def __init__(self, wandb_config: WandbConfig):
        self._config = wandb_config
        self._run = None

    def start_run(
        self,
        config: dict,
        run_name: str,
        extra_config: dict | None = None,
        run_id: str | None = None,
    ):
        """wandb.init(...). `config` should be the same manifest dict written
        to the result file's meta row -- wandb.config must not drift from it.

        Pass `run_id` (the previous call's `.run_id`, persisted in the meta
        row of the upstream result file) to resume the SAME run across
        separate script invocations -- run_judge_over_validation.py,
        build_paired_labels.py, agreement_analysis.py, and report.py all log
        into one run for V, per the spec ("keep the same run open through
        the agreement step").
        """
        if not self._config.enabled:
            return None
        import wandb

        merged_config = dict(config)
        if extra_config:
            merged_config.update(extra_config)

        init_kwargs = dict(
            entity=self._config.entity,
            project=self._config.project,
            group=self._config.group,
            job_type=self._config.job_type,
            tags=list(self._config.tags),
            mode=self._config.mode,
            name=run_name,
            config=merged_config,
        )
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        self._run = wandb.init(**init_kwargs)
        return self._run

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run is not None else None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self._run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def log_table(self, name: str, rows) -> None:
        """rows: list[dict] or a pandas DataFrame."""
        if self._run is None:
            return
        import pandas as pd

        import wandb

        df = rows if hasattr(rows, "columns") else pd.DataFrame(rows)
        wandb.log({name: wandb.Table(dataframe=df)})

    def log_confusion_matrix(
        self,
        name: str,
        y_true: list[int],
        y_pred: list[int],
        labels: list[int],
    ) -> None:
        """Logs both a queryable wandb.Table of raw counts and the
        wandb.plot.confusion_matrix UI widget, for the same (y_true, y_pred).
        """
        if self._run is None:
            return
        import wandb

        label_index = {label: i for i, label in enumerate(labels)}
        counts = [[0] * len(labels) for _ in labels]
        for t, p in zip(y_true, y_pred):
            counts[label_index[t]][label_index[p]] += 1

        rows = [
            {
                "human_label": t,
                "judge_label": p,
                "count": counts[label_index[t]][label_index[p]],
            }
            for t in labels
            for p in labels
        ]
        self.log_table(f"{name}_counts", rows)

        wandb.log(
            {
                f"{name}_plot": wandb.plot.confusion_matrix(
                    preds=[label_index[p] for p in y_pred],
                    y_true=[label_index[t] for t in y_true],
                    class_names=[str(label) for label in labels],
                )
            }
        )

    def log_artifact(
        self, path: str | Path, artifact_type: str, name: str | None = None
    ) -> None:
        if self._run is None:
            return
        import wandb

        path = Path(path)
        artifact = wandb.Artifact(name or path.stem, type=artifact_type)
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def set_summary(self, key: str, value) -> None:
        if self._run is None:
            return
        self._run.summary[key] = value

    def finish(self) -> None:
        if self._run is None:
            return
        import wandb

        wandb.finish()
        self._run = None
