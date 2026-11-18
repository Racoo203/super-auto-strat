"""Run tracking: one Optuna trial per training run, plus TensorBoard curves and a metrics file.

    with Run(cfg, "baseline") as run:
        run.log(update, {"train/win_rate": 0.4, "loss/policy": 0.01})
        run.report(update, eval_win_rate)   # the trial's objective (best value wins)

Where things go (runs_dir defaults to cfg.runs_dir = "runs"):
    runs/optuna.db                  every run as a trial of study "ppo" (params, value, attrs)
    runs/<run>/tb/                  TensorBoard event files
    runs/<run>/metrics.jsonl        one JSON line per update (pandas.read_json(lines=True))
    runs/<run>/config.json          the PPOConfig
    runs/<run>/checkpoints/, replays/   written by train.py

View:
    optuna-dashboard sqlite:///runs/optuna.db
    tensorboard --logdir runs
"""

import json
import subprocess
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import optuna

from .config import PPOConfig

STUDY_NAME = "ppo"
LOG_SCALE_PARAMS = {"lr"}  # must match how tune.py suggests them (Optuna checks the log flag)


def default_storage(runs_dir: str | Path = "runs") -> str:
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{Path(runs_dir).as_posix()}/optuna.db"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def record_config(trial: optuna.Trial, cfg: PPOConfig) -> None:
    """Store every numeric config field as a (fixed, single-value) Optuna param, the rest as attrs.

    Params already suggested by a tuner (tune.py) are left alone. Point distributions (low == high)
    keep the dashboard's parameter plots numeric.
    """
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name in trial.params:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            trial.set_user_attr(f"cfg.{f.name}", list(v) if isinstance(v, tuple) else v)
        elif isinstance(v, int):
            trial.suggest_int(f.name, v, v)
        else:
            trial.suggest_float(f.name, v, v, log=f.name in LOG_SCALE_PARAMS)


class Run:
    """Context manager around one training run. See the module docstring.

    trial=None: this run creates (and finishes) its own trial in `study_name`.
    trial given (tuning): the caller's study.optimize owns it; `report` can raise TrialPruned.
    """

    def __init__(
        self,
        cfg: PPOConfig,
        name: str | None = None,
        *,
        storage: str | None = None,
        study_name: str = STUDY_NAME,
        trial: optuna.Trial | None = None,
        tensorboard: bool = True,
    ):
        self.cfg = cfg
        self.name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(cfg.runs_dir) / self.name
        self.storage = storage or default_storage(cfg.runs_dir)
        self.study_name = study_name
        self.trial = trial
        self._owns_trial = trial is None
        self.use_tb = tensorboard
        self.best_value: float | None = None
        self.last_train_win_rate: float | None = None
        self.writer = None
        self._metrics_file = None
        self.study: optuna.Study | None = None

    # --- context manager -------------------------------------------------------
    def __enter__(self) -> "Run":
        if self.dir.exists() and any(self.dir.iterdir()):
            self.name = f"{self.name}-{datetime.now().strftime('%H%M%S')}"
            self.dir = Path(self.cfg.runs_dir) / self.name
        for sub in ("checkpoints", "replays"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))

        if self._owns_trial:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(
                study_name=self.study_name, storage=self.storage, load_if_exists=True, direction="maximize",
            )
            self.trial = self.study.ask()
        record_config(self.trial, self.cfg)
        for k, v in {
            "run_name": self.name, "run_dir": str(self.dir), "git_commit": _git_commit(),
            "started": datetime.now().isoformat(timespec="seconds"),
        }.items():
            self.trial.set_user_attr(k, v)

        if self.use_tb:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.dir / "tb"))
        self._metrics_file = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.trial.set_user_attr("duration_s", round(time.time() - self._t0, 1))
            if exc_type is not None:
                self.trial.set_user_attr("error", f"{exc_type.__name__}: {exc}"[:500])
            if self._owns_trial:
                if exc_type is None:
                    self.study.tell(self.trial, self.objective_value())
                elif issubclass(exc_type, optuna.TrialPruned):
                    self.study.tell(self.trial, state=optuna.trial.TrialState.PRUNED)
                else:  # crash or KeyboardInterrupt: keep the record, marked FAIL
                    self.study.tell(self.trial, state=optuna.trial.TrialState.FAIL)
        finally:
            if self.writer:
                self.writer.close()
            if self._metrics_file:
                self._metrics_file.close()
        return False  # never swallow the exception

    # --- logging -----------------------------------------------------------------
    def objective_value(self) -> float:
        """Best eval win rate seen; falls back to the last training win rate, then 0."""
        for v in (self.best_value, self.last_train_win_rate):
            if v is not None:
                return float(v)
        return 0.0

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if "train/win_rate" in metrics:
            self.last_train_win_rate = metrics["train/win_rate"]
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
            self.writer.flush()
        self._metrics_file.write(json.dumps({"update": step, "time": round(time.time() - self._t0, 2), **metrics}) + "\n")
        self._metrics_file.flush()

    def report(self, step: int, value: float, prune: bool = False) -> bool:
        """Report the objective (eval win rate). Returns True if this is the best so far.
        With prune=True (tuning) raises optuna.TrialPruned when the pruner says so."""
        self.trial.report(value, step)
        best = self.best_value is None or value > self.best_value
        if best:
            self.best_value = value
            self.trial.set_user_attr("best_update", step)
        if prune and self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned at update {step}")
        return best


def load_runs(storage: str | None = None, study_name: str = STUDY_NAME, runs_dir: str = "runs"):
    """All runs as a pandas DataFrame: value (best eval win rate), state, params_*, user_attrs_*."""
    study = optuna.load_study(study_name=study_name, storage=storage or default_storage(runs_dir))
    return study.trials_dataframe()


def load_metrics(run_dir: str | Path):
    """A run's per-update metrics (metrics.jsonl) as a DataFrame indexed by update."""
    import pandas as pd

    return pd.read_json(Path(run_dir) / "metrics.jsonl", lines=True).set_index("update")
