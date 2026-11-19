"""Re-check the best tuning trials with longer, multi-seed runs before choosing one for the long run.

Tuning trials are single-seed and short (200 updates), so the top of the leaderboard is partly luck.
This reruns the top --top COMPLETE trials of --study with each of --seeds for --updates updates,
then prints, per config, the objective (eval/battle_win_rate) averaged over seeds.

    python -m agents.ppo.validate --study ppo-tune --top 3 --seeds 1 2 3 --updates 500
    python -m agents.ppo.validate ... --baseline     # also the default PPOConfig, for reference
    python -m agents.ppo.validate ... --dry-run      # list the runs, train nothing

Runs are named val-<study>-t<NNN>-s<seed> (val-<study>-default-s<seed> for --baseline), go to
Optuna study "ppo-validate", and resume: rerun the same command after an interruption and finished
runs are skipped, unfinished ones continue. The summary is read from each run's metrics.jsonl.
Launch the winner with `python -m agents.ppo.run --from-trial <study>:<NNN> ...` (EXTRA_ARGS in .env).
"""

import argparse
import logging
import signal
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna

from .config import PPOConfig
from .experiments import default_storage
from .train import OBJECTIVE, train
from .tune import TUNED_PARAMS

VALIDATE_STUDY = "ppo-validate"
log = logging.getLogger(__name__)


def top_trials(study_name: str, storage: str, k: int) -> list[optuna.trial.FrozenTrial]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    done = [t for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)) if t.value is not None]
    return sorted(done, key=lambda t: t.value, reverse=True)[:k]


def plan(study_name: str, storage: str, top: int, seeds: list[int], base: PPOConfig, baseline: bool):
    """[(label, source, run name, cfg)]: one entry per (config, seed)."""
    configs = [(f"t{t.number:03d}", f"{study_name}:{t.number}", {k: t.params[k] for k in TUNED_PARAMS if k in t.params})
               for t in top_trials(study_name, storage, top)]
    if baseline:
        configs.append(("default", "default", {}))
    return [(label, src, f"val-{study_name}-{label}-s{seed}", replace(base, seed=seed, **params))
            for label, src, params in configs for seed in seeds]


def objective_curve(run_dir: Path) -> list[float]:
    """The run's OBJECTIVE at each eval, in order ([] if it never evaluated)."""
    from .experiments import load_metrics

    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    m = load_metrics(run_dir)
    if OBJECTIVE not in m.columns:
        return []
    return m[OBJECTIVE].dropna().tolist()


def summarize(runs, runs_dir: str, last_k: int = 3) -> list[dict]:
    """Per config: mean/std over seeds of the best eval and of the mean of the last `last_k` evals
    (less luck-driven than the best), sorted by the latter."""
    by_label: dict[str, dict] = {}
    for label, src, name, _ in runs:
        curve = objective_curve(Path(runs_dir) / name)
        row = by_label.setdefault(label, {"config": label, "source": src, "best": [], "last": []})
        if curve:
            row["best"].append(max(curve))
            row["last"].append(float(np.mean(curve[-last_k:])))
    out = []
    for row in by_label.values():
        n = len(row["last"])
        out.append({
            "config": row["config"], "source": row["source"], "seeds_done": n,
            "last_mean": float(np.mean(row["last"])) if n else float("nan"),
            "last_std": float(np.std(row["last"])) if n else float("nan"),
            "best_mean": float(np.mean(row["best"])) if n else float("nan"),
        })
    return sorted(out, key=lambda r: (r["seeds_done"] == 0, -r["last_mean"] if r["seeds_done"] else 0.0))


def validate(study_name: str, storage: str, top: int, seeds: list[int], base: PPOConfig, *,
             baseline: bool = False, dry_run: bool = False, log_fn=print) -> list[dict]:
    from .run import _completed_update

    runs = plan(study_name, storage, top, seeds, base, baseline)
    if not runs:
        raise SystemExit(f"study {study_name!r} has no COMPLETE trials to validate")
    for i, (label, src, name, cfg) in enumerate(runs, 1):
        status = "done" if _completed_update(Path(cfg.runs_dir) / name) >= cfg.n_updates - 1 else "todo"
        if log_fn:
            log_fn(f"[{i}/{len(runs)}] {name} ({src}, seed {cfg.seed}): {status}")
        if dry_run or status == "done":
            continue
        try:
            train(cfg, name, storage=storage, resume=True, study_name=VALIDATE_STUDY,
                  user_attrs={"params_from_trial": src, "validate_of": study_name}, log_fn=log_fn,
                  log_every=max(1, cfg.eval_every))
        except Exception as e:  # noqa: BLE001 -- one broken run shouldn't stop the others
            log.warning("validation run %s crashed, moving on: %s: %s", name, type(e).__name__, e)
    table = summarize(runs, base.runs_dir)
    if log_fn:
        log_fn(f"\n{OBJECTIVE}: mean of each run's last 3 evals, then mean ± std over seeds")
        for r in table:
            log_fn(f"  {r['config']:>8}  {r['source']:<16} {r['last_mean']:.3f} ± {r['last_std']:.3f}"
                   f"   (best eval {r['best_mean']:.3f}, {r['seeds_done']}/{len(seeds)} seeds)")
        top_row = next((r for r in table if r["source"] != "default" and r["seeds_done"]), None)
        if top_row:
            log_fn(f"\nlaunch it: EXTRA_ARGS=--from-trial {top_row['source']}   (in .env, with a new RUN_NAME)")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="ppo-tune", help="the tuning study to take the top trials from")
    ap.add_argument("--top", type=int, default=3, help="how many of its best COMPLETE trials")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="tuning trials used seed 0")
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=PPOConfig.eval_every)
    ap.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes)
    ap.add_argument("--n-envs", type=int, default=PPOConfig.n_envs)
    ap.add_argument("--baseline", action="store_true", help="also run the default PPOConfig")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and the summary so far; train nothing")
    ap.add_argument("--storage", default=None, help="default sqlite:///runs/optuna.db")
    args = ap.parse_args(argv)

    base = PPOConfig(n_updates=args.updates, eval_every=args.eval_every, eval_episodes=args.eval_episodes, n_envs=args.n_envs)
    storage = args.storage or default_storage(base.runs_dir)
    from .run import _raise_keyboard_interrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    validate(args.study, storage, args.top, args.seeds, base, baseline=args.baseline, dry_run=args.dry_run)


if __name__ == "__main__":  # required: async envs spawn subprocesses (see envs.py)
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to continue (finished runs are skipped)", file=sys.stderr)
        sys.exit(130)
