"""Hyperparameter search with Optuna. Every trial is a full tracked training run (see train.py).

    python -m agents.ppo.tune --trials 20 --updates 200
    optuna-dashboard sqlite:///runs/optuna.db

Trials go to study "ppo-tune" by default, separate from plain runs (study "ppo"), so pruning
compares tuning trials with each other. A trial's value is its best eval battle win rate vs
random teams (train.OBJECTIVE); eval/win_rate saturates near 1.0 and can't rank trials.

Rerunning the same command continues the study (load_if_exists) and runs --trials *more* trials.
A trial that crashes is marked FAIL and the study moves on to the next one; Ctrl+C / SIGTERM
(docker stop) marks the current trial FAIL and stops. Interrupted trials are not resumed.

Use a winner for a long run with `python -m agents.ppo.run --from-trial ppo-tune:<number> ...`,
ideally after re-checking the top few with several seeds (python -m agents.ppo.validate).
"""

import argparse
import logging
import signal
import sys
from dataclasses import replace

import optuna

from .config import PPOConfig
from .experiments import default_storage
from .train import train

# The PPOConfig fields this search tunes (suggest_config). run.py --from-trial copies exactly these.
TUNED_PARAMS = ("lr", "entropy_coef", "clip_eps", "gae_lambda", "n_epochs")

log = logging.getLogger(__name__)


def suggest_config(trial: optuna.Trial, base: PPOConfig) -> PPOConfig:
    # lr is log-scaled: experiments.LOG_SCALE_PARAMS must agree.
    return replace(
        base,
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        entropy_coef=trial.suggest_float("entropy_coef", 0.0, 0.05),
        clip_eps=trial.suggest_float("clip_eps", 0.1, 0.3),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        n_epochs=trial.suggest_int("n_epochs", 2, 8),
    )


def make_objective(base: PPOConfig, storage: str):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(trial, base)
        result = train(cfg, f"{trial.study.study_name}-{trial.number:03d}", storage=storage, trial=trial, prune=True)
        return result.best_objective or 0.0

    return objective


def run_trials(study: optuna.Study, objective, n_trials: int) -> None:
    """n_trials trials, one at a time, so that nothing short of Ctrl+C/SIGTERM ends the search.

    catch=(Exception,): a trial whose training raises is marked FAIL and the study goes on.
    The outer try covers what `catch` can't: Optuna's own tell/storage failing after a trial.
    """
    for i in range(n_trials):
        try:
            study.optimize(objective, n_trials=1, catch=(Exception,))
        except Exception as e:  # noqa: BLE001 -- KeyboardInterrupt (BaseException) still stops
            log.warning("tuning trial %d/%d hit a storage error, moving on: %s: %s", i + 1, n_trials, type(e).__name__, e)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20, help="trials to run now (added to the study's existing ones)")
    ap.add_argument("--updates", type=int, default=200, help="n_updates per trial")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=PPOConfig.eval_episodes, help="games per evaluation")
    ap.add_argument("--n-envs", type=int, default=PPOConfig.n_envs, help="env worker processes (~ physical cores)")
    ap.add_argument("--seed", type=int, default=PPOConfig.seed)
    ap.add_argument("--study", default="ppo-tune")
    ap.add_argument("--storage", default=None, help="default sqlite:///runs/optuna.db")
    args = ap.parse_args(argv)

    base = PPOConfig(
        n_updates=args.updates, eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        n_envs=args.n_envs, seed=args.seed,
    )
    storage = args.storage or default_storage(base.runs_dir)
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True, direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.eval_every * 2),
    )
    from .run import _raise_keyboard_interrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)  # docker stop: mark the trial FAIL, save latest.pt
    run_trials(study, make_objective(base, storage), args.trials)
    try:
        print(f"best: trial #{study.best_trial.number}, {study.best_value:.2%} with {study.best_params}")
    except ValueError:  # no completed trial yet
        print("no completed trials yet")


if __name__ == "__main__":  # required on Windows: async envs spawn subprocesses
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; the running trial is marked FAIL. Rerun the same command to continue the study.", file=sys.stderr)
        sys.exit(130)
