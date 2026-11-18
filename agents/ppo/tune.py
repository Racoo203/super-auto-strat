"""Hyperparameter search with Optuna. Every trial is a full tracked training run (see train.py).

    python -m agents.ppo.tune --trials 20 --updates 200
    optuna-dashboard sqlite:///runs/optuna.db

Trials go to study "ppo-tune" by default, separate from plain runs (study "ppo"), so pruning
compares tuning trials with each other. The objective is the best eval win rate.
"""

import argparse
from dataclasses import replace

import optuna

from .config import PPOConfig
from .experiments import default_storage
from .train import train


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
        return result.best_eval_win_rate or 0.0

    return objective


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--updates", type=int, default=200, help="n_updates per trial")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--study", default="ppo-tune")
    ap.add_argument("--storage", default=None, help="default sqlite:///runs/optuna.db")
    args = ap.parse_args(argv)

    base = PPOConfig(n_updates=args.updates, eval_every=args.eval_every)
    storage = args.storage or default_storage(base.runs_dir)
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True, direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.eval_every * 2),
    )
    study.optimize(make_objective(base, storage), n_trials=args.trials)
    print(f"best: {study.best_value:.2%} with {study.best_params}")


if __name__ == "__main__":  # required on Windows: async envs spawn subprocesses
    main()
