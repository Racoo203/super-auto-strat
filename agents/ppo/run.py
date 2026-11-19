"""Launch one long PPO training run from a terminal, not a notebook, so it can run for hours
unattended and be resumed if it's interrupted (crash, OOM, laptop sleep, closed terminal).

    python -m agents.ppo.run --name final --updates 5000
    python -m agents.ppo.run --name final --updates 5000 --resume   # after a crash/interrupt
    python -m agents.ppo.run --name final --updates 5000 --from-trial ppo-tune:7   # tuned params
    python -m agents.ppo.run --name final --params-json '{"lr": 1e-4, "n_epochs": 6}'

Everything it writes goes to runs/<name>/ (see train.py's module docstring for the layout).

Hyperparameters, lowest to highest precedence: PPOConfig defaults, then --from-trial (the tuned
params of that Optuna trial: tune.TUNED_PARAMS) and --params-json (any PPOConfig fields, inline
JSON or a .json file path), then explicit flags such as --lr. So `--from-trial ppo-tune:7 --lr 1e-4`
takes trial 7's params except lr.

Server use (see deploy/README.md): always pass --resume. With no checkpoint yet it starts fresh,
after a crash/reboot it continues, and once the run has reached --updates it exits immediately, so
a supervisor (Docker restart policy, systemd) can relaunch the same command blindly. Keep the
hyperparameter flags the same across restarts of one run (they are re-read on every launch).
SIGTERM (docker stop, systemctl stop, shutdown) is handled like Ctrl+C: latest.pt is saved first.
"""

import argparse
import json
import signal
import sys
from dataclasses import fields
from pathlib import Path

import torch

from .config import PPOConfig
from .train import train

# flag -> (PPOConfig field, type). Each defaults to None ("not given") so it can override
# --from-trial / --params-json only when passed explicitly.
CONFIG_FLAGS = {
    "--updates": ("n_updates", int),
    "--lr": ("lr", float),
    "--entropy-coef": ("entropy_coef", float),
    "--clip-eps": ("clip_eps", float),
    "--gae-lambda": ("gae_lambda", float),
    "--n-epochs": ("n_epochs", int),
    "--n-envs": ("n_envs", int),
    "--n-policies": ("n_policies", int),
    "--seed": ("seed", int),
    "--eval-every": ("eval_every", int),
    "--eval-episodes": ("eval_episodes", int),
    "--checkpoint-every": ("checkpoint_every", int),
    "--keep-checkpoints": ("keep_checkpoints", int),
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="run name (-> runs/<name>/); default: a timestamp")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/checkpoints/latest*.pt")
    for flag, (field, typ) in CONFIG_FLAGS.items():
        ap.add_argument(flag, type=typ, default=None, help=f"PPOConfig.{field} (default {getattr(PPOConfig, field)})")
    ap.add_argument("--opponent", choices=["ghosts", "random"], default=None, help=f"default {PPOConfig.opponent}")
    ap.add_argument("--runs-dir", default=None, help=f"default {PPOConfig.runs_dir}")
    ap.add_argument("--from-trial", metavar="STUDY:NUMBER", help="use the tuned params of this Optuna trial, e.g. ppo-tune:7")
    ap.add_argument("--params-json", metavar="JSON|FILE", help="PPOConfig fields as inline JSON or a .json file")
    ap.add_argument("--storage", default=None, help="Optuna storage for --from-trial (default sqlite:///<runs-dir>/optuna.db)")
    return ap


def trial_params(spec: str, storage: str) -> dict:
    """The tuned hyperparameters (tune.TUNED_PARAMS) of trial `study:number`."""
    import optuna

    from .tune import TUNED_PARAMS

    study_name, sep, number = spec.rpartition(":")
    if not sep or not study_name or not number.isdigit():
        raise SystemExit(f"--from-trial expects STUDY:NUMBER (e.g. ppo-tune:7), got {spec!r}")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        raise SystemExit(f"--from-trial: no study {study_name!r} in {storage}") from None
    matches = [t for t in study.get_trials(deepcopy=False) if t.number == int(number)]
    if not matches:
        raise SystemExit(f"--from-trial: study {study_name!r} has no trial #{number}")
    t = matches[0]
    params = {k: t.params[k] for k in TUNED_PARAMS if k in t.params}
    if not params:
        raise SystemExit(f"--from-trial: trial {spec} has none of the tuned params {TUNED_PARAMS}")
    value = "n/a" if t.value is None else f"{t.value:.3f}"
    print(f"hyperparameters from {spec} ({t.state.name}, value {value}): {params}")
    return params


def json_params(spec: str) -> dict:
    """PPOConfig fields from inline JSON ('{...}') or a JSON file."""
    text = spec if spec.lstrip().startswith("{") else Path(spec).read_text(encoding="utf-8")
    params = json.loads(text)
    if not isinstance(params, dict):
        raise SystemExit("--params-json must be a JSON object")
    unknown = set(params) - {f.name for f in fields(PPOConfig)}
    if unknown:
        raise SystemExit(f"--params-json: not PPOConfig fields: {sorted(unknown)}")
    return params


def build_config(args: argparse.Namespace) -> PPOConfig:
    """Defaults < --from-trial < --params-json < explicit flags."""
    from .experiments import default_storage

    runs_dir = args.runs_dir or PPOConfig.runs_dir
    values: dict = {}
    if args.from_trial:
        values.update(trial_params(args.from_trial, args.storage or default_storage(runs_dir)))
    if args.params_json:
        values.update(json_params(args.params_json))
    for flag, (field, _) in CONFIG_FLAGS.items():
        v = getattr(args, flag.lstrip("-").replace("-", "_"))
        if v is not None:
            values[field] = v
    if args.opponent is not None:
        values["opponent"] = args.opponent
    if args.runs_dir is not None:
        values["runs_dir"] = args.runs_dir
    return PPOConfig.from_dict({**PPOConfig().to_dict(), **values})


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    cfg = build_config(args)
    if args.resume and args.name and _completed_update(Path(cfg.runs_dir) / args.name) >= cfg.n_updates - 1:
        print(f"run '{args.name}' already reached {cfg.n_updates} updates; nothing to do")
        return
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    attrs = {k: v for k, v in (("params_from_trial", args.from_trial), ("params_json", args.params_json)) if v}
    result = train(cfg, args.name, resume=args.resume, user_attrs=attrs)
    print(f"done: {result.run_name} -> {result.run_dir}  (best eval battle win rate: {result.best_objective})")


def _completed_update(run_dir: Path) -> int:
    """Last update saved in run_dir's latest.pt, or -1 if there is none."""
    path = run_dir / "checkpoints" / "latest.pt"
    if not path.exists():
        return -1
    return int(torch.load(path, map_location="cpu", weights_only=False)["update"])


def _raise_keyboard_interrupt(signum, frame) -> None:
    # Python's default SIGTERM kills the process without unwinding, skipping train()'s `finally`
    # (final latest.pt save, Optuna trial marked FAIL). Turn it into a normal Ctrl+C instead.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # a second SIGTERM kills outright
    raise KeyboardInterrupt(f"received signal {signum}")


if __name__ == "__main__":  # required: async envs spawn subprocesses (see envs.py)
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; latest checkpoint saved, relaunch with --resume", file=sys.stderr)
        sys.exit(130)  # non-zero, so a restart-on-failure supervisor brings it back after a reboot
