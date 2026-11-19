"""Launch one long PPO training run from a terminal, not a notebook, so it can run for hours
unattended and be resumed if it's interrupted (crash, OOM, laptop sleep, closed terminal).

    python -m agents.ppo.run --name final --updates 5000
    python -m agents.ppo.run --name final --updates 5000 --resume   # after a crash/interrupt

Everything it writes goes to runs/<name>/ (see train.py's module docstring for the layout).

Server use (see deploy/README.md): always pass --resume. With no checkpoint yet it starts fresh,
after a crash/reboot it continues, and once the run has reached --updates it exits immediately, so
a supervisor (Docker restart policy, systemd) can relaunch the same command blindly.
SIGTERM (docker stop, systemctl stop, shutdown) is handled like Ctrl+C: latest.pt is saved first.
"""

import argparse
import signal
import sys
from pathlib import Path

import torch

from .config import PPOConfig
from .train import train


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="run name (-> runs/<name>/); default: a timestamp")
    ap.add_argument("--updates", type=int, default=PPOConfig.n_updates)
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/checkpoints/latest*.pt")
    ap.add_argument("--lr", type=float, default=PPOConfig.lr)
    ap.add_argument("--n-envs", type=int, default=PPOConfig.n_envs)
    ap.add_argument("--n-policies", type=int, default=PPOConfig.n_policies)
    ap.add_argument("--opponent", choices=["ghosts", "random"], default=PPOConfig.opponent)
    ap.add_argument("--seed", type=int, default=PPOConfig.seed)
    ap.add_argument("--eval-every", type=int, default=PPOConfig.eval_every)
    ap.add_argument("--checkpoint-every", type=int, default=PPOConfig.checkpoint_every)
    ap.add_argument("--keep-checkpoints", type=int, default=PPOConfig.keep_checkpoints)
    ap.add_argument("--runs-dir", default=PPOConfig.runs_dir)
    args = ap.parse_args(argv)

    cfg = PPOConfig(
        n_updates=args.updates, lr=args.lr, n_envs=args.n_envs, n_policies=args.n_policies,
        opponent=args.opponent, seed=args.seed, eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every, keep_checkpoints=args.keep_checkpoints,
        runs_dir=args.runs_dir,
    )
    if args.resume and args.name and _completed_update(Path(cfg.runs_dir) / args.name) >= cfg.n_updates - 1:
        print(f"run '{args.name}' already reached {cfg.n_updates} updates; nothing to do")
        return
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    result = train(cfg, args.name, resume=args.resume)
    print(f"done: {result.run_name} -> {result.run_dir}  (best eval win rate: {result.best_eval_win_rate})")


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
