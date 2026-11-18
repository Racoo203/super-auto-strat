"""Launch one long PPO training run from a terminal, not a notebook, so it can run for hours
unattended and be resumed if it's interrupted (crash, OOM, laptop sleep, closed terminal).

    python -m agents.ppo.run --name final --updates 5000
    python -m agents.ppo.run --name final --updates 5000 --resume   # after a crash/interrupt

Everything it writes goes to runs/<name>/ (see train.py's module docstring for the layout).
"""

import argparse

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
    result = train(cfg, args.name, resume=args.resume)
    print(f"done: {result.run_name} -> {result.run_dir}  (best eval win rate: {result.best_eval_win_rate})")


if __name__ == "__main__":  # required on Windows: async envs spawn subprocesses
    main()
