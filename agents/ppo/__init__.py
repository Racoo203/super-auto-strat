"""PPO, one concept per module:

  config.py       hyperparameters, device
  network.py      ActorCritic (policy logits + V(s)), masked action sampling
  envs.py         N parallel environments (vector env)
  rollout.py      run pi_old for T steps into the buffer
  buffer.py       (T, N) rollout storage + minibatching
  advantages.py   GAE (pure function)
  losses.py       one function per loss term
  tracking.py     episode stats (return, length, win/wins) for logging

Training and observability:

  train.py        train(cfg, run_name): the full loop + metrics, evals, checkpoints
  experiments.py  Run: Optuna trial (runs/optuna.db) + TensorBoard + metrics.jsonl; load_runs()
  checkpoints.py  save_checkpoint / load_checkpoint
  evaluate.py     play_episode / evaluate: full games with every decision recorded (replays)
  replay.py       summarize / step through recorded games   (python -m agents.ppo.replay)
  watch.py        watch a policy play live, as text          (python -m agents.ppo.watch)
  tune.py         Optuna hyperparameter search               (python -m agents.ppo.tune)
"""

from .advantages import compute_gae
from .buffer import RolloutBuffer
from .checkpoints import load_checkpoint, save_checkpoint
from .config import PPOConfig, get_device
from .envs import make_vec_env
from .evaluate import EpisodeRecord, evaluate, load_replays, play_episode, save_replays
from .experiments import Run, load_metrics, load_runs
from .losses import entropy_bonus, ppo_clip_loss, value_loss
from .network import ActorCritic
from .rollout import collect_rollout
from .tracking import EpisodeStats, EpisodeTracker
from .train import TrainResult, train

__all__ = [
    "ActorCritic", "EpisodeRecord", "EpisodeStats", "EpisodeTracker", "PPOConfig", "RolloutBuffer",
    "Run", "TrainResult", "collect_rollout", "compute_gae", "entropy_bonus", "evaluate",
    "get_device", "load_checkpoint", "load_metrics", "load_replays", "load_runs", "make_vec_env",
    "play_episode", "ppo_clip_loss", "save_checkpoint", "save_replays", "train", "value_loss",
]
