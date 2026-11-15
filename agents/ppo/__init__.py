"""PPO, one concept per module:

  config.py      hyperparameters, device
  network.py     ActorCritic (policy logits + V(s)), masked action sampling
  envs.py        N parallel environments (vector env)
  rollout.py     run pi_old for T steps into the buffer
  buffer.py      (T, N) rollout storage + minibatching
  advantages.py  GAE (pure function)
  losses.py      one function per loss term
  tracking.py    episode stats for logging
"""

from .advantages import compute_gae
from .buffer import RolloutBuffer
from .config import PPOConfig, get_device
from .envs import make_vec_env
from .losses import entropy_bonus, ppo_clip_loss, value_loss
from .network import ActorCritic
from .rollout import collect_rollout
from .tracking import EpisodeTracker

__all__ = [
    "ActorCritic", "EpisodeTracker", "PPOConfig", "RolloutBuffer", "collect_rollout",
    "compute_gae", "entropy_bonus", "get_device", "make_vec_env", "ppo_clip_loss", "value_loss",
]
