"""The "run pi_old in N envs for T steps" part of PPO."""

import numpy as np
import torch

from .buffer import RolloutBuffer
from .network import ActorCritic
from .policies import PolicySet
from .tracking import EpisodeTracker


def collect_rollout(
    envs, model, buffer: RolloutBuffer, obs, mask, device, tracker: EpisodeTracker | None = None
):
    """Fill `buffer` with T steps from all N envs using the current (= old) policy.

    `model`: an ActorCritic (plays every env) or a PolicySet (env i played by seat_policy[i]).
    Each buffer column is one seat's own trajectory with its own policy's values, so GAE needs no
    change when several policies share the batch.

    Returns the obs/mask to continue from and V(s_T) for bootstrapping GAE.
    """
    if isinstance(model, ActorCritic):
        model = PolicySet.single(model, buffer.n_envs)
    buffer.reset()

    def to_t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device)

    while not buffer.full:
        obs_t, mask_t = to_t(obs), to_t(mask, torch.bool)
        action, log_prob, value = model.act(obs_t, mask_t)

        obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        buffer.add(obs_t, mask_t, action, log_prob, value, to_t(reward), to_t(done))
        mask = info["action_mask"]
        if tracker is not None:
            tracker.update(reward, done, info)

    last_value = model.value(to_t(obs))
    return obs, mask, last_value
