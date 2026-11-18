"""The "run pi_old in N envs for T steps" part of PPO."""

import numpy as np
import torch

from .buffer import RolloutBuffer
from .tracking import EpisodeTracker


def collect_rollout(
    envs, model, buffer: RolloutBuffer, obs, mask, device, tracker: EpisodeTracker | None = None
):
    """Fill `buffer` with T steps from all N envs using the current (= old) policy.

    Returns the obs/mask to continue from and V(s_T) for bootstrapping GAE.
    """
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

    with torch.no_grad():
        _, last_value = model(to_t(obs))
    return obs, mask, last_value
