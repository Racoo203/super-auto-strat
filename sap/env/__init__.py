"""Gymnasium environment for SuperAutoStrat.

    import gymnasium as gym
    import sap.env  # registers "SuperAutoStrat-v0"
    env = gym.make("SuperAutoStrat-v0")
    obs, info = env.reset(seed=0)
    action = env.action_space.sample(mask=info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym

from .actions import ACTIONS, apply_action, describe_action
from .masking import action_mask, is_legal, undoes_last
from .observation import OBS_HIGH, OBS_SIZE, encode_observation, observation_layout
from .sap_env import SapEnv

ENV_ID = "SuperAutoStrat-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="sap.env:SapEnv")

__all__ = [
    "ACTIONS", "ENV_ID", "OBS_HIGH", "OBS_SIZE", "SapEnv", "action_mask", "apply_action",
    "describe_action", "encode_observation", "is_legal", "observation_layout", "undoes_last",
]
