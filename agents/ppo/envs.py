"""Parallel actors: N copies of the game in a gymnasium vector env.

Each env lives in its own process (AsyncVectorEnv) and all N are stepped in lockstep with one
batched forward pass: the synchronous A2C-style setup the PPO paper uses.

Async envs always use `spawn` (Windows' only option, and Linux's default `fork` is unsafe once the
parent has threads, e.g. TensorBoard's writer, or CUDA initialised), so the laptop and the Linux
server behave the same. In a *script* the entry point must sit under `if __name__ == "__main__":`.
In a notebook it just works.
"""

import gymnasium as gym

import sap.env  # noqa: F401  (registers SuperAutoStrat-v0)
from sap.env import ENV_ID


def make_vec_env(n_envs: int, async_envs: bool = True, seed: int | None = None, **env_kwargs):
    """Returns (envs, obs, action_mask) with obs (N, n_states) and mask (N, n_actions).

    SAME_STEP autoreset: when an env finishes, the obs/info it returns already belong to the
    fresh episode (and info["action_mask"] is the new episode's mask), which keeps the buffer
    convention in buffer.py simple. The default NEXT_STEP mode would instead ignore the action
    on the step after `done`, which you'd have to special-case.
    """
    envs = gym.make_vec(
        ENV_ID,
        num_envs=n_envs,
        vectorization_mode="async" if async_envs else "sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
                       **({"context": "spawn"} if async_envs else {})},
        **env_kwargs,
    )
    obs, info = envs.reset(seed=seed)
    return envs, obs, info["action_mask"]
