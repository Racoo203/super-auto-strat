"""Hyperparameters and device selection, in one place so the notebook and a script share them."""

from dataclasses import dataclass

import torch


@dataclass
class PPOConfig:
    # --- parallel actors (the "for actor = 1..N" loop in the paper) ---
    n_envs: int = 8          # N: number of environments stepped in lockstep
    n_steps: int = 128       # T: steps collected per env per update -> batch = N * T
    async_envs: bool = True  # True: one process per env (real parallelism). False: in-process.

    # --- returns / advantages ---
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # --- optimisation ---
    lr: float = 2.5e-4
    n_epochs: int = 4        # K: passes over the collected batch
    minibatch_size: int = 256  # M <= N * T
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5

    # --- loss weights (paper: L = L_clip - c1 * L_vf + c2 * S) ---
    value_coef: float = 0.5    # c1
    entropy_coef: float = 0.01  # c2

    n_updates: int = 1000

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps


def get_device() -> torch.device:
    """CUDA/MPS/etc. if available (your PC), otherwise CPU (your laptop)."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")
