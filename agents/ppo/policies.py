"""Several policies sharing one vector env: seat (env column) i is played by policy seat_policy[i].

    n_policies = 1   shared-policy self-play (every seat, one network)
    n_policies = 2   two agents trained side by side (seats alternate 0, 1, 0, 1, ...)
    n_policies = P   a league

Acting is one batched forward per policy on that policy's rows, scattered back into (N,) tensors,
so P=1 costs exactly what a bare ActorCritic does.
"""

import numpy as np
import torch

from .network import ActorCritic


def seat_policies(n_envs: int, n_policies: int) -> np.ndarray:
    """Round-robin assignment of env columns to policy ids: (N,) int array."""
    if not 1 <= n_policies <= n_envs:
        raise ValueError(f"need 1 <= n_policies ({n_policies}) <= n_envs ({n_envs})")
    return np.arange(n_envs) % n_policies


class PolicySet:
    def __init__(self, models: list[ActorCritic], seat_policy: np.ndarray, optimizers: list | None = None):
        self.models = list(models)
        self.optimizers = list(optimizers) if optimizers is not None else [None] * len(self.models)
        self.seat_policy = np.asarray(seat_policy)
        if self.seat_policy.max() >= len(self.models):
            raise ValueError("seat_policy refers to a policy that doesn't exist")
        self._rows = [torch.as_tensor(np.flatnonzero(self.seat_policy == p)) for p in range(len(self.models))]

    @classmethod
    def single(cls, model: ActorCritic, n_envs: int) -> "PolicySet":
        return cls([model], np.zeros(n_envs, dtype=int))

    def __len__(self) -> int:
        return len(self.models)

    def seats(self, policy: int) -> np.ndarray:
        """Env columns played by `policy`."""
        return self._rows[policy].numpy()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        """Like ActorCritic.act for all N seats: (action, log_prob, value), each (N,)."""
        if len(self.models) == 1:
            return self.models[0].act(obs, mask)
        n = obs.shape[0]
        action = torch.zeros(n, dtype=torch.long, device=obs.device)
        log_prob = torch.zeros(n, device=obs.device)
        value = torch.zeros(n, device=obs.device)
        for model, rows in zip(self.models, self._rows):
            rows = rows.to(obs.device)
            a, lp, v = model.act(obs[rows], mask[rows])
            action[rows], log_prob[rows], value[rows] = a, lp, v
        return action, log_prob, value

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) of every seat under its own policy: (N,)."""
        if len(self.models) == 1:
            return self.models[0](obs)[1]
        out = torch.zeros(obs.shape[0], device=obs.device)
        for model, rows in zip(self.models, self._rows):
            rows = rows.to(obs.device)
            out[rows] = model(obs[rows])[1]
        return out
