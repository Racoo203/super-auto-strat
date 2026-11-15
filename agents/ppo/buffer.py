"""Rollout storage for N parallel envs x T steps, laid out as (T, N, ...) tensors.

Storage only: advantages/returns are computed by `advantages.compute_gae` and written into
`buffer.advantages` / `buffer.returns` before calling `minibatches`.

Convention (matters for GAE): row t holds the obs s_t the agent saw, the action a_t it took,
and what came back from that step: r_t and done_t. If done_t is True, the obs in row t+1 belongs
to a *new* episode (the vector env auto-resets), so you must not bootstrap V(s_{t+1}) into t.
"""

from collections.abc import Iterator

import torch


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, n_states: int, n_actions: int, device):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        shape = (n_steps, n_envs)
        kw = dict(device=device)
        self.obs = torch.zeros(*shape, n_states, **kw)
        self.masks = torch.zeros(*shape, n_actions, dtype=torch.bool, **kw)
        self.actions = torch.zeros(shape, dtype=torch.long, **kw)
        self.log_probs = torch.zeros(shape, **kw)  # log pi_old(a_t|s_t)
        self.values = torch.zeros(shape, **kw)     # V_old(s_t)
        self.rewards = torch.zeros(shape, **kw)
        self.dones = torch.zeros(shape, **kw)      # 1.0 if the episode ended at this step
        self.advantages = torch.zeros(shape, **kw)  # filled by compute_gae
        self.returns = torch.zeros(shape, **kw)    # value targets (filled by compute_gae)
        self.t = 0

    def add(self, obs, mask, action, log_prob, value, reward, done) -> None:
        t = self.t
        self.obs[t], self.masks[t], self.actions[t] = obs, mask, action
        self.log_probs[t], self.values[t] = log_prob, value
        self.rewards[t], self.dones[t] = reward, done
        self.t += 1

    @property
    def full(self) -> bool:
        return self.t == self.n_steps

    def reset(self) -> None:
        self.t = 0

    def minibatches(self, minibatch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        """Flatten (T, N) -> (T*N,) and yield shuffled minibatches. Call once per epoch K."""
        b = self.n_steps * self.n_envs
        flat = {
            "obs": self.obs.reshape(b, -1),
            "masks": self.masks.reshape(b, -1),
            "actions": self.actions.reshape(b),
            "log_probs": self.log_probs.reshape(b),
            "values": self.values.reshape(b),
            "advantages": self.advantages.reshape(b),
            "returns": self.returns.reshape(b),
        }
        perm = torch.randperm(b, device=self.device)
        for start in range(0, b, minibatch_size):
            idx = perm[start:start + minibatch_size]
            yield {k: v[idx] for k, v in flat.items()}
