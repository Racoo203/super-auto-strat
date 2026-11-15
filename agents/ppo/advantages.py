"""Generalized Advantage Estimation, as a pure function of the rollout tensors."""

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(gamma, lambda). Returns (advantages, returns), both shaped like `rewards`.

    rewards, values, dones: (T, N), laid out as in buffer.py (row t: r_t, V(s_t), done_t).
    last_value: V(s_T), shape (N,), the value of the obs *after* the last stored step, used to
    bootstrap the final row (episodes are usually cut mid-way by T).

    Loop backwards over t = T-1 ... 0:
        next_value  = last_value if t == T-1 else values[t+1]
        nonterminal = 1 - dones[t]
        delta_t = r_t + gamma * nonterminal * next_value - V(s_t)
        A_t     = delta_t + gamma * lambda * nonterminal * A_{t+1}      (A_T = 0)
    then returns = advantages + values (the value-function targets).

    Note: this treats truncation (max_rounds hit) like termination. Strictly you'd bootstrap
    V(final_obs) on truncation; with max_rounds=40 it's rare, so it's fine to start with.

    tests/test_ppo.py::test_compute_gae checks a hand-computed case once you implement this.
    """
    raise NotImplementedError("your turn: implement GAE here")
