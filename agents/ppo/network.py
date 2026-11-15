"""Actor-critic network: one shared trunk, a policy head (logits) and a value head V(s).

Why logits instead of a final Softmax:
  * Illegal actions are masked by setting their logit to a huge negative number *before* the
    softmax. The softmax then gives them probability ~0 and the legal ones already sum to 1,
    so there's no "multiply by mask and renormalise" step (that's what the old `select_act` did,
    because masking *after* a softmax breaks the sum-to-1 property).
  * PPO needs log pi(a|s). `Categorical(logits=...)` computes it with log-softmax, which is
    numerically stable; `log(softmax(x))` can underflow to log(0) = -inf.
  * The same masked distribution is rebuilt during the update, so the new log-probs are
    comparable with the old ones stored in the buffer.
"""

import torch
from torch import nn
from torch.distributions import Categorical

MASKED_LOGIT = -1e9  # not -inf: -inf gives 0 * -inf = NaN in the entropy


class ActorCritic(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden: tuple[int, ...] = (2048, 1024, 512)):
        super().__init__()
        layers, last = [], n_states
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, n_actions)  # pi(.|s) as logits
        self.value_head = nn.Linear(last, 1)           # V(s): one number per state

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, n_states) -> logits (B, n_actions), value (B,)."""
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    def distribution(
        self, obs: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        """Masked action distribution and V(s). Used both when acting and when updating."""
        logits, value = self(obs)
        logits = torch.where(action_mask, logits, torch.full_like(logits, MASKED_LOGIT))
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """Sample actions for a batch of envs. Returns (action, log_prob, value), each (B,).

        This replaces the old `select_act`. No gradients: these are the "old policy" numbers
        that get stored in the rollout buffer.
        """
        dist, value = self.distribution(obs, action_mask)
        action = dist.sample()
        return action, dist.log_prob(action), value
