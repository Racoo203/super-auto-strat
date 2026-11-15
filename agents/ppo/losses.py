"""PPO surrogate losses.

A "custom loss" in PyTorch is just a function that returns a scalar tensor built from torch ops
on tensors that require grad. Adam doesn't need anything special:

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    ...
    loss = ppo_clip_loss(new_log_probs, old_log_probs, advantages)  # + value / entropy terms
    optimizer.zero_grad()
    loss.backward()        # autograd: d loss / d theta, flows back through new_log_probs
    optimizer.step()       # Adam updates theta

Only `new_log_probs` may carry gradients. `old_log_probs` and `advantages` come from the rollout
buffer (collected under torch.no_grad()), so they are constants w.r.t. theta. If they did carry
grads, the ratio's gradient would be wrong.
"""

import torch


def ppo_clip_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """-L^CLIP: negated so that *minimising* it with Adam maximises the clipped objective.

    All inputs are (B,) for one minibatch: log pi_theta(a_t|s_t), log pi_theta_old(a_t|s_t), A_t.
    """
    # (B,) vs (B, 1) would silently broadcast to (B, B) and give a meaningless loss.
    assert new_log_probs.shape == old_log_probs.shape == advantages.shape, (
        new_log_probs.shape, old_log_probs.shape, advantages.shape,
    )
    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()

    # r_t(theta) = pi_new / pi_old = exp(log pi_new - log pi_old)
    ratios = torch.exp(new_log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

    # Pessimistic bound, averaged over the minibatch; minus sign for gradient *descent*.
    return -torch.min(surr1, surr2).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """L^VF = mean((V_theta(s_t) - V_t^targ)^2), with V_t^targ = the GAE returns (A_t + V_old(s_t))."""
    raise NotImplementedError("your turn")


def entropy_bonus(entropy: torch.Tensor) -> torch.Tensor:
    """S = mean entropy of the policy over the minibatch (pass in `dist.entropy()`)."""
    raise NotImplementedError("your turn")


# Total to minimise: ppo_clip_loss(...) + c1 * value_loss(...) - c2 * entropy_bonus(...)
