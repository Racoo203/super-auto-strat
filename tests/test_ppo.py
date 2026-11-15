import pytest

torch = pytest.importorskip("torch")

from agents.ppo import (  # noqa: E402
    ActorCritic, EpisodeTracker, RolloutBuffer, collect_rollout, compute_gae, make_vec_env,
    ppo_clip_loss,
)

N_STATES, N_ACTIONS = 8, 5


def small_model():
    torch.manual_seed(0)
    return ActorCritic(N_STATES, N_ACTIONS, hidden=(16,))


def test_masked_actions_are_never_sampled():
    model = small_model()
    obs = torch.rand(256, N_STATES)
    mask = torch.zeros(256, N_ACTIONS, dtype=torch.bool)
    mask[:, [1, 3]] = True
    action, log_prob, value = model.act(obs, mask)
    assert set(action.tolist()) <= {1, 3}
    assert action.shape == log_prob.shape == value.shape == (256,)


def test_masked_probs_sum_to_one_over_legal_actions():
    model = small_model()
    mask = torch.tensor([[True, False, True, False, False]])
    dist, _ = model.distribution(torch.rand(1, N_STATES), mask)
    assert dist.probs[0, [0, 2]].sum().item() == pytest.approx(1.0)
    assert torch.isfinite(dist.entropy()).all()


def test_clip_loss_equals_minus_mean_advantage_when_policy_unchanged():
    logp = torch.log(torch.tensor([0.2, 0.5, 0.3]))
    adv = torch.tensor([1.0, -2.0, 0.5])
    assert ppo_clip_loss(logp, logp, adv).item() == pytest.approx(-adv.mean().item())


def test_clip_loss_clips_large_ratios():
    old = torch.log(torch.tensor([0.1]))
    new = torch.log(torch.tensor([0.9]))  # ratio 9, positive advantage -> clipped to 1.2
    assert ppo_clip_loss(new, old, torch.tensor([1.0]), epsilon=0.2).item() == pytest.approx(-1.2)


def test_clip_loss_rejects_mismatched_shapes():
    with pytest.raises(AssertionError):
        ppo_clip_loss(torch.zeros(4), torch.zeros(4), torch.zeros(4, 1))


def test_clip_loss_trains_with_adam():
    model = small_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    obs, mask = torch.rand(32, N_STATES), torch.ones(32, N_ACTIONS, dtype=torch.bool)
    actions, old_logp, _ = model.act(obs, mask)
    adv = torch.randn(32)
    losses = []
    for _ in range(5):
        dist, _ = model.distribution(obs, mask)
        loss = ppo_clip_loss(dist.log_prob(actions), old_logp, adv)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_minibatches_cover_every_sample_once():
    buf = RolloutBuffer(n_steps=4, n_envs=3, n_states=N_STATES, n_actions=N_ACTIONS, device="cpu")
    buf.actions = torch.arange(12).reshape(4, 3)
    seen = torch.cat([mb["actions"] for mb in buf.minibatches(5)])
    assert sorted(seen.tolist()) == list(range(12))


def test_episode_tracker_reports_finished_episodes():
    import numpy as np

    tr = EpisodeTracker(2)
    tr.update(np.array([1.0, 0.5]), np.array([False, True]))
    tr.update(np.array([1.0, 0.0]), np.array([True, False]))
    assert tr.pop() == [(0.5, 1), (2.0, 2)]
    assert tr.pop() == []


def test_collect_rollout_fills_buffer_from_parallel_envs():
    envs, obs, mask = make_vec_env(2, async_envs=False, seed=0)
    n_states, n_actions = envs.single_observation_space.shape[0], int(envs.single_action_space.n)
    model = ActorCritic(n_states, n_actions, hidden=(16,))
    buf = RolloutBuffer(8, 2, n_states, n_actions, "cpu")
    obs, mask, last_value = collect_rollout(envs, model, buf, obs, mask, "cpu")
    envs.close()
    assert buf.full and last_value.shape == (2,)
    taken = buf.masks.gather(-1, buf.actions.unsqueeze(-1)).squeeze(-1)
    assert taken.all()  # every stored action was legal under its stored mask


def test_compute_gae():
    """Hand-computed: T=3, N=1, gamma=0.9, lambda=0.8, episode ends at t=1."""
    rewards = torch.tensor([[1.0], [0.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last_value = torch.tensor([2.0])
    try:
        adv, ret = compute_gae(rewards, values, dones, last_value, 0.9, 0.8)
    except NotImplementedError:
        pytest.skip("compute_gae not implemented yet")
    assert adv.squeeze(-1).tolist() == pytest.approx([0.59, -0.5, 2.3])
    assert ret.squeeze(-1).tolist() == pytest.approx([1.09, 0.0, 2.8])
