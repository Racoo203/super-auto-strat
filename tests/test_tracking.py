"""Observability: env rendering, recorded episodes, checkpoints and run tracking (Optuna + files)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")

from agents.ppo import (  # noqa: E402
    ActorCritic, PPOConfig, Run, load_checkpoint, load_metrics, load_replays, load_runs,
    play_episode, save_checkpoint, save_replays, train,
)
from agents.ppo.replay import summarize  # noqa: E402
from sap.env import ACTIONS, OBS_SIZE, SapEnv, describe_action  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    return ActorCritic(OBS_SIZE, len(ACTIONS), hidden=(16,))


def tiny_cfg(tmp_path, **kw):
    base = dict(
        n_envs=2, n_steps=16, n_updates=3, minibatch_size=16, hidden=(16,), async_envs=False,
        eval_every=2, eval_episodes=2, checkpoint_every=2, runs_dir=str(tmp_path),
    )
    return PPOConfig(**{**base, **kw})


def test_ansi_render_shows_state_last_action_and_battle():
    env = SapEnv(render_mode="ansi")
    obs, info = env.reset(seed=0)
    assert "Round 1" in env.render() and "Store" in env.render()
    end_turn = len(ACTIONS) - 1
    env.step(end_turn)
    frame = env.render()
    assert f"Last action: {describe_action(end_turn)}" in frame
    assert "--- Battle:" in frame and "Opponent" in frame and "Round 2" in frame


def test_render_policy_overlay_marks_chosen_action():
    env = SapEnv(render_mode="ansi")
    env.reset(seed=0)
    from sap.env.render import render_policy

    frame = env.render(policy=render_policy([(0, 0.7), (131, 0.3)], value=0.5, chosen=131))
    assert "V(s) = +0.50" in frame and "> " in frame and "70.0%" in frame


def test_play_episode_records_a_full_game_and_round_trips(tmp_path):
    rec = play_episode(tiny_model(), SapEnv(render_mode="ansi"), seed=1)
    assert rec.steps and rec.steps[0].frame and rec.final_frame
    assert rec.wins + rec.losses + rec.draws == sum(s.battle is not None for s in rec.steps)
    assert all(abs(sum(p for _, p in s.probs)) <= 1.0 + 1e-6 for s in rec.steps)
    assert rec.ret == pytest.approx(sum(s.reward for s in rec.steps))
    (back,) = load_replays(save_replays([rec], tmp_path / "r.jsonl"))
    assert back == rec
    assert f"seed 1:" in summarize(back)


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    cfg = PPOConfig(hidden=(16,))
    path = save_checkpoint(tmp_path / "c.pt", model, None, cfg, update=7)
    loaded, cfg2, ckpt = load_checkpoint(path, device="cpu")
    obs = torch.rand(3, OBS_SIZE)
    assert torch.allclose(model(obs)[0], loaded(obs)[0])
    assert cfg2 == cfg and ckpt["update"] == 7


def test_train_tracks_the_run(tmp_path):
    cfg = tiny_cfg(tmp_path)
    result = train(cfg, "smoke", log_fn=None)
    d = result.run_dir
    for f in ("checkpoints/latest.pt", "checkpoints/best.pt", "checkpoints/update_0001.pt",
              "replays/update_0001.jsonl", "replays/update_0002.jsonl", "metrics.jsonl", "config.json"):
        assert (d / f).exists(), f
    assert list((d / "tb").iterdir())  # TensorBoard events
    metrics = load_metrics(d)
    assert list(metrics.index) == [0, 1, 2]
    assert {"loss/policy", "loss/approx_kl", "perf/sps"} <= set(metrics.columns)
    assert metrics["eval/win_rate"].notna().sum() == 2

    runs = load_runs(runs_dir=str(tmp_path))
    row = runs.iloc[-1]
    assert row["state"] == "COMPLETE" and row["user_attrs_run_name"] == "smoke"
    assert row["value"] == pytest.approx(result.best_eval_win_rate)
    assert row["params_lr"] == pytest.approx(cfg.lr) and row["user_attrs_cfg.hidden"] == [16]


def test_crashed_run_is_marked_failed_and_keeps_its_logs(tmp_path):
    cfg = tiny_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        with Run(cfg, "crash", tensorboard=False) as run:
            run.log(0, {"train/win_rate": 0.25})
            raise RuntimeError("boom")
    row = load_runs(runs_dir=str(tmp_path)).iloc[-1]
    assert row["state"] == "FAIL" and "boom" in row["user_attrs_error"]
    lines = (tmp_path / "crash" / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["train/win_rate"] == 0.25


def test_run_name_is_not_reused(tmp_path):
    cfg = tiny_cfg(tmp_path)
    with Run(cfg, "same", tensorboard=False) as a:
        a.log(0, {})
    with Run(cfg, "same", tensorboard=False) as b:
        pass
    assert a.dir != b.dir
    assert np.all(load_runs(runs_dir=str(tmp_path))["state"] == "COMPLETE")


def test_two_policy_self_play_trains_both_and_tracks_each(tmp_path):
    cfg = tiny_cfg(tmp_path, n_envs=4, n_policies=2, n_updates=2, eval_every=2, checkpoint_every=0)
    result = train(cfg, "duel", log_fn=None)
    ck = result.run_dir / "checkpoints"
    for f in ("latest.pt", "latest_p1.pt", "best.pt", "best_p1.pt"):
        assert (ck / f).exists(), f
    p1, _, raw = load_checkpoint(ck / "latest_p1.pt", device="cpu")
    assert raw["policy"] == 1 and len(result.models) == 2
    m = load_metrics(result.run_dir)
    assert {"loss/policy", "loss/p1/policy", "ghosts/pool_size"} <= set(m.columns)
    assert m["ghosts/pool_size"].iloc[-1] > 0
    assert {"eval/vs_ghosts/battle_win_rate", "eval/h2h_p0_vs_p1/battle_win_rate"} <= set(m.columns)


def test_random_opponent_mode_has_no_ghosts(tmp_path):
    result = train(tiny_cfg(tmp_path, opponent="random", n_updates=1, eval_every=0), "rand", log_fn=None)
    assert "ghosts/pool_size" not in load_metrics(result.run_dir).columns


def test_watch_against_another_policys_ghosts():
    from agents.ppo.watch import watch

    frames = []
    (rec,) = watch(tiny_model(), vs=tiny_model(), vs_games=2, seed=0, delay=0, printer=frames.append)
    sources = {s.battle["opponent_source"] for s in rec.steps if s.battle}
    assert 1 in sources and frames
