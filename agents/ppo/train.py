"""The PPO training loop (formerly inline in PPO.ipynb), with tracking.

    from agents.ppo import PPOConfig, train
    result = train(PPOConfig(n_updates=300), "baseline")                  # shared-policy self-play
    result = train(PPOConfig(n_updates=300, n_policies=2), "duel")        # two agents vs each other
    result = train(PPOConfig(n_updates=300, opponent="random"), "random")  # the original setup

Opponents (cfg.opponent):
    "random"  every seat fights random teams.
    "ghosts"  self-play: seats fight teams the policies built (sap/ghosts.py, selfplay.py).
              cfg.n_policies > 1 trains that many agents at once, env i played by policy i % P.
              The first update fights random teams while the pool fills up.

Per update it logs (TensorBoard + metrics.jsonl, see experiments.py):
    train/*   episodes finished this update: ret, length, win_rate, wins, losses, lives, rounds
    loss/*    policy, value, entropy, approx_kl, clip_frac, explained_var
    ghosts/*  self-play: pool size, share of battles fought against random teams
    perf/sps  environment steps per second
With several policies, policy 0 keeps the plain names and policy p logs train/p{p}/*, loss/p{p}/*.

Every cfg.eval_every updates it plays cfg.eval_episodes games on fixed seeds against random teams
(eval/*: a fixed yardstick; self-play win rates drift toward 50% and say little), saves them as
runs/<run>/replays/update_XXXX.jsonl, reports eval/win_rate to Optuna and saves checkpoints.
With ghosts it also plays policy 0 against the live pool (eval/vs_ghosts/*) and, with 2+ policies,
against policy 1's own ghosts (eval/h2h_p0_vs_p1/*).
Checkpoints: latest.pt / best.pt are policy 0; other policies get latest_p1.pt, best_p1.pt, ...
cfg.keep_checkpoints caps how many numbered update_XXXX.pt checkpoints are kept per policy.
If training crashes or is interrupted, latest.pt is still written and the trial is marked FAIL.
train(cfg, run_name, resume=True) continues runs/<run_name>/checkpoints/latest*.pt instead of
starting over (see agents/ppo/run.py for a script that can relaunch with --resume after a crash).
A non-finite loss (training diverged) raises instead of quietly burning the rest of n_updates.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sap.ghosts import GhostOpponent, GhostPool

from .advantages import compute_gae
from .buffer import RolloutBuffer
from .checkpoints import prune_checkpoints, save_checkpoint
from .config import PPOConfig, get_device
from .envs import make_vec_env
from .evaluate import evaluate, head_to_head, save_replays, summarize_records
from .experiments import Run
from .losses import entropy_bonus, ppo_clip_loss, value_loss
from .network import ActorCritic
from .policies import PolicySet, seat_policies
from .rollout import collect_rollout
from .selfplay import harvest, make_pool, opponent_sources, push_opponents, random_opponent_fraction
from .tracking import EpisodeTracker, summarize_episodes


@dataclass
class TrainResult:
    run_name: str
    run_dir: Path
    best_eval_win_rate: float | None
    model: ActorCritic                       # policy 0
    models: list[ActorCritic] | None = None  # every policy (index = policy id)


def ppo_update(model, optimizer, buffer: RolloutBuffer, cfg: PPOConfig, envs=None) -> dict[str, float]:
    """K epochs of minibatch updates on the collected batch. Returns mean loss diagnostics.

    envs: only these env columns (this policy's seats, when several policies share the buffer).
    """
    stats: dict[str, list[float]] = {k: [] for k in ("policy", "value", "entropy", "approx_kl", "clip_frac")}
    for _ in range(cfg.n_epochs):
        for mb in buffer.minibatches(cfg.minibatch_size, envs):
            dist, new_values = model.distribution(mb["obs"], mb["masks"])
            new_log_probs = dist.log_prob(mb["actions"])
            entropy = dist.entropy()

            advantages = mb["advantages"]
            if cfg.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            pl = ppo_clip_loss(new_log_probs, mb["log_probs"], advantages, cfg.clip_eps)
            vl = value_loss(new_values, mb["returns"])
            ent = entropy_bonus(entropy)
            loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite PPO loss (policy={pl.item():.4g}, value={vl.item():.4g}, "
                    f"entropy={ent.item():.4g}) -- training diverged, stopping before it corrupts "
                    "the model further"
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_probs - mb["log_probs"]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()  # k3 estimator
                clip_frac = ((ratio - 1).abs() > cfg.clip_eps).float().mean()
            if not torch.isfinite(approx_kl):
                raise FloatingPointError(f"non-finite approx_kl ({approx_kl.item()}) -- training diverged")
            stats["approx_kl"].append(approx_kl.item())
            stats["clip_frac"].append(clip_frac.item())
            stats["policy"].append(pl.item())
            stats["value"].append(vl.item())
            stats["entropy"].append(ent.item())

    out = {f"loss/{k}": float(np.mean(v)) for k, v in stats.items()}
    cols = slice(None) if envs is None else torch.as_tensor(envs, device=buffer.device)
    y, v = buffer.returns[:, cols].flatten(), buffer.values[:, cols].flatten()
    var_y = torch.var(y).item()
    out["loss/explained_var"] = float("nan") if var_y == 0 else 1 - torch.var(y - v).item() / var_y
    return out


def _resume_from(ckpt_dir: Path, models: list[ActorCritic], optimizers: list, device) -> int:
    """Load each policy's latest.pt in place (model + optimizer state). Returns the update to
    resume from (one past the highest `update` recorded), or 0 if no checkpoint exists yet."""
    starts = []
    for p, (model, opt) in enumerate(zip(models, optimizers)):
        stem = "latest" if p == 0 else f"latest_p{p}"
        path = ckpt_dir / f"{stem}.pt"
        if not path.exists():
            return 0
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        starts.append(ckpt["update"] + 1)
    return max(starts)


def train(
    cfg: PPOConfig,
    run_name: str | None = None,
    *,
    storage: str | None = None,
    trial=None,
    prune: bool = False,
    env_kwargs: dict | None = None,
    log_every: int = 1,
    log_fn: Callable[[str], None] | None = print,
    resume: bool = False,
) -> TrainResult:
    """Train PPO and track the run. See the module docstring.

    storage: Optuna storage URL (default sqlite:///<runs_dir>/optuna.db).
    trial / prune: used by tune.py; a normal run creates its own trial.
    resume: continue runs/<run_name>/checkpoints/latest*.pt instead of starting from scratch
    (model + optimizer state and the update counter; the self-play ghost pool is not persisted
    and rebuilds within a few updates). Writes append to the same run directory. Opens a new
    Optuna trial either way -- Optuna's ask/tell doesn't support resuming one across restarts.
    """
    if cfg.opponent not in ("ghosts", "random"):
        raise ValueError(f"cfg.opponent must be 'ghosts' or 'random', not {cfg.opponent!r}")
    device = get_device()
    torch.manual_seed(cfg.seed)
    env_kwargs = env_kwargs or {}
    selfplay = cfg.opponent == "ghosts"
    n_policies = cfg.n_policies if selfplay else 1

    with Run(cfg, run_name, storage=storage, trial=trial, resume=resume) as run:
        run.trial.set_user_attr("device", str(device))
        envs, obs, mask = make_vec_env(cfg.n_envs, async_envs=cfg.async_envs, seed=cfg.seed, **env_kwargs)
        n_states = envs.single_observation_space.shape[0]
        n_actions = int(envs.single_action_space.n)
        models = [ActorCritic(n_states, n_actions, hidden=cfg.hidden).to(device) for _ in range(n_policies)]
        optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr, eps=1e-5) for m in models]
        policies = PolicySet(models, seat_policies(cfg.n_envs, n_policies), optimizers)
        pool, pool_rng = make_pool(cfg), random.Random(cfg.seed)
        buffer = RolloutBuffer(cfg.n_steps, cfg.n_envs, n_states, n_actions, device)
        tracker = EpisodeTracker(cfg.n_envs)
        ckpt_dir = run.dir / "checkpoints"

        start_update = _resume_from(ckpt_dir, models, optimizers, device) if resume else 0
        if start_update:
            run.trial.set_user_attr("resumed_from_update", start_update)
        update = start_update - 1

        def save(name: str, p: int, **extra) -> None:
            stem = name if p == 0 else f"{name}_p{p}"
            save_checkpoint(ckpt_dir / f"{stem}.pt", models[p], optimizers[p], cfg, update, run=run.name, policy=p, **extra)

        if log_fn:
            mode = f"self-play, {n_policies} polic{'y' if n_policies == 1 else 'ies'}" if selfplay else "vs random teams"
            resume_note = f", resuming from update {start_update}" if start_update else ""
            log_fn(f"run '{run.name}' -> {run.dir}  (trial #{run.trial.number}, {device}, {mode}{resume_note})")
        try:
            for update in range(start_update, cfg.n_updates):
                t0 = time.time()
                obs, mask, last_value = collect_rollout(envs, policies, buffer, obs, mask, device, tracker)
                if selfplay:  # this rollout's teams become the next rollout's opponents
                    harvest(envs, policies.seat_policy, pool, pool_rng)
                    push_opponents(envs, policies.seat_policy, pool, cfg)
                buffer.advantages, buffer.returns = compute_gae(
                    buffer.rewards, buffer.values, buffer.dones, last_value, cfg.gamma, cfg.gae_lambda
                )

                metrics: dict[str, float] = {}
                episodes, battles = tracker.pop(), tracker.pop_opponents()
                for p in range(n_policies):
                    pre = "" if p == 0 else f"p{p}/"
                    seats = None if n_policies == 1 else policies.seats(p)
                    losses = ppo_update(models[p], optimizers[p], buffer, cfg, envs=seats)
                    metrics.update({k.replace("loss/", f"loss/{pre}", 1): v for k, v in losses.items()})
                    mine = [e for e in episodes if policies.seat_policy[e.env] == p]
                    metrics.update(summarize_episodes(mine, prefix=f"train/{pre}"))
                metrics["perf/sps"] = cfg.batch_size / (time.time() - t0)
                if selfplay:
                    metrics["ghosts/pool_size"] = float(len(pool))
                    if battles:
                        metrics["ghosts/random_frac"] = random_opponent_fraction([s for _, s in battles])

                last = update == cfg.n_updates - 1
                if cfg.eval_every and ((update + 1) % cfg.eval_every == 0 or last):
                    records = evaluate(models[0], cfg.eval_episodes, seed=cfg.seed, greedy=cfg.eval_greedy, env_kwargs=env_kwargs)
                    save_replays(records, run.dir / "replays" / f"update_{update:04d}.jsonl")
                    ev = summarize_records(records)
                    metrics.update(ev)
                    if selfplay:
                        metrics.update(selfplay_evals(models, pool, cfg, env_kwargs))
                    for p in range(n_policies):
                        save("latest", p)
                    if run.report(update, ev["eval/win_rate"], prune=prune):
                        save("best", 0, **ev)
                        for p in range(1, n_policies):
                            save("best", p)
                if cfg.checkpoint_every and (update + 1) % cfg.checkpoint_every == 0:
                    for p in range(n_policies):
                        save(f"update_{update:04d}", p)
                    prune_checkpoints(ckpt_dir, keep=cfg.keep_checkpoints)

                run.log(update, metrics)
                if log_fn and (update % log_every == 0 or "eval/win_rate" in metrics):
                    log_fn(_progress_line(update, metrics))
        finally:
            envs.close()
            if update >= 0:  # whatever happened, keep the latest weights
                for p in range(n_policies):
                    save("latest", p)

    return TrainResult(run.name, run.dir, run.best_value, models[0], models)


def selfplay_evals(models: list[ActorCritic], pool: GhostPool, cfg: PPOConfig, env_kwargs: dict | None = None) -> dict[str, float]:
    """Policy 0 against the live ghost pool (only the ghosts it trains against), and with 2+
    policies against policy 1's own freshly played ghosts."""
    env_kwargs = env_kwargs or {}
    sources = opponent_sources(0, len(models), cfg.opponents)
    vs_pool = evaluate(
        models[0], cfg.eval_episodes, seed=cfg.seed, greedy=cfg.eval_greedy, record_frames=False,
        env_kwargs={**env_kwargs, "opponent_fn": GhostOpponent(pool, sources)},
    )
    out = summarize_records(vs_pool, prefix="eval/vs_ghosts/")
    if len(models) > 1:
        h2h = head_to_head(models[0], models[1], cfg.eval_episodes, seed=cfg.seed, greedy=cfg.eval_greedy, env_kwargs=env_kwargs)
        out.update(summarize_records(h2h, prefix="eval/h2h_p0_vs_p1/"))
    return out


def _progress_line(update: int, m: dict) -> str:
    parts = [f"update {update:4d}"]
    if "train/episodes" in m:
        parts.append(
            f"{int(m['train/episodes'])} eps, ret {m['train/ret']:.2f}, win {m['train/win_rate']:.0%}, "
            f"wins {m['train/wins']:.1f}, len {m['train/length']:.0f}"
        )
    parts.append(f"ent {m['loss/entropy']:.2f}, kl {m['loss/approx_kl']:.1e}, {m['perf/sps']:.0f} sps")
    if "ghosts/pool_size" in m:
        parts.append(f"ghosts {m['ghosts/pool_size']:.0f}")
    if "eval/win_rate" in m:
        parts.append(f"EVAL win {m['eval/win_rate']:.0%} wins {m['eval/wins']:.1f} rounds {m['eval/rounds']:.1f}")
    if "eval/vs_ghosts/win_rate" in m:
        parts.append(f"vs ghosts: battles {m['eval/vs_ghosts/battle_win_rate']:.0%}")
    if "eval/h2h_p0_vs_p1/win_rate" in m:
        parts.append(f"p0 vs p1: battles {m['eval/h2h_p0_vs_p1/battle_win_rate']:.0%}")
    return " | ".join(parts)
