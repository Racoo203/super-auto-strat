"""Play full games with a policy and record every decision: evaluation, replays and live watching.

One code path (`play_episode`) is shared by
  * evaluation during training (`evaluate` -> runs/<run>/replays/update_XXXX.jsonl),
  * live watching (agents/ppo/watch.py passes an `on_step` callback that prints the frame),
so what you watch live and what you replay later look the same.

A frame is shown *before* each decision: the state the agent sees, the result of its previous
action (and the battle, if that action ended the turn), and the policy overlay with the chosen
action marked '>'.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from sap.env import SapEnv, describe_action
from sap.env.render import render_policy

from .network import ActorCritic


@dataclass
class StepRecord:
    round: int
    action: int
    label: str
    probs: list[tuple[int, float]]  # top-k (action, probability) under the masked policy
    value: float                    # V(s) before acting
    reward: float
    invalid: bool = False
    battle: dict | None = None      # {"outcome", "opponent", "log"} if this action ended the turn
    frame: str = ""                 # text frame shown before acting (empty if not recorded)


@dataclass
class EpisodeRecord:
    seed: int | None
    steps: list[StepRecord] = field(default_factory=list)
    final_frame: str = ""
    won: bool = False
    wins: int = 0
    losses: int = 0
    draws: int = 0
    lives: int = 0
    rounds: int = 0
    ret: float = 0.0
    truncated: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        steps = [StepRecord(**{**s, "probs": [tuple(p) for p in s["probs"]]}) for s in d.pop("steps")]
        return cls(steps=steps, **d)


@torch.no_grad()
def policy_view(model: ActorCritic, obs, mask, device, top_k: int = 5):
    """Masked action probabilities and V(s) for a single observation.

    Returns (probs (n_actions,) numpy, value float, top-k [(action, p)]).
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    dist, value = model.distribution(obs_t, mask_t)
    probs = dist.probs[0].cpu().numpy()
    top = np.argsort(-probs)[:top_k]
    return probs, float(value[0]), [(int(a), float(probs[a])) for a in top if mask[a]]


def play_episode(
    model: ActorCritic,
    env: SapEnv,
    *,
    seed: int | None = None,
    greedy: bool = False,
    device=None,
    record_frames: bool = True,
    top_k: int = 5,
    on_step: Callable[[str, StepRecord | None], None] | None = None,
    rng: np.random.Generator | None = None,
) -> EpisodeRecord:
    """Play one game to the end. `on_step(frame, step)` is called before each action is applied
    (and once more at the end with step=None, for the final frame)."""
    device = device or next(model.parameters()).device
    rng = rng or np.random.default_rng(seed)
    was_training = model.training
    model.eval()

    obs, info = env.reset(seed=seed)
    rec = EpisodeRecord(seed=seed)
    done = False
    while not done:
        mask = info["action_mask"]
        probs, value, top = policy_view(model, obs, mask, device, top_k)
        action = int(np.argmax(probs)) if greedy else int(rng.choice(len(probs), p=probs / probs.sum()))
        if action not in (a for a, _ in top):  # sampled outside the top-k: still show it
            top.append((action, float(probs[action])))
        frame = ""
        if record_frames or on_step:
            frame = env.render(policy=render_policy(top, value, chosen=action))
        step = StepRecord(
            round=env.game.round, action=action, label=describe_action(action), probs=top,
            value=value, reward=0.0, frame=frame if record_frames else "",
        )
        if on_step:
            on_step(frame, step)

        obs, reward, terminated, truncated, info = env.step(action)
        step.reward = float(reward)
        step.invalid = bool(info.get("invalid_action"))
        step.battle = info.get("battle_record")
        rec.steps.append(step)
        rec.ret += float(reward)
        done = terminated or truncated
        rec.truncated = bool(truncated)

    g = env.game
    rec.won, rec.wins, rec.losses, rec.draws = g.won, g.wins, g.losses, g.draws
    rec.lives, rec.rounds = g.lives, g.round
    rec.final_frame = env.render() if (record_frames or on_step) else ""
    if on_step:
        on_step(rec.final_frame, None)
    model.train(was_training)
    return rec


EVAL_SEED_OFFSET = 10_000  # eval games use fixed seeds, so evaluations are comparable over time


def evaluate(
    model: ActorCritic,
    n_episodes: int,
    *,
    seed: int = 0,
    greedy: bool = False,
    env_kwargs: dict | None = None,
    record_frames: bool = True,
) -> list[EpisodeRecord]:
    """Play `n_episodes` games on fixed seeds (seed + EVAL_SEED_OFFSET + i)."""
    env = SapEnv(render_mode="ansi", **(env_kwargs or {}))
    base = seed + EVAL_SEED_OFFSET
    return [
        play_episode(model, env, seed=base + i, greedy=greedy, record_frames=record_frames)
        for i in range(n_episodes)
    ]


def summarize_records(records: list[EpisodeRecord], prefix: str = "eval/") -> dict[str, float]:
    if not records:
        return {}
    steps = [s for r in records for s in r.steps]
    return {
        f"{prefix}win_rate": float(np.mean([r.won for r in records])),
        f"{prefix}wins": float(np.mean([r.wins for r in records])),
        f"{prefix}losses": float(np.mean([r.losses for r in records])),
        f"{prefix}rounds": float(np.mean([r.rounds for r in records])),
        f"{prefix}ret": float(np.mean([r.ret for r in records])),
        f"{prefix}steps_per_round": len(steps) / max(1, sum(r.rounds for r in records)),
        f"{prefix}invalid_rate": float(np.mean([s.invalid for s in steps])) if steps else 0.0,
    }


def save_replays(records: list[EpisodeRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
    return path


def load_replays(path: str | Path) -> list[EpisodeRecord]:
    with open(path, encoding="utf-8") as f:
        return [EpisodeRecord.from_dict(json.loads(line)) for line in f if line.strip()]
