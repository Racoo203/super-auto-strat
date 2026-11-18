"""Self-play glue between the vector env and the ghost pool (sap/ghosts.py).

Once per update, after the rollout:

    harvest(envs, seat_policy, pool, rng)       # teams that fought -> ghosts, tagged by policy
    push_opponents(envs, seat_policy, pool, cfg)  # every seat now fights ghosts from the new pool

Each seat is an ordinary single-agent SapEnv, so nobody waits for anybody: opponents are at most
one update old, and older policies stay in the pool's history reservoir at no extra cost.
Traffic is two vector-env calls per update (the pool is pickled once per env process).
"""

import random

import numpy as np

from sap.ghosts import RANDOM_SOURCE, Ghost, GhostOpponent, GhostPool

from .config import PPOConfig


def make_pool(cfg: PPOConfig) -> GhostPool:
    return GhostPool(cfg.ghost_recent, cfg.ghost_history)


def harvest(envs, seat_policy: np.ndarray, pool: GhostPool, rng: random.Random) -> int:
    """Move every env's fought teams into `pool`, tagged with the seat's policy id. Returns how many."""
    per_env = envs.call("pop_snapshots")
    ghosts = [Ghost(team, rnd, int(seat_policy[i])) for i, snaps in enumerate(per_env) for rnd, team in snaps]
    return pool.add(ghosts, rng)


def opponent_sources(policy: int, n_policies: int, opponents: str) -> set[int] | None:
    """Whose ghosts a seat of `policy` may fight (None = anyone)."""
    if n_policies == 1 or opponents == "all":
        return None
    if opponents == "others":
        return set(range(n_policies)) - {policy}
    raise ValueError(f"opponents must be 'others' or 'all', not {opponents!r}")


def push_opponents(envs, seat_policy: np.ndarray, pool: GhostPool, cfg: PPOConfig) -> None:
    """Give every seat a GhostOpponent over the current pool (falls back to random teams while
    the pool has nothing for a round)."""
    n_policies = int(seat_policy.max()) + 1
    fns = [
        GhostOpponent(
            pool, opponent_sources(int(p), n_policies, cfg.opponents),
            history_prob=cfg.ghost_history_prob, random_prob=cfg.random_opponent_prob,
        )
        for p in seat_policy
    ]
    envs.set_attr("opponent_fn", fns)


def random_opponent_fraction(sources: list[int]) -> float:
    """Share of battles fought against random teams instead of ghosts (from info["opponent_source"])."""
    return float(np.mean([s == RANDOM_SOURCE for s in sources])) if sources else float("nan")
