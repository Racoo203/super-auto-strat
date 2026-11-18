"""Ghost opponents: snapshots of teams other players built, fought asynchronously.

As in Super Auto Pets, a player never waits for anyone: at round r it fights a *ghost*, a frozen
copy of a team some player had at round r. Players (and the policies driving them) only interact
through the pool of ghosts, so any number of them can play in parallel.

    pool = GhostPool()
    pool.add([Ghost(team, round=3, source=0)], rng)        # source = id of the policy that built it
    env = SapEnv(opponent_fn=GhostOpponent(pool, sources={1}))  # fight only policy 1's teams

Pure data: no RL dependencies, and everything pickles (vector envs ship the pool to subprocesses).
"""

import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from sap.team import Team

RANDOM_SOURCE = -1  # `GhostOpponent.last_source` when it fell back to a random team


@dataclass
class Ghost:
    team: Team
    round: int
    source: int = 0  # who built it (a policy id)


class _RoundBucket:
    """Ghosts of one round: the most recent ones (FIFO) plus a uniform reservoir of older ones."""

    def __init__(self, recent: int, history: int):
        self.recent: deque[Ghost] = deque(maxlen=recent)
        self.history: list[Ghost] = []
        self.history_size = history
        self.evicted = 0  # ghosts pushed out of `recent` so far (reservoir sampling count)

    def add(self, ghost: Ghost, rng: random.Random) -> None:
        if len(self.recent) == self.recent.maxlen and self.history_size:
            self._keep_old(self.recent[0], rng)
        self.recent.append(ghost)

    def _keep_old(self, ghost: Ghost, rng: random.Random) -> None:
        self.evicted += 1
        if len(self.history) < self.history_size:
            self.history.append(ghost)
        else:
            j = rng.randrange(self.evicted)
            if j < self.history_size:
                self.history[j] = ghost

    def candidates(self, sources: set[int] | None, history: bool) -> list[Ghost]:
        pool = self.history if history else self.recent
        return [g for g in pool if sources is None or g.source in sources]


class GhostPool:
    """Ghosts keyed by round. Per round: `recent` newest ghosts + `history` older ones (reservoir)."""

    def __init__(self, recent_per_round: int = 32, history_per_round: int = 64):
        self.recent_per_round = recent_per_round
        self.history_per_round = history_per_round
        self.rounds: dict[int, _RoundBucket] = {}

    def add(self, ghosts: Iterable[Ghost], rng: random.Random) -> int:
        """Store copies of `ghosts` (empty teams are skipped: they'd be a free win). Returns how many."""
        n = 0
        for g in ghosts:
            if not len(g.team):
                continue
            team = Team(f"Ghost p{g.source} r{g.round}", [p.copy() for p in g.team], g.team.max_size)
            bucket = self.rounds.get(g.round)
            if bucket is None:
                bucket = self.rounds[g.round] = _RoundBucket(self.recent_per_round, self.history_per_round)
            bucket.add(Ghost(team, g.round, g.source), rng)
            n += 1
        return n

    def sample(
        self,
        round: int,
        rng: random.Random,
        sources: set[int] | None = None,
        history_prob: float = 0.0,
    ) -> Ghost | None:
        """A ghost from `round` (else the nearest earlier round that has one), built by one of
        `sources` (None = anyone). With probability `history_prob` it comes from the older
        reservoir when that has a match. None if nothing matches."""
        use_history = history_prob > 0 and rng.random() < history_prob
        for r in sorted((r for r in self.rounds if r <= round), reverse=True):
            bucket = self.rounds[r]
            # preferred part of the bucket first, then the other one
            found = bucket.candidates(sources, use_history) or bucket.candidates(sources, not use_history)
            if found:
                return rng.choice(found)
        return None

    def __len__(self) -> int:
        return sum(len(b.recent) + len(b.history) for b in self.rounds.values())


class GhostOpponent:
    """An `opponent_fn(game) -> Team` for `SapEnv` that fights ghosts from a pool.

    Uses the game's round and seeded rng, so a seeded game stays reproducible. Falls back to a
    random team (`game.make_opponent()`) with probability `random_prob`, or when no ghost matches.
    `last_source` is the source of the last opponent (RANDOM_SOURCE for a random team).
    """

    def __init__(
        self,
        pool: GhostPool,
        sources: Iterable[int] | None = None,
        history_prob: float = 0.0,
        random_prob: float = 0.0,
    ):
        self.pool = pool
        self.sources = None if sources is None else set(sources)
        self.history_prob = history_prob
        self.random_prob = random_prob
        self.last_source = RANDOM_SOURCE

    def __call__(self, game) -> Team:
        ghost = None
        if not (self.random_prob and game.rng.random() < self.random_prob):
            ghost = self.pool.sample(game.round, game.rng, self.sources, self.history_prob)
        if ghost is None:
            self.last_source = RANDOM_SOURCE
            return game.make_opponent()
        self.last_source = ghost.source
        return ghost.team
