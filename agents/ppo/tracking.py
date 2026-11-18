"""Episode statistics for logging: running per-env return and length, plus the game result.

With SAME_STEP autoreset the vector env returns the *new* episode's info; the finished episode's
last info is in `info["final_info"]` (a dict of arrays, `final_info["_wins"][i]` True for env i).
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EpisodeStats:
    ret: float
    length: int
    won: bool = False  # reached WIN_TARGET wins
    wins: int = 0
    losses: int = 0
    lives: int = 0
    rounds: int = 0
    env: int = -1  # env column it was played in (-> which policy, with several policies)

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_episodes(episodes: list[EpisodeStats], prefix: str = "") -> dict[str, float]:
    """Mean of each stat (and the win rate) over a list of episodes; {} if empty."""
    if not episodes:
        return {}
    keys = ("ret", "length", "wins", "losses", "lives", "rounds")
    out = {f"{prefix}{k}": float(np.mean([getattr(e, k) for e in episodes])) for k in keys}
    out[f"{prefix}win_rate"] = float(np.mean([e.won for e in episodes]))
    out[f"{prefix}episodes"] = float(len(episodes))
    return out


class EpisodeTracker:
    def __init__(self, n_envs: int):
        self.ret = np.zeros(n_envs)
        self.len = np.zeros(n_envs, dtype=int)
        self.finished: list[EpisodeStats] = []
        self.opponents: list[tuple[int, int]] = []  # (env, opponent source) per battle

    def update(self, reward: np.ndarray, done: np.ndarray, info: dict | None = None) -> None:
        self.ret += reward
        self.len += 1
        final = (info or {}).get("final_info", {})
        # a battle's info is in `info`, or in `final_info` when that battle ended the game
        for src in (info or {}, final or {}):
            if "opponent_source" in src:
                for i in np.flatnonzero(src["_opponent_source"]):
                    self.opponents.append((int(i), int(src["opponent_source"][i])))
        for i in np.flatnonzero(done):
            stats = EpisodeStats(float(self.ret[i]), int(self.len[i]), env=int(i))
            if final and final.get("_wins", np.zeros(len(done), bool))[i]:
                stats.won = bool(final["won"][i])
                stats.wins, stats.losses = int(final["wins"][i]), int(final["losses"][i])
                stats.lives = int(final["lives"][i])
                # `round` is the store round the game ended in (a finished game stops counting).
                stats.rounds = int(final["round"][i])
            self.finished.append(stats)
            self.ret[i], self.len[i] = 0.0, 0

    def pop(self) -> list[EpisodeStats]:
        """Stats of every episode finished since the last pop."""
        out, self.finished = self.finished, []
        return out

    def pop_opponents(self) -> list[tuple[int, int]]:
        """(env, opponent source) of every battle since the last call (-1 = random team)."""
        out, self.opponents = self.opponents, []
        return out
