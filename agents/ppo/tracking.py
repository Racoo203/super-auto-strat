"""Episode statistics for logging: running per-env return and length."""

import numpy as np


class EpisodeTracker:
    def __init__(self, n_envs: int):
        self.ret = np.zeros(n_envs)
        self.len = np.zeros(n_envs, dtype=int)
        self.finished: list[tuple[float, int]] = []

    def update(self, reward: np.ndarray, done: np.ndarray) -> None:
        self.ret += reward
        self.len += 1
        for i in np.flatnonzero(done):
            self.finished.append((float(self.ret[i]), int(self.len[i])))
            self.ret[i], self.len[i] = 0.0, 0

    def pop(self) -> list[tuple[float, int]]:
        """(return, length) of every episode finished since the last pop."""
        out, self.finished = self.finished, []
        return out
