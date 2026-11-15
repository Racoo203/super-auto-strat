"""SapEnv: an agent plays the shop phase; battles resolve automatically.

This file only wires the gym API together. The pieces live next to it:
  actions.py      the Discrete action table, labels, applying an action to a Game
  masking.py      which actions are legal (`info["action_mask"]`)
  observation.py  Game -> float32 vector

Illegal actions change nothing and cost `invalid_action_penalty`. Actions that only undo the
previous one (`move a->b` right after `move b->a`, or freezing the same shop slot twice in a row)
are masked too, so an agent can't waste the round toggling back and forth.

Reward: +1 for a won battle, -1 for a lost one, 0 for a draw (tunable).
The episode ends when the game is over (out of lives or WIN_TARGET wins), and is truncated after
`max_rounds` rounds.
"""

from typing import Callable

import gymnasium as gym
from gymnasium import spaces

from sap import config
from sap.battle import Outcome
from sap.errors import ActionError
from sap.game import Game
from sap.team import Team

from .actions import ACTIONS, apply_action, describe_action
from .masking import action_mask, is_legal
from .observation import OBS_HIGH, OBS_SIZE, encode_observation


class SapEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}
    describe_action = staticmethod(describe_action)

    def __init__(
        self,
        render_mode: str | None = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        draw_reward: float = 0.0,
        invalid_action_penalty: float = -0.05,
        max_shop_actions: int = 50,
        max_rounds: int = 40,
        opponent_fn: Callable[[Game], Team] | None = None,
    ):
        self.render_mode = render_mode
        self.rewards = {Outcome.WIN: win_reward, Outcome.LOSS: loss_reward, Outcome.DRAW: draw_reward}
        self.invalid_action_penalty = invalid_action_penalty
        self.max_shop_actions = max_shop_actions  # the round ends automatically after this many
        self.max_rounds = max_rounds
        self.opponent_fn = opponent_fn  # e.g. self-play; default: a random team (Game.make_opponent)

        self.actions = ACTIONS
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0.0, OBS_HIGH, shape=(OBS_SIZE,), dtype="float32")
        self.game: Game | None = None
        self.shop_actions = 0
        self.last_action: int | None = None  # last applied shop action this round (undo masking)

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.game = Game(seed=int(self.np_random.integers(2**31)))
        self.game.start("Agent")
        self.shop_actions = 0
        self.last_action = None
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.game is not None, "call reset() first"
        action = int(action)
        info: dict = {}
        reward = 0.0
        kind = self.actions[action][0]

        if not is_legal(self.game, action, self.last_action):
            reward += self.invalid_action_penalty
            info["invalid_action"] = True
        elif kind == "end_turn":
            reward += self._end_turn(info)
        else:
            try:
                apply_action(self.game, action)
                self.last_action = action
            except ActionError:  # should not happen: the mask mirrors the rules (tested)
                reward += self.invalid_action_penalty
                info["invalid_action"] = True

        if kind != "end_turn" and not self.game.is_over:
            self.shop_actions += 1
            if self.shop_actions >= self.max_shop_actions:
                reward += self._end_turn(info)
                info["forced_end_turn"] = True

        terminated = self.game.is_over
        truncated = not terminated and self.game.round > self.max_rounds
        info.update(self._info())
        if self.render_mode == "human":
            print(self.render())
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> str | None:
        g = self.game
        if g is None:
            return None
        return (
            f"Round {g.round} | Lives {g.lives} | Wins {g.wins}/{config.WIN_TARGET} | "
            f"Losses {g.losses} | Draws {g.draws}\n{g.store}\n{g.team}"
        )

    def action_masks(self):
        """The method sb3-contrib's MaskablePPO looks for."""
        return action_mask(self.game, self.last_action)

    # --- helpers ---------------------------------------------------------------
    def _end_turn(self, info: dict) -> float:
        opponent = self.opponent_fn(self.game) if self.opponent_fn else None
        result = self.game.end_turn(opponent)
        self.shop_actions = 0
        self.last_action = None  # new round: nothing to undo
        info["battle"] = result.outcome.value
        info["battle_log"] = result.log
        return self.rewards[result.outcome]

    def _info(self) -> dict:
        g = self.game
        return {
            "action_mask": self.action_masks(),
            "round": g.round, "lives": g.lives, "wins": g.wins, "credits": g.store.credits,
        }

    def _obs(self):
        return encode_observation(self.game, self.shop_actions, self.max_shop_actions, self.max_rounds)
