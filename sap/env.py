"""Gymnasium environment: an agent plays the shop phase; battles resolve automatically.

    import gymnasium as gym
    import sap.env  # registers "SuperAutoStrat-v0"
    env = gym.make("SuperAutoStrat-v0")
    obs, info = env.reset(seed=0)
    action = env.action_space.sample(mask=info["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)

Actions: one Discrete space covering every shop action (see `SapEnv.actions` and
`describe_action`). Most actions are illegal in a given state; `info["action_mask"]` (and
`env.action_masks()`, the method sb3-contrib's MaskablePPO looks for) marks the legal ones. An
illegal action changes nothing and costs `invalid_action_penalty`.

Observation: a flat float32 vector (see `observation_layout`):
  globals    round (/max_rounds), lives, wins, credits, unlocked tier, shop actions used this round
  team       TEAM_SIZE pet blocks
  shop pets  MAX_STORE_PET_SLOTS pet blocks
  shop items MAX_STORE_ITEM_SLOTS item blocks
  pet block  present, species one-hot (+ "other" for summoned tokens), attack, health, level, xp,
             tier, perk one-hot, frozen
  item block present, item one-hot, price, frozen

Reward: +1 for a won battle, -1 for a lost one, 0 for a draw (tunable).
The episode ends when the game is over (out of lives or WIN_TARGET wins), and is truncated after
`max_rounds` rounds.
"""

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sap import config
from sap.battle import Outcome
from sap.errors import ActionError
from sap.game import Game
from sap.items import ITEM_POOL
from sap.perks import PERKS
from sap.pets import PET_POOL, Pet
from sap.team import Team

T = config.TEAM_SIZE
P = config.MAX_STORE_PET_SLOTS
I = config.MAX_STORE_ITEM_SLOTS

SPECIES = list(PET_POOL)  # + one "other" bucket for tokens (Zombie Cricket, Ram...)
PERK_NAMES = list(PERKS)
ITEM_NAMES = list(ITEM_POOL)

N_GLOBAL = 6
PET_BLOCK = 1 + len(SPECIES) + 1 + 5 + len(PERK_NAMES) + 1
ITEM_BLOCK = 1 + len(ITEM_NAMES) + 2
OBS_SIZE = N_GLOBAL + (T + P) * PET_BLOCK + I * ITEM_BLOCK
OBS_HIGH = 5.0  # features are scaled to ~[0, 1]; values are clipped to [0, OBS_HIGH]


def _build_actions() -> list[tuple]:
    acts: list[tuple] = []
    acts += [("buy_pet", s, None) for s in range(P)]  # into a free team slot
    acts += [("buy_pet", s, t) for s in range(P) for t in range(T)]  # stack onto team[t]
    acts += [("buy_item", s, t) for s in range(I) for t in range(T)]  # untargeted items: t = 0
    acts += [("sell", t, None) for t in range(T)]
    acts += [("combine", a, b) for a in range(T) for b in range(T) if a != b]
    acts += [("move", a, b) for a in range(T) for b in range(T) if a != b]
    acts += [("freeze_pet", s, None) for s in range(P)]
    acts += [("freeze_item", s, None) for s in range(I)]
    acts += [("reroll", None, None), ("end_turn", None, None)]
    return acts


ACTIONS = _build_actions()


def observation_layout() -> dict[str, tuple[int, int]]:
    """Name -> (start, end) index ranges in the observation vector."""
    start, layout = 0, {}
    for name, size in [
        ("globals", N_GLOBAL), ("team", T * PET_BLOCK), ("shop_pets", P * PET_BLOCK),
        ("shop_items", I * ITEM_BLOCK),
    ]:
        layout[name] = (start, start + size)
        start += size
    return layout


class SapEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

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
        self.observation_space = spaces.Box(0.0, OBS_HIGH, shape=(OBS_SIZE,), dtype=np.float32)
        self.game: Game | None = None
        self.shop_actions = 0

    # --- gym API -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.game = Game(seed=int(self.np_random.integers(2**31)))
        self.game.start("Agent")
        self.shop_actions = 0
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.game is not None, "call reset() first"
        action = int(action)
        info: dict = {}
        reward = 0.0
        kind = self.actions[action][0]

        if not self._is_legal(action):
            reward += self.invalid_action_penalty
            info["invalid_action"] = True
        elif kind == "end_turn":
            reward += self._end_turn(info)
        else:
            try:
                self._apply(action)
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
        text = (
            f"Round {g.round} | Lives {g.lives} | Wins {g.wins}/{config.WIN_TARGET} | "
            f"Losses {g.losses} | Draws {g.draws}\n{g.store}\n{g.team}"
        )
        return text

    def action_masks(self) -> np.ndarray:
        return np.array([self._is_legal(a) for a in range(len(self.actions))], dtype=bool)

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def describe_action(action: int) -> str:
        kind, a, b = ACTIONS[int(action)]
        def n(i):
            return i + 1 if i is not None else None  # 1-based, as on screen

        return {
            "buy_pet": f"buy shop pet {n(a)}" + (f" onto team pet {n(b)}" if b is not None else ""),
            "buy_item": f"buy shop item {n(a)} for team pet {n(b)}",
            "sell": f"sell team pet {n(a)}",
            "combine": f"stack team pet {n(a)} onto team pet {n(b)}",
            "move": f"move team pet {n(a)} to position {n(b)}",
            "freeze_pet": f"freeze/unfreeze shop pet {n(a)}",
            "freeze_item": f"freeze/unfreeze shop item {n(a)}",
        }.get(kind, kind.replace("_", " "))

    def _end_turn(self, info: dict) -> float:
        opponent = self.opponent_fn(self.game) if self.opponent_fn else None
        result = self.game.end_turn(opponent)
        self.shop_actions = 0
        info["battle"] = result.outcome.value
        info["battle_log"] = result.log
        return self.rewards[result.outcome]

    def _apply(self, action: int) -> None:
        kind, a, b = self.actions[action]
        g = self.game
        if kind == "buy_pet":
            g.buy_pet(a, b)
        elif kind == "buy_item":
            item = g.store.item_slots[a].item
            g.buy_item(a, b if item.targeted else None)
        elif kind == "sell":
            g.sell(a)
        elif kind == "combine":
            g.combine(a, b)
        elif kind == "move":
            g.move(a, b)
        elif kind == "freeze_pet":
            g.freeze_pet(a)
        elif kind == "freeze_item":
            g.freeze_item(a)
        elif kind == "reroll":
            g.reroll()

    def _is_legal(self, action: int) -> bool:
        """Mirror of the store rules, without touching the game (kept in sync by tests)."""
        g = self.game
        if g is None or g.is_over:
            return False
        kind, a, b = self.actions[action]
        store, team = g.store, g.team
        n = len(team)
        if kind == "end_turn":
            return True
        if kind == "reroll":
            return store.credits >= config.REROLL_COST
        if kind == "sell":
            return a < n
        if kind == "move":
            return a < n and b < n
        if kind == "combine":
            return a < n and b < n and team[a].name == team[b].name and not team[b].max_level
        if kind in ("buy_pet", "freeze_pet"):
            pet = store.pet_slots[a] if a < len(store.pet_slots) else None
            if pet is None:
                return False
            if kind == "freeze_pet":
                return True
            if store.credits < config.PET_COST:
                return False
            if b is None:
                return not team.is_full
            return b < n and team[b].name == pet.name and not team[b].max_level
        if kind in ("buy_item", "freeze_item"):
            slot = store.item_slots[a] if a < len(store.item_slots) else None
            if slot is None:
                return False
            if kind == "freeze_item":
                return True
            if store.credits < slot.price:
                return False
            return b < n if slot.item.targeted else b == 0
        return False

    # --- encoding ---------------------------------------------------------------
    def _info(self) -> dict:
        g = self.game
        return {
            "action_mask": self.action_masks(),
            "round": g.round, "lives": g.lives, "wins": g.wins, "credits": g.store.credits,
        }

    def _obs(self) -> np.ndarray:
        g = self.game
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[:N_GLOBAL] = [
            g.round / self.max_rounds, g.lives / config.STARTING_LIVES, g.wins / config.WIN_TARGET,
            g.store.credits / 20, g.max_tier / config.MAX_TIER,
            self.shop_actions / self.max_shop_actions,
        ]
        pos = N_GLOBAL
        team = list(g.team)
        for i in range(T):
            self._encode_pet(obs, pos, team[i] if i < len(team) else None, frozen=False)
            pos += PET_BLOCK
        for i in range(P):
            pet = g.store.pet_slots[i] if i < len(g.store.pet_slots) else None
            self._encode_pet(obs, pos, pet, frozen=g.store.is_frozen(pet))
            pos += PET_BLOCK
        for i in range(I):
            slot = g.store.item_slots[i] if i < len(g.store.item_slots) else None
            if slot is not None:
                obs[pos] = 1
                obs[pos + 1 + ITEM_NAMES.index(slot.item.name)] = 1
                obs[pos + 1 + len(ITEM_NAMES)] = slot.price / config.ITEM_COST
                obs[pos + 2 + len(ITEM_NAMES)] = float(g.store.is_frozen(slot))
            pos += ITEM_BLOCK
        return np.clip(obs, 0.0, OBS_HIGH)

    @staticmethod
    def _encode_pet(obs: np.ndarray, pos: int, pet: Pet | None, frozen: bool) -> None:
        if pet is None:
            return
        obs[pos] = 1
        species = SPECIES.index(pet.name) if pet.name in PET_POOL else len(SPECIES)
        obs[pos + 1 + species] = 1
        k = pos + 2 + len(SPECIES)
        obs[k:k + 5] = [
            pet.attack / config.MAX_STAT, pet.health / config.MAX_STAT,
            pet.level / config.MAX_LEVEL, pet.xp / config.MAX_XP, pet.tier / config.MAX_TIER,
        ]
        if pet.perk is not None:
            obs[k + 5 + PERK_NAMES.index(pet.perk.name)] = 1
        obs[k + 5 + len(PERK_NAMES)] = float(frozen)


ENV_ID = "SuperAutoStrat-v0"
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="sap.env:SapEnv")
