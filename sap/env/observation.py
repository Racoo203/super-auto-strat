"""Game state -> flat float32 vector.

  globals    round (/max_rounds), lives, wins, credits, unlocked tier, shop actions used this round
  team       TEAM_SIZE pet blocks
  shop pets  MAX_STORE_PET_SLOTS pet blocks
  shop items MAX_STORE_ITEM_SLOTS item blocks
  pet block  present, species one-hot (+ "other" for summoned tokens), attack, health, level, xp,
             tier, perk one-hot, frozen
  item block present, item one-hot, price, frozen
"""

import numpy as np

from sap import config
from sap.game import Game
from sap.items import ITEM_POOL
from sap.perks import PERKS
from sap.pets import PET_POOL, Pet

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


def encode_pet(obs: np.ndarray, pos: int, pet: Pet | None, frozen: bool) -> None:
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


def encode_observation(game: Game, shop_actions: int, max_shop_actions: int, max_rounds: int) -> np.ndarray:
    g = game
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[:N_GLOBAL] = [
        g.round / max_rounds, g.lives / config.STARTING_LIVES, g.wins / config.WIN_TARGET,
        g.store.credits / 20, g.max_tier / config.MAX_TIER, shop_actions / max_shop_actions,
    ]
    pos = N_GLOBAL
    team = list(g.team)
    for i in range(T):
        encode_pet(obs, pos, team[i] if i < len(team) else None, frozen=False)
        pos += PET_BLOCK
    for i in range(P):
        pet = g.store.pet_slots[i] if i < len(g.store.pet_slots) else None
        encode_pet(obs, pos, pet, frozen=g.store.is_frozen(pet))
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
