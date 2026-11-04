"""Game constants. Tune balance here."""

# Content pack: which pets and food are in the game (see sap/data/pets.csv and sap/items.py).
PACK = "Turtle Pack"

STARTING_LIVES = 5
WIN_TARGET = 10

CREDITS_PER_TURN = 10
PET_COST = 3
ITEM_COST = 3  # default food price; a few foods override it (e.g. Sleeping Pill)
SELL_PRICE_PER_LEVEL = 1  # a pet sells for level x this (1/2/3), plus any sell bonus (Cake)
REROLL_COST = 1

TEAM_SIZE = 5

# Shop slots grow with the rounds: {first round: slot count}.
# Pets: 3 slots, 4 from round 5, 5 from round 9. Food: 1 slot, 2 from round 3.
STORE_PET_SLOTS = {1: 3, 5: 4, 9: 5}
STORE_ITEM_SLOTS = {1: 1, 3: 2}
# Abilities, level-ups and frozen leftovers can add slots beyond these, up to these caps.
MAX_STORE_PET_SLOTS = 7
MAX_STORE_ITEM_SLOTS = 6

# Tier unlocks: a new pet/food tier opens every TIER_UNLOCK_ROUNDS rounds.
# Rounds 1-2 -> tier 1, rounds 3-4 -> tier 2, ..., round 11+ -> tier 6.
TIER_UNLOCK_ROUNDS = 2
MAX_TIER = 6

# Leveling: xp needed to reach level 2 and level 3. Each stacked copy adds 1 xp,
# so 3 copies -> level 2 and 6 copies -> level 3.
LEVEL_XP = (2, 5)
MAX_LEVEL = len(LEVEL_XP) + 1
MAX_XP = LEVEL_XP[-1]

# Attack and health can never exceed this.
MAX_STAT = 50

# Safety cap so a battle can never loop forever (e.g. two 0-attack pets).
MAX_BATTLE_TURNS = 100


def max_tier_for_round(round_number: int) -> int:
    return min(MAX_TIER, (max(round_number, 1) - 1) // TIER_UNLOCK_ROUNDS + 1)


def slots_for_round(schedule: dict[int, int], round_number: int) -> int:
    return max(count for first, count in schedule.items() if first <= max(round_number, 1))
