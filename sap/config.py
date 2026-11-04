"""Game constants. Tune balance here."""

STARTING_LIVES = 5
WIN_TARGET = 10

CREDITS_PER_TURN = 10
PET_COST = 3
ITEM_COST = 3
SELL_PRICE = 1
REROLL_COST = 1

TEAM_SIZE = 5
STORE_PET_SLOTS = 3
STORE_ITEM_SLOTS = 2

# Safety cap so a battle can never loop forever (e.g. two 0-attack pets).
MAX_BATTLE_TURNS = 100
