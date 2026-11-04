# Context — Super Auto Strat

A simplified, text-based auto-battler inspired by Super Auto Pets. The long-term goal is to let an
agent (e.g. a `gymnasium` environment) play the game, so **game logic must be separate from
input/output**: the engine exposes actions and state; a CLI (or an agent) drives it.

---

## 1. Game flow

```
START ──► STORE ──► FIGHT ──► (game over?) ──► STORE ...
```

1. **START** — player picks a team name. Starts with **5 lives**, **0 wins**, **0 losses**.
2. **STORE** — player gets **10 credits** and a fresh display, then takes actions until they choose to fight.
3. **FIGHT** — the team battles an opponent. Win → +1 win. Loss → +1 loss, −1 life.
4. **Game over** when lives reach 0 (defeat) or wins reach the target (victory). Otherwise back to STORE.

## 2. Rules (confirmed by the brief)

| Rule | Value |
|---|---|
| Starting lives | 5 |
| Credits per store visit | 10 |
| Pet cost | 3 (all pets) |
| Sell price | 1 (all pets) |
| Reroll cost | 1 |
| Items held per pet | max 1; buying a new one replaces the old one |
| Team order | matters; position 0 fights first |

### Store actions
- **Buy pet** — needs ≥ 3 credits and a free team slot. The display slot is left empty until the next reroll or store visit.
- **Buy item** — needs enough credits and a target pet on the team.
- **Reroll** — 1 credit; refills **every** display slot with new random choices.
- **Sell pet** — removes the pet (and its item) from the team; +1 credit.
- **Swap / move** — reorder the team freely, for free.
- **End turn** — leave the store and fight. Allowed at any time.

### Fight
- Only the pet at position 0 of each team fights.
- **Simultaneous strikes**: both front pets hit at the same moment (A deals its attack to B,
  B deals its attack to A), then both take damage, then faint checks run. A fainted pet is removed
  and the next one moves to the front.
- Repeat until one or both teams have no pets left.
- Damage is **deterministic** (= the pet's attack). No randomness in combat.

## 3. Assumptions (please confirm or correct)

| # | Topic | Proposed assumption |
|---|---|---|
| A1 | Leftover credits | **Not** carried over; every store visit resets to exactly 10. |
| A2 | Team size | Max **5** pets. |
| A3 | Display size | **3 pet slots + 2 item slots**. A fresh display is rolled at every store visit. |
| A4 | Item cost | **3 credits** (same as pets). |
| A5 | Item effect | Flat stat bonus applied **while held**. Replacing an item removes the old bonus. |
| A6 | Health after battle | Fights run on **copies** of the team; every pet returns at full stats for the next store. |
| A7 | Draw | Both teams wiped in the same exchange → **draw**: no win, no life lost. |
| A8 | Win target | **10 wins** (Super Auto Pets standard). The current code uses 5; either way it'll be a constant. |
| A9 | Opponent | A random team of pets from the pool, **the same size as the player's team** (min. 1). Later this can be another player or an agent. |
| A10 | Empty team | Fighting with no pets is an automatic loss. |
| A11 | Abilities | None for now: pets are just attack/health. The structure should make abilities easy to add later. |
| A12 | Interface | Terminal menu with numbered choices; the store screen always shows credits, lives, wins, the display and the team. |

### Starting pet pool (stats can be tuned; all cost 3)

| Pet | Attack | Health |
|---|---|---|
| Fish  | 2 | 3 |
| Dog   | 3 | 3 |
| Cat   | 4 | 2 |
| Snake | 5 | 1 |
| Fox   | 3 | 4 |
| Duck  | 1 | 5 |

### Starting item pool (all cost 3)

| Item | Effect while held |
|---|---|
| Apple  | +1 attack, +1 health |
| Bone   | +3 attack |
| Garlic | +3 health |

---

## 4. Target structure

```
sap/
  __init__.py
  config.py        # constants: lives, credits, costs, team size, win target
  errors.py        # ActionError: raised by any invalid action, state left unchanged
  pets.py          # Pet class + PET_POOL data table (name, attack, health)
  items.py         # Item class + ITEM_POOL data table
  team.py          # Team: fixed-size roster, add/remove/swap/move, alive pets
  store.py         # Store: display, credits, buy/sell/reroll (no input())
  battle.py        # simultaneous-strike battle on copied teams -> result + log
  game.py          # Game: state (lives, wins, stage) + action API, no I/O
  cli.py           # text interface: render state, read choices, call Game
main.py            # entry point: `python main.py`
tests/             # pytest tests for store, team, battle, game flow
```

Design principles:
- **No `input()`/`print()` inside the engine** (store, team, battle, game). Methods return results
  or raise or return clear errors; the CLI handles display. That keeps the engine usable by an agent.
- **Data-driven pets/items**: adding a pet means adding a row to a table, not writing a new class.
- **A seedable RNG** (`random.Random(seed)`) is passed to Game/Store, so runs can be reproduced for testing and RL.

---

## 5. Plan of tasks

1. **Scaffold**: add `sap/__init__.py`, `config.py`, `main.py`, `tests/`. Remove the empty `item.py`.
2. **Pets** (`pets.py`): replace `Animal` with `Pet(name, attack, health, item=None)`, with
   effective-stat properties that include the item bonus, `copy()`, `__str__`, plus `PET_POOL` and a `random_pet(rng)` helper.
3. **Items** (`items.py`): `Item(name, attack_bonus, health_bonus, cost)` + `ITEM_POOL`.
4. **Team** (`team.py`): roster capped at 5 (a compact list, no `None` padding), `add`, `remove(idx)`,
   `swap(i, j)`, `move(i, j)`, `is_full`, `copy()`, `__str__`.
5. **Store** (`store.py`): `new_visit()` (credits = 10, fresh display), `buy_pet`, `buy_item`,
   `reroll`, `sell`, each validating credits and slots and returning success or a reason.
6. **Battle** (`battle.py`): `fight(team_a, team_b) -> BattleResult(winner|draw, log)` using
   simultaneous strikes on copies.
7. **Game** (`game.py`): state machine START → STORE → FIGHT, lives/wins/losses, opponent
   generation, win/loss/draw handling, game-over detection. Replaces `environment.py`.
8. **CLI** (`cli.py` + `main.py`): menu loop showing status, display and team; actions: buy pet,
   buy item, reroll, sell, swap, fight, quit. Prints the battle log turn by turn.
9. **Tests**: pytest for purchase rules, item replacement, reroll cost, sell, swap, simultaneous
   damage, draw, and game-over conditions.
10. **Cleanup**: update `temp.ipynb` imports to the new modules. Add a short README section on how to run it.

Out of scope for now: pet abilities, tiers and leveling, frozen shop slots, and the gymnasium env
wrapper (the structure prepares for it).
