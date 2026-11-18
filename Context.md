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
| Sell price | 1 per level (1/2/3) |
| Reroll cost | 1 |
| Perks held per pet | max 1; a new perk replaces the old one |
| Team order | matters; position 0 fights first |

### Store actions
- **Buy pet** — needs ≥ 3 credits and a free team slot. The display slot is left empty until the next reroll or store visit.
- **Buy item** — needs enough credits; targeted food needs a pet on the team. Eaten food applies
  its effect; perk food gives the pet that perk.
- **Reroll** — 1 credit; refills **every** display slot with new random choices.
- **Sell pet** — fires the pet's Sell ability, removes it (and its perk); + its level (+ Cake bonus) in credits.
- **Stack** — buy a pet onto the same pet on the team (allowed even when the team is full), or
  **combine** two of the same pet already on the team (free). Cannot be undone.
- **Swap / move** — reorder the team freely, for free.
- **Freeze / unfreeze** a shop pet or item slot — free. Frozen slots survive rerolls and the next
  store visit (moved to the front); buying a frozen slot clears the freeze.
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
| A3 | Display size | Grows with rounds: pets **3** (4 from round 5, 5 from round 9), food **1** (2 from round 3). Frozen slots are kept; the rest is re-rolled every visit. |
| A4 | Item cost | **3 credits** (Sleeping Pill 1; ability-stocked food can be cheaper or free). |
| A5 | Item effect | Items are Super Auto Pets food: eaten for an effect, or held as a perk. See §7. |
| A6 | Health after battle | Fights run on **copies** of the team; every pet returns at full stats for the next store. |
| A7 | Draw | Both teams wiped in the same exchange → **draw**: no win, no life lost. |
| A8 | Win target | **10 wins** (Super Auto Pets standard). The current code uses 5; either way it'll be a constant. |
| A9 | Opponent | A random team of pets from the pool, **the same size as the player's team** (min. 1), or a **ghost**: a snapshot of a team another player (agent) built at the same round (`sap/ghosts.py`, §9). |
| A10 | Empty team | Fighting with no pets is an automatic loss. |
| A11 | Abilities | Trigger → effect, scaled by level. See §6. |
| A12 | Interface | Terminal menu with numbered choices; the store screen always shows credits, lives, wins, the display and the team. |

Pets and food come from the configured pack (`config.PACK`, default Turtle Pack): see §6 and §7.

---

## 4. Target structure

```
sap/
  __init__.py
  config.py        # constants: lives, credits, costs, team size, win target
  errors.py        # ActionError: raised by any invalid action, state left unchanged
  pets.py          # PetSpec/Pet + PET_POOL loaded from data/pets.csv; leveling
  perks.py         # Perk data (Honey, Garlic, Melon, Coconut, Peanut...)
  items.py         # shop food: ITEM_POOL, eaten effects, feed()
  events.py        # Trigger, Event, ability registry, Context + dispatcher
  abilities.py     # every pet (and perk) ability
  data/pets.csv    # scraped pet table (all packs)
  env/             # gymnasium environment (optional: needs gymnasium + numpy)
    actions.py     #   Discrete action table, labels, apply_action(game, a)
    masking.py     #   is_legal / action_mask (mirror of the store rules) + undo masking
    observation.py #   Game -> float32 vector, observation_layout()
    sap_env.py     #   SapEnv: wires the gym API; __init__.py registers SuperAutoStrat-v0
  team.py          # Team: fixed-size roster, add/remove/swap/move, alive pets
  store.py         # Store: display, credits, buy/sell/reroll (no input())
  battle.py        # simultaneous-strike battle on copied teams -> result + log
  game.py          # Game: state (lives, wins, stage) + action API, no I/O
  ghosts.py        # Ghost, GhostPool, GhostOpponent: asynchronous opponents (self-play)
  cli.py           # text interface: render state, read choices, call Game
agents/
  ppo/             # PPO: config, network, envs (parallel actors), rollout, buffer,
                   # advantages (GAE), losses, tracking, policies (seat -> policy),
                   # selfplay (ghost pool glue); one concept per module
                   # + train, experiments (Optuna/TensorBoard), evaluate, replay, watch, tune
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

Opponents smarter than a random team plug in via `opponent_fn` (self-play ghosts: §9).

---

## 6. Leveling and abilities

### Leveling
- Each stacked copy adds 1 xp (a stacked pet brings its own xp too). **3 copies → level 2,
  6 copies → level 3** (max). Thresholds: `config.LEVEL_XP = (2, 5)`.
- A stacked pet keeps the **higher attack and higher health** of the two, **+1/+1**, and keeps a
  perk if either had one.

### Pet data
- `sap/data/pets.csv` (restored scrape, all packs). The pool is the pack named by
  `config.PACK` (default **Turtle Pack**, 61 pets).
- **Tier unlocks**: a new tier opens every `config.TIER_UNLOCK_ROUNDS` (2) rounds: rounds 1–2 →
  tier 1, 3–4 → tier 2, 5–6 → tier 3 … round 11+ → tier 6. The shop's pets *and* food, and random
  opponents, only roll unlocked tiers. Stats are capped at `config.MAX_STAT` (50).
- **Level-up bonus**: whenever a pet levels up in the shop (stacking or Chocolate), one pet from the
  tier *above* the unlocked one (max tier 6) is added to the shop as an extra slot.
- Known data issue: the scraped level-3 ability text repeats level 2's. It is display-only; the
  code is the source of truth.

### Abilities (`sap/events.py`, `sap/abilities.py`)
- Every ability is **trigger + effect**, registered with `@ability("Name", Trigger.X)`; a pet may
  register several (Whale: swallow at start of battle, release on faint). Effects receive the level
  to act at, so a Parrot (copy) or Tiger (repeat) can run them at a different level.
- Triggers: start of battle, before/after attack, friend ahead attacks, hurt, friend hurt, faint,
  friend faints, friend ahead faints, knock out, summoned, friend summoned, buy, friend bought, sell,
  level up, eats food, friendly ate food, start/end of turn. Cat and Tiger are passive rules.
- Effects act through a `Context` (`damage`, `reduce_health`, `buff`, `give_perk`, `summon`,
  `gain_gold`, `stock`, `ahead`/`behind`, `random_friends`/`random_enemies`, `use` for "N times
  per turn/battle") and may cause further events; a queue resolves the chain.
- Several pets reacting to the same event act **highest attack first**, ties broken by the seeded rng.
- Battle changes land on the battle copies; store changes are permanent. "Until end of battle"
  buffs bought in the shop (Cupcake, Horse, Dog) are temporary stats cleared after each fight.
- Buying a pet into a slot also counts as **summoning** it (Horse, Dog, Turkey, Scorpion react).
- Usage limits reset at the start of each turn and of each battle.
- Scaling: level 3 = 3× level 1 (the scrape only has trustworthy level 1 and 2 text).
- **All 61 Turtle Pack pets are implemented** (Sloth has no ability).

---

## 7. Items (food) and perks

Shop items are Super Auto Pets food (`sap/items.py`). Values from the Super Auto Pets wiki.

| Tier | Eaten (instant) | Perk (held) |
|---|---|---|
| 1 | Apple +1/+1 | Honey: faint → 1/1 Bee |
| 2 | Cupcake +3/+3 until end of battle; Sleeping Pill (1c): faint | Meat Bone: +3 attack damage; Lime: −1 damage taken (min 1) |
| 3 | Salad Bowl: 2 random +1/+1 | Garlic: −2 damage taken (min 1); Cake: end turn → +1 sell value |
| 4 | Canned Food: shop pets +1/+1 now and later; Pear +2/+2 | — |
| 5 | Chocolate +1 xp; Sushi: 3 random +1/+1 | Chili: also hit 2nd enemy for 5 |
| 6 | Pizza: 2 random +2/+2 | Melon: −20 once; Mushroom: come back 1/1; Steak: +20 once; Pancakes: start of battle → friends +2/+2 |
| — (stocked by abilities) | Bread Crumbs +1/+0 (Pigeon); Milk +1/+2, Better +2/+4, Best +3/+6 (Cow); Apple / Better Apple +2/+2 / Best Apple +3/+3 (Worm)* | Coconut: ignore a hit once (Gorilla); Peanut: knock out what it attacks and hurts (Scorpion) |

\* Better/Best Apple values are assumed by analogy with Milk.

- A Cat multiplies food stats (×2/×3/×4), twice per turn.
- Perk damage rules apply to all damage (attacks and abilities); Peanut and Chili only on attacks.

---

## 8. Gymnasium environment (`sap/env/`)

`import sap.env` registers **`SuperAutoStrat-v0`**. The agent plays the shop; battles resolve
automatically when it ends the turn.

- **Actions**: one `Discrete(132)` space covering buy pet (into a free slot, or stacked onto a team
  pet), buy item (onto a team pet; untargeted food uses target 0), sell, combine, move, freeze pet,
  freeze item, reroll and end turn. `info["action_mask"]` / `env.action_masks()` marks the legal
  ones (compatible with sb3-contrib `MaskablePPO`). Illegal actions change nothing and cost
  `invalid_action_penalty` (-0.05). `SapEnv.describe_action(a)` gives a readable label.
- **Observation**: flat `float32` vector (1146): globals (round, lives, wins, credits, tier, shop
  actions used), then 5 team pets, 7 shop pets and 6 shop items, each as present flag + one-hot
  species/perk/item + scaled stats + frozen flag. `observation_layout()` gives the index ranges.
- **Reward**: +1 battle won, -1 lost, 0 draw (constructor arguments).
- **Episode**: ends when the game ends; truncated after `max_rounds` (40). A round is ended
  automatically after `max_shop_actions` (50) shop actions, so an agent can't stall.
- **Opponent**: random team of the round's tiers by default; pass `opponent_fn(game) -> Team` for
  self-play (`GhostOpponent`) or scripted opponents. Every team that fights is kept in
  `env.snapshots` (`pop_snapshots()`), and `info["opponent_source"]` says whose ghost was fought
  (-1 = random team).
- **Rendering**: `render_mode="ansi"` makes `render()` return a text frame (`sap/env/render.py`):
  status line, last action and reward, the last battle (opponent team and log) when a turn just
  ended, the store and the team. `"human"` prints it every step. `render(policy=...)` adds an overlay
  of the agent's top actions (used by `agents/ppo/watch.py`).
- Tests check the mask against the real rules: every allowed action succeeds and every forbidden
  one is refused.

---

## 9. Self-play: ghosts, seats and policies

As in Super Auto Pets, players never wait for each other: at round r a player fights a **ghost**,
a frozen copy of a team some player had at round r. So K agents need no lockstep lobby:

- **Seats** are the N vector envs, each an ordinary single-agent `SapEnv` (no idle seats, no
  masked waiting steps, GAE unchanged).
- **`Game.end_turn`** is three atomic steps: `finish_shop()` (end-of-turn abilities, returns the
  team that fights), `play_battle(opponent)`, `record_result(result)`. The env snapshots the team
  between the first two.
- **`GhostPool`** (`sap/ghosts.py`) keeps, per round, the newest `recent` ghosts (FIFO) and a
  reservoir of `history` older ones: opponents from past policies, a league at no extra cost.
  `sample(round, ...)` takes the exact round, else the nearest earlier one; empty teams are never
  ghosts. `GhostOpponent` is the `opponent_fn`, seeded by the game's rng, falling back to a random
  team when nothing matches (cold start) or with `random_prob`.
- **Policies**: `seat_policy[i]` says which policy plays env i (`agents/ppo/policies.py`,
  `PolicySet`: one batched forward per policy). `n_policies=1` is shared-policy self-play,
  `n_policies=2` with `opponents="others"` is two agents trained against each other, P > 2 a league.
- **Per update** (`agents/ppo/selfplay.py`): rollout, `harvest` (`envs.call("pop_snapshots")` ->
  ghosts tagged with the seat's policy), `push_opponents` (`envs.set_attr("opponent_fn", ...)`,
  one GhostOpponent per seat), then one PPO update per policy on its own buffer columns.
  Opponents are at most one update old.
- **Evaluation**: `eval/*` stays against random teams (a fixed yardstick: self-play win rates drift
  toward 50%); `eval/vs_ghosts/*` against the live pool; `eval/h2h_p0_vs_p1/*` policy 0 against
  policy 1's ghosts (`evaluate.head_to_head`). Against an equal opponent whole games are rarely won,
  so watch `battle_win_rate` there (50% = even).
