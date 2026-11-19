# Super Auto Strat

A text-based auto-battler inspired by Super Auto Pets. Rules and design notes are in [Context.md](Context.md).

## Play

```
python main.py
```

## Install

The game itself needs only Python 3.10+. The RL stack (Python 3.13) is pinned in `pyproject.toml`
and `uv.lock`:

```
uv sync                                   # or, in a conda/venv env:
python -m pip install -r requirements.txt # generated from uv.lock
```

## Test

```
python -m pytest
```

## Use the engine from code (e.g. an agent)

```python
from sap.game import Game

game = Game(seed=0)
game.start("Rauls")
game.buy_pet(0)            # 0-based indexes
game.buy_pet(1, target=0)  # stack onto the same pet at position 0 (3 copies -> Lv2, 6 -> Lv3)
game.combine(1, 0)         # stack two team pets of the same kind
game.freeze_pet(2)         # keep shop pet 3 through rerolls and into next round
game.buy_item(0, pet_idx=0)  # food: eaten, or held as a perk
result = game.end_turn()   # fights, updates lives/wins, opens the next store
print(result.outcome, game.lives, game.wins)
```

Pet abilities live in `sap/abilities.py` (one decorated function per pet), food in `sap/items.py`,
perks in `sap/perks.py`. The content pack is `config.PACK` (default Turtle Pack). See Context.md §6–7.

Invalid actions raise `sap.errors.ActionError` and leave the state unchanged.

## Reinforcement learning

```python
import gymnasium as gym
import sap.env  # registers SuperAutoStrat-v0

env = gym.make("SuperAutoStrat-v0")
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample(mask=info["action_mask"].astype("int8"))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

With sb3-contrib, `MaskablePPO("MlpPolicy", env)` picks up `env.action_masks()` automatically.
See Context.md §8 for the action, observation and reward layout.

## Training and monitoring (PPO)

`agents/ppo/` is a from-scratch PPO; [PPO.ipynb](PPO.ipynb) walks through it.

```python
from agents.ppo import PPOConfig, train
result = train(PPOConfig(n_updates=300), "baseline")   # tracked run -> runs/baseline/
```

Opponents are **ghosts** by default (self-play, as in Super Auto Pets: each player fights a snapshot of a
team another player built at the same round; see Context.md §9):

```python
train(PPOConfig(n_updates=300), "selfplay")                  # one shared policy plays every seat
train(PPOConfig(n_updates=300, n_policies=2), "duel")        # two agents trained against each other
train(PPOConfig(n_updates=300, opponent="random"), "random") # the original: random opponent teams
```

Run `python -m agents.ppo.tune --trials 20` from a script rather than a notebook (async envs need the
`__main__` guard on Windows). For one long run meant to go for hours, use `agents/ppo/run.py` instead
of a notebook so it isn't tied to a kernel staying open, and so it can be resumed if interrupted:

```
python -m agents.ppo.run --name final --updates 5000
python -m agents.ppo.run --name final --updates 5000 --resume   # after a crash/interrupt
```

To train on a GPU server (Docker, automatic resume, dashboards over SSH), see
[deploy/README.md](deploy/README.md): `cp .env.example .env && make check test smoke up`.

| Question | Tool |
|---|---|
| How is it training? (live) | `tensorboard --logdir runs`: win rate, wins, return, entropy, KL, clip fraction |
| How did it play? | `python -m agents.ppo.replay runs/baseline` summarises the latest evaluation games; add `-e 0 --play [--battles-only]` to step through one |
| How is it playing? (live) | `python -m agents.ppo.watch runs/baseline --delay 0.5` plays a fresh game, showing the board, the last battle and the policy's top actions with V(s); `--vs runs/duel/checkpoints/latest_p1.pt` fights another policy's ghosts |
| Who beats whom? | TensorBoard `eval/vs_ghosts/*`, `eval/h2h_p0_vs_p1/*`; in code `head_to_head(model_a, model_b, 20)` |
| How do runs compare? | `optuna-dashboard sqlite:///runs/optuna.db`, or `load_runs()` (a DataFrame) in a notebook |
| Hyperparameter search | `python -m agents.ppo.tune --trials 20 --updates 200` (study `ppo-tune`, median pruning) |

Each run writes `runs/<run>/{metrics.jsonl, config.json, tb/, replays/, checkpoints/{latest,best,update_XXXX}.pt}`
and is one trial in `runs/optuna.db`. The objective is the best eval win rate, and crashed runs are kept and marked FAIL.
Evaluation games use fixed seeds, so evaluations of the same run are comparable over time.

The environment renders itself as text: `gym.make("SuperAutoStrat-v0", render_mode="ansi")` and `env.render()`
(or `render_mode="human"` to print every step).
