# Super Auto Strat

A text-based auto-battler inspired by Super Auto Pets. Rules and design notes are in [Context.md](Context.md).

## Play

```
python main.py
```

## Install

The game itself needs only Python 3.10+. For the RL environment and tests:

```
python -m pip install -r requirements.txt
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
