# Super Auto Strat

A text-based auto-battler inspired by Super Auto Pets. Rules and design notes are in [Context.md](Context.md).

## Play

```
python main.py
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
game.buy_item(0, pet_idx=0)
result = game.end_turn()   # fights, updates lives/wins, opens the next store
print(result.outcome, game.lives, game.wins)
```

Invalid actions raise `sap.errors.ActionError` and leave the state unchanged.
