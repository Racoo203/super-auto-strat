import random
from enum import Enum

from sap import config
from sap.battle import BattleResult, Outcome, fight
from sap.errors import ActionError
from sap.items import Item
from sap.pets import Pet, random_pet
from sap.store import Store
from sap.team import Team


class Stage(Enum):
    START = "start"
    STORE = "store"
    OVER = "over"
    # The FIGHT stage runs inside `end_turn()`, so it is never a resting state.


class Game:
    """Game state and action API. No input()/print(): a CLI or an agent drives it."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.store = Store(self.rng)
        self.team: Team | None = None
        self.lives = config.STARTING_LIVES
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.round = 0
        self.stage = Stage.START
        self.last_result: BattleResult | None = None

    # --- lifecycle -------------------------------------------------------
    def start(self, team_name: str) -> None:
        if self.stage is not Stage.START:
            raise ActionError("Game already started")
        self.team = Team(team_name.strip() or "My Team")
        self._begin_round()

    def _begin_round(self) -> None:
        self.round += 1
        self.store.new_visit()
        self.stage = Stage.STORE

    def _require_store(self) -> None:
        if self.stage is not Stage.STORE:
            raise ActionError(f"Not in the store (stage: {self.stage.value})")

    @property
    def is_over(self) -> bool:
        return self.stage is Stage.OVER

    @property
    def won(self) -> bool:
        return self.wins >= config.WIN_TARGET

    # --- store actions ---------------------------------------------------
    def buy_pet(self, slot: int) -> Pet:
        self._require_store()
        return self.store.buy_pet(slot, self.team)

    def buy_item(self, slot: int, pet_idx: int) -> Item | None:
        self._require_store()
        return self.store.buy_item(slot, self.team, pet_idx)

    def reroll(self) -> None:
        self._require_store()
        self.store.reroll()

    def sell(self, pet_idx: int) -> Pet:
        self._require_store()
        return self.store.sell(self.team, pet_idx)

    def swap(self, i: int, j: int) -> None:
        self._require_store()
        self.team.swap(i, j)

    def move(self, src: int, dst: int) -> None:
        self._require_store()
        self.team.move(src, dst)

    # --- fight -----------------------------------------------------------
    def make_opponent(self) -> Team:
        size = max(1, len(self.team))
        return Team("Opponent", [random_pet(self.rng) for _ in range(size)])

    def end_turn(self, opponent: Team | None = None) -> BattleResult:
        """Leave the store and fight, apply the result, then start the next round or end the game."""
        self._require_store()
        opponent = opponent or self.make_opponent()

        if not self.team:
            result = BattleResult(Outcome.LOSS, 0, ["Your team has no pets: automatic loss."])
        else:
            result = fight(self.team, opponent)

        if result.outcome is Outcome.WIN:
            self.wins += 1
        elif result.outcome is Outcome.LOSS:
            self.losses += 1
            self.lives -= 1
        else:
            self.draws += 1

        self.last_result = result
        if self.lives <= 0 or self.won:
            self.stage = Stage.OVER
        else:
            self._begin_round()
        return result
