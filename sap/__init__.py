"""Super Auto Strat: a small Super Auto Pets-style auto-battler engine."""

from sap.errors import ActionError
from sap.game import Game, Stage
from sap.battle import BattleResult, Outcome, fight

__all__ = ["ActionError", "Game", "Stage", "BattleResult", "Outcome", "fight"]
