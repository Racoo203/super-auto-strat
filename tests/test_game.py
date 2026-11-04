import random

import pytest

from sap import config
from sap.battle import Outcome, fight
from sap.errors import ActionError
from sap.game import Game, Stage
from sap.pets import Pet, make_pet
from sap.store import Store
from sap.team import Team


def new_store(seed=0):
    store = Store(random.Random(seed))
    store.new_visit()
    return store


# --- team ---------------------------------------------------------------
def test_team_capped_at_five():
    team = Team("T")
    for _ in range(config.TEAM_SIZE):
        team.add(make_pet("Dog"))
    with pytest.raises(ActionError):
        team.add(make_pet("Dog"))


def test_swap_and_move():
    team = Team("T", [make_pet(n) for n in ("Fish", "Dog", "Cat")])
    team.swap(0, 2)
    assert [p.name for p in team] == ["Cat", "Dog", "Fish"]
    team.move(2, 0)
    assert [p.name for p in team] == ["Fish", "Cat", "Dog"]
    with pytest.raises(ActionError):
        team.swap(0, 3)


# --- store --------------------------------------------------------------
def test_new_visit_resets_credits():
    store = new_store()
    store.credits = 2
    store.new_visit()
    assert store.credits == config.CREDITS_PER_TURN
    assert len(store.pet_slots) == store.pet_slot_count == 3
    assert len(store.item_slots) == store.item_slot_count == 1


def test_buy_pet_costs_credits_and_empties_slot():
    store, team = new_store(), Team("T")
    pet = store.buy_pet(0, team)
    assert store.credits == config.CREDITS_PER_TURN - config.PET_COST
    assert team.pets == [pet]
    assert store.pet_slots[0] is None
    with pytest.raises(ActionError):
        store.buy_pet(0, team)  # already sold


def test_cannot_buy_without_credits_or_space():
    store, team = new_store(), Team("T")
    store.credits = config.PET_COST - 1
    with pytest.raises(ActionError):
        store.buy_pet(0, team)
    assert len(team) == 0 and store.pet_slots[0] is not None

    store.credits = 100
    full = Team("F", [make_pet("Dog") for _ in range(config.TEAM_SIZE)])
    with pytest.raises(ActionError):
        store.buy_pet(0, full)
    assert store.credits == 100


def test_new_perk_replaces_old_perk():
    store, team = new_store(), Team("T", [Pet("Dog", 3, 3)])
    store.stock("Honey", 3)
    store.stock("Garlic", 3)
    honey, garlic = len(store.item_slots) - 2, len(store.item_slots) - 1
    assert store.buy_item(honey, team, 0) is None
    assert team[0].perk.name == "Honey"
    assert store.buy_item(garlic, team, 0).name == "Honey"
    assert team[0].perk.name == "Garlic"


def test_eaten_item_gives_permanent_stats():
    store, team = new_store(), Team("T", [Pet("Dog", 3, 3)])
    store.stock("Apple", 3)
    store.buy_item(len(store.item_slots) - 1, team, 0)
    assert (team[0].attack, team[0].health, team[0].perk) == (4, 4, None)


def test_buy_item_needs_valid_pet():
    store, team = new_store(), Team("T")
    with pytest.raises(ActionError):
        store.buy_item(0, team, 0)
    assert store.credits == config.CREDITS_PER_TURN


def test_reroll_costs_one_and_refills_all_slots():
    store, team = new_store(), Team("T")
    store.buy_pet(0, team)
    store.reroll()
    assert store.credits == config.CREDITS_PER_TURN - config.PET_COST - config.REROLL_COST
    assert all(store.pet_slots) and all(store.item_slots)


def test_sell():
    store, team = new_store(), Team("T", [make_pet("Dog"), make_pet("Cat")])
    sold = store.sell(team, 0)
    assert sold.name == "Dog" and [p.name for p in team] == ["Cat"]
    assert store.credits == config.CREDITS_PER_TURN + config.SELL_PRICE_PER_LEVEL


# --- battle -------------------------------------------------------------
def test_simultaneous_strikes_can_draw():
    # 3/3 vs 3/3: both faint in the same exchange
    result = fight(Team("A", [Pet("Dog", 3, 3)]), Team("B", [Pet("Dog", 3, 3)]))
    assert result.outcome is Outcome.DRAW
    assert result.turns == 1


def test_next_pet_steps_up_and_damage_is_deterministic():
    a = Team("A", [Pet("Glass", 10, 1), Pet("Tank", 1, 10)])
    b = Team("B", [Pet("Mid", 2, 5)])
    # Turn 1: Glass and Mid trade, both faint (10 >= 5, 2 >= 1). A still has Tank -> win.
    result = fight(a, b)
    assert result.outcome is Outcome.WIN


def test_fight_does_not_mutate_teams():
    a = Team("A", [Pet("Glass", 5, 1)])
    b = Team("B", [Pet("Wall", 1, 5)])
    fight(a, b)
    assert a[0].health == 1 and b[0].health == 5


def test_loss():
    result = fight(Team("A", [Pet("Weak", 2, 3)]), Team("B", [Pet("Strong", 3, 4)]))
    assert result.outcome is Outcome.LOSS


# --- game ---------------------------------------------------------------
def test_game_start_state():
    game = Game(seed=1)
    game.start("Rauls")
    assert game.stage is Stage.STORE
    assert game.lives == config.STARTING_LIVES
    assert (game.wins, game.losses) == (0, 0)
    assert game.store.credits == config.CREDITS_PER_TURN


def test_loss_costs_a_life_and_empty_team_auto_loses():
    game = Game(seed=1)
    game.start("T")
    result = game.end_turn()
    assert result.outcome is Outcome.LOSS
    assert game.lives == config.STARTING_LIVES - 1
    assert game.round == 2 and game.store.credits == config.CREDITS_PER_TURN


def test_win_and_draw_bookkeeping():
    game = Game(seed=1)
    game.start("T")
    game.team.add(Pet("Big", 50, 50))
    game.end_turn(Team("E", [make_pet("Fish")]))
    assert game.wins == 1 and game.lives == config.STARTING_LIVES

    game.team.pets = [make_pet("Dog")]
    game.end_turn(Team("E", [make_pet("Dog")]))
    assert game.draws == 1 and game.lives == config.STARTING_LIVES


def test_game_over_on_zero_lives():
    game = Game(seed=1)
    game.start("T")
    for _ in range(config.STARTING_LIVES):
        game.end_turn()
    assert game.is_over and not game.won
    with pytest.raises(ActionError):
        game.reroll()


def test_game_over_on_win_target():
    game = Game(seed=1)
    game.start("T")
    game.team.add(Pet("Big", 50, 50))
    for _ in range(config.WIN_TARGET):
        game.end_turn()
    assert game.is_over and game.won
