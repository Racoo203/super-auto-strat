"""Freezing, slot growth by round, and the level-up bonus pet."""

import random

import pytest

from sap import config
from sap.errors import ActionError
from sap.game import Game
from sap.pets import make_pet
from sap.store import Store
from sap.team import Team


def new_store(round_number=1, seed=0):
    store = Store(random.Random(seed))
    store.round = round_number
    store.max_tier = config.max_tier_for_round(round_number)
    store.new_visit()
    return store


# --- slot growth ----------------------------------------------------------------
@pytest.mark.parametrize("rnd, pets, items", [(1, 3, 1), (2, 3, 1), (3, 3, 2), (5, 4, 2), (8, 4, 2), (9, 5, 2), (20, 5, 2)])
def test_shop_slots_grow_with_rounds(rnd, pets, items):
    store = new_store(rnd)
    assert (len(store.pet_slots), len(store.item_slots)) == (pets, items)


def test_game_passes_round_to_store():
    game = Game(seed=0)
    game.start("T")
    for _ in range(4):
        game.end_turn()
    assert game.round == 5 and len(game.store.pet_slots) == 4


# --- freezing ---------------------------------------------------------------------
def test_frozen_pet_and_item_survive_reroll():
    store = new_store(3)
    pet, item = store.pet_slots[1], store.item_slots[1]
    assert store.freeze_pet(1) and store.freeze_item(1)
    store.reroll()
    assert store.pet_slots[0] is pet and store.item_slots[0] is item  # moved to the front
    assert len(store.pet_slots) == 3 and len(store.item_slots) == 2
    assert store.is_frozen(pet) and store.is_frozen(item)


def test_frozen_slots_survive_into_next_round():
    game = Game(seed=1)
    game.start("T")
    pet = game.store.pet_slots[2]
    game.freeze_pet(2)
    game.end_turn()
    assert game.store.pet_slots[0] is pet


def test_unfreeze_toggles_back():
    store = new_store()
    pet = store.pet_slots[0]
    assert store.freeze_pet(0) is True
    assert store.freeze_pet(0) is False
    store.reroll()
    assert all(p is not pet for p in store.pet_slots)


def test_buying_a_frozen_pet_clears_the_freeze():
    store, team = new_store(), Team("T")
    pet = store.pet_slots[0]
    store.freeze_pet(0)
    store.buy_pet(0, team)
    assert not store.is_frozen(pet)
    store.reroll()
    assert len(store.pet_slots) == 3


def test_cannot_freeze_a_sold_slot():
    store, team = new_store(), Team("T")
    store.buy_pet(0, team)
    with pytest.raises(ActionError):
        store.freeze_pet(0)


def test_frozen_pet_keeps_its_shop_buffs():
    store, team = new_store(), Team("T", [make_pet("Duck")])
    store.freeze_pet(0)
    store.sell(team, 0)  # Duck: shop pets +1 health
    health = store.pet_slots[0].health
    store.reroll()
    assert store.pet_slots[0].health == health


# --- level-up bonus pet --------------------------------------------------------------
def test_level_up_adds_a_pet_from_the_next_tier():
    store = new_store(1)
    team = Team("T", [make_pet("Sloth")])
    team[0].xp = 1
    store.pet_slots[0] = make_pet("Sloth")
    store.buy_pet(0, team, target=0)
    assert team[0].level == 2
    assert len(store.pet_slots) == 4 and store.pet_slots[-1].tier == 2
    assert any("Level-up bonus" in line for line in store.log)


def test_level_up_bonus_caps_at_tier_6():
    store = new_store(11)
    team = Team("T", [make_pet("Sloth"), make_pet("Sloth")])
    team[0].xp = 1
    store.combine(team, 1, 0)
    assert store.pet_slots[-1].tier == 6


def test_level_up_via_chocolate_also_adds_bonus_pet():
    store = new_store(9)
    team = Team("T", [make_pet("Sloth")])
    team[0].xp = 1
    store.stock("Chocolate", 3)
    store.buy_item(len(store.item_slots) - 1, team, 0)
    assert len(store.pet_slots) == 6


def test_no_bonus_pet_for_stacking_without_level_up():
    store = new_store(1)
    team = Team("T", [make_pet("Sloth"), make_pet("Sloth")])
    store.combine(team, 1, 0)
    assert len(store.pet_slots) == 3
