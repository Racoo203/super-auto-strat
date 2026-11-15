import copy

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from sap.env import ACTIONS, ENV_ID, OBS_SIZE, SapEnv, apply_action, undoes_last  # noqa: E402
from sap.errors import ActionError  # noqa: E402


def test_passes_gymnasium_env_checker():
    check_env(SapEnv(), skip_render_check=True)


def test_registered_and_makeable():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,) and info["action_mask"].shape == (len(ACTIONS),)


def test_reset_is_reproducible_with_seed():
    a, _ = SapEnv().reset(seed=42)
    b, _ = SapEnv().reset(seed=42)
    assert np.array_equal(a, b)


def test_illegal_action_is_penalised_and_changes_nothing():
    env = SapEnv()
    obs, info = env.reset(seed=0)
    sell_first = ACTIONS.index(("sell", 0, None))  # empty team: illegal
    assert not info["action_mask"][sell_first]
    obs2, reward, *_ , info2 = env.step(sell_first)
    assert reward == env.invalid_action_penalty and info2["invalid_action"]
    assert np.array_equal(np.delete(obs, 5), np.delete(obs2, 5))  # only the action counter moved


def test_end_turn_gives_battle_reward():
    env = SapEnv()
    env.reset(seed=0)
    _, reward, *_ , info = env.step(ACTIONS.index(("end_turn", None, None)))
    assert info["battle"] == "loss" and reward == -1.0  # empty team loses


def test_round_ends_after_max_shop_actions():
    env = SapEnv(max_shop_actions=3)
    env.reset(seed=0)
    freezes = [ACTIONS.index(("freeze_pet", s, None)) for s in (0, 1, 0)]  # alternate: no undo
    for a in freezes[:2]:
        *_, info = env.step(a)
        assert "forced_end_turn" not in info
    *_, info = env.step(freezes[2])
    assert info["forced_end_turn"] and info["round"] == 2


@pytest.mark.parametrize("seed", range(3))
def test_mask_matches_the_real_rules(seed):
    """Every action the mask allows must succeed, and every one it forbids must be refused."""
    env = SapEnv()
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(15):
        mask = info["action_mask"]
        for a in range(len(ACTIONS)):
            if ACTIONS[a][0] == "end_turn":
                continue
            probe = copy.deepcopy(env)
            try:
                apply_action(probe.game, a)
                ok = True
            except ActionError:
                ok = False
            except (IndexError, AttributeError, TypeError):
                ok = False  # slot doesn't exist
            kind, slot, target = ACTIONS[a]
            if ok and kind == "buy_item" and target != 0:
                # untargeted items (Salad Bowl...) are only offered once, as target 0
                ok = env.game.store.item_slots[slot].item.targeted
            ok = ok and not undoes_last(a, env.last_action)  # undo moves are masked on purpose
            assert ok == mask[a], (SapEnv.describe_action(a), ok, mask[a])
        _, _, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        if term or trunc:
            _, info = env.reset(seed=seed + 100)


def test_random_masked_episodes_terminate():
    env = SapEnv()
    rng = np.random.default_rng(0)
    for ep in range(5):
        _, info = env.reset(seed=ep)
        for steps in range(5000):
            _, _, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(info["action_mask"]))))
            if term or trunc:
                break
        assert term or trunc


def test_render_shows_state():
    env = SapEnv(render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert "Store" in text and "Agent" in text


def test_undo_moves_and_double_freezes_are_masked():
    env = SapEnv()
    env.reset(seed=0)
    env.game.store.credits = 10
    for slot in range(2):  # a bought shop slot stays empty, so buy slots 0 and 1
        env.step(ACTIONS.index(("buy_pet", slot, None)))
    move_01, move_10 = ACTIONS.index(("move", 0, 1)), ACTIONS.index(("move", 1, 0))
    *_, info = env.step(move_01)
    # neighbours: 0->1 is a swap, so both 1->0 and a repeated 0->1 undo it
    assert not info["action_mask"][move_10] and not info["action_mask"][move_01]

    freeze0, freeze1 = ACTIONS.index(("freeze_pet", 2, None)), ACTIONS.index(("freeze_item", 0, None))
    *_, info = env.step(freeze0)
    assert not info["action_mask"][freeze0] and info["action_mask"][freeze1]
    assert info["action_mask"][move_10]  # the move is no longer the last action

    *_, info = env.step(ACTIONS.index(("end_turn", None, None)))
    assert env.last_action is None  # a new round forgets the last action


def test_non_adjacent_move_only_masks_its_reverse():
    env = SapEnv()
    env.reset(seed=0)
    env.game.store.credits = 10
    for slot in range(3):
        env.step(ACTIONS.index(("buy_pet", slot, None)))
    *_, info = env.step(ACTIONS.index(("move", 0, 2)))
    assert not info["action_mask"][ACTIONS.index(("move", 2, 0))]
    assert info["action_mask"][ACTIONS.index(("move", 0, 2))]  # [b,c,a] -> [c,a,b]: a new order
