import copy

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from sap.env import ACTIONS, ENV_ID, OBS_SIZE, SapEnv  # noqa: E402
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
    freeze = ACTIONS.index(("freeze_pet", 0, None))
    for _ in range(2):
        *_, info = env.step(freeze)
        assert "forced_end_turn" not in info
    *_, info = env.step(freeze)
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
                probe._apply(a)
                ok = True
            except ActionError:
                ok = False
            except (IndexError, AttributeError, TypeError):
                ok = False  # slot doesn't exist
            kind, slot, target = ACTIONS[a]
            if ok and kind == "buy_item" and target != 0:
                # untargeted items (Salad Bowl...) are only offered once, as target 0
                ok = env.game.store.item_slots[slot].item.targeted
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
