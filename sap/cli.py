"""Terminal interface. All input()/print() lives here; the engine stays I/O-free."""

from sap import config
from sap.errors import ActionError
from sap.game import Game

MENU = (
    "Actions:\n"
    f"  [1] Buy pet   [2] Buy item   [3] Reroll ({config.REROLL_COST}c)\n"
    "  [4] Sell pet  [5] Swap pets  [6] Move pet\n"
    "  [7] FIGHT!    [0] Quit"
)


def ask_index(prompt: str, count: int) -> int | None:
    """Ask for a 1-based number in [1, count]. Returns a 0-based index, or None if cancelled."""
    while True:
        raw = input(f"{prompt} (1-{count}, Enter to cancel): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print("  Invalid choice.")


def render(game: Game) -> None:
    print("\n" + "=" * 44)
    print(
        f"Round {game.round} | Lives: {game.lives} | "
        f"Wins: {game.wins}/{config.WIN_TARGET} | Losses: {game.losses} | Draws: {game.draws}"
    )
    print("=" * 44)
    print(game.store)
    print()
    print(game.team)
    print()
    print(MENU)


def do_action(game: Game, choice: str) -> str | None:
    """Run one store action. Returns a message, or None if the player cancelled."""
    store, team = game.store, game.team
    if choice == "1":
        slot = ask_index("Pet slot", len(store.pet_slots))
        if slot is None:
            return None
        return f"Bought {game.buy_pet(slot)}."
    if choice == "2":
        slot = ask_index("Item slot", len(store.item_slots))
        if slot is None or not team:
            return None if slot is None else "You have no pets to give items to."
        pet_idx = ask_index("Give to pet at position", len(team))
        if pet_idx is None:
            return None
        replaced = game.buy_item(slot, pet_idx)
        msg = f"{team[pet_idx].name} now holds {team[pet_idx].item.name}."
        return msg + (f" ({replaced.name} was replaced.)" if replaced else "")
    if choice == "3":
        game.reroll()
        return "Store rerolled."
    if choice == "4":
        if not team:
            return "You have no pets to sell."
        idx = ask_index("Sell pet at position", len(team))
        return None if idx is None else f"Sold {game.sell(idx).name} for {config.SELL_PRICE}c."
    if choice in ("5", "6"):
        if len(team) < 2:
            return "You need at least two pets to reorder."
        i = ask_index("From position", len(team))
        j = None if i is None else ask_index("To position", len(team))
        if j is None:
            return None
        (game.swap if choice == "5" else game.move)(i, j)
        return "Team reordered."
    return "Unknown option."


def store_phase(game: Game) -> bool:
    """Loop over store actions. Returns True to fight, False to quit."""
    while True:
        render(game)
        choice = input("> ").strip()
        if choice == "7":
            return True
        if choice == "0":
            return False
        try:
            msg = do_action(game, choice)
        except ActionError as err:
            msg = f"Can't do that: {err}"
        if msg:
            print(f"\n>> {msg}")


def main(seed: int | None = None) -> None:
    game = Game(seed)
    try:
        game.start(input("Choose your team name! "))
        while not game.is_over:
            if not store_phase(game):
                print("Bye!")
                return
            print("\n" + "#" * 15 + " FIGHT " + "#" * 15)
            result = game.end_turn()
            print("\n".join(result.log))
            input("\nPress Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        return

    print("\nYou won the game!!!" if game.won else "\nYou ran out of lives :(")
    print(f"Final record: {game.wins}W / {game.losses}L / {game.draws}D")
