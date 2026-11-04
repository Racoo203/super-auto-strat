from sap.config import TEAM_SIZE
from sap.errors import ActionError
from sap.pets import Pet


class Team:
    """An ordered roster of pets. Index 0 is the front (the pet that fights)."""

    def __init__(self, name: str, pets: list[Pet] | None = None, max_size: int = TEAM_SIZE):
        self.name = name
        self.max_size = max_size
        self.pets: list[Pet] = list(pets or [])
        if len(self.pets) > max_size:
            raise ValueError(f"A team holds at most {max_size} pets")

    def __len__(self):
        return len(self.pets)

    def __iter__(self):
        return iter(self.pets)

    def __getitem__(self, idx: int) -> Pet:
        self._check(idx)
        return self.pets[idx]

    @property
    def is_full(self) -> bool:
        return len(self.pets) >= self.max_size

    @property
    def front(self) -> Pet | None:
        return self.pets[0] if self.pets else None

    def _check(self, idx: int) -> None:
        if not 0 <= idx < len(self.pets):
            raise ActionError(f"No pet at position {idx + 1}")

    def add(self, pet: Pet) -> None:
        if self.is_full:
            raise ActionError("Team is full")
        self.pets.append(pet)

    def remove(self, idx: int) -> Pet:
        self._check(idx)
        return self.pets.pop(idx)

    def swap(self, i: int, j: int) -> None:
        self._check(i)
        self._check(j)
        self.pets[i], self.pets[j] = self.pets[j], self.pets[i]

    def move(self, src: int, dst: int) -> None:
        """Take the pet at `src` and insert it at `dst`, shifting the others."""
        self._check(src)
        self._check(dst)
        self.pets.insert(dst, self.pets.pop(src))

    def remove_fainted(self) -> list[Pet]:
        fainted = [p for p in self.pets if p.fainted]
        self.pets = [p for p in self.pets if not p.fainted]
        return fainted

    def copy(self) -> "Team":
        return Team(self.name, [p.copy() for p in self.pets], self.max_size)

    def __str__(self):
        lines = [f"[+] {self.name} [+]"]
        for i in range(self.max_size):
            lines.append(f"  [{i + 1}] {self.pets[i] if i < len(self.pets) else '(empty)'}")
        return "\n".join(lines)
