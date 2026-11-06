"""NoteBoard: messages posted during discussion, with reader-side visibility rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm_teams.comms.graph import CommGraph

# structured message format: token -> canonical text posted on the board
MESSAGE_TOKENS: dict[str, str] = {
    "raise": "I will raise",
    "lower": "I will lower",
    "hold": "I will hold",
}
STRUCTURED_TEXTS = frozenset(MESSAGE_TOKENS.values())


@dataclass(frozen=True)
class Note:
    author: int
    after_turn: int  # 0 = planning, before turn 1
    round: int
    text: str
    slot: int = 0  # speaking position within the round (always 0 when discussion is simultaneous)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NoteBoard:
    def __init__(self) -> None:
        self.notes: list[Note] = []

    def post(self, after_turn: int, round: int, author: int, text: str, slot: int = 0) -> None:
        self.notes.append(Note(author, after_turn, round, text, slot))

    def visible_to(
        self,
        reader: int,
        graph: CommGraph,
        before: tuple[int, int, int],
        history: str = "all",
    ) -> list[Note]:
        """Notes from the reader's message-neighbors (and itself) posted strictly before
        (after_turn, round, slot) = `before`. With simultaneous discussion every slot is 0, so a
        round never sees itself. history='last_turn' keeps only the latest after_turn."""
        allowed = graph.neighbors_in(reader) | {reader}
        vis = [n for n in self.notes
               if n.author in allowed and (n.after_turn, n.round, n.slot) < before]
        if history == "last_turn" and vis:
            latest = max(n.after_turn for n in vis)
            vis = [n for n in vis if n.after_turn == latest]
        return sorted(vis, key=lambda n: (n.after_turn, n.round, n.slot, n.author))

    def readers_of(self, note: Note, graph: CommGraph) -> list[int]:
        return sorted(graph.neighbors_out(note.author))

    def render_markdown(self, after_turn: int, round: int, graph: CommGraph) -> str:
        lines = [f"# After turn {after_turn}, round {round}", ""]
        notes = sorted((n for n in self.notes if n.after_turn == after_turn and n.round == round),
                       key=lambda n: (n.slot, n.author))
        sequential = any(n.slot for n in notes)
        for n in notes:
            order = f"speaker {n.slot + 1}, " if sequential else ""
            lines += [f"## Agent {n.author} ({order}read by {self.readers_of(n, graph)})", "", n.text, ""]
        return "\n".join(lines)


TOKEN_OF_TEXT = {text: token for token, text in MESSAGE_TOKENS.items()}
TALLY_KEYS = {"raise": "raised", "lower": "lowered", "hold": "held"}


def tally(notes: list[Note]) -> list[dict[str, Any]]:
    """Per after_turn (rounds summed): how many structured notes said raise / lower / hold."""
    out: dict[int, dict[str, Any]] = {}
    for n in sorted(notes, key=lambda n: n.after_turn):
        row = out.setdefault(n.after_turn, {"after_turn": n.after_turn,
                                            **{k: 0 for k in TALLY_KEYS.values()}})
        token = TOKEN_OF_TEXT.get(n.text)
        if token:
            row[TALLY_KEYS[token]] += 1
    return list(out.values())
