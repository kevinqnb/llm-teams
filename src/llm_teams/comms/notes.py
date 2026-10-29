"""NoteBoard: messages posted during discussion, with reader-side visibility rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm_teams.comms.graph import CommGraph


@dataclass(frozen=True)
class Note:
    author: int
    after_turn: int  # 0 = planning, before turn 1
    round: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NoteBoard:
    def __init__(self) -> None:
        self.notes: list[Note] = []

    def post(self, after_turn: int, round: int, author: int, text: str) -> None:
        self.notes.append(Note(author, after_turn, round, text))

    def visible_to(
        self,
        reader: int,
        graph: CommGraph,
        before: tuple[int, int],
        history: str = "all",
    ) -> list[Note]:
        """Notes from the reader's message-neighbors (and itself) posted strictly before
        (after_turn, round) = `before`. history='last_turn' keeps only the latest after_turn."""
        allowed = graph.neighbors_in(reader) | {reader}
        vis = [n for n in self.notes if n.author in allowed and (n.after_turn, n.round) < before]
        if history == "last_turn" and vis:
            latest = max(n.after_turn for n in vis)
            vis = [n for n in vis if n.after_turn == latest]
        return sorted(vis, key=lambda n: (n.after_turn, n.round, n.author))

    def readers_of(self, note: Note, graph: CommGraph) -> list[int]:
        return sorted(graph.neighbors_out(note.author))

    def render_markdown(self, after_turn: int, round: int, graph: CommGraph) -> str:
        lines = [f"# After turn {after_turn}, round {round}", ""]
        for n in self.notes:
            if n.after_turn == after_turn and n.round == round:
                lines += [f"## Agent {n.author} (read by {self.readers_of(n, graph)})", "", n.text, ""]
        return "\n".join(lines)
