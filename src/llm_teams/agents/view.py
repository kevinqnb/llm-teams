"""The single place where information access is decided.

`build_view` assembles everything agent i may see at a given moment. Nothing else in the
code base hands information to agents, so leaks can only come from here (see tests).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from llm_teams.comms.graph import CommGraph
from llm_teams.comms.notes import NoteBoard, tally
from llm_teams.tasks.base import Task


@dataclass
class EpisodeState:
    """Public history of an episode. Never handed to agents directly."""

    n: int
    task: Task
    actions: dict[int, dict[int, int]] = field(default_factory=dict)  # turn -> agent -> action
    feedback: dict[int, dict[int, dict[str, Any]]] = field(default_factory=dict)
    noteboard: NoteBoard = field(default_factory=NoteBoard)
    action_graph: CommGraph | None = None
    message_graph: CommGraph | None = None


@dataclass
class ViewConfig:
    know_n: bool = True
    know_id: bool = True
    reveal_group: bool = True
    action_history: str = "last"
    notes_history: str = "all"
    message_format: str = "free"
    board_view: str = "named"
    discussion_order: str = "simultaneous"
    max_turns: int = 20


@dataclass
class AgentView:
    phase: str  # "act" | "discuss" | "planning"
    turn: int  # act: turn being chosen (1-based); discuss: turn just completed (0 = planning)
    round: int  # discussion round (0 for act)
    self_id: int | None
    n: int | None
    max_turns: int
    task_info: dict[str, Any]
    own_actions: list[int]  # one per completed turn
    feedback_history: list[dict[str, Any]]  # this agent's feedback, one per completed turn
    action_neighbors: list[int]
    visible_actions: dict[int, dict[int, int]]  # completed turn -> neighbor -> action
    message_neighbors: list[int]
    group: dict[str, Any] | None  # {"members": [...], "outside": k} or {"members": []} if isolated
    notes: list[dict[str, Any]]
    message_format: str = "free"  # "free" | "structured"
    discussion_order: str = "simultaneous"  # "simultaneous" | "sequential"
    board_view: str = "named"  # "named" (notes) | "aggregate" (tallies only; notes stays empty)
    tallies: list[dict[str, Any]] = field(default_factory=list)  # aggregate: counts per turn
    # sequential discussion only: {"position", "total", "passed", "posted"} -- this agent is
    # speaker `position` of `total` group members; `passed` earlier group members have had their
    # turn this round and `posted` of them actually posted a valid note.
    progress: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_view(
    agent_id: int,
    state: EpisodeState,
    cfg: ViewConfig,
    phase: str,
    turn: int,
    round: int = 0,
    slot: int = 0,
    order: list[int] | None = None,
) -> AgentView:
    """`slot`/`order` are the speaker's position and the full speaking order in a sequential
    discussion round; simultaneous discussion and the act phase leave them at the defaults."""
    assert state.action_graph is not None and state.message_graph is not None
    completed = turn - 1 if phase == "act" else turn
    done = range(1, completed + 1)

    a_nb = sorted(state.action_graph.neighbors_in(agent_id))
    m_nb = sorted(state.message_graph.neighbors_in(agent_id))
    shown_turns = [completed] if cfg.action_history == "last" else list(done)
    visible = {
        t: {j: state.actions[t][j] for j in a_nb}
        for t in shown_turns
        if t >= 1 and t in state.actions
    }

    group: dict[str, Any] | None = None
    if cfg.reveal_group:
        members = state.message_graph.group_of(agent_id)
        members = members if len(members) > 1 else []
        group = {"members": members, "outside": state.n - len(members) if members else None}

    notes = state.noteboard.visible_to(
        agent_id,
        state.message_graph,
        # act at turn k reads everything posted after turns < k; (k, 0) sorts before (k, 1).
        # discussion after turn k, round r reads rounds < r of turn k and all earlier turns, plus
        # (sequential only) notes of the same round from speakers with a smaller slot.
        before=(turn, round, slot),
        history=cfg.notes_history,
    )

    progress = None
    if order is not None:
        earlier = [j for j in order[:order.index(agent_id)] if j in m_nb]
        posted = sum(1 for nt in notes
                     if (nt.after_turn, nt.round) == (turn, round) and nt.author in earlier)
        progress = {"position": len(earlier) + 1, "total": len(m_nb) + 1,
                    "passed": len(earlier), "posted": posted}
    aggregate = cfg.board_view == "aggregate"

    return AgentView(
        phase=phase,
        turn=turn,
        round=round,
        self_id=agent_id if cfg.know_id else None,
        n=state.n if cfg.know_n else None,
        max_turns=cfg.max_turns,
        task_info=state.task.describe(agent_id),
        own_actions=[state.actions[t][agent_id] for t in done],
        feedback_history=[state.feedback[t][agent_id] for t in done],
        action_neighbors=a_nb,
        visible_actions=visible,
        message_neighbors=m_nb,
        group=group,
        notes=[] if aggregate else [n.to_dict() for n in notes],  # aggregate never exposes authors
        message_format=cfg.message_format,
        discussion_order=cfg.discussion_order,
        board_view=cfg.board_view,
        tallies=tally(notes) if aggregate else [],
        progress=progress,
    )
