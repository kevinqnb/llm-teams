---
id: 2026-11-06-sequential-board-01
kind: build
---

## Session 2026-11-06

### Prompts
- Asked for a high-level overview of how the message board works for agents, with an example single-turn conversation.
- Asked whether agents see each other's responses before deciding to raise or lower (they did not: posting was simultaneous within a round).
- Brainstormed turn-based posting: each agent sees how many others have written so far, and either their free text or, for structured messages, a tally ("X raised, Y lowered, Z held"). Then: "The plan sounds good. I'm ready to go ahead and implement."
- "Commit and /devlog."

### Implemented
Added `comms.discussion_order` (`simultaneous` default, or `sequential`: one speaker at a time in a seeded random order reshuffled per turn and round, each seeing earlier same-round notes and their position) and `comms.board_view` (`named` default, or `aggregate`: counts of raise/lower/hold per turn, no authors; requires structured messages). Notes gain a `slot`, `speaking_order` is logged per turn, and defaults leave condition hashes and simultaneous behaviour unchanged. The prompt template, `analysis/` grouping and README were updated. New smoke configs `experiments/experiment-configs/sum_clique_sequential_smoke.yaml` (order x format) and `sum_clique_aggregate_smoke.yaml` (named vs aggregate); tests in `tests/test_sequential.py`. Not yet run on a real model.

### Commits
5b25559 feat: sequential board discussion and aggregate (counts-only) board view
