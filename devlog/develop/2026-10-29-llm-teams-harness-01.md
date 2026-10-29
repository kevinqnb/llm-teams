---
id: 2026-10-29-llm-teams-harness-01
config:
---

## Session 2026-10-29

### Prompts
- Pasted a full implementation plan for `llm-teams`: a harness for teams of local LLM agents (shared
  Qwen3.5-4B via vLLM) on group tasks, varying how much agents see of and talk to each other. The graph
  is a first-class object (action-visibility graph and message graph, built by registered generators);
  collaboration is one random clique; turn structure is act, feedback, discuss, act; scripted agents
  share the Agent interface; episodes are generators batched in lockstep; everything is logged as
  append-only JSONL; first task is `sum_to_target`. Work through milestones M0-M5 in order, stopping
  after each with a summary and any deviations.
- "start M1", "start M2", "start M3", "start M4", "start M5", one per milestone, after reviewing each
  summary.
- "Good, commit this work and write a /developlog note." Chose one commit per milestone (approximated
  by dependency groups, see below) and the id `2026-10-29-llm-teams-harness-01`.

### Implemented
Package `llm_teams` (task, comms graphs/generators/notes, a single `build_view` access point, scripted
and LLM agents, mock and vLLM backends, lockstep runner with resume, CLI), `experiments/submit.sh` for
local runs and SGE array jobs, and an `analysis/` package (pure metrics, loaders, plots, strategy report,
optional LLM judge). Entry points are `llm-teams run|sweep|validate-configs|summarize` and
`experiments/submit.sh <config> [--local|--dry-run|--print]`. 82 tests pass without a GPU, including
information-leak, turn-ordering, lockstep-equals-serial, submit.sh (stub `qsub`) and analysis acceptance
tests (the oracle sits at log2). Not yet run: the vLLM backend and the GPU smoke configs
(`sum_smoke`, `sum_clique_smoke`), since the dev node had no GPU. Deviations from the plan: the optimal
bound is `ceil(log2(n+2))` rather than `ceil(log2(n+1))`; the bisection score measures distance to the
nearest integer midpoint; `agents.memory: transcript` is not implemented; rounding for `m`/`k` is
half-up.

### Commits
- 8d4b06e chore: scaffold uv project, package layout and tooling (M0)
- 16d99e1 feat: core harness and LLM agents (M1, M2)
- 0b78c1c feat: collaboration configs and clique-controller reference tests (M3)
- 62b016c feat: SGE array-job submission via experiments/submit.sh (M4)
- 5ce2313 feat: analysis metrics, loaders, plots, report and strategy judge (M5)

Note: M1 and M2 share one commit, and the M3 agent code sits in the M1/M2 commit, because several files
(`cli.py`, `runner/run.py`, `agents/scripted.py`, `README.md`) span milestones and each commit had to
import cleanly.
