# llm-teams

[![attested by humans](https://github.com/kevinqnb/llm-teams/actions/workflows/git-signoff.yml/badge.svg)](https://github.com/kevinqnb/llm-teams/actions/workflows/git-signoff.yml)

Harness for testing teams of LLM agents on group tasks while varying how much the agents can
see of, and talk to, each other. See the implementation plan for the design.

## Setup

```
uv sync
uv run pytest
uv run llm-teams --help
```

## Running experiments

All experiment configs run through `experiments/submit.sh`:

```
bash experiments/submit.sh experiments/experiment-configs/sum_scripted_baselines.yaml --local
bash experiments/submit.sh experiments/experiment-configs/sum_smoke.yaml --dry-run   # mock backend, no GPU
```

### LLM smoke test (needs a GPU node)

vLLM is an optional extra and is not installed by default. On a GPU node:

```
uv sync --extra vllm
bash experiments/submit.sh experiments/experiment-configs/sum_smoke.yaml --local
uv run llm-teams summarize experiments/results/sum_smoke/<run_dir>
```

`sum_smoke.yaml` runs n=4, 5 episodes, with action-graph p in {0, 1}. Check that
`generations.jsonl` has the expected prompts and a low invalid rate before running larger sweeps.

### Collaboration mode

`sum_clique.yaml` uses a message graph (`generator: clique`, `p` = fraction of agents in the one clique)
and an empty action graph. After each turn's feedback, clique members each write one message that their
clique-mates read; everyone else makes no discussion calls. Notes are written to
`results/<exp>/<run>/notes/episode_<id>/after_turn_<k>_round_<r>.md`.
`sum_clique_controller.yaml` is the scripted reference (perfectly coordinated clique, random outsiders)
and runs without a GPU. `sum_clique_smoke.yaml` is the GPU smoke test (p ∈ {0.5, 1.0}).

### Sweeps and cluster submission

A config's `sweep:` block is a cartesian product over dotted config keys. Inspect it with
`uv run llm-teams sweep <cfg> --list` / `--count`. `experiments/submit.sh` submits one SGE array job
(`qsub -t 1-N`), one task per sweep condition by default, or `--shards N` to group conditions
(task `i` runs conditions with index mod N = i-1). Preview without submitting:

```
bash experiments/submit.sh experiments/experiment-configs/sum_clique.yaml --print \
    --gpus 1 --gpu-type A40 --runtime 04:00:00 -- --set dataset.num_episodes=20
```

Cluster-specific values come from the environment: `LLM_TEAMS_SGE_PROJECT`, `LLM_TEAMS_SGE_CORES`,
`LLM_TEAMS_GPUS`, `LLM_TEAMS_GPU_TYPE`, `LLM_TEAMS_GPU_MEMORY`, `LLM_TEAMS_RUNTIME`, `LLM_TEAMS_PRELUDE`
(e.g. `module load cuda`), `LLM_TEAMS_UV_ARGS`, `UV_CACHE_DIR`, `HF_HOME`; see `--help`.
Logs go to `experiments/results/<experiment>/logs/`, and each submission is appended to
`experiments/results/<experiment>/submissions.jsonl`. Re-submitting resumes unfinished conditions.

## Analysis

Needs the analysis extra: `uv sync --extra analysis`. Outputs go to `analysis/results/` (PDF + PNG + CSV).

```
uv run python -m analysis.report experiments/results/<experiment> \
    --baselines experiments/results/sum_scripted_baselines [--controller experiments/results/sum_clique_controller]
```

- `analysis/metrics.py` — pure metric functions (success@k / Kaplan–Meier, consistent interval, bisection
  score, info gain, step profile, strategy classifier with documented `Thresholds`, flips, over-correction,
  clique-vs-outsider behaviour, plan parsing, stated-strategy tags).
- `analysis/load.py`, `features.py` — run outputs to DataFrames and a per-episode feature table.
- `plot_performance.py`, `plot_strategy.py`, `plot_clique.py`, `report.py` — figures and tables.
- `judge_strategies.py` — optional LLM-judge labelling of notes (`--mock` for a dry run).
