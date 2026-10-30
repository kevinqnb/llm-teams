import json
import os
import shutil
import subprocess

import pytest

from llm_teams.config import repo_root

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")

ROOT = repo_root()
SUBMIT = ROOT / "experiments/submit.sh"
BASELINES = ROOT / "experiments/experiment-configs/sum_scripted_baselines.yaml"


def sh(args, tmp_path, extra_env=None, check=True):
    env = {**os.environ, "LLM_TEAMS_RESULTS_DIR": str(tmp_path / "results"), **(extra_env or {})}
    for k in ("LLM_TEAMS_SGE_PROJECT", "LLM_TEAMS_GPU_TYPE", "LLM_TEAMS_PRELUDE"):
        env.pop(k, None) if not extra_env or k not in extra_env else None
    return subprocess.run(["bash", str(SUBMIT), *map(str, args)], capture_output=True, text=True,
                          env=env, check=check, cwd=tmp_path)


def test_dry_run_end_to_end(tmp_path):
    r = sh([BASELINES, "--dry-run", "--", "--set", "dataset.n=4"], tmp_path)
    assert "ran 3 episodes" in r.stdout
    runs = list((tmp_path / "results/sum_scripted_baselines").glob("*/episodes.jsonl"))
    assert len(runs) == 12 and all(len(p.read_text().splitlines()) == 3 for p in runs)


def test_print_array_job(tmp_path):
    r = sh([BASELINES, "--print", "--shards", "3", "--gpus", "2", "--gpu-type", "A40", "--gpu-memory",
            "40", "--runtime", "01:30:00", "--", "--set", "seed=7"], tmp_path,
           {"LLM_TEAMS_SGE_PROJECT": "someproj", "HF_HOME": "/hf/cache"})
    out = r.stdout
    for frag in ("qsub", "-t 1-3", "-N sum_scripted_baselines", "-l h_rt=01:30:00", "-l gpus=2",
                 "-l gpu_type=A40", "-l gpu_memory=40", "-P someproj", "--shard $SGE_TASK_ID/3",
                 "--set seed=7", "export HF_HOME=/hf/cache", "set -euo pipefail"):
        assert frag in out, frag
    assert not (tmp_path / "results").exists()  # --print has no side effects


def test_no_overrides_adds_no_stray_argument(tmp_path):
    last = sh([BASELINES, "--print"], tmp_path).stdout.strip().splitlines()[-1]
    assert last.endswith("--shard $SGE_TASK_ID/12") and "''" not in last


def test_default_one_task_per_condition_and_shard_clamp(tmp_path):
    assert "-t 1-12" in sh([BASELINES, "--print"], tmp_path).stdout
    r = sh([BASELINES, "--print", "--shards", "50"], tmp_path)
    assert "-t 1-12" in r.stdout and "exceeds" in r.stderr


def test_no_optional_flags_when_unset(tmp_path):
    out = sh([BASELINES, "--print", "--gpus", "0"], tmp_path, {"LLM_TEAMS_SGE_CORES": "0"}).stdout
    for frag in ("-P ", "gpu_type", "gpu_memory", "gpus=", "-pe"):
        assert frag not in out.splitlines()[1], frag


def test_submit_with_stub_qsub_records_submission(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub = bin_dir / "qsub"
    qsub.write_text(f'#!/usr/bin/env bash\necho "$@" > {tmp_path}/qsub_args\necho "4242.1-3:1"\n')
    qsub.chmod(0o755)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}"}
    sh([BASELINES, "--shards", "3"], tmp_path, env)
    assert "-t 1-3" in (tmp_path / "qsub_args").read_text()
    rec = json.loads((tmp_path / "results/sum_scripted_baselines/submissions.jsonl").read_text())
    assert rec["job_id"] == "4242" and rec["shards"] == 3 and rec["git_commit"]
    (job,) = (tmp_path / "results/sum_scripted_baselines/logs").glob("job_*.sh")
    assert "--shard $SGE_TASK_ID/3" in job.read_text()


def test_errors(tmp_path):
    r = sh([tmp_path / "missing.yaml"], tmp_path, check=False)
    assert r.returncode != 0 and "config not found" in r.stderr
    r = sh([BASELINES, "--shards", "x"], tmp_path, check=False)
    assert r.returncode != 0 and "positive integer" in r.stderr
    r = sh([BASELINES, "--nope"], tmp_path, check=False)
    assert r.returncode != 0 and "unknown option" in r.stderr
    assert "Usage:" in sh(["--help"], tmp_path).stdout
