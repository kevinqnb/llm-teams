"""`llm-teams` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_teams import __version__
from llm_teams.config import build_conditions, parse_override, validate_all


def _overrides(items: list[str] | None) -> dict:
    return dict(parse_override(s) for s in items or [])


def _shard(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    i, n = (int(x) for x in s.split("/"))
    if not 1 <= i <= n:
        raise SystemExit(f"--shard must be i/N with 1 <= i <= N, got {s}")
    return i, n


def cmd_run(a: argparse.Namespace) -> int:
    from llm_teams.runner.run import run_experiment

    run_experiment(a.config, _overrides(a.set), a.dry_run, _shard(a.shard), argv=sys.argv)
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    conds = build_conditions(a.config, _overrides(a.set))
    if a.name:
        print(conds[0][1].name)
    elif a.count:
        print(len(conds))
    else:
        for i, (point, _) in enumerate(conds, 1):
            print(f"{i}\t{json.dumps(point)}")
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    if a.configs:
        errors = []
        for c in a.configs:
            try:
                build_conditions(c)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{c}: {e}")
    else:
        errors = validate_all()
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    if not errors:
        print("all configs valid")
    return 1 if errors else 0


def cmd_summarize(a: argparse.Namespace) -> int:
    from llm_teams.runner.run import summarize

    print(json.dumps(summarize(Path(a.run_dir)), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-teams", description="Experiments on teams of LLM agents."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("run", help="run an experiment config")
    p.add_argument("config")
    p.add_argument("--set", action="append", metavar="key=value")
    p.add_argument("--dry-run", action="store_true", help="mock backend, few episodes")
    p.add_argument("--shard", metavar="i/N", help="run only conditions with index %% N == i-1")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("sweep", help="list or count sweep conditions")
    p.add_argument("config")
    p.add_argument("--set", action="append", metavar="key=value")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--name", action="store_true", help="print the experiment name")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("validate-configs", help="load every config in experiments/")
    p.add_argument("configs", nargs="*", help="specific experiment configs (default: all)")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("summarize", help="print summary of a run directory")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
