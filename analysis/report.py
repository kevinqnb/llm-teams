"""One-shot report for an experiment: tables (CSV) and figures in analysis/results/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import plot_clique, plot_performance, plot_strategy
from analysis.common import RESULTS, x_column
from analysis.features import episode_features
from analysis.load import load_experiment
from analysis.plot_strategy import label_table


def performance_table(eps: pd.DataFrame) -> pd.DataFrame:
    xcol = x_column(eps)
    keys = ["agents_type", "message_format", "discussion_order", "board_view", "n", xcol]
    keys = [k for k in keys if k in eps]
    return eps.groupby(list(dict.fromkeys(keys))).agg(
        episodes=("success", "size"), success_rate=("success", "mean"), mean_turns=("turns", "mean"),
        mean_turns_success=("turns", lambda s: s[eps.loc[s.index, "success"]].mean()),
        normalized_turns=("normalized_turns", "mean"), oracle_turns=("oracle_turns", "mean"),
        messages=("total_messages", "mean"), tokens=("total_tokens", "mean"),
        invalid_rate=("invalid_count", lambda s: s.sum() / (eps.loc[s.index, "turns"]
                                                            * eps.loc[s.index, "n"]).sum()),
        consistency=("consistency", "mean"), bisection_score=("bisection_score", "mean"),
        invalid_messages=("invalid_messages", "mean"),
        intent_coherent=("intent_coherent_rate", "mean"), intent_follow=("intent_follow_rate", "mean"),
    ).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment results directory")
    ap.add_argument("--baselines")
    ap.add_argument("--controller")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = load_experiment(a.experiment)
    eps = episode_features(run)
    name = run.config["name"]
    performance_table(eps).to_csv(out / f"{name}_performance.csv", index=False)
    eps.drop(columns=["graphs_turn1"], errors="ignore").to_csv(out / f"{name}_episode_features.csv",
                                                               index=False)
    base = load_experiment(a.baselines).episodes if a.baselines else None
    # one set of tables/figures per communication setting, so e.g. free and structured, or
    # simultaneous and sequential, are never pooled
    split = [k for k in ("message_format", "discussion_order", "board_view") if eps[k].nunique() > 1]
    groups = [(v if isinstance(v, tuple) else (v,), sub) for v, sub in eps.groupby(split)] \
        if split else [((), eps)]
    for vals, sub in groups:
        tag = "_".join([name, *map(str, vals)])
        label_table(sub).to_csv(out / f"{tag}_strategy_labels.csv")
        plot_performance.plot_success_at_k(sub, f"{tag}_success_at_k", out)
        plot_performance.plot_turns_vs_x(sub, base, f"{tag}_turns", out)
        plot_strategy.plot_labels(sub, f"{tag}_strategy_labels", out)
        plot_strategy.plot_trajectories(run, sub, name=f"{tag}_trajectories", outdir=out)
        if sub["m"].nunique() > 1:
            ctrl = load_experiment(a.controller).episodes if a.controller else None
            plot_clique.plot_turns_vs_m(sub, ctrl, f"{tag}_turns_vs_m", out)
            plot_clique.plot_clique_vs_outsiders(sub, f"{tag}_clique_vs_outsiders", out)
    print(f"wrote report for {name} to {out}")


if __name__ == "__main__":
    main()
