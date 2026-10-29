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
    return eps.groupby(list(dict.fromkeys(["agents_type", "n", xcol]))).agg(
        episodes=("success", "size"), success_rate=("success", "mean"), mean_turns=("turns", "mean"),
        mean_turns_success=("turns", lambda s: s[eps.loc[s.index, "success"]].mean()),
        normalized_turns=("normalized_turns", "mean"), oracle_turns=("oracle_turns", "mean"),
        messages=("total_messages", "mean"), tokens=("total_tokens", "mean"),
        invalid_rate=("invalid_count", lambda s: s.sum() / (eps.loc[s.index, "turns"]
                                                            * eps.loc[s.index, "n"]).sum()),
        consistency=("consistency", "mean"), bisection_score=("bisection_score", "mean"),
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
    label_table(eps).to_csv(out / f"{name}_strategy_labels.csv")
    eps.drop(columns=["graphs_turn1"], errors="ignore").to_csv(out / f"{name}_episode_features.csv",
                                                               index=False)
    base = load_experiment(a.baselines).episodes if a.baselines else None
    plot_performance.plot_success_at_k(eps, f"{name}_success_at_k", out)
    plot_performance.plot_turns_vs_x(eps, base, f"{name}_turns", out)
    plot_strategy.plot_labels(eps, f"{name}_strategy_labels", out)
    plot_strategy.plot_trajectories(run, eps, name=f"{name}_trajectories", outdir=out)
    if eps["m"].nunique() > 1:
        ctrl = load_experiment(a.controller).episodes if a.controller else None
        plot_clique.plot_turns_vs_m(eps, ctrl, f"{name}_turns_vs_m", out)
        plot_clique.plot_clique_vs_outsiders(eps, f"{name}_clique_vs_outsiders", out)
    print(f"wrote report for {name} to {out}")


if __name__ == "__main__":
    main()
