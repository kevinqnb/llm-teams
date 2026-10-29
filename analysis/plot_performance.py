"""Performance vs communication: success@k curves and turns vs p, with scripted baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import metrics as M
from analysis.common import BASELINE_STYLE, PALETTE, RESULTS, baseline_means, save, x_column
from analysis.features import episode_features
from analysis.load import load_experiment


def plot_success_at_k(eps: pd.DataFrame, name: str = "success_at_k", outdir: Path | None = None):
    xcol = x_column(eps)
    if xcol == "n":
        xcol = "agents_type"
    ns = sorted(eps["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.4), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], ns, strict=True):
        sub = eps[eps["n"] == n]
        kmax = int(sub["max_turns"].iloc[0])
        for color, (x, g) in zip(PALETTE, sub.groupby(xcol), strict=False):
            ax.plot(range(1, kmax + 1), M.success_at_k(list(g["turns"]), list(g["success"]), kmax),
                    color=color, label=f"{xcol}={x}")
        ax.axvline(sub["optimal_turns"].iloc[0], color="#888", ls=":", lw=1)
        ax.set(title=f"n={n}", xlabel="turns k", ylim=(0, 1.02))
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("success within k turns")
    return save(fig, name, outdir)


def plot_turns_vs_x(eps: pd.DataFrame, baselines: pd.DataFrame | None = None,
                    name: str = "turns_vs_p", outdir: Path | None = None):
    xcol = x_column(eps)
    base = baseline_means(baselines)
    ns = sorted(eps["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.4), squeeze=False, sharey=False)
    for ax, n in zip(axes[0], ns, strict=True):
        sub = eps[eps["n"] == n]
        g = sub.groupby(xcol)["turns"].agg(["mean", "sem"]).fillna(0)
        xs = [str(i) for i in g.index] if xcol == "agents_type" else g.index
        ax.errorbar(xs, g["mean"], yerr=1.96 * g["sem"], color=PALETTE[0], marker="o",
                    label="agents (censored at max_turns)")
        ax.axhline(sub["oracle_turns"].mean(), color=BASELINE_STYLE["oracle"][0], ls="-", lw=1,
                   label="oracle bisection")
        ax.axhline(sub["optimal_turns"].iloc[0], color="#888", ls=":", lw=1, label="worst-case optimum")
        for kind in ("id_bisection", "random"):
            if (kind, n) in base:
                c, ls = BASELINE_STYLE[kind]
                ax.axhline(base[(kind, n)][0], color=c, ls=ls, lw=1, label=kind.replace("_", " "))
        ax.set(title=f"n={n}", xlabel=xcol, ylabel="mean turns")
        ax.legend(fontsize=7, frameon=False)
    return save(fig, name, outdir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--baselines", help="sum_scripted_baselines results directory")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    eps = episode_features(load_experiment(a.experiment))
    base = load_experiment(a.baselines).episodes if a.baselines else None
    plot_success_at_k(eps, outdir=Path(a.out))
    plot_turns_vs_x(eps, base, outdir=Path(a.out))


if __name__ == "__main__":
    main()
