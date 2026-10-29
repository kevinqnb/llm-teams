"""Collaboration: performance vs clique size m, clique vs outsiders, and the controller baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.common import PALETTE, RESULTS, save
from analysis.features import episode_features
from analysis.load import load_experiment


def plot_turns_vs_m(eps: pd.DataFrame, controller: pd.DataFrame | None = None,
                    name: str = "turns_vs_clique_size", outdir: Path | None = None):
    ns = sorted(eps["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.4), squeeze=False)
    for ax, n in zip(axes[0], ns, strict=True):
        sub = eps[eps["n"] == n]
        g = sub.groupby("m")["turns"].agg(["mean", "sem"]).fillna(0)
        ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", color=PALETTE[0], label="agents")
        if controller is not None:
            c = controller[controller["n"] == n].groupby("m")["turns"].mean()
            ax.plot(c.index, c.values, "--", color="#666", label="scripted clique controller")
        ax.axhline(sub["oracle_turns"].mean(), color="#999", lw=1, label="oracle")
        ax.axvline(n / 2, color="#ccc", lw=1, ls=":")  # m = n - m
        ax.set(title=f"n={n}", xlabel="clique size m", ylabel="mean turns")
        ax.legend(fontsize=7, frameon=False)
    return save(fig, name, outdir)


def plot_clique_vs_outsiders(eps: pd.DataFrame, name: str = "clique_vs_outsiders",
                             outdir: Path | None = None):
    d = eps.dropna(subset=["clique_flip_rate"]) if "clique_flip_rate" in eps else eps.iloc[0:0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, (a, b, title) in zip(axes, [("clique_flip_rate", "outsider_flip_rate", "flip rate"),
                                        ("clique_dir_consistent", "outsider_dir_consistent",
                                         "direction-consistent flips")], strict=True):
        if not d.empty:
            g = d.groupby("m")[[a, b]].mean()
            ax.plot(g.index, g[a], "o-", color=PALETTE[0], label="clique")
            ax.plot(g.index, g[b], "o--", color=PALETTE[1], label="outsiders")
        ax.set(title=title, xlabel="clique size m")
        if not d.empty:
            ax.legend(fontsize=7, frameon=False)
    return save(fig, name, outdir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--controller", help="sum_clique_controller results directory")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    eps = episode_features(load_experiment(a.experiment))
    ctrl = load_experiment(a.controller).episodes if a.controller else None
    plot_turns_vs_m(eps, ctrl, outdir=Path(a.out))
    plot_clique_vs_outsiders(eps, outdir=Path(a.out))


if __name__ == "__main__":
    main()
