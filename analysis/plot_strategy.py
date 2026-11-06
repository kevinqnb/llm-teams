"""Strategy-label distribution across the sweep, and trajectories overlaid on oracle bisection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import metrics as M
from analysis.common import PALETTE, RESULTS, save, x_column
from analysis.features import episode_features, trajectories
from analysis.load import Run, load_experiment
from llm_teams.tasks.sum_to_target import bisect_guess, update_interval

LABEL_COLORS = dict(zip(M.LABELS, [PALETTE[2], PALETTE[0], PALETTE[1], "#999999", PALETTE[4]],
                        strict=True))


def label_table(eps: pd.DataFrame) -> pd.DataFrame:
    xcol = x_column(eps)
    t = eps.groupby([xcol, "label"]).size().unstack(fill_value=0).reindex(columns=M.LABELS, fill_value=0)
    return t.div(t.sum(axis=1), axis=0)


def plot_labels(eps: pd.DataFrame, name: str = "strategy_labels", outdir: Path | None = None):
    t = label_table(eps)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bottom = np.zeros(len(t))
    for lab in M.LABELS:
        ax.bar([str(i) for i in t.index], t[lab], bottom=bottom, color=LABEL_COLORS[lab], label=lab)
        bottom += t[lab].to_numpy()
    ax.set(xlabel=t.index.name, ylabel="fraction of episodes")
    ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1))
    return save(fig, name, outdir)


def oracle_trajectory(n: int, target: int, max_turns: int = 30) -> list[int]:
    lo, hi, out = 0, n, []
    for _ in range(max_turns):
        g = bisect_guess(lo, hi)
        out.append(g)
        if g == target:
            break
        lo, hi = update_interval(lo, hi, g, "too_high" if g > target else "too_low")
    return out


def plot_trajectories(run: Run, eps: pd.DataFrame, name: str = "trajectories", outdir: Path | None = None,
                      p_message: float | None = None, ncols: int = 5):
    """One panel per game, in game order: team sum per turn vs the oracle bisection and the target.

    With p_message set, only games of that message-graph probability are drawn.
    """
    if p_message is not None:
        eps = eps[eps["p_message"] == p_message]
    eps = eps.set_index(["condition_hash", "episode_id"])
    sums_by_game = {(h, e): sums for h, e, sums, _, _ in trajectories(run) if (h, e) in eps.index}
    games = sorted(sums_by_game, key=lambda k: (k[1], k[0]))
    ncols = min(ncols, len(games))
    nrows = -(-len(games) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False,
                             layout="constrained")
    for ax in axes.flat[len(games):]:
        ax.set_visible(False)
    for ax, (h, e) in zip(axes.flat, games, strict=False):
        row = eps.loc[(h, e)]
        sums = sums_by_game[(h, e)]
        ax.plot(range(1, len(sums) + 1), sums, "o-", color=PALETTE[0], label="agents")
        orc = oracle_trajectory(int(row["n"]), int(row["target"]))
        ax.plot(range(1, len(orc) + 1), orc, "s--", color="#666", label="oracle")
        ax.axhline(row["target"], color=PALETTE[1], lw=1, label="target")
        ax.set(title=f"Game {e + 1}", xlabel="turn", ylabel="sum")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0][0].legend(fontsize=7, frameon=False)
    return save(fig, name, outdir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    run = load_experiment(a.experiment)
    eps = episode_features(run)
    plot_labels(eps, outdir=Path(a.out))
    for p in sorted(eps["p_message"].dropna().unique()):
        plot_trajectories(run, eps, name=f"trajectories_p{p:g}", outdir=Path(a.out), p_message=p)


if __name__ == "__main__":
    main()
