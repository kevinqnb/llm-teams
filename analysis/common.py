"""Shared plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
BASELINE_STYLE = {"oracle": ("#555555", "-"), "id_bisection": ("#555555", "--"),
                  "random": ("#999999", ":")}
RESULTS = Path(__file__).parent / "results"


def save(fig, name: str, outdir: Path | None = None) -> list[Path]:
    outdir = Path(outdir or RESULTS)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def x_column(df: pd.DataFrame) -> str:
    """Which sweep parameter varies: message-graph p, action-graph p, agent type, else n."""
    for c in ("p_message", "p_action", "agents_type"):
        if c in df and df[c].nunique() > 1:
            return c
    return "n"


def baseline_means(baselines: pd.DataFrame | None) -> dict[tuple[str, int], tuple[float, float]]:
    """(agents_type, n) -> (mean turns over all episodes, success rate) from scripted runs."""
    if baselines is None or baselines.empty:
        return {}
    g = baselines.groupby(["agents_type", "n"]).agg(t=("turns", "mean"), s=("success", "mean"))
    return {k: (v.t, v.s) for k, v in g.iterrows()}
