"""Per-episode strategy features, built on metrics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import metrics as M
from analysis.load import Run


def trajectories(run: Run):
    """Yield (condition_hash, episode_id, sums, results, actions) per episode."""
    t = run.turns.sort_values(["condition_hash", "episode_id", "turn"])
    for (h, e), g in t.groupby(["condition_hash", "episode_id"], sort=False):
        yield (h, e, list(g["sum"]), [f["result"] for f in g["feedback"]],
               [list(a) for a in g["actions"]])


def episode_features(run: Run, th: M.Thresholds = M.DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Episodes table plus consistency, bisection score, info gain, steps, label, etc."""
    eps = run.episodes.set_index(["condition_hash", "episode_id"], drop=False)
    rows = []
    for h, e, sums, results, actions in trajectories(run):
        n = int(eps.loc[(h, e), "n"])
        iv = M.intervals_before(n, sums, results)
        gains = M.info_gains(n, sums, results)
        members = eps.loc[(h, e), "clique_members"]
        row = {
            "condition_hash": h, "episode_id": e, "label": M.classify_episode(n, sums, results, th),
            "consistency": M.consistency_rate(sums, iv),
            "bisection_score": float(np.mean(M.bisection_scores(sums, iv))),
            "info_gain_frac": (sum(g for g, _ in gains) / sum(m for _, m in gains)
                               if gains and sum(m for _, m in gains) > 0 else np.nan),
            "mean_step": float(np.mean(M.step_profile(sums))) if len(sums) > 1 else np.nan,
            "oscillations": M.oscillations(sums), "overshoots": M.overshoots(results),
            "flip_rate": float(np.nanmean(M.flip_rates(actions))) if len(sums) > 1 else np.nan,
            "id_corr": M.id_correlation(actions),
        }
        dc = M.direction_flips(actions, results)
        row["dir_consistent_flips"] = dc.consistent_fraction
        oc = M.overcorrection(n, actions, results)
        row["mean_excess_flips"] = float(np.mean([o["excess"] for o in oc])) if oc else np.nan
        if len(members) > 1:
            gb = M.group_behavior(actions, results, members)
            row["clique_flip_rate"] = gb.get("clique", {}).get("flip_rate", np.nan)
            row["outsider_flip_rate"] = gb.get("outsiders", {}).get("flip_rate", np.nan)
            row["clique_dir_consistent"] = gb.get("clique", {}).get("direction_consistent", np.nan)
            row["outsider_dir_consistent"] = gb.get("outsiders", {}).get("direction_consistent", np.nan)
            row["compensation"] = M.compensation(actions, members)
        rows.append(row)
    feats = pd.DataFrame(rows)
    out = run.episodes.merge(feats, on=["condition_hash", "episode_id"], how="left")
    out["normalized_turns"] = out["turns"] / out["optimal_turns"]
    return out
