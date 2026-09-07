"""Shared utilities for the adaptive-experimental-design experiments.

Loading, seeding, bootstrap CIs and the small statistics helpers used by E1-E4.
Everything here is deterministic given SEED.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_BOOT = 1000

for _d in (RESULTS, FIGURES, RESULTS / "offline", RESULTS / "live"):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Trajectory loading
# --------------------------------------------------------------------------

def load_outcomes() -> pd.DataFrame:
    """Per-run outcomes for all 14 random-search configurations (672 rows)."""
    o = pd.read_csv(DATA / "rs_trajectories" / "_run_outcomes.csv")
    # Two judges are deliberately kept; they disagree and the disagreement is a result.
    o["success_judge"] = (o["judge_score_final"] == 10).astype(int)
    o["success_rule"] = o["jailbroken_rule_final"].astype(bool).astype(int)
    return o


def load_trajectory(config: str) -> pd.DataFrame:
    return pd.read_csv(DATA / "rs_trajectories" / f"{config}.csv.gz")


@dataclass
class RunTrace:
    """One (config, behaviour) attack run, reduced to what a replay needs.

    Attributes
    ----------
    goal        : harmful request string (may be unrecoverable -> None)
    n_iters     : iterations the original run actually consumed (T_r)
    success     : terminal outcome under the chosen judge
    best_prob   : monotone non-decreasing "best target-token probability so far",
                  indexed by iteration 1..n_iters. This is the continuous signal
                  that converts a null into feedback.
    """

    config: str
    run_id: int
    goal: str | None
    n_iters: int
    success: int
    success_rule: int
    best_prob: np.ndarray


def build_traces(config: str, outcomes: pd.DataFrame) -> dict[int, RunTrace]:
    """Reduce one configuration's per-iteration log to a dict of RunTrace."""
    traj = load_trajectory(config)
    # a few configs have duplicate recovered run_ids (restart-splitting, README caveat 1)
    oc = (outcomes[outcomes["config"] == config]
          .drop_duplicates(subset="run_id", keep="first").set_index("run_id"))
    traces: dict[int, RunTrace] = {}
    for run_id, g in traj.groupby("run_id"):
        if run_id not in oc.index:
            continue
        g = g.sort_values("iteration")
        bp = g["best_prob"].to_numpy(dtype=float)
        bp = np.nan_to_num(bp, nan=0.0, neginf=0.0)
        # enforce monotone "best so far" (log files occasionally reset on restart)
        bp = np.maximum.accumulate(bp)
        row = oc.loc[run_id]
        goal = row.get("goal")
        traces[int(run_id)] = RunTrace(
            config=config,
            run_id=int(run_id),
            goal=None if pd.isna(goal) else str(goal),
            n_iters=int(len(bp)),
            success=int(row["success_judge"]),
            success_rule=int(row["success_rule"]),
            best_prob=bp,
        )
    return traces


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def boot_ci(values: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05,
            rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean, resampling over units (behaviours)."""
    rng = rng or np.random.default_rng(SEED)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def paired_boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                     alpha: float = 0.05, rng: np.random.Generator | None = None):
    """Paired bootstrap of mean(a) - mean(b); a and b are aligned per behaviour."""
    rng = rng or np.random.default_rng(SEED)
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert len(a) == len(b)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    obs = float(a.mean() - b.mean())
    lo, hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
    # two-sided bootstrap p-value (proportion of resamples crossing zero, doubled)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, lo, hi, float(min(p, 1.0))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes. Returns (n01, n10, p)."""
    from scipy import stats
    a, b = np.asarray(a, int), np.asarray(b, int)
    n10 = int(((a == 1) & (b == 0)).sum())   # a wins
    n01 = int(((a == 0) & (b == 1)).sum())   # b wins
    n = n10 + n01
    if n == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(n10, n, 0.5).pvalue)
    return n01, n10, p


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference of two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U / (n_pos * n_neg))."""
    from scipy import stats
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  wrote {path.relative_to(ROOT)}")
