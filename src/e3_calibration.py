"""E3 - Are the NULLS an adaptive protocol emits calibrated?

The hypothesis has two halves. E1/E2 test only the first (does adaptivity find more?). This
experiment tests the second: when a protocol stops and reports "we found nothing", is the
bound it attaches to that null correct?

Set-up. An evaluator runs n behaviours against a target, monitoring a grid of budgets. It
stops (adaptively - that is the whole point) when the discovery curve looks flat, observes
r successes out of n, and must publish an upper bound on the fraction of behaviours that
are *actually* elicitable, where ground truth is what the full trace eventually shows.

Four bounds are compared:

  FIXED-t CP        Clopper-Pearson upper bound at a budget fixed in advance. Valid as a
                    bound on the rate *at that budget* - the textbook-correct procedure.
  ADAPTIVE CP       The same interval, but computed at an adaptively chosen stopping time.
                    This is what the field actually does, and it is doubly wrong: optional
                    stopping inflates error, and it bounds the wrong estimand.
  UNION-BOUND CP    Bonferroni over the monitoring grid: a simple, genuinely anytime-valid
                    correction for the optional-stopping half of the problem.
  EXTRAPOLATIVE     Fits the Best-of-N power law -log(ASR(t)) = a t^-b (Hughes et al. 2024)
                    to the observed discovery curve and extrapolates to the full horizon,
                    i.e. it models the *arrival process* instead of only the counts.

The interesting question is whether the statistical fix (anytime-validity) is sufficient,
or whether the binding problem is extrapolation rather than optional stopping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, SEED, build_traces, load_outcomes, save_json, set_seed

ALPHA = 0.05
GRID = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 1000, 2500, 5000, 10000]
PATIENCE_CHECKS = 3        # "looks converged": no new discovery for 3 consecutive checks
N_BOOT = 400


def cp_upper(r: int, n: int, alpha: float) -> float:
    """Clopper-Pearson one-sided upper confidence limit for a binomial proportion."""
    if n == 0:
        return 1.0
    if r >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, r + 1, n - r))


def discovery_curve(n_iters: np.ndarray, success: np.ndarray, grid) -> np.ndarray:
    """Fraction of behaviours discovered by each budget in `grid`."""
    return np.array([float(np.mean(success & (n_iters <= t))) for t in grid])


def adaptive_stop(curve: np.ndarray, patience: int = PATIENCE_CHECKS) -> int:
    """Index of the grid point where a data-driven evaluator would declare convergence."""
    for i in range(patience, len(curve)):
        if curve[i] <= curve[i - patience] + 1e-12:
            return i
    return len(curve) - 1


def extrapolate_powerlaw(curve: np.ndarray, grid, stop_idx: int, horizon: float) -> float:
    """Fit -log(ASR(t)) = a * t^-b on the observed prefix and extrapolate to `horizon`.

    Returns an upper estimate of the eventual rate, or NaN when the observed prefix is a
    *total* null (ASR = 0 everywhere), in which case the power law is unidentifiable -
    which is itself the point: extrapolation cannot rescue a design that has seen nothing.
    """
    ts = np.array(grid[: stop_idx + 1], dtype=float)
    ys = curve[: stop_idx + 1]
    ok = (ys > 0) & (ys < 1)
    if ok.sum() < 3:
        return float("nan")
    x = np.log(ts[ok])
    y = np.log(-np.log(ys[ok]))
    b, loga = np.polyfit(x, y, 1)          # y = log(a) - b_eff * x  (slope = -b)
    pred = np.exp(-np.exp(loga + b * np.log(horizon)))
    return float(min(1.0, max(0.0, pred)))


def main():
    set_seed(SEED)
    outcomes = load_outcomes()
    rng = np.random.default_rng(SEED)
    recs = []

    for config in sorted(outcomes["config"].unique()):
        traces = build_traces(config, outcomes)
        if len(traces) < 15:
            continue
        n_iters = np.array([t.n_iters for t in traces.values()])
        succ = np.array([t.success for t in traces.values()]).astype(bool)
        horizon = float(max(GRID))
        n = len(succ)

        for b in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            ni, sc = n_iters[idx], succ[idx]
            truth = float(np.mean(sc))                      # eventual elicitable fraction
            curve = discovery_curve(ni, sc, GRID)
            si = adaptive_stop(curve)
            r_stop = int(np.sum(sc & (ni <= GRID[si])))

            # a budget fixed in advance, deliberately modest - the common practice
            fixed_i = GRID.index(100)
            r_fixed = int(np.sum(sc & (ni <= GRID[fixed_i])))

            bounds = {
                "FIXED-t CP": cp_upper(r_fixed, n, ALPHA),
                "ADAPTIVE CP": cp_upper(r_stop, n, ALPHA),
                "UNION-BOUND CP": cp_upper(r_stop, n, ALPHA / len(GRID)),
                "EXTRAPOLATIVE": extrapolate_powerlaw(curve, GRID, si, horizon),
            }
            for name, ub in bounds.items():
                recs.append(dict(config=config, boot=b, method=name, bound=ub,
                                 truth=truth, stop_budget=GRID[si],
                                 observed_at_stop=curve[si],
                                 total_null=int(curve[si] == 0),
                                 covered=int((not np.isnan(ub)) and ub >= truth - 1e-12),
                                 defined=int(not np.isnan(ub))))

    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "offline" / "e3_raw.csv.gz", index=False)

    # coverage: over all bootstrap replicates, pooled and per config
    def agg(g):
        d = g[g.defined == 1]
        return pd.Series({
            "n": len(g),
            "frac_defined": float(g["defined"].mean()),
            "coverage": float(d["covered"].mean()) if len(d) else np.nan,
            "mean_bound": float(d["bound"].mean()) if len(d) else np.nan,
            "mean_truth": float(g["truth"].mean()),
            "mean_excess": float((d["bound"] - d["truth"]).mean()) if len(d) else np.nan,
        })

    pooled = df.groupby("method", group_keys=False)[df.columns].apply(agg).round(3)
    per_config = df.groupby(["config", "method"], group_keys=False)[df.columns].apply(agg).round(3)
    pooled.to_csv(RESULTS / "offline" / "e3_coverage_pooled.csv")
    per_config.to_csv(RESULTS / "offline" / "e3_coverage_per_config.csv")

    # split by whether the evaluator stopped on a TOTAL null (the case that matters most)
    tn = df[df.total_null == 1].groupby("method", group_keys=False)[df.columns].apply(agg).round(3)
    tn.to_csv(RESULTS / "offline" / "e3_coverage_total_nulls.csv")

    save_json(dict(alpha=ALPHA, nominal_coverage=1 - ALPHA, grid=GRID,
                   n_boot=N_BOOT, patience_checks=PATIENCE_CHECKS,
                   pooled=pooled.to_dict(), total_null_only=tn.to_dict()),
              RESULTS / "offline" / "e3_summary.json")

    print("[E3] pooled coverage of the published upper bound "
          f"(nominal {1 - ALPHA:.0%}):")
    print(pooled.to_string())
    print("\n[E3] restricted to evaluators that stopped on a TOTAL null "
          f"(n={int((df.total_null == 1).sum() / df.method.nunique())} replicates):")
    print(tn.to_string())
    print("\n[E3] per-config coverage (ADAPTIVE CP):")
    print(per_config.xs("ADAPTIVE CP", level="method")[["coverage", "mean_bound",
                                                        "mean_truth"]].to_string())
    return pooled


if __name__ == "__main__":
    main()
