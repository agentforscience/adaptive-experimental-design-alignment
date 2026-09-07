"""E1 - Null-as-signal escalation ladder, replayed at MATCHED iteration budget.

Substrate: `datasets/rs_trajectories/` - the released per-iteration random-search logs
from Andriushchenko et al. (2024). Nothing is re-run; these are the authors' own traces.

The experimenter is given a budget of B iterations *per behaviour* and must decide how to
spend it across a set of available design families. Arms:

  STATIC-d0      all B on the default design (== "continue-only": maximal within-family
                 persistence, the compute-matched control that rules out "adaptive just
                 means more compute")
  STATIC-d1/d2   all B on a single alternative design (shows design choice matters)
  FIXED-SPLIT    B divided equally over d0,d1,d2 in order, with carry-over, on a schedule
                 that ignores the data (design *diversification* without adaptivity)
  LADDER         same design set, same total B, but the switch is triggered by the
                 continuous null signal (stall in best target-token probability)
  ORACLE         best design per behaviour, all B (upper bound)

The FIXED-SPLIT vs LADDER contrast is the sharpest test of the hypothesis: both switch
design family, both spend exactly B, and they differ *only* in whether the switch is
triggered by the null.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, SEED, RunTrace, auc, boot_ci, build_traces, cohens_h,
                    holm, load_outcomes, mcnemar_exact, paired_boot_diff, save_json,
                    set_seed)

# Design ladders available in the released logs, per target model.
# Rung 0 = default / plain design; rung 1 = re-initialisation (self-transfer);
# rung 2 = switch of prompt-template family (in-context one-shot).
LADDERS = {
    "llama-2-7b-chat": ["llama2-7b_plain_init", "llama2_7b", "llama2-7b_icl_one_shot"],
    "llama-2-13b-chat": ["llama2-13b", "llama2-13b_icl_one_shot"],
}
# Which rungs can be joined per-behaviour on the goal string (exact pairing) and which
# cannot (goal unrecoverable for in-context templates -- datasets/README.md caveat 2).
JOINABLE = {"llama2-7b_plain_init", "llama2_7b", "llama2-13b"}

BUDGETS = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 1000, 2500, 5000, 10000]


# --------------------------------------------------------------------------
# The null trigger
# --------------------------------------------------------------------------

def stall_iteration(best_prob: np.ndarray, patience: int, tau: float) -> int:
    """First iteration at which the search is declared *stalled* (a null with no momentum).

    `best_prob` is monotone non-decreasing. A stall is declared at the first iteration s
    (1-indexed, s > patience) where the best-so-far signal has not improved over the last
    `patience` iterations AND is still below the escalation threshold `tau`.

    Returns len(best_prob) + 1 if the run never stalls (i.e. never triggers escalation).
    """
    n = len(best_prob)
    if n <= patience:
        return n + 1
    for s in range(patience, n):           # 0-indexed position s -> iteration s+1
        if best_prob[s] < tau and best_prob[s] <= best_prob[s - patience] + 1e-12:
            return s + 1
    return n + 1


# --------------------------------------------------------------------------
# Protocol replay
# --------------------------------------------------------------------------

def run_design(tr: RunTrace, budget: int, stop_at: int | None = None) -> tuple[int, bool]:
    """Replay one design under a budget cap and an optional early-abandon iteration.

    Returns (iterations_consumed, detected).

    Semantics: the original run consumed `tr.n_iters` iterations and ended with outcome
    `tr.success`. A replayed protocol detects the vulnerability iff it lets the run reach
    its natural end, i.e. it neither runs out of budget nor abandons first.
    Conservative assumption: a run that the original authors ended without success is
    treated as a failure even if more budget were nominally available -- we have no data
    beyond `tr.n_iters`. This biases *against* the persistence arms only in the sense that
    it caps them; it is applied identically to every arm.
    """
    cap = budget if stop_at is None else min(budget, stop_at)
    if tr.n_iters <= cap:
        return tr.n_iters, bool(tr.success)
    return cap, False


def protocol_static(traces: list[RunTrace | None], budget: int, rung: int):
    tr = traces[rung]
    if tr is None:
        return 0, False, 0
    used, det = run_design(tr, budget)
    return used, det, 0 if det else -1


def escalate_at(bp: np.ndarray, cap: int, patience: int, tau: float, extend: float) -> int:
    """When the adaptive ladder abandons the current rung.

    The *schedule* is identical to the non-adaptive comparator (switch at `cap`); the
    signal is allowed to move that point in either direction, which is what "using the
    null as a trigger" concretely means:

      - stalled before the scheduled switch  -> switch EARLY (bank the unspent budget)
      - still a null at `cap` but the signal is warm (best_prob >= tau) -> PERSIST up to
        `extend * cap`, on the theory that this design is about to pay off
      - still a null at `cap` and the signal is cold -> switch on schedule

    Setting the signal aside (tau = 0, extend = 1) recovers FIXED-SPLIT exactly, so the
    two arms differ *only* in the trigger.
    """
    s = stall_iteration(bp, patience, tau)
    if s <= cap:
        return s
    idx = min(cap, len(bp)) - 1
    if idx >= 0 and bp[idx] >= tau:
        return int(min(len(bp), max(cap, round(cap * extend))))
    return cap


def _sequential(traces, budget, cap_fn):
    """Run rungs in order; `cap_fn(i, tr, share, remaining)` gives the abandon point."""
    avail = [t for t in traces if t is not None]
    if not avail:
        return 0, False, -1
    share = max(1, budget // len(avail))
    spent = 0
    for i, tr in enumerate(avail):
        remaining = budget - spent
        if remaining <= 0:
            break
        last = (i == len(avail) - 1)
        stop_at = None if last else cap_fn(i, tr, share, remaining)
        used, det = run_design(tr, remaining, stop_at=stop_at)
        spent += used
        if det:
            return spent, True, i
    return spent, False, -1


def protocol_fixed_split(traces, budget: int):
    """Non-adaptive diversification: switch on a fixed schedule, ignoring the data."""
    return _sequential(traces, budget, lambda i, tr, share, rem: min(share, rem))


def protocol_ladder(traces, budget: int, patience: int, tau: float, extend: float = 3.0):
    """Signal-triggered escalation: continue -> re-initialise -> switch family."""
    return _sequential(
        traces, budget,
        lambda i, tr, share, rem: min(rem, escalate_at(tr.best_prob, min(share, rem),
                                                       patience, tau, extend)))


def protocol_oracle(traces: list[RunTrace | None], budget: int):
    best = None
    for i, tr in enumerate(traces):
        if tr is None:
            continue
        used, det = run_design(tr, budget)
        if det and (best is None or used < best[0]):
            best = (used, True, i)
    if best:
        return best
    return budget, False, -1


# --------------------------------------------------------------------------
# Behaviour-level assembly
# --------------------------------------------------------------------------

def assemble_behaviours(ladder: list[str], outcomes: pd.DataFrame, rng: np.random.Generator,
                        joined_only: bool = False):
    """Build per-behaviour tuples of RunTrace, one per rung.

    Rungs whose goal strings are recoverable are joined exactly on the (normalised) goal.
    Rungs whose goals are unrecoverable (in-context templates) are matched by *random
    assignment* -- an explicit exchangeability assumption, flagged in the output.
    """
    per_rung = {c: build_traces(c, outcomes) for c in ladder}
    exact = [c for c in ladder if c in JOINABLE]
    approx = [c for c in ladder if c not in JOINABLE]
    if joined_only:
        approx = []

    # index the exactly-joinable rungs by normalised goal
    def norm(g):
        return None if g is None else " ".join(str(g).lower().split())

    goal_index = {}
    for c in exact:
        goal_index[c] = {norm(t.goal): t for t in per_rung[c].values() if t.goal}

    anchor = exact[0]
    keys = list(goal_index[anchor].keys())
    rows = []
    for k in keys:
        row = []
        ok = True
        for c in ladder:
            if joined_only and c not in exact:
                continue
            if c in exact:
                t = goal_index[c].get(k)
                if t is None:
                    ok = False
                row.append(t)
            else:
                pool = list(per_rung[c].values())
                row.append(pool[rng.integers(0, len(pool))] if pool else None)
        if ok:
            rows.append((k, row))
    return rows, ("exact" if not approx else "exact+resampled")


def sweep(rows, patience: int, tau: float, extend: float = 3.0):
    """Detection outcome for every arm x budget x behaviour."""
    recs = []
    n_rungs = len(rows[0][1])
    for budget in BUDGETS:
        for key, traces in rows:
            arms = {}
            for r in range(n_rungs):
                arms[f"STATIC-d{r}"] = protocol_static(traces, budget, r)
            arms["FIXED-SPLIT"] = protocol_fixed_split(traces, budget)
            arms["LADDER"] = protocol_ladder(traces, budget, patience, tau, extend)
            arms["ORACLE"] = protocol_oracle(traces, budget)
            for arm, (u, d, rung) in arms.items():
                recs.append(dict(budget=budget, behaviour=key, arm=arm,
                                 iters=u, detected=int(d), rung=rung))
    return pd.DataFrame(recs)


def trigger_sensitivity(rows, budgets=(12, 60, 200, 1000)):
    """Is the ladder's behaviour a knife-edge in its trigger hyper-parameters?"""
    recs = []
    for patience, tau, extend in itertools.product((2, 5, 20), (0.05, 0.3, 0.7), (1.0, 3.0, 10.0)):
        for budget in budgets:
            det = [protocol_ladder(tr, budget, patience, tau, extend)[1] for _, tr in rows]
            fs = [protocol_fixed_split(tr, budget)[1] for _, tr in rows]
            recs.append(dict(patience=patience, tau=tau, extend=extend, budget=budget,
                             ladder=float(np.mean(det)), fixed_split=float(np.mean(fs)),
                             delta=float(np.mean(det) - np.mean(fs))))
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# E1c - is a null actionable? diagnosticity of the stall signal
# --------------------------------------------------------------------------

def null_diagnosticity(outcomes: pd.DataFrame, checkpoints=(1, 2, 3, 5, 10, 25, 50, 100, 250)):
    """At iteration t, does the continuous signal predict the eventual outcome?

    This tests the mechanism the entire hypothesis rests on: an adaptive protocol can only
    use a null as a trigger if the null carries information. AUC ~ 0.5 refutes it.

    Crucially the population is conditioned on *still being a null at t* -- that is exactly
    the decision context an adaptive protocol faces, and it is a much harder population
    than "all runs", because the easy successes have already left.
    """
    recs, pooled = [], []
    configs = sorted(outcomes["config"].unique())
    for c in configs:
        traces = build_traces(c, outcomes)
        if not traces:
            continue
        for t in checkpoints:
            live = [tr for tr in traces.values() if tr.n_iters >= t]
            if len(live) < 8:
                continue
            sig = np.array([tr.best_prob[t - 1] for tr in live])
            lab = np.array([tr.success for tr in live])
            a = float("nan") if lab.sum() in (0, len(lab)) else auc(sig, lab)
            recs.append(dict(config=c, t=t, n_live=len(live), base_rate=float(lab.mean()),
                             auc=a, mean_signal=float(sig.mean())))
            if not np.isnan(a):
                # within-config rank-normalise so configs can be pooled without a
                # between-config difficulty offset driving the pooled AUC
                order = sig.argsort().argsort() / max(1, len(sig) - 1)
                for r, l in zip(order, lab):
                    pooled.append(dict(t=t, rank=float(r), label=int(l), config=c))
    per_config = pd.DataFrame(recs)

    # pooled AUC per checkpoint, with a bootstrap CI clustered on config
    pool_df = pd.DataFrame(pooled)
    prows = []
    rng = np.random.default_rng(SEED)
    for t, g in pool_df.groupby("t"):
        a = auc(g["rank"].to_numpy(), g["label"].to_numpy())
        cfgs = g["config"].unique()
        boots = []
        for _ in range(500):
            pick = rng.choice(cfgs, size=len(cfgs), replace=True)
            sub = pd.concat([g[g.config == c] for c in pick])
            if sub["label"].nunique() < 2:
                continue
            boots.append(auc(sub["rank"].to_numpy(), sub["label"].to_numpy()))
        lo, hi = (np.nanquantile(boots, 0.025), np.nanquantile(boots, 0.975)) if boots else (np.nan, np.nan)
        prows.append(dict(t=t, n=len(g), n_configs=len(cfgs), pooled_auc=a,
                          lo=float(lo), hi=float(hi), base_rate=float(g["label"].mean())))
    return per_config, pd.DataFrame(prows)


def main():
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    outcomes = load_outcomes()
    all_sweeps, summaries, tests = [], [], {}

    for target, ladder in LADDERS.items():
        for joined_only in ([True, False] if len(ladder) > 2 else [False]):
            rows, join_kind = assemble_behaviours(ladder, outcomes, rng, joined_only=joined_only)
            tag = f"{target}{'_joined-only' if joined_only else ''}"
            print(f"[E1] {tag}: {len(rows)} behaviours, join={join_kind}, "
                  f"rungs={len(rows[0][1])}")
            df = sweep(rows, patience=5, tau=0.5)
            df["target"] = target
            df["setting"] = tag
            df["join"] = join_kind
            all_sweeps.append(df)

            # detection-vs-budget curve with bootstrap CIs over behaviours
            for (budget, arm), g in df.groupby(["budget", "arm"]):
                m, lo, hi = boot_ci(g["detected"].to_numpy(), rng=np.random.default_rng(SEED))
                summaries.append(dict(setting=tag, target=target, budget=budget, arm=arm,
                                      n=len(g), detect=m, lo=lo, hi=hi,
                                      mean_iters=float(g["iters"].mean())))

            # paired tests, LADDER vs each comparator, at every budget
            for budget in BUDGETS:
                sub = df[df["budget"] == budget]
                piv = sub.pivot_table(index="behaviour", columns="arm", values="detected")
                if "LADDER" not in piv:
                    continue
                for comp in [c for c in piv.columns if c != "LADDER"]:
                    obs, lo, hi, p = paired_boot_diff(piv["LADDER"].to_numpy(),
                                                      piv[comp].to_numpy(),
                                                      rng=np.random.default_rng(SEED))
                    n01, n10, pmc = mcnemar_exact(piv["LADDER"].to_numpy().astype(int),
                                                  piv[comp].to_numpy().astype(int))
                    tests[f"{tag}|B={budget}|LADDER-vs-{comp}"] = dict(
                        diff=obs, lo=lo, hi=hi, p_boot=p, mcnemar_p=pmc,
                        n_ladder_only=n10, n_comp_only=n01,
                        h=cohens_h(float(piv['LADDER'].mean()), float(piv[comp].mean())))

    sweeps = pd.concat(all_sweeps, ignore_index=True)
    sweeps.to_csv(RESULTS / "offline" / "e1_sweep_raw.csv.gz", index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(RESULTS / "offline" / "e1_detection_curves.csv", index=False)

    # Holm correction within the primary family: LADDER vs FIXED-SPLIT and vs STATIC-d0
    primary = {k: v["mcnemar_p"] for k, v in tests.items()
               if ("LADDER-vs-FIXED-SPLIT" in k or "LADDER-vs-STATIC-d0" in k)
               and "joined-only" not in k}
    adj = holm(primary)
    for k, a in adj.items():
        tests[k]["mcnemar_p_holm"] = a
    save_json(tests, RESULTS / "offline" / "e1_tests.json")

    # trigger sensitivity on the full 3-rung ladder
    rows3, _ = assemble_behaviours(LADDERS["llama-2-7b-chat"], outcomes,
                                   np.random.default_rng(SEED), joined_only=False)
    sens = trigger_sensitivity(rows3)
    sens.to_csv(RESULTS / "offline" / "e1_trigger_sensitivity.csv", index=False)
    print("\n[E1] trigger sensitivity (LADDER - FIXED-SPLIT, detection rate):")
    print(sens.pivot_table(index=["patience", "tau"], columns="budget",
                           values="delta").round(3).to_string())

    # E1c
    diag, pooled = null_diagnosticity(outcomes)
    diag.to_csv(RESULTS / "offline" / "e1c_null_diagnosticity.csv", index=False)
    pooled.to_csv(RESULTS / "offline" / "e1c_pooled_auc.csv", index=False)
    print("\n[E1c] pooled AUC of the signal among runs STILL NULL at iteration t:")
    print(pooled.round(3).to_string(index=False))

    print("\n[E1] detection at matched budget (full ladder, llama-2-7b-chat):")
    piv = summ[summ.setting == "llama-2-7b-chat"].pivot_table(
        index="budget", columns="arm", values="detect")
    print(piv.round(3).to_string())
    print("\n[E1] rung at which LADDER detects (share of detections):")
    lad = sweeps[(sweeps.arm == "LADDER") & (sweeps.detected == 1)
                 & (sweeps.setting == "llama-2-7b-chat")]
    print(lad.groupby(["budget", "rung"]).size().unstack(fill_value=0).to_string())
    return summ, diag


if __name__ == "__main__":
    main()
