"""E2 - Compute-matched adaptive budget allocation over an attack x behaviour portfolio.

Substrate: `datasets/jbb_artifacts/jbb_queries_to_jailbreak.csv` - the official
JailbreakBench artifacts, a complete (method x target model x behaviour) outcome matrix.

Framing. A safety team has a budget of B *attack runs* against one target model. Each run
costs 1 unit and applies one method to one behaviour, returning a binary outcome. The team
wants to jailbreak as many distinct behaviours as possible. Static protocols fix the
allocation in advance; adaptive protocols re-allocate as nulls come in.

This is the same hypothesis as E1 on a completely different substrate and at a different
level of the design hierarchy (across methods rather than within a search), so agreement
between E1 and E2 is evidence of generality rather than of a substrate artifact.

The decisive contrast, as in E1, is ADAPTIVE vs a *non-adaptive but equally diversified*
comparator (ROUND-ROBIN): both spend B, both touch every method, and they differ only in
whether the observed nulls steer the remaining budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, SEED, boot_ci, holm, save_json, set_seed

N_REPS = 300          # policy-randomisation replicates per (model, budget, arm)
BUDGET_GRID = [25, 50, 100, 150, 200, 300, 400, 500]


def load_matrix() -> dict[str, tuple[list[str], list[int], np.ndarray]]:
    """Return {target_model: (methods, behaviour_ids, M[method, behaviour] in {0,1})}."""
    j = pd.read_csv(DATA / "jbb_artifacts" / "jbb_queries_to_jailbreak.csv")
    out = {}
    for model, g in j.groupby("target_model"):
        piv = g.pivot_table(index="method", columns="index", values="jailbroken",
                            aggfunc="max")
        piv = piv.dropna(axis=0, how="any")          # keep methods with full coverage
        out[model] = (list(piv.index), list(piv.columns), piv.to_numpy().astype(int))
    return out


# --------------------------------------------------------------------------
# Allocation policies. Each consumes exactly `budget` cell-evaluations and returns the
# set of behaviours found jailbroken. Cells are never re-evaluated (deterministic matrix).
# --------------------------------------------------------------------------

def _found(M, tried):
    """Distinct behaviours with at least one successful cell among `tried`."""
    hit = set()
    for (m, b) in tried:
        if M[m, b] == 1:
            hit.add(b)
    return hit


def policy_single_method(M, budget, rng, method_idx=None):
    """The field default: pick one method, run it on as many behaviours as budget allows."""
    K, N = M.shape
    m = rng.integers(0, K) if method_idx is None else method_idx
    order = rng.permutation(N)[:budget]
    return _found(M, [(m, b) for b in order])


def policy_round_robin(M, budget, rng):
    """Non-adaptive diversification: cycle methods over a fixed behaviour order."""
    K, N = M.shape
    order = rng.permutation(N)
    tried, i = [], 0
    while len(tried) < budget:
        b = order[i % N]
        m = (i // N) % K
        tried.append((m, b))
        i += 1
        if i >= K * N:
            break
    return _found(M, tried[:budget])


def policy_escalation(M, budget, rng):
    """Null-triggered per-behaviour escalation: on a null, move that behaviour to the
    next method; on a success, drop the behaviour and move on. Adaptive at the
    behaviour level, non-adaptive at the method level."""
    K, N = M.shape
    order = list(rng.permutation(N))
    rung = {b: 0 for b in order}
    tried, spent, queue = [], 0, list(order)
    while spent < budget and queue:
        nxt = []
        for b in queue:
            if spent >= budget:
                nxt.append(b)
                continue
            m = rung[b]
            if m >= K:
                continue
            tried.append((m, b))
            spent += 1
            if M[m, b] == 1:
                continue                     # solved, retire this behaviour
            rung[b] = m + 1
            if rung[b] < K:
                nxt.append(b)
        queue = nxt
    return _found(M, tried)


def policy_thompson(M, budget, rng, prior=(1.0, 1.0)):
    """Adaptive at the method level: Beta-Bernoulli Thompson sampling over methods,
    applied to a random untried behaviour of the sampled method."""
    K, N = M.shape
    a = np.full(K, prior[0]); b_ = np.full(K, prior[1])
    untried = [list(rng.permutation(N)) for _ in range(K)]
    tried = []
    for _ in range(budget):
        theta = rng.beta(a, b_)
        cand = [k for k in range(K) if untried[k]]
        if not cand:
            break
        m = max(cand, key=lambda k: theta[k])
        b = untried[m].pop()
        tried.append((m, b))
        if M[m, b] == 1:
            a[m] += 1
        else:
            b_[m] += 1
    return _found(M, tried)


def policy_successive_halving(M, budget, rng):
    """Adaptive at the method level: spend a pilot on every method, keep the best half,
    repeat. The canonical 'drop the arms that keep returning nulls' protocol."""
    K, N = M.shape
    alive = list(range(K))
    untried = {k: list(rng.permutation(N)) for k in range(K)}
    tried, spent = [], 0
    while spent < budget and alive:
        per = max(1, (budget - spent) // (len(alive) * max(1, int(np.ceil(np.log2(len(alive) + 1))))))
        scores = {}
        for m in list(alive):
            hits = 0
            for _ in range(per):
                if spent >= budget or not untried[m]:
                    break
                b = untried[m].pop()
                tried.append((m, b)); spent += 1
                hits += M[m, b]
            scores[m] = hits / max(1, per)
        if len(alive) == 1:
            # single survivor: spend everything left on it
            m = alive[0]
            while spent < budget and untried[m]:
                b = untried[m].pop(); tried.append((m, b)); spent += 1
            break
        alive = sorted(alive, key=lambda k: -scores[k])[:max(1, len(alive) // 2)]
    return _found(M, tried)


def policy_escalation_thompson(M, budget, rng):
    """Both levels adaptive: Thompson over methods, but a behaviour already solved is
    retired and a behaviour that has nulled on a method is never retried there."""
    K, N = M.shape
    a = np.full(K, 1.0); b_ = np.full(K, 1.0)
    untried = {k: list(rng.permutation(N)) for k in range(K)}
    solved, tried = set(), []
    for _ in range(budget):
        theta = rng.beta(a, b_)
        cand = [k for k in range(K) if untried[k]]
        if not cand:
            break
        m = max(cand, key=lambda k: theta[k])
        b = None
        while untried[m]:
            cb = untried[m].pop()
            if cb not in solved:
                b = cb
                break
        if b is None:
            continue
        tried.append((m, b))
        if M[m, b] == 1:
            a[m] += 1; solved.add(b)
        else:
            b_[m] += 1
    return _found(M, tried)


def policy_best_method(M, budget, rng):
    """Static allocation, but with the single best method chosen ORACULARLY in advance.
    This is the fair upper bound on any *a priori* method choice: whatever ADAPTIVE beats
    here is attributable to online method identification rather than to diversification."""
    K, N = M.shape
    m = int(np.argmax(M.mean(axis=1)))
    order = rng.permutation(N)[:budget]
    return _found(M, [(m, b) for b in order])


def policy_oracle(M, budget, rng):
    """Upper bound: spend each unit on the cheapest known-successful cell."""
    K, N = M.shape
    solvable = [b for b in range(N) if M[:, b].max() == 1]
    rng.shuffle(solvable)
    return set(solvable[:budget])


POLICIES = {
    "STATIC-single-method": policy_single_method,
    "STATIC-round-robin": policy_round_robin,
    "STATIC-best-method(oracle)": policy_best_method,
    "ADAPTIVE-escalation": policy_escalation,
    "ADAPTIVE-thompson": policy_thompson,
    "ADAPTIVE-succ-halving": policy_successive_halving,
    "ADAPTIVE-escal+thompson": policy_escalation_thompson,
    "ORACLE": policy_oracle,
}


def main():
    set_seed(SEED)
    mats = load_matrix()
    recs = []
    for model, (methods, beh, M) in mats.items():
        print(f"[E2] {model}: {len(methods)} methods x {len(beh)} behaviours; "
              f"per-method ASR = "
              f"{ {m: round(float(M[i].mean()), 2) for i, m in enumerate(methods)} }; "
              f"union-of-methods ceiling = {float((M.max(axis=0) == 1).mean()):.2f}")
        for budget in BUDGET_GRID:
            for arm, fn in POLICIES.items():
                for rep in range(N_REPS):
                    rng = np.random.default_rng(SEED + 9973 * rep)
                    found = fn(M, budget, rng)
                    recs.append(dict(model=model, budget=budget, arm=arm, rep=rep,
                                     n_found=len(found),
                                     frac_found=len(found) / len(beh)))
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "offline" / "e2_raw.csv.gz", index=False)

    summ = (df.groupby(["model", "budget", "arm"])["frac_found"]
              .agg(["mean", "std", "count"]).reset_index())
    ci = []
    for (m, b, a), g in df.groupby(["model", "budget", "arm"]):
        mu, lo, hi = boot_ci(g["frac_found"].to_numpy(), rng=np.random.default_rng(SEED))
        ci.append(dict(model=m, budget=b, arm=a, mean=mu, lo=lo, hi=hi))
    summ = pd.DataFrame(ci)
    summ.to_csv(RESULTS / "offline" / "e2_curves.csv", index=False)

    # The decisive test, mirroring E1: adaptive vs an equally-diversified static schedule.
    tests = {}
    for (model, budget), g in df.groupby(["model", "budget"]):
        piv = g.pivot_table(index="rep", columns="arm", values="frac_found")
        base = piv["STATIC-round-robin"]
        for arm in [c for c in piv.columns if c.startswith("ADAPTIVE")]:
            from scipy import stats
            d = piv[arm] - base
            t = stats.wilcoxon(d, zero_method="zsplit") if d.abs().sum() > 0 else None
            tests[f"{model}|B={budget}|{arm}-vs-round-robin"] = dict(
                diff=float(d.mean()),
                lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)),
                p=float(t.pvalue) if t is not None else 1.0)
    adj = holm({k: v["p"] for k, v in tests.items()})
    for k, a in adj.items():
        tests[k]["p_holm"] = a
    save_json(tests, RESULTS / "offline" / "e2_tests.json")

    print("\n[E2] fraction of behaviours jailbroken at matched budget:")
    for model in sorted(mats):
        print(f"\n  --- {model} ---")
        print(summ[summ.model == model].pivot_table(index="budget", columns="arm",
                                                    values="mean").round(3).to_string())
    return summ


if __name__ == "__main__":
    main()
