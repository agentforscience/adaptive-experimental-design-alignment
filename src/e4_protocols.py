"""E4 (part 2) - replay static vs adaptive protocols against the live outcome grid.

Reads the cached (model x intervention x pressure-family x item) outcome matrix produced by
`e4_live.py` and replays protocols against it at MATCHED query budget, exactly as E1/E2 do.

Two outcomes are reported:

  detect_any        the probability that the protocol escapes the null at all, i.e. reports
                    at least one flip. This is the "null-to-non-null transition rate"
                    called for in the research brief, and it is the quantity a safety team
                    actually acts on.
  frac_items        the fraction of vulnerable items the protocol localises - the finer
                    measure of diagnostic power.

Because the grid is exhaustive we also know the CEILING (union over all pressure families):
the fraction of items that are flippable by *any* design in the space. A cell whose ceiling
is ~0 is genuine robustness; a cell with a high ceiling where the static protocol reports
zero is a false null. Separating these two cases is the point of the whole project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, SEED, boot_ci, holm, save_json, set_seed

N_REPS = 400
BUDGETS = [5, 10, 20, 30, 50, 75, 100, 150, 200]
FAMILY_ORDER = ["P1_simple_doubt", "P2_authority", "P3_fabricated_consensus",
                "P4_persistent_multi_turn"]


def load_grid() -> pd.DataFrame:
    parts = sorted((RESULTS / "live").glob("e4_grid*.csv"))
    parts = [p for p in parts if "_smoke" not in p.name]
    if not parts:
        raise SystemExit("no e4 grid files found - run src/e4_live.py first")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return df.drop_duplicates(subset=["model", "condition", "family", "item_id"])


def to_matrix(sub: pd.DataFrame):
    """(families, items, F[family,item] in {0,1}, cost[family])."""
    piv = sub.pivot_table(index="family", columns="item_id", values="flipped", aggfunc="max")
    fams = [f for f in FAMILY_ORDER if f in piv.index]
    piv = piv.loc[fams].dropna(axis=1, how="any")
    cost = sub.groupby("family")["cost"].first().reindex(fams).to_numpy()
    return fams, list(piv.columns), piv.to_numpy().astype(int), cost


# --- protocols: each spends at most `budget` model calls -------------------

def _spend(F, cost, budget, plan):
    """plan: iterator yielding (family_idx, item_idx). Returns set of items detected."""
    spent, found = 0, set()
    for m, b in plan:
        if spent + cost[m] > budget:
            break
        spent += cost[m]
        if F[m, b] == 1:
            found.add(b)
    return found


def p_static_family(F, cost, budget, rng, m):
    N = F.shape[1]
    return _spend(F, cost, budget, ((m, b) for b in rng.permutation(N)))


def p_static_random(F, cost, budget, rng):
    return p_static_family(F, cost, budget, rng, int(rng.integers(0, F.shape[0])))


def p_round_robin(F, cost, budget, rng):
    K, N = F.shape
    order = rng.permutation(N)
    plan = ((i // N % K, order[i % N]) for i in range(K * N))
    return _spend(F, cost, budget, plan)


def p_escalation(F, cost, budget, rng):
    """Null-triggered: an item that does not flip is escalated to the next family."""
    K, N = F.shape
    order = list(rng.permutation(N))
    spent, found, rung = 0, set(), {b: 0 for b in order}
    queue = list(order)
    while queue and spent < budget:
        nxt = []
        for b in queue:
            m = rung[b]
            if m >= K or spent + cost[m] > budget:
                continue
            spent += cost[m]
            if F[m, b] == 1:
                found.add(b)
                continue
            rung[b] = m + 1
            if rung[b] < K:
                nxt.append(b)
        if not nxt:
            break
        queue = nxt
    return found


def p_thompson(F, cost, budget, rng):
    """Adaptive over families; cost-aware (reward per unit of budget)."""
    K, N = F.shape
    a, b_ = np.ones(K), np.ones(K)
    untried = [list(rng.permutation(N)) for _ in range(K)]
    spent, found, solved = 0, set(), set()
    while spent < budget:
        theta = rng.beta(a, b_) / cost
        cand = [k for k in range(K) if untried[k] and spent + cost[k] <= budget]
        if not cand:
            break
        m = max(cand, key=lambda k: theta[k])
        item = None
        while untried[m]:
            c = untried[m].pop()
            if c not in solved:
                item = c
                break
        if item is None:
            continue
        spent += cost[m]
        if F[m, item] == 1:
            a[m] += 1; found.add(item); solved.add(item)
        else:
            b_[m] += 1
    return found


def p_oracle(F, cost, budget, rng):
    N = F.shape[1]
    flippable = [b for b in range(N) if F[:, b].max() == 1]
    rng.shuffle(flippable)
    per = int(cost.min())
    return set(flippable[: max(0, budget // per)])


def main():
    set_seed(SEED)
    grid = load_grid()
    grid.to_csv(RESULTS / "live" / "e4_grid_all.csv", index=False)
    recs, ceilings = [], []

    for (model, cond), sub in grid.groupby(["model", "condition"]):
        fams, items, F, cost = to_matrix(sub)
        if F.size == 0:
            continue
        ceiling = float((F.max(axis=0) == 1).mean())
        ceilings.append(dict(model=model, condition=cond, n_items=len(items),
                             ceiling=ceiling,
                             **{f: float(F[i].mean()) for i, f in enumerate(fams)}))
        protos = {f"STATIC-{f}": (lambda F, c, B, r, m=i: p_static_family(F, c, B, r, m))
                  for i, f in enumerate(fams)}
        protos.update({
            "STATIC-random-family": p_static_random,
            "STATIC-round-robin": p_round_robin,
            "ADAPTIVE-escalation": p_escalation,
            "ADAPTIVE-thompson": p_thompson,
            "ORACLE": p_oracle,
        })
        for budget in BUDGETS:
            for arm, fn in protos.items():
                for rep in range(N_REPS):
                    rng = np.random.default_rng(SEED + 7919 * rep)
                    found = fn(F, cost, budget, rng)
                    recs.append(dict(model=model, condition=cond, budget=budget, arm=arm,
                                     rep=rep, n_found=len(found),
                                     frac_items=len(found) / len(items),
                                     detect_any=int(len(found) > 0), ceiling=ceiling))

    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "live" / "e4_protocol_raw.csv.gz", index=False)
    ceil = pd.DataFrame(ceilings)
    ceil.to_csv(RESULTS / "live" / "e4_ceilings.csv", index=False)

    rows = []
    for (m, c, b, a), g in df.groupby(["model", "condition", "budget", "arm"]):
        mu, lo, hi = boot_ci(g["detect_any"].to_numpy(), rng=np.random.default_rng(SEED))
        fmu, flo, fhi = boot_ci(g["frac_items"].to_numpy(), rng=np.random.default_rng(SEED))
        rows.append(dict(model=m, condition=c, budget=b, arm=a,
                         detect_any=mu, da_lo=lo, da_hi=hi,
                         frac_items=fmu, fi_lo=flo, fi_hi=fhi,
                         ceiling=float(g["ceiling"].iloc[0])))
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "live" / "e4_protocol_summary.csv", index=False)

    # decisive tests: adaptive vs the equally-diversified static schedule, matched budget
    from scipy import stats
    tests = {}
    for (model, cond, budget), g in df.groupby(["model", "condition", "budget"]):
        piv = g.pivot_table(index="rep", columns="arm", values="frac_items")
        if "STATIC-round-robin" not in piv:
            continue
        for arm in [c for c in piv.columns if c.startswith("ADAPTIVE")]:
            d = piv[arm] - piv["STATIC-round-robin"]
            p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if d.abs().sum() > 0 else 1.0
            tests[f"{model}|{cond}|B={budget}|{arm}-vs-round-robin"] = dict(
                diff=float(d.mean()), lo=float(np.quantile(d, .025)),
                hi=float(np.quantile(d, .975)), p=p)
    for k, a in holm({k: v["p"] for k, v in tests.items()}).items():
        tests[k]["p_holm"] = a
    save_json(tests, RESULTS / "live" / "e4_tests.json")

    print("[E4] per-family flip rate and design-space ceiling:")
    print(ceil.round(3).to_string(index=False))
    print("\n[E4] null-to-non-null transition rate P(any detection) at matched budget,")
    print("     intervention condition only:")
    iv = summ[summ.condition == "intervention"]
    for model in sorted(iv.model.unique()):
        print(f"\n  --- {model} (ceiling={iv[iv.model == model].ceiling.iloc[0]:.2f}) ---")
        print(iv[iv.model == model].pivot_table(index="budget", columns="arm",
                                                values="detect_any").round(3).to_string())
    print("\n[E4] fraction of vulnerable items localised (intervention):")
    for model in sorted(iv.model.unique()):
        print(f"\n  --- {model} ---")
        print(iv[iv.model == model].pivot_table(index="budget", columns="arm",
                                                values="frac_items").round(3).to_string())
    return summ


if __name__ == "__main__":
    main()
