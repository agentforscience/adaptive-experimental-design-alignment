"""Figures and the consolidated results table for E1-E4."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, RESULTS

# A single brand-neutral categorical palette, used consistently across every figure.
C = {"STATIC": "#8c8c8c", "STATIC2": "#c0c0c0", "DIVERSE": "#3b7dd8",
     "ADAPTIVE": "#d1495b", "ADAPTIVE2": "#f0a202", "ORACLE": "#2a9d8f"}


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(alpha=.25, linewidth=.6); ax.spines[["top", "right"]].set_visible(False)


def fig_e1():
    f = RESULTS / "offline" / "e1_detection_curves.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.setting == "llama-2-7b-chat"]
    colors = {"STATIC-d0": C["STATIC"], "STATIC-d1": C["STATIC2"], "STATIC-d2": "#5a5a5a",
              "FIXED-SPLIT": C["DIVERSE"], "LADDER": C["ADAPTIVE"], "ORACLE": C["ORACLE"]}
    labels = {"STATIC-d0": "STATIC d0 (default design, = continue-only)",
              "STATIC-d1": "STATIC d1 (re-initialised)",
              "STATIC-d2": "STATIC d2 (other template family)",
              "FIXED-SPLIT": "FIXED-SPLIT (diversified, non-adaptive)",
              "LADDER": "LADDER (null-triggered escalation)",
              "ORACLE": "ORACLE (best design per behaviour)"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, g in d.groupby("arm"):
        g = g.sort_values("budget")
        ax.plot(g.budget, g.detect, marker="o", ms=3.5, lw=1.8,
                color=colors.get(arm, "k"), label=labels.get(arm, arm),
                ls="--" if arm == "ORACLE" else "-")
        ax.fill_between(g.budget, g.lo, g.hi, color=colors.get(arm, "k"), alpha=.10, lw=0)
    ax.set_xscale("log")
    _style(ax, "matched iteration budget per behaviour (log scale)",
           "detection rate", "E1  Llama-2-7B-chat: detection at matched compute\n"
           "(n=50 behaviours, shaded = 95% bootstrap CI)")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=.9)
    fig.tight_layout(); fig.savefig(FIGURES / "e1_detection_vs_budget.png", dpi=180)
    plt.close(fig)


def fig_e1c():
    f = RESULTS / "offline" / "e1c_pooled_auc.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).sort_values("t")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.axhline(.5, color="#555", ls=":", lw=1.2, label="uninformative (AUC = 0.5)")
    ax.errorbar(d.t, d.pooled_auc,
                yerr=[d.pooled_auc - d.lo, d.hi - d.pooled_auc],
                marker="o", ms=5, lw=1.8, capsize=3, color=C["ADAPTIVE"],
                label="pooled AUC (95% CI, clustered on config)")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    _style(ax, "iteration t at which the run is still a null",
           "AUC for predicting eventual success",
           "E1c  Is a null actionable?\nDiagnosticity of the continuous signal, "
           "conditional on still being null at t")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "e1c_null_diagnosticity.png", dpi=180)
    plt.close(fig)


def fig_e2():
    f = RESULTS / "offline" / "e2_curves.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    models = sorted(d.model.unique())
    colors = {"STATIC-single-method": C["STATIC"],
              "STATIC-best-method(oracle)": C["STATIC2"],
              "STATIC-round-robin": C["DIVERSE"],
              "ADAPTIVE-thompson": C["ADAPTIVE"],
              "ADAPTIVE-escal+thompson": C["ADAPTIVE2"],
              "ADAPTIVE-succ-halving": "#7b2cbf",
              "ADAPTIVE-escalation": "#9fbfe0",
              "ORACLE": C["ORACLE"]}
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.8), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        g0 = d[d.model == m]
        for arm, g in g0.groupby("arm"):
            g = g.sort_values("budget")
            ax.plot(g.budget, g["mean"], marker="o", ms=3, lw=1.6,
                    color=colors.get(arm, "k"), label=arm,
                    ls="--" if arm == "ORACLE" else "-")
        _style(ax, "matched budget (attack runs)",
               "fraction of behaviours jailbroken" if m == models[0] else "", m)
    np.atleast_1d(axes)[-1].legend(fontsize=6.5, loc="lower right", framealpha=.9)
    fig.suptitle("E2  Compute-matched budget allocation over the attack x behaviour "
                 "portfolio (JailbreakBench, 100 behaviours/model)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "e2_portfolio_allocation.png", dpi=180)
    plt.close(fig)


def fig_e3():
    f = RESULTS / "offline" / "e3_coverage_pooled.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).set_index("method")
    order = ["FIXED-t CP", "ADAPTIVE CP", "UNION-BOUND CP", "EXTRAPOLATIVE"]
    d = d.reindex([o for o in order if o in d.index])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    cols = [C["STATIC"], C["ADAPTIVE"], C["DIVERSE"], C["ORACLE"]]
    ax.bar(range(len(d)), d.coverage, color=cols[:len(d)], width=.6)
    ax.axhline(.95, color="#333", ls="--", lw=1.4, label="nominal 95% coverage")
    for i, (v, fd) in enumerate(zip(d.coverage, d.frac_defined)):
        ax.text(i, v + .015, f"{v:.2f}" + ("" if fd > .99 else f"\n({fd:.0%} defined)"),
                ha="center", fontsize=8)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.index, fontsize=8, rotation=12)
    ax.set_ylim(0, 1.05)
    _style(ax, "", "empirical coverage of the published upper bound",
           "E3  Nulls are badly uncalibrated, and anytime-validity barely helps\n"
           "(14 configs x 400 bootstrap replicates)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "e3_null_calibration.png", dpi=180)
    plt.close(fig)


def fig_e4():
    f = RESULTS / "live" / "e4_protocol_summary.csv"
    if not f.exists():
        print("  (E4 summary missing - skipping E4 figures)")
        return
    d = pd.read_csv(f)
    iv = d[d.condition == "intervention"]
    models = sorted(iv.model.unique())
    keep = ["STATIC-P1_simple_doubt", "STATIC-random-family", "STATIC-round-robin",
            "ADAPTIVE-escalation", "ADAPTIVE-thompson", "ORACLE"]
    colors = {"STATIC-P1_simple_doubt": C["STATIC"], "STATIC-random-family": C["STATIC2"],
              "STATIC-round-robin": C["DIVERSE"], "ADAPTIVE-escalation": C["ADAPTIVE2"],
              "ADAPTIVE-thompson": C["ADAPTIVE"], "ORACLE": C["ORACLE"]}
    for metric, fname, ylab in [
            ("detect_any", "e4_null_to_nonnull.png",
             "P(protocol reports >=1 flip)"),
            ("frac_items", "e4_items_localised.png",
             "fraction of items localised")]:
        fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.8),
                                 sharey=True)
        for ax, m in zip(np.atleast_1d(axes), models):
            g0 = iv[(iv.model == m) & (iv.arm.isin(keep))]
            ceiling = g0.ceiling.iloc[0] if len(g0) else np.nan
            for arm, g in g0.groupby("arm"):
                g = g.sort_values("budget")
                ax.plot(g.budget, g[metric], marker="o", ms=3, lw=1.6,
                        color=colors.get(arm, "k"), label=arm,
                        ls="--" if arm == "ORACLE" else "-")
            _style(ax, "matched budget (model calls)",
                   ylab if m == models[0] else "",
                   f"{m.split('/')[-1]}\ndesign-space ceiling = {ceiling:.2f}")
        np.atleast_1d(axes)[-1].legend(fontsize=6.5, loc="lower right", framealpha=.9)
        fig.suptitle("E4  LIVE: static vs adaptive protocols against real LLMs, "
                     "anti-sycophancy intervention active", fontsize=10)
        fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=180)
        plt.close(fig)

    # per-family flip rates: shows WHICH design family carries the detection
    cf = RESULTS / "live" / "e4_ceilings.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        fams = [x for x in c.columns if x.startswith("P")]
        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        width = .8 / max(1, len(fams))
        idx = np.arange(len(c))
        pal = ["#8c8c8c", "#3b7dd8", "#f0a202", "#d1495b"]
        for k, fam in enumerate(fams):
            ax.bar(idx + k * width, c[fam], width=width, label=fam, color=pal[k % len(pal)])
        ax.plot(idx + .4, c.ceiling, "k*", ms=11, label="ceiling (union of families)")
        ax.set_xticks(idx + .4)
        ax.set_xticklabels([f"{r.model.split('/')[-1].replace('-Instruct', '').replace('-instruct', '')}\n{r.condition}"
                            for r in c.itertuples()], fontsize=7.5)
        ax.set_xlim(-.3, len(c) - .1)
        _style(ax, "", "flip rate",
               "E4  Flip rate by pressure-design family. A static protocol sees only one bar;\n"
               "the design-space ceiling (star) is what is actually there to be found.")
        ax.legend(fontsize=7.5)
        fig.tight_layout(); fig.savefig(FIGURES / "e4_family_flip_rates.png", dpi=180)
        plt.close(fig)


def main():
    for fn in (fig_e1, fig_e1c, fig_e2, fig_e3, fig_e4):
        try:
            fn()
            print(f"  ok {fn.__name__}")
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    print("figures in", FIGURES)


if __name__ == "__main__":
    main()
