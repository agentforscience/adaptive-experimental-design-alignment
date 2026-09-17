# Adaptive Experimental Design for Detecting Hidden Robustness in AI Alignment Interventions

Does an *adaptive* evaluation protocol — one that treats its own null results as triggers for
experimental redesign — find more vulnerabilities than a static protocol at the **same compute
budget**, and does it produce better-calibrated evidence when it still finds nothing? We test
this by replaying protocols against real published attack traces and a real cost-to-discovery
matrix at matched budget, plus a live arm run against open-weights LLMs on local GPUs.

**Full write-up: [REPORT.md](REPORT.md).** Pre-registered plan: [planning.md](planning.md).

## Key findings

- **The false-null phenomenon is real and extreme.** The default design detects **0/50**
  behaviours at every budget up to 10,000 iterations on Llama-2-7B-chat; reallocating the
  *same* budget across three design families reaches **1.00** (McNemar p < 10⁻⁶, Cohen's
  h ≈ 2.4). A static protocol reports total robustness where near-total vulnerability exists.
- **But the "null as trigger" mechanism does not work.** Against a comparator that switches
  design family on a **blind fixed schedule** with identical budget and identical designs,
  signal-triggered escalation gains nothing (Δ = +0.02, p = 0.73) and loses at large budgets
  (Δ = −0.12, 95% CI [−0.22, −0.04]). A 27-cell sweep of the trigger's hyper-parameters never
  gains more than +0.06.
- **Because nulls are not diagnostic.** Conditional on a run still being null at iteration
  *t*, the continuous signal predicts eventual success at pooled **AUC 0.47–0.61**, CIs
  covering 0.5 everywhere.
- **Adaptivity *does* work one level up.** Thompson sampling over an attack portfolio beats an
  equally diversified static round-robin by **+0.65** on GPT-4 at matched budget (0.70 vs
  0.04, Holm p < 0.001) and exceeds even an oracle single-method choice.
- **Adaptive nulls are badly calibrated.** Nominally 95% upper bounds achieve **63–67%**
  coverage, and an anytime-valid correction moves it only 0.638 → 0.671 — the binding problem
  is the estimand, not optional stopping.

**Bottom line:** the hypothesis is right about the conclusion and wrong about the mechanism.
Budget for several design families up front and spend adaptivity *across* them, not inside one.

## Reproducing

```bash
uv venv && source .venv/bin/activate      # isolated env; deps in pyproject.toml
uv sync

# rebuild the datasets if datasets/ is empty (data files are git-ignored by design)
python scripts/build_datasets.py

# offline experiments (CPU, < 5 min total)
python src/e1_ladder.py          # escalation ladder + null-diagnosticity (E1, E1c)
python src/e2_portfolio.py       # compute-matched portfolio allocation (E2)
python src/e3_calibration.py     # calibration of terminal nulls (E3)

# live arm (needs a CUDA GPU; ~10 min per model on an RTX A6000)
HF_HOME=$PWD/.hf_cache CUDA_VISIBLE_DEVICES=0 \
  E4_MODEL=Qwen/Qwen2.5-7B-Instruct E4_ITEMS=32 E4_BATCH=16 E4_MAXTOK=576 \
  E4_OUT=results/live/e4_grid_qwen7b.csv .venv/bin/python src/e4_run.py
python src/e4_protocols.py       # replay protocols against the live grid

python src/make_figures.py       # all figures
```

Seed 42 throughout; live decoding is greedy (temperature 0) so the grid is deterministic.

## Layout

```
src/
  common.py           loading, seeding, bootstrap CIs, McNemar, Holm, AUC
  e1_ladder.py        E1  escalation ladder replay at matched budget; E1c null diagnosticity
  e2_portfolio.py     E2  budget allocation over the attack x behaviour portfolio
  e3_calibration.py   E3  coverage of upper bounds emitted on a terminal null
  e4_run.py           E4  live grid against real open-weights LLMs (self-contained runner)
  e4_live.py          E4  reference implementation + full experiment documentation
  e4_protocols.py     E4  protocol replay against the cached live grid
  make_figures.py     all figures
results/offline/      E1-E3 raw outputs, summaries, statistical tests
results/live/         E4 grid, protocol replay, per-family flip rates
figures/              publication figures
datasets/             substrates (data git-ignored; see datasets/README.md to rebuild)
papers/ code/         44 papers and 10 repos gathered in the resource phase
literature_review.md  synthesis  ·  resources.md  catalogue  ·  planning.md  plan
```

## Scope and ethics

E1–E3 replay **already-published** benchmark artifacts and generate no new attack content.
E4 uses a non-harmful failure mode (sycophantic answer-flipping on grade-school maths) and
produces no jailbreak artifacts. This project studies *how to design robustness experiments*;
it does not develop new attack capability.
