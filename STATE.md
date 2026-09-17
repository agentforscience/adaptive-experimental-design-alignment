# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-08-05T23:03:07.653438Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/adaptive-experimental-design-ead2`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase 1 (`resource_finder`) COMPLETE — 2026-08-05.** Marker: `.resource_finder_complete`.

### What was completed
- Isolated env: `uv venv` + `.venv/` + `pyproject.toml` with `[tool.uv] package = false`
  (needed — hatchling could not infer a package layout). Installed: pypdf, httpx, requests,
  pandas, numpy, scipy, matplotlib, datasets, pydantic, and `bayes_evals` (editable).
  **Activate with `source .venv/bin/activate`.**
- **44 papers** downloaded to `papers/` (0 failures); 4 deep-read chunk-by-chunk.
  528 arXiv records screened → `paper_search_results/`.
- **4 dataset groups** built in `datasets/` (rebuild: `python scripts/build_datasets.py`).
- **10 repos** cloned to `code/`; `bayes_evals` and `tinybenchmarks` installed and smoke-tested.
- Deliverables: `literature_review.md`, `resources.md`, `planning.md`,
  `papers/README.md`, `datasets/README.md` (+ `.gitignore`), `code/README.md`.

### Key findings
1. **`paper-finder` at localhost:8000 was NOT running.** Fell back to a manual arXiv sweep
   (`scripts/search_arxiv.py`, `scripts/fetch_by_id.py`). Semantic Scholar returned HTTP 429.
   If a later phase needs more literature, expect to use the same fallback.
2. **The hypothesis has a near-exact precedent**: Andriushchenko et al. 2024 (`papers/2404.02151`)
   argues adaptivity is essential for *attacks*; we generalize it to *protocols*.
   Llama-2-7B: prompt alone 0% → +random search 50% → +self-transfer **100%**.
3. **Critical nuance — "adaptive" is two different moves.** On R2D2 (adversarially trained),
   self-transfer gives only 12%, but switching *template family* gives 90%→100%. Parameter
   adjustment within a design family ≠ switching design family. This is the sharpest
   contribution available and the built-in negative control.
4. **Verified matched counterfactual from real released logs [computed this phase]**, same attack
   family and same 10k-iteration cap, differing only in initialization:
   `llama2-7b_plain_init` (static) = **0.00** ASR (judge 10/10, n=50), median 10,000 iters,
   56% censored at the cap; `llama2_7b` (self-transfer) = **0.96** ASR (n=51), **median 2** iters.
   Caveats (censoring; heuristic run boundaries; `plain_init` is harsher than the paper's
   headline "Prompt+RS" row) are in `datasets/README.md` §3 — do not cite it as reproducing
   the paper's Table 2.
5. **Closest prior work**: Jones et al. 2025 (`papers/2502.16797`) — elicitation probability turns
   a binary null into a continuous signal; Gumbel-tail extrapolation; §6 already does
   forecast-driven adaptive allocation of red-team compute. Read it before designing D1.
6. **Static baseline is well characterized**: BoN (`papers/2412.03556`) is memoryless by
   construction, follows −log(ASR)=aN^(−b), and — crucially — **resampling a *successful*
   jailbreak reproduces harm only ~20% of the time**, so single trials are very noisy.

### Evidence paths
- `planning.md` — direction budget: 11 directions enumerated, **3 kept**, 8 pruned with reasons.
- `literature_review.md` §6 — the numbers computed this phase, with caveats.
- `datasets/README.md` — schemas, rebuild instructions, 5 named caveats on the trajectories.
- `code/README.md` — tested examples for `bayes_evals` and `tinybenchmarks`; the
  `jailbreakbench`-needs-`litellm` workaround.
- `datasets/_build_report.json` — row counts and per-cell ASRs.

### Kept directions (do not re-expand the search space)
- **D1** Adaptive budget allocation over an attack × behavior portfolio at *matched compute*.
  Substrate: `datasets/jbb_artifacts/jbb_queries_to_jailbreak.csv` (5 attacks × 4 models × 100
  behaviors, with `queries_to_jailbreak`).
- **D2** Null-as-signal escalation ladder: continue vs. re-initialize vs. switch design family.
  Substrate: `datasets/rs_trajectories/` (~437k iteration rows, 672 run outcomes, 14 configs).
- **D3** Calibrated nulls: anytime-valid stopping + Gumbel-tail extrapolation, scored on coverage.
  Substrate: same trajectories, truncated at *t* and scored against the rest of the trace.

### Next phase: `experiment_runner` — concrete next steps
1. `source .venv/bin/activate`; if `datasets/` is empty, run `python scripts/build_datasets.py`
   (data files are git-ignored by design; the rebuild takes ~1 min and needs `code/` populated).
2. Read `planning.md` first — it contains 5 cross-cutting design requirements that constrain
   every arm. In particular: **equalize total compute**, or the result is vacuous.
3. Start with D2 on `datasets/rs_trajectories/`; it has the cleanest counterfactual and the
   built-in negative control. Then D1, then D3.
4. Treat `queries_to_jailbreak` and iteration counts as **right-censored survival data**.
5. Report both judge-based and rule-based ASR — they disagree (e.g. `llama2-7b_icl_one_shot`:
   0.66 vs. 0.10) and the disagreement is a finding, not noise.
6. Include the "continue-only adaptive" arm (more compute, no redesign). Without it, any win
   is attributable to compute rather than adaptivity.

### Unresolved uncertainties / risks
- **Elicitation probability is not directly observable** in the traces; we substitute
  target-token logprob (which Jones et al. themselves use for most experiments). The mapping is
  an approximation and must be stated in the write-up, not buried.
- **One realization per cell** in the JBB matrix — no repeat-run variance. All uncertainty must
  come from bootstrapping over behaviors/trajectories, following Hughes et al.'s protocol.
- **Run-boundary recovery is heuristic** (iteration-counter decreases). Counts land within ±2 of
  the expected 50 per config, but `run_id` is approximate in multi-restart configs.
- **Goal strings unrecoverable for 3 configs** (`r2d2`, `*_icl_one_shot`) — those trajectories
  cannot be joined to behavior metadata.
- **Judge noise may be the binding constraint** on any efficiency claim: ~20% GPT-4-judge false
  positives on Claude 2.1 (Andriushchenko §4.4), and Dorner et al. (`papers/2410.13341`) prove a
  hard ceiling on how far an LLM judge can substitute for labels.
- **Disputed dimensionality of the vulnerability boundary** (Arditi 1-D vs. Wollschläger concept
  cones vs. Che et al. low-dimensional subspace). Treat as a moderator, not as settled.
- No live-model arm and no GPU; all three directions were selected to run CPU-only and offline.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase 2 (`experiment_runner`) COMPLETE — 2026-08-05.**

### What was completed
Four experiments, all at **matched compute** (the non-negotiable from `planning.md`).
Code in `src/`, outputs in `results/offline/`, `results/live/`, `figures/`.
Deliverables: `REPORT.md`, `README.md`, updated `planning.md` (Phase 0/1 sections appended).

- **E1** (`src/e1_ladder.py`) — escalation-ladder replay on `rs_trajectories`, 3 design
  families on Llama-2-7B-chat, 16 budgets from 1 to 10,000 iterations, n=50 behaviours.
- **E1c** — null-diagnosticity: AUC of the continuous signal among runs *still null* at t.
- **E2** (`src/e2_portfolio.py`) — portfolio allocation on the JBB 1,800-cell matrix,
  8 policies x 8 budgets x 4 target models x 300 replicates.
- **E3** (`src/e3_calibration.py`) — coverage of upper bounds emitted on a terminal null,
  14 configs x 400 bootstrap replicates.
- **E4** (`src/e4_run.py`, `src/e4_protocols.py`) — LIVE arm on real open-weights LLMs.

### Key findings (all compute-matched)
1. **The false null is real and total.** `STATIC-d0` detects **0/50** at *every* budget up
   to 10,000 iters; reallocating the same budget across 3 design families reaches **1.00**
   (McNemar p<1e-6, h~2.4). This is the motivating phenomenon, confirmed at extreme size.
2. **The hypothesised mechanism FAILS.** vs `FIXED-SPLIT` (same budget, same designs, blind
   switch schedule), signal-triggered `LADDER` gains +0.02 (p=0.73) at B=60 and *loses*
   -0.12 (95% CI [-0.22,-0.04]) at B=1000. 27-cell trigger sweep: max gain +0.06.
   **All of the "adaptivity" benefit is design-family diversification, not the trigger.**
   Rung attribution: 0 detections from rung 0, 31 from rung 1, 13 from rung 2.
3. **E1c explains it:** pooled AUC 0.47-0.61, every CI covering 0.5. Nulls carry no
   within-run signal about whether persistence pays off.
4. **E2 is the positive result:** portfolio-level adaptivity works. Thompson vs round-robin
   at B=100: **+0.648** on GPT-4 (0.703 vs 0.040), +0.321 on GPT-3.5 (Holm p<0.001).
   `escal+thompson` (0.860) beats even the ORACLE best-single-method (0.780) on GPT-4.
   **Negative case:** loses -0.185 on Llama-2-7B/Vicuna where a dominant method (DSN 0.94)
   exists — exploration is pure overhead there. This is the built-in negative control.
5. **E3 refutes the calibration half:** nominal 95% bounds get **0.626-0.671** coverage.
   Anytime-valid (union-bound) correction moves it only 0.638 -> 0.671. **The binding
   problem is the estimand, not optional stopping.** On total nulls, power-law
   extrapolation is undefined 100% of the time. Per-config coverage is **0.00** for
   llama2-13b/70b/llama2_7b (truth 0.96-1.00).

### Deviation from the resource phase
`OPENROUTER_KEY` was **unset** (verified), so the brief's API route was unavailable. But
**4x RTX A6000 were free**, so R8 was un-pruned in a cheaper offline-safe form: E4 runs
**real open-weights models locally** (Qwen2.5-7B-Instruct, Qwen2.5-1.5B-Instruct,
Phi-3.5-mini-instruct) on GSM8K sycophancy. No simulated LLM anywhere. Rationale recorded
in `planning.md` under "Deviation from the resource phase's assumptions".

### Evidence paths
- `REPORT.md` §4 — all result tables; `figures/*.png` — 5 figures.
- `results/offline/e1_tests.json`, `e2_tests.json` — Holm-adjusted tests.
- `results/offline/e1c_pooled_auc.csv` — the mechanism result.
- `results/offline/e3_coverage_pooled.csv`, `e3_coverage_total_nulls.csv`.
- `results/live/e4_grid_*.csv`, `e4_ceilings.csv`, `e4_protocol_summary.csv`.

### Debugging notes worth keeping
- **`src/e4_live.py` was reverted mid-session by an external process**, silently undoing
  in-place patches (twice). `src/e4_run.py` is a self-contained rewrite with every fix
  baked in; prefer it. `e4_live.py` is retained for its experiment documentation.
- Two real bugs cost significant GPU time and would corrupt any rerun:
  (a) re-tokenising an already-templated chat string **must** use
  `add_special_tokens=False` — double-BOS drove Qwen2.5-7B GSM8K accuracy to 0.31 and
  produced degenerate repetition loops; with the fix it is **0.844**.
  (b) Qwen-family models emit `\boxed{...}` rather than the requested `ANSWER:` line;
  answer extraction must accept both or the last-number fallback silently grabs an
  intermediate value.
- Phi-3.5-mini needs `trust_remote_code=False` (its bundled remote code calls
  `DynamicCache.seen_tokens`, removed in current transformers).
- Duplicate `run_id`s exist in `_run_outcomes.csv` (restart splitting) — dedupe on load.

### Unresolved uncertainties
- E1c's verdict rests on target-token probability as the elicitation-probability proxy.
  A better signal (repeated-sampling elicitation probability, or a learned probe) could
  flip E1. This is the single most important threat and is stated in `REPORT.md` §6.6.
- Replay semantics treat a run the original authors ended without success as a failure even
  if budget remained (no data beyond `n_iters`). Applied identically to all arms.
- Rung 2 is matched by resampling (goal strings unrecoverable for in-context configs);
  the exactly-paired 2-rung variant is reported alongside and agrees qualitatively.
- E4 items are GSM8K, so contamination cannot be excluded; the outcome is answer *flipping*
  under pressure, not accuracy, which mitigates but does not eliminate the concern.

### Next steps if the work continues
1. Test whether a better within-run signal is diagnostic where target-token prob is not.
2. Build a bound whose estimand is elicitability *under a specified design space* and
   score its coverage against held-out design families.
3. Extend E2 to heterogeneous method costs (unit-cost accounting flatters cheap methods).
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
