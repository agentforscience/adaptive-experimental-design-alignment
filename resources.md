# Resources Catalog

**Project:** Adaptive Experimental Design for Detecting Hidden Robustness in AI Alignment Interventions
**Phase:** `resource_finder` (Phase 1) — complete
**Date:** 2026-08-05

## Summary

| Resource | Count | Location |
|---|---|---|
| Papers downloaded (PDF) | **44** | `papers/` |
| Papers deep-read chunk-by-chunk | 4 | `papers/pages/` |
| arXiv records screened | 528 | `paper_search_results/` |
| Datasets built | **4 groups / 17 files** | `datasets/` |
| Repositories cloned | **10** | `code/` |
| Research directions kept | 3 of 11 | `planning.md` |

Everything needed to run the three planned directions is **on disk and offline**. No GPU and no
API key is required.

---

## Papers

44 PDFs, all verified as valid PDFs, 0 failed downloads. Full annotated list with per-paper
relevance notes: **`papers/README.md`**. Synthesis: **`literature_review.md`**.

### Deep-read (chunked and read in full)

| Title | Authors | Year | File | Key contribution to this project |
|---|---|---|---|---|
| Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks | Andriushchenko, Croce, Flammarion | 2024 | `papers/2404.02151_*.pdf` | Its thesis *is* our hypothesis. Llama-2-7B: 0% → 50% → **100%** as adaptation is added. R2D2 negative control: self-transfer 12% but template switch 90%. Ships the attack logs we use as data. |
| Best-of-N Jailbreaking | Hughes et al. | 2024 | `papers/2412.03556_*.pdf` | The memoryless static baseline. Power law −log(ASR)=aN^(−b); forecast from 10× fewer samples at 4.4% error; **resampling a *successful* jailbreak reproduces harm only ~20% of the time.** |
| Forecasting Rare Language Model Behaviors | Jones et al. (Anthropic) | 2025 | `papers/2502.16797_*.pdf` | Closest prior work. Elicitation probability converts a null into a continuous signal; Gumbel-tail extrapolation; §6 does forecast-driven adaptive compute allocation. |
| Adding Error Bars to Evals | Miller (Anthropic) | 2024 | `papers/2411.00640_*.pdf` | Evals as experiments. Clustered SEs up to **3.05×** wider; sample-size/MDE formula; K-resampling cuts MDE 13.2%→7.5%. |

### Screened (title + abstract), grouped

- **A. Automated / adaptive red teaming (17):** GCG, PAIR, TAP, Rainbow Teaming, RainbowPlus,
  QDRT, Curiosity-driven RT, HarmBench, GOAT, MHJ multi-turn human jailbreaks, Perez et al.,
  Ganguli et al., Jailbreak Foundry (2026), Jailbreak Scaling Laws (2026), Process Mining (2026).
- **B. Hidden robustness / boundary conditions (13):** Jailbroken, Shallow Safety Alignment,
  Sleeper Agents, Fine-tuning Compromises Safety, Refusal Direction, Geometry of Refusal, LAT,
  Model Tampering, Adversarial Unlearning, Constitutional Classifiers, Emergent Misalignment,
  Lynch thesis (2026), Inference-Time Vulnerability (2026).
- **C. Statistical / adaptive evaluation methodology (14):** Safe Anytime-Valid Inference,
  Anytime-Valid Linear Models, Don't Use the CLT, tinyBenchmarks, Limits to Scalable Evaluation,
  Deep Adaptive Design, Modern Bayesian Experimental Design, ConSol/SPRT, CAT for medical
  benchmarking (2026), FAQ (2026), Valid Best-Model Identification (2026), WILD+OED (2026).

---

## Datasets

Detailed schemas, download instructions, and caveats: **`datasets/README.md`**.
Rebuild everything with `python scripts/build_datasets.py`. Data files are git-ignored; 10-row
samples are committed in `datasets/samples/`.

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| **Random-search trajectories** | Parsed from `llm-adaptive-attacks/attack_logs` (21 MB raw) | ~437k iteration rows + 672 run outcomes, 3.5 MB gz | Sequential attack traces with a continuous objective | `datasets/rs_trajectories/` | **Primary substrate for D2/D3.** 14 configs. Contains a matched adaptive-vs-static pair. 5 documented caveats |
| **JBB cost-to-discovery matrix** | JailbreakBench artifacts | 1,800 rows, 81 KB | Queries-to-jailbreak per (attack, model, behavior) | `datasets/jbb_artifacts/` | **Primary substrate for D1.** 5 attacks × 4 models × 100 behaviors × 10 categories. `queries_to_jailbreak` is **right-censored** |
| HarmBench behaviors | `centerforaisafety/HarmBench` | 400 / 320 / 80 | Harmful behavior prompts | `datasets/behaviors/` | Official all/test/val splits |
| AdvBench | `llm-attacks/llm-attacks` | 520 + 574 | Harmful behaviors & strings | `datasets/behaviors/` | The `goal`/`target` format every suffix attack uses |
| AdvBench-50 (PAIR curation) | `tml-epfl/llm-adaptive-attacks` | 50 | Harmful behaviors | `datasets/behaviors/` | **Join key** for `rs_trajectories` |
| JBB-Behaviors | HF `JailbreakBench/JBB-Behaviors` | 100 + 100 | Harmful + index-aligned benign | `datasets/behaviors/` | The benign set is the **over-refusal control** |
| StrongREJECT | `alexandrasouly/strongreject` | 313 + 60 | Harmful prompts + rubric | `datasets/behaviors/` | Fine-grained rubric grader in `code/strong_reject` |
| BoN direct requests | `jplhughes/bon-jailbreaking` | 159 | Harmful requests | `datasets/behaviors/` | Exactly the subset used in the Best-of-N paper |

### The headline empirical finding from this phase

Parsing the released logs produced a **matched adaptive-vs-static counterfactual** on
Llama-2-Chat-7B — same attack family, same 10,000-iteration cap, differing only in initialization
policy:

| Config | ASR (GPT-4 judge 10/10) | ASR (rule) | median iterations | mean iterations | % runs hitting the cap |
|---|---|---|---|---|---|
| `llama2-7b_plain_init` (static init) | **0.00** (n=50) | 0.00 | 10,000 | 6,735 | 56% |
| `llama2_7b` (self-transfer init) | **0.96** (n=51) | 0.88 | **2** | 14.2 | 0% |

One adaptive re-initialization step turns a total null into near-total success at roughly three
orders of magnitude lower query cost. Caveats (censoring, heuristic run boundaries, and the fact
that `plain_init` is a harsher ablation than the paper's headline row) are in
`datasets/README.md` §3.

---

## Code repositories

Detailed per-repo notes, tested examples, and running requirements: **`code/README.md`**.

| Name | URL | Purpose | Location | Status |
|---|---|---|---|---|
| llm-adaptive-attacks | `tml-epfl/llm-adaptive-attacks` | Released per-iteration attack logs | `code/llm-adaptive-attacks/` | **Data extracted** |
| jailbreakbench-artifacts | `JailbreakBench/artifacts` | Cost-to-discovery matrix | `code/jailbreakbench-artifacts/` | **Data extracted** |
| bayes_evals | `sambowyer/bayes_evals` | Small-sample Bayesian eval intervals | `code/bayes_evals/` | **Installed & smoke-tested** |
| tinybenchmarks | `felipemaiapolo/tinyBenchmarks` | Offline IRT parameters + `fit_theta` | `code/tinybenchmarks/` | **Verified offline-usable** |
| harmbench | `centerforaisafety/HarmBench` | Behavior sets; canonical harness | `code/harmbench/` | Data used; not run (needs GPU + 5 API SDKs) |
| jailbreakbench | `JailbreakBench/jailbreakbench` | Artifact schema; judges | `code/jailbreakbench/` | Schema read; package import needs `litellm` (documented workaround) |
| bon-jailbreaking | `jplhughes/bon-jailbreaking` | BoN augmentations; 159 requests | `code/bon-jailbreaking/` | Data used; trajectories not shipped |
| strong_reject | `dsbowen/strong_reject` | 12 judge/grader templates | `code/strong_reject/` | Templates extracted |
| pair | `patrickrchao/JailbreakingLLMs` | PAIR reference | `code/pair/` | Algorithmic reference |
| tap | `RICommunity/TAP` | TAP reference | `code/tap/` | Algorithmic reference |

---

## Resource gathering notes

### Search strategy

The `paper-finder` service at `localhost:8000` **was not running**
(`ConnectError`), so the prescribed primary method was unavailable and a manual sweep was used
instead:

1. **22 arXiv API queries** across four topic clusters (adaptive/sequential design; automated red
   teaming; robustness of alignment interventions; evaluation statistics) → 269 unique records.
2. **10 supplementary queries** for terms the first pass missed (quality-diversity, anytime-valid,
   SPRT, HarmBench/AdvBench, circuit breakers, latent adversarial training, unlearning) → 528
   unique records total.
3. **30 anchor papers fetched by arXiv ID** — canonical works known to be load-bearing that
   keyword search ranked poorly (GCG, PAIR, TAP, Sleeper Agents, BoN, Forecasting Rare Behaviors,
   Modern BED, …). 27 resolved to the intended paper; 3 ID guesses were wrong and were dropped
   (two were unrelated papers; one, `2410.13341`, turned out to be *Limits to Scalable Evaluation*
   and was kept because it is genuinely relevant).
4. Keyword scoring + manual review of the top ~110 to pick 44 for download.

Scripts: `scripts/search_arxiv.py`, `scripts/fetch_by_id.py`, `scripts/download_papers.py`.
Raw results: `paper_search_results/*.jsonl`.

### Selection criteria

A paper was downloaded if it (a) makes adaptivity or sequential design the object of study,
(b) documents a case where a static evaluation produced a false null about an alignment
intervention, or (c) supplies statistical machinery for inference under adaptive stopping. Papers
that merely propose another jailbreak were largely excluded — the keyword sweep surfaced dozens,
and they neither test the hypothesis nor calibrate it.

### Challenges encountered

| Challenge | Resolution |
|---|---|
| `paper-finder` service down | Fell back to arXiv API + anchor-ID fetching. Cost: relevance ranking had to be done by hand, and the keyword sweep was heavily skewed toward jailbreak-method papers |
| Semantic Scholar API returned HTTP 429 immediately | Not used. arXiv coverage was sufficient for this literature, which is almost entirely on arXiv |
| Attack logs have no run delimiters | Inferred run boundaries from iteration-counter decreases; validated against the expected 50 behaviors per config (recovered counts within ±2). Documented as a caveat |
| Goal strings unparseable in 3 configs | The in-context templates phrase requests differently. Added two fallback regexes; three configs still unjoinable to behavior metadata. Documented |
| `jailbreakbench` package import fails without `litellm` | Read the artifact JSON directly in `build_datasets.py`; documented the workaround |
| BoN attack trajectories not in the repo | Behind a manual Google Drive link. Used the published power law as the analytic stand-in for the memoryless baseline |
| `uv add` failed on the initial `pyproject.toml` | hatchling could not infer a package layout; switched to `[tool.uv] package = false` |

### Gaps and workarounds

- **No repeat-run variance in the JBB matrix.** One realization per cell. Workaround: uncertainty
  must come from bootstrapping over behaviors and trajectories (following Hughes et al.'s
  bootstrap protocol), not from re-running attacks.
- **Elicitation probability is not directly observable** in the on-disk traces. Workaround: use
  target-token logprob as the proxy — which is what Jones et al. themselves use for most of their
  experiments — and state the approximation rather than hiding it.
- **No live-model arm.** All three kept directions were chosen to be executable offline. If API
  keys are supplied later, a small live arm can be added, but Andriushchenko et al. §4.3 documents
  that API nondeterminism corrupts the search signal, so it would need its own controls.
- **Human-adaptive red teaming (MHJ) unavailable.** Gated dataset; direction pruned (R6). Cited as
  motivation only.

---

## Recommendations for experiment design

**1. Primary datasets**
- `datasets/rs_trajectories/` for directions D2 and D3 — real sequential traces with a continuous
  objective, a matched adaptive/static pair, and a built-in negative control (R2D2).
- `datasets/jbb_artifacts/jbb_queries_to_jailbreak.csv` for direction D1 — a complete
  cost-to-discovery matrix that any allocation policy can be replayed against.
- `datasets/behaviors/jbb_behaviors_benign.csv` as the over-refusal control in any arm that
  claims to improve detection.

**2. Baselines**
- Static uniform allocation (the field default) — the thing the hypothesis must beat.
- Round-robin, and an oracle allocation as the upper bound.
- BoN-style memoryless resampling, using the published power law as its analytic form.
- Continue-only adaptive (more compute, no redesign) — **this is the critical control**, because
  without it any win is attributable to compute rather than adaptivity.

**3. Metrics**
- Vulnerabilities found at **matched total budget** (primary outcome).
- Queries-to-discovery as **censored survival data** (56% of `plain_init` runs are censored;
  naive means are badly biased).
- Calibration of the terminal null: forecast vs. realized worst-query risk, using the Gumbel-tail
  method truncated at iteration *t* and scored against the rest of the same trace.
- CI coverage under optional stopping (anytime-valid vs. naive CLT).
- **Both** judge-based and rule-based ASR, reported side by side — they disagree, and the
  disagreement is itself a result.

**4. Code to reuse**
- `code/bayes_evals` — installed and working; use for every small-*n* interval.
- `code/tinybenchmarks` — offline IRT parameters and `fit_theta` for item-selection machinery.
- `scripts/build_datasets.py` — the parsers; extend it rather than re-parsing logs ad hoc.
- `code/pair` and `code/tap` — read for control structure (refine-on-refusal, prune-before-query);
  the planned allocation policies are generalizations of these.

**5. Non-negotiables**
- Equalize compute in every adaptive-vs-static comparison.
- Use anytime-valid inference wherever the stopping rule depends on the data.
- Include an adaptation that is expected to fail (R2D2 + self-transfer). A method that wins
  everywhere is leaking information.
- Do not fit power laws below N ≈ 10, and expect under-estimation on high-ASR arms.

**Ethical scope.** All data here is published safety-benchmark material used for defensive
research into evaluation methodology. The project studies *how to design robustness experiments*;
it does not develop new attack capability, and the experiment phase should not produce or release
new jailbreak artifacts.
