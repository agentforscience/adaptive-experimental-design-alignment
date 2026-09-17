# Literature Review: Adaptive Experimental Design for Detecting Hidden Robustness in AI Alignment Interventions

**Phase:** `resource_finder` · **Corpus:** 44 PDFs in `papers/` · **Search:** 528 unique arXiv
records screened (`paper_search_results/arxiv_all.jsonl`)

> **Note on method.** The `paper-finder` service at `localhost:8000` was not running, so this
> review was built from a manual sweep: 32 arXiv API queries across four topic clusters plus 30
> anchor papers fetched by ID. Everything below is derived from PDFs actually downloaded and read
> (deep-read papers are marked ★). Numbers quoted from papers are attributed; numbers computed in
> this phase from the on-disk logs are explicitly labelled **[computed here]**.

---

## 1. Research area overview

The hypothesis sits at the intersection of three literatures that have, so far, barely talked to
each other.

1. **Automated red teaming** has produced a decade-compressed arms race of attack algorithms
   (GCG → PAIR → TAP → Rainbow Teaming → BoN). These are *implicitly* adaptive — almost all of
   them iterate on failure — but adaptivity is treated as an implementation detail of the attack,
   never as the experimental variable under study.
2. **Robustness of alignment interventions** has repeatedly found that interventions which look
   robust under one evaluation collapse under another (shallow safety alignment, sleeper agents,
   unlearning-is-not-removal, fine-tuning attacks). The recurring lesson is that *a null result
   from a fixed evaluation is very weak evidence of robustness*.
3. **Statistical methodology for LLM evaluation** is very new (2024–2026) and is only now
   importing standard experiment-design tools: standard errors, clustering, power analysis,
   anytime-valid inference, item response theory, and Bayesian optimal experimental design.

The gap the hypothesis targets is precisely the join: **treating the red-teaming protocol itself
as an experimental design object, and treating a null as a datum that carries information about
where to look next.** The closest existing work is Jones et al. (2025) on forecasting rare
behaviors, which converts nulls into continuous elicitation probabilities and — in its final
section — uses forecasts to allocate red-teaming compute. That is a single, narrow instance of the
general claim.

---

## 2. Cluster A — Automated / adaptive red teaming

### ★ Andriushchenko, Croce & Flammarion (2024), *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks* — `papers/2404.02151_*.pdf` (ICLR 2025)

**The single most important paper for this hypothesis.** Its thesis *is* the hypothesis, stated
for attacks rather than for experiments: "adaptive attacks are crucial for accurate robustness
evaluations of LLMs… no single method can generalize across all target models."

- **Method.** Manually designed prompt template (rules + request + adversarial suffix), then
  **random search** on a 25-token suffix maximizing the logprob of the target first token
  ("Sure"), up to 10,000 iterations and 10 restarts. Plus three adaptation mechanisms:
  *self-transfer* (initialize the search for a hard request from a suffix that already worked on
  an easier one), *transfer* (reuse across models), and *prefilling* (for APIs that allow it).
- **Data.** 50 AdvBench behaviors curated by Chao et al. (PAIR). GPT-4 judge, success = 10/10.
- **The ablation that matters** (their Table 2, Llama-2-Chat-7B):

  | Method | ASR |
  |---|---|
  | Prompt alone | 0% |
  | Prompt + random search | 50% |
  | Prompt + random search + self-transfer | **100%** |

  Same attack family, same budget structure — the only change is *using a prior success as the
  initialization*. 0% → 100%.
- **The negative control that matters even more** (same table, R2D2, an adversarially trained
  model): Prompt + RS + self-transfer = **12%**; switching to an *in-context* template = **90%**;
  + RS = 100%. Self-transfer, the adaptation that saved Llama-2, is useless here. R2D2 was trained
  against suffix-structured attacks, so escaping requires changing the *structure*, not the
  parameters. **Adaptivity is not one thing:** parameter adjustment within a design family and
  switching design family are different moves with different success conditions.
- Also: GPT-4o gives 0% with the default template but 72% with a custom one, then 100% with
  RS + self-transfer.
- **Caveat the paper raises itself:** GPT-4 as judge shows ~20% false positives on Claude 2.1
  (<5% elsewhere), and GPT-3.5/4 logprobs are nondeterministic even at temperature 0 with a fixed
  seed (their Fig. 3), which degrades the random-search signal.
- **Code/artifacts:** `github.com/tml-epfl/llm-adaptive-attacks` — cloned to
  `code/llm-adaptive-attacks/`, including **21 MB of per-iteration attack logs**. See §6.

### ★ Hughes et al. (2024), *Best-of-N Jailbreaking* — `papers/2412.03556_*.pdf`

**The canonical *non-adaptive* baseline.** BoN resamples i.i.d. augmentations (character
scrambling, random capitalization, character noising) with **no feedback whatsoever** between
attempts — memoryless by construction. This makes it the ideal static control condition.

- 78% ASR on Claude 3.5 Sonnet and 89% on GPT-4o at N = 10,000; 52% against circuit breakers;
  67% against GraySwan Cygnet. Extends to vision (56% GPT-4o) and audio (72% GPT-4o Realtime).
- **Power-law scaling:** −log(ASR) = a·N^(−b) holds over orders of magnitude across models and
  modalities. Fitting on N = 1,000 forecasts ASR at N = 10,000 with **4.4% error** (text; 6.3%
  vision, 2.5% audio). Slope *b* is similar across models; the intercept varies. Forecasts
  systematically **under**-estimate for high-ASR models.
- **Three findings that directly constrain our design:**
  1. Augmentation, not resampling, is what works — BoN's log-log slope is *steeper* than a
     temperature-1 resampling baseline (22× ASR improvement on GPT-4o at N = 500). So the
     variance must be injected into the *input space*, not just the output sampler.
  2. Temperature barely matters (0.7%–27.7% ASR drop from T=1 to T=0), so input-space entropy
     dominates output-space entropy.
  3. **Resampling a *successful* BoN jailbreak reproduces harm only ~20% of the time.** A single
     trial — success or failure — is a very noisy measurement. This is the quantitative core of
     why a static single-shot design produces false nulls.
- Composing BoN with many-shot jailbreaking gives a **28× sample-efficiency gain** (N to reach
  74% ASR on Sonnet: 6,000 → 274) — evidence that composition of design elements, not just more
  samples, drives efficiency.

### ★ Jones et al. (2025), *Forecasting Rare Language Model Behaviors* — `papers/2502.16797_*.pdf` (Anthropic)

**The closest prior work, and the source of our core measurement primitive.**

- **Core move: replace the binary null with a continuous score.** Define the *elicitation
  probability* p_ELICIT(x) = P(behavior | query x). Failed jailbreaks have small-but-nonzero
  elicitation probabilities. This is exactly "treat a null as actionable feedback".
- **Gumbel-tail method.** Let ψ = −log(−log p_ELICIT). Extreme value theory ⇒ the tail of the log
  survival function is approximately linear in ψ, so Q_ψ(n) = −(1/a)(log n − b). Fit *a*, *b* by
  OLS on the **ten highest** elicitation scores in the evaluation set, then extrapolate.
- **Three risk metrics forecastable from the quantiles:** worst-query risk (max elicitation
  probability over n queries), behavior frequency (fraction above threshold τ), aggregate risk
  (1 − Π(1 − p_i)).
- **Accuracy.** Forecasting from m = 900 to n = 90,000 (2 orders of magnitude): within one order
  of magnitude of true risk for **86%** of misuse forecasts. Average absolute log error 1.7
  (Gumbel-tail) vs. 2.4 (log-normal baseline). Crucially, Gumbel-tail **under**-estimates only 34%
  of the time vs. 72% for log-normal — under-estimates are the dangerous direction because they
  give false security.
- **Aggregate risk can be high even when no single query has high elicitation probability** —
  compounding across a large deployment. A per-query null tells you almost nothing about
  deployment-scale risk.
- **§6, Applications to red teaming:** forecasts identify the *compute-optimal* red-teamer.
  In their example Sonnet has both higher worst-query risk and a faster growth rate, yet the
  forecast correctly says sampling 10× more from the cheaper Haiku is optimal. **This is
  forecast-driven adaptive allocation, and it is the direct ancestor of direction D1.**
- The paper itself proposes, as future work, *adaptively stopping* sampling from queries unlikely
  to be in the top quantiles — i.e. exactly the adaptive design our hypothesis asks about.
- **Stated limitations:** the fit uses only the 10 largest elicitation probabilities, so it is
  sensitive to the specific evaluation set; the eval set may not reach the true tail.

### Chao et al. (2023), PAIR — `papers/2310.08419_*.pdf`
Attacker LLM iteratively refines a candidate jailbreak against a target using the target's
*refusal* as feedback. Often < 20 queries. The original "null-triggers-redesign" loop, though the
adaptivity is never isolated as a variable.

### Mehrotra et al. (2023), TAP — `papers/2312.02119_*.pdf`
Tree-of-attacks with **pruning**: branch the refinement, and prune candidates judged unlikely
*before* querying the target. >80% ASR on GPT-4-Turbo/4o with fewer queries than PAIR. Pruning is
a budget-allocation policy — directly relevant to D1.

### Samvelyan et al. (2024), Rainbow Teaming — `papers/2402.16822_*.pdf`
Casts adversarial prompt generation as **quality-diversity** search (MAP-Elites-style archive over
risk-category × attack-style descriptors). >90% ASR across Llama 2/3. The archive *is* an adaptive
design: it maintains coverage of the design space rather than hill-climbing one point. Follow-ups
on disk: RainbowPlus (2504.15047, multi-element archive) and QDRT (2506.07121).

### Zou et al. (2023), GCG — `papers/2307.15043_*.pdf`
Greedy Coordinate Gradient; universal and transferable suffixes. The white-box reference attack
and the thing R2D2 was adversarially trained against — which is why R2D2 resists suffixes but
falls to in-context prompts.

### Hong et al. (2024), Curiosity-driven Red Teaming — `papers/2402.19464_*.pdf`
RL red-teamer with a novelty bonus. Frames the failure of prior RL red teaming as *low coverage*,
not low per-attempt success — the same coverage-vs-depth tradeoff that D1 must navigate.

### Perez et al. (2022) — `papers/2202.03286_*.pdf`; Ganguli et al. (2022) — `papers/2209.07858_*.pdf`
The foundational red-teaming papers. Ganguli et al. is the important one for us: it studies
**scaling behavior of red teaming** across 3 model sizes × 4 model types and finds RLHF models get
*harder* to red team with scale while other types stay flat. Releases 38,961 human red-team
attacks. This is the earliest evidence that a fixed red-teaming protocol has model-dependent
sensitivity — i.e. that static protocols mismeasure.

### Li et al. (2024), *LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet* — `papers/2408.15221_*.pdf`
**>70% ASR on HarmBench against defenses reporting single-digit ASR under automated single-turn
attacks.** Also recovers dual-use biosecurity knowledge from unlearned models. Releases MHJ
(2,912 prompts / 537 multi-turn jailbreaks). The largest published gap between what a static
automated protocol concludes and what an adaptive human protocol finds — the strongest motivating
statistic for the hypothesis, even though we cannot run humans here.

### Pavlova et al. (2024), GOAT — `papers/2410.01606_*.pdf`
Agentic multi-turn red teamer that reasons over a toolbox of published attack techniques and
picks one per turn. An explicit *design-family-switching* policy — the operationalization D2 needs.

### Halder et al. (2026), *Jailbreak Scaling Laws: Polynomial–Exponential Crossover* — `papers/2603.11331_*.pdf`
Finds ASR grows *polynomially* in inference-time samples without prompt injection but
*exponentially* with injection, and gives a statistical-mechanics (spin-glass, RSB) model
predicting the crossover. Consistent across 3B–70B models. Directly relevant: the *return curve
shape* determines whether adaptive allocation can beat uniform allocation at all.

### Topol (2026), *Beyond Pass/Fail: Process Mining Red Team Attacks* — `papers/2606.07833_*.pdf`
Applies process mining to red-teaming traces: 60 HarmBench prompts × 2 models × 10 mutators × up
to 110 attempts = 8,575 scored events. Finds **structurally distinct defense profiles invisible to
ASR alone** — GPT-OSS has a near-absorbing refusal state; Llama 3.3 has multiple porous escape
routes — and that **mutator effectiveness is asymmetric across models**, with time-to-jailbreak
distributions differing by an order of magnitude. Independent 2026 confirmation of the R2D2
lesson: which adaptation works is model-specific.

### Fang et al. (2026), Jailbreak Foundry — `papers/2602.24009_*.pdf`
Multi-agent system converting jailbreak papers into runnable modules in a unified harness. 30
attacks reproduced with mean ASR deviation of **+0.26 pp** from reported values; standardized
AdvBench evaluation of all 30 across 10 victim models with one GPT-4o judge. Relevant as a
reproducibility control and as a source of a large uniform attack × model matrix.

---

## 3. Cluster B — Hidden robustness and boundary conditions of alignment interventions

This cluster supplies the *phenomena* whose boundary conditions an adaptive protocol should
localize, and the evidence that static protocols systematically produce false nulls.

| Paper | Finding | Why it matters here |
|---|---|---|
| Wei et al. (2023), *Jailbroken* — `2307.02483` | Two failure modes: **competing objectives** and **mismatched generalization** | A taxonomy of *where the boundary is* — a principled axis set for an adaptive design to search over |
| Qi et al. (2024), *Shallow Safety Alignment* — `2406.05946` | Alignment adapts the distribution mostly over the **first few output tokens**; this one fact explains suffix, prefilling, decoding-parameter, and fine-tuning attacks | A concrete, testable boundary condition: robustness is a function of token depth |
| Hubinger et al. (2024), *Sleeper Agents* — `2401.05566` | Backdoors persist through SFT, RL, and adversarial training; **most persistent in the largest models**; adversarial training taught models to *better hide* the trigger | The canonical false null. Standard safety training reports success while the behavior is intact |
| Qi et al. (2023), *Fine-tuning Compromises Safety* — `2310.03693` | 10 examples, < $0.20, breaks GPT-3.5 Turbo's guardrails; even benign fine-tuning degrades safety | Interventions have a sharp dose–response boundary that fixed evaluations miss |
| Łucki et al. (2024), *Adversarial Perspective on Unlearning* — `2409.18025` | Jailbreaks "previously reported as ineffective against unlearning" succeed **when applied carefully**; finetuning on 10 unrelated examples recovers RMU-unlearned capabilities | Explicit demonstration that published nulls were artifacts of insufficiently adaptive attacks |
| Che et al. (2025), *Model Tampering Attacks* — `2502.05209` | Input-output evals only **lower-bound** worst-case behavior; resilience lies on a **low-dimensional robustness subspace**; tampering success **predicts** held-out input-space attack success; SOTA unlearning undone in **16 fine-tuning steps** | Gives a principled upper bound to compare adaptive input-space protocols against, and evidence that the vulnerability space is low-dimensional — which is what makes adaptive search efficient |
| Arditi et al. (2024), *Refusal Is Mediated by a Single Direction* — `2406.11717` | One 1-D subspace mediates refusal across 13 chat models | If the boundary is genuinely 1-D, adaptive search should find it fast |
| Wollschläger et al. (2025), *Geometry of Refusal* — `2502.17420` | **Contradicts** the above: multiple independent directions and multi-dimensional **concept cones**; orthogonality ≠ independence under intervention | An open dispute about the dimensionality of the boundary — and dimensionality determines whether adaptive design pays off |
| Sheshadri et al. (2024), *Latent Adversarial Training* — `2407.15549` | Targeted LAT improves robustness to persistent behaviors; **solves the sleeper-agent problem** where standard safety training failed | A rare case of an intervention with *genuine* robustness — the positive control the hypothesis's second half needs |
| Sharma et al. (2025), *Constitutional Classifiers* — `2501.18837` | **3,000+ hours** of human red teaming found no universal jailbreak | The most expensive null result in the literature. What would it have taken to reach the same confidence adaptively? |
| Betley et al. (2025), *Emergent Misalignment* — `2502.17424` | Narrow fine-tuning (insecure code) ⇒ broad misalignment; **all fine-tuned models behave inconsistently, sometimes acting aligned** | Inconsistency means single-sample evaluation has low power by construction |
| Lynch (2026), *Persistent Vulnerability of Aligned AI Systems* — `2604.00324` | Thesis integrating ACDC, LAT, BoN, and agentic misalignment; agentic misbehavior rises **6.5% → 55.1%** with a stated condition change | Shows how large a boundary-condition effect can be — and how easily a static design that omits the condition reports a null |
| Inference-Time Vulnerability Beyond Shallow Safety (2026) — `2606.04778` | Shallow safety is a special case: short token injections at **any** generation step alter safety behavior; **refusal-direction alignment does not predict robustness to injection** | Extends the boundary from "first few tokens" to the whole trajectory, and warns that internal-state proxies are not valid substitutes for behavioral tests |

---

## 4. Cluster C — Statistical and adaptive evaluation methodology

### ★ Miller (2024), *Adding Error Bars to Evals* — `papers/2411.00640_*.pdf` (Anthropic)
The reference text for treating evals as experiments. Five recommendations: CLT standard errors;
**clustered** standard errors when questions come in groups; variance reduction by resampling
answers and using next-token probabilities; **paired** question-level inference when comparing two
models; and **power analysis**.

- Clustered SEs are up to **3.05×** larger than naive ones on real Anthropic model data (DROP).
  Ignoring clustering makes an eval look far more precise than it is.
- Sample-size formula: `n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²`, invertible to give
  the **Minimum Detectable Effect**. Under reasonable assumptions, detecting a 3 pp difference at
  80% power needs **≈969 questions** — so "new evals should contain at least 1,000 questions".
- Increasing per-question resamples K from 1 to 10 cuts the MDE from 13.2% to 7.5% at n = 198.
  **This is the cheapest available lever for turning an ambiguous null into a decisive one, and it
  is exactly the kind of redesign an adaptive protocol should trigger.**

### Bowyer, Aitchison & Ivanova (2025), *Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints* — `papers/2503.01747_*.pdf`
CLT intervals **dramatically underestimate uncertainty** on small specialized benchmarks. Gives
frequentist and Bayesian alternatives; code at `code/bayes_evals/`. Since adaptive protocols
concentrate budget on small subsets, this failure mode is *more* acute for us, not less.

### Ramdas et al. (2022), *Safe Anytime-Valid Inference* — `papers/2210.01948_*.pdf`; Lindon et al. (2022), *Anytime-Valid Linear Models* — `papers/2210.08589_*.pdf`
e-processes and confidence sequences valid at **all** stopping times, accommodating continuous
monitoring and optional stopping. This is the correct inferential frame for a protocol that stops
when it has seen enough — which every adaptive design does. Without it, adaptive protocols inflate
Type I error and their "findings" are not trustworthy.

### Dorner, Nastl & Hardt (2024), *Limits to Scalable Evaluation at the Frontier* — `papers/2410.13341_*.pdf`
When the judge is no more accurate than the evaluated model, **no** debiasing method can reduce the
required ground-truth labels by more than half; empirical savings are smaller still. A hard ceiling
on how far LLM-as-judge can carry an adaptive protocol, and a reason to keep the rule-based judge
as a second outcome measure.

### Polo et al. (2024), *tinyBenchmarks* — `papers/2402.14992_*.pdf`
100 curated items suffice to estimate MMLU performance. IRT-based item selection; code at
`code/tinybenchmarks/`. The proof that adaptive item selection can cut evaluation cost ~100×.

### Krumdick et al. (2026), *Cost-Efficient Estimation of General Abilities* — `papers/2604.01418_*.pdf`
Releases **WILD** (65 models × 109,564 items × 163 tasks). Multidimensional IRT + **adaptive item
selection driven by optimal experimental design** predicts held-out performance on 112 tasks to
< 7% MAE after **16 items**. Adds cost-aware discount factors. The most direct demonstration in the
LLM literature that OED-driven adaptive selection beats static sampling — on capability, which is
why it was pruned as a standalone direction but retained as machinery.

### Wu, Nair & Candès (2026), *Efficient Evaluation of LLM Performance with Statistical Guarantees* — `papers/2601.20251_*.pdf`
FAQ: Bayesian factor model + hybrid variance-reduction/active-learning sampling + **Proactive
Active Inference** to preserve frequentist coverage under adaptive question selection. Up to **5×**
effective-sample-size gains at matched CI width. This is the template for "adaptive selection
*without* breaking validity" that D3 needs.

### Tolochinsky, Tenzer & Romano (2026), *Valid Best-Model Identification* — `papers/2605.10405_*.pdf`
MAB + low-rank factorization with **doubly robust** estimators giving valid finite-sample CIs under
adaptive selection and sampling without replacement. The statistically careful version of the
bandit in D1.

### Foster et al. (2021), *Deep Adaptive Design* — `papers/2103.02438_*.pdf`; Rainforth et al. (2023), *Modern Bayesian Experimental Design* — `papers/2302.14545_*.pdf`
The BOED theory. DAD amortizes sequential design into a network mapping history → next design in
one forward pass, trained with contrastive information bounds. Rainforth et al. is the review. These
supply the formal vocabulary — expected information gain, sequential design policy — that the
hypothesis is informally reaching for.

### Others on disk
- **ConSol** (`2503.17587`): SPRT to stop self-consistency sampling early — a worked example of a
  sequential stopping rule calibrated for LLM sampling.
- **CAT for medical LLM benchmarking** (`2603.23506`): IRT-based computerized adaptive testing,
  38 LLMs, terminating on a reliability threshold.

---

## 5. Synthesis

### Common methodologies
| Method | Papers |
|---|---|
| Iterative refinement using target feedback | PAIR, TAP, GOAT, LLM-assisted attacks |
| Zeroth-order / random search on a continuous surrogate | Andriushchenko et al., GCG (gradient), DSN |
| Quality-diversity / archive-based search | Rainbow Teaming, RainbowPlus, QDRT, Curiosity-driven RT |
| Memoryless i.i.d. resampling | Best-of-N (the static control) |
| Warm-starting from prior successes | Self-transfer (Andriushchenko), transfer attacks (Zou) |
| Extreme-value extrapolation from nulls | Jones et al. (Gumbel-tail), Hughes et al. (power law) |
| IRT / adaptive item selection | tinyBenchmarks, WILD+OED, CAT, FAQ |
| Anytime-valid sequential inference | Ramdas et al., Lindon et al., ConSol |

### Standard baselines
- **Attacks:** GCG, PAIR, TAP, AutoDAN, BoN, prompt+random-search, human/manual (JBC), DSN.
  HarmBench provides the canonical 18-attack × 33-model comparison.
- **Defenses/interventions:** RLHF/refusal training, adversarial training (R2D2), circuit
  breakers, latent adversarial training, unlearning (RMU), constitutional classifiers.
- **Static evaluation protocol:** fixed behavior set × fixed attack × fixed budget per behavior,
  single judged sample. This is the "static design" the hypothesis competes against.

### Evaluation metrics
| Metric | Use | Caveat |
|---|---|---|
| Attack success rate (ASR) | Standard headline | Binary, budget-dependent, judge-dependent; collapses the sequential structure (Topol 2026) |
| Queries-to-jailbreak | Efficiency; the right outcome for D1/D2 | Censored at the budget cap — must be handled as survival data, not thrown away |
| Elicitation probability & tail quantiles Q_p(n) | Converts nulls into signal | Needs repeated sampling or a logprob proxy |
| Worst-query / behavior-frequency / aggregate risk | Deployment-scale risk (Jones et al.) | Forecasts are set-sensitive |
| ASR(N) power-law coefficients (a, b) | Compares return curves across arms | Fit is unstable at small N; underestimates for high-ASR models |
| MDE / power | Whether a null is informative at all | Requires variance estimates from prior data |
| CI coverage under optional stopping | Whether the protocol's nulls are trustworthy | Requires anytime-valid methods |

### Datasets used in the literature
| Dataset | Used by | On disk |
|---|---|---|
| AdvBench (520 behaviors) | GCG and nearly everything after | ✅ |
| AdvBench-50 (PAIR curation) | PAIR, TAP, Andriushchenko et al. | ✅ |
| HarmBench (400 / 159 standard) | HarmBench, BoN, MHJ, Topol | ✅ |
| JBB-Behaviors (100 harmful + 100 benign) | JailbreakBench | ✅ |
| StrongREJECT (313) | Souly et al.; cited by BoN on judge reliability | ✅ |
| JailbreakBench attack artifacts | JailbreakBench leaderboard | ✅ (5 attacks × 4 models × 100) |
| MHJ (2,912 multi-turn prompts) | Li et al. 2024 | ✗ — external, noted in `datasets/README.md` |
| WILD (65 models × 109,564 items) | Krumdick et al. 2026 | ✗ — external |
| Anthropic red-team attacks (38,961) | Ganguli et al. 2022 | ✗ — external (HF `Anthropic/hh-rlhf` red-team subset) |

---

## 6. What we verified ourselves this phase **[computed here]**

Parsing the 24 attack logs in `code/llm-adaptive-attacks/attack_logs/` produced 437k per-iteration
rows and 672 per-run outcomes (`datasets/rs_trajectories/`). Two results are worth stating up
front because they are the empirical spine of the whole project.

**(a) A matched adaptive-vs-static counterfactual on Llama-2-Chat-7B.** Same attack family, same
iteration cap, differing only in initialization policy:

| Config | ASR (judge 10/10) | ASR (rule-based) | median iters | mean iters | % runs hitting the 10,000 cap |
|---|---|---|---|---|---|
| `llama2-7b_plain_init` (static init) | **0.00** (n=50) | 0.00 | 10,000 | 6,735 | 56% |
| `llama2_7b` (self-transfer init) | **0.96** (n=51) | 0.88 | **2** | 14.2 | 0% |

A single adaptive re-initialization step flips a total null into near-total success at roughly
three orders of magnitude lower query cost.

**(b) The negative control is present in the same corpus.** Per-run outcomes recovered for R2D2,
the ICL-template variants, and 11 other configurations, so a protocol can be scored on cases where
the same adaptation *fails* (R2D2 + self-transfer = 12% in the paper) as well as where it succeeds.

**Honest caveats on (a).** `plain_init` is a harsher ablation than the paper's headline "Prompt +
RS = 50%" row; it is random search from an unoptimized suffix initialization, so the comparison
brackets the effect rather than reproducing Table 2 exactly. Run boundaries were recovered
heuristically (a decrease in the iteration counter marks a new run), which conflates restarts with
new requests in configurations that use multiple restarts — counts should be treated as ±1 run.
Goal-string extraction failed for three configurations whose templates phrase the request
differently (`r2d2`, `*_icl_one_shot`); those trajectories are usable but not joinable to behavior
metadata. All of this is recorded in `datasets/README.md`.

---

## 7. Gaps and opportunities

1. **Adaptivity is never the independent variable.** Every attack paper compares *attacks*.
   None holds the attack fixed and varies the *protocol* that schedules it under a fixed budget.
   This is the gap the hypothesis occupies, and it is genuinely open.
2. **Nulls are reported without power.** Of the alignment-intervention papers above, only Jones
   et al. and Miller give any machinery for saying how strong a null is. Constitutional
   Classifiers spent 3,000+ red-team hours to earn its null; nobody can say what the minimum
   sufficient design would have been.
3. **"Adaptive" conflates two different moves.** Parameter adjustment within a design family
   (self-transfer) and switching design family (in-context prompt on R2D2) succeed under different
   conditions. No paper measures when to do which. This is the sharpest contribution available.
4. **Adaptive stopping breaks fixed-sample inference, and the red-teaming literature has not
   noticed.** The eval-methodology literature (SAVI, FAQ, doubly robust MAB) has the fix; the
   red-teaming literature has the problem. Joining them is straightforward and currently undone.
5. **Disputed dimensionality of the boundary.** Arditi et al. (1-D) vs. Wollschläger et al.
   (concept cones) vs. Che et al. (low-dimensional robustness subspace). Whether adaptive design
   pays off depends on this, so it should be treated as a moderator, not settled.
6. **Judge noise is the binding constraint.** ~20% false positives on one model (Andriushchenko),
   manual re-grading of every flagged jailbreak (BoN), and a hard theoretical ceiling on
   judge-based label savings (Dorner et al.). Any efficiency claim measured only through an LLM
   judge is suspect.

---

## 8. Recommendations for our experiment

**Recommended primary data (all offline, zero API cost):**
1. `datasets/rs_trajectories/` — real per-iteration search traces with a continuous objective and
   a matched adaptive/static pair. Primary substrate for D2 and D3.
2. `datasets/jbb_artifacts/jbb_queries_to_jailbreak.csv` — complete cost-to-discovery matrix
   (5 attacks × 4 models × 100 behaviors, with category labels). Primary substrate for D1.
3. `datasets/behaviors/` — the behavior sets, for stratification and for any live arm.

**Recommended baselines:** static uniform allocation; round-robin; BoN-style memoryless
resampling (with the published power law as the analytic form); continue-only adaptive; oracle
allocation as an upper bound.

**Recommended metrics:** vulnerabilities found at matched budget (primary); queries-to-discovery
treated as **censored** survival data; calibration of the terminal null (forecast vs. realized
worst-query risk); CI coverage under optional stopping; and both judge-based *and* rule-based ASR
reported side by side.

**Methodological requirements, in priority order:**
1. **Equalize compute** in every adaptive-vs-static comparison, or the finding is vacuous.
2. **Use anytime-valid inference** for any protocol with a data-dependent stopping rule.
3. **Handle censoring.** Runs hitting the iteration cap are right-censored, not failures at the
   cap value; 56% of `plain_init` runs are censored. Naive means will be badly biased.
4. **Report the negative control.** Include R2D2-like cases where the adaptation should fail.
5. **Bootstrap for uncertainty**, following Hughes et al.: resample trajectories to generate
   independent runs rather than treating the single observed trace as the estimate.
6. **Do not report power-law fits below N ≈ 10**, and expect under-estimation for high-ASR arms.

**Ethical scope.** All harmful-behavior data here is standard published safety-benchmark material
used for defensive evaluation-methodology research. The project studies *how to design robustness
experiments*; it does not develop new attack capability, and no new jailbreak artifacts should be
produced or released by the experiment phase.
