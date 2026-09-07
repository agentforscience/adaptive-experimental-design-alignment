# Planning: Direction Budget and Ranking

**Research hypothesis.** Adaptive experimental protocols that use null results as iterative
triggers for experimental redesign will increase the likelihood of identifying subtle
vulnerabilities or genuine robustness in LLMs compared to static experimental designs, and will
more efficiently localize the boundary conditions under which alignment interventions succeed or
fail.

**Phase:** `resource_finder` (Phase 1). This file records the enumeration, scoring, and pruning of
candidate research directions, per the DIRECTION BUDGET contract. **Three** directions are kept
for implementation; the rest are pruned with reasons.

---

## Scoring rubric

Each direction scored 1–5 on four axes, summed (max 20):

| Axis | Meaning |
|---|---|
| **EV** — Evidence in literature | Is there published work establishing the mechanism and giving us calibration targets? |
| **REL** — Relevance to hypothesis | Does it test *adaptivity of the experimental design*, rather than the quality of a particular attack? |
| **IG** — Expected information gain | Will the result change what a practitioner does, and can it separate "adaptive helps" from "adaptive is just more compute"? |
| **FEAS** — Implementation feasibility | Can it run in this workspace (CPU-only, no model weights, no paid API keys assumed)? |

---

## Directions kept (top 3)

### D1 — Adaptive budget allocation over an attack × behavior portfolio (score 20)

**Claim under test.** Given a fixed total query budget, a protocol that reallocates budget away
from (attack, behavior) arms that keep returning nulls — and toward arms whose partial evidence is
promising — discovers strictly more vulnerabilities than a static uniform grid allocation.

**Why it is a real test.** Total compute is *held constant*; only the allocation policy varies.
This isolates adaptivity from "more compute", which is the main confound the hypothesis faces.

**Data already on disk (no API cost).**
- `datasets/jbb_artifacts/jbb_queries_to_jailbreak.csv` — 1,800 rows: per-behavior
  `queries_to_jailbreak` and `jailbroken` for **5 attacks × 4 target models × 100 behaviors**,
  with 10 harm-category labels. This is a complete cost-to-discovery matrix; any allocation
  policy can be replayed offline against it.
- `datasets/rs_trajectories/*.csv.gz` — 14 configurations of per-iteration random-search traces
  (≈437k iteration rows) giving the *within-arm* reward curve, so a bandit can be simulated at
  iteration granularity rather than only at the arm level.

**Baselines.** Uniform/static allocation (the current field default, e.g. HarmBench's fixed
budget-per-behavior protocol); round-robin; oracle allocation (upper bound). Adaptive:
successive halving / LUCB / Thompson sampling over arms; Jones et al.'s forecast-driven allocation.

**Calibration targets from the literature.** JailbreakBench artifact ASRs already on disk range
from 0.00 (PAIR vs. Llama-2-7B) to 0.95 (DSN vs. Vicuna); Jones et al. (2502.16797 §6) show
forecast-driven allocation picks the compute-optimal red-teamer.

**Risk.** The artifact matrix is one realization per cell, so re-randomization must come from
bootstrapping the trajectories, not from re-running attacks. Documented in
`datasets/README.md`.

---

### D2 — Null-as-signal: escalation ladder over the redesign space (score 20)

**Claim under test.** A null is only actionable if it carries a continuous score. A protocol that
(i) reads a continuous elicitation signal out of a failed attempt and (ii) uses it to choose
*which kind* of redesign to make — continue the same search, re-initialize from a related success
(self-transfer), or switch template family — localizes boundary conditions faster than either a
static design or a naive "just run longer" adaptive design.

**Why it is a real test.** It distinguishes two very different things that both get called
"adaptive": *parameter adjustment within a fixed design family* versus *switching design family*.
The literature contains a clean natural experiment on exactly this distinction (below).

**Data already on disk, with a verified matched counterfactual.** Parsing
`code/llm-adaptive-attacks/attack_logs/` yielded a matched pair on Llama-2-Chat-7B where only the
initialization policy differs:

| Config | GPT-4-judge 10/10 ASR | median iters to stop | mean iters | runs hitting the 10,000 cap |
|---|---|---|---|---|
| `llama2-7b_plain_init` (static init) | **0.00** (n=50) | 10,000 | 6,735 | 56% |
| `llama2_7b` (self-transfer init) | **0.96** (n=51) | **2** | 14.2 | 0% |

*(computed in this phase from `datasets/rs_trajectories/`; see `datasets/README.md` for the
derivation and caveats)*

One adaptive re-initialization step moves the measured outcome from a total null to near-total
success at ~3 orders of magnitude lower query cost. This is the single strongest offline anchor
for the hypothesis available without spending a dollar of API budget.

**And the crucial negative control, from the same source.** On R2D2 (adversarially trained),
Andriushchenko et al. report Prompt + RS + **self-transfer = 12%**, while switching *template
family* to an in-context prompt gives **90%**, and +RS gives 100%. So more of the same adaptation
is worthless there; only a change of design family works. Any protocol we build must be scored on
both cases, or it will look good for the wrong reason.

**Baselines.** Static single-shot; static-with-restarts; continue-only adaptive (no family
switch); full escalation ladder. Signal for the ladder: target-token logprob (available in the
traces) and Jones et al.'s elicitation probability.

---

### D3 — Calibrated nulls: separating genuine robustness from an underpowered experiment (score 19)

**Claim under test.** The hypothesis has two halves; the second ("or genuine robustness") is the
one the field currently gets wrong. A protocol that stops adaptively must emit a *calibrated upper
bound* on residual vulnerability, not a bare "no jailbreak found". We test whether
anytime-valid sequential stopping plus tail extrapolation gives correct coverage under adaptive
stopping, where naive CLT error bars do not.

**Why it is a real test.** Adaptive stopping invalidates fixed-sample inference. If an adaptive
protocol reports more findings but its nulls are uncalibrated, the hypothesis is only half
supported — and that is a publishable, decision-relevant distinction.

**Method stack, all with papers and code on disk.**
- Safe anytime-valid inference / e-processes and confidence sequences (Ramdas et al., 2210.01948;
  Lindon et al., 2210.08589) for stopping without inflating Type I error.
- Gumbel-tail extrapolation of extreme elicitation quantiles (Jones et al., 2502.16797) to convert
  "we saw no success in m queries" into a forecast of worst-query risk at n ≫ m.
- Power analysis / MDE formula (Miller, 2411.00640 Eq. 9–10) and small-sample Bayesian intervals
  (`code/bayes_evals`, Bowyer et al. 2503.01747) for the fixed-sample comparison.

**Ground truth for coverage checking.** The trajectories contain the eventual outcome, so a
protocol can be truncated at iteration *t*, made to emit a bound, and scored against what actually
happened later in the same trace. That is a genuine calibration test, not a self-consistency check.

**Feasibility docked one point:** the Gumbel-tail method needs repeated sampling per query to
estimate elicitation probabilities. The on-disk traces provide a *proxy* (target-token logprob),
which is what Jones et al. themselves use for most experiments, but the mapping from logprob to
elicitation probability is an approximation that must be stated, not hidden.

---

## Directions enumerated and pruned

| # | Direction | Score | Reason for pruning |
|---|---|---|---|
| R1 | Train an RL / curiosity-driven / GFlowNet red-team attacker policy and compare to static prompt sets | 11 | Needs GPU fine-tuning of an attacker LM (Perez 2202.03286, Hong 2402.19464). Worse, it confounds *policy quality* with *adaptivity* — a better attacker winning tells us nothing about experimental design. |
| R2 | Mechanistic localization of boundary conditions via refusal directions / concept cones | 10 | Needs white-box weights + GPUs (Arditi 2406.11717, Wollschläger 2502.17420). Measures representation geometry, a different construct from design efficiency. Retained as *interpretation* of results, not as an experiment. |
| R3 | Multimodal (vision / audio) adaptive red teaming | 9 | BoN (2412.03556) shows modality changes ASR a lot, but running it needs VLM/ALM API access and 61 MB of audio assets. Adds cost without testing the design-methodology claim. |
| R4 | Model-tampering / latent-space attacks as the elicitation channel | 10 | Che et al. (2502.05209) show tampering attacks upper-bound input-space attacks — genuinely important, but requires weights + GPU. Kept as a cited argument that input-space nulls are weak evidence. |
| R5 | Fine-tuning dose–response: how many adversarial examples to break alignment | 9 | Qi et al. (2310.03693) is the reference result; reproducing needs training compute and produces a harm-uplift artifact. Out of scope for this workspace. |
| R6 | Human multi-turn red teaming as the adaptive condition | 7 | MHJ (2408.15221) is the strongest evidence that human adaptivity beats automated single-turn (>70% vs. single-digit ASR), but there are no human subjects here. Cited as motivation; the MHJ dataset is noted as an *external* resource, not used as an experimental arm. |
| R7 | Adaptive item selection (IRT / CAT / active querying) for capability benchmarks | 14 | Strong, feasible methods (tinyBenchmarks 2402.14992; FAQ 2601.20251; WILD/OED 2604.01418; CAT 2603.23506). Pruned **as a standalone direction** because its outcome variable is *capability estimation error*, not vulnerability discovery. Its machinery is imported into D1 (item selection) and D3 (valid CIs under adaptive selection). |
| R8 | Live adaptive attack harness against frontier APIs | 12 | Duplicates artifacts we already have offline, costs real money, and adds nondeterminism that Andriushchenko et al. (§4.3, Fig. 3) show corrupts the search signal. Deferred; the experiment runner may add a small live arm if keys are provided. |

**Pruning discipline.** Per the contract, the search space is not to be re-expanded later unless
new evidence invalidates this ranking. If that happens, the ranking must be updated here and the
change explained in `STATE.md`.

---

## Cross-cutting design requirements (apply to all three kept directions)

1. **Hold compute constant.** Every adaptive-vs-static comparison must equalize total queries, or
   the result reduces to "more compute helps", which BoN already established.
2. **Report both halves.** A protocol that finds more vulnerabilities but emits uncalibrated nulls
   only supports half the hypothesis. Score discovery *and* calibration.
3. **Include an adaptation that should fail.** R2D2 with self-transfer (12%) is the built-in
   negative control. A protocol that "wins" on every arm is almost certainly leaking information.
4. **Judge noise is not negligible.** Andriushchenko et al. observed ~20% GPT-4-judge false
   positives on Claude 2.1; Hughes et al. manually re-graded every flagged jailbreak; Dorner et al.
   (2410.13341) bound how much an LLM judge can substitute for labels. Use the rule-based and
   judge-score columns as *two* outcome measures and report both.
5. **Sequential ≠ fixed-sample inference.** Any stopping rule needs anytime-valid treatment (D3),
   including in D1 and D2.

---
---

# Phase 2 (`experiment_runner`) — Motivation, Novelty, and Experimental Design

*Added 2026-08-05 at the start of the experiment phase. The direction budget above
(3 kept of 11) is **not** re-expanded; everything below instantiates D1, D2, D3, plus one
live-model arm that the resource phase deferred (R8) because it assumed no compute. Four
free RTX A6000 GPUs were found at the start of this phase, which changes that assumption —
the change and its justification are recorded in STATE.md.*

## Motivation & Novelty Assessment

### Why This Research Matters

Safety evaluations report nulls constantly: "we could not elicit the behaviour." That
sentence is load-bearing in deployment decisions, model cards, and policy, and it is
systematically ambiguous — it can mean the model is robust, or it can mean the experiment
was badly designed. The literature already contains proof that the second reading is often
correct: PAIR scores **0.00** ASR against Llama-2-7B while prompt+random-search scores
**0.90** on the *same 100 behaviours of the same model* (`datasets/jbb_artifacts/`). A lab
that ran only PAIR would have published a false claim of robustness. The practical question
is therefore not "is adaptivity good" but "**given a fixed evaluation budget, does a protocol
that treats its own nulls as feedback find more, and does it also say something calibrated
when it still finds nothing?**" Anyone who has to act on an eval result — safety teams,
auditors, regulators — depends on the answer.

### Gap in Existing Work

From `literature_review.md`: adaptivity has been argued for **attacks** (Andriushchenko et
al. 2024) and for **forecasting** rare behaviours (Jones et al. 2025), and evaluation
statistics has been improved on the fixed-sample side (Miller 2024; Bowyer et al. 2025).
What is missing is a controlled comparison in which *the experimental protocol itself* is the
independent variable and **total compute is held constant**. Every existing adaptivity result
is confounded with spending more queries. Nobody has separated the two moves that both get
called "adaptive": **parameter adjustment inside a design family** versus **switching design
family**. And nobody scores the *null* branch — what an adaptive protocol should output when
it still finds nothing.

### Our Novel Contribution

1. A **compute-matched** head-to-head of static vs. adaptive protocols, replayed against real
   released attack traces and a real cost-to-discovery matrix, so the compute confound is
   removed by construction rather than by argument.
2. Decomposition of "adaptive" into a **three-rung escalation ladder** (continue → re-initialise
   → switch design family) and measurement of which rung carries the effect.
3. A direct test of the mechanism the whole hypothesis rests on: **is a null actually
   informative?** We ask whether the continuous signal available at the moment of a null
   predicts whether persistence would have paid off (AUC). If it does not, no adaptive
   protocol can work in principle, however clever.
4. The **calibration half**: when an adaptive protocol stops and reports a null, we score the
   coverage of the bound it emits, under adaptive stopping, against the ground truth contained
   in the remainder of the same trace.
5. A **live end-to-end replication on real open-weights LLMs** of the static-vs-adaptive
   contrast, on a non-harmful alignment failure mode (sycophantic answer-flipping under
   social pressure), so the finding is not an artifact of replaying somebody else's logs.

### Experiment Justification

| # | Experiment | Why it is necessary |
|---|---|---|
| **E1** | Escalation-ladder replay on 437k real random-search iterations (D2) | The only way to compare protocols at *matched iteration budget* on real traces. Isolates which rung of adaptivity matters. Without it we cannot separate adaptivity from compute. |
| **E1c** | Null-diagnosticity (AUC of the stall signal) | Tests the *mechanism*. A protocol can only use nulls as triggers if nulls carry signal. This can refute the hypothesis at its root, so it must be run. |
| **E2** | Compute-matched portfolio allocation on the JBB 1,800-cell matrix (D1) | Tests the second, independent form of adaptivity: reallocating budget across *methods and behaviours* rather than within a search. Different substrate, different failure mode — needed for generality. |
| **E3** | Calibration of the terminal null under adaptive stopping (D3) | The hypothesis has two halves. E1/E2 test discovery only. If adaptive protocols find more but emit invalid nulls, the hypothesis is half-refuted, and that is decision-relevant. |
| **E4** | Live protocol race on real LLMs (Qwen2.5-7B, Qwen2.5-1.5B, Phi-3.5-mini) | Replays are retrospective and inherit the original authors' design choices. E4 runs the actual adaptive loop against real model weights, on a fresh failure mode, with a real alignment intervention (an anti-sycophancy system prompt) whose robustness is what is being probed. Also supplies a **genuine-robustness** case, which the offline data cannot. |

## Experimental Design

### Independent variable
Protocol class: `STATIC` | `RESTART-ONLY` | `CONTINUE-ONLY` | `LADDER (adaptive)` | `ORACLE`.

### Held constant (the non-negotiable)
**Total query/iteration budget `B` is identical across all arms in every comparison.** Every
figure sweeps `B` and reports the whole detection-vs-budget curve, not a single point.

### Dependent variables
1. Detection rate at matched budget (primary).
2. Queries/iterations to first detection, treated as **right-censored survival data**.
3. AUC of the null-trigger signal for predicting eventual success.
4. Coverage and width of the upper bound emitted on a terminal null.
5. (E4) Flip rate under pressure, and detection of a non-zero flip rate at matched budget.

### Statistical analysis plan (pre-registered here, before running)
- Uncertainty by **bootstrap over behaviours** (B = 1,000 resamples, seed 42), never from
  re-running attacks — the artifact matrix has one realisation per cell.
- Paired comparisons where a per-behaviour join exists (`llama2-7b_plain_init` ↔ `llama2_7b`
  join on `goal`): **paired bootstrap + McNemar's exact test** on discordant behaviours.
- Unpaired comparisons (family switch, where goal strings are unrecoverable — see
  `datasets/README.md` caveat 2): resampled pairing, **explicitly flagged as an exchangeability
  assumption**, not as a paired result.
- Survival: Kaplan–Meier with the 10,000-iteration cap as the censoring time; log-rank test.
- α = 0.05; Holm correction within each experiment's family of tests.
- Effect sizes reported alongside every p-value (risk difference, Cohen's h, hazard ratio).

### Expected outcomes and what would refute the hypothesis
- **Supports:** LADDER ≥ STATIC at every matched `B`, with the gap driven by the
  family-switch rung; AUC of the stall signal meaningfully > 0.5.
- **Refutes:** LADDER ≈ STATIC at matched `B` (i.e. published adaptivity gains are just extra
  compute); or AUC ≈ 0.5 (nulls carry no actionable signal); or adaptive stopping destroys
  coverage so badly that adaptive nulls are uninterpretable.
- We expect a **mixed** result and will report it as such: the negative control
  (R2D2 + self-transfer = 12% in Andriushchenko et al.) predicts that the *continue* and
  *re-initialise* rungs fail where only a family switch works.

### Deviation from the resource phase's assumptions
The resource phase assumed **no GPU and no API key**. There is no API key (`OPENROUTER_KEY`
is unset, confirmed at the start of this phase), so the OpenRouter route in the brief is
unavailable. However **4× RTX A6000 are free**, so the "use real models, not simulations"
requirement is satisfied by running real open-weights instruct models locally instead. This
un-prunes R8 in a cheaper, offline-safe form (E4). No simulated LLM is used anywhere.

### Ethical scope
E1–E3 replay **already-published** benchmark artifacts and generate no new attack content.
E4 uses a **non-harmful** failure mode (sycophantic answer-flipping on grade-school maths) and
produces no jailbreak artifacts. Consistent with the scope set in `datasets/README.md`.

## Timeline (6 h budget)
| Time | Milestone |
|---|---|
| 0:00–0:35 | Resource review, plan, env, model download started |
| 0:35–1:40 | Implement E1/E2/E3 harness (`src/`) |
| 1:40–2:20 | Run E1/E2/E3 |
| 2:20–4:00 | Implement + run E4 live on real models |
| 4:00–4:45 | Analysis, figures, statistics |
| 4:45–5:30 | REPORT.md, README.md, STATE.md, validation |
