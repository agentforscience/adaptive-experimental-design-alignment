# Cloned Repositories

10 repositories, cloned with `--depth 1`. Nothing here is git-ignored by `datasets/.gitignore`,
but the clones themselves are large (~780 MB total, dominated by HarmBench and bon-jailbreaking
assets) and are reproducible from the commands below.

**No repository listed in the research spec** — there was no `code_references` section — so all of
these were selected during this phase. Selection criterion: it either (a) supplies data we can
replay offline, (b) supplies a statistical method one of the three kept directions needs, or
(c) is the reference implementation of a baseline we must name correctly.

| Repo | Size | Commit | Role |
|---|---|---|---|
| [`llm-adaptive-attacks`](#1-llm-adaptive-attacks) | 36 MB | `7787faa` | **Primary data source.** Released per-iteration attack logs |
| [`jailbreakbench-artifacts`](#2-jailbreakbench-artifacts) | 2.0 MB | `909e68c` | **Primary data source.** Cost-to-discovery matrix |
| [`bayes_evals`](#3-bayes_evals) | 2.5 MB | `b971184` | **Tested & working.** Small-sample Bayesian eval intervals |
| [`tinybenchmarks`](#4-tinybenchmarks) | 65 MB | `e9a8b10` | **Tested & working.** Offline IRT parameters + `fit_theta` |
| [`harmbench`](#5-harmbench) | 228 MB | `8e1604d` | Behavior sets; canonical attack/defense harness |
| [`jailbreakbench`](#6-jailbreakbench) | 3.7 MB | `23dbdf6` | Artifact schema; judges |
| [`bon-jailbreaking`](#7-bon-jailbreaking) | 501 MB | `c118210` | Best-of-N reference implementation; 159 direct requests |
| [`strong_reject`](#8-strong_reject) | 487 KB | `7a551d5` | 12 judge/grader templates including the StrongREJECT rubric |
| [`pair`](#9-pair-and-10-tap) | 273 KB | `6379ef7` | PAIR reference implementation |
| [`tap`](#9-pair-and-10-tap) | 1.2 MB | `7bcdad3` | TAP reference implementation |

Reproduce all clones:

```bash
cd code
git clone --depth 1 https://github.com/tml-epfl/llm-adaptive-attacks.git llm-adaptive-attacks
git clone --depth 1 https://github.com/JailbreakBench/artifacts.git    jailbreakbench-artifacts
git clone --depth 1 https://github.com/sambowyer/bayes_evals.git       bayes_evals
git clone --depth 1 https://github.com/felipemaiapolo/tinyBenchmarks.git tinybenchmarks
git clone --depth 1 https://github.com/centerforaisafety/HarmBench.git harmbench
git clone --depth 1 https://github.com/JailbreakBench/jailbreakbench.git jailbreakbench
git clone --depth 1 https://github.com/jplhughes/bon-jailbreaking.git  bon-jailbreaking
git clone --depth 1 https://github.com/dsbowen/strong_reject.git       strong_reject
git clone --depth 1 https://github.com/patrickrchao/JailbreakingLLMs.git pair
git clone --depth 1 https://github.com/RICommunity/TAP.git             tap
```

---

## 1. `llm-adaptive-attacks`

Reference implementation for Andriushchenko, Croce & Flammarion (2024), *Jailbreaking Leading
Safety-Aligned LLMs with Simple Adaptive Attacks* (ICLR 2025). Apache-2.0.

**Why it matters most:** it ships **`attack_logs/` — 24 log files, 21 MB** of the authors' own
per-iteration random-search traces. These are already-paid-for experiments containing a matched
adaptive/static counterfactual. `scripts/build_datasets.py` parses them into
`datasets/rs_trajectories/`; see `datasets/README.md` §3 for the schema, the headline numbers,
and five caveats.

**Key files**

| Path | What it is |
|---|---|
| `attack_logs/exps_*.log` | Per-iteration traces: `it=N [best] logprob=… [curr] logprob=… len_adv=… n_change=…`, plus a per-run summary line with `max_prob`, `judge_llm_score=X/10->Y/10`, `jailbroken_judge_rule` |
| `jailbreak_artifacts/exps_*.json` | Final jailbreak strings in JailbreakBench format (15 configs) |
| `harmful_behaviors/harmful_behaviors_pair.csv` | The AdvBench-50 curation every log uses |
| `main.py` | Random-search driver; `--n-iterations`, `--n-restarts`, `--target-str`, `--prompt-template` |
| `main_claude_transfer.py`, `main_claude_prefilling.py` | The two Claude-specific adaptations |
| `prompts.py` | The rule-based prompt templates, including the in-context template that beats R2D2 |

**Log format note.** Run boundaries are not delimited; our parser infers them from decreases in
the iteration counter. Documented as caveat 1 in `datasets/README.md`.

**Running it live** would need GPU + `vllm`/`transformers` for open-weight targets and API keys
for the rest. Not attempted — the logs already contain what we need.

---

## 2. `jailbreakbench-artifacts`

The official JailbreakBench artifact repository. MIT.

29 JSON files under `attack-artifacts/<method>/<attack_type>/<model>.json`, covering
**GCG, PAIR, DSN, JBC, prompt_with_random_search × 4 target models × 100 behaviors**. Each entry
carries `queries_to_jailbreak`, `number_of_queries`, `jailbroken`, token counts, and the
prompt/response text.

Flattened to `datasets/jbb_artifacts/jbb_queries_to_jailbreak.csv` (1,800 rows) — the substrate
for direction D1. Prompt/response *text* is deliberately not copied into `datasets/`; only lengths.

---

## 3. `bayes_evals` — installed and verified working

Implements Bowyer, Aitchison & Ivanova (2025), *Don't Use the CLT in LLM Evals With Fewer Than a
Few Hundred Datapoints* (ICML 2025 spotlight). Needed by direction D3, where adaptive protocols
concentrate budget on small subsets and CLT intervals fail exactly there.

```bash
uv pip install -e code/bayes_evals
```

Verified smoke test (a deliberately null-heavy small eval, n=50 per arm):

```python
import bayes_evals as be, pandas as pd, numpy as np
rng = np.random.default_rng(0)
df = pd.DataFrame({"static": rng.binomial(1, 0.02, 50), "adaptive": rng.binomial(1, 0.20, 50)})
# successes: static 2/50, adaptive 15/50
be.independent_intervals(df, alpha=0.05)
#         static  adaptive
# lower   0.0123    0.1911
# upper   0.1346    0.4383
be.paired_comparisons(df)          # P(adaptive > static) = 0.9999
```

Note the `static` interval: `[0.012, 0.135]` from 2/50. A near-null still has a non-trivial upper
bound — precisely the quantity direction D3 needs to report instead of "no jailbreak found".

API: `independent_intervals`, `independent_comparisons`, `paired_comparisons`, `plot_intervals`,
`plot_comparisons`. Input is a binary DataFrame, Q rows (questions) × M columns (models/arms).

---

## 4. `tinybenchmarks` — verified working offline

Polo et al. (2024). Ships `tinyBenchmarks/tinyBenchmarks.pkl` (5.2 MB) with **fitted IRT item
parameters for `lb`, `mmlu`, `helm_lite`, `alpaca`** — no download or model access required.

```python
import sys; sys.path.insert(0, "code/tinybenchmarks")
import tinyBenchmarks as tb   # exposes: evaluate, fit_theta, item_curve, sigmoid
```

Relevant because directions D1 and D3 borrow its machinery (IRT ability estimation, item
information for selection) even though capability benchmarking was pruned as a standalone
direction (see `planning.md`, R7).

---

## 5. `harmbench`

Mazeika et al. (2024). MIT. The standardized red-teaming harness: 18 attacks × 33 models.

Used here for `data/behavior_datasets/harmbench_behaviors_text_{all,test,val}.csv`
(400/320/80 behaviors), copied into `datasets/behaviors/`.

**Not run.** `requirements.txt` pins `vllm>=0.3.0`, `spacy==3.7.2`, `transformers`, `fschat`,
`ray`, plus five commercial API SDKs — i.e. it needs GPUs and paid keys. Its evaluation classifier
(`cais/HarmBench-Llama-2-13b-cls`, referenced from `evaluate_completions.py` and
`docs/evaluation_pipeline.md`) is a 13B model download and was not fetched.

Also contains the **R2D2** adversarially trained model config — the negative-control target from
`planning.md` D2.

---

## 6. `jailbreakbench`

Chao et al. MIT. Provides the artifact schema (`src/jailbreakbench/artifact.py`:
`JailbreakInfo`, `AttackParameters`, `AttackArtifact`) and the judge implementations.

**Import caveat, discovered while testing:** `from jailbreakbench.artifact import read_artifact`
triggers `jailbreakbench/__init__.py`, which imports `classifier.py`, which requires `litellm`.
With `pydantic` installed but not `litellm` the import fails. Since we already have the artifacts
locally, `scripts/build_datasets.py` **reads the JSON directly** and does not depend on the
package. If the experiment phase wants the package API, install `litellm` first.

`read_artifact(method=..., model_name=...)` otherwise fetches from GitHub at runtime — offline
runs should use the local clone.

---

## 7. `bon-jailbreaking`

Hughes et al. (2024) reference implementation. The `bon/` package holds the augmentation
primitives (character scrambling, random capitalization, character noising for text; image and
audio augmentations for the other modalities) and the BoN driver loop.

**Used for:** `data/direct_request.jsonl` — the 159 HarmBench standard direct requests used in the
paper — copied to `datasets/behaviors/bon_direct_requests.jsonl`.

**Not used:** the repo does **not** ship the attack trajectories (those are behind a Google Drive
link in the README). 501 MB of the clone is `docs/assets` (website media) and `data/audio_files`
+ `background_sounds` (61 MB of audio for the ALM experiments). Running BoN live needs
`openai`, `anthropic`, `grayswan-api` keys.

The published power law `−log(ASR) = a·N^(−b)` is a sufficient analytic stand-in for the
memoryless baseline (see `literature_review.md` §2).

---

## 8. `strong_reject`

Souly et al. MIT. Its real value here is
`strong_reject/eval_files/judge_templates.json`, which contains **12 grader templates**:
`gpt4_judge`, `pair`, `pair_system`, `jailbroken_binary`, `category_binary`,
`strongreject_rubric`, `strongreject_rubric_system`, `strongreject_finetuned`,
`strongreject_finetuned_v2`, `strongreject_aisi`, `harmbench`, `accuracy_rubric`.

Because judge disagreement is a first-order confound in this project (Andriushchenko et al. saw
~20% GPT-4-judge false positives on Claude 2.1; Dorner et al. bound how far judges can substitute
for labels), having the exact prompt text of the standard graders matters for reporting.

Also ships `jailbreak_files/` with PAIR templates and PAP persuasion templates.
Running the graders needs an API key.

---

## 9. `pair` and 10. `tap`

Reference implementations of PAIR (Chao et al. 2023) and Tree-of-Attacks-with-Pruning
(Mehrotra et al. 2023). Both are small, readable, and structurally identical in layout
(`main.py`/`main_TAP.py`, `conversers.py`, `judges.py`/`evaluators.py`, `system_prompts.py`).

Cloned as **algorithmic references**, not to be run: both need attacker + target + judge model
access. What we want from them is the *control structure* — PAIR's refine-on-refusal loop and
TAP's prune-before-query rule — since direction D1's allocation policies are generalizations of
TAP's pruning and direction D2's escalation ladder generalizes PAIR's refinement.

---

## Environment notes for the experiment runner

Installed in `.venv` so far (via `uv add`, recorded in `pyproject.toml`):
`pypdf`, `httpx`, `requests`, `pandas`, `numpy`, `scipy`, `matplotlib`, `datasets`, `pydantic`,
plus `bayes_evals` as an editable install.

Not installed, and only needed if a live-model arm is added: `vllm`, `transformers`, `torch`,
`litellm`, `openai`, `anthropic`, `fschat`, `ray`.

**All three kept directions run CPU-only and offline against `datasets/`.** No GPU and no API key
is required to execute the planned experiments.
