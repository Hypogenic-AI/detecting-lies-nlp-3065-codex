# Planning: Mechanism-Separated Falsehood Analysis in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Current LLM lie-detection evaluations often score “false output” as a single category, even though falsehoods can come from very different mechanisms: not knowing vs knowingly misreporting under social or utility pressure. This matters for safety and governance because interventions differ: epistemic failures need calibration/retrieval/abstention, while incentive-driven misreporting needs objective and policy controls. A mechanism-separated estimate of falsehood sources enables more targeted detection and mitigation.

### Gap in Existing Work
The literature distinguishes hallucination/confabulation from strategic deception, but most detector evaluations do not use counterfactual controls that hold knowledge constant while varying incentives. This causes label conflation and makes it unclear whether a detector is tracking ignorance, compliance, or true deceptive intent.

### Our Novel Contribution
We run a counterfactual, mechanism-aware benchmark that separates (a) epistemic falsehoods from (b) incentive-induced falsehoods on paired prompts. We then test cross-mechanism detector transfer to quantify conflation risk in current lie-detection practice.

### Experiment Justification
- Experiment 1: Epistemic baseline on factual QA (no incentive pressure).
  - Why needed: Establishes base falsehood rate under neutral conditions, approximating knowledge/epistemic failure.
- Experiment 2: Paired incentive test on sycophancy prompts (control vs pressure).
  - Why needed: Isolates misreporting induced by incentive/social pressure while keeping question identity constant.
- Experiment 3: Detector transfer across mechanisms.
  - Why needed: Tests whether detectors trained on one mechanism generalize to another or simply pick up superficial artifacts.

## Research Question
When LLMs generate false statements, what proportion are attributable to epistemic failure versus incentive-driven misreporting under controlled prompt conditions?

## Background and Motivation
Prior studies (Truth is Universal; Internal State Knows When It’s Lying; AI-LieDar; DeceptionBench; Can LLMs Lie?) indicate that deception and hallucination are behaviorally distinct but often jointly evaluated. We leverage pre-gathered local datasets and code resources to build a practical mechanism-separated evaluation pipeline with real API model outputs.

## Hypothesis Decomposition
- H1 (Epistemic): In neutral factual QA, a non-trivial false-answer rate remains, reflecting epistemic limitations.
- H2 (Incentive): In paired sycophancy conditions, false-answer rate increases under incentive/social-pressure prompts relative to control.
- H3 (Conflation): A detector trained to identify one mechanism (epistemic vs incentive) transfers poorly to the other.

Independent variables:
- Prompt condition: neutral/control vs incentive-pressure.
- Dataset mechanism context: factual QA vs sycophancy.
- Detector training source: epistemic-only vs incentive-only.

Dependent variables:
- Binary correctness/truthfulness.
- Falsehood mechanism label (epistemic, incentive-induced, mixed/ambiguous).
- Detector AUROC/F1 under in-domain and cross-domain evaluation.

Success criteria:
- Significant pressure-induced falsehood increase in paired tests.
- Non-overlapping confidence intervals between key mechanism rates or statistically significant paired test.
- Detector transfer drop demonstrating mechanism non-equivalence.

## Proposed Methodology

### Approach
Use real LLM API responses on pre-downloaded datasets. Build mechanism labels through explicit controls:
1. Measure neutral epistemic error on factual QA.
2. For sycophancy-style items, compare control vs pressured prompt to infer whether falsehood emerged due to incentive.
3. Train lightweight text-feature detectors on one mechanism and evaluate transfer.

### Experimental Steps
1. Environment setup and reproducibility logging.
   - Rationale: deterministic, reproducible pipeline with versions/seeds/hardware.
2. Dataset loading and schema audit (`truthful_qa_multiple_choice`, `halueval_qa`, `sycophancy_answer`, `open_ended_sycophancy`, `deception_evals`).
   - Rationale: select fields enabling automatic correctness scoring.
3. Prompt template construction.
   - Neutral/control templates for factual answering.
   - Pressure templates for agreement/reward/social-approval bias.
4. API inference runs (primary model: GPT-4.1; optional secondary if budget permits).
   - Rationale: real-model behavior is required; no simulation.
5. Mechanism attribution logic.
   - Epistemic: false in neutral condition.
   - Incentive-induced: true in control but false under pressure (paired item).
   - Mixed/ambiguous: false in both or inconsistent mappings where attribution uncertain.
6. Detector training and transfer test.
   - Baseline features: refusal/certainty markers, answer length, hedging terms, contradiction cues.
7. Statistical analysis + visualization.
   - Paired tests, confidence intervals, effect sizes, failure case audit.

### Baselines
- Majority and random baselines for mechanism-class proportions.
- Simple lexical detector (logistic regression with TF-IDF) trained separately on each mechanism.
- Optional self-report confidence baseline (if prompt includes confidence scoring).

### Evaluation Metrics
- Primary: mechanism share among false outputs
  - % epistemic false, % incentive-induced false, % ambiguous.
- Correctness metrics:
  - Accuracy (MC/labelable tasks), falsehood rate per condition.
- Paired effect metrics:
  - Absolute risk increase under pressure.
  - McNemar test on paired binary outcomes.
  - Cohen’s g / odds ratio for paired change.
- Detector metrics:
  - AUROC, F1, precision, recall (in-domain vs cross-mechanism).

### Statistical Analysis Plan
- Alpha = 0.05.
- 95% bootstrap confidence intervals (10,000 resamples) for mechanism proportions.
- McNemar test for paired control-vs-pressure correctness on sycophancy set.
- Two-proportion z-test for selected unpaired comparisons.
- Benjamini-Hochberg correction across multiple hypothesis tests.

## Expected Outcomes
- A measurable epistemic falsehood base rate on neutral factual QA.
- A significant subset of additional falsehoods attributable to incentive pressure in paired prompts.
- Detector degradation under cross-mechanism evaluation, supporting mechanism conflation concerns.

## Timeline and Milestones
- M1 (Planning, 20-30 min): finalize design and mapping.
- M2 (Setup + EDA, 20-30 min): env, data schema, sample extraction.
- M3 (Implementation, 45-60 min): experiment scripts and scoring.
- M4 (Runs, 45-75 min): API calls and result generation.
- M5 (Analysis + Report, 30-45 min): stats, plots, write-up.
- Buffer: 25% reserved for API retries/schema edge cases.

## Potential Challenges
- Dataset schema heterogeneity.
  - Mitigation: per-dataset adapters and strict validation.
- Automated correctness scoring for open-ended outputs.
  - Mitigation: prefer multiple-choice/structured subsets and constrained parsing.
- API rate/cost limits.
  - Mitigation: sample-size caps, caching, retry/backoff, incremental checkpoints.
- Imperfect mechanism attribution.
  - Mitigation: explicit “ambiguous/mixed” bucket and transparent limitations.

## Success Criteria
- End-to-end reproducible pipeline producing `results/*.json` and figures.
- Empirical estimate of falsehood mechanism split with confidence intervals.
- At least one statistically supported pressure effect.
- Clear evidence (positive or negative) regarding detector transfer/conflation.
