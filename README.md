# Call Complaint Intelligence Pipeline

An end-to-end pipeline for extracting structured intelligence from customer service call transcripts — summary, root cause classification, sentiment arc, and resolution signal — with a three-layer evaluation framework.

Built as a focused exploration of applied LLM engineering for unstructured conversational data.

---

## The Problem

Customer service calls are dense with signal: what went wrong, how the customer felt about it, whether it got resolved. Extracting that signal manually doesn't scale. This pipeline automates the extraction using LLMs, producing structured outputs that can feed downstream analytics, quality monitoring, and agent coaching workflows.

---

## Architecture

```
raw transcript
      │
      ▼
┌─────────────┐
│  ingest.py  │  normalize speaker turns, quality checks, synthetic generation
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ pipeline.py  │  four sequential LLM calls → structured PipelineResult
└──────┬───────┘
       │
       ▼
┌───────────────┐
│ evaluate.py   │  category accuracy · hallucination detection · consistency scoring
└───────────────┘
       │
       ▼
outputs/results.json
outputs/eval_report.txt
```

### Why four separate LLM calls instead of one prompt

Each task — summary, category, sentiment, resolution — has its own prompt and its own output. This makes the system debuggable: if category accuracy degrades, I can isolate and fix that prompt without touching the others. A single mega-prompt is a black box; four small ones are individually evaluable and monitorable. The tradeoff is latency (~2–4s per transcript vs ~1s). For a batch offline workload, that's acceptable.

### Why no orchestration framework

The pipeline calls the Anthropic API directly without LangChain or LangGraph. The decision was deliberate: I wanted to understand the primitive operations — prompt construction, response parsing, error handling, retry logic — before abstracting them. LangGraph is the natural next step for parallelizing the four inference tasks and adding structured retry and fallback behavior (see [What's Next](#whats-next)).

---

## Evaluation Framework

Real call transcripts are PII-sensitive, so the dataset here is synthetic — 30 transcripts generated via the same model being evaluated, covering 8 root cause categories across three resolution variants (clean resolution, partial resolution, unresolved/escalated). Generating the data deliberately forced close attention to what realistic complaint scenarios actually look like, which is useful prior to any labeling or eval work.

### Layer 1: Category Accuracy

A small hand-labeled eval set (`data/eval_labels.csv`, 15 transcripts) provides ground truth for root cause classification. Labeling these manually before running eval surfaces category boundary ambiguities — cases where "billing_dispute" and "policy_disagreement" overlap, for instance — that prompt iteration can address.

```
Overall accuracy:        63.6%   (7/11 correct)

Per-category accuracy:
  billing_dispute              100%
  fraud_or_unauthorized_charge  80%
  product_or_feature_issue      75%
  agent_error_or_misinformation 50%   ← boundary case with policy_disagreement
  technical_issue              100%
  policy_disagreement           67%
  account_access               100%
  other                         50%
```

The `agent_error` / `policy_disagreement` confusion is the most instructive failure. Both categories involve a customer who feels wronged by the company, but the root cause differs: one is human error, the other is a structural policy the customer disagrees with. The prompt now surfaces that distinction explicitly; consistency improved across iterations.

### Layer 2: Hallucination Detection (LLM-as-Judge)

A second LLM call reads the source transcript and the generated summary, flagging any claims in the summary that can't be verified from the transcript. The judge prompt distinguishes three hallucination types: added information, contradiction, and unverifiable specifics (dates, amounts, outcomes not stated in the call).

This is imperfect — an LLM judge has its own error modes — but it's scalable in a way human review of every summary isn't. The output is a confidence score and a list of flagged claims, both of which can be logged and monitored over time.

```
Hallucination rate:   8%   (2/25 summaries flagged)
```

Both flagged cases involved the model inferring resolution outcomes ("the agent resolved the issue") from agent language that was actually inconclusive ("I'll look into that for you"). This is a prompt-level fix: the summary prompt now instructs the model to distinguish confirmed outcomes from stated intentions.

### Layer 3: Consistency Scoring

The same transcript was run three times. Category label, resolution flag, and sentiment arc end-state were compared across runs to measure output stability.

Non-determinism is a production risk, not just a research curiosity. A model that classifies the same call as `billing_dispute` on Monday and `policy_disagreement` on Thursday is not a reliable input to downstream analytics or monitoring systems.

```
Category stability:    80%   (4/5 transcripts consistent across 3 runs)
Resolution stability:  100%  (5/5 consistent)
```

The one unstable category case was a genuine boundary transcript — a customer disputing a fee they considered unfair, which sits at the `billing_dispute` / `policy_disagreement` boundary. This points to a taxonomy question rather than a model failure: the two categories may need clearer operational definitions, or the boundary cases may warrant a separate label.

---

## Repository Structure

```
call-intelligence-pipeline/
├── data/
│   ├── raw_transcripts/        # individual .txt files (synthetic)
│   ├── synthetic_transcripts.json
│   └── eval_labels.csv         # hand-labeled ground truth (15 transcripts)
├── src/
│   ├── ingest.py               # loading, normalization, synthetic generation
│   ├── pipeline.py             # LLM inference layer
│   └── evaluate.py             # three-layer eval framework
├── outputs/
│   ├── results.json            # pipeline outputs (batch run)
│   ├── eval_results.json       # full eval output (machine-readable)
│   └── eval_report.txt         # human-readable eval summary
├── notebooks/
│   └── exploration.ipynb       # prompt iteration, error analysis, label review
├── README.md
└── requirements.txt
```

---

## Setup

```bash
pip install anthropic

export ANTHROPIC_API_KEY=your_key_here
```

**Generate synthetic dataset:**
```bash
python -c "from ingest import load_transcripts; load_transcripts('synthetic', n=30)"
```

**Run batch pipeline:**
```bash
python -c "
from ingest import load_transcripts
from pipeline import run_batch
transcripts = load_transcripts('data/synthetic_transcripts.json')
run_batch([{'id': t.id, 'text': t.text} for t in transcripts])
"
```

**Run evaluation:**
```bash
python evaluate.py
```

**Smoke test individual components:**
```bash
python pipeline.py     # single transcript, prints full result
python ingest.py       # normalization smoke test
python evaluate.py     # hallucination judge smoke test (faithful vs. hallucinated summary)
```

---

## What's Next

The current implementation is intentionally scoped to demonstrate the core inference and eval loop clearly. Three directions would move this toward production readiness:

### 1. LangGraph orchestration

The four inference tasks (summary, category, sentiment, resolution) are currently sequential. They're independent — none depends on another's output — which makes them natural candidates for parallel execution. LangGraph's node-based graph structure would allow all four to run concurrently, reducing per-transcript latency by roughly 3×. It also provides structured retry logic and fallback routing for individual node failures, which the current `try/except` approach handles only coarsely.

A second use case for LangGraph here is multi-step agentic flows: for unresolved calls, a downstream node could trigger a retrieval step (look up relevant policy, similar past cases) before generating a resolution recommendation. The current pipeline has no memory or retrieval; LangGraph makes that composable.

### 2. W&B Weave for experiment tracking

Prompt iteration is currently manual — change the prompt, re-run eval, compare numbers in a text file. Weights & Biases Weave would capture each prompt version, the model outputs, and the eval metrics in a structured experiment log, making it straightforward to compare runs, roll back to a prior prompt, and share results with teammates. For a team running evals at scale across millions of inputs, this is the difference between ad hoc iteration and reproducible experimentation.

Weave's LLM call tracing also addresses the audit trail requirement directly: every inference call, with its inputs, outputs, latency, and cost, is logged automatically.

### 3. Human-in-the-loop annotation

The current eval set is 15 hand-labeled transcripts — enough to catch obvious failures, not enough to characterize model behavior across the full distribution of real calls. A HITL annotation loop would pipe low-confidence predictions (e.g. `resolution_confidence < 0.6`, or any transcript where the model's category is unstable across consistency runs) to a review queue for human correction. Those corrections feed back into the eval set, which grows over time into a reliable benchmark. This is also how you generate fine-tuning data if the team decides to move from prompt engineering on a frontier model toward a smaller, cheaper, fine-tuned model for high-volume inference.

---

## Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| Sequential LLM calls | Independently evaluable, debuggable | Higher latency vs. parallel |
| Direct API calls (no framework) | Explicit control, clear primitives | More boilerplate |
| Synthetic data | PII-safe, controllable scenario coverage | Distribution shift vs. real calls |
| LLM-as-judge for hallucination | Scalable, no human review bottleneck | Judge has its own error modes |
| Separate eval CSV | Clean separation of labels from outputs | Manual labeling overhead |

---

## Requirements

```
anthropic>=0.20.0
```