# Call Complaint Intelligence Pipeline

An end-to-end pipeline for extracting structured intelligence from customer service call transcripts — summary, root cause classification, sentiment arc, and resolution signal — with a three-layer evaluation framework and two execution models: a sequential baseline and a LangGraph-orchestrated version with conditional routing.

Built as a focused exploration of applied LLM engineering for unstructured conversational data.

---

## The Problem

Customer service calls are dense with signal: what went wrong, how the customer felt about it, whether it got resolved. Extracting that signal manually doesn't scale. This pipeline automates the extraction using LLMs, producing structured outputs that can feed downstream analytics, quality monitoring, and agent coaching workflows.

---

## Architecture

Sequential pipeline (`pipeline.py`)
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

Graph pipeline (`pipeline_graph.py`)

```
START
  │
  ▼
ingest
  │
  ├─────────────────┬─────────────────┐
  ▼                 ▼                 ▼
summarize      categorize        sentiment     ← parallel fan-out
  │                 │                 │
  └─────────────────┴─────────────────┘
                    │
                    ▼
                 resolve
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  confidence ≥ 0.6        confidence < 0.6    ← conditional routing
        │                       │
        ▼                       ▼
    finalize                 escalate
                                │
                                ▼
                            finalize

```


### Why four separate LLM calls instead of one prompt

Each task — summary, category, sentiment, resolution — has its own prompt and its own output. This makes the system debuggable: if category accuracy degrades, I can isolate and fix that prompt without touching the others. A single mega-prompt is a black box; four small ones are individually evaluable and monitorable. The tradeoff is latency (~2–4s per transcript vs ~1s). For a batch offline workload, that's acceptable.

### Why two execution models
`pipeline.py` was built first, without an orchestration framework, to understand the primitive operations — prompt construction, response parsing, error handling — before abstracting them. `pipeline_graph.py` was built second, as a LangGraph refactor, to solve two specific problems the sequential version can't address cleanly: independent tasks running serially when they don't need to, and conditional routing logic that belongs inside the pipeline rather than in post-processing. Both versions produce identical `PipelineResult` outputs and can be evaluated with the same `evaluate.py`.

---

## Evaluation Framework

Real call transcripts are PII-sensitive, so the dataset here is synthetic — 30 transcripts generated via the same model being evaluated, covering 8 root cause categories across three resolution variants (clean resolution, partial resolution, unresolved/escalated). Generating the data deliberately forced close attention to what realistic complaint scenarios actually look like, which is useful prior to any labeling or eval work.

### Layer 1: Category Accuracy

A small hand-labeled eval set (`data/eval_labels.csv`, 15 transcripts) provides ground truth for root cause classification. Labeling these manually before running eval surfaces category boundary ambiguities — cases where "billing_dispute" and "policy_disagreement" overlap, for instance — that prompt iteration can address.

```
Overall accuracy:        63.6%   (7/11 correct)

Per-category accuracy:
  account_access                      0.0%
  agent_error_or_misinformation       100.0%
  billing_dispute                     50.0%
  fraud_or_unauthorized_charge        100.0%
  product_or_feature_issue            0.0%
  technical_issue                     100.0%
```

The `agent_error` / `policy_disagreement` confusion is the most instructive failure. Both categories involve a customer who feels wronged by the company, but the root cause differs: one is human error, the other is a structural policy the customer disagrees with. The prompt now surfaces that distinction explicitly; consistency improved across iterations.

### Layer 2: Hallucination Detection (LLM-as-Judge)

A second LLM call reads the source transcript and the generated summary, flagging any claims in the summary that can't be verified from the transcript. The judge prompt distinguishes three hallucination types: added information, contradiction, and unverifiable specifics (dates, amounts, outcomes not stated in the call).

This is imperfect — an LLM judge has its own error modes — but it's scalable in a way human review of every summary isn't. The output is a confidence score and a list of flagged claims, both of which can be logged and monitored over time.

```
Hallucination rate:   6.7%   (2/30 summaries flagged)
```

Both flagged cases involved the model inferring resolution outcomes ("the agent resolved the issue") from agent language that was actually inconclusive ("I'll look into that for you"). This is a prompt-level fix: the summary prompt now instructs the model to distinguish confirmed outcomes from stated intentions.

### Layer 3: Consistency Scoring

The same transcript was run three times. Category label, resolution flag, and sentiment arc end-state were compared across runs to measure output stability.

Non-determinism is a production risk, not just a research curiosity. A model that classifies the same call as `billing_dispute` on Monday and `policy_disagreement` on Thursday is not a reliable input to downstream analytics or monitoring systems.

```
Category stability:    100%   (5/5 transcripts consistent across 3 runs)
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
│   ├── pipeline.py             # sequential LLM inference layer
│   ├── pipeline_graph.py       # LangGraph refactor with parallel fan-out and conditional routing
│   └── evaluate.py             # three-layer eval framework
├── outputs/
│   ├── results.json            # sequential pipeline outputs
│   ├── results_graph.json      # graph pipeline outputs
│   ├── eval_results.json       # full eval output (machine-readable)
│   └── eval_report.txt         # human-readable eval summary
├── notebooks/
│   └── exploration.ipynb       # prompt iteration, error analysis, label review
├── run.py                      # batch entry point — sequential pipeline
├── run_graph.py                # batch entry point — graph pipeline
├── .env                        # ANTHROPIC_API_KEY (not committed)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install anthropic langgraph python-dotenv
pip freeze > requirements.txt
```

Add to `.env`:
```bash
ANTHROPIC_API_KEY=your_key_here
```

**Generate synthetic dataset:**
```bash
python -c "from ingest import load_transcripts; load_transcripts('synthetic', n=30)"
```

**Run both batch pipelines and evaluation:**
```bash
python run.py
```

**Smoke test individual components:**
```bash
PYTHONPATH=src python src/pipeline.py        # sequential smoke test
PYTHONPATH=src python src/pipeline_graph.py  # graph smoke test (triggers escalation branch)
PYTHONPATH=src python src/ingest.py          # normalization smoke test
PYTHONPATH=src python src/evaluate.py        # hallucination judge smoke test
```

---
## LangGraph Implementation

`pipeline_graph.py` refactors the sequential pipeline into a LangGraph `StateGraph`. Two concrete improvements over `pipeline.py`:

### Parallel fan-out

`summarize`, `categorize`, and `sentiment` are independent — none depends on another's output. In `pipeline.py` they run sequentially. In the graph they fan out from `ingest` as parallel branches and merge at `resolve`. With async LLM calls this reduces per-transcript latency by roughly 3×. The current implementation uses synchronous `_call_llm` calls inside each node, so the parallelism is structural but not yet realized in latency — the next step is swapping in `anthropic.AsyncAnthropic` calls (see [What's Next](#whats-next)).

### Conditional routing

Low-confidence resolutions (below 0.6) route to a dedicated `escalate` node that generates a structured recommendation: what information is missing, what follow-up action is needed, and which team should own it. In `pipeline.py` this logic would live outside the pipeline as post-processing — invisible to any tracing system. Inside the graph it's explicit, auditable, and extensible: adding a new branch (e.g. routing fraud cases to a different handler) is a one-line graph change.

### State management

LangGraph passes a shared `CallState` TypedDict between nodes, merging return values at each step. For keys written by multiple parallel nodes simultaneously — `node_latencies` — an `Annotated` type with a dict-merge reducer handles the fan-in:

```python
node_latencies: Annotated[dict[str, float], lambda a, b: {**a, **b}]
```

Without the reducer, LangGraph raises `InvalidUpdateError` when parallel nodes write to the same key in the same step.

### Comparing both versions

```bash
# Run sequential pipeline
python run.py

# Run graph pipeline
python run_graph.py

# Side-by-side comparison on a single transcript
PYTHONPATH=src python -c "
from pipeline_graph import compare_vs_sequential
transcript = open('data/raw_transcripts/synthetic_0001.txt').read()
compare_vs_sequential(transcript)
"
```

Both write to `outputs/` and produce identical `PipelineResult` objects, so `evaluate.py` works against either output file.

---

## What's Next

The current implementation is intentionally scoped to demonstrate the core inference and eval loop clearly. Three directions would move this toward production readiness:

### 1. Async LLM calls for true parallel execution

The graph is wired for parallel execution but `_call_llm` is synchronous, so the three parallel branches run sequentially in practice. Swapping in `anthropic.AsyncAnthropic` and making node functions `async def` would realize the ~3× latency improvement the graph structure is designed for. The wiring doesn't change — only the LLM call inside each node.

### 2. W&B Weave for experiment tracking

Prompt iteration is currently manual — change the prompt, re-run eval, compare numbers in a text file. Weights & Biases Weave would capture each prompt version, the model outputs, and the eval metrics in a structured experiment log, making it straightforward to compare runs, roll back to a prior prompt, and share results with teammates. For a team running evals at scale across millions of inputs, this is the difference between ad hoc iteration and reproducible experimentation.

Weave's LLM call tracing also addresses the audit trail requirement directly: every inference call, with its inputs, outputs, latency, and cost, is logged automatically.

---

## Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| Sequential pipeline built first | Understand primitives before abstracting | Redundant once graph version exists |
| Four separate LLM calls | Independently evaluable, debuggable per task | Higher latency vs. single prompt |
| LangGraph for graph version | Explicit topology, conditional routing, observability hooks | More boilerplate, framework dependency |
| Synchronous LLM calls in graph nodes | Simpler to reason about during development | Parallel fan-out latency benefit not yet realized |
| Conditional routing inside graph | Escalation logic is auditable and extensible | Routing threshold (0.6) needs calibration on real data |
| Annotated reducer for node_latencies | Required for parallel nodes writing to the same state key | Less intuitive than plain TypedDict fields |
| Synthetic data | PII-safe, controllable scenario coverage | Distribution shift vs. real calls |
| LLM-as-judge for hallucination | Scalable, no human review bottleneck | Judge has its own error modes |
| Human labels for category accuracy | Only genuine ground truth for boundary judgment calls | Manual labeling overhead |
| Separate eval CSV | Clean separation of labels from outputs | Must be maintained as taxonomy evolves |

---

## Requirements

```
anthropic>=0.20.0
langgraph>=0.2.0
python-dotenv>=1.0.0
```