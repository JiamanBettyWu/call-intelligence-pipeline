# Call Complaint Intelligence Pipeline

End-to-end LLM system for extracting structured intelligence from customer service call transcripts, with a multi-transcript RAG Q&A layer and a three-layer evaluation framework.

---

## Architecture

Two modes of execution:

```
                        transcripts (synthetic)
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
             EXTRACTION MODE                    Q&A MODE
                    │                               │
              ingest.py                       chunk by speaker turn
         (normalize, validate)                      │
                    │                               ▼
                    │                    embed (all-MiniLM-L6-v2)
          ┌─────────┴──────────┐                    │
          │                    │                    ▼
     pipeline.py       pipeline_graph.py     ChromaDB index
     (sequential)       (LangGraph)                 │
          │                    │            question → retrieve
          │          ┌─────────┴──────┐             │
          │        ingest             │             ▼
          │          │                │       answer.py
          │    ┌─────┼─────┐          │    (RAGAnswer + citations)
          │  sum.  cat.  sent.        │ 
          │    └─────┼─────┘          │
          │        resolve            │
          │       ┌──┴──┐             │
          │    final  escalate        │
          │                           │
          └──────────┬────────────────┘
                     │
                evaluate.py
      (accuracy · hallucination · consistency)
                     │
              outputs/results.json
              outputs/eval_report.txt
```

**Extraction vs Q&A — when to use which:**

|Question|Right tool|
|---|---|
|How do agents handle fraud disputes?|RAG — semantic lookup|
|What fraction of calls ended unresolved?|Pipeline outputs → `df["resolution_flag"].value_counts()`|
|What did the customer say in transcript 0003?|RAG with transcript filter|
|Which root cause category appears most?|Pipeline outputs → `value_counts()`|

RAG retrieves by relevance — the most similar chunks, not all matching chunks. Counting and aggregation questions belong to the structured `outputs/results.json` layer.

---

## Key Design Decisions

**Four separate LLM calls, not one prompt.** Summary, category, sentiment, and resolution each have their own prompt and output. Each task is self-contained.

**Sequential pipeline built before LangGraph.** `pipeline.py` executes prompt construction, JSON parsing, error handling, before abstracting them. `pipeline_graph.py` solves two problems the sequential version can't: independent tasks running serially, and conditional routing that belongs inside the pipeline. The escalation node (low-confidence resolutions → structured recommendation) is the new behavior.

**Chunk by speaker turn, not token windows.** Speaker turns are natural semantic units in conversational data. Splitting mid-turn breaks meaning. Turns under 8 words are filtered at index time.

**Pydantic for LLM output, dataclass for internal data.** `RAGAnswer` and `Citation` use Pydantic — runtime validation catches malformed LLM output at the boundary. `PipelineResult` and `SentimentArc` use dataclasses — internal data you construct and trust doesn't need coercion.

**Human labels for category accuracy, LLM-as-judge for hallucination.** Category ground truth requires a judgment call about boundary definitions — only a human can settle where `billing_dispute` ends and `policy_disagreement` begins. Hallucination detection is a text entailment check — verifiable against the source transcript, scalable with an LLM judge.

---

## Evaluation

Dataset: 30 synthetic transcripts covering 8 root cause categories × 3 resolution variants. Generated via the same model being evaluated — PII-safe and forces close attention to what realistic complaint scenarios look like.

**Layer 1 — Category accuracy** (hand-labeled ground truth, 15 transcripts)

```
Overall: 63.6%
```

**Layer 2 — Hallucination detection** (LLM-as-judge)

```
Rate: 6.7% (2/30 flagged)
```

**Layer 3 — Consistency scoring** (3 runs per transcript, 5 transcripts)

```
Category stability: 100%
Resolution stability: 100%
```

---

## Repository Structure

```
call-intelligence-pipeline/
├── data/
│   ├── raw_transcripts/            # synthetic .txt files
│   ├── synthetic_transcripts.json
│   ├── chroma_db/                  # persisted vector index
│   └── eval_labels.csv             # hand-labeled ground truth
├── src/
│   ├── ingest.py                   # loading, normalization, generation
│   ├── pipeline.py                 # sequential inference
│   ├── pipeline_graph.py           # LangGraph: parallel fan-out + conditional routing
│   ├── evaluate.py                 # three-layer eval framework
│   ├── rag_qa.py                   # chunking, embedding, indexing, retrieval
│   └── answer.py                   # LLM answer node, Pydantic output schema
├── outputs/
├── notebooks/
│   └── exploration.ipynb
├── run.py                          # sequential batch entry point
├── run_graph.py                    # graph batch entry point
├── .env
└── requirements.txt
```

---

## Setup & Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install anthropic langgraph python-dotenv chromadb sentence-transformers pydantic
```

```bash
# Generate data
PYTHONPATH=src python -c "from ingest import load_transcripts; load_transcripts('synthetic', n=30)"

# Run pipelines
python run.py                           # sequential
python run_graph.py                     # LangGraph

# Build RAG index
PYTHONPATH=src python src/rag_qa.py

# Ask a question
PYTHONPATH=src python -c "
from answer import ask, format_answer
print(format_answer(ask('How do agents handle fraud disputes?')))
"

# Evaluate
PYTHONPATH=src python src/evaluate.py
```

---

## What's Next

**Async LLM calls** — swap `_call_llm` for `anthropic.AsyncAnthropic` inside each graph node to realize the ~3× latency improvement the parallel fan-out is designed for.

**Agentic RAG with question routing** — classify questions before retrieval (lookup vs. aggregation vs. hybrid) and route to the appropriate tool. Conditional routing in LangGraph is already demonstrated in the extraction pipeline; this applies the same pattern to the Q&A layer. Multi-step retrieval with a sufficiency check is the full agentic version.

**W&B Weave** — per-call tracing (inputs, outputs, latency, cost) and structured experiment tracking for prompt iteration. Currently prompt changes are compared manually against a text file.

**HITL annotation loop** — pipe low-confidence and consistency-unstable predictions to a review queue. Corrections grow the eval set over time and generate fine-tuning data if the team moves toward a smaller, specialized model for high-volume inference.

---

## Requirements

```
anthropic>=0.20.0
langgraph>=0.2.0
python-dotenv>=1.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
pydantic>=2.0.0
```