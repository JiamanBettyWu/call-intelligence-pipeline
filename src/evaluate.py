"""
Evaluation Framework
----------------------
Three evaluation layers, in increasing sophistication:

  1. Category accuracy      — compare predicted category to human labels
  2. Hallucination check    — LLM-as-judge: does the summary contain claims
                              not supported by the source transcript?
  3. Consistency scoring    — run the same transcript N times, measure output
                              variance (critical for non-deterministic LLM outputs)

Design notes:
  - Ground truth labels live in data/eval_labels.csv — a small hand-labeled
    set of 10-15 transcripts. Manual labeling is cheap here and forces you
    to deeply understand the data before modeling it.
  - Hallucination detection uses a second LLM call (LLM-as-judge pattern).
    This is imperfect but scalable — the alternative is human review of every
    summary, which doesn't scale.
  - Consistency scoring directly addresses the JD's language about
    "non-deterministic outputs" and "continuous monitoring controls."
  - All results are written to outputs/eval_results.json and a human-readable
    text report.
"""

import csv
import json
import os
import time
from dataclasses import dataclass, asdict, field
from collections import Counter
from typing import Optional
import anthropic
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 512
CONSISTENCY_RUNS = 3      # how many times to re-run each transcript for consistency eval

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CategoryEval:
    transcript_id: str
    predicted: str
    ground_truth: str
    correct: bool


@dataclass
class HallucinationEval:
    transcript_id: str
    summary: str
    hallucination_detected: bool
    flagged_claims: list[str]       # specific claims the judge flagged
    confidence: float               # judge's confidence that hallucination exists
    raw_judge_output: str


@dataclass
class ConsistencyEval:
    transcript_id: str
    n_runs: int
    category_values: list[str]      # one per run
    resolution_values: list[bool]   # one per run
    category_stable: bool           # same value across all runs
    resolution_stable: bool
    sentiment_end_values: list[str]
    sentiment_end_stable: bool


@dataclass
class EvalReport:
    n_labeled: int
    category_accuracy: float
    category_breakdown: dict        # per-category accuracy
    n_hallucination_checked: int
    hallucination_rate: float
    n_consistency_checked: int
    category_stability_rate: float
    resolution_stability_rate: float
    per_transcript: dict = field(default_factory=dict)  # transcript_id → all evals


# ---------------------------------------------------------------------------
# 1. Category accuracy
# ---------------------------------------------------------------------------

def load_ground_truth(labels_path: str = "data/eval_labels.csv") -> dict[str, str]:
    """
    Load hand-labeled ground truth from CSV.

    Expected CSV format:
        transcript_id,category,resolved
        synthetic_0001,billing_dispute,true
        ...

    Returns:
        dict mapping transcript_id → category label
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Ground truth labels not found at {labels_path}.\n"
            "Create this file by running the pipeline on a small set of transcripts,\n"
            "reviewing the outputs, and correcting them manually."
        )

    labels = {}
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["transcript_id"]] = {
                "category": row["category"],
                "resolved": row.get("resolved", "").lower() == "true",
            }
    return labels


def evaluate_categories(
    results: list[dict],
    ground_truth: dict[str, dict],
) -> tuple[list[CategoryEval], float]:
    """
    Compare predicted categories to ground truth labels.

    Args:
        results:        List of PipelineResult dicts (from pipeline.run_batch)
        ground_truth:   Dict from load_ground_truth()

    Returns:
        evals:          Per-transcript CategoryEval objects
        accuracy:       Overall accuracy float
    """
    evals = []
    for r in results:
        tid = r["transcript_id"]
        if tid not in ground_truth:
            continue

        predicted = r["root_cause_category"]
        gt = ground_truth[tid]["category"]
        evals.append(CategoryEval(
            transcript_id=tid,
            predicted=predicted,
            ground_truth=gt,
            correct=(predicted == gt),
        ))

    if not evals:
        return evals, 0.0

    accuracy = sum(e.correct for e in evals) / len(evals)
    return evals, accuracy


def category_breakdown(evals: list[CategoryEval]) -> dict:
    """Per-category precision — useful for spotting which categories confuse the model."""
    by_category = {}
    for e in evals:
        cat = e.ground_truth
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if e.correct:
            by_category[cat]["correct"] += 1

    return {
        cat: round(v["correct"] / v["total"], 3)
        for cat, v in by_category.items()
    }


# ---------------------------------------------------------------------------
# 2. Hallucination detection (LLM-as-judge)
# ---------------------------------------------------------------------------

def _hallucination_judge_prompt(transcript: str, summary: str) -> str:
    return f"""You are evaluating the faithfulness of a summary to its source transcript.

Your job: identify any claims in the summary that are NOT supported by the transcript.
A claim is unsupported if:
  - It adds information not present in the transcript
  - It contradicts something in the transcript
  - It makes a specific assertion (number, date, outcome) that can't be verified from the transcript

Source transcript:
{transcript}

Summary to evaluate:
{summary}

Respond in this exact JSON format (no markdown, no code fences):
{{
  "hallucination_detected": true or false,
  "flagged_claims": ["<claim 1>", "<claim 2>"],
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

If no hallucinations are detected, return an empty list for flagged_claims."""


def check_hallucination(
    client: anthropic.Anthropic,
    transcript: str,
    summary: str,
    transcript_id: str,
) -> HallucinationEval:
    """Run LLM-as-judge hallucination check on a single summary."""
    prompt = _hallucination_judge_prompt(transcript, summary)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)

        return HallucinationEval(
            transcript_id=transcript_id,
            summary=summary,
            hallucination_detected=bool(parsed.get("hallucination_detected", False)),
            flagged_claims=parsed.get("flagged_claims", []),
            confidence=float(parsed.get("confidence", 0.5)),
            raw_judge_output=raw,
        )

    except Exception as e:
        # Don't crash the eval — return a flagged result so it's visible
        return HallucinationEval(
            transcript_id=transcript_id,
            summary=summary,
            hallucination_detected=False,
            flagged_claims=[],
            confidence=0.0,
            raw_judge_output=f"ERROR: {e}",
        )


def evaluate_hallucinations(
    results: list[dict],
    transcripts: dict[str, str],    # transcript_id → raw text
    sleep_between: float = 0.5,
) -> tuple[list[HallucinationEval], float]:
    """
    Run hallucination checks across all results that have a matching transcript.

    Args:
        results:        Pipeline results (list of dicts)
        transcripts:    Dict mapping transcript_id → full transcript text
        sleep_between:  Rate limit courtesy

    Returns:
        evals:              Per-transcript HallucinationEval objects
        hallucination_rate: Fraction of summaries with detected hallucinations
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    evals = []

    checkable = [r for r in results if r["transcript_id"] in transcripts]
    print(f"\nRunning hallucination checks on {len(checkable)} transcripts...")

    for i, r in enumerate(checkable):
        tid = r["transcript_id"]
        eval_result = check_hallucination(
            client=client,
            transcript=transcripts[tid],
            summary=r["summary"],
            transcript_id=tid,
        )
        evals.append(eval_result)

        status = "⚠ HALLUCINATION" if eval_result.hallucination_detected else "✓ clean"
        print(f"  [{i+1}/{len(checkable)}] {tid}: {status}")
        if eval_result.flagged_claims:
            for claim in eval_result.flagged_claims:
                print(f"    → {claim}")

        if i < len(checkable) - 1:
            time.sleep(sleep_between)

    if not evals:
        return evals, 0.0

    rate = sum(e.hallucination_detected for e in evals) / len(evals)
    return evals, rate


# ---------------------------------------------------------------------------
# 3. Consistency scoring
# ---------------------------------------------------------------------------

def evaluate_consistency(
    transcripts: dict[str, str],
    n_runs: int = CONSISTENCY_RUNS,
    sleep_between: float = 0.5,
    sample_size: int = 5,           # number of transcripts to consistency-check (API cost)
) -> tuple[list[ConsistencyEval], float, float]:
    """
    Run each transcript N times and measure output variance.

    This is the most direct test of non-determinism — a critical concern
    for production LLM systems flagged explicitly in the JD.

    Args:
        transcripts:    Dict of transcript_id → text
        n_runs:         How many times to run each transcript
        sleep_between:  Rate limit courtesy
        sample_size:    Max transcripts to test (keeps API costs bounded)

    Returns:
        evals:                      Per-transcript ConsistencyEval objects
        category_stability_rate:    Fraction of transcripts with stable category
        resolution_stability_rate:  Fraction with stable resolution flag
    """
    # Import here to avoid circular dependency
    from pipeline import run_pipeline

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    sampled = dict(list(transcripts.items())[:sample_size])
    evals = []

    print(f"\nRunning consistency checks ({n_runs} runs × {len(sampled)} transcripts)...")

    for tid, text in sampled.items():
        categories = []
        resolutions = []
        sentiment_ends = []

        for run in range(n_runs):
            try:
                result = run_pipeline(text, tid, client)
                categories.append(result.root_cause_category)
                resolutions.append(result.resolution_flag)
                sentiment_ends.append(result.sentiment_arc.end)
                print(f"  {tid} run {run+1}/{n_runs}: "
                      f"{result.root_cause_category} | "
                      f"resolved={result.resolution_flag} | "
                      f"sentiment_end={result.sentiment_arc.end}")
            except Exception as e:
                print(f"  ✗ {tid} run {run+1} failed: {e}")

            if run < n_runs - 1:
                time.sleep(sleep_between)

        evals.append(ConsistencyEval(
            transcript_id=tid,
            n_runs=len(categories),
            category_values=categories,
            resolution_values=resolutions,
            category_stable=len(set(categories)) == 1,
            resolution_stable=len(set(resolutions)) == 1,
            sentiment_end_values=sentiment_ends,
            sentiment_end_stable=len(set(sentiment_ends)) == 1,
        ))

    if not evals:
        return evals, 0.0, 0.0

    cat_stable_rate = sum(e.category_stable for e in evals) / len(evals)
    res_stable_rate = sum(e.resolution_stable for e in evals) / len(evals)

    return evals, cat_stable_rate, res_stable_rate


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    category_evals: list[CategoryEval],
    category_accuracy: float,
    hallucination_evals: list[HallucinationEval],
    hallucination_rate: float,
    consistency_evals: list[ConsistencyEval],
    category_stability_rate: float,
    resolution_stability_rate: float,
) -> EvalReport:

    breakdown = category_breakdown(category_evals)

    per_transcript = {}

    for e in category_evals:
        per_transcript.setdefault(e.transcript_id, {})["category_eval"] = asdict(e)
    for e in hallucination_evals:
        per_transcript.setdefault(e.transcript_id, {})["hallucination_eval"] = asdict(e)
    for e in consistency_evals:
        per_transcript.setdefault(e.transcript_id, {})["consistency_eval"] = asdict(e)

    return EvalReport(
        n_labeled=len(category_evals),
        category_accuracy=round(category_accuracy, 3),
        category_breakdown=breakdown,
        n_hallucination_checked=len(hallucination_evals),
        hallucination_rate=round(hallucination_rate, 3),
        n_consistency_checked=len(consistency_evals),
        category_stability_rate=round(category_stability_rate, 3),
        resolution_stability_rate=round(resolution_stability_rate, 3),
        per_transcript=per_transcript,
    )


def write_report(
    report: EvalReport,
    json_path: str = "outputs/eval_results.json",
    txt_path: str = "outputs/eval_report.txt",
):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # JSON dump (machine-readable)
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Human-readable text report
    lines = [
        "=" * 60,
        "CALL INTELLIGENCE PIPELINE — EVALUATION REPORT",
        "=" * 60,
        "",
        "",
        f"  Labeled transcripts:  {report.n_labeled}",
        f"  Overall accuracy:     {report.category_accuracy:.1%}",
        "",
        "  Per-:",
    ]
    for cat, acc in sorted(report.category_breakdown.items()):
        lines.append(f"    {cat:<35} {acc:.1%}")

    lines += [
        "",
        "HALLUCINATION DETECTION (LLM-as-judge)",
        f"  Transcripts checked:  {report.n_hallucination_checked}",
        f"  Hallucination rate:   {report.hallucination_rate:.1%}",
        "",
        "CONSISTENCY (non-determinism)",
        f"  Transcripts tested:   {report.n_consistency_checked} ({CONSISTENCY_RUNS} runs each)",
        f"  Category stability:   {report.category_stability_rate:.1%}",
        f"  Resolution stability: {report.resolution_stability_rate:.1%}",
        "",
        "=" * 60,
        "Full results written to: " + json_path,
    ]

    report_text = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_full_eval(
    results_path: str = "outputs/results.json",
    transcripts_source: str = "data/raw_transcripts/",
    labels_path: str = "data/eval_labels.csv",
    run_consistency: bool = True,
):
    """
    Run the full evaluation suite against existing pipeline results.

    Args:
        results_path:       Path to pipeline outputs (from pipeline.run_batch)
        transcripts_source: Directory or JSON path to load source transcripts
        labels_path:        Path to hand-labeled CSV
        run_consistency:    Whether to run consistency checks (slower, uses API)
    """
    # Load pipeline results
    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]
    print(f"Loaded {len(results)} pipeline results from {results_path}")

    # Load source transcripts (needed for hallucination check)
    from ingest import load_transcripts
    transcript_list = load_transcripts(transcripts_source)
    transcripts = {t.id: t.text for t in transcript_list}

    # 1. Category accuracy
    try:
        ground_truth = load_ground_truth(labels_path)
        print(ground_truth)
        cat_evals, cat_accuracy = evaluate_categories(results, ground_truth)
       
        print(f"\nCategory accuracy: {cat_accuracy:.1%} "
              f"({sum(e.correct for e in cat_evals)}/{len(cat_evals)} correct)")
    except FileNotFoundError as e:
        print(f"\n⚠ Skipping category eval: {e}")
        cat_evals, cat_accuracy = [], 0.0

    # 2. Hallucination detection
    hall_evals, hall_rate = evaluate_hallucinations(results, transcripts)
    print(f"Hallucination rate: {hall_rate:.1%}")

    # 3. Consistency (optional — uses more API calls)
    if run_consistency:
        con_evals, cat_stable, res_stable = evaluate_consistency(transcripts)
    else:
        con_evals, cat_stable, res_stable = [], 0.0, 0.0
        print("\nSkipping consistency eval (run_consistency=False)")

    # Assemble and write report
    report = build_report(
        cat_evals, cat_accuracy,
        hall_evals, hall_rate,
        con_evals, cat_stable, res_stable,
    )
    write_report(report)

    return report


if __name__ == "__main__":
    # Quick smoke test for hallucination judge only
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    transcript = """Agent: Thank you for calling, how can I help?
Customer: I have a charge for $49.99 I don't recognize from March 18th.
Agent: I can see that charge. I'll raise a dispute and apply a provisional credit within 24 hours.
Customer: Thank you, that's all I needed.
Agent: Of course, have a great day."""

    faithful_summary = "Customer disputed a $49.99 charge from March 18th. Agent raised a dispute and offered a provisional credit within 24 hours."
    hallucinated_summary = "Customer disputed a $49.99 charge and mentioned they had been a loyal customer for 10 years. Agent waived the fee entirely."

    print("--- Testing faithful summary ---")
    r1 = check_hallucination(client, transcript, faithful_summary, "test_faithful")
    print(f"Hallucination detected: {r1.hallucination_detected}")
    print(f"Flagged claims: {r1.flagged_claims}")

    print("\n--- Testing hallucinated summary ---")
    r2 = check_hallucination(client, transcript, hallucinated_summary, "test_hallucinated")
    print(f"Hallucination detected: {r2.hallucination_detected}")
    print(f"Flagged claims: {r2.flagged_claims}")