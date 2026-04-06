"""
Call Complaint Intelligence Pipeline
-------------------------------------
Takes a customer service call transcript and produces structured intelligence:
  - Extractive summary
  - Root cause category
  - Sentiment arc (beginning / middle / end)
  - Resolution flag + confidence

Design notes:
  - Calls Anthropic API directly (no orchestration framework) to keep primitive
    operations explicit. LangGraph orchestration is a natural next step for
    parallelization and retry logic.
  - Each task is a separate LLM call with its own prompt — easier to evaluate,
    iterate, and monitor independently than a single mega-prompt.
  - Outputs are typed via dataclasses for downstream eval compatibility.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional
import anthropic

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

# Root cause taxonomy — tune these to match the team's actual categories
ROOT_CAUSE_CATEGORIES = [
    "billing_dispute",
    "fraud_or_unauthorized_charge",
    "product_or_feature_issue",
    "agent_error_or_misinformation",
    "technical_issue",
    "policy_disagreement",
    "account_access",
    "other",
]

SENTIMENT_LABELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class SentimentArc:
    beginning: str   # one of SENTIMENT_LABELS
    middle: str
    end: str

    def __post_init__(self):
        for field, val in [("beginning", self.beginning),
                           ("middle", self.middle),
                           ("end", self.end)]:
            if val not in SENTIMENT_LABELS:
                raise ValueError(f"Invalid sentiment '{val}' for {field}. "
                                 f"Must be one of {SENTIMENT_LABELS}")


@dataclass
class PipelineResult:
    transcript_id: str
    summary: str                        # 2-3 sentence extractive summary
    root_cause_category: str            # one of ROOT_CAUSE_CATEGORIES
    root_cause_explanation: str         # 1-sentence rationale
    sentiment_arc: SentimentArc
    resolution_flag: bool               # was the issue resolved?
    resolution_confidence: float        # 0.0 - 1.0
    resolution_notes: str               # brief explanation
    model: str
    latency_seconds: float
    raw_outputs: dict                   # store raw LLM responses for debugging

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------

def _summary_prompt(transcript: str) -> str:
    return f"""You are analyzing a customer service call transcript.

Write a 2-3 sentence summary of this call. Focus on:
1. What the customer's issue was
2. What actions were taken during the call
3. How the call ended

Be extractive — only include information present in the transcript. Do not infer or embellish.

Transcript:
{transcript}

Respond with only the summary text. No preamble, no labels."""


def _category_prompt(transcript: str) -> str:
    categories_str = "\n".join(f"- {c}" for c in ROOT_CAUSE_CATEGORIES)
    return f"""You are analyzing a customer service call transcript.

Classify the root cause of the customer's complaint into exactly one of these categories:
{categories_str}

Then provide a one-sentence explanation of why you chose that category.

Transcript:
{transcript}

Respond in this exact JSON format (no markdown, no code fences):
{{
  "category": "<category_name>",
  "explanation": "<one sentence>"
}}"""


def _sentiment_prompt(transcript: str) -> str:
    labels_str = ", ".join(SENTIMENT_LABELS)
    return f"""You are analyzing the emotional tone of a customer service call.

Assess the customer's sentiment at three points in the call:
- beginning: their tone in the first ~20% of the conversation
- middle: their tone in the middle ~60%
- end: their tone in the final ~20%

Valid sentiment labels: {labels_str}

Transcript:
{transcript}

Respond in this exact JSON format (no markdown, no code fences):
{{
  "beginning": "<label>",
  "middle": "<label>",
  "end": "<label>"
}}"""


def _resolution_prompt(transcript: str) -> str:
    return f"""You are analyzing a customer service call transcript.

Determine whether the customer's issue was resolved by the end of the call.

A call is "resolved" if the customer's core problem was addressed to a reasonable conclusion —
even if they weren't fully satisfied. A call is "unresolved" if the issue was left open,
escalated without closure, or the customer hung up without a path forward.

Transcript:
{transcript}

Respond in this exact JSON format (no markdown, no code fences):
{{
  "resolved": true or false,
  "confidence": <float between 0.0 and 1.0>,
  "notes": "<one sentence explaining your assessment>"
}}"""


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------

def _call_llm(client: anthropic.Anthropic, prompt: str, task_name: str) -> str:
    """
    Single LLM call with basic error handling.
    In production: add retry logic with exponential backoff here.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except anthropic.APIError as e:
        raise RuntimeError(f"LLM call failed for task '{task_name}': {e}") from e


def _parse_json_response(raw: str, task_name: str) -> dict:
    """
    Safely parse JSON from LLM output.
    Guards against markdown fences or stray whitespace.
    """
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON for task '{task_name}'.\n"
                         f"Raw output: {raw}\nError: {e}") from e


# ---------------------------------------------------------------------------
# Individual task runners
# ---------------------------------------------------------------------------

def run_summary(client: anthropic.Anthropic, transcript: str) -> str:
    raw = _call_llm(client, _summary_prompt(transcript), "summary")
    return raw


def run_category(client: anthropic.Anthropic, transcript: str) -> dict:
    raw = _call_llm(client, _category_prompt(transcript), "category")
    parsed = _parse_json_response(raw, "category")

    if parsed.get("category") not in ROOT_CAUSE_CATEGORIES:
        # Fallback: don't crash, but flag it
        parsed["category"] = "other"
        parsed["explanation"] = f"[Fallback] Original: {parsed.get('explanation', '')}"

    return parsed


def run_sentiment(client: anthropic.Anthropic, transcript: str) -> dict:
    raw = _call_llm(client, _sentiment_prompt(transcript), "sentiment")
    parsed = _parse_json_response(raw, "sentiment")
    return parsed


def run_resolution(client: anthropic.Anthropic, transcript: str) -> dict:
    raw = _call_llm(client, _resolution_prompt(transcript), "resolution")
    parsed = _parse_json_response(raw, "resolution")

    # Coerce confidence to float in [0, 1]
    confidence = float(parsed.get("confidence", 0.5))
    parsed["confidence"] = max(0.0, min(1.0, confidence))

    return parsed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    transcript: str,
    transcript_id: str,
    client: Optional[anthropic.Anthropic] = None,
) -> PipelineResult:
    """
    Run all four inference tasks sequentially on a single transcript.

    Args:
        transcript:     Raw call transcript text (speaker-labeled preferred)
        transcript_id:  Unique identifier for this transcript
        client:         Anthropic client instance (created from env if not provided)

    Returns:
        PipelineResult dataclass with all structured outputs
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    start = time.time()
    raw_outputs = {}

    # --- Task 1: Summary ---
    summary = run_summary(client, transcript)
    raw_outputs["summary"] = summary

    # --- Task 2: Root cause category ---
    category_result = run_category(client, transcript)
    raw_outputs["category"] = category_result

    # --- Task 3: Sentiment arc ---
    sentiment_result = run_sentiment(client, transcript)
    raw_outputs["sentiment"] = sentiment_result

    # --- Task 4: Resolution flag ---
    resolution_result = run_resolution(client, transcript)
    raw_outputs["resolution"] = resolution_result

    latency = round(time.time() - start, 2)

    return PipelineResult(
        transcript_id=transcript_id,
        summary=summary,
        root_cause_category=category_result["category"],
        root_cause_explanation=category_result["explanation"],
        sentiment_arc=SentimentArc(
            beginning=sentiment_result["beginning"],
            middle=sentiment_result["middle"],
            end=sentiment_result["end"],
        ),
        resolution_flag=bool(resolution_result["resolved"]),
        resolution_confidence=resolution_result["confidence"],
        resolution_notes=resolution_result["notes"],
        model=MODEL,
        latency_seconds=latency,
        raw_outputs=raw_outputs,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    transcripts: list[dict],  # each dict: {"id": str, "text": str}
    output_path: str = "outputs/results.json",
    sleep_between: float = 0.5,  # basic rate limit courtesy
) -> list[dict]:
    """
    Run pipeline over a list of transcripts and write results to JSON.

    Args:
        transcripts:    List of {"id": ..., "text": ...} dicts
        output_path:    Where to write the output JSON
        sleep_between:  Seconds to sleep between API calls (avoid rate limits)

    Returns:
        List of result dicts
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []
    errors = []

    for i, item in enumerate(transcripts):
        transcript_id = item["id"]
        transcript_text = item["text"]

        print(f"[{i+1}/{len(transcripts)}] Processing transcript: {transcript_id}")

        try:
            result = run_pipeline(transcript_text, transcript_id, client)
            results.append(result.to_dict())
            print(f"  ✓ Done in {result.latency_seconds}s | "
                  f"Category: {result.root_cause_category} | "
                  f"Resolved: {result.resolution_flag}")
        except Exception as e:
            print(f"  ✗ Error on {transcript_id}: {e}")
            errors.append({"id": transcript_id, "error": str(e)})

        if i < len(transcripts) - 1:
            time.sleep(sleep_between)

    # Write results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)

    print(f"\nDone. {len(results)} succeeded, {len(errors)} failed.")
    print(f"Output written to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_TRANSCRIPT = """
    Agent: Thank you for calling Capital One, this is Marcus. How can I help you today?
    Customer: Hi Marcus, I'm calling because there's a charge on my account I don't recognize. 
    It's from a place called "Streamline Media" for $49.99 and I've never heard of them.
    Agent: I'm sorry to hear that. Let me pull up your account. Can I get the last four digits of your card?
    Customer: Sure, it's 4821.
    Agent: Thank you. I can see the charge from March 18th. Have you tried contacting the merchant directly?
    Customer: No, I don't even know who they are. This feels like fraud to me.
    Agent: I understand your concern. I'm going to flag this as a disputed charge and initiate a 
    provisional credit to your account within 1-2 business days while we investigate.
    Customer: Okay, that's good. Will I get a new card number?
    Agent: Yes, I'll go ahead and order a replacement card now. You should receive it in 5-7 business days.
    Customer: Alright, thank you. That makes me feel better.
    Agent: Of course. Is there anything else I can help you with today?
    Customer: No, that's everything. Thanks Marcus.
    Agent: Thank you for calling Capital One. Have a great day.
    """

    result = run_pipeline(
        transcript=SAMPLE_TRANSCRIPT,
        transcript_id="smoke_test_001",
    )

    print("\n--- Pipeline Result ---")
    print(json.dumps(result.to_dict(), indent=2))