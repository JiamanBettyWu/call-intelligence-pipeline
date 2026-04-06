"""
Transcript Ingestion & Preprocessing
--------------------------------------
Loads raw call transcripts from various formats, normalizes speaker turns,
and prepares them for the pipeline.

Design notes:
  - Handles both real transcript files (txt, json, csv) and synthetic generation
    via LLM — because real call transcripts are PII-sensitive, synthetic data
    is often the only viable option for a portfolio project, and generating it
    deliberately is itself a talking point.
  - Speaker normalization produces a consistent "Agent: / Customer:" format
    regardless of input variation ("Rep:", "CSR:", "Caller:", etc.)
  - Basic quality checks flag transcripts that are too short or malformed
    before they hit the pipeline.
"""

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import anthropic

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"

MIN_TRANSCRIPT_WORDS = 50   # flag anything shorter as suspect
MAX_TRANSCRIPT_CHARS = 8000  # rough context window guard

# Patterns for common agent speaker labels (case-insensitive)
AGENT_ALIASES = re.compile(
    r"^(agent|rep|representative|csr|advisor|specialist|associate|support|"
    r"marcus|sarah|james|emily|alex|operator)\s*:",
    re.IGNORECASE,
)
CUSTOMER_ALIASES = re.compile(
    r"^(customer|caller|client|user|member|guest)\s*:",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Transcript:
    id: str
    text: str                           # normalized full text
    turns: list[dict]                   # [{"speaker": "Agent"|"Customer", "text": str}]
    word_count: int
    source: str                         # "file" | "synthetic" | "csv" | "json"
    warnings: list[str]                 # quality flags


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_speaker_turns(raw_text: str) -> tuple[str, list[dict]]:
    """
    Parse raw transcript text into normalized speaker turns.

    Accepts messy formats like:
        "Rep: Hello..."
        "CUSTOMER: I have a problem..."
        "Agent (Marcus): ..."

    Returns:
        normalized_text: clean "Agent: / Customer:" format
        turns: list of {"speaker": ..., "text": ...} dicts
    """
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    turns = []
    current_speaker = None
    current_text = []

    for line in lines:
        # Strip parenthetical names: "Agent (Marcus):" → "Agent:"
        line = re.sub(r"\(.*?\)", "", line).strip()

        if AGENT_ALIASES.match(line):
            if current_speaker:
                turns.append({"speaker": current_speaker,
                               "text": " ".join(current_text).strip()})
            current_speaker = "Agent"
            current_text = [AGENT_ALIASES.sub("", line).strip()]

        elif CUSTOMER_ALIASES.match(line):
            if current_speaker:
                turns.append({"speaker": current_speaker,
                               "text": " ".join(current_text).strip()})
            current_speaker = "Customer"
            current_text = [CUSTOMER_ALIASES.sub("", line).strip()]

        else:
            # Continuation of previous speaker
            if current_speaker:
                current_text.append(line)

    # Flush last turn
    if current_speaker and current_text:
        turns.append({"speaker": current_speaker,
                      "text": " ".join(current_text).strip()})

    # Rebuild normalized text
    normalized = "\n".join(
        f"{t['speaker']}: {t['text']}" for t in turns if t["text"]
    )

    return normalized, turns


def _quality_check(transcript: Transcript) -> Transcript:
    """Add warnings for transcripts that may cause pipeline issues."""
    if transcript.word_count < MIN_TRANSCRIPT_WORDS:
        transcript.warnings.append(
            f"Short transcript ({transcript.word_count} words). "
            f"Pipeline output may be unreliable."
        )
    if len(transcript.text) > MAX_TRANSCRIPT_CHARS:
        transcript.warnings.append(
            f"Long transcript ({len(transcript.text)} chars). "
            f"Consider chunking before inference."
        )
    if not any(t["speaker"] == "Customer" for t in transcript.turns):
        transcript.warnings.append("No customer turns detected. Check speaker normalization.")
    if not any(t["speaker"] == "Agent" for t in transcript.turns):
        transcript.warnings.append("No agent turns detected. Check speaker normalization.")
    return transcript


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_txt(file_path: str, transcript_id: Optional[str] = None) -> Transcript:
    """Load a single .txt transcript file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    raw = path.read_text(encoding="utf-8")
    tid = transcript_id or path.stem
    normalized, turns = normalize_speaker_turns(raw)

    t = Transcript(
        id=tid,
        text=normalized,
        turns=turns,
        word_count=len(normalized.split()),
        source="file",
        warnings=[],
    )
    return _quality_check(t)


def load_from_json(file_path: str) -> list[Transcript]:
    """
    Load transcripts from a JSON file.

    Expected format:
        [{"id": "...", "text": "..."}, ...]
    or
        [{"id": "...", "turns": [{"speaker": "...", "text": "..."}]}, ...]
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    transcripts = []
    for item in data:
        tid = item.get("id", f"transcript_{len(transcripts):04d}")

        if "text" in item:
            raw = item["text"]
        elif "turns" in item:
            # Pre-formatted turns — reconstruct raw text and re-normalize
            raw = "\n".join(
                f"{t['speaker']}: {t['text']}" for t in item["turns"]
            )
        else:
            raise ValueError(f"Item {tid} has neither 'text' nor 'turns' field.")

        normalized, turns = normalize_speaker_turns(raw)
        t = Transcript(
            id=tid,
            text=normalized,
            turns=turns,
            word_count=len(normalized.split()),
            source="json",
            warnings=[],
        )
        transcripts.append(_quality_check(t))

    return transcripts


def load_from_csv(file_path: str,
                  id_col: str = "id",
                  text_col: str = "transcript") -> list[Transcript]:
    """
    Load transcripts from a CSV file.

    Args:
        file_path:  Path to CSV
        id_col:     Column name for transcript IDs
        text_col:   Column name for transcript text
    """
    transcripts = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tid = row.get(id_col, f"row_{i:04d}")
            raw = row.get(text_col, "")
            if not raw:
                print(f"  Warning: empty transcript at row {i}, skipping.")
                continue

            normalized, turns = normalize_speaker_turns(raw)
            t = Transcript(
                id=tid,
                text=normalized,
                turns=turns,
                word_count=len(normalized.split()),
                source="csv",
                warnings=[],
            )
            transcripts.append(_quality_check(t))

    return transcripts


def load_from_directory(dir_path: str, extension: str = ".txt") -> list[Transcript]:
    """Load all transcript files from a directory."""
    path = Path(dir_path)
    files = sorted(path.glob(f"*{extension}"))
    if not files:
        raise FileNotFoundError(f"No {extension} files found in {dir_path}")

    transcripts = []
    for f in files:
        try:
            t = load_from_txt(str(f))
            transcripts.append(t)
        except Exception as e:
            print(f"  Warning: could not load {f.name}: {e}")

    return transcripts


# ---------------------------------------------------------------------------
# Synthetic transcript generation
# ---------------------------------------------------------------------------

COMPLAINT_SCENARIOS = [
    ("billing_dispute",            "Customer disputes a recurring charge they didn't authorize"),
    ("fraud_or_unauthorized_charge","Customer reports a fraudulent transaction on their card"),
    ("product_or_feature_issue",   "Customer can't access a feature they've been using for months"),
    ("agent_error_or_misinformation","Customer was given wrong information by a previous agent"),
    ("technical_issue",            "Customer's online account is locked and they can't log in"),
    ("policy_disagreement",        "Customer is upset about a late fee they think should be waived"),
    ("account_access",             "Customer can't remember security questions and is locked out"),
    ("other",                      "Customer has a general inquiry that escalates into a complaint"),
]

RESOLUTION_VARIANTS = ["resolved cleanly", "partially resolved", "unresolved — escalated"]


def _synthetic_generation_prompt(scenario: tuple[str, str], resolution: str) -> str:
    category, description = scenario
    return f"""Generate a realistic customer service call transcript for a credit card company.

Scenario: {description}
Root cause category: {category}
Resolution: {resolution}

Requirements:
- 15-25 speaker turns total
- Realistic dialogue — not perfectly polite, some frustration is natural
- Agent and Customer labels only (no names in speaker labels)
- Include specific details: dollar amounts, dates, account numbers (fake), product names
- The resolution should be clear from how the call ends
- No meta-commentary, just the transcript

Format each line as:
Agent: <text>
Customer: <text>

Start with the agent greeting."""


def generate_synthetic_transcripts(
    n: int = 30,
    output_dir: str = "data/raw_transcripts",
    output_json: str = "data/synthetic_transcripts.json",
    sleep_between: float = 1.0,
) -> list[Transcript]:
    """
    Generate n synthetic call transcripts covering all scenario/resolution combinations.

    Args:
        n:              Number of transcripts to generate
        output_dir:     Where to save individual .txt files
        output_json:    Where to save the combined JSON
        sleep_between:  Seconds between API calls

    Returns:
        List of Transcript objects
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    os.makedirs(output_dir, exist_ok=True)

    transcripts = []
    scenario_pool = COMPLAINT_SCENARIOS * 10    # repeat to fill n
    resolution_pool = RESOLUTION_VARIANTS * 10

    print(f"Generating {n} synthetic transcripts...")

    for i in range(n):
        scenario = scenario_pool[i % len(COMPLAINT_SCENARIOS)]
        resolution = resolution_pool[i % len(RESOLUTION_VARIANTS)]
        tid = f"synthetic_{i:04d}"

        prompt = _synthetic_generation_prompt(scenario, resolution)

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1500,
                output_config={
                    "effort": "low"  
                },
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Save raw .txt
            txt_path = os.path.join(output_dir, f"{tid}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw)

            normalized, turns = normalize_speaker_turns(raw)
            t = Transcript(
                id=tid,
                text=normalized,
                turns=turns,
                word_count=len(normalized.split()),
                source="synthetic",
                warnings=[],
            )
            t = _quality_check(t)
            transcripts.append(t)

            print(f"  [{i+1}/{n}] {tid} — {scenario[0]} / {resolution} "
                  f"({t.word_count} words)")

            if t.warnings:
                for w in t.warnings:
                    print(f"    ⚠ {w}")

        except Exception as e:
            print(f"  ✗ Failed to generate {tid}: {e}")

        if i < n - 1:
            time.sleep(sleep_between)

    # Save combined JSON
    payload = [{"id": t.id, "text": t.text, "source": t.source,
                "warnings": t.warnings} for t in transcripts]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nGenerated {len(transcripts)} transcripts → {output_json}")
    return transcripts


# ---------------------------------------------------------------------------
# Unified loader (entry point for pipeline)
# ---------------------------------------------------------------------------

def load_transcripts(source: str, **kwargs) -> list[Transcript]:
    """
    Unified loader. Detects source type and dispatches accordingly.

    Args:
        source: One of "synthetic", path to .json, .csv, .txt, or directory

    Examples:
        load_transcripts("synthetic", n=30)
        load_transcripts("data/transcripts.json")
        load_transcripts("data/transcripts.csv", id_col="call_id", text_col="body")
        load_transcripts("data/raw_transcripts/")   # directory of .txt files
    """
    if source == "synthetic":
        return generate_synthetic_transcripts(**kwargs)

    path = Path(source)

    if path.is_dir():
        return load_from_directory(str(path), **kwargs)
    elif path.suffix == ".json":
        return load_from_json(str(path))
    elif path.suffix == ".csv":
        return load_from_csv(str(path), **kwargs)
    elif path.suffix == ".txt":
        return [load_from_txt(str(path))]
    else:
        raise ValueError(f"Unsupported source: {source}. "
                         f"Use 'synthetic', a .json/.csv/.txt path, or a directory.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """
    Rep (James): Thanks for calling, how can I help you today?
    Caller: Yeah hi, I got charged twice for the same transaction and nobody's helping me.
    Rep (James): I'm sorry about that. Can I get your account number?
    Caller: It's 4821. The charge is from March 12th, $87.50, and it shows up twice.
    Rep (James): I see both charges here. I'll initiate a dispute on the duplicate right now.
    Caller: How long is this going to take?
    Rep (James): You'll see a provisional credit within 24 hours while we investigate.
    Caller: Fine. That's all I needed.
    Rep (James): Of course, is there anything else I can help with?
    Caller: No.
    """

    normalized, turns = normalize_speaker_turns(SAMPLE)
    print("--- Normalized transcript ---")
    print(normalized)
    print(f"\n{len(turns)} turns detected")
    print(turns)

    t = Transcript(
        id="smoke_test",
        text=normalized,
        turns=turns,
        word_count=len(normalized.split()),
        source="test",
        warnings=[],
    )
    t = _quality_check(t)
    if t.warnings:
        print("\nWarnings:", t.warnings)
    else:
        print("\nNo quality warnings.")