"""
RAG Answer Node
----------------
LLM call on top of retrieved chunks, with Pydantic-validated structured output.

Sits on top of rag_qa.py — retrieval is already done by the time this runs.
This module only handles: context formatting → LLM call → validated response.

Why Pydantic here:
    LLM outputs are strings. When you ask the model to return JSON, it usually
    does — but "usually" isn't good enough for a production system. The model
    might return extra fields, missing fields, wrong types, or JSON wrapped in
    markdown code fences. Pydantic validates the parsed output against a schema
    and raises a clear error when something is wrong, rather than letting a
    malformed response propagate silently downstream.

    This is the same problem dataclasses solve for internal data structures,
    but Pydantic adds runtime type coercion and validation on top — important
    when the data is coming from an external, non-deterministic source.

Pydantic vs dataclass:
    dataclass  — defines structure, Python doesn't enforce types at runtime
    Pydantic   — defines structure AND validates/coerces types at runtime

    Use dataclass for internal data you control.
    Use Pydantic for data coming from outside your code — LLM outputs,
    API responses, user input, config files.
"""

import json
import os
from typing import Optional
import anthropic
from pydantic import BaseModel, Field, field_validator, ValidationError

from rag_qa import (
    RetrievalResult,
    RetrievedChunk,
    build_context,
    retrieve,
    load_index,
    load_embedding_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """
    A single source citation.

    Pydantic BaseModel is similar to a dataclass but with runtime validation.
    Field() adds metadata: description for the LLM prompt, and constraints
    like min_length that Pydantic enforces when parsing.
    """
    source_number: int = Field(
        description="The [Source N] number as it appears in the answer text"
    )
    transcript_id: str = Field(
        description="ID of the source transcript"
    )
    speaker: str = Field(
        description="Speaker of the cited turn: Agent or Customer"
    )
    turn_index: int = Field(
        description="Turn position within the transcript (0-indexed)"
    )
    relevant_quote: str = Field(
        description="Brief verbatim quote from the source that supports the answer",
        min_length=5,
    )

    @field_validator("speaker")
    @classmethod
    def speaker_must_be_valid(cls, v: str) -> str:
        """
        field_validator runs when a Citation is created from parsed JSON.
        If the LLM returns speaker="agent" (lowercase), this normalizes it.
        If it returns something invalid entirely, it raises ValidationError.
        """
        normalized = v.strip().title()     # "agent" → "Agent"
        if normalized not in ("Agent", "Customer"):
            raise ValueError(
                f"Invalid speaker '{v}'. Must be 'Agent' or 'Customer'."
            )
        return normalized


class RAGAnswer(BaseModel):
    """
    Structured answer with inline citations and a sources list.

    The answer field uses [Source N] inline references that correspond
    to entries in the citations list — same pattern as academic citations.

    Example:
        answer: "Fraud disputes are typically handled by flagging the charge
                 and issuing a provisional credit [Source 1]. Agents generally
                 advise customers to expect resolution within 5-7 days [Source 2]."
        citations: [
            Citation(source_number=1, transcript_id="synthetic_0003", ...),
            Citation(source_number=2, transcript_id="synthetic_0011", ...),
        ]
    """
    answer: str = Field(
        description=(
            "Answer to the question in 2-5 sentences. "
            "Reference sources inline as [Source 1], [Source 2], etc."
        ),
        min_length=20,
    )
    citations: list[Citation] = Field(
        description="Structured list of sources cited in the answer",
        min_length=1,
    )
    confidence: float = Field(
        description=(
            "Confidence that the retrieved context adequately answers "
            "the question. Float between 0.0 and 1.0."
        ),
        ge=0.0,   # Pydantic constraint: greater than or equal to 0.0
        le=1.0,   # Pydantic constraint: less than or equal to 1.0
    )
    coverage_note: Optional[str] = Field(
        default=None,
        description=(
            "Optional note if the retrieved context was insufficient "
            "to fully answer the question."
        ),
    )

    @field_validator("answer")
    @classmethod
    def answer_must_contain_citation(cls, v: str) -> str:
        """
        Validate that the answer actually uses inline citations.
        An answer without [Source N] references ignored the context entirely.
        """
        if "[Source" not in v:
            raise ValueError(
                "Answer must contain at least one inline citation [Source N]. "
                "Ground your answer in the retrieved context."
            )
        return v


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _answer_prompt(question: str, context: str) -> str:
    """
    Instruct the LLM to answer from context with inline citations.

    The schema description is derived from the Pydantic model fields,
    keeping the prompt and the validation schema in sync — if you add
    a field to RAGAnswer, the prompt needs to reflect it.
    """
    return f"""You are analyzing customer service call transcripts to answer questions.

Answer the question using ONLY the provided context. Do not use outside knowledge.

Each context chunk is labeled: [Source N: transcript_id, Speaker, turn position]

Question:
{question}

Context:
{context}

Instructions:
- Answer in 2-5 sentences
- Reference sources inline as [Source 1], [Source 2], etc.
- Every factual claim must be grounded in a source
- If the context is insufficient, say so clearly and set confidence low

Respond in this exact JSON format (no markdown, no code fences):
{{
  "answer": "<2-5 sentences with inline [Source N] references>",
  "citations": [
    {{
      "source_number": <integer>,
      "transcript_id": "<transcript id>",
      "speaker": "Agent" or "Customer",
      "turn_index": <integer>,
      "relevant_quote": "<brief verbatim quote from that source>"
    }}
  ],
  "confidence": <float 0.0-1.0>,
  "coverage_note": "<optional note if context was insufficient, else null>"
}}"""


# ---------------------------------------------------------------------------
# Answer node
# ---------------------------------------------------------------------------

def answer(
    question: str,
    retrieval_result: RetrievalResult,
    client: Optional[anthropic.Anthropic] = None,
) -> RAGAnswer:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Args:
        question:           The original question
        retrieval_result:   Output of rag_qa.retrieve()
        client:             Anthropic client (created from env if not provided)

    Returns:
        RAGAnswer — Pydantic-validated structured response

    Raises:
        ValidationError:    If LLM output doesn't match the schema
        ValueError:         If LLM output isn't parseable JSON
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    context = build_context(retrieval_result.retrieved)
    prompt = _answer_prompt(question, context)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    cleaned = (raw
               .removeprefix("```json")
               .removeprefix("```")
               .removesuffix("```")
               .strip())

    # Parse JSON
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned non-JSON output.\nRaw: {raw}\nError: {e}"
        ) from e

    # Pydantic validation
    # This is where Pydantic earns its place: if the LLM returns confidence=1.5,
    # or omits citations, or returns speaker="agent", Pydantic catches it here
    # with a clear error rather than letting it propagate silently.
    try:
        validated = RAGAnswer(**parsed)
    except ValidationError as e:
        raise ValidationError(
            f"LLM output failed schema validation.\nRaw: {raw}\nErrors: {e}"
        ) from e

    return validated


def format_answer(result: RAGAnswer) -> str:
    """Human-readable formatting of a RAGAnswer."""
    lines = [
        f"Answer: {result.answer}",
        f"Confidence: {result.confidence:.0%}",
    ]

    if result.coverage_note:
        lines.append(f"Note: {result.coverage_note}")

    lines.append("\nSources:")
    for c in result.citations:
        lines.append(
            f"  [Source {c.source_number}] {c.transcript_id} | "
            f"{c.speaker} | turn {c.turn_index}"
        )
        lines.append(f"    \"{c.relevant_quote}\"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full RAG pipeline (retrieval + answer)
# ---------------------------------------------------------------------------

def ask(
    question: str,
    top_k: int = 5,
    speaker_filter: Optional[str] = None,
) -> RAGAnswer:
    """
    End-to-end RAG: question → retrieve → answer.
    Loads index and model from disk — no setup required if build_full_index
    has already been run.

    Args:
        question:       Natural language question
        top_k:          Number of chunks to retrieve
        speaker_filter: Optionally restrict to "Agent" or "Customer" turns

    Returns:
        Validated RAGAnswer
    """
    collection = load_index()
    model = load_embedding_model()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    retrieval_result = retrieve(
        question=question,
        collection=collection,
        model=model,
        top_k=top_k,
        speaker_filter=speaker_filter,
    )

    return answer(
        question=question,
        retrieval_result=retrieval_result,
        client=client,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_questions = [
        "What are the most common reasons customers call about fraud?",
        "How do agents typically resolve billing disputes?",
        "What do agents say when they can't immediately resolve an issue?",
    ]

    for question in test_questions:
        print("=" * 60)
        print(f"Q: {question}")
        print("=" * 60)

        try:
            result = ask(question, top_k=5)
            print(format_answer(result))
        except Exception as e:
            print(f"Error: {e}")

        print()