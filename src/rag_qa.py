"""
RAG Q&A — Multi-Transcript Retrieval
--------------------------------------
Given a question, retrieve the most relevant passages across all indexed
transcripts and answer from them with source attribution.

Architecture:
    question
        │
        ▼
    retrieve — semantic search across chunked + indexed transcript corpus
        │
        ▼
    answer — LLM call with retrieved chunks as context  [tomorrow]
        │
        ▼
    response with source attribution (transcript_id, speaker, chunk position)


Design decisions:
    - Chunk by speaker turn, not arbitrary token windows. Speaker turns are
      natural semantic units in conversational data — a customer explaining
      their problem, an agent describing a resolution. Splitting mid-turn
      would break the meaning. This is specific to call transcript data and
      worth explaining to a hiring manager.

    - ChromaDB for the vector store. Stores embeddings alongside metadata
      (transcript_id, speaker, turn_index, chunk_text) in a persistent local
      directory. Source attribution comes for free — every retrieved chunk
      knows exactly where it came from.

    - sentence-transformers all-MiniLM-L6-v2 for embeddings. Fast, local,
      no API key required. 384-dimensional embeddings, good general semantic
      quality for English conversational text. The tradeoff vs. a hosted
      embedding API (e.g. OpenAI text-embedding-3-small) is quality vs.
      cost/simplicity — for a portfolio project, local wins.

    - Minimum chunk length filter. Very short turns ("Yes.", "I see.",
      "Hold on.") add noise to the index without meaningful semantic content.
      Filtered out at index time, not query time.
"""

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, no API key
CHROMA_DIR = "data/chroma_db"           # persistent local storage
COLLECTION_NAME = "call_transcripts"
MIN_CHUNK_WORDS = 8                     # filter trivially short turns
TOP_K_DEFAULT = 5                       # chunks returned per query

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str           # "{transcript_id}_turn_{turn_index}"
    transcript_id: str
    speaker: str            # "Agent" or "Customer"
    turn_index: int         # position in the conversation (0-indexed)
    text: str
    word_count: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float            # cosine similarity (higher = more relevant)


@dataclass
class RetrievalResult:
    question: str
    retrieved: list[RetrievedChunk]
    n_chunks_searched: int


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(transcript_id: str, transcript_text: str) -> list[Chunk]:
    """
    Split a transcript into chunks by speaker turn.

    Each speaker turn becomes one chunk. Turns shorter than MIN_CHUNK_WORDS
    are filtered out — "Yes.", "Hold on.", "One moment." add noise without
    semantic content.

    Args:
        transcript_id:      Unique ID for this transcript
        transcript_text:    Normalized transcript text ("Agent: ...\nCustomer: ...")

    Returns:
        List of Chunk objects, one per substantive speaker turn
    """
    chunks = []
    turn_index = 0

    for line in transcript_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # Parse speaker and text from "Speaker: content" format
        if ":" not in line:
            continue

        speaker, _, text = line.partition(":")
        speaker = speaker.strip()
        text = text.strip()

        if not text:
            continue

        word_count = len(text.split())
        if word_count < MIN_CHUNK_WORDS:
            turn_index += 1
            continue

        chunks.append(Chunk(
            chunk_id=f"{transcript_id}_turn_{turn_index:04d}",
            transcript_id=transcript_id,
            speaker=speaker,
            turn_index=turn_index,
            text=text,
            word_count=word_count,
        ))
        turn_index += 1

    return chunks


def chunk_all_transcripts(transcripts: list[dict]) -> list[Chunk]:
    """
    Chunk a list of transcripts.

    Args:
        transcripts:    List of {"id": str, "text": str} dicts
                        (same format as pipeline.py run_batch input)

    Returns:
        Flat list of all chunks across all transcripts
    """
    all_chunks = []
    for t in transcripts:
        chunks = chunk_transcript(t["id"], t["text"])
        all_chunks.extend(chunks)
        print(f"  {t['id']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)} across {len(transcripts)} transcripts")
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load sentence-transformers model.
    Downloads on first use (~90MB), cached locally afterward.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: list[Chunk],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> list[list[float]]:
    """
    Embed a list of chunks using sentence-transformers.

    Args:
        chunks:     Chunk objects to embed
        model:      Loaded SentenceTransformer model
        batch_size: Embedding batch size (tune for memory)

    Returns:
        List of embedding vectors, one per chunk
    """
    texts = [chunk.text for chunk in chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


# ---------------------------------------------------------------------------
# ChromaDB index
# ---------------------------------------------------------------------------

def get_chroma_client(persist_dir: str = CHROMA_DIR) -> chromadb.PersistentClient:
    """
    Get or create a persistent ChromaDB client.
    Data is stored in persist_dir and survives process restarts.
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def build_index(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    overwrite: bool = False,
) -> chromadb.Collection:
    """
    Build (or rebuild) the ChromaDB vector index from chunks and embeddings.

    ChromaDB stores three things per chunk:
        - The embedding vector (for similarity search)
        - The metadata dict (transcript_id, speaker, turn_index, word_count)
        - The document text (returned with results, no need to look up separately)

    Args:
        chunks:             Chunk objects
        embeddings:         Corresponding embedding vectors
        persist_dir:        Where ChromaDB persists to disk
        collection_name:    Name of the ChromaDB collection
        overwrite:          If True, delete and recreate the collection

    Returns:
        ChromaDB Collection object
    """
    client = get_chroma_client(persist_dir)

    # Handle existing collection
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        if overwrite:
            print(f"Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists. "
                  f"Use overwrite=True to rebuild.")
            return client.get_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    print(f"Indexing {len(chunks)} chunks into '{collection_name}'...")

    # ChromaDB add() accepts batches — use batches of 500 to stay within limits
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c.text for c in batch_chunks],
            metadatas=[{
                "transcript_id": c.transcript_id,
                "speaker":       c.speaker,
                "turn_index":    c.turn_index,
                "word_count":    c.word_count,
            } for c in batch_chunks],
        )

        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"Index built. Total documents: {collection.count()}")
    return collection


def load_index(
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk.
    Raises if the collection doesn't exist — run build_index first.
    """
    client = get_chroma_client(persist_dir)
    existing = [c.name for c in client.list_collections()]

    if collection_name not in existing:
        raise FileNotFoundError(
            f"Collection '{collection_name}' not found in {persist_dir}. "
            f"Run build_index() first."
        )

    collection = client.get_collection(collection_name)
    print(f"Loaded index: {collection.count()} chunks in '{collection_name}'")
    return collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = TOP_K_DEFAULT,
    speaker_filter: Optional[str] = None,  # "Agent", "Customer", or None for both
) -> RetrievalResult:
    """
    Retrieve the most relevant chunks for a question.

    Args:
        question:       Natural language question
        collection:     ChromaDB collection to search
        model:          SentenceTransformer model (same one used for indexing)
        top_k:          Number of chunks to return
        speaker_filter: Optionally restrict to Agent or Customer turns

    Returns:
        RetrievalResult with ranked chunks and source metadata
    """
    # Embed the question using the same model as the index
    query_embedding = model.encode(question).tolist()

    # Build optional metadata filter
    where = None
    if speaker_filter:
        where = {"speaker": {"$eq": speaker_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB response structure
    # results["ids"][0], results["documents"][0], etc. are lists
    # (outer list is per query — we only have one query)
    retrieved = []
    for chunk_id, doc, meta, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunk = Chunk(
            chunk_id=chunk_id,
            transcript_id=meta["transcript_id"],
            speaker=meta["speaker"],
            turn_index=meta["turn_index"],
            text=doc,
            word_count=meta["word_count"],
        )
        # ChromaDB cosine distance is 1 - similarity, so convert to similarity
        similarity = round(1 - distance, 4)
        retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))

    return RetrievalResult(
        question=question,
        retrieved=retrieved,
        n_chunks_searched=collection.count(),
    )


def format_retrieval_result(result: RetrievalResult) -> str:
    """
    Human-readable formatting of retrieval results.
    Used for debugging and as context builder for the LLM answer node (tomorrow).
    """
    lines = [
        f"Question: {result.question}",
        f"Retrieved {len(result.retrieved)} chunks "
        f"from {result.n_chunks_searched} total\n",
    ]

    for i, rc in enumerate(result.retrieved, 1):
        lines += [
            f"[{i}] {rc.chunk.transcript_id} | "
            f"{rc.chunk.speaker} | "
            f"turn {rc.chunk.turn_index} | "
            f"score {rc.score:.4f}",
            f"    {rc.chunk.text}",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build context string for LLM 
# ---------------------------------------------------------------------------

def build_context(retrieved: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a context string for the LLM answer node.
    Each chunk is labeled with its source for attribution in the answer.

    This function is used by the answer node tomorrow — defined here so
    the retrieval and answer layers share the same context format.
    """
    lines = []
    for i, rc in enumerate(retrieved, 1):
        lines.append(
            f"[Source {i}: {rc.chunk.transcript_id}, "
            f"{rc.chunk.speaker}, turn {rc.chunk.turn_index}]\n"
            f"{rc.chunk.text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Full index build pipeline 
# ---------------------------------------------------------------------------

def build_full_index(
    transcripts_path: str = "data/synthetic_transcripts.json",
    persist_dir: str = CHROMA_DIR,
    overwrite: bool = False,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    End-to-end: load transcripts → chunk → embed → index.

    Args:
        transcripts_path:   Path to synthetic_transcripts.json
        persist_dir:        Where to persist ChromaDB
        overwrite:          Rebuild index even if it already exists

    Returns:
        (collection, model) — both needed for retrieval
    """
    # Load transcripts
    with open(transcripts_path) as f:
        transcripts = json.load(f)

    print(f"Loaded {len(transcripts)} transcripts from {transcripts_path}\n")

    # Chunk
    print("--- Chunking ---")
    chunks = chunk_all_transcripts(transcripts)

    # Embed
    print("\n--- Embedding ---")
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)

    # Index
    print("\n--- Indexing ---")
    collection = build_index(chunks, embeddings, persist_dir, overwrite=overwrite)

    return collection, model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # --- Step 1: Build index ---
    print("=" * 60)
    print("BUILDING INDEX")
    print("=" * 60)
    collection, model = build_full_index(overwrite=True)

    # --- Step 2: Test retrieval with a few questions ---
    print("\n" + "=" * 60)
    print("RETRIEVAL SMOKE TEST")
    print("=" * 60)

    test_questions = [
        "What are the most common reasons customers call about fraud?",
        "How do agents handle billing disputes?",
        "Which calls end without the customer's issue being resolved?",
        "What do agents say when they escalate a case?",
    ]

    for question in test_questions:
        print(f"\n{'─' * 60}")
        result = retrieve(question, collection, model, top_k=3)
        print(format_retrieval_result(result))

    # --- Step 3: Show what the context string looks like for the LLM ---
    print("=" * 60)
    print("CONTEXT STRING FOR LLM ANSWER NODE (preview)")
    print("=" * 60)
    sample_result = retrieve(test_questions[0], collection, model, top_k=3)
    print(build_context(sample_result.retrieved))