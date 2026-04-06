import sys
sys.path.insert(0, "src")

from ingest import load_transcripts
from pipeline import run_batch
from evaluate import run_full_eval

if __name__ == "__main__":
    transcripts = load_transcripts("data/synthetic_transcripts.json")
    run_batch([{"id": t.id, "text": t.text} for t in transcripts])

    report = run_full_eval()
