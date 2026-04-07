import sys
sys.path.insert(0, "src")
from pipeline_graph import run_batch_graph


from ingest import load_transcripts
from pipeline import run_batch
from evaluate import run_full_eval

if __name__ == "__main__":
    transcripts = load_transcripts("data/synthetic_transcripts.json")

    # Run the standard pipeline and save results to a separate file for analysis.
    run_batch([{"id": t.id, "text": t.text} for t in transcripts])

    # Run the graph pipeline and save results to a separate file for analysis.
    run_batch_graph([{"id": t.id, "text": t.text} for t in transcripts],
                    output_path="outputs/results_graph.json")
    
    # run evaluation over the graph pipeline results
    report = run_full_eval()
