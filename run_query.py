"""Quick script to run a RAG query and display top-5 results with scores."""
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.rag.pipeline import RAGPipeline

QUERY = "voltage sag protection switching procedure"

def main():
    pipeline = RAGPipeline()
    pipeline.initialize()

    stats = pipeline.get_stats()
    print(f"\n{'='*80}")
    print(f"Pipeline Stats: {stats['num_chunks']} chunks indexed | "
          f"embedding={stats['embedding_model']} | reranker={stats['reranker_model']}")
    print(f"{'='*80}")
    print(f"\nQuery: \"{QUERY}\"\n")

    results = pipeline.query_with_scores(QUERY, top_k=5)

    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        print(f"\n{'-'*80}")
        print(f"  Rank #{i}  |  Score: {score:.4f}  |  Source: {source}")
        if section:
            print(f"  Section: {section}")
        if chunk_id:
            print(f"  Chunk ID: {chunk_id}")
        print(f"{'-'*80}")
        # Show first 400 chars of content
        content = doc.page_content
        if len(content) > 400:
            print(f"  {content[:400]}...")
        else:
            print(f"  {content}")

    print(f"\n{'='*80}")
    print(f"Done -- {len(results)} results returned.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
