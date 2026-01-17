import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

load_dotenv()

def fit_bm25_from_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("resume-index")

    print("Fetching all namespaces...")
    stats = index.describe_index_stats()
    namespaces = list(stats.get('namespaces', {}).keys())

    if not namespaces:
        print(" No namespaces found in index. Have you ingested any resumes?")
        return

    print(f"Found {len(namespaces)} namespaces (resumes).")

    corpus = []
    dummy_vector = [0.0] * 768

    for ns in namespaces:
        print(f"  Fetching chunks from namespace: {ns}...")
        try:
            # Fetch as many as possible (serverless usually allows high top_k)
            results = index.query(
                vector=dummy_vector,
                top_k=10000,                  
                include_metadata=True,
                namespace=ns
            )

            texts = [
                match['metadata']['text']
                for match in results['matches']
                if 'metadata' in match and 'text' in match['metadata']
            ]

            corpus.extend(texts)
            print(f"Got {len(texts)} chunks")

        except Exception as e:
            print(f"Error fetching namespace {ns}: {str(e)}")
            continue

    if not corpus:
        print("No valid text chunks found across all namespaces.")
        return

    print(f"\nTotal documents collected: {len(corpus)}")
    print("Fitting BM25 on full corpus...")

    bm25 = BM25Encoder()
    bm25.fit(corpus)

    output_file = "utilities/fitted_bm25.json"
    bm25.dump(output_file)
    print(f" Success! Custom BM25 encoder saved to: {output_file}")
    print(f"   Ready to load in your main script with: bm25.load('{output_file}')")

if __name__ == "__main__":
    fit_bm25_from_pinecone()