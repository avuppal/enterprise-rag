import time
import random
from src.retriever import EnterpriseRetriever

def generate_mock_docs(num_docs):
    return [{"id": f"doc_{i}", "text": f"This is mock document {i} with some generic content.", "metadata": {}} for i in range(num_docs)]

def run_benchmark():
    sizes = [100, 1000, 5000]
    results = []
    
    for size in sizes:
        print(f"Benchmarking corpus size: {size}")
        docs = generate_mock_docs(size)
        
        # Dense only
        retriever_dense = EnterpriseRetriever(dense_only=True)
        retriever_dense.add_documents(docs)
        
        start = time.time()
        retriever_dense.search("mock document content", k=5)
        dense_time = time.time() - start
        
        # Hybrid
        retriever_hybrid = EnterpriseRetriever(dense_only=False)
        retriever_hybrid.add_documents(docs)
        
        start = time.time()
        retriever_hybrid.search("mock document content", k=5)
        hybrid_time = time.time() - start
        
        results.append({
            "size": size,
            "dense_time": dense_time,
            "hybrid_time": hybrid_time
        })
        
    print("\nBenchmark Results:")
    print("Corpus Size | Dense Time (s) | Hybrid Time (s)")
    print("-" * 50)
    for r in results:
        print(f"{r['size']:<11} | {r['dense_time']:<14.4f} | {r['hybrid_time']:<15.4f}")

if __name__ == "__main__":
    run_benchmark()
