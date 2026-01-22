"""Example usage of Context Budget Optimizer."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextllm.utils.logging_setup import setup_logging
from contextllm.ingestion.pipeline import ingest_documents
from contextllm.retrieval.searcher import search_chunks
from contextllm.optimization.optimizer import optimize_context
from contextllm.generation.generator import generate_answer
from contextllm.optimization.explainer import explain_optimization

# Setup logging
setup_logging()
import logging
logger = logging.getLogger(__name__)


def example_ingestion():
    """Example: Ingest documents."""
    print("=" * 60)
    print("Example 1: Document Ingestion")
    print("=" * 60)
    
    # Example: Ingest text files
    # Replace with your actual document paths
    files = [
        # "path/to/document1.txt",
        # "path/to/document2.pdf",
    ]
    
    if files:
        results = ingest_documents(files)
        print(f"Ingested {len(results)} documents")
        for result in results:
            print(f"  - {result.get('filename')}: {result.get('num_chunks')} chunks")
    else:
        print("No files specified. Add file paths to the 'files' list.")


def example_query():
    """Example: Query the system."""
    print("\n" + "=" * 60)
    print("Example 2: Query Processing")
    print("=" * 60)
    
    query = "What is the main topic of the documents?"
    budget = 2000
    
    print(f"Query: {query}")
    print(f"Budget: {budget} tokens")
    print("\nProcessing...")
    
    # Retrieve chunks
    chunks = search_chunks(query, top_k=50)
    print(f"Retrieved {len(chunks)} chunks")
    
    # Optimize
    optimization_result = optimize_context(chunks, budget=budget)
    selected_chunks = optimization_result.get('selected_chunks', [])
    print(f"Selected {len(selected_chunks)} chunks")
    
    # Generate answer
    if selected_chunks:
        generation_result = generate_answer(
            query=query,
            selected_chunks=selected_chunks
        )
        
        print("\nAnswer:")
        print("-" * 60)
        print(generation_result.get('answer', ''))
        print("-" * 60)
        
        # Show explanation
        print("\nOptimization Explanation:")
        print(explain_optimization(optimization_result))
    else:
        print("No chunks selected within budget.")


def example_budget_comparison():
    """Example: Compare different budgets."""
    print("\n" + "=" * 60)
    print("Example 3: Budget Comparison")
    print("=" * 60)
    
    query = "What are the key points?"
    
    budgets = [1000, 2000, 4000]
    
    chunks = search_chunks(query, top_k=50)
    
    for budget in budgets:
        print(f"\nBudget: {budget} tokens")
        result = optimize_context(chunks, budget=budget)
        selected = result.get('selected_chunks', [])
        tokens_used = result.get('total_tokens', 0)
        budget_used = result.get('budget_used', 0)
        
        print(f"  Chunks selected: {len(selected)}")
        print(f"  Tokens used: {tokens_used}")
        print(f"  Budget utilization: {budget_used:.1f}%")


if __name__ == '__main__':
    print("Context Budget Optimizer - Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_ingestion()
    # example_query()  # Uncomment after ingesting documents
    # example_budget_comparison()  # Uncomment after ingesting documents
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nTo use the system:")
    print("1. Ingest documents: python -m contextllm.main ingest file1.txt file2.pdf")
    print("2. Query: python -m contextllm.main query 'Your question' --budget 2000")
    print("3. Start web server: python -m contextllm.main server")
