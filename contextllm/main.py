"""Main entry point for Context Budget Optimizer CLI."""

import argparse
import sys
import logging
from pathlib import Path

from contextllm.utils.logging_setup import setup_logging
from contextllm.utils.config import get_config
from contextllm.ingestion.pipeline import ingest_documents
from contextllm.retrieval.searcher import search_chunks
from contextllm.optimization.optimizer import optimize_context
from contextllm.generation.generator import generate_answer
from contextllm.optimization.explainer import explain_optimization

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def ingest_command(args):
    """Handle ingest command."""
    if not args.files:
        logger.error("No files provided for ingestion")
        return 1
    
    logger.info(f"Ingesting {len(args.files)} file(s)...")
    
    try:
        results = ingest_documents(args.files)
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        print(f"\nIngestion complete:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        for result in successful:
            print(f"  ✓ {result.get('filename')}: {result.get('num_chunks')} chunks")
        
        for result in failed:
            print(f"  ✗ {result.get('filename')}: {result.get('error', 'Unknown error')}")
        
        return 0 if not failed else 1
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        return 1


def query_command(args):
    """Handle query command."""
    if not args.query:
        logger.error("No query provided")
        return 1
    
    logger.info(f"Processing query: {args.query[:100]}...")
    
    try:
        # Retrieve chunks
        chunks = search_chunks(args.query, top_k=args.top_k)
        
        if not chunks:
            print("No relevant chunks found.")
            return 0
        
        print(f"\nRetrieved {len(chunks)} chunks")
        
        # Optimize
        budget = args.budget
        optimization_result = optimize_context(chunks, budget=budget)
        selected_chunks = optimization_result.get('selected_chunks', [])
        
        if not selected_chunks:
            print("No chunks could fit within the budget.")
            return 0
        
        print(f"Selected {len(selected_chunks)} chunks within budget")
        
        # Generate answer
        generation_result = generate_answer(
            query=args.query,
            selected_chunks=selected_chunks,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(generation_result.get('answer', ''))
        print("\n" + "=" * 60)
        
        # Display optimization explanation if requested
        if args.explain:
            print("\nOPTIMIZATION EXPLANATION")
            print("=" * 60)
            explanation = explain_optimization(optimization_result)
            print(explanation)
        
        # Display usage
        usage = generation_result.get('usage', {})
        print(f"\nToken Usage:")
        print(f"  Prompt: {usage.get('prompt_tokens', 0)}")
        print(f"  Completion: {usage.get('completion_tokens', 0)}")
        print(f"  Total: {usage.get('total_tokens', 0)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return 1


def server_command(args):
    """Handle server command."""
    import uvicorn
    from contextllm.api.server import app
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Context Budget Optimizer - Intelligent document chunk selection for LLMs"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('files', nargs='+', help='Files to ingest')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--budget', type=int, help='Token budget (default from config)')
    query_parser.add_argument('--top-k', type=int, default=50, help='Number of chunks to retrieve (default: 50)')
    query_parser.add_argument('--temperature', type=float, help='Temperature for generation')
    query_parser.add_argument('--max-tokens', type=int, help='Max tokens for response')
    query_parser.add_argument('--explain', action='store_true', help='Show optimization explanation')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start web server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    server_parser.add_argument('--log-level', default='info', help='Log level (default: info)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'ingest':
            return ingest_command(args)
        elif args.command == 'query':
            return query_command(args)
        elif args.command == 'server':
            server_command(args)
            return 0
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
