"""FastAPI routes for the Context Budget Optimizer API."""

import logging
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from contextllm.retrieval.searcher import ChunkSearcher, search_chunks
from contextllm.optimization.optimizer import optimize_context
from contextllm.generation.generator import ResponseGenerator, generate_answer
from contextllm.utils.metadata_db import QueryMetadataStore
from contextllm.utils.observability import get_decision_logger
from contextllm.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

# Initialize components
setup_logging()
searcher = ChunkSearcher()
generator = ResponseGenerator()
metadata_store = QueryMetadataStore()
decision_logger = get_decision_logger()

router = APIRouter()


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query text")
    budget: Optional[int] = Field(None, description="Token budget (uses config default if not provided)")
    temperature: Optional[float] = Field(None, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Max tokens for response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query_id: str
    answer: str
    usage: Dict[str, int]
    optimization: Dict[str, Any]
    chunks_used: int
    model: str


class ChunkInfo(BaseModel):
    """Chunk information model."""
    chunk_id: str
    text: str
    similarity_score: float
    token_count: int
    value_per_token: Optional[float] = None
    included: bool
    inclusion_reason: Optional[str] = None
    metadata: Dict[str, Any]


@router.post("/api/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest) -> QueryResponse:
    """
    Submit a query and get optimized answer.
    
    Args:
        request: Query request with query text and optional parameters
        
    Returns:
        Query response with answer and metadata
    """
    try:
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query {query_id}: {request.query[:100]}...")
        
        # Retrieve chunks
        chunks = search_chunks(request.query, top_k=50)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found")
        
        # Log retrieval
        decision_logger.log_retrieval(request.query, chunks, top_k=50)
        
        # Optimize context
        optimization_result = optimize_context(chunks, budget=request.budget)
        selected_chunks = optimization_result.get('selected_chunks', [])
        
        if not selected_chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks could fit within the budget. Try increasing the budget."
            )
        
        # Log optimization
        decision_logger.log_optimization(request.query, chunks, optimization_result)
        
        # Generate answer
        generation_result = generator.generate(
            query=request.query,
            selected_chunks=selected_chunks,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Log generation
        decision_logger.log_generation(request.query, generation_result)
        
        # Save to metadata store
        metadata_store.save_query(
            query_id=query_id,
            query_text=request.query,
            budget=request.budget,
            model=generation_result.get('model'),
            temperature=request.temperature
        )
        
        # Save chunks
        all_chunks = selected_chunks + optimization_result.get('excluded_chunks', [])
        metadata_store.save_query_chunks(query_id, all_chunks, optimization_result)
        
        # Save response
        response_id = str(uuid.uuid4())
        metadata_store.save_response(
            response_id=response_id,
            query_id=query_id,
            answer_text=generation_result.get('answer', ''),
            usage=generation_result.get('usage', {}),
            chunks_included_count=len(selected_chunks),
            budget_used=optimization_result.get('budget_used', 0)
        )
        
        # Prepare response
        response = QueryResponse(
            query_id=query_id,
            answer=generation_result.get('answer', ''),
            usage=generation_result.get('usage', {}),
            optimization={
                'chunks_evaluated': optimization_result.get('selection_metadata', {}).get('chunks_evaluated', 0),
                'chunks_selected': len(selected_chunks),
                'chunks_excluded': len(optimization_result.get('excluded_chunks', [])),
                'total_tokens': optimization_result.get('total_tokens', 0),
                'budget_used': optimization_result.get('budget_used', 0)
            },
            chunks_used=len(selected_chunks),
            model=generation_result.get('model', 'unknown')
        )
        
        logger.info(f"Query {query_id} completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/api/chunks/{query_id}", response_model=List[ChunkInfo])
async def get_chunks(query_id: str) -> List[ChunkInfo]:
    """
    Get chunks for a specific query.
    
    Args:
        query_id: Query ID
        
    Returns:
        List of chunk information
    """
    try:
        query_chunks = metadata_store.get_query_chunks(query_id)
        
        if not query_chunks:
            raise HTTPException(status_code=404, detail=f"No chunks found for query {query_id}")
        
        # Format chunks
        chunk_infos = []
        for chunk_data in query_chunks:
            # Get full chunk from vector store if needed
            chunk_id = chunk_data.get('chunk_id', '')
            chunk = searcher.get_chunk_by_id(chunk_id)
            
            if chunk:
                chunk_infos.append(ChunkInfo(
                    chunk_id=chunk_id,
                    text=chunk.get('text', ''),
                    similarity_score=chunk_data.get('similarity_score', 0),
                    token_count=chunk_data.get('token_count', 0),
                    value_per_token=chunk_data.get('value_score', None),
                    included=bool(chunk_data.get('included', False)),
                    inclusion_reason=chunk_data.get('inclusion_reason'),
                    metadata=chunk.get('metadata', {})
                ))
        
        return chunk_infos
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/api/history")
async def get_history(limit: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """
    Get query history.
    
    Args:
        limit: Maximum number of queries to return
        
    Returns:
        List of query history entries
    """
    try:
        history = metadata_store.get_query_history(limit=limit)
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get system statistics.
    
    Returns:
        Dictionary with system statistics
    """
    try:
        from contextllm.ingestion.pipeline import IngestionPipeline
        
        pipeline = IngestionPipeline()
        ingestion_stats = pipeline.get_stats()
        
        # Get query stats
        history = metadata_store.get_query_history(limit=1000)
        total_queries = len(history)
        
        stats = {
            'ingestion': ingestion_stats,
            'queries': {
                'total': total_queries,
                'recent': len([h for h in history if h.get('timestamp')])
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/api/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy", "service": "context-budget-optimizer"}
