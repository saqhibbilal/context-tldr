"""SQLite operations for query and response metadata tracking."""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextllm.utils.config import get_config

logger = logging.getLogger(__name__)


class QueryMetadataStore:
    """Stores query and response metadata in SQLite."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize query metadata store.
        
        Args:
            db_path: Path to SQLite database (uses config if None)
        """
        config = get_config()
        self.db_path = db_path or config.get("metadata.db_path", "./data/metadata.db")
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._init_schema()
        
        logger.info(f"Query metadata store initialized at {self.db_path}")
    
    def _init_schema(self) -> None:
        """Initialize database schema for queries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    budget INTEGER,
                    model TEXT,
                    temperature REAL
                )
            """)
            
            # Query chunks table (tracks which chunks were retrieved/evaluated)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    similarity_score REAL,
                    token_count INTEGER,
                    value_score REAL,
                    included BOOLEAN,
                    inclusion_reason TEXT,
                    FOREIGN KEY (query_id) REFERENCES queries(id)
                )
            """)
            
            # Responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    chunks_included_count INTEGER,
                    budget_used REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES queries(id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_chunks_query_id 
                ON query_chunks(query_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_responses_query_id 
                ON responses(query_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_timestamp 
                ON queries(timestamp)
            """)
            
            conn.commit()
            logger.debug("Query metadata schema initialized")
            
        except Exception as e:
            logger.error(f"Error initializing query metadata schema: {e}")
            raise
        finally:
            conn.close()
    
    def save_query(
        self,
        query_id: str,
        query_text: str,
        budget: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> None:
        """
        Save a query record.
        
        Args:
            query_id: Unique query ID
            query_text: Query text
            budget: Token budget used
            model: Model name
            temperature: Temperature used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO queries 
                (id, query_text, timestamp, budget, model, temperature)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query_id, query_text, datetime.now(), budget, model, temperature))
            
            conn.commit()
            logger.debug(f"Saved query: {query_id}")
            
        except Exception as e:
            logger.error(f"Error saving query: {e}")
            raise
        finally:
            conn.close()
    
    def save_query_chunks(
        self,
        query_id: str,
        chunks: List[Dict[str, Any]],
        optimization_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save chunk evaluation data for a query.
        
        Args:
            query_id: Query ID
            chunks: List of chunks (all evaluated chunks)
            optimization_result: Optional optimization result to determine inclusion
        """
        if not chunks:
            return
        
        # Create a set of selected chunk IDs if optimization result provided
        selected_ids = set()
        if optimization_result:
            selected = optimization_result.get('selected_chunks', [])
            selected_ids = {chunk.get('chunk_id', '') for chunk in selected}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id', '')
                similarity_score = chunk.get('similarity_score', 0)
                token_count = chunk.get('token_count', 0)
                value_score = chunk.get('value_per_token', 0)
                
                # Determine if included
                included = chunk_id in selected_ids if optimization_result else False
                inclusion_reason = chunk.get('metadata', {}).get('inclusion_reason', 'not_optimized')
                
                cursor.execute("""
                    INSERT INTO query_chunks 
                    (query_id, chunk_id, similarity_score, token_count, value_score, included, inclusion_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (query_id, chunk_id, similarity_score, token_count, value_score, included, inclusion_reason))
            
            conn.commit()
            logger.debug(f"Saved {len(chunks)} query chunks for query {query_id}")
            
        except Exception as e:
            logger.error(f"Error saving query chunks: {e}")
            raise
        finally:
            conn.close()
    
    def save_response(
        self,
        response_id: str,
        query_id: str,
        answer_text: str,
        usage: Dict[str, int],
        chunks_included_count: int,
        budget_used: Optional[float] = None
    ) -> None:
        """
        Save a response record.
        
        Args:
            response_id: Unique response ID
            query_id: Associated query ID
            answer_text: Generated answer text
            usage: Token usage dictionary
            chunks_included_count: Number of chunks included
            budget_used: Percentage of budget used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO responses 
                (id, query_id, answer_text, prompt_tokens, completion_tokens, 
                 total_tokens, chunks_included_count, budget_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                response_id,
                query_id,
                answer_text,
                usage.get('prompt_tokens', 0),
                usage.get('completion_tokens', 0),
                usage.get('total_tokens', 0),
                chunks_included_count,
                budget_used
            ))
            
            conn.commit()
            logger.debug(f"Saved response: {response_id}")
            
        except Exception as e:
            logger.error(f"Error saving response: {e}")
            raise
        finally:
            conn.close()
    
    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a query record.
        
        Args:
            query_id: Query ID
            
        Returns:
            Query dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM queries WHERE id = ?", (query_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting query: {e}")
            return None
        finally:
            conn.close()
    
    def get_query_chunks(self, query_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            List of chunk dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM query_chunks 
                WHERE query_id = ? 
                ORDER BY similarity_score DESC
            """, (query_id,))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting query chunks: {e}")
            return []
        finally:
            conn.close()
    
    def get_response(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get response for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            Response dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM responses WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return None
        finally:
            conn.close()
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent query history.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query dictionaries with response info
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT q.*, r.answer_text, r.total_tokens, r.chunks_included_count
                FROM queries q
                LEFT JOIN responses r ON q.id = r.query_id
                ORDER BY q.timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return []
        finally:
            conn.close()
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total queries
            cursor.execute("SELECT COUNT(*) FROM queries")
            total_queries = cursor.fetchone()[0] or 0
            
            # Successful queries (with responses)
            cursor.execute("SELECT COUNT(*) FROM queries q INNER JOIN responses r ON q.id = r.query_id")
            successful_queries = cursor.fetchone()[0] or 0
            
            # Success rate
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            # Total tokens used
            cursor.execute("SELECT SUM(total_tokens) FROM responses")
            total_tokens = cursor.fetchone()[0] or 0
            
            # Average tokens per query
            avg_tokens = (total_tokens / successful_queries) if successful_queries > 0 else 0
            
            # Average chunks per query
            cursor.execute("SELECT AVG(chunks_included_count) FROM responses")
            avg_chunks = cursor.fetchone()[0] or 0
            
            # Recent activity (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) FROM queries 
                WHERE timestamp > datetime('now', '-1 day')
            """)
            recent_queries = cursor.fetchone()[0] or 0
            
            return {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'failed_queries': total_queries - successful_queries,
                'success_rate': round(success_rate, 2),
                'total_tokens': total_tokens,
                'avg_tokens_per_query': round(avg_tokens, 2),
                'avg_chunks_per_query': round(avg_chunks, 2),
                'recent_queries_24h': recent_queries
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {e}")
            return {}
        finally:
            conn.close()
