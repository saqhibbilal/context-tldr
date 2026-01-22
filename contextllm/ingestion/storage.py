"""Storage layer for vector database (ChromaDB) and metadata (SQLite)."""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings
import numpy as np

from contextllm.utils.config import get_config
from contextllm.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for embeddings."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist database (uses config if None)
            collection_name: Name of the collection (uses config if None)
        """
        config = get_config()
        
        self.persist_directory = persist_directory or config.get("vector_db.persist_directory", "./data/vector_db")
        self.collection_name = collection_name or config.get("vector_db.collection_name", "context_chunks")
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Context chunks for budget optimizer"}
            )
            
            logger.info(f"Initialized ChromaDB vector store at {self.persist_directory}")
            logger.info(f"Collection '{self.collection_name}' has {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        chunk_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            embeddings: NumPy array of embeddings (shape: [num_chunks, embedding_dim])
            chunk_ids: Optional list of chunk IDs (generated if None)
            
        Returns:
            List of chunk IDs that were added
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
        
        if chunk_ids is None:
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {}).copy()
            # ChromaDB requires metadata values to be strings, numbers, or booleans
            # Convert any non-serializable values
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            metadatas.append(clean_metadata)
        
        # Convert embeddings to list of lists
        embeddings_list = embeddings.tolist()
        
        try:
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings_list,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dictionaries with 'id', 'text', 'metadata', 'distance', 'score'
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'score': 1 - results['distances'][0][i] if 'distances' in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk dictionary or None if not found
        """
        try:
            results = self.collection.get(ids=[chunk_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
            return None
    
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise


class MetadataStore:
    """SQLite database for metadata tracking."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite metadata store.
        
        Args:
            db_path: Path to SQLite database (uses config if None)
        """
        config = get_config()
        self.db_path = db_path or config.get("metadata.db_path", "./data/metadata.db")
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"Initialized metadata store at {self.db_path}")
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT,
                    file_size INTEGER,
                    num_chunks INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER,
                    embedding_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            # Ingestion log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    status TEXT NOT NULL,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            conn.commit()
            logger.debug("Database schema initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise
        finally:
            conn.close()
    
    def add_document(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
        num_chunks: int = 0
    ) -> None:
        """
        Add a document record.
        
        Args:
            document_id: Unique document ID
            filename: Document filename
            file_path: Full path to document
            file_type: Type of file (text, pdf, etc.)
            file_size: File size in bytes
            num_chunks: Number of chunks created from document
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, filename, file_path, file_type, file_size, num_chunks, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (document_id, filename, file_path, file_type, file_size, num_chunks, datetime.now()))
            
            conn.commit()
            logger.debug(f"Added document record: {document_id}")
            
        except Exception as e:
            logger.error(f"Error adding document record: {e}")
            raise
        finally:
            conn.close()
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str
    ) -> None:
        """
        Add chunk records.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'text', 'metadata'
            document_id: ID of the parent document
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for chunk in chunks:
                chunk_id = chunk.get('id') or chunk.get('metadata', {}).get('chunk_id')
                if not chunk_id:
                    continue
                
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                chunk_index = metadata.get('chunk_index', 0)
                embedding_id = chunk_id  # Same as chunk_id for ChromaDB
                
                # Count tokens
                token_count = count_tokens(text)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, document_id, chunk_index, text, token_count, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, document_id, chunk_index, text, token_count, embedding_id))
            
            conn.commit()
            logger.debug(f"Added {len(chunks)} chunk records for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error adding chunk records: {e}")
            raise
        finally:
            conn.close()
    
    def log_ingestion(
        self,
        document_id: Optional[str],
        status: str,
        message: Optional[str] = None
    ) -> None:
        """
        Log an ingestion event.
        
        Args:
            document_id: Document ID (None for general events)
            status: Status (success, error, etc.)
            message: Optional message
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO ingestion_log (document_id, status, message)
                VALUES (?, ?, ?)
            """, (document_id, status, message))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error logging ingestion event: {e}")
        finally:
            conn.close()
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document record.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document dictionary or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
        finally:
            conn.close()
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []
        finally:
            conn.close()
