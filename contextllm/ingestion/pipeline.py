"""End-to-end ingestion pipeline."""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from contextllm.ingestion.loader import load_documents
from contextllm.ingestion.chunker import TextChunker
from contextllm.ingestion.embedder import generate_embeddings
from contextllm.ingestion.storage import VectorStore, MetadataStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Complete ingestion pipeline from documents to stored embeddings."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        metadata_store: Optional[MetadataStore] = None,
        chunker: Optional[TextChunker] = None
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            vector_store: VectorStore instance (created if None)
            metadata_store: MetadataStore instance (created if None)
            chunker: TextChunker instance (created if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.metadata_store = metadata_store or MetadataStore()
        self.chunker = chunker or TextChunker()
        
        logger.info("Ingestion pipeline initialized")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        document_id = str(uuid.uuid4())
        file_path_obj = Path(file_path)
        
        try:
            logger.info(f"Ingesting document: {file_path}")
            self.metadata_store.log_ingestion(document_id, "started", f"Processing {file_path}")
            
            # Load document
            documents = load_documents([file_path])
            if not documents:
                raise ValueError(f"Failed to load document: {file_path}")
            
            document = documents[0]
            doc_metadata = document.get('metadata', {})
            
            # Chunk document
            chunks = self.chunker.chunk_document(document)
            if not chunks:
                raise ValueError(f"No chunks created from document: {file_path}")
            
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = generate_embeddings(chunk_texts)
            
            # Assign IDs to chunks
            chunk_ids = []
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                chunk['id'] = chunk_id
                chunk['metadata']['chunk_id'] = chunk_id
            
            # Store in vector database
            self.vector_store.add_chunks(chunks, embeddings, chunk_ids)
            
            # Store metadata
            self.metadata_store.add_document(
                document_id=document_id,
                filename=doc_metadata.get('filename', file_path_obj.name),
                file_path=str(file_path_obj.absolute()),
                file_type=doc_metadata.get('file_type', 'unknown'),
                file_size=doc_metadata.get('file_size', 0),
                num_chunks=len(chunks)
            )
            
            self.metadata_store.add_chunks(chunks, document_id)
            
            # Log success
            self.metadata_store.log_ingestion(
                document_id,
                "success",
                f"Successfully ingested {len(chunks)} chunks"
            )
            
            result = {
                'document_id': document_id,
                'filename': doc_metadata.get('filename', file_path_obj.name),
                'num_chunks': len(chunks),
                'status': 'success'
            }
            
            logger.info(f"Successfully ingested document: {file_path} ({len(chunks)} chunks)")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            self.metadata_store.log_ingestion(document_id, "error", str(e))
            
            return {
                'document_id': document_id,
                'filename': file_path_obj.name if file_path_obj.exists() else file_path,
                'num_chunks': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def ingest_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents.
        
        Args:
            file_paths: List of file paths to ingest
            
        Returns:
            List of ingestion result dictionaries
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.ingest_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch ingestion for {file_path}: {e}")
                results.append({
                    'filename': Path(file_path).name if Path(file_path).exists() else file_path,
                    'num_chunks': 0,
                    'status': 'error',
                    'error': str(e)
                })
        
        successful = sum(1 for r in results if r.get('status') == 'success')
        logger.info(f"Batch ingestion complete: {successful}/{len(file_paths)} documents successful")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get vector store count
            vector_count = self.vector_store.collection.count()
            
            # Get metadata store stats
            conn = self.metadata_store.db_path
            import sqlite3
            conn_obj = sqlite3.connect(conn)
            cursor = conn_obj.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(num_chunks) FROM documents")
            total_chunks = cursor.fetchone()[0] or 0
            
            conn_obj.close()
            
            return {
                'vector_store_chunks': vector_count,
                'documents': doc_count,
                'chunks_in_metadata': chunk_count,
                'total_chunks_processed': total_chunks
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Convenience function
def ingest_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest documents.
    
    Args:
        file_paths: List of file paths to ingest
        
    Returns:
        List of ingestion result dictionaries
    """
    pipeline = IngestionPipeline()
    return pipeline.ingest_documents(file_paths)
