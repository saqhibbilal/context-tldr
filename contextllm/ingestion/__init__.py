"""Ingestion layer for document processing, chunking, and embedding."""

from contextllm.ingestion.loader import (
    DocumentLoader,
    TextLoader,
    PDFLoader,
    get_loader,
    load_documents
)
from contextllm.ingestion.chunker import TextChunker
from contextllm.ingestion.embedder import (
    get_embedding_model,
    generate_embeddings,
    generate_embedding
)
from contextllm.ingestion.storage import VectorStore, MetadataStore
from contextllm.ingestion.pipeline import IngestionPipeline, ingest_documents

__all__ = [
    "DocumentLoader",
    "TextLoader",
    "PDFLoader",
    "get_loader",
    "load_documents",
    "TextChunker",
    "get_embedding_model",
    "generate_embeddings",
    "generate_embedding",
    "VectorStore",
    "MetadataStore",
    "IngestionPipeline",
    "ingest_documents",
]
