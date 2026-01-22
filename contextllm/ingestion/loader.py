"""Document loaders for various file formats."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Base class for document loaders."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document and return its content and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        raise NotImplementedError


class TextLoader(DocumentLoader):
    """Loader for plain text files."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = {
                'source': str(path),
                'filename': path.name,
                'file_type': 'text',
                'file_size': path.stat().st_size,
            }
            
            logger.info(f"Loaded text file: {path.name} ({len(text)} characters)")
            return {
                'text': text,
                'metadata': metadata
            }
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()
            
            metadata = {
                'source': str(path),
                'filename': path.name,
                'file_type': 'text',
                'file_size': path.stat().st_size,
            }
            
            return {
                'text': text,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF loading. Install it with: pip install pypdf")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            text_parts = []
            with open(path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {path.name}: {e}")
                        continue
            
            text = '\n\n'.join(text_parts)
            
            metadata = {
                'source': str(path),
                'filename': path.name,
                'file_type': 'pdf',
                'file_size': path.stat().st_size,
                'num_pages': num_pages,
            }
            
            logger.info(f"Loaded PDF file: {path.name} ({num_pages} pages, {len(text)} characters)")
            return {
                'text': text,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            raise


def get_loader(file_path: str) -> DocumentLoader:
    """
    Get appropriate loader for a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        DocumentLoader instance
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.txt':
        return TextLoader()
    elif extension == '.pdf':
        return PDFLoader()
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported types: .txt, .pdf")


def load_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load multiple documents.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of document dictionaries (text + metadata)
    """
    documents = []
    
    for file_path in file_paths:
        try:
            loader = get_loader(file_path)
            document = loader.load(file_path)
            documents.append(document)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue
    
    return documents
