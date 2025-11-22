"""
NPCPU Storage Adapters

Provides implementations of VectorStorageProtocol for various backends:
- ChromaDB: Local/remote vector storage
- Pinecone: Cloud-scale vector storage
"""

from .chromadb_adapter import ChromaDBAdapter

# Conditional imports for optional dependencies
try:
    from .pinecone_adapter import PineconeAdapter
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    PineconeAdapter = None

__all__ = [
    "ChromaDBAdapter",
    "PineconeAdapter",
    "PINECONE_AVAILABLE"
]
