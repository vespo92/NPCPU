"""
NPCPU Storage Protocol

Abstract interface for vector storage, allowing multiple backends:
- ChromaDB (reference implementation)
- Pinecone
- Milvus
- Weaviate
- Qdrant
- Custom implementations

This decouples NPCPU from any specific storage backend.
"""

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum
import asyncio


# ============================================================================
# Core Data Types
# ============================================================================

@dataclass
class StorageResult:
    """Result of a storage operation"""
    success: bool
    id: str
    collection: str
    timestamp: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResult:
    """Single result from a query"""
    id: str
    score: float  # Similarity/relevance score
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    document: Optional[str] = None


@dataclass
class UpdateResult:
    """Result of an update operation"""
    success: bool
    id: str
    collection: str
    fields_updated: List[str]
    error: Optional[str] = None


@dataclass
class DeleteResult:
    """Result of a delete operation"""
    success: bool
    id: str
    collection: str
    error: Optional[str] = None


@dataclass
class CollectionInfo:
    """Metadata about a collection"""
    name: str
    vector_dimension: int
    count: int
    metadata_schema: Dict[str, type]
    created_at: float
    updated_at: float


class DistanceMetric(Enum):
    """Distance metrics for similarity search"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


# ============================================================================
# Vector Storage Protocol
# ============================================================================

@runtime_checkable
class VectorStorageProtocol(Protocol):
    """
    Abstract vector storage interface.

    This protocol allows NPCPU to work with any vector database backend.
    Implementations can use ChromaDB, Pinecone, Milvus, or custom solutions.

    Design Principles:
    - Async-first for non-blocking I/O
    - Backend-agnostic (works with any vector DB)
    - Rich metadata support
    - Flexible querying (vector, text, filters)
    - Collection management
    """

    # ========================================================================
    # Collection Management
    # ========================================================================

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        vector_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        metadata_schema: Optional[Dict[str, type]] = None
    ) -> CollectionInfo:
        """
        Create a new collection.

        Args:
            name: Collection name
            vector_dimension: Dimensionality of vectors
            distance_metric: How to measure similarity
            metadata_schema: Expected metadata fields and types

        Returns:
            CollectionInfo with details
        """
        pass

    @abstractmethod
    async def get_collection(self, name: str) -> CollectionInfo:
        """Get information about a collection"""
        pass

    @abstractmethod
    async def list_collections(self) -> List[CollectionInfo]:
        """List all collections"""
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection (returns success status)"""
        pass

    # ========================================================================
    # Storage Operations
    # ========================================================================

    @abstractmethod
    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """
        Store a vector with metadata.

        Args:
            collection: Collection name
            id: Unique identifier
            vector: Embedding vector
            metadata: Associated metadata
            document: Optional text document

        Returns:
            StorageResult indicating success/failure
        """
        pass

    @abstractmethod
    async def store_batch(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> List[StorageResult]:
        """Store multiple vectors in batch (more efficient)"""
        pass

    # ========================================================================
    # Query Operations
    # ========================================================================

    @abstractmethod
    async def query(
        self,
        collection: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        include_vectors: bool = False
    ) -> List[QueryResult]:
        """
        Query for similar vectors.

        Args:
            collection: Collection to query
            query_vector: Vector to search for (if None, use query_text)
            query_text: Text to search for (will be embedded)
            filters: Metadata filters (e.g., {"consciousness_state": "AWARE"})
            limit: Maximum results to return
            include_vectors: Whether to include vectors in results

        Returns:
            List of QueryResult ordered by similarity
        """
        pass

    @abstractmethod
    async def query_by_id(
        self,
        collection: str,
        id: str,
        include_vector: bool = False
    ) -> Optional[QueryResult]:
        """Get a specific item by ID"""
        pass

    @abstractmethod
    async def query_batch(
        self,
        collection: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        include_vectors: bool = False
    ) -> List[List[QueryResult]]:
        """Query multiple vectors in batch"""
        pass

    # ========================================================================
    # Update Operations
    # ========================================================================

    @abstractmethod
    async def update(
        self,
        collection: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> UpdateResult:
        """
        Update an existing item.

        Args:
            collection: Collection name
            id: Item identifier
            vector: New vector (if None, keep existing)
            metadata: New metadata (if None, keep existing)
            document: New document (if None, keep existing)

        Returns:
            UpdateResult indicating what was updated
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """Insert or update (create if doesn't exist, update if exists)"""
        pass

    # ========================================================================
    # Delete Operations
    # ========================================================================

    @abstractmethod
    async def delete(self, collection: str, id: str) -> DeleteResult:
        """Delete a single item"""
        pass

    @abstractmethod
    async def delete_batch(
        self,
        collection: str,
        ids: List[str]
    ) -> List[DeleteResult]:
        """Delete multiple items"""
        pass

    @abstractmethod
    async def delete_by_filter(
        self,
        collection: str,
        filters: Dict[str, Any]
    ) -> int:
        """
        Delete all items matching filters.

        Returns:
            Number of items deleted
        """
        pass

    # ========================================================================
    # Advanced Operations
    # ========================================================================

    @abstractmethod
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count items in collection (optionally with filters)"""
        pass

    @abstractmethod
    async def scroll(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[List[QueryResult]]:
        """
        Iterate through all items in collection in batches.

        Useful for processing large collections without loading all into memory.
        """
        pass


# ============================================================================
# Storage Backend Registry
# ============================================================================

class StorageBackendRegistry:
    """
    Registry for storage backend implementations.

    Allows runtime selection of storage backend via configuration.
    """

    _backends: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, backend_class: type):
        """Register a storage backend implementation"""
        cls._backends[name] = backend_class

    @classmethod
    def create(cls, backend_name: str, **config) -> VectorStorageProtocol:
        """
        Create storage backend instance.

        Args:
            backend_name: Name of registered backend
            **config: Configuration parameters for backend

        Returns:
            Instance of storage backend

        Raises:
            ValueError: If backend not registered
        """
        if backend_name not in cls._backends:
            raise ValueError(
                f"Unknown backend '{backend_name}'. "
                f"Available: {list(cls._backends.keys())}"
            )

        backend_class = cls._backends[backend_name]
        return backend_class(**config)

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names"""
        return list(cls._backends.keys())


# ============================================================================
# In-Memory Storage (Reference Implementation for Testing)
# ============================================================================

class InMemoryStorage:
    """
    Reference implementation using in-memory storage.

    Useful for:
    - Testing without external dependencies
    - Development and prototyping
    - Small-scale deployments

    Not suitable for:
    - Production use with large datasets
    - Persistent storage requirements
    - Distributed deployments
    """

    def __init__(self):
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def create_collection(
        self,
        name: str,
        vector_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        metadata_schema: Optional[Dict[str, type]] = None
    ) -> CollectionInfo:
        """Create collection in memory"""
        import time

        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists")

        self.collections[name] = {
            "vector_dimension": vector_dimension,
            "distance_metric": distance_metric,
            "metadata_schema": metadata_schema or {},
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self.data[name] = {}

        return await self.get_collection(name)

    async def get_collection(self, name: str) -> CollectionInfo:
        """Get collection info"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' does not exist")

        coll = self.collections[name]
        return CollectionInfo(
            name=name,
            vector_dimension=coll["vector_dimension"],
            count=len(self.data[name]),
            metadata_schema=coll["metadata_schema"],
            created_at=coll["created_at"],
            updated_at=coll["updated_at"]
        )

    async def list_collections(self) -> List[CollectionInfo]:
        """List all collections"""
        return [await self.get_collection(name) for name in self.collections.keys()]

    async def delete_collection(self, name: str) -> bool:
        """Delete collection"""
        if name in self.collections:
            del self.collections[name]
            del self.data[name]
            return True
        return False

    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """Store vector"""
        import time

        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        # Validate vector dimension
        expected_dim = self.collections[collection]["vector_dimension"]
        if len(vector) != expected_dim:
            return StorageResult(
                success=False,
                id=id,
                collection=collection,
                timestamp=time.time(),
                error=f"Vector dimension {len(vector)} != {expected_dim}"
            )

        self.data[collection][id] = {
            "vector": vector,
            "metadata": metadata or {},
            "document": document,
            "timestamp": time.time()
        }

        self.collections[collection]["updated_at"] = time.time()

        return StorageResult(
            success=True,
            id=id,
            collection=collection,
            timestamp=time.time(),
            metadata=metadata
        )

    async def store_batch(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> List[StorageResult]:
        """Store batch"""
        results = []
        for i, (id_, vector) in enumerate(zip(ids, vectors)):
            metadata = metadatas[i] if metadatas else None
            document = documents[i] if documents else None
            result = await self.store(collection, id_, vector, metadata, document)
            results.append(result)
        return results

    async def query(
        self,
        collection: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        include_vectors: bool = False
    ) -> List[QueryResult]:
        """Query vectors"""
        import numpy as np

        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        if query_vector is None:
            # In real implementation, would embed query_text
            raise NotImplementedError("Text query requires embedding model")

        # Calculate distances
        results = []
        distance_metric = self.collections[collection]["distance_metric"]

        for id_, item in self.data[collection].items():
            # Apply filters
            if filters:
                match = all(
                    item["metadata"].get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue

            # Calculate distance
            stored_vec = np.array(item["vector"])
            query_vec = np.array(query_vector)

            if distance_metric == DistanceMetric.COSINE:
                score = np.dot(query_vec, stored_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                )
            elif distance_metric == DistanceMetric.EUCLIDEAN:
                score = -np.linalg.norm(query_vec - stored_vec)
            else:
                score = np.dot(query_vec, stored_vec)

            results.append(QueryResult(
                id=id_,
                score=float(score),
                vector=item["vector"] if include_vectors else None,
                metadata=item["metadata"],
                document=item["document"]
            ))

        # Sort by score (descending) and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def query_by_id(
        self,
        collection: str,
        id: str,
        include_vector: bool = False
    ) -> Optional[QueryResult]:
        """Query by ID"""
        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        if id not in self.data[collection]:
            return None

        item = self.data[collection][id]
        return QueryResult(
            id=id,
            score=1.0,
            vector=item["vector"] if include_vector else None,
            metadata=item["metadata"],
            document=item["document"]
        )

    async def query_batch(
        self,
        collection: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        include_vectors: bool = False
    ) -> List[List[QueryResult]]:
        """Query batch"""
        results = []
        for query_vec in query_vectors:
            result = await self.query(
                collection,
                query_vector=query_vec,
                limit=limit,
                include_vectors=include_vectors
            )
            results.append(result)
        return results

    async def update(
        self,
        collection: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> UpdateResult:
        """Update item"""
        import time

        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        if id not in self.data[collection]:
            return UpdateResult(
                success=False,
                id=id,
                collection=collection,
                fields_updated=[],
                error=f"ID '{id}' not found"
            )

        item = self.data[collection][id]
        fields_updated = []

        if vector is not None:
            item["vector"] = vector
            fields_updated.append("vector")

        if metadata is not None:
            item["metadata"].update(metadata)
            fields_updated.append("metadata")

        if document is not None:
            item["document"] = document
            fields_updated.append("document")

        item["timestamp"] = time.time()
        self.collections[collection]["updated_at"] = time.time()

        return UpdateResult(
            success=True,
            id=id,
            collection=collection,
            fields_updated=fields_updated
        )

    async def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """Upsert (insert or update)"""
        if id in self.data.get(collection, {}):
            await self.update(collection, id, vector, metadata, document)
        return await self.store(collection, id, vector, metadata, document)

    async def delete(self, collection: str, id: str) -> DeleteResult:
        """Delete item"""
        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        if id in self.data[collection]:
            del self.data[collection][id]
            self.collections[collection]["updated_at"] = time.time()
            return DeleteResult(success=True, id=id, collection=collection)

        return DeleteResult(
            success=False,
            id=id,
            collection=collection,
            error=f"ID '{id}' not found"
        )

    async def delete_batch(
        self,
        collection: str,
        ids: List[str]
    ) -> List[DeleteResult]:
        """Delete batch"""
        return [await self.delete(collection, id_) for id_ in ids]

    async def delete_by_filter(
        self,
        collection: str,
        filters: Dict[str, Any]
    ) -> int:
        """Delete by filter"""
        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        deleted_count = 0
        ids_to_delete = []

        for id_, item in self.data[collection].items():
            match = all(
                item["metadata"].get(k) == v
                for k, v in filters.items()
            )
            if match:
                ids_to_delete.append(id_)

        for id_ in ids_to_delete:
            del self.data[collection][id_]
            deleted_count += 1

        if deleted_count > 0:
            self.collections[collection]["updated_at"] = time.time()

        return deleted_count

    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count items"""
        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        if not filters:
            return len(self.data[collection])

        count = 0
        for item in self.data[collection].values():
            match = all(
                item["metadata"].get(k) == v
                for k, v in filters.items()
            )
            if match:
                count += 1

        return count

    async def scroll(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[List[QueryResult]]:
        """Scroll through items"""
        if collection not in self.collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        batch = []
        for id_, item in self.data[collection].items():
            # Apply filters
            if filters:
                match = all(
                    item["metadata"].get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue

            batch.append(QueryResult(
                id=id_,
                score=1.0,
                vector=item["vector"],
                metadata=item["metadata"],
                document=item["document"]
            ))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


# Register the in-memory backend
StorageBackendRegistry.register("memory", InMemoryStorage)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def main():
        # Create storage backend
        storage = StorageBackendRegistry.create("memory")

        # Create collection
        collection_info = await storage.create_collection(
            name="test_personas",
            vector_dimension=384,
            distance_metric=DistanceMetric.COSINE
        )
        print(f"Created collection: {collection_info.name}")

        # Store some vectors
        import random
        vectors = [[random.random() for _ in range(384)] for _ in range(10)]
        ids = [f"persona_{i}" for i in range(10)]
        metadatas = [{"consciousness_state": "AWARE", "index": i} for i in range(10)]

        results = await storage.store_batch(
            collection="test_personas",
            ids=ids,
            vectors=vectors,
            metadatas=metadatas
        )
        print(f"Stored {len(results)} vectors")

        # Query
        query_results = await storage.query(
            collection="test_personas",
            query_vector=vectors[0],
            limit=5
        )
        print(f"\nTop 5 similar vectors:")
        for result in query_results:
            print(f"  {result.id}: score={result.score:.3f}")

        # Query with filter
        filtered_results = await storage.query(
            collection="test_personas",
            query_vector=vectors[0],
            filters={"consciousness_state": "AWARE"},
            limit=3
        )
        print(f"\nFiltered results: {len(filtered_results)}")

        # Count
        count = await storage.count("test_personas")
        print(f"\nTotal items: {count}")

        # Scroll through all items
        print("\nScrolling through items:")
        async for batch in storage.scroll("test_personas", batch_size=3):
            print(f"  Batch of {len(batch)} items")

    asyncio.run(main())
