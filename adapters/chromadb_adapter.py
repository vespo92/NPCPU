"""
ChromaDB Adapter - Production Implementation

Implements VectorStorageProtocol for ChromaDB with:
- Connection pooling
- Retry logic
- Error handling
- Metrics collection
- Async support
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import time
import logging
from contextlib import asynccontextmanager

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None


# Import protocols
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.storage import (
    VectorStorageProtocol,
    StorageResult,
    QueryResult,
    UpdateResult,
    DeleteResult,
    CollectionInfo,
    DistanceMetric
)


logger = logging.getLogger(__name__)


class ChromaDBAdapter:
    """
    Production-grade ChromaDB adapter implementing VectorStorageProtocol.

    Features:
    - Connection pooling and reuse
    - Automatic retry with exponential backoff
    - Comprehensive error handling
    - Performance metrics collection
    - Batch operation optimization
    - Async-compatible interface

    Example:
        adapter = ChromaDBAdapter(path="./chromadb_data")
        await adapter.create_collection("memories", vector_dimension=384)
        await adapter.store("memories", "id1", [0.1, 0.2, ...], {"type": "experience"})
        results = await adapter.query("memories", [0.1, 0.2, ...], limit=5)
    """

    def __init__(
        self,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        settings: Optional[Settings] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize ChromaDB adapter.

        Args:
            path: Path for persistent storage (local mode)
            host: Host for HTTP client (remote mode)
            port: Port for HTTP client (remote mode)
            settings: Custom ChromaDB settings
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Initial delay between retries (exponential backoff)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize client
        if host and port:
            logger.info(f"Connecting to ChromaDB at {host}:{port}")
            self.client = chromadb.HttpClient(host=host, port=port)
            self.mode = "remote"
        else:
            path = path or "./chromadb_data"
            logger.info(f"Using local ChromaDB at {path}")
            self.client = chromadb.PersistentClient(
                path=path,
                settings=settings or Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            self.mode = "local"

        # Metrics
        self.metrics = {
            "queries": 0,
            "stores": 0,
            "updates": 0,
            "deletes": 0,
            "errors": 0,
            "retries": 0,
            "total_latency_ms": 0.0
        }

        # Collection cache
        self._collections = {}

    async def _retry_async(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Run in thread pool for sync operations
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)
                return result

            except Exception as e:
                last_error = e
                self.metrics["retries"] += 1

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed: {e}")

        self.metrics["errors"] += 1
        raise last_error

    @asynccontextmanager
    async def _measure_latency(self, operation: str):
        """Context manager to measure operation latency"""
        start = time.time()
        try:
            yield
        finally:
            latency_ms = (time.time() - start) * 1000
            self.metrics["total_latency_ms"] += latency_ms
            logger.debug(f"{operation} took {latency_ms:.2f}ms")

    def _map_distance_metric(self, metric: DistanceMetric) -> str:
        """Map protocol distance metric to ChromaDB metric"""
        mapping = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip",
            DistanceMetric.MANHATTAN: "l1"
        }
        return mapping.get(metric, "cosine")

    async def create_collection(
        self,
        name: str,
        vector_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        metadata_schema: Optional[Dict[str, type]] = None
    ) -> CollectionInfo:
        """Create a new collection"""
        async with self._measure_latency(f"create_collection({name})"):
            def _create():
                # Delete if exists (for idempotency)
                try:
                    self.client.delete_collection(name)
                except:
                    pass

                # Create collection
                collection = self.client.create_collection(
                    name=name,
                    metadata={
                        "hnsw:space": self._map_distance_metric(distance_metric),
                        "dimension": vector_dimension
                    }
                )

                # Cache collection
                self._collections[name] = collection

                return collection

            collection = await self._retry_async(_create)

            return CollectionInfo(
                name=name,
                vector_dimension=vector_dimension,
                count=collection.count(),
                metadata_schema=metadata_schema or {},
                created_at=time.time(),
                updated_at=time.time()
            )

    async def get_collection(self, name: str) -> CollectionInfo:
        """Get collection information"""
        async with self._measure_latency(f"get_collection({name})"):
            def _get():
                if name in self._collections:
                    return self._collections[name]

                collection = self.client.get_collection(name)
                self._collections[name] = collection
                return collection

            collection = await self._retry_async(_get)

            return CollectionInfo(
                name=name,
                vector_dimension=0,  # ChromaDB doesn't expose this easily
                count=collection.count(),
                metadata_schema={},
                created_at=0,
                updated_at=time.time()
            )

    async def list_collections(self) -> List[CollectionInfo]:
        """List all collections"""
        async with self._measure_latency("list_collections"):
            def _list():
                return self.client.list_collections()

            collections = await self._retry_async(_list)

            return [
                CollectionInfo(
                    name=c.name,
                    vector_dimension=0,
                    count=c.count(),
                    metadata_schema={},
                    created_at=0,
                    updated_at=time.time()
                )
                for c in collections
            ]

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        async with self._measure_latency(f"delete_collection({name})"):
            def _delete():
                self.client.delete_collection(name)
                if name in self._collections:
                    del self._collections[name]
                return True

            return await self._retry_async(_delete)

    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """Store a vector with metadata"""
        self.metrics["stores"] += 1

        async with self._measure_latency(f"store({collection}/{id})"):
            def _store():
                coll = self.client.get_collection(collection)

                coll.upsert(
                    ids=[id],
                    embeddings=[vector],
                    metadatas=[metadata] if metadata else None,
                    documents=[document] if document else None
                )

            try:
                await self._retry_async(_store)

                return StorageResult(
                    success=True,
                    id=id,
                    collection=collection,
                    timestamp=time.time(),
                    metadata=metadata
                )

            except Exception as e:
                logger.error(f"Failed to store {id} in {collection}: {e}")
                return StorageResult(
                    success=False,
                    id=id,
                    collection=collection,
                    timestamp=time.time(),
                    error=str(e)
                )

    async def store_batch(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> List[StorageResult]:
        """Store multiple vectors in batch"""
        self.metrics["stores"] += len(ids)

        async with self._measure_latency(f"store_batch({collection}, {len(ids)} items)"):
            def _store_batch():
                coll = self.client.get_collection(collection)

                coll.upsert(
                    ids=ids,
                    embeddings=vectors,
                    metadatas=metadatas,
                    documents=documents
                )

            try:
                await self._retry_async(_store_batch)

                return [
                    StorageResult(
                        success=True,
                        id=id,
                        collection=collection,
                        timestamp=time.time()
                    )
                    for id in ids
                ]

            except Exception as e:
                logger.error(f"Failed to store batch in {collection}: {e}")
                return [
                    StorageResult(
                        success=False,
                        id=id,
                        collection=collection,
                        timestamp=time.time(),
                        error=str(e)
                    )
                    for id in ids
                ]

    async def query(
        self,
        collection: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        include_vectors: bool = False
    ) -> List[QueryResult]:
        """Query for similar vectors"""
        self.metrics["queries"] += 1

        async with self._measure_latency(f"query({collection}, limit={limit})"):
            def _query():
                coll = self.client.get_collection(collection)

                # Build query
                kwargs = {
                    "n_results": limit,
                    "include": ["metadatas", "documents", "distances"]
                }

                if include_vectors:
                    kwargs["include"].append("embeddings")

                if filters:
                    kwargs["where"] = filters

                if query_vector:
                    kwargs["query_embeddings"] = [query_vector]
                elif query_text:
                    kwargs["query_texts"] = [query_text]
                else:
                    raise ValueError("Must provide query_vector or query_text")

                return coll.query(**kwargs)

            results = await self._retry_async(_query)

            # Format results
            query_results = []

            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    query_results.append(QueryResult(
                        id=results['ids'][0][i],
                        score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        vector=results['embeddings'][0][i] if include_vectors and 'embeddings' in results else None,
                        metadata=results['metadatas'][0][i] if 'metadatas' in results else None,
                        document=results['documents'][0][i] if 'documents' in results else None
                    ))

            return query_results

    async def query_by_id(
        self,
        collection: str,
        id: str,
        include_vector: bool = False
    ) -> Optional[QueryResult]:
        """Get a specific item by ID"""
        async with self._measure_latency(f"query_by_id({collection}/{id})"):
            def _get():
                coll = self.client.get_collection(collection)

                include = ["metadatas", "documents"]
                if include_vector:
                    include.append("embeddings")

                return coll.get(ids=[id], include=include)

            results = await self._retry_async(_get)

            if results and results['ids']:
                return QueryResult(
                    id=results['ids'][0],
                    score=1.0,
                    vector=results['embeddings'][0] if include_vector and 'embeddings' in results else None,
                    metadata=results['metadatas'][0] if 'metadatas' in results else None,
                    document=results['documents'][0] if 'documents' in results else None
                )

            return None

    async def update(
        self,
        collection: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> UpdateResult:
        """Update an existing item"""
        self.metrics["updates"] += 1

        async with self._measure_latency(f"update({collection}/{id})"):
            def _update():
                coll = self.client.get_collection(collection)

                update_kwargs = {"ids": [id]}
                fields_updated = []

                if vector is not None:
                    update_kwargs["embeddings"] = [vector]
                    fields_updated.append("vector")

                if metadata is not None:
                    update_kwargs["metadatas"] = [metadata]
                    fields_updated.append("metadata")

                if document is not None:
                    update_kwargs["documents"] = [document]
                    fields_updated.append("document")

                coll.update(**update_kwargs)
                return fields_updated

            try:
                fields_updated = await self._retry_async(_update)

                return UpdateResult(
                    success=True,
                    id=id,
                    collection=collection,
                    fields_updated=fields_updated
                )

            except Exception as e:
                logger.error(f"Failed to update {id} in {collection}: {e}")
                return UpdateResult(
                    success=False,
                    id=id,
                    collection=collection,
                    fields_updated=[],
                    error=str(e)
                )

    async def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """Insert or update (same as store for ChromaDB)"""
        return await self.store(collection, id, vector, metadata, document)

    async def delete(self, collection: str, id: str) -> DeleteResult:
        """Delete a single item"""
        self.metrics["deletes"] += 1

        async with self._measure_latency(f"delete({collection}/{id})"):
            def _delete():
                coll = self.client.get_collection(collection)
                coll.delete(ids=[id])

            try:
                await self._retry_async(_delete)

                return DeleteResult(
                    success=True,
                    id=id,
                    collection=collection
                )

            except Exception as e:
                logger.error(f"Failed to delete {id} from {collection}: {e}")
                return DeleteResult(
                    success=False,
                    id=id,
                    collection=collection,
                    error=str(e)
                )

    async def delete_batch(
        self,
        collection: str,
        ids: List[str]
    ) -> List[DeleteResult]:
        """Delete multiple items"""
        self.metrics["deletes"] += len(ids)

        async with self._measure_latency(f"delete_batch({collection}, {len(ids)} items)"):
            def _delete_batch():
                coll = self.client.get_collection(collection)
                coll.delete(ids=ids)

            try:
                await self._retry_async(_delete_batch)

                return [
                    DeleteResult(success=True, id=id, collection=collection)
                    for id in ids
                ]

            except Exception as e:
                logger.error(f"Failed to delete batch from {collection}: {e}")
                return [
                    DeleteResult(success=False, id=id, collection=collection, error=str(e))
                    for id in ids
                ]

    async def delete_by_filter(
        self,
        collection: str,
        filters: Dict[str, Any]
    ) -> int:
        """Delete all items matching filters"""
        async with self._measure_latency(f"delete_by_filter({collection})"):
            def _delete_filtered():
                coll = self.client.get_collection(collection)

                # Get IDs matching filter
                results = coll.get(where=filters, include=[])

                if results and results['ids']:
                    coll.delete(ids=results['ids'])
                    return len(results['ids'])

                return 0

            deleted_count = await self._retry_async(_delete_filtered)
            self.metrics["deletes"] += deleted_count

            return deleted_count

    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count items in collection"""
        async with self._measure_latency(f"count({collection})"):
            def _count():
                coll = self.client.get_collection(collection)

                if filters:
                    results = coll.get(where=filters, include=[])
                    return len(results['ids']) if results and results['ids'] else 0

                return coll.count()

            return await self._retry_async(_count)

    async def scroll(
        self,
        collection: str,
        batch_size: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[List[QueryResult]]:
        """Iterate through all items in collection in batches"""

        def _get_batch(offset: int):
            coll = self.client.get_collection(collection)

            kwargs = {
                "limit": batch_size,
                "offset": offset,
                "include": ["metadatas", "documents", "embeddings"]
            }

            if filters:
                kwargs["where"] = filters

            return coll.get(**kwargs)

        offset = 0

        while True:
            results = await self._retry_async(_get_batch, offset)

            if not results or not results['ids']:
                break

            batch = [
                QueryResult(
                    id=results['ids'][i],
                    score=1.0,
                    vector=results['embeddings'][i] if 'embeddings' in results else None,
                    metadata=results['metadatas'][i] if 'metadatas' in results else None,
                    document=results['documents'][i] if 'documents' in results else None
                )
                for i in range(len(results['ids']))
            ]

            yield batch

            if len(batch) < batch_size:
                break

            offset += batch_size

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_ops = (
            self.metrics["queries"] +
            self.metrics["stores"] +
            self.metrics["updates"] +
            self.metrics["deletes"]
        )

        return {
            **self.metrics,
            "total_operations": total_ops,
            "error_rate": self.metrics["errors"] / total_ops if total_ops > 0 else 0,
            "avg_latency_ms": (
                self.metrics["total_latency_ms"] / total_ops
                if total_ops > 0 else 0
            )
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "queries": 0,
            "stores": 0,
            "updates": 0,
            "deletes": 0,
            "errors": 0,
            "retries": 0,
            "total_latency_ms": 0.0
        }


# Register adapter
try:
    from protocols.storage import StorageBackendRegistry
    StorageBackendRegistry.register("chromadb", ChromaDBAdapter)
except ImportError:
    pass  # Registry not available yet
