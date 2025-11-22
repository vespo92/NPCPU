"""
Pinecone Adapter - Cloud-Scale Vector Storage

Implements VectorStorageProtocol for Pinecone with:
- Cloud-scale storage (billions of vectors)
- Low latency worldwide
- Managed infrastructure
- Async support
- Retry logic

Based on Week 1 roadmap: Day 3-4 Pinecone Adapter
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import time
import logging
from contextlib import asynccontextmanager

try:
    import pinecone
    from pinecone import Pinecone as PineconeClient
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None
    PineconeClient = None


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


class PineconeAdapter:
    """
    Pinecone adapter for cloud-scale vector storage.

    Benefits:
    - Scales to billions of vectors
    - Low latency worldwide
    - Managed infrastructure
    - High availability

    Features:
    - Async-compatible interface
    - Automatic retry with exponential backoff
    - Namespace support for multi-tenant
    - Metadata filtering
    - Performance metrics

    Example:
        adapter = PineconeAdapter(
            api_key="your-api-key",
            environment="us-west1-gcp"
        )
        await adapter.create_collection("memories", vector_dimension=384)
        await adapter.store("memories", "id1", [0.1, 0.2, ...], {"type": "experience"})
        results = await adapter.query("memories", [0.1, 0.2, ...], limit=5)

    Note:
        Requires pinecone-client package: pip install pinecone-client
    """

    def __init__(
        self,
        api_key: str,
        environment: Optional[str] = None,
        host: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Pinecone adapter.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., "us-west1-gcp")
            host: Optional host override for serverless
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries
        """
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone is not installed. Install with: pip install pinecone-client"
            )

        self.api_key = api_key
        self.environment = environment
        self.host = host
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize client
        logger.info(f"Connecting to Pinecone (environment: {environment})")
        self.client = PineconeClient(api_key=api_key)

        # Index cache
        self._indexes: Dict[str, Any] = {}

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

    async def _retry_async(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                return result

            except Exception as e:
                last_error = e
                self.metrics["retries"] += 1

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
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
        """Map protocol distance metric to Pinecone metric"""
        mapping = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "euclidean",
            DistanceMetric.DOT_PRODUCT: "dotproduct"
        }
        return mapping.get(metric, "cosine")

    def _get_index(self, name: str):
        """Get or create index reference"""
        if name not in self._indexes:
            self._indexes[name] = self.client.Index(name)
        return self._indexes[name]

    async def create_collection(
        self,
        name: str,
        vector_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        metadata_schema: Optional[Dict[str, type]] = None,
        replicas: int = 1,
        pods: int = 1,
        pod_type: str = "p1.x1"
    ) -> CollectionInfo:
        """
        Create a new Pinecone index (collection).

        Args:
            name: Index name
            vector_dimension: Dimension of vectors
            distance_metric: Distance metric for similarity
            metadata_schema: Optional metadata schema
            replicas: Number of replicas (for availability)
            pods: Number of pods (for capacity)
            pod_type: Pod type (p1.x1, p1.x2, s1.x1, etc.)
        """
        async with self._measure_latency(f"create_collection({name})"):
            def _create():
                metric = self._map_distance_metric(distance_metric)

                # Check if index exists
                existing = self.client.list_indexes()
                if name in [idx.name for idx in existing]:
                    logger.info(f"Index {name} already exists")
                    return self._get_index(name)

                # Create index
                self.client.create_index(
                    name=name,
                    dimension=vector_dimension,
                    metric=metric,
                    spec={
                        "pod": {
                            "environment": self.environment,
                            "replicas": replicas,
                            "pods": pods,
                            "pod_type": pod_type
                        }
                    } if self.environment else {
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )

                # Wait for index to be ready
                time.sleep(1)
                return self._get_index(name)

            await self._retry_async(_create)

            return CollectionInfo(
                name=name,
                vector_dimension=vector_dimension,
                count=0,
                metadata_schema=metadata_schema or {},
                created_at=time.time(),
                updated_at=time.time()
            )

    async def get_collection(self, name: str) -> CollectionInfo:
        """Get collection (index) information"""
        async with self._measure_latency(f"get_collection({name})"):
            def _get():
                index = self._get_index(name)
                stats = index.describe_index_stats()
                return stats

            stats = await self._retry_async(_get)

            return CollectionInfo(
                name=name,
                vector_dimension=stats.get("dimension", 0),
                count=stats.get("total_vector_count", 0),
                metadata_schema={},
                created_at=0,
                updated_at=time.time()
            )

    async def list_collections(self) -> List[CollectionInfo]:
        """List all indexes"""
        async with self._measure_latency("list_collections"):
            def _list():
                return self.client.list_indexes()

            indexes = await self._retry_async(_list)

            return [
                CollectionInfo(
                    name=idx.name,
                    vector_dimension=idx.dimension,
                    count=0,
                    metadata_schema={},
                    created_at=0,
                    updated_at=time.time()
                )
                for idx in indexes
            ]

    async def delete_collection(self, name: str) -> bool:
        """Delete an index"""
        async with self._measure_latency(f"delete_collection({name})"):
            def _delete():
                self.client.delete_index(name)
                if name in self._indexes:
                    del self._indexes[name]
                return True

            return await self._retry_async(_delete)

    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> StorageResult:
        """Store a vector with metadata"""
        self.metrics["stores"] += 1

        async with self._measure_latency(f"store({collection}/{id})"):
            def _store():
                index = self._get_index(collection)
                index.upsert(
                    vectors=[{
                        "id": id,
                        "values": vector,
                        "metadata": metadata or {}
                    }],
                    namespace=namespace
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
        namespace: str = "",
        batch_size: int = 100
    ) -> List[StorageResult]:
        """Store multiple vectors in batches"""
        self.metrics["stores"] += len(ids)

        async with self._measure_latency(f"store_batch({collection}, {len(ids)} items)"):
            def _store_batch():
                index = self._get_index(collection)

                # Build vectors list
                vectors_list = []
                for i, (id_, vec) in enumerate(zip(ids, vectors)):
                    item = {
                        "id": id_,
                        "values": vec
                    }
                    if metadatas and i < len(metadatas):
                        item["metadata"] = metadatas[i]
                    vectors_list.append(item)

                # Upsert in batches
                for i in range(0, len(vectors_list), batch_size):
                    batch = vectors_list[i:i + batch_size]
                    index.upsert(vectors=batch, namespace=namespace)

            try:
                await self._retry_async(_store_batch)

                return [
                    StorageResult(
                        success=True,
                        id=id_,
                        collection=collection,
                        timestamp=time.time()
                    )
                    for id_ in ids
                ]

            except Exception as e:
                logger.error(f"Failed to store batch in {collection}: {e}")
                return [
                    StorageResult(
                        success=False,
                        id=id_,
                        collection=collection,
                        timestamp=time.time(),
                        error=str(e)
                    )
                    for id_ in ids
                ]

    async def query(
        self,
        collection: str,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        namespace: str = "",
        include_vectors: bool = False,
        include_metadata: bool = True
    ) -> List[QueryResult]:
        """Query for similar vectors"""
        self.metrics["queries"] += 1

        async with self._measure_latency(f"query({collection}, limit={limit})"):
            def _query():
                index = self._get_index(collection)

                return index.query(
                    vector=query_vector,
                    top_k=limit,
                    namespace=namespace,
                    filter=filters,
                    include_values=include_vectors,
                    include_metadata=include_metadata
                )

            results = await self._retry_async(_query)

            query_results = []
            for match in results.get("matches", []):
                query_results.append(QueryResult(
                    id=match["id"],
                    score=match.get("score", 0.0),
                    vector=match.get("values") if include_vectors else None,
                    metadata=match.get("metadata"),
                    document=None
                ))

            return query_results

    async def query_by_id(
        self,
        collection: str,
        id: str,
        namespace: str = "",
        include_vector: bool = False
    ) -> Optional[QueryResult]:
        """Fetch a specific item by ID"""
        async with self._measure_latency(f"query_by_id({collection}/{id})"):
            def _fetch():
                index = self._get_index(collection)
                return index.fetch(ids=[id], namespace=namespace)

            results = await self._retry_async(_fetch)

            vectors = results.get("vectors", {})
            if id in vectors:
                item = vectors[id]
                return QueryResult(
                    id=id,
                    score=1.0,
                    vector=item.get("values") if include_vector else None,
                    metadata=item.get("metadata"),
                    document=None
                )

            return None

    async def update(
        self,
        collection: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> UpdateResult:
        """Update an existing item"""
        self.metrics["updates"] += 1

        async with self._measure_latency(f"update({collection}/{id})"):
            def _update():
                index = self._get_index(collection)
                fields_updated = []

                update_kwargs = {"id": id, "namespace": namespace}

                if vector is not None:
                    update_kwargs["values"] = vector
                    fields_updated.append("vector")

                if metadata is not None:
                    update_kwargs["set_metadata"] = metadata
                    fields_updated.append("metadata")

                index.update(**update_kwargs)
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

    async def delete(
        self,
        collection: str,
        id: str,
        namespace: str = ""
    ) -> DeleteResult:
        """Delete a single item"""
        self.metrics["deletes"] += 1

        async with self._measure_latency(f"delete({collection}/{id})"):
            def _delete():
                index = self._get_index(collection)
                index.delete(ids=[id], namespace=namespace)

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
        ids: List[str],
        namespace: str = ""
    ) -> List[DeleteResult]:
        """Delete multiple items"""
        self.metrics["deletes"] += len(ids)

        async with self._measure_latency(f"delete_batch({collection}, {len(ids)} items)"):
            def _delete_batch():
                index = self._get_index(collection)
                index.delete(ids=ids, namespace=namespace)

            try:
                await self._retry_async(_delete_batch)

                return [
                    DeleteResult(success=True, id=id_, collection=collection)
                    for id_ in ids
                ]

            except Exception as e:
                logger.error(f"Failed to delete batch from {collection}: {e}")
                return [
                    DeleteResult(success=False, id=id_, collection=collection, error=str(e))
                    for id_ in ids
                ]

    async def delete_by_filter(
        self,
        collection: str,
        filters: Dict[str, Any],
        namespace: str = ""
    ) -> int:
        """Delete all items matching filters"""
        async with self._measure_latency(f"delete_by_filter({collection})"):
            def _delete_filtered():
                index = self._get_index(collection)
                index.delete(filter=filters, namespace=namespace)
                return -1  # Pinecone doesn't return count

            await self._retry_async(_delete_filtered)
            return -1  # Unknown count

    async def count(
        self,
        collection: str,
        namespace: str = ""
    ) -> int:
        """Count items in collection"""
        async with self._measure_latency(f"count({collection})"):
            def _count():
                index = self._get_index(collection)
                stats = index.describe_index_stats()

                if namespace:
                    ns_stats = stats.get("namespaces", {}).get(namespace, {})
                    return ns_stats.get("vector_count", 0)

                return stats.get("total_vector_count", 0)

            return await self._retry_async(_count)

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
    StorageBackendRegistry.register("pinecone", PineconeAdapter)
except ImportError:
    pass
