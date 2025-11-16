"""
Tests for Storage Protocol

Tests InMemoryStorage implementation, collection management,
CRUD operations, querying, and performance.
"""

import pytest
import asyncio
import numpy as np
from protocols.storage import (
    InMemoryStorage,
    StorageBackendRegistry,
    DistanceMetric,
    StorageResult,
    QueryResult,
    UpdateResult,
    DeleteResult,
    CollectionInfo
)


# ============================================================================
# Collection Management Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_create_collection():
    """Test creating a collection"""
    storage = InMemoryStorage()

    collection_info = await storage.create_collection(
        name="test_collection",
        vector_dimension=128,
        distance_metric=DistanceMetric.COSINE
    )

    assert collection_info.name == "test_collection"
    assert collection_info.vector_dimension == 128
    assert collection_info.count == 0


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_create_duplicate_collection():
    """Test creating duplicate collection raises error"""
    storage = InMemoryStorage()

    await storage.create_collection("test", vector_dimension=128)

    # Creating again should raise error
    with pytest.raises(ValueError, match="already exists"):
        await storage.create_collection("test", vector_dimension=128)


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_get_collection(storage):
    """Test retrieving collection info"""
    collection_info = await storage.get_collection("test_collection")

    assert collection_info.name == "test_collection"
    assert collection_info.vector_dimension == 128


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_get_nonexistent_collection():
    """Test getting nonexistent collection raises error"""
    storage = InMemoryStorage()

    with pytest.raises(ValueError, match="does not exist"):
        await storage.get_collection("nonexistent")


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_list_collections():
    """Test listing all collections"""
    storage = InMemoryStorage()

    await storage.create_collection("collection1", vector_dimension=64)
    await storage.create_collection("collection2", vector_dimension=128)
    await storage.create_collection("collection3", vector_dimension=256)

    collections = await storage.list_collections()

    assert len(collections) == 3
    names = {c.name for c in collections}
    assert names == {"collection1", "collection2", "collection3"}


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_delete_collection():
    """Test deleting a collection"""
    storage = InMemoryStorage()

    await storage.create_collection("to_delete", vector_dimension=128)

    # Verify it exists
    collections = await storage.list_collections()
    assert any(c.name == "to_delete" for c in collections)

    # Delete it
    success = await storage.delete_collection("to_delete")
    assert success is True

    # Verify it's gone
    collections = await storage.list_collections()
    assert not any(c.name == "to_delete" for c in collections)


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_delete_nonexistent_collection():
    """Test deleting nonexistent collection returns False"""
    storage = InMemoryStorage()

    success = await storage.delete_collection("nonexistent")
    assert success is False


# ============================================================================
# Storage Operations Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_store_vector(storage):
    """Test storing a single vector"""
    vector = np.random.randn(128).tolist()
    metadata = {"type": "test", "index": 0}

    result = await storage.store(
        collection="test_collection",
        id="test_item_1",
        vector=vector,
        metadata=metadata,
        document="Test document"
    )

    assert isinstance(result, StorageResult)
    assert result.success is True
    assert result.id == "test_item_1"
    assert result.collection == "test_collection"


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_store_wrong_dimension():
    """Test storing vector with wrong dimension fails"""
    storage = InMemoryStorage()

    await storage.create_collection("test", vector_dimension=128)

    # Try to store 64-dim vector in 128-dim collection
    vector = np.random.randn(64).tolist()

    result = await storage.store(
        collection="test",
        id="wrong_dim",
        vector=vector
    )

    assert result.success is False
    assert "dimension" in result.error.lower()


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_store_to_nonexistent_collection():
    """Test storing to nonexistent collection raises error"""
    storage = InMemoryStorage()

    vector = np.random.randn(128).tolist()

    with pytest.raises(ValueError, match="does not exist"):
        await storage.store(
            collection="nonexistent",
            id="test",
            vector=vector
        )


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_store_batch(storage):
    """Test batch storage operation"""
    vectors = [np.random.randn(128).tolist() for _ in range(10)]
    ids = [f"batch_item_{i}" for i in range(10)]
    metadatas = [{"index": i} for i in range(10)]

    results = await storage.store_batch(
        collection="test_collection",
        ids=ids,
        vectors=vectors,
        metadatas=metadatas
    )

    assert len(results) == 10
    assert all(r.success for r in results)


# ============================================================================
# Query Operations Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_query_basic(storage_with_data):
    """Test basic vector query"""
    # Get one of the stored vectors
    result = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_0",
        include_vector=True
    )

    query_vector = result.vector

    # Query with it
    results = await storage_with_data.query(
        collection="test_collection",
        query_vector=query_vector,
        limit=5
    )

    assert len(results) <= 5
    assert isinstance(results[0], QueryResult)

    # First result should be the query vector itself
    assert results[0].id == "item_0"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_query_with_filters(storage_with_data):
    """Test query with metadata filters"""
    # Get query vector
    result = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_0",
        include_vector=True
    )

    query_vector = result.vector

    # Query with filter
    results = await storage_with_data.query(
        collection="test_collection",
        query_vector=query_vector,
        filters={"type": "test"},
        limit=5
    )

    # All results should match filter
    for r in results:
        assert r.metadata["type"] == "test"


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_query_limit(storage_with_data):
    """Test query limit parameter"""
    result = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_0",
        include_vector=True
    )

    query_vector = result.vector

    results_3 = await storage_with_data.query(
        collection="test_collection",
        query_vector=query_vector,
        limit=3
    )

    results_7 = await storage_with_data.query(
        collection="test_collection",
        query_vector=query_vector,
        limit=7
    )

    assert len(results_3) == 3
    assert len(results_7) == 7


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_query_by_id(storage_with_data):
    """Test querying specific item by ID"""
    result = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_5",
        include_vector=True
    )

    assert result is not None
    assert result.id == "item_5"
    assert result.metadata["index"] == 5
    assert result.vector is not None


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_query_by_nonexistent_id(storage_with_data):
    """Test querying nonexistent ID returns None"""
    result = await storage_with_data.query_by_id(
        collection="test_collection",
        id="nonexistent"
    )

    assert result is None


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_query_batch(storage_with_data):
    """Test batch query operation"""
    # Get 3 query vectors
    query_vectors = []
    for i in range(3):
        result = await storage_with_data.query_by_id(
            collection="test_collection",
            id=f"item_{i}",
            include_vector=True
        )
        query_vectors.append(result.vector)

    # Batch query
    results = await storage_with_data.query_batch(
        collection="test_collection",
        query_vectors=query_vectors,
        limit=5
    )

    assert len(results) == 3  # One result list per query
    assert all(len(r) <= 5 for r in results)


# ============================================================================
# Update Operations Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_update_vector(storage_with_data):
    """Test updating a vector"""
    new_vector = np.random.randn(128).tolist()

    result = await storage_with_data.update(
        collection="test_collection",
        id="item_0",
        vector=new_vector
    )

    assert isinstance(result, UpdateResult)
    assert result.success is True
    assert "vector" in result.fields_updated


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_update_metadata(storage_with_data):
    """Test updating metadata"""
    new_metadata = {"updated": True, "new_field": "value"}

    result = await storage_with_data.update(
        collection="test_collection",
        id="item_0",
        metadata=new_metadata
    )

    assert result.success is True
    assert "metadata" in result.fields_updated

    # Verify update
    item = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_0"
    )

    assert item.metadata["updated"] is True
    assert item.metadata["new_field"] == "value"


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_update_nonexistent_item(storage_with_data):
    """Test updating nonexistent item fails"""
    result = await storage_with_data.update(
        collection="test_collection",
        id="nonexistent",
        metadata={"test": True}
    )

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_upsert_new_item(storage):
    """Test upsert creates new item if not exists"""
    vector = np.random.randn(128).tolist()

    result = await storage.upsert(
        collection="test_collection",
        id="new_item",
        vector=vector,
        metadata={"new": True}
    )

    assert result.success is True

    # Verify it was created
    item = await storage.query_by_id(
        collection="test_collection",
        id="new_item"
    )

    assert item is not None


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_upsert_existing_item(storage_with_data):
    """Test upsert updates existing item"""
    new_vector = np.random.randn(128).tolist()

    result = await storage_with_data.upsert(
        collection="test_collection",
        id="item_0",
        vector=new_vector,
        metadata={"updated": True}
    )

    assert result.success is True


# ============================================================================
# Delete Operations Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_delete_item(storage_with_data):
    """Test deleting a single item"""
    result = await storage_with_data.delete(
        collection="test_collection",
        id="item_0"
    )

    assert isinstance(result, DeleteResult)
    assert result.success is True

    # Verify deletion
    item = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_0"
    )

    assert item is None


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_delete_nonexistent_item(storage_with_data):
    """Test deleting nonexistent item returns failure"""
    result = await storage_with_data.delete(
        collection="test_collection",
        id="nonexistent"
    )

    assert result.success is False


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_delete_batch(storage_with_data):
    """Test batch delete operation"""
    ids_to_delete = ["item_0", "item_1", "item_2"]

    results = await storage_with_data.delete_batch(
        collection="test_collection",
        ids=ids_to_delete
    )

    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_delete_by_filter(storage_with_data):
    """Test deleting items by filter"""
    # Delete all items with index < 5
    deleted_count = await storage_with_data.delete_by_filter(
        collection="test_collection",
        filters={"index": 3}
    )

    # Should have deleted item_3
    assert deleted_count == 1

    # Verify deletion
    item = await storage_with_data.query_by_id(
        collection="test_collection",
        id="item_3"
    )

    assert item is None


# ============================================================================
# Count and Scroll Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_count_all(storage_with_data):
    """Test counting all items in collection"""
    count = await storage_with_data.count("test_collection")

    assert count == 10


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_count_with_filter(storage_with_data):
    """Test counting with metadata filter"""
    count = await storage_with_data.count(
        collection="test_collection",
        filters={"type": "test"}
    )

    # All items have type="test"
    assert count == 10


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_scroll_all(storage_with_data):
    """Test scrolling through all items"""
    all_items = []

    async for batch in storage_with_data.scroll(
        collection="test_collection",
        batch_size=3
    ):
        all_items.extend(batch)

    assert len(all_items) == 10


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_scroll_with_filter(storage_with_data):
    """Test scrolling with filter"""
    filtered_items = []

    async for batch in storage_with_data.scroll(
        collection="test_collection",
        batch_size=3,
        filters={"type": "test"}
    ):
        filtered_items.extend(batch)

    # All items match filter
    assert len(filtered_items) == 10


# ============================================================================
# Distance Metric Tests
# ============================================================================

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_cosine_distance_metric():
    """Test cosine similarity metric"""
    storage = InMemoryStorage()

    await storage.create_collection(
        name="cosine_test",
        vector_dimension=3,
        distance_metric=DistanceMetric.COSINE
    )

    # Store orthogonal vectors
    await storage.store(
        collection="cosine_test",
        id="vec1",
        vector=[1.0, 0.0, 0.0]
    )

    await storage.store(
        collection="cosine_test",
        id="vec2",
        vector=[0.0, 1.0, 0.0]
    )

    # Query vec1
    results = await storage.query(
        collection="cosine_test",
        query_vector=[1.0, 0.0, 0.0],
        limit=2
    )

    # vec1 should be most similar (score=1.0)
    # vec2 should be least similar (score=0.0, orthogonal)
    assert results[0].id == "vec1"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.unit
async def test_euclidean_distance_metric():
    """Test Euclidean distance metric"""
    storage = InMemoryStorage()

    await storage.create_collection(
        name="euclidean_test",
        vector_dimension=2,
        distance_metric=DistanceMetric.EUCLIDEAN
    )

    await storage.store(
        collection="euclidean_test",
        id="origin",
        vector=[0.0, 0.0]
    )

    await storage.store(
        collection="euclidean_test",
        id="far",
        vector=[10.0, 10.0]
    )

    # Query from origin
    results = await storage.query(
        collection="euclidean_test",
        query_vector=[0.0, 0.0],
        limit=2
    )

    # Origin should be first (distance=0, but score is negative distance)
    assert results[0].id == "origin"


# ============================================================================
# Storage Backend Registry Tests
# ============================================================================

@pytest.mark.unit
def test_backend_registry_memory():
    """Test memory backend is registered"""
    backends = StorageBackendRegistry.list_backends()

    assert "memory" in backends


@pytest.mark.unit
def test_backend_registry_create():
    """Test creating backend from registry"""
    storage = StorageBackendRegistry.create("memory")

    assert isinstance(storage, InMemoryStorage)


@pytest.mark.unit
def test_backend_registry_unknown():
    """Test creating unknown backend raises error"""
    with pytest.raises(ValueError, match="Unknown backend"):
        StorageBackendRegistry.create("nonexistent_backend")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="function")
async def test_large_batch_storage():
    """Test storing large batch of vectors"""
    storage = InMemoryStorage()

    await storage.create_collection(
        name="large_test",
        vector_dimension=128
    )

    # Store 1000 vectors
    num_vectors = 1000
    vectors = [np.random.randn(128).tolist() for _ in range(num_vectors)]
    ids = [f"vec_{i}" for i in range(num_vectors)]

    results = await storage.store_batch(
        collection="large_test",
        ids=ids,
        vectors=vectors
    )

    assert len(results) == num_vectors
    assert all(r.success for r in results)


@pytest.mark.slow
@pytest.mark.asyncio(loop_scope="function")
async def test_query_performance_large_collection():
    """Test query performance on large collection"""
    storage = InMemoryStorage()

    await storage.create_collection(
        name="perf_test",
        vector_dimension=128
    )

    # Store 1000 vectors
    num_vectors = 1000
    vectors = [np.random.randn(128).tolist() for _ in range(num_vectors)]
    ids = [f"vec_{i}" for i in range(num_vectors)]

    await storage.store_batch(
        collection="perf_test",
        ids=ids,
        vectors=vectors
    )

    # Query should be reasonably fast
    import time

    query_vector = np.random.randn(128).tolist()

    start = time.time()
    results = await storage.query(
        collection="perf_test",
        query_vector=query_vector,
        limit=10
    )
    duration = time.time() - start

    assert len(results) == 10
    assert duration < 1.0  # Should complete in < 1 second


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="function")
async def test_full_storage_lifecycle():
    """Test complete storage lifecycle"""
    storage = InMemoryStorage()

    # Create collection
    await storage.create_collection(
        name="lifecycle_test",
        vector_dimension=64
    )

    # Store items
    vectors = [np.random.randn(64).tolist() for _ in range(5)]
    ids = [f"item_{i}" for i in range(5)]

    await storage.store_batch(
        collection="lifecycle_test",
        ids=ids,
        vectors=vectors
    )

    # Query
    results = await storage.query(
        collection="lifecycle_test",
        query_vector=vectors[0],
        limit=3
    )

    assert len(results) == 3

    # Update
    await storage.update(
        collection="lifecycle_test",
        id="item_0",
        metadata={"updated": True}
    )

    # Delete
    await storage.delete(
        collection="lifecycle_test",
        id="item_4"
    )

    # Verify
    count = await storage.count("lifecycle_test")
    assert count == 4  # 5 - 1 deleted

    # Clean up
    await storage.delete_collection("lifecycle_test")

    collections = await storage.list_collections()
    assert not any(c.name == "lifecycle_test" for c in collections)
