"""
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
import os
import sys
import tempfile
import shutil
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import numpy (optional for core tests)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Try to import protocol modules (require numpy)
HAS_PROTOCOLS = False
try:
    from protocols.consciousness import GradedConsciousness, ConsciousnessAdapter
    from protocols.storage import InMemoryStorage, DistanceMetric
    from protocols.transformations import Manifold, TransformationLibrary
    from factory.consciousness_factory import ConsciousnessFactory
    HAS_PROTOCOLS = True
except ImportError:
    # Protocol modules not available (likely missing numpy)
    GradedConsciousness = None
    ConsciousnessAdapter = None
    InMemoryStorage = None
    DistanceMetric = None
    Manifold = None
    TransformationLibrary = None
    ConsciousnessFactory = None


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Consciousness Fixtures (require numpy/protocols)
# ============================================================================

@pytest.fixture
def basic_consciousness():
    """Create basic consciousness for testing"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return GradedConsciousness(
        perception_fidelity=0.7,
        reaction_speed=0.6,
        memory_depth=0.8,
        memory_recall_accuracy=0.7,
        introspection_capacity=0.5,
        meta_cognitive_ability=0.3,
        information_integration=0.6,
        intentional_coherence=0.7,
        qualia_richness=0.5,
    )


@pytest.fixture
def high_consciousness():
    """Create high-level consciousness for testing"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return GradedConsciousness(
        perception_fidelity=0.9,
        reaction_speed=0.8,
        memory_depth=0.95,
        memory_recall_accuracy=0.9,
        introspection_capacity=0.9,
        meta_cognitive_ability=0.85,
        information_integration=0.9,
        intentional_coherence=0.95,
        qualia_richness=0.9,
    )


@pytest.fixture
def low_consciousness():
    """Create low-level consciousness for testing"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return GradedConsciousness(
        perception_fidelity=0.2,
        reaction_speed=0.3,
        memory_depth=0.1,
        memory_recall_accuracy=0.1,
        introspection_capacity=0.0,
        meta_cognitive_ability=0.0,
        information_integration=0.1,
        intentional_coherence=0.1,
        qualia_richness=0.05,
    )


@pytest.fixture
def consciousness_adapter():
    """Create consciousness adapter for testing"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return ConsciousnessAdapter()


# ============================================================================
# Storage Fixtures (require numpy/protocols)
# ============================================================================

@pytest.fixture
async def storage():
    """Create in-memory storage for testing"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    storage = InMemoryStorage()

    # Create test collection
    await storage.create_collection(
        name="test_collection",
        vector_dimension=128,
        distance_metric=DistanceMetric.COSINE
    )

    return storage


@pytest.fixture
async def storage_with_data():
    """Create storage with test data"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    storage = InMemoryStorage()

    await storage.create_collection(
        name="test_collection",
        vector_dimension=128,
        distance_metric=DistanceMetric.COSINE
    )

    # Add test data
    test_vectors = [np.random.randn(128).tolist() for _ in range(10)]
    test_ids = [f"item_{i}" for i in range(10)]
    test_metadata = [{"index": i, "type": "test"} for i in range(10)]

    await storage.store_batch(
        collection="test_collection",
        ids=test_ids,
        vectors=test_vectors,
        metadatas=test_metadata
    )

    return storage


# ============================================================================
# Transformation Fixtures (require numpy/protocols)
# ============================================================================

@pytest.fixture
def simple_manifold():
    """Create simple manifold for testing"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    vectors = np.random.randn(50, 64)
    return Manifold(vectors=vectors)


@pytest.fixture
def complex_manifold():
    """Create complex manifold with adjacency matrix"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    vectors = np.random.randn(100, 128)

    # Create random adjacency matrix (sparse)
    adjacency = np.zeros((100, 100))
    for i in range(100):
        # Connect to 3-5 random neighbors
        num_neighbors = np.random.randint(3, 6)
        neighbors = np.random.choice(100, num_neighbors, replace=False)
        adjacency[i, neighbors] = 1
        adjacency[neighbors, i] = 1

    metadata = {
        "name": "complex_test_manifold",
        "type": "test",
        "created": "pytest"
    }

    return Manifold(vectors=vectors, adjacency=adjacency, metadata=metadata)


@pytest.fixture
def transformation_library():
    """Return transformation library"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return TransformationLibrary


# ============================================================================
# Factory Fixtures
# ============================================================================

@pytest.fixture
def temp_config_dir():
    """Create temporary config directory for testing"""
    temp_dir = tempfile.mkdtemp()

    # Create a test YAML config
    test_yaml = """
model_type: "graded"
name: "Test Consciousness"
description: "Test model for pytest"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.8
    weight: 1.0
    min: 0.0
    max: 1.0
    description: "Test perception"

  reaction_speed:
    default: 0.7
    weight: 0.8

  memory_depth:
    default: 0.6
    weight: 1.2

aggregation: "weighted_mean"

thresholds:
  dormant: 0.0
  reactive: 0.2
  aware: 0.4
  reflective: 0.6
  meta_aware: 0.8
  transcendent: 0.95
"""

    with open(os.path.join(temp_dir, "test_model.yaml"), 'w') as f:
        f.write(test_yaml)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def consciousness_factory(temp_config_dir):
    """Create consciousness factory with temp config dir"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return ConsciousnessFactory(config_dir=temp_config_dir)


@pytest.fixture
def real_consciousness_factory():
    """Create consciousness factory with real config dir"""
    if not HAS_PROTOCOLS:
        pytest.skip("Requires protocols module (numpy)")
    return ConsciousnessFactory()


# ============================================================================
# Helper Fixtures (require numpy)
# ============================================================================

@pytest.fixture
def random_vectors():
    """Generate random test vectors"""
    if not HAS_NUMPY:
        pytest.skip("Requires numpy")
    def _generate(num_vectors: int, dimension: int):
        return np.random.randn(num_vectors, dimension)
    return _generate


@pytest.fixture
def similarity_calculator():
    """Calculate cosine similarity between vectors"""
    if not HAS_NUMPY:
        pytest.skip("Requires numpy")
    def _calculate(vec1, vec2) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return _calculate


# ============================================================================
# Core Module Fixtures (for new abstraction layer tests)
# ============================================================================

@pytest.fixture
def fresh_event_bus():
    """Provide a fresh event bus for each test"""
    from core.events import EventBus
    return EventBus()


@pytest.fixture
def fresh_hook_manager():
    """Provide a fresh hook manager for each test"""
    from core.plugins import HookManager
    return HookManager()


@pytest.fixture
def simple_organism():
    """Provide a simple organism for testing"""
    from core.simple_organism import SimpleOrganism
    return SimpleOrganism("TestOrganism")


@pytest.fixture
def simple_population():
    """Provide a simple population for testing"""
    from core.simple_organism import SimplePopulation
    return SimplePopulation("TestPopulation")


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests"""
    from core import events, plugins

    # Store original global instances
    original_bus = events._global_bus
    original_manager = plugins._global_manager

    yield

    # Restore original state
    events._global_bus = original_bus
    plugins._global_manager = original_manager
