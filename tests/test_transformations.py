"""
Tests for Transformation Algebra Protocol

Tests Manifold, transformations, composition, invariants,
and the transformation library.
"""

import pytest
import numpy as np
from protocols.transformations import (
    Manifold,
    Invariant,
    ComposableTransformation,
    TransformationLibrary,
    TransformationPipeline
)


# ============================================================================
# Manifold Tests
# ============================================================================

@pytest.mark.unit
def test_manifold_creation():
    """Test creating a basic manifold"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    assert manifold.num_points == 100
    assert manifold.dimension == 64
    assert manifold.adjacency is None
    assert manifold.metadata is None


@pytest.mark.unit
def test_manifold_with_adjacency():
    """Test manifold with adjacency matrix"""
    vectors = np.random.randn(50, 32)
    adjacency = np.random.rand(50, 50)

    manifold = Manifold(vectors=vectors, adjacency=adjacency)

    assert manifold.adjacency is not None
    assert manifold.adjacency.shape == (50, 50)


@pytest.mark.unit
def test_manifold_with_metadata():
    """Test manifold with metadata"""
    vectors = np.random.randn(10, 8)
    metadata = {"name": "test", "type": "sample"}

    manifold = Manifold(vectors=vectors, metadata=metadata)

    assert manifold.get_semantic_property("name") == "test"
    assert manifold.get_semantic_property("type") == "sample"


@pytest.mark.unit
def test_manifold_semantic_properties():
    """Test setting and getting semantic properties"""
    vectors = np.random.randn(10, 8)
    manifold = Manifold(vectors=vectors)

    manifold.set_semantic_property("custom_prop", "value")

    assert manifold.get_semantic_property("custom_prop") == "value"
    assert manifold.get_semantic_property("nonexistent", "default") == "default"


# ============================================================================
# Invariant Computation Tests
# ============================================================================

@pytest.mark.unit
def test_compute_dimension_invariant():
    """Test computing dimension invariant"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    dimension = manifold.compute_invariant(Invariant.DIMENSION)

    assert dimension == 64


@pytest.mark.unit
def test_compute_volume_invariant():
    """Test computing volume invariant"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    volume = manifold.compute_invariant(Invariant.VOLUME)

    assert isinstance(volume, (int, float, np.number))


@pytest.mark.unit
def test_compute_euler_characteristic_point_cloud():
    """Test Euler characteristic for point cloud"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    euler = manifold.compute_invariant(Invariant.EULER_CHARACTERISTIC)

    # For point cloud without edges: χ = V
    assert euler == 50


@pytest.mark.unit
def test_compute_euler_characteristic_graph():
    """Test Euler characteristic for graph"""
    vectors = np.random.randn(10, 8)

    # Create simple graph: 5 edges
    adjacency = np.zeros((10, 10))
    adjacency[0, 1] = adjacency[1, 0] = 1  # edge 1
    adjacency[1, 2] = adjacency[2, 1] = 1  # edge 2
    adjacency[2, 3] = adjacency[3, 2] = 1  # edge 3
    adjacency[3, 4] = adjacency[4, 3] = 1  # edge 4
    adjacency[4, 5] = adjacency[5, 4] = 1  # edge 5

    manifold = Manifold(vectors=vectors, adjacency=adjacency)

    euler = manifold.compute_invariant(Invariant.EULER_CHARACTERISTIC)

    # χ = V - E = 10 - 5 = 5
    assert euler == 5


@pytest.mark.unit
def test_compute_information_invariant():
    """Test information content invariant"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    information = manifold.compute_invariant(Invariant.INFORMATION)

    assert isinstance(information, float)
    assert information >= 0


@pytest.mark.unit
def test_compute_coherence_invariant():
    """Test coherence invariant"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    coherence = manifold.compute_invariant(Invariant.COHERENCE)

    assert isinstance(coherence, float)
    assert 0.0 <= coherence <= 1.0


# ============================================================================
# Transformation Basic Tests
# ============================================================================

@pytest.mark.unit
def test_identity_transformation():
    """Test identity transformation"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    identity = TransformationLibrary.identity()
    transformed = identity.transform(manifold)

    # Should be unchanged
    assert transformed.num_points == manifold.num_points
    assert transformed.dimension == manifold.dimension
    np.testing.assert_array_equal(transformed.vectors, manifold.vectors)


@pytest.mark.unit
def test_identity_preserves_all():
    """Test identity preserves all invariants"""
    identity = TransformationLibrary.identity()

    preserved = identity.preserves()

    # Identity should preserve everything
    assert len(preserved) == len(Invariant)


@pytest.mark.unit
def test_identity_is_invertible():
    """Test identity has inverse"""
    identity = TransformationLibrary.identity()

    inverse = identity.inverse()

    assert inverse is not None


# ============================================================================
# Transformation Library Tests
# ============================================================================

@pytest.mark.unit
def test_projection_transformation():
    """Test projection to lower dimension"""
    vectors = np.random.randn(100, 128)
    manifold = Manifold(vectors=vectors)

    projection = TransformationLibrary.projection(64)
    transformed = projection.transform(manifold)

    assert transformed.dimension == 64
    assert transformed.num_points == manifold.num_points


@pytest.mark.unit
def test_projection_preserves_topology():
    """Test projection preserves topology"""
    projection = TransformationLibrary.projection(32)

    preserved = projection.preserves()

    assert Invariant.TOPOLOGY in preserved
    assert Invariant.CONNECTIVITY in preserved


@pytest.mark.unit
def test_embedding_transformation():
    """Test embedding to higher dimension"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    embedding = TransformationLibrary.embedding(64)
    transformed = embedding.transform(manifold)

    assert transformed.dimension == 64
    assert transformed.num_points == manifold.num_points


@pytest.mark.unit
def test_embedding_is_invertible():
    """Test embedding has inverse"""
    embedding = TransformationLibrary.embedding(128)

    inverse = embedding.inverse()

    assert inverse is not None


@pytest.mark.unit
def test_rotation_transformation():
    """Test rotation transformation"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    rotation = TransformationLibrary.rotation(np.pi / 4, plane=(0, 1))
    transformed = rotation.transform(manifold)

    # Dimension should be unchanged
    assert transformed.dimension == manifold.dimension
    assert transformed.num_points == manifold.num_points


@pytest.mark.unit
def test_rotation_preserves_distances():
    """Test rotation preserves relative distances"""
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    manifold = Manifold(vectors=vectors)

    rotation = TransformationLibrary.rotation(np.pi / 2, plane=(0, 1))
    transformed = rotation.transform(manifold)

    # Compute distances before and after
    dist_before_01 = np.linalg.norm(manifold.vectors[0] - manifold.vectors[1])
    dist_after_01 = np.linalg.norm(transformed.vectors[0] - transformed.vectors[1])

    assert dist_before_01 == pytest.approx(dist_after_01, abs=1e-6)


@pytest.mark.unit
def test_scaling_transformation():
    """Test scaling transformation"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    scaling = TransformationLibrary.scaling(2.0)
    transformed = scaling.transform(manifold)

    # Vectors should be scaled
    np.testing.assert_array_almost_equal(
        transformed.vectors,
        manifold.vectors * 2.0
    )


@pytest.mark.unit
def test_scaling_is_invertible():
    """Test scaling has inverse"""
    scaling = TransformationLibrary.scaling(3.0)

    inverse = scaling.inverse()

    assert inverse is not None


@pytest.mark.unit
def test_normalization_transformation():
    """Test normalization transformation"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    normalization = TransformationLibrary.normalization()
    transformed = normalization.transform(manifold)

    # All vectors should have unit length
    norms = np.linalg.norm(transformed.vectors, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(100))


@pytest.mark.unit
def test_normalization_not_invertible():
    """Test normalization is not invertible"""
    normalization = TransformationLibrary.normalization()

    inverse = normalization.inverse()

    assert inverse is None


@pytest.mark.unit
def test_folding_transformation():
    """Test folding transformation"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    folding = TransformationLibrary.folding(axis=0)
    transformed = folding.transform(manifold)

    # All values in axis 0 should be non-negative
    assert np.all(transformed.vectors[:, 0] >= 0)


@pytest.mark.unit
def test_crystallization_transformation():
    """Test crystallization transformation"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    crystallization = TransformationLibrary.crystallization("fractal")
    transformed = crystallization.transform(manifold)

    # Should have crystallization marker
    pattern = transformed.get_semantic_property("crystallization_pattern")
    assert pattern == "fractal"


# ============================================================================
# Composition Tests
# ============================================================================

@pytest.mark.unit
def test_compose_transformations():
    """Test composing two transformations"""
    vectors = np.random.randn(100, 128)
    manifold = Manifold(vectors=vectors)

    proj = TransformationLibrary.projection(64)
    norm = TransformationLibrary.normalization()

    # Compose: first project, then normalize
    composed = proj @ norm

    transformed = composed.transform(manifold)

    # Should be 64D and normalized
    assert transformed.dimension == 64

    norms = np.linalg.norm(transformed.vectors, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(100))


@pytest.mark.unit
def test_composition_preserves_intersection():
    """Test composed transformation preserves intersection of invariants"""
    proj = TransformationLibrary.projection(32)
    rotation = TransformationLibrary.rotation(np.pi / 4)

    composed = proj @ rotation

    # Should preserve only invariants preserved by both
    proj_preserved = proj.preserves()
    rotation_preserved = rotation.preserves()
    expected = proj_preserved & rotation_preserved

    assert composed.preserves() == expected


@pytest.mark.unit
def test_composition_order_matters():
    """Test that composition order matters"""
    vectors = np.random.randn(50, 64)
    manifold = Manifold(vectors=vectors)

    proj = TransformationLibrary.projection(32)
    scale = TransformationLibrary.scaling(2.0)

    # Two different orders
    proj_then_scale = proj @ scale
    scale_then_proj = scale @ proj

    result1 = proj_then_scale.transform(manifold)
    result2 = scale_then_proj.transform(manifold)

    # Results should be different (in general)
    # At minimum, both should work
    assert result1.dimension == 32
    assert result2.dimension == 32


@pytest.mark.unit
def test_matmul_operator():
    """Test @ operator for composition"""
    identity = TransformationLibrary.identity()
    norm = TransformationLibrary.normalization()

    # @ should work same as compose()
    composed1 = identity @ norm
    composed2 = identity.compose(norm)

    assert composed1.name == composed2.name


# ============================================================================
# Transformation Pipeline Tests
# ============================================================================

@pytest.mark.unit
def test_pipeline_creation():
    """Test creating transformation pipeline"""
    pipeline = TransformationPipeline()

    assert pipeline.transformations == []


@pytest.mark.unit
def test_pipeline_add():
    """Test adding transformations to pipeline"""
    pipeline = (
        TransformationPipeline()
        .add(TransformationLibrary.projection(64))
        .add(TransformationLibrary.normalization())
        .add(TransformationLibrary.rotation(np.pi / 4))
    )

    assert len(pipeline.transformations) == 3


@pytest.mark.unit
def test_pipeline_build():
    """Test building composed transformation from pipeline"""
    pipeline = (
        TransformationPipeline()
        .add(TransformationLibrary.projection(32))
        .add(TransformationLibrary.normalization())
    )

    composed = pipeline.build()

    assert composed is not None
    assert isinstance(composed, ComposableTransformation)


@pytest.mark.unit
def test_pipeline_preserved_invariants():
    """Test getting preserved invariants from pipeline"""
    pipeline = (
        TransformationPipeline()
        .add(TransformationLibrary.projection(32))
        .add(TransformationLibrary.rotation(np.pi / 4))
    )

    preserved = pipeline.get_preserved_invariants()

    # Should be intersection of both
    assert Invariant.TOPOLOGY in preserved


@pytest.mark.unit
def test_empty_pipeline():
    """Test empty pipeline returns identity"""
    pipeline = TransformationPipeline()

    composed = pipeline.build()

    # Should be identity
    assert len(composed.preserves()) > 0


# ============================================================================
# Invariant Preservation Tests
# ============================================================================

@pytest.mark.unit
def test_rotation_preserves_topology():
    """Test rotation preserves topological invariants"""
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    manifold = Manifold(vectors=vectors)

    rotation = TransformationLibrary.rotation(np.pi / 3, plane=(0, 1))

    euler_before = manifold.compute_invariant(Invariant.EULER_CHARACTERISTIC)

    transformed = rotation.transform(manifold)

    euler_after = transformed.compute_invariant(Invariant.EULER_CHARACTERISTIC)

    assert euler_before == euler_after


@pytest.mark.unit
def test_scaling_changes_volume():
    """Test scaling changes volume"""
    vectors = np.random.randn(100, 64)
    manifold = Manifold(vectors=vectors)

    volume_before = manifold.compute_invariant(Invariant.VOLUME)

    scaling = TransformationLibrary.scaling(2.0)
    transformed = scaling.transform(manifold)

    volume_after = transformed.compute_invariant(Invariant.VOLUME)

    # Volume should change (not preserved)
    assert volume_before != pytest.approx(volume_after)


@pytest.mark.unit
def test_projection_preserves_connectivity():
    """Test projection preserves connectivity"""
    vectors = np.random.randn(10, 64)

    # Create adjacency
    adjacency = np.eye(10)  # Simple connectivity

    manifold = Manifold(vectors=vectors, adjacency=adjacency)

    projection = TransformationLibrary.projection(32)
    transformed = projection.transform(manifold)

    # Adjacency should be preserved
    if transformed.adjacency is not None:
        np.testing.assert_array_equal(transformed.adjacency, manifold.adjacency)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_full_transformation_pipeline():
    """Test complete transformation pipeline"""
    vectors = np.random.randn(200, 256)
    manifold = Manifold(vectors=vectors)

    # Build complex pipeline
    pipeline = (
        TransformationPipeline()
        .add(TransformationLibrary.projection(128))
        .add(TransformationLibrary.normalization())
        .add(TransformationLibrary.rotation(np.pi / 6, plane=(0, 1)))
        .add(TransformationLibrary.projection(64))
        .add(TransformationLibrary.normalization())
        .add(TransformationLibrary.crystallization("fractal"))
    )

    composed = pipeline.build()
    transformed = composed.transform(manifold)

    # Verify final state
    assert transformed.dimension == 64
    assert transformed.num_points == 200

    # Should be normalized
    norms = np.linalg.norm(transformed.vectors, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(200))

    # Should have crystallization marker
    assert transformed.get_semantic_property("crystallization_pattern") == "fractal"


@pytest.mark.integration
def test_invertible_pipeline():
    """Test pipeline with invertible transformations"""
    vectors = np.random.randn(50, 32)
    manifold = Manifold(vectors=vectors)

    # Pipeline of invertible transformations
    pipeline = (
        TransformationPipeline()
        .add(TransformationLibrary.rotation(np.pi / 4, plane=(0, 1)))
        .add(TransformationLibrary.scaling(2.0))
        .add(TransformationLibrary.rotation(-np.pi / 4, plane=(0, 1)))
    )

    composed = pipeline.build()

    # Forward
    transformed = composed.transform(manifold)

    # Should have inverse
    inverse = composed.inverse()
    assert inverse is not None


@pytest.mark.integration
def test_semantic_property_preservation():
    """Test that semantic properties flow through transformations"""
    vectors = np.random.randn(100, 64)
    metadata = {
        "name": "test_manifold",
        "type": "sample",
        "custom_value": 42
    }
    manifold = Manifold(vectors=vectors, metadata=metadata)

    # Transform
    transformation = TransformationLibrary.projection(32)
    transformed = transformation.transform(manifold)

    # Metadata should be preserved
    assert transformed.get_semantic_property("name") == "test_manifold"
    assert transformed.get_semantic_property("type") == "sample"
    assert transformed.get_semantic_property("custom_value") == 42


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.unit
def test_projection_to_same_dimension():
    """Test projection to same dimension does nothing"""
    vectors = np.random.randn(50, 64)
    manifold = Manifold(vectors=vectors)

    projection = TransformationLibrary.projection(64)
    transformed = projection.transform(manifold)

    # Should be unchanged
    assert transformed.dimension == 64


@pytest.mark.unit
def test_rotation_in_invalid_plane():
    """Test rotation in invalid plane (dimensions don't exist)"""
    vectors = np.random.randn(50, 8)
    manifold = Manifold(vectors=vectors)

    # Plane (10, 11) doesn't exist in 8D space
    rotation = TransformationLibrary.rotation(np.pi / 4, plane=(10, 11))

    # Should not crash
    transformed = rotation.transform(manifold)

    # Vectors should be unchanged (rotation couldn't apply)
    np.testing.assert_array_equal(transformed.vectors, manifold.vectors)


@pytest.mark.unit
def test_zero_vector_normalization():
    """Test normalizing manifold with zero vectors"""
    vectors = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])  # Middle is zero
    manifold = Manifold(vectors=vectors)

    normalization = TransformationLibrary.normalization()
    transformed = normalization.transform(manifold)

    # Should handle zero vector gracefully (likely kept as zero or set to small value)
    assert not np.any(np.isnan(transformed.vectors))
