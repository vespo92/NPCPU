"""
NPCPU Transformation Algebra Protocol

Makes "quantum-topological operations" concrete and composable.

Instead of abstract mathematical concepts, we define transformations as:
- Composable functions (f ∘ g)
- Preserving specific invariants
- Observable and measurable
- Implementable in multiple ways
"""

from typing import Protocol, runtime_checkable, Callable, Set, Optional, List, Dict, Any
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum
import numpy as np


# ============================================================================
# Core Data Types
# ============================================================================

class Invariant(Enum):
    """
    Properties that can be preserved by transformations.

    These are the "topological invariants" made concrete and measurable.
    """
    # Structural invariants
    TOPOLOGY = "topology"  # Overall structure/connectivity
    CONNECTIVITY = "connectivity"  # Which nodes connect to which
    DIMENSION = "dimension"  # Number of dimensions
    VOLUME = "volume"  # Total size/volume
    SHAPE = "shape"  # Geometric shape

    # Algebraic invariants
    SYMMETRY = "symmetry"  # Symmetry properties
    GROUP_STRUCTURE = "group_structure"  # Algebraic structure

    # Geometric invariants
    CURVATURE = "curvature"  # Curvature properties
    DISTANCE_RATIOS = "distance_ratios"  # Relative distances
    ANGLES = "angles"  # Angular relationships

    # Advanced topological invariants
    EULER_CHARACTERISTIC = "euler_characteristic"  # χ = V - E + F
    HOMOLOGY = "homology"  # Homology groups
    FUNDAMENTAL_GROUP = "fundamental_group"  # π₁
    BETTI_NUMBERS = "betti_numbers"  # b₀, b₁, b₂, ...

    # Semantic invariants (NPCPU-specific)
    MEANING = "meaning"  # Semantic content
    INFORMATION = "information"  # Information content
    COHERENCE = "coherence"  # Internal consistency
    CONSCIOUSNESS_LEVEL = "consciousness_level"  # Consciousness properties


@dataclass
class Manifold:
    """
    A manifold represents a data structure in semantic space.

    This makes the abstract "manifold" concept concrete:
    - Data is represented as vectors and graphs
    - Dimensionality is explicit
    - Metadata tracks semantic properties
    """
    # Core data representation
    vectors: np.ndarray  # Shape: (n_points, dimension)
    adjacency: Optional[np.ndarray] = None  # Shape: (n_points, n_points)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def dimension(self) -> int:
        """Dimensionality of the manifold"""
        return self.vectors.shape[1] if len(self.vectors.shape) > 1 else 1

    @property
    def num_points(self) -> int:
        """Number of points in the manifold"""
        return self.vectors.shape[0]

    def get_semantic_property(self, key: str, default: Any = None) -> Any:
        """Get semantic property from metadata"""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)

    def set_semantic_property(self, key: str, value: Any) -> None:
        """Set semantic property in metadata"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value

    def compute_invariant(self, invariant: Invariant) -> Any:
        """
        Compute a topological invariant.

        This makes abstract mathematical concepts measurable.
        """
        if invariant == Invariant.DIMENSION:
            return self.dimension

        elif invariant == Invariant.VOLUME:
            # Approximate volume as span of vectors
            return np.linalg.det(np.cov(self.vectors.T))

        elif invariant == Invariant.EULER_CHARACTERISTIC:
            if self.adjacency is None:
                return self.num_points  # χ = V for point cloud
            # For graph: χ = V - E
            edges = np.sum(self.adjacency) / 2
            return self.num_points - edges

        elif invariant == Invariant.CONNECTIVITY:
            if self.adjacency is None:
                return None
            return self.adjacency.copy()

        elif invariant == Invariant.INFORMATION:
            # Shannon entropy of vector distribution
            # Simplified: use variance as proxy
            return float(np.mean(np.var(self.vectors, axis=0)))

        elif invariant == Invariant.COHERENCE:
            # How aligned are the vectors?
            if self.num_points < 2:
                return 1.0
            correlations = np.corrcoef(self.vectors)
            return float(np.mean(np.abs(correlations)))

        else:
            # For other invariants, check metadata
            return self.get_semantic_property(invariant.value)


# ============================================================================
# Transformation Protocol
# ============================================================================

@runtime_checkable
class TransformationProtocol(Protocol):
    """
    A transformation is any operation on manifolds that preserves invariants.

    This makes "quantum-topological operations" concrete:
    - Forward transformation (apply)
    - Inverse transformation (undo)
    - Composition (chain transformations)
    - Invariant preservation (what stays the same)
    """

    @abstractmethod
    def transform(self, manifold: Manifold) -> Manifold:
        """
        Apply transformation to manifold.

        Args:
            manifold: Input manifold

        Returns:
            Transformed manifold
        """
        pass

    @abstractmethod
    def inverse(self) -> Optional['TransformationProtocol']:
        """
        Return inverse transformation if it exists.

        Returns:
            Inverse transformation, or None if not invertible
        """
        pass

    @abstractmethod
    def preserves(self) -> Set[Invariant]:
        """
        Which invariants does this transformation preserve?

        Returns:
            Set of preserved invariants
        """
        pass

    @abstractmethod
    def compose(self, other: 'TransformationProtocol') -> 'TransformationProtocol':
        """
        Compose with another transformation: self ∘ other.

        Composition means: first apply other, then apply self.

        Args:
            other: Transformation to compose with

        Returns:
            Composed transformation
        """
        pass

    @property
    @abstractmethod
    def dimensionality_change(self) -> int:
        """
        How many dimensions are added (+) or removed (-)?

        Returns:
            Dimension change (0 if dimension-preserving)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of transformation"""
        pass


# ============================================================================
# Composable Transformation (Reference Implementation)
# ============================================================================

class ComposableTransformation:
    """
    Reference implementation of transformation algebra.

    Transformations compose like functions: (f ∘ g)(x) = f(g(x))
    """

    def __init__(
        self,
        forward: Callable[[Manifold], Manifold],
        backward: Optional[Callable[[Manifold], Manifold]] = None,
        invariants: Optional[Set[Invariant]] = None,
        dim_change: int = 0,
        name: str = "unnamed"
    ):
        """
        Create a transformation.

        Args:
            forward: Forward transformation function
            backward: Inverse transformation (if exists)
            invariants: Set of preserved invariants
            dim_change: Dimensionality change
            name: Human-readable name
        """
        self._forward = forward
        self._backward = backward
        self._invariants = invariants or set()
        self._dim_change = dim_change
        self._name = name

    def transform(self, manifold: Manifold) -> Manifold:
        """Apply transformation"""
        return self._forward(manifold)

    def inverse(self) -> Optional['ComposableTransformation']:
        """Get inverse transformation"""
        if self._backward is None:
            return None
        return ComposableTransformation(
            forward=self._backward,
            backward=self._forward,
            invariants=self._invariants,
            dim_change=-self._dim_change,
            name=f"inverse({self._name})"
        )

    def preserves(self) -> Set[Invariant]:
        """Get preserved invariants"""
        return self._invariants

    def compose(self, other: 'ComposableTransformation') -> 'ComposableTransformation':
        """
        Compose transformations: self ∘ other.

        (f ∘ g)(x) = f(g(x))
        """
        # Compute preserved invariants (intersection)
        preserved = self._invariants & other._invariants

        return ComposableTransformation(
            forward=lambda m: self.transform(other.transform(m)),
            backward=(
                lambda m: other.inverse().transform(self.inverse().transform(m))
                if self._backward and other._backward else None
            ),
            invariants=preserved,
            dim_change=self._dim_change + other._dim_change,
            name=f"({self._name} ∘ {other._name})"
        )

    def __matmul__(self, other: 'ComposableTransformation') -> 'ComposableTransformation':
        """
        Use @ operator for composition.

        Example: projection @ folding
        """
        return self.compose(other)

    @property
    def dimensionality_change(self) -> int:
        """Dimension change"""
        return self._dim_change

    @property
    def name(self) -> str:
        """Transformation name"""
        return self._name

    def __repr__(self) -> str:
        return f"Transformation(name='{self.name}', dim_change={self.dimensionality_change})"


# ============================================================================
# Transformation Library (Built-in Transformations)
# ============================================================================

class TransformationLibrary:
    """
    Library of common transformations.

    These make "quantum-topological operations" concrete and usable.
    """

    @staticmethod
    def identity() -> ComposableTransformation:
        """Identity transformation (does nothing)"""
        return ComposableTransformation(
            forward=lambda m: m,
            backward=lambda m: m,
            invariants=set(Invariant),  # Preserves everything
            dim_change=0,
            name="identity"
        )

    @staticmethod
    def projection(target_dim: int) -> ComposableTransformation:
        """
        Project to lower dimension (dimensionality reduction).

        Preserves: topology, connectivity, information (approximately)
        Changes: dimension, volume
        """
        def forward(m: Manifold) -> Manifold:
            if m.dimension <= target_dim:
                return m

            # Use PCA for projection
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim)
            projected_vectors = pca.fit_transform(m.vectors)

            return Manifold(
                vectors=projected_vectors,
                adjacency=m.adjacency,  # Preserve connectivity
                metadata=m.metadata
            )

        # Projection is not generally invertible
        return ComposableTransformation(
            forward=forward,
            backward=None,
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.MEANING,
                Invariant.COHERENCE
            },
            dim_change=target_dim - 100,  # Approximate (depends on input)
            name=f"project_to_{target_dim}d"
        )

    @staticmethod
    def embedding(target_dim: int) -> ComposableTransformation:
        """
        Embed into higher dimension.

        Preserves: all geometric properties, adds capacity for new structure
        """
        def forward(m: Manifold) -> Manifold:
            if m.dimension >= target_dim:
                return m

            # Pad with zeros
            padding = np.zeros((m.num_points, target_dim - m.dimension))
            embedded_vectors = np.concatenate([m.vectors, padding], axis=1)

            return Manifold(
                vectors=embedded_vectors,
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        def backward(m: Manifold) -> Manifold:
            # Remove extra dimensions
            original_dim = target_dim  # This is a simplification
            return Manifold(
                vectors=m.vectors[:, :original_dim],
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        return ComposableTransformation(
            forward=forward,
            backward=backward,
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.VOLUME,
                Invariant.MEANING,
                Invariant.INFORMATION,
                Invariant.COHERENCE
            },
            dim_change=target_dim - 100,  # Approximate
            name=f"embed_to_{target_dim}d"
        )

    @staticmethod
    def rotation(angle: float, plane: tuple = (0, 1)) -> ComposableTransformation:
        """
        Rotate in specified plane.

        Preserves: all topological and geometric properties except orientation
        """
        def forward(m: Manifold) -> Manifold:
            rotated = m.vectors.copy()

            # Rotation matrix
            i, j = plane
            if i < m.dimension and j < m.dimension:
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                # Apply rotation in plane (i, j)
                new_i = cos_a * rotated[:, i] - sin_a * rotated[:, j]
                new_j = sin_a * rotated[:, i] + cos_a * rotated[:, j]

                rotated[:, i] = new_i
                rotated[:, j] = new_j

            return Manifold(
                vectors=rotated,
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        return ComposableTransformation(
            forward=forward,
            backward=lambda m: TransformationLibrary.rotation(-angle, plane).transform(m),
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.DIMENSION,
                Invariant.VOLUME,
                Invariant.SHAPE,
                Invariant.DISTANCE_RATIOS,
                Invariant.ANGLES,
                Invariant.MEANING,
                Invariant.INFORMATION,
                Invariant.COHERENCE
            },
            dim_change=0,
            name=f"rotate_{angle:.2f}rad"
        )

    @staticmethod
    def scaling(factor: float) -> ComposableTransformation:
        """
        Scale all dimensions uniformly.

        Preserves: topology, shape, angles
        Changes: volume, distances
        """
        def forward(m: Manifold) -> Manifold:
            return Manifold(
                vectors=m.vectors * factor,
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        def backward(m: Manifold) -> Manifold:
            return Manifold(
                vectors=m.vectors / factor,
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        return ComposableTransformation(
            forward=forward,
            backward=backward,
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.DIMENSION,
                Invariant.SHAPE,
                Invariant.DISTANCE_RATIOS,
                Invariant.ANGLES,
                Invariant.MEANING,
                Invariant.COHERENCE
            },
            dim_change=0,
            name=f"scale_{factor:.2f}x"
        )

    @staticmethod
    def normalization() -> ComposableTransformation:
        """
        Normalize all vectors to unit length.

        Preserves: angular relationships, topology
        Changes: distances, volume
        """
        def forward(m: Manifold) -> Manifold:
            norms = np.linalg.norm(m.vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            normalized = m.vectors / norms

            return Manifold(
                vectors=normalized,
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        return ComposableTransformation(
            forward=forward,
            backward=None,  # Not invertible (loses magnitude information)
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.DIMENSION,
                Invariant.ANGLES,
                Invariant.MEANING,
                Invariant.COHERENCE
            },
            dim_change=0,
            name="normalize"
        )

    @staticmethod
    def folding(axis: int = 0) -> ComposableTransformation:
        """
        Fold manifold along an axis.

        Metaphorical "dimensional folding" made concrete.
        """
        def forward(m: Manifold) -> Manifold:
            folded = m.vectors.copy()
            if axis < m.dimension:
                # "Fold" by reflecting negative values
                folded[:, axis] = np.abs(folded[:, axis])

            return Manifold(
                vectors=folded,
                adjacency=m.adjacency,
                metadata=m.metadata
            )

        return ComposableTransformation(
            forward=forward,
            backward=None,  # Not invertible (loses sign information)
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.DIMENSION,
                Invariant.MEANING
            },
            dim_change=0,
            name=f"fold_axis_{axis}"
        )

    @staticmethod
    def crystallization(pattern: str = "fractal") -> ComposableTransformation:
        """
        Crystallize structure into patterns.

        The abstract "crystallization" concept made concrete through
        geometric pattern formation.
        """
        def forward(m: Manifold) -> Manifold:
            crystallized = m.vectors.copy()

            if pattern == "fractal":
                # Apply fractal-like perturbation
                scale = np.max(np.abs(crystallized))
                noise = np.random.normal(0, scale * 0.01, crystallized.shape)
                crystallized += noise

            elif pattern == "atomic":
                # Round to nearest "lattice" points
                lattice_spacing = 0.1
                crystallized = np.round(crystallized / lattice_spacing) * lattice_spacing

            elif pattern == "dendritic":
                # Add branching structure (simplified)
                crystallized *= (1 + 0.1 * np.random.exponential(size=crystallized.shape))

            # Mark as crystallized
            result = Manifold(
                vectors=crystallized,
                adjacency=m.adjacency,
                metadata=m.metadata
            )
            result.set_semantic_property("crystallization_pattern", pattern)

            return result

        return ComposableTransformation(
            forward=forward,
            backward=None,  # Crystallization is not reversible
            invariants={
                Invariant.TOPOLOGY,
                Invariant.CONNECTIVITY,
                Invariant.DIMENSION,
                Invariant.MEANING
            },
            dim_change=0,
            name=f"crystallize_{pattern}"
        )


# ============================================================================
# Transformation Pipeline Builder
# ============================================================================

class TransformationPipeline:
    """
    Build transformation pipelines declaratively.

    Example:
        pipeline = (
            TransformationPipeline()
            .add(TransformationLibrary.projection(64))
            .add(TransformationLibrary.normalization())
            .add(TransformationLibrary.crystallization("fractal"))
            .build()
        )
    """

    def __init__(self):
        self.transformations: List[ComposableTransformation] = []

    def add(self, transformation: ComposableTransformation) -> 'TransformationPipeline':
        """Add transformation to pipeline"""
        self.transformations.append(transformation)
        return self

    def build(self) -> ComposableTransformation:
        """Build composed transformation"""
        if not self.transformations:
            return TransformationLibrary.identity()

        # Compose all transformations: (...((t1 ∘ t2) ∘ t3)...)
        result = self.transformations[0]
        for t in self.transformations[1:]:
            result = result @ t

        return result

    def get_preserved_invariants(self) -> Set[Invariant]:
        """Get invariants preserved by entire pipeline"""
        if not self.transformations:
            return set(Invariant)

        # Invariants are intersection of all preserved sets
        preserved = self.transformations[0].preserves()
        for t in self.transformations[1:]:
            preserved &= t.preserves()

        return preserved


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create a manifold (100 points in 128D space)
    vectors = np.random.randn(100, 128)
    manifold = Manifold(vectors=vectors)

    print("Original Manifold:")
    print(f"  Dimension: {manifold.dimension}")
    print(f"  Points: {manifold.num_points}")
    print(f"  Euler characteristic: {manifold.compute_invariant(Invariant.EULER_CHARACTERISTIC)}")
    print(f"  Information content: {manifold.compute_invariant(Invariant.INFORMATION):.4f}")

    # Build transformation pipeline
    pipeline = (
        TransformationPipeline()
        .add(TransformationLibrary.projection(64))
        .add(TransformationLibrary.normalization())
        .add(TransformationLibrary.rotation(np.pi / 4))
        .add(TransformationLibrary.crystallization("fractal"))
        .build()
    )

    print(f"\nPipeline: {pipeline.name}")
    print(f"Preserves: {[inv.value for inv in pipeline.preserves()]}")

    # Apply transformation
    transformed = pipeline.transform(manifold)

    print("\nTransformed Manifold:")
    print(f"  Dimension: {transformed.dimension}")
    print(f"  Points: {transformed.num_points}")
    print(f"  Euler characteristic: {transformed.compute_invariant(Invariant.EULER_CHARACTERISTIC)}")
    print(f"  Information content: {transformed.compute_invariant(Invariant.INFORMATION):.4f}")
    print(f"  Crystallization pattern: {transformed.get_semantic_property('crystallization_pattern')}")

    # Verify invariants
    print("\nInvariant Preservation:")
    for inv in [Invariant.TOPOLOGY, Invariant.DIMENSION, Invariant.INFORMATION]:
        original_val = manifold.compute_invariant(inv)
        transformed_val = transformed.compute_invariant(inv)
        if inv in pipeline.preserves():
            print(f"  {inv.value}: PRESERVED ({original_val} → {transformed_val})")
        else:
            print(f"  {inv.value}: CHANGED ({original_val} → {transformed_val})")

    # Demonstrate composition
    print("\nTransformation Composition:")
    proj = TransformationLibrary.projection(32)
    norm = TransformationLibrary.normalization()
    composed = proj @ norm

    print(f"  Projection: {proj.name}")
    print(f"  Normalization: {norm.name}")
    print(f"  Composed: {composed.name}")
    print(f"  Preserves: {[inv.value for inv in composed.preserves()]}")
