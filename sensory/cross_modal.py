"""
SYNAPSE-M: Cross-Modal Binding

Advanced cross-modal integration for binding perceptions across modalities.
Implements solutions to the binding problem using temporal synchrony,
spatial coherence, and semantic alignment.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import deque

from .modality_types import (
    ExtendedModality, ModalityCharacteristics, ProcessedModality,
    get_modality_characteristics, ModalityDomain
)


# ============================================================================
# Binding Types
# ============================================================================

class BindingType(Enum):
    """Types of cross-modal binding"""
    TEMPORAL = auto()     # Bound by temporal synchrony
    SPATIAL = auto()      # Bound by spatial co-location
    SEMANTIC = auto()     # Bound by meaning alignment
    CAUSAL = auto()       # Bound by causal relationship
    ATTENTIONAL = auto()  # Bound by shared attention


class BindingStrength(Enum):
    """Strength levels of binding"""
    WEAK = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    FUSED = 1.0


# ============================================================================
# Binding Results
# ============================================================================

@dataclass
class BindingPair:
    """A pair of bound modalities"""
    modality_a: ExtendedModality
    modality_b: ExtendedModality
    binding_type: BindingType
    strength: float
    evidence: Dict[str, float] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.modality_a, self.modality_b))


@dataclass
class BoundPercept:
    """
    A unified percept created from bound modalities.
    """
    id: str = field(default_factory=lambda: str(id(object())))
    modalities: List[ExtendedModality] = field(default_factory=list)
    unified_representation: np.ndarray = field(default_factory=lambda: np.array([]))
    bindings: List[BindingPair] = field(default_factory=list)

    # Coherence metrics
    temporal_coherence: float = 0.0
    spatial_coherence: float = 0.0
    semantic_coherence: float = 0.0
    overall_binding_strength: float = 0.0

    # Source information
    source_perceptions: Dict[ExtendedModality, ProcessedModality] = field(default_factory=dict)
    dominant_modality: Optional[ExtendedModality] = None

    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_num_modalities(self) -> int:
        """Get number of contributing modalities"""
        return len(self.modalities)

    def has_modality(self, modality: ExtendedModality) -> bool:
        """Check if modality contributes to this percept"""
        return modality in self.modalities


# ============================================================================
# Binding Mechanisms
# ============================================================================

class TemporalBindingMechanism:
    """
    Binds percepts based on temporal synchrony.

    Neurons that fire together wire together - percepts occurring
    within a synchrony window are likely from the same source.
    """

    def __init__(self, synchrony_window_ms: float = 100.0):
        self.synchrony_window_ms = synchrony_window_ms
        self.synchrony_window_s = synchrony_window_ms / 1000.0

    def compute_binding(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> Tuple[float, List[BindingPair]]:
        """
        Compute temporal binding between perceptions.

        Returns:
            Tuple of (coherence_score, list of binding pairs)
        """
        if len(perceptions) < 2:
            return 1.0, []

        bindings = []
        timestamps = [p.timestamp for p in perceptions.values()]
        modalities = list(perceptions.keys())

        # Calculate pairwise temporal proximity
        coherence_scores = []
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                m_a, m_b = modalities[i], modalities[j]
                t_a = perceptions[m_a].timestamp
                t_b = perceptions[m_b].timestamp

                # Time difference
                dt = abs(t_a - t_b)

                # Compute coherence (exponential decay)
                coherence = np.exp(-dt / self.synchrony_window_s)
                coherence_scores.append(coherence)

                if coherence > 0.3:  # Threshold for binding
                    bindings.append(BindingPair(
                        modality_a=m_a,
                        modality_b=m_b,
                        binding_type=BindingType.TEMPORAL,
                        strength=coherence,
                        evidence={"time_diff_ms": dt * 1000}
                    ))

        overall_coherence = np.mean(coherence_scores) if coherence_scores else 1.0
        return float(overall_coherence), bindings


class SpatialBindingMechanism:
    """
    Binds percepts based on spatial co-location.

    Perceptions from the same spatial location are likely
    from the same object/event.
    """

    def __init__(self, spatial_threshold: float = 0.5):
        self.spatial_threshold = spatial_threshold

    def compute_binding(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> Tuple[float, List[BindingPair]]:
        """Compute spatial binding between perceptions."""
        # Extract spatial information from perceptions
        spatial_info: Dict[ExtendedModality, np.ndarray] = {}

        for modality, perception in perceptions.items():
            if perception.raw_input.source_location is not None:
                spatial_info[modality] = perception.raw_input.source_location
            else:
                # Infer spatial info from features (simplified)
                # Use first few dimensions as spatial proxy
                spatial_info[modality] = perception.features[:3] if len(perception.features) >= 3 else np.zeros(3)

        if len(spatial_info) < 2:
            return 0.8, []

        bindings = []
        modalities = list(spatial_info.keys())
        coherence_scores = []

        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                m_a, m_b = modalities[i], modalities[j]
                loc_a = spatial_info[m_a]
                loc_b = spatial_info[m_b]

                # Compute spatial distance
                distance = np.linalg.norm(loc_a - loc_b)

                # Convert to coherence (inverse of distance)
                coherence = 1.0 / (1.0 + distance / self.spatial_threshold)
                coherence_scores.append(coherence)

                if coherence > 0.4:
                    bindings.append(BindingPair(
                        modality_a=m_a,
                        modality_b=m_b,
                        binding_type=BindingType.SPATIAL,
                        strength=coherence,
                        evidence={"spatial_distance": float(distance)}
                    ))

        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.8
        return float(overall_coherence), bindings


class SemanticBindingMechanism:
    """
    Binds percepts based on semantic alignment.

    Perceptions with aligned meanings are bound together.
    """

    def __init__(self, semantic_threshold: float = 0.3):
        self.semantic_threshold = semantic_threshold

    def compute_binding(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> Tuple[float, List[BindingPair]]:
        """Compute semantic binding between perceptions."""
        if len(perceptions) < 2:
            return 1.0, []

        bindings = []
        modalities = list(perceptions.keys())
        coherence_scores = []

        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                m_a, m_b = modalities[i], modalities[j]
                emb_a = perceptions[m_a].semantic_embedding
                emb_b = perceptions[m_b].semantic_embedding

                # Align dimensions if needed
                min_dim = min(len(emb_a), len(emb_b))
                emb_a_aligned = emb_a[:min_dim]
                emb_b_aligned = emb_b[:min_dim]

                # Compute cosine similarity
                norm_a = np.linalg.norm(emb_a_aligned) + 1e-8
                norm_b = np.linalg.norm(emb_b_aligned) + 1e-8
                similarity = np.dot(emb_a_aligned, emb_b_aligned) / (norm_a * norm_b)

                # Normalize to [0, 1]
                coherence = (similarity + 1) / 2
                coherence_scores.append(coherence)

                if coherence > self.semantic_threshold:
                    bindings.append(BindingPair(
                        modality_a=m_a,
                        modality_b=m_b,
                        binding_type=BindingType.SEMANTIC,
                        strength=coherence,
                        evidence={"cosine_similarity": float(similarity)}
                    ))

        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        return float(overall_coherence), bindings


# ============================================================================
# Cross-Modal Integrator
# ============================================================================

class CrossModalIntegrator:
    """
    Integrates information across sensory modalities.

    Combines temporal, spatial, and semantic binding mechanisms
    to create unified percepts from multi-modal input.
    """

    def __init__(
        self,
        integration_dim: int = 256,
        temporal_weight: float = 0.3,
        spatial_weight: float = 0.3,
        semantic_weight: float = 0.4,
        binding_threshold: float = 0.4
    ):
        self.integration_dim = integration_dim
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.semantic_weight = semantic_weight
        self.binding_threshold = binding_threshold

        # Binding mechanisms
        self.temporal_binder = TemporalBindingMechanism()
        self.spatial_binder = SpatialBindingMechanism()
        self.semantic_binder = SemanticBindingMechanism()

        # Integration weights (learnable projection)
        np.random.seed(47)
        self._init_projection_weights()

        # History for tracking binding patterns
        self.binding_history: deque = deque(maxlen=100)

    def _init_projection_weights(self):
        """Initialize projection weights for fusion"""
        # Maximum input dimension (sum of all modality features)
        self.max_input_dim = 2048
        self.projection = np.random.randn(self.max_input_dim, self.integration_dim) * 0.1

    def integrate(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> BoundPercept:
        """
        Integrate perceptions across modalities.

        Args:
            perceptions: Dict mapping modalities to processed perceptions

        Returns:
            BoundPercept with unified representation
        """
        if not perceptions:
            return BoundPercept(
                unified_representation=np.zeros(self.integration_dim),
                overall_binding_strength=0.0
            )

        # Compute binding from each mechanism
        temporal_coherence, temporal_bindings = self.temporal_binder.compute_binding(perceptions)
        spatial_coherence, spatial_bindings = self.spatial_binder.compute_binding(perceptions)
        semantic_coherence, semantic_bindings = self.semantic_binder.compute_binding(perceptions)

        # Combine all bindings
        all_bindings = temporal_bindings + spatial_bindings + semantic_bindings

        # Compute weighted coherence
        overall_coherence = (
            self.temporal_weight * temporal_coherence +
            self.spatial_weight * spatial_coherence +
            self.semantic_weight * semantic_coherence
        )

        # Fuse representations
        unified_rep = self._fuse_representations(perceptions, all_bindings)

        # Determine dominant modality
        dominant = self._find_dominant_modality(perceptions)

        # Create bound percept
        percept = BoundPercept(
            modalities=list(perceptions.keys()),
            unified_representation=unified_rep,
            bindings=all_bindings,
            temporal_coherence=temporal_coherence,
            spatial_coherence=spatial_coherence,
            semantic_coherence=semantic_coherence,
            overall_binding_strength=overall_coherence,
            source_perceptions=perceptions,
            dominant_modality=dominant
        )

        # Store in history
        self.binding_history.append({
            "modalities": [m.value for m in perceptions.keys()],
            "coherence": overall_coherence,
            "timestamp": time.time()
        })

        return percept

    def _fuse_representations(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality],
        bindings: List[BindingPair]
    ) -> np.ndarray:
        """Fuse modality representations into unified vector."""
        # Collect weighted features
        weighted_features = []
        total_weight = 0.0

        for modality, perception in perceptions.items():
            # Get modality characteristics for weighting
            chars = get_modality_characteristics(modality)
            weight = chars.binding_weight * perception.confidence

            # Weight semantic embeddings more heavily for integration
            features = perception.semantic_embedding
            weighted_features.append(features * weight)
            total_weight += weight

        if not weighted_features or total_weight == 0:
            return np.zeros(self.integration_dim)

        # Concatenate and average
        combined = np.concatenate(weighted_features) / total_weight

        # Project to integration space
        if len(combined) > self.max_input_dim:
            combined = combined[:self.max_input_dim]
        else:
            padded = np.zeros(self.max_input_dim)
            padded[:len(combined)] = combined
            combined = padded

        unified = np.tanh(combined @ self.projection)
        return unified

    def _find_dominant_modality(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> Optional[ExtendedModality]:
        """Find the dominant modality in a set of perceptions."""
        if not perceptions:
            return None

        # Score each modality by salience * confidence * binding_weight
        scores = {}
        for modality, perception in perceptions.items():
            chars = get_modality_characteristics(modality)
            score = perception.salience * perception.confidence * chars.binding_weight
            scores[modality] = score

        return max(scores, key=scores.get)

    def get_binding_statistics(self) -> Dict[str, Any]:
        """Get statistics about binding patterns."""
        if not self.binding_history:
            return {"status": "No binding history"}

        coherences = [h["coherence"] for h in self.binding_history]
        modality_counts: Dict[str, int] = {}

        for h in self.binding_history:
            for m in h["modalities"]:
                modality_counts[m] = modality_counts.get(m, 0) + 1

        return {
            "total_bindings": len(self.binding_history),
            "avg_coherence": float(np.mean(coherences)),
            "min_coherence": float(min(coherences)),
            "max_coherence": float(max(coherences)),
            "modality_frequency": modality_counts
        }


# ============================================================================
# Sensory Fusion Pipeline
# ============================================================================

class SensoryFusion:
    """
    High-level API for sensory fusion across modalities.

    Provides a clean interface for binding multi-modal inputs
    into unified percepts.
    """

    def __init__(self, integration_dim: int = 256):
        self.integrator = CrossModalIntegrator(integration_dim=integration_dim)
        self._pending_inputs: Dict[ExtendedModality, ProcessedModality] = {}
        self._fusion_window_ms: float = 100.0

    def add_perception(self, perception: ProcessedModality):
        """Add a perception to the pending fusion window."""
        self._pending_inputs[perception.modality] = perception

        # Clean old perceptions
        current_time = time.time()
        window_s = self._fusion_window_ms / 1000.0

        self._pending_inputs = {
            m: p for m, p in self._pending_inputs.items()
            if current_time - p.timestamp < window_s
        }

    def fuse(self) -> Optional[BoundPercept]:
        """Fuse all pending perceptions."""
        if not self._pending_inputs:
            return None

        percept = self.integrator.integrate(self._pending_inputs)
        self._pending_inputs.clear()
        return percept

    def fuse_immediate(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> BoundPercept:
        """Immediately fuse provided perceptions."""
        return self.integrator.integrate(perceptions)

    def get_pending_modalities(self) -> List[ExtendedModality]:
        """Get list of modalities with pending input."""
        return list(self._pending_inputs.keys())


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Cross-Modal Binding Demo")
    print("=" * 50)

    from .modality_types import ModalityInput

    # Create test perceptions
    np.random.seed(42)
    current_time = time.time()

    vision_input = ModalityInput(
        modality=ExtendedModality.VISION,
        raw_data=np.random.randn(256),
        intensity=0.8
    )

    audio_input = ModalityInput(
        modality=ExtendedModality.AUDITION,
        raw_data=np.random.randn(128),
        intensity=0.7
    )

    vision_processed = ProcessedModality(
        modality=ExtendedModality.VISION,
        raw_input=vision_input,
        features=np.random.randn(256),
        semantic_embedding=np.random.randn(128),
        salience=0.8,
        confidence=0.9,
        timestamp=current_time
    )

    audio_processed = ProcessedModality(
        modality=ExtendedModality.AUDITION,
        raw_input=audio_input,
        features=np.random.randn(128),
        semantic_embedding=np.random.randn(64),
        salience=0.7,
        confidence=0.85,
        timestamp=current_time + 0.05  # 50ms later
    )

    # Create integrator and bind
    integrator = CrossModalIntegrator()

    perceptions = {
        ExtendedModality.VISION: vision_processed,
        ExtendedModality.AUDITION: audio_processed
    }

    print("\n1. Integrating vision + audition...")
    bound = integrator.integrate(perceptions)

    print(f"   Modalities: {[m.value for m in bound.modalities]}")
    print(f"   Dominant: {bound.dominant_modality.value if bound.dominant_modality else 'None'}")
    print(f"   Temporal coherence: {bound.temporal_coherence:.3f}")
    print(f"   Spatial coherence: {bound.spatial_coherence:.3f}")
    print(f"   Semantic coherence: {bound.semantic_coherence:.3f}")
    print(f"   Overall binding: {bound.overall_binding_strength:.3f}")
    print(f"   Unified representation shape: {bound.unified_representation.shape}")
    print(f"   Number of bindings: {len(bound.bindings)}")

    print("\n2. Binding details:")
    for binding in bound.bindings:
        print(f"   {binding.modality_a.value} <-> {binding.modality_b.value}")
        print(f"      Type: {binding.binding_type.name}")
        print(f"      Strength: {binding.strength:.3f}")

    print("\n3. Using SensoryFusion API:")
    fusion = SensoryFusion()
    fusion.add_perception(vision_processed)
    fusion.add_perception(audio_processed)

    result = fusion.fuse()
    if result:
        print(f"   Fused {result.get_num_modalities()} modalities")
        print(f"   Binding strength: {result.overall_binding_strength:.3f}")

    print("\n4. Binding statistics:")
    stats = integrator.get_binding_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
