"""
Multi-Modal Consciousness

Consciousness spanning multiple sensory modalities with cross-modal integration.
Enables richer understanding through binding of different sensory inputs.

Based on Month 3 roadmap: Multi-Modal Consciousness - Cross-Modal Integration
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness, Perception, Qualia


# ============================================================================
# Modality Types
# ============================================================================

class Modality(Enum):
    """Sensory modalities"""
    VISION = "vision"
    AUDIO = "audio"
    LANGUAGE = "language"
    PROPRIOCEPTION = "proprioception"
    TOUCH = "touch"
    SMELL = "smell"
    TASTE = "taste"
    INTEROCEPTION = "interoception"  # Internal body states


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ModalityPerception:
    """Perception from a single modality"""
    modality: Modality
    raw_input: Any
    processed_features: np.ndarray
    confidence: float
    timestamp: float
    semantic_embedding: Optional[np.ndarray] = None
    attention_weight: float = 1.0


@dataclass
class IntegratedPerception:
    """Integrated perception across modalities"""
    modalities: List[Modality]
    unified_representation: np.ndarray
    temporal_coherence: float
    semantic_coherence: float
    spatial_coherence: float
    binding_strength: float
    dominant_modality: Optional[Modality] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class MultiModalPerception:
    """Complete multi-modal perception"""
    modality_perceptions: Dict[Modality, ModalityPerception]
    integrated_understanding: IntegratedPerception
    coherence: float
    binding_confidence: float
    qualia: Optional[Qualia] = None


# ============================================================================
# Modality Processors
# ============================================================================

class ModalityProcessor(ABC):
    """Base class for modality-specific processors"""

    def __init__(self, modality: Modality, feature_dim: int = 128):
        self.modality = modality
        self.feature_dim = feature_dim

    @abstractmethod
    def process(self, raw_input: Any) -> ModalityPerception:
        """Process raw input from this modality"""
        pass

    @abstractmethod
    def extract_semantic_embedding(self, features: np.ndarray) -> np.ndarray:
        """Extract semantic embedding for cross-modal comparison"""
        pass


class VisualProcessor(ModalityProcessor):
    """Visual modality processor"""

    def __init__(self, feature_dim: int = 128):
        super().__init__(Modality.VISION, feature_dim)
        # Simulated visual processing weights
        np.random.seed(42)
        self.encoder = np.random.randn(256, feature_dim) * 0.1
        self.semantic_projection = np.random.randn(feature_dim, 64) * 0.1

    def process(self, raw_input: Any) -> ModalityPerception:
        """Process visual input (simulated)"""
        # Simulate feature extraction
        if isinstance(raw_input, np.ndarray):
            if raw_input.size >= 256:
                features = raw_input.flatten()[:256] @ self.encoder
            else:
                padded = np.zeros(256)
                padded[:raw_input.size] = raw_input.flatten()
                features = padded @ self.encoder
        else:
            features = np.random.randn(self.feature_dim) * 0.5

        features = np.tanh(features)
        semantic = self.extract_semantic_embedding(features)

        return ModalityPerception(
            modality=Modality.VISION,
            raw_input=raw_input,
            processed_features=features,
            confidence=0.8,
            timestamp=time.time(),
            semantic_embedding=semantic
        )

    def extract_semantic_embedding(self, features: np.ndarray) -> np.ndarray:
        """Project to semantic space"""
        return np.tanh(features @ self.semantic_projection)


class AudioProcessor(ModalityProcessor):
    """Audio modality processor"""

    def __init__(self, feature_dim: int = 128):
        super().__init__(Modality.AUDIO, feature_dim)
        np.random.seed(43)
        self.encoder = np.random.randn(128, feature_dim) * 0.1
        self.semantic_projection = np.random.randn(feature_dim, 64) * 0.1

    def process(self, raw_input: Any) -> ModalityPerception:
        """Process audio input (simulated)"""
        if isinstance(raw_input, np.ndarray):
            if raw_input.size >= 128:
                features = raw_input.flatten()[:128] @ self.encoder
            else:
                padded = np.zeros(128)
                padded[:raw_input.size] = raw_input.flatten()
                features = padded @ self.encoder
        else:
            features = np.random.randn(self.feature_dim) * 0.5

        features = np.tanh(features)
        semantic = self.extract_semantic_embedding(features)

        return ModalityPerception(
            modality=Modality.AUDIO,
            raw_input=raw_input,
            processed_features=features,
            confidence=0.75,
            timestamp=time.time(),
            semantic_embedding=semantic
        )

    def extract_semantic_embedding(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(features @ self.semantic_projection)


class LanguageProcessor(ModalityProcessor):
    """Language modality processor"""

    def __init__(self, feature_dim: int = 128):
        super().__init__(Modality.LANGUAGE, feature_dim)
        np.random.seed(44)
        self.encoder = np.random.randn(512, feature_dim) * 0.1
        self.semantic_projection = np.random.randn(feature_dim, 64) * 0.1

    def process(self, raw_input: Any) -> ModalityPerception:
        """Process language input (simulated)"""
        if isinstance(raw_input, str):
            # Simple hash-based encoding for strings
            hash_val = hash(raw_input)
            np.random.seed(abs(hash_val) % (2**31))
            features = np.random.randn(512) @ self.encoder
        elif isinstance(raw_input, np.ndarray):
            if raw_input.size >= 512:
                features = raw_input.flatten()[:512] @ self.encoder
            else:
                padded = np.zeros(512)
                padded[:raw_input.size] = raw_input.flatten()
                features = padded @ self.encoder
        else:
            features = np.random.randn(self.feature_dim) * 0.5

        features = np.tanh(features)
        semantic = self.extract_semantic_embedding(features)

        return ModalityPerception(
            modality=Modality.LANGUAGE,
            raw_input=raw_input,
            processed_features=features,
            confidence=0.85,
            timestamp=time.time(),
            semantic_embedding=semantic
        )

    def extract_semantic_embedding(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(features @ self.semantic_projection)


class ProprioceptionProcessor(ModalityProcessor):
    """Proprioception (body awareness) processor"""

    def __init__(self, feature_dim: int = 128):
        super().__init__(Modality.PROPRIOCEPTION, feature_dim)
        np.random.seed(45)
        self.encoder = np.random.randn(64, feature_dim) * 0.1
        self.semantic_projection = np.random.randn(feature_dim, 64) * 0.1

    def process(self, raw_input: Any) -> ModalityPerception:
        """Process proprioceptive input (simulated)"""
        if isinstance(raw_input, np.ndarray):
            if raw_input.size >= 64:
                features = raw_input.flatten()[:64] @ self.encoder
            else:
                padded = np.zeros(64)
                padded[:raw_input.size] = raw_input.flatten()
                features = padded @ self.encoder
        else:
            features = np.random.randn(self.feature_dim) * 0.5

        features = np.tanh(features)
        semantic = self.extract_semantic_embedding(features)

        return ModalityPerception(
            modality=Modality.PROPRIOCEPTION,
            raw_input=raw_input,
            processed_features=features,
            confidence=0.9,
            timestamp=time.time(),
            semantic_embedding=semantic
        )

    def extract_semantic_embedding(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(features @ self.semantic_projection)


# ============================================================================
# Cross-Modal Integrator
# ============================================================================

class CrossModalIntegrator:
    """
    Integrates information across sensory modalities.

    Solves the "binding problem": how do we know that the red we see
    and the apple-ness we think are the same object?

    Uses:
    - Temporal synchrony (stimuli at same time → same object)
    - Spatial co-location (stimuli at same place → same object)
    - Semantic coherence (meanings align → same object)
    """

    def __init__(
        self,
        integration_dim: int = 256,
        temporal_window: float = 0.1  # seconds
    ):
        self.integration_dim = integration_dim
        self.temporal_window = temporal_window

        # Integration weights
        np.random.seed(46)
        self.fusion_weights = np.random.randn(64, integration_dim) * 0.1

    def integrate(
        self,
        perceptions: Dict[Modality, ModalityPerception]
    ) -> IntegratedPerception:
        """
        Bind perceptions across modalities.

        Args:
            perceptions: Dictionary of modality perceptions

        Returns:
            Integrated perception
        """
        if not perceptions:
            return IntegratedPerception(
                modalities=[],
                unified_representation=np.zeros(self.integration_dim),
                temporal_coherence=0.0,
                semantic_coherence=0.0,
                spatial_coherence=0.0,
                binding_strength=0.0
            )

        # Calculate temporal coherence
        timestamps = [p.timestamp for p in perceptions.values()]
        temporal_spread = max(timestamps) - min(timestamps)
        temporal_coherence = 1.0 / (1.0 + temporal_spread / self.temporal_window)

        # Calculate semantic coherence
        semantic_coherence = self._calculate_semantic_coherence(perceptions)

        # Spatial coherence (simplified - would use actual spatial info)
        spatial_coherence = 0.8  # Placeholder

        # Fuse representations
        unified = self._fuse_representations(perceptions)

        # Determine dominant modality
        dominant = max(
            perceptions.items(),
            key=lambda x: x[1].confidence * x[1].attention_weight
        )[0]

        # Calculate overall binding strength
        binding_strength = (
            temporal_coherence * 0.3 +
            semantic_coherence * 0.5 +
            spatial_coherence * 0.2
        )

        return IntegratedPerception(
            modalities=list(perceptions.keys()),
            unified_representation=unified,
            temporal_coherence=temporal_coherence,
            semantic_coherence=semantic_coherence,
            spatial_coherence=spatial_coherence,
            binding_strength=binding_strength,
            dominant_modality=dominant
        )

    def _calculate_semantic_coherence(
        self,
        perceptions: Dict[Modality, ModalityPerception]
    ) -> float:
        """Calculate semantic coherence between modalities"""
        if len(perceptions) < 2:
            return 1.0

        embeddings = [
            p.semantic_embedding
            for p in perceptions.values()
            if p.semantic_embedding is not None
        ]

        if len(embeddings) < 2:
            return 0.8

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                e1, e2 = embeddings[i], embeddings[j]
                norm1 = np.linalg.norm(e1) + 1e-8
                norm2 = np.linalg.norm(e2) + 1e-8
                sim = np.dot(e1, e2) / (norm1 * norm2)
                similarities.append((sim + 1) / 2)  # Normalize to [0, 1]

        return np.mean(similarities) if similarities else 0.8

    def _fuse_representations(
        self,
        perceptions: Dict[Modality, ModalityPerception]
    ) -> np.ndarray:
        """Fuse modality representations into unified representation"""
        # Weighted concatenation of semantic embeddings
        fused_features = []
        total_weight = 0.0

        for modality, perception in perceptions.items():
            if perception.semantic_embedding is not None:
                weight = perception.confidence * perception.attention_weight
                fused_features.append(perception.semantic_embedding * weight)
                total_weight += weight

        if not fused_features:
            return np.zeros(self.integration_dim)

        # Average weighted features
        combined = np.sum(fused_features, axis=0) / (total_weight + 1e-8)

        # Project to integration space
        unified = combined @ self.fusion_weights
        return np.tanh(unified)


# ============================================================================
# Multi-Modal Consciousness
# ============================================================================

class MultiModalConsciousness:
    """
    Consciousness spanning multiple sensory modalities.

    Integrates vision, audio, language, proprioception, etc. to create
    a unified conscious experience richer than any single modality.

    Features:
    - Multiple modality processors
    - Cross-modal integration
    - Attention-based weighting
    - Coherence monitoring
    - Qualia generation

    Example:
        mm_consciousness = MultiModalConsciousness()

        # Perceive multi-modal stimuli
        perception = mm_consciousness.perceive_multimodal({
            Modality.VISION: image_array,
            Modality.AUDIO: audio_array,
            Modality.LANGUAGE: "a barking dog"
        })

        print(f"Coherence: {perception.coherence:.3f}")
        print(f"Dominant modality: {perception.integrated_understanding.dominant_modality}")
    """

    def __init__(self):
        self.modalities: Dict[Modality, ModalityProcessor] = {
            Modality.VISION: VisualProcessor(),
            Modality.AUDIO: AudioProcessor(),
            Modality.LANGUAGE: LanguageProcessor(),
            Modality.PROPRIOCEPTION: ProprioceptionProcessor()
        }

        self.cross_modal_integration = CrossModalIntegrator()

        # Attention weights for each modality
        self.attention_weights: Dict[Modality, float] = {
            modality: 1.0 for modality in self.modalities
        }

        # Perception history
        self.perception_history: List[MultiModalPerception] = []

    def add_modality(self, processor: ModalityProcessor):
        """Add a new modality processor"""
        self.modalities[processor.modality] = processor
        self.attention_weights[processor.modality] = 1.0

    def set_attention(self, modality: Modality, weight: float):
        """Set attention weight for a modality"""
        if modality in self.attention_weights:
            self.attention_weights[modality] = np.clip(weight, 0.0, 2.0)

    def perceive_multimodal(
        self,
        stimuli: Dict[Modality, Any]
    ) -> MultiModalPerception:
        """
        Perceive across modalities and integrate.

        Args:
            stimuli: Dictionary mapping modalities to raw inputs

        Returns:
            MultiModalPerception with integrated understanding
        """
        perceptions: Dict[Modality, ModalityPerception] = {}

        # Process each modality
        for modality, stimulus in stimuli.items():
            if modality in self.modalities:
                perception = self.modalities[modality].process(stimulus)
                perception.attention_weight = self.attention_weights.get(modality, 1.0)
                perceptions[modality] = perception

        # Cross-modal integration
        integrated = self.cross_modal_integration.integrate(perceptions)

        # Calculate overall coherence
        coherence = self._calculate_coherence(perceptions, integrated)

        # Generate qualia
        qualia = self._generate_qualia(perceptions, integrated)

        # Create multi-modal perception
        result = MultiModalPerception(
            modality_perceptions=perceptions,
            integrated_understanding=integrated,
            coherence=coherence,
            binding_confidence=integrated.binding_strength,
            qualia=qualia
        )

        # Store in history
        self.perception_history.append(result)
        if len(self.perception_history) > 100:
            self.perception_history = self.perception_history[-100:]

        return result

    def _calculate_coherence(
        self,
        perceptions: Dict[Modality, ModalityPerception],
        integrated: IntegratedPerception
    ) -> float:
        """Calculate overall perceptual coherence"""
        if len(perceptions) < 2:
            return 1.0

        # Combine different coherence measures
        coherence = (
            integrated.temporal_coherence * 0.3 +
            integrated.semantic_coherence * 0.5 +
            integrated.spatial_coherence * 0.2
        )

        # Adjust by number of modalities (more modalities = harder to integrate)
        modality_penalty = 1.0 / (1.0 + 0.1 * (len(perceptions) - 2))

        return coherence * modality_penalty

    def _generate_qualia(
        self,
        perceptions: Dict[Modality, ModalityPerception],
        integrated: IntegratedPerception
    ) -> Qualia:
        """Generate subjective experience marker"""
        # Intensity based on number of modalities and confidence
        avg_confidence = np.mean([p.confidence for p in perceptions.values()])
        intensity = avg_confidence * (1 + 0.1 * len(perceptions))

        # Valence from unified representation (simplified)
        valence = float(np.mean(integrated.unified_representation[:10]))
        valence = np.clip(valence, -1.0, 1.0)

        # Experience type based on dominant modality
        if integrated.dominant_modality:
            experience_type = f"multimodal_{integrated.dominant_modality.value}"
        else:
            experience_type = "multimodal_integrated"

        # Uniqueness based on how different this is from recent perceptions
        uniqueness = self._calculate_uniqueness(integrated)

        return Qualia(
            experience_type=experience_type,
            intensity=np.clip(intensity, 0.0, 1.0),
            valence=valence,
            content=f"Integrated perception across {len(perceptions)} modalities",
            timestamp=time.time(),
            uniqueness=uniqueness
        )

    def _calculate_uniqueness(self, integrated: IntegratedPerception) -> float:
        """Calculate how unique this perception is"""
        if len(self.perception_history) < 2:
            return 0.5

        # Compare to recent perceptions
        recent_representations = [
            p.integrated_understanding.unified_representation
            for p in self.perception_history[-10:]
        ]

        similarities = []
        for rep in recent_representations:
            sim = np.dot(integrated.unified_representation, rep)
            sim /= (np.linalg.norm(integrated.unified_representation) + 1e-8)
            sim /= (np.linalg.norm(rep) + 1e-8)
            similarities.append(abs(sim))

        avg_similarity = np.mean(similarities) if similarities else 0.5
        uniqueness = 1.0 - avg_similarity

        return np.clip(uniqueness, 0.0, 1.0)

    def get_modality_stats(self) -> Dict[str, Any]:
        """Get statistics about modality processing"""
        if not self.perception_history:
            return {"status": "No perceptions recorded"}

        modality_counts = {m.value: 0 for m in Modality}
        coherence_scores = []
        dominant_counts = {m.value: 0 for m in Modality}

        for perception in self.perception_history:
            for modality in perception.modality_perceptions:
                modality_counts[modality.value] += 1

            coherence_scores.append(perception.coherence)

            if perception.integrated_understanding.dominant_modality:
                dominant_counts[perception.integrated_understanding.dominant_modality.value] += 1

        return {
            "total_perceptions": len(self.perception_history),
            "modality_usage": modality_counts,
            "dominant_modality_frequency": dominant_counts,
            "avg_coherence": np.mean(coherence_scores),
            "min_coherence": min(coherence_scores),
            "max_coherence": max(coherence_scores)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Multi-Modal Consciousness Demo")
    print("=" * 50)

    # Create multi-modal consciousness
    mm = MultiModalConsciousness()

    # Simulate perceiving a scene
    print("\n1. Perceiving multi-modal scene (vision + audio + language)...")

    visual_input = np.random.randn(256)  # Simulated image features
    audio_input = np.random.randn(128)   # Simulated audio features
    language_input = "a dog barking in the park"

    perception = mm.perceive_multimodal({
        Modality.VISION: visual_input,
        Modality.AUDIO: audio_input,
        Modality.LANGUAGE: language_input
    })

    print(f"   Coherence: {perception.coherence:.3f}")
    print(f"   Binding confidence: {perception.binding_confidence:.3f}")
    print(f"   Dominant modality: {perception.integrated_understanding.dominant_modality}")
    print(f"   Temporal coherence: {perception.integrated_understanding.temporal_coherence:.3f}")
    print(f"   Semantic coherence: {perception.integrated_understanding.semantic_coherence:.3f}")

    # Perceive with different attention
    print("\n2. Same scene with visual attention boosted...")
    mm.set_attention(Modality.VISION, 2.0)
    mm.set_attention(Modality.AUDIO, 0.5)

    perception2 = mm.perceive_multimodal({
        Modality.VISION: visual_input,
        Modality.AUDIO: audio_input,
        Modality.LANGUAGE: language_input
    })

    print(f"   Dominant modality: {perception2.integrated_understanding.dominant_modality}")

    # Show qualia
    print("\n3. Qualia generated:")
    if perception.qualia:
        print(f"   Type: {perception.qualia.experience_type}")
        print(f"   Intensity: {perception.qualia.intensity:.3f}")
        print(f"   Valence: {perception.qualia.valence:.3f}")
        print(f"   Uniqueness: {perception.qualia.uniqueness:.3f}")

    # Statistics
    print("\n4. Modality Statistics:")
    stats = mm.get_modality_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
