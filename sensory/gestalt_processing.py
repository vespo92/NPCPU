"""
SYNAPSE-M: Gestalt Processing

Pattern completion and perceptual grouping based on Gestalt principles.
Implements proximity, similarity, continuity, closure, and figure-ground.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

from .modality_types import (
    ExtendedModality, ProcessedModality, ModalityCharacteristics,
    get_modality_characteristics
)


# ============================================================================
# Gestalt Principles
# ============================================================================

class GestaltPrinciple(Enum):
    """Gestalt principles of perceptual organization"""
    PROXIMITY = auto()        # Elements close together group together
    SIMILARITY = auto()       # Similar elements group together
    CONTINUITY = auto()       # Elements forming smooth paths group
    CLOSURE = auto()          # Incomplete shapes completed
    COMMON_FATE = auto()      # Elements moving together group
    FIGURE_GROUND = auto()    # Separate foreground from background
    SYMMETRY = auto()         # Symmetric elements group
    COMMON_REGION = auto()    # Elements in same region group
    CONNECTEDNESS = auto()    # Connected elements group


# ============================================================================
# Gestalt Data Structures
# ============================================================================

@dataclass
class PerceptualGroup:
    """
    A group of perceptions organized by Gestalt principles.
    """
    id: str = field(default_factory=lambda: str(id(object())))
    members: List[ProcessedModality] = field(default_factory=list)
    principle: GestaltPrinciple = GestaltPrinciple.PROXIMITY
    grouping_strength: float = 0.5
    centroid: Optional[np.ndarray] = None

    # Group properties
    coherence: float = 0.5
    salience: float = 0.5
    is_figure: bool = True  # True=figure, False=ground

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def size(self) -> int:
        return len(self.members)

    def add_member(self, perception: ProcessedModality):
        """Add member to group"""
        self.members.append(perception)
        self._update_centroid()

    def _update_centroid(self):
        """Update group centroid"""
        if self.members:
            features = [m.features for m in self.members]
            self.centroid = np.mean(features, axis=0)


@dataclass
class CompletedPattern:
    """
    A pattern with closure/completion applied.
    """
    original: ProcessedModality
    completed_features: np.ndarray
    completion_confidence: float
    missing_regions: List[Tuple[int, int]] = field(default_factory=list)
    completion_source: str = "inference"


@dataclass
class FigureGroundSegmentation:
    """
    Figure-ground separation result.
    """
    figures: List[PerceptualGroup] = field(default_factory=list)
    ground: Optional[PerceptualGroup] = None
    separation_strength: float = 0.5
    ambiguity: float = 0.0  # 0=clear, 1=ambiguous (like Rubin's vase)


# ============================================================================
# Gestalt Processors
# ============================================================================

class GestaltProcessor(ABC):
    """Abstract base for Gestalt principle processors"""

    def __init__(self, principle: GestaltPrinciple):
        self.principle = principle

    @abstractmethod
    def apply(
        self,
        perceptions: List[ProcessedModality]
    ) -> List[PerceptualGroup]:
        """Apply Gestalt principle to find groups"""
        pass


class ProximityProcessor(GestaltProcessor):
    """
    Groups perceptions by proximity in feature space.
    """

    def __init__(self, distance_threshold: float = 0.5):
        super().__init__(GestaltPrinciple.PROXIMITY)
        self.distance_threshold = distance_threshold

    def apply(
        self,
        perceptions: List[ProcessedModality]
    ) -> List[PerceptualGroup]:
        """Group by proximity"""
        if not perceptions:
            return []

        # Compute pairwise distances
        n = len(perceptions)
        features = [p.features for p in perceptions]

        # Simple greedy clustering by proximity
        assigned = set()
        groups = []

        for i in range(n):
            if i in assigned:
                continue

            # Start new group
            group = PerceptualGroup(
                principle=self.principle,
                members=[perceptions[i]]
            )
            assigned.add(i)

            # Find nearby members
            for j in range(i + 1, n):
                if j in assigned:
                    continue

                # Compute distance
                min_dim = min(len(features[i]), len(features[j]))
                dist = np.linalg.norm(features[i][:min_dim] - features[j][:min_dim])
                normalized_dist = dist / np.sqrt(min_dim)

                if normalized_dist < self.distance_threshold:
                    group.add_member(perceptions[j])
                    assigned.add(j)

            # Calculate grouping strength
            if len(group.members) > 1:
                intra_distances = []
                for a in range(len(group.members)):
                    for b in range(a + 1, len(group.members)):
                        d = np.linalg.norm(
                            group.members[a].features[:min_dim] -
                            group.members[b].features[:min_dim]
                        )
                        intra_distances.append(d)
                group.grouping_strength = 1.0 / (1.0 + np.mean(intra_distances))
            else:
                group.grouping_strength = 0.3

            groups.append(group)

        return groups


class SimilarityProcessor(GestaltProcessor):
    """
    Groups perceptions by feature similarity.
    """

    def __init__(self, similarity_threshold: float = 0.6):
        super().__init__(GestaltPrinciple.SIMILARITY)
        self.similarity_threshold = similarity_threshold

    def apply(
        self,
        perceptions: List[ProcessedModality]
    ) -> List[PerceptualGroup]:
        """Group by similarity"""
        if not perceptions:
            return []

        # Use semantic embeddings for similarity
        n = len(perceptions)
        embeddings = [p.semantic_embedding for p in perceptions]

        assigned = set()
        groups = []

        for i in range(n):
            if i in assigned:
                continue

            group = PerceptualGroup(
                principle=self.principle,
                members=[perceptions[i]]
            )
            assigned.add(i)

            # Find similar members
            for j in range(i + 1, n):
                if j in assigned:
                    continue

                # Cosine similarity
                min_dim = min(len(embeddings[i]), len(embeddings[j]))
                e1 = embeddings[i][:min_dim]
                e2 = embeddings[j][:min_dim]
                similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)

                if similarity > self.similarity_threshold:
                    group.add_member(perceptions[j])
                    assigned.add(j)

            # Set grouping strength
            group.grouping_strength = self._compute_coherence(group.members)
            groups.append(group)

        return groups

    def _compute_coherence(self, members: List[ProcessedModality]) -> float:
        """Compute within-group coherence"""
        if len(members) < 2:
            return 0.3

        embeddings = [m.semantic_embedding for m in members]
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                min_dim = min(len(embeddings[i]), len(embeddings[j]))
                sim = np.dot(embeddings[i][:min_dim], embeddings[j][:min_dim])
                sim /= (np.linalg.norm(embeddings[i][:min_dim]) + 1e-8)
                sim /= (np.linalg.norm(embeddings[j][:min_dim]) + 1e-8)
                similarities.append(sim)

        return float(np.mean(similarities))


class ClosureProcessor(GestaltProcessor):
    """
    Completes incomplete patterns using closure principle.
    """

    def __init__(self, completion_threshold: float = 0.3):
        super().__init__(GestaltPrinciple.CLOSURE)
        self.completion_threshold = completion_threshold

        # Learned pattern templates (simplified)
        np.random.seed(50)
        self._templates: Dict[str, np.ndarray] = {}

    def apply(
        self,
        perceptions: List[ProcessedModality]
    ) -> List[PerceptualGroup]:
        """Apply closure to complete patterns"""
        groups = []

        for perception in perceptions:
            completed = self.complete_pattern(perception)

            group = PerceptualGroup(
                principle=self.principle,
                members=[perception],
                grouping_strength=completed.completion_confidence
            )

            # Store completion info
            group.metadata["completed_pattern"] = completed

            groups.append(group)

        return groups

    def complete_pattern(self, perception: ProcessedModality) -> CompletedPattern:
        """Complete an incomplete pattern"""
        features = perception.features

        # Detect "missing" regions (low activation areas)
        missing_mask = np.abs(features) < self.completion_threshold
        missing_indices = np.where(missing_mask)[0]

        if len(missing_indices) == 0:
            # Pattern is complete
            return CompletedPattern(
                original=perception,
                completed_features=features.copy(),
                completion_confidence=1.0,
                completion_source="none_needed"
            )

        # Complete using interpolation/inference
        completed = features.copy()

        for idx in missing_indices:
            # Simple interpolation from neighbors
            left = max(0, idx - 1)
            right = min(len(features) - 1, idx + 1)

            if left != idx and right != idx:
                completed[idx] = (features[left] + features[right]) / 2
            elif left != idx:
                completed[idx] = features[left]
            else:
                completed[idx] = features[right]

        # Confidence based on how much was missing
        missing_ratio = len(missing_indices) / len(features)
        confidence = 1.0 - missing_ratio

        return CompletedPattern(
            original=perception,
            completed_features=completed,
            completion_confidence=float(confidence),
            missing_regions=[(int(missing_indices[0]), int(missing_indices[-1]))] if len(missing_indices) > 0 else [],
            completion_source="interpolation"
        )

    def add_template(self, name: str, template: np.ndarray):
        """Add a pattern template for completion"""
        self._templates[name] = template


class FigureGroundProcessor(GestaltProcessor):
    """
    Separates figure from ground in perceptual scene.
    """

    def __init__(
        self,
        salience_threshold: float = 0.5,
        convexity_weight: float = 0.3
    ):
        super().__init__(GestaltPrinciple.FIGURE_GROUND)
        self.salience_threshold = salience_threshold
        self.convexity_weight = convexity_weight

    def apply(
        self,
        perceptions: List[ProcessedModality]
    ) -> List[PerceptualGroup]:
        """Separate figure and ground"""
        if not perceptions:
            return []

        segmentation = self.segment(perceptions)

        # Return figures as groups
        groups = segmentation.figures
        if segmentation.ground:
            groups.append(segmentation.ground)

        return groups

    def segment(
        self,
        perceptions: List[ProcessedModality]
    ) -> FigureGroundSegmentation:
        """Perform figure-ground segmentation"""
        if not perceptions:
            return FigureGroundSegmentation()

        # Score each perception for "figureness"
        figure_scores = []
        for p in perceptions:
            score = self._compute_figure_score(p)
            figure_scores.append(score)

        # Threshold to separate
        threshold = np.median(figure_scores)

        figures_list = []
        ground_list = []

        for p, score in zip(perceptions, figure_scores):
            if score > threshold:
                figures_list.append(p)
            else:
                ground_list.append(p)

        # Create figure groups
        figures = []
        if figures_list:
            figure_group = PerceptualGroup(
                principle=self.principle,
                members=figures_list,
                is_figure=True,
                salience=float(np.mean([self._compute_figure_score(p) for p in figures_list]))
            )
            figures.append(figure_group)

        # Create ground group
        ground = None
        if ground_list:
            ground = PerceptualGroup(
                principle=self.principle,
                members=ground_list,
                is_figure=False,
                salience=float(np.mean([self._compute_figure_score(p) for p in ground_list]))
            )

        # Compute ambiguity
        if figure_scores:
            variance = np.var(figure_scores)
            ambiguity = 1.0 / (1.0 + variance * 10)
        else:
            ambiguity = 0.5

        return FigureGroundSegmentation(
            figures=figures,
            ground=ground,
            separation_strength=float(1.0 - ambiguity),
            ambiguity=float(ambiguity)
        )

    def _compute_figure_score(self, perception: ProcessedModality) -> float:
        """Compute how figure-like a perception is"""
        # Factors that make something a figure:
        # - Higher salience
        # - More coherent features (lower variance)
        # - Smaller/bounded (simplified)

        salience_score = perception.salience

        # Feature coherence
        feature_var = np.var(perception.features)
        coherence_score = 1.0 / (1.0 + feature_var)

        # Combine scores
        figure_score = (
            0.5 * salience_score +
            0.3 * coherence_score +
            0.2 * perception.confidence
        )

        return float(np.clip(figure_score, 0.0, 1.0))


# ============================================================================
# Gestalt Integration System
# ============================================================================

class GestaltProcessor:
    """
    Unified Gestalt processing system.

    Applies multiple Gestalt principles to organize perception.
    """

    def __init__(self):
        # Individual processors
        self.proximity = ProximityProcessor()
        self.similarity = SimilarityProcessor()
        self.closure = ClosureProcessor()
        self.figure_ground = FigureGroundProcessor()

        # Weights for combining principles
        self.principle_weights = {
            GestaltPrinciple.PROXIMITY: 0.25,
            GestaltPrinciple.SIMILARITY: 0.25,
            GestaltPrinciple.CLOSURE: 0.20,
            GestaltPrinciple.FIGURE_GROUND: 0.30
        }

    def process(
        self,
        perceptions: List[ProcessedModality]
    ) -> Dict[str, Any]:
        """
        Apply all Gestalt principles to organize perceptions.

        Returns:
            Dictionary with groupings from each principle
        """
        results = {
            "proximity_groups": self.proximity.apply(perceptions),
            "similarity_groups": self.similarity.apply(perceptions),
            "closure_groups": self.closure.apply(perceptions),
            "figure_ground": self.figure_ground.segment(perceptions)
        }

        # Find best grouping
        results["best_grouping"] = self._select_best_grouping(results)

        return results

    def _select_best_grouping(
        self,
        results: Dict[str, Any]
    ) -> List[PerceptualGroup]:
        """Select best grouping based on coherence"""
        candidates = [
            ("proximity", results["proximity_groups"]),
            ("similarity", results["similarity_groups"]),
            ("closure", results["closure_groups"]),
            ("figure_ground", results["figure_ground"].figures)
        ]

        best_score = -1
        best_groups = []

        for name, groups in candidates:
            if not groups:
                continue

            # Score by average grouping strength
            avg_strength = np.mean([g.grouping_strength for g in groups])

            # Get weight for this principle
            principle = next(
                (p for p in GestaltPrinciple if p.name.lower() == name.split("_")[0]),
                GestaltPrinciple.PROXIMITY
            )
            weight = self.principle_weights.get(principle, 0.25)

            score = avg_strength * weight

            if score > best_score:
                best_score = score
                best_groups = groups

        return best_groups

    def complete_pattern(
        self,
        perception: ProcessedModality
    ) -> CompletedPattern:
        """Apply closure to complete a single pattern"""
        return self.closure.complete_pattern(perception)

    def get_figure_ground(
        self,
        perceptions: List[ProcessedModality]
    ) -> FigureGroundSegmentation:
        """Get figure-ground segmentation"""
        return self.figure_ground.segment(perceptions)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Gestalt Processing Demo")
    print("=" * 50)

    from .modality_types import ModalityInput

    # Create test perceptions with varying similarity
    np.random.seed(42)
    perceptions = []

    # Group 1: Similar features
    for i in range(3):
        input_data = ModalityInput(
            modality=ExtendedModality.VISION,
            raw_data=np.random.randn(256),
            intensity=0.8
        )
        features = np.random.randn(256) * 0.1 + 1.0  # Centered around 1
        perceptions.append(ProcessedModality(
            modality=ExtendedModality.VISION,
            raw_input=input_data,
            features=features,
            semantic_embedding=np.random.randn(128) * 0.1 + 0.5,
            salience=0.7,
            confidence=0.9
        ))

    # Group 2: Different features
    for i in range(3):
        input_data = ModalityInput(
            modality=ExtendedModality.VISION,
            raw_data=np.random.randn(256),
            intensity=0.6
        )
        features = np.random.randn(256) * 0.1 - 1.0  # Centered around -1
        perceptions.append(ProcessedModality(
            modality=ExtendedModality.VISION,
            raw_input=input_data,
            features=features,
            semantic_embedding=np.random.randn(128) * 0.1 - 0.5,
            salience=0.4,
            confidence=0.8
        ))

    # Create processor
    gestalt = GestaltProcessor()

    print("\n1. Applying Gestalt processing...")
    results = gestalt.process(perceptions)

    print("\n2. Proximity groups:")
    for i, group in enumerate(results["proximity_groups"]):
        print(f"   Group {i}: {group.size()} members, strength={group.grouping_strength:.3f}")

    print("\n3. Similarity groups:")
    for i, group in enumerate(results["similarity_groups"]):
        print(f"   Group {i}: {group.size()} members, strength={group.grouping_strength:.3f}")

    print("\n4. Figure-Ground segmentation:")
    fg = results["figure_ground"]
    print(f"   Figures: {len(fg.figures)} groups")
    print(f"   Ground: {'Yes' if fg.ground else 'No'}")
    print(f"   Separation strength: {fg.separation_strength:.3f}")
    print(f"   Ambiguity: {fg.ambiguity:.3f}")

    print("\n5. Best grouping:")
    best = results["best_grouping"]
    print(f"   {len(best)} groups")
    for i, group in enumerate(best):
        print(f"   Group {i}: principle={group.principle.name}, "
              f"members={group.size()}, strength={group.grouping_strength:.3f}")

    print("\n6. Pattern completion test:")
    # Create perception with "missing" values
    sparse_features = np.zeros(256)
    sparse_features[::3] = np.random.randn(86)  # Only every 3rd value set
    input_data = ModalityInput(
        modality=ExtendedModality.VISION,
        raw_data=sparse_features,
        intensity=0.5
    )
    sparse_perception = ProcessedModality(
        modality=ExtendedModality.VISION,
        raw_input=input_data,
        features=sparse_features,
        semantic_embedding=np.random.randn(128),
        salience=0.5,
        confidence=0.6
    )

    completed = gestalt.complete_pattern(sparse_perception)
    print(f"   Completion confidence: {completed.completion_confidence:.3f}")
    print(f"   Source: {completed.completion_source}")
