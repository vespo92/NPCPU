"""
SYNAPSE-M: Attention Filter

Advanced selective attention mechanisms for multi-modal perception.
Implements bottom-up and top-down attention, saliency computation,
and attentional filtering.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod

from .modality_types import (
    ExtendedModality, ProcessedModality, ModalityCharacteristics,
    get_modality_characteristics, ModalityDomain
)


# ============================================================================
# Attention Types
# ============================================================================

class AttentionType(Enum):
    """Types of attention mechanisms"""
    BOTTOM_UP = auto()      # Stimulus-driven (salience)
    TOP_DOWN = auto()       # Goal-directed (voluntary)
    ALERTING = auto()       # Arousal/vigilance
    ORIENTING = auto()      # Spatial attention
    EXECUTIVE = auto()      # Conflict resolution


class AttentionPriority(Enum):
    """Priority levels for attention"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ============================================================================
# Attention Data Structures
# ============================================================================

@dataclass
class AttentionWeight:
    """Attention weight for a modality or target"""
    target_id: str
    modality: Optional[ExtendedModality] = None
    weight: float = 1.0
    attention_type: AttentionType = AttentionType.BOTTOM_UP
    priority: AttentionPriority = AttentionPriority.NORMAL
    decay_rate: float = 0.01
    timestamp: float = field(default_factory=time.time)

    def decay(self, dt: float) -> float:
        """Apply decay to weight and return new weight"""
        self.weight *= np.exp(-self.decay_rate * dt)
        return self.weight


@dataclass
class SaliencyMap:
    """
    Multi-modal saliency map for attention allocation.
    """
    modality_saliency: Dict[ExtendedModality, float] = field(default_factory=dict)
    spatial_saliency: Optional[np.ndarray] = None
    feature_saliency: Optional[np.ndarray] = None
    peak_saliency: float = 0.0
    peak_location: Optional[Tuple[ExtendedModality, int]] = None
    timestamp: float = field(default_factory=time.time)

    def get_top_k_modalities(self, k: int = 3) -> List[Tuple[ExtendedModality, float]]:
        """Get top-k salient modalities"""
        sorted_items = sorted(
            self.modality_saliency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_items[:k]


@dataclass
class AttentionFocus:
    """Current focus of attention"""
    primary_target: Optional[str] = None
    primary_modality: Optional[ExtendedModality] = None
    focus_strength: float = 1.0
    focus_duration: float = 0.0
    started_at: float = field(default_factory=time.time)
    suppressed_modalities: Set[ExtendedModality] = field(default_factory=set)

    def get_duration(self) -> float:
        """Get current focus duration"""
        return time.time() - self.started_at


# ============================================================================
# Saliency Computation
# ============================================================================

class SaliencyComputer:
    """
    Computes bottom-up saliency for perceptions.

    Uses multiple cues:
    - Intensity/contrast
    - Novelty
    - Motion/change
    - Emotional valence
    """

    def __init__(
        self,
        intensity_weight: float = 0.3,
        novelty_weight: float = 0.3,
        change_weight: float = 0.25,
        valence_weight: float = 0.15
    ):
        self.intensity_weight = intensity_weight
        self.novelty_weight = novelty_weight
        self.change_weight = change_weight
        self.valence_weight = valence_weight

        # History for novelty/change detection
        self._perception_history: Dict[ExtendedModality, deque] = {}
        self._history_size = 20

    def compute_saliency(
        self,
        perception: ProcessedModality
    ) -> float:
        """
        Compute bottom-up saliency score for a perception.

        Returns:
            Saliency score in [0, 1]
        """
        modality = perception.modality

        # Initialize history if needed
        if modality not in self._perception_history:
            self._perception_history[modality] = deque(maxlen=self._history_size)

        history = self._perception_history[modality]

        # Intensity component
        intensity_saliency = perception.raw_input.intensity

        # Novelty component (how different from history)
        novelty_saliency = self._compute_novelty(perception, history)

        # Change component (rate of change)
        change_saliency = self._compute_change(perception, history)

        # Valence component (emotional significance)
        valence_saliency = self._compute_valence(perception)

        # Combine components
        saliency = (
            self.intensity_weight * intensity_saliency +
            self.novelty_weight * novelty_saliency +
            self.change_weight * change_saliency +
            self.valence_weight * valence_saliency
        )

        # Store in history
        history.append({
            "features": perception.features.copy(),
            "timestamp": perception.timestamp
        })

        return float(np.clip(saliency, 0.0, 1.0))

    def _compute_novelty(
        self,
        perception: ProcessedModality,
        history: deque
    ) -> float:
        """Compute novelty based on feature divergence from history"""
        if len(history) < 2:
            return 0.5  # Neutral novelty for insufficient history

        # Compare to average of history
        history_features = np.array([h["features"] for h in history])
        mean_features = np.mean(history_features, axis=0)

        # Align dimensions
        current = perception.features
        min_dim = min(len(current), len(mean_features))
        current_aligned = current[:min_dim]
        mean_aligned = mean_features[:min_dim]

        # Compute distance (normalized)
        distance = np.linalg.norm(current_aligned - mean_aligned)
        max_distance = np.sqrt(min_dim)  # Maximum possible distance

        novelty = distance / max_distance
        return float(np.clip(novelty, 0.0, 1.0))

    def _compute_change(
        self,
        perception: ProcessedModality,
        history: deque
    ) -> float:
        """Compute rate of change from recent history"""
        if len(history) < 1:
            return 0.5

        # Compare to most recent
        last = history[-1]
        last_features = last["features"]
        dt = perception.timestamp - last["timestamp"]

        if dt <= 0:
            return 0.5

        # Feature velocity
        min_dim = min(len(perception.features), len(last_features))
        diff = perception.features[:min_dim] - last_features[:min_dim]
        velocity = np.linalg.norm(diff) / dt

        # Normalize (assuming typical velocity range)
        normalized_velocity = velocity / 10.0
        return float(np.clip(normalized_velocity, 0.0, 1.0))

    def _compute_valence(self, perception: ProcessedModality) -> float:
        """Compute emotional valence/significance"""
        # Check modality domain - internal states are more salient
        chars = get_modality_characteristics(perception.modality)

        if chars.domain == ModalityDomain.INTEROCEPTIVE:
            base_valence = 0.7  # Internal states are salient
        elif perception.modality == ExtendedModality.NOCICEPTION:
            base_valence = 1.0  # Pain is maximally salient
        else:
            base_valence = 0.3

        # Modulate by confidence
        return base_valence * perception.confidence

    def compute_saliency_map(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> SaliencyMap:
        """Compute saliency map across all perceptions."""
        modality_saliency: Dict[ExtendedModality, float] = {}
        peak_saliency = 0.0
        peak_modality = None

        for modality, perception in perceptions.items():
            saliency = self.compute_saliency(perception)
            modality_saliency[modality] = saliency

            if saliency > peak_saliency:
                peak_saliency = saliency
                peak_modality = modality

        return SaliencyMap(
            modality_saliency=modality_saliency,
            peak_saliency=peak_saliency,
            peak_location=(peak_modality, 0) if peak_modality else None
        )


# ============================================================================
# Attention Filter
# ============================================================================

class AttentionFilter:
    """
    Filters perceptions based on attention allocation.

    Implements:
    - Selective attention (focus on relevant)
    - Inhibition of return (avoid re-attending)
    - Attentional blink (reduced sensitivity after attention capture)
    - Capacity limits (can't attend to everything)
    """

    def __init__(
        self,
        attention_capacity: float = 1.0,
        inhibition_duration_ms: float = 500.0,
        blink_duration_ms: float = 300.0,
        focus_decay_rate: float = 0.1
    ):
        self.attention_capacity = attention_capacity
        self.inhibition_duration_ms = inhibition_duration_ms
        self.blink_duration_ms = blink_duration_ms
        self.focus_decay_rate = focus_decay_rate

        # Current attention state
        self.current_focus = AttentionFocus()
        self.attention_weights: Dict[str, AttentionWeight] = {}
        self.attention_spent: float = 0.0

        # Inhibition tracking
        self.inhibited_targets: Dict[str, float] = {}  # target -> inhibition_end_time

        # Blink tracking
        self.in_blink: bool = False
        self.blink_end_time: float = 0.0

        # Saliency computer
        self.saliency_computer = SaliencyComputer()

        # Goals for top-down attention
        self.attention_goals: Dict[ExtendedModality, float] = {}

    def set_goal(self, modality: ExtendedModality, priority: float):
        """Set top-down attention goal for a modality"""
        self.attention_goals[modality] = np.clip(priority, 0.0, 1.0)

    def clear_goal(self, modality: ExtendedModality):
        """Clear attention goal"""
        self.attention_goals.pop(modality, None)

    def filter_perceptions(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> Dict[ExtendedModality, ProcessedModality]:
        """
        Filter perceptions based on attention.

        Returns:
            Filtered perceptions with attention weights applied
        """
        if not perceptions:
            return {}

        current_time = time.time()

        # Update blink state
        if self.in_blink and current_time > self.blink_end_time:
            self.in_blink = False

        # Compute saliency
        saliency_map = self.saliency_computer.compute_saliency_map(perceptions)

        # Combine bottom-up (saliency) and top-down (goals) attention
        combined_attention = self._compute_combined_attention(perceptions, saliency_map)

        # Apply capacity limit - keep top perceptions
        sorted_perceptions = sorted(
            perceptions.items(),
            key=lambda x: combined_attention.get(x[0], 0.0),
            reverse=True
        )

        # Select perceptions that fit within capacity
        filtered: Dict[ExtendedModality, ProcessedModality] = {}
        remaining_capacity = self.attention_capacity

        for modality, perception in sorted_perceptions:
            # Check if inhibited
            if self._is_inhibited(modality.value):
                continue

            # Apply blink reduction
            attention = combined_attention.get(modality, 0.0)
            if self.in_blink:
                attention *= 0.3

            # Cost proportional to attention required
            cost = attention * 0.3

            if cost <= remaining_capacity:
                # Apply attention weight to perception
                weighted_perception = self._apply_attention_weight(
                    perception, attention
                )
                filtered[modality] = weighted_perception
                remaining_capacity -= cost

        # Update focus
        if filtered:
            top_modality = list(filtered.keys())[0]
            self._update_focus(top_modality)

        self.attention_spent = self.attention_capacity - remaining_capacity
        return filtered

    def _compute_combined_attention(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality],
        saliency_map: SaliencyMap
    ) -> Dict[ExtendedModality, float]:
        """Combine bottom-up and top-down attention"""
        combined = {}

        for modality in perceptions:
            # Bottom-up: saliency
            bottom_up = saliency_map.modality_saliency.get(modality, 0.5)

            # Top-down: goal priority
            top_down = self.attention_goals.get(modality, 0.5)

            # Combine (biased toward whichever is higher)
            combined[modality] = max(bottom_up, top_down) * 0.7 + min(bottom_up, top_down) * 0.3

        return combined

    def _is_inhibited(self, target_id: str) -> bool:
        """Check if target is currently inhibited"""
        if target_id not in self.inhibited_targets:
            return False

        current_time = time.time()
        end_time = self.inhibited_targets[target_id]

        if current_time > end_time:
            del self.inhibited_targets[target_id]
            return False

        return True

    def _apply_attention_weight(
        self,
        perception: ProcessedModality,
        attention: float
    ) -> ProcessedModality:
        """Apply attention weight to perception"""
        # Create weighted copy
        weighted = ProcessedModality(
            modality=perception.modality,
            raw_input=perception.raw_input,
            features=perception.features * attention,
            semantic_embedding=perception.semantic_embedding * attention,
            salience=perception.salience * attention,
            confidence=perception.confidence,
            processing_time_ms=perception.processing_time_ms,
            timestamp=perception.timestamp,
            metadata={**perception.metadata, "attention_weight": attention}
        )
        return weighted

    def _update_focus(self, modality: ExtendedModality):
        """Update attention focus"""
        if self.current_focus.primary_modality != modality:
            # Switching focus - apply inhibition to old target
            if self.current_focus.primary_modality:
                old_target = self.current_focus.primary_modality.value
                self.inhibited_targets[old_target] = (
                    time.time() + self.inhibition_duration_ms / 1000.0
                )

            # Start new focus
            self.current_focus = AttentionFocus(
                primary_modality=modality,
                focus_strength=1.0
            )

            # Trigger attentional blink
            self.in_blink = True
            self.blink_end_time = time.time() + self.blink_duration_ms / 1000.0
        else:
            # Continuing focus - update duration
            self.current_focus.focus_duration = self.current_focus.get_duration()

    def get_attention_state(self) -> Dict[str, Any]:
        """Get current attention state"""
        return {
            "capacity": self.attention_capacity,
            "spent": self.attention_spent,
            "available": self.attention_capacity - self.attention_spent,
            "focus_modality": (
                self.current_focus.primary_modality.value
                if self.current_focus.primary_modality else None
            ),
            "focus_duration": self.current_focus.focus_duration,
            "in_blink": self.in_blink,
            "inhibited_count": len(self.inhibited_targets),
            "goals": {m.value: p for m, p in self.attention_goals.items()}
        }


# ============================================================================
# Multi-Modal Attention Manager
# ============================================================================

class MultiModalAttentionManager:
    """
    Manages attention across all modalities with coordination.
    """

    def __init__(self, attention_capacity: float = 1.0):
        self.filter = AttentionFilter(attention_capacity=attention_capacity)
        self.modality_history: Dict[ExtendedModality, deque] = {}
        self._history_size = 50

    def allocate_attention(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> Dict[ExtendedModality, ProcessedModality]:
        """Allocate attention to perceptions."""
        return self.filter.filter_perceptions(perceptions)

    def set_modality_priority(self, modality: ExtendedModality, priority: float):
        """Set top-down priority for a modality."""
        self.filter.set_goal(modality, priority)

    def clear_modality_priority(self, modality: ExtendedModality):
        """Clear top-down priority."""
        self.filter.clear_goal(modality)

    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get attention allocation statistics."""
        return {
            "current_state": self.filter.get_attention_state(),
            "modality_priorities": {
                m.value: p for m, p in self.filter.attention_goals.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Attention Filter Demo")
    print("=" * 50)

    from .modality_types import ModalityInput

    # Create test perceptions
    np.random.seed(42)
    current_time = time.time()

    perceptions: Dict[ExtendedModality, ProcessedModality] = {}

    for modality in [ExtendedModality.VISION, ExtendedModality.AUDITION,
                     ExtendedModality.TACTILE, ExtendedModality.NOCICEPTION]:
        chars = get_modality_characteristics(modality)
        input_data = ModalityInput(
            modality=modality,
            raw_data=np.random.randn(chars.feature_dim),
            intensity=np.random.uniform(0.3, 1.0)
        )
        perceptions[modality] = ProcessedModality(
            modality=modality,
            raw_input=input_data,
            features=np.random.randn(chars.feature_dim),
            semantic_embedding=np.random.randn(chars.semantic_dim),
            salience=np.random.uniform(0.3, 1.0),
            confidence=np.random.uniform(0.7, 1.0),
            timestamp=current_time + np.random.uniform(-0.05, 0.05)
        )

    # Create attention manager
    manager = MultiModalAttentionManager(attention_capacity=1.0)

    print("\n1. Initial filtering (no goals):")
    filtered = manager.allocate_attention(perceptions)
    print(f"   Input modalities: {[m.value for m in perceptions.keys()]}")
    print(f"   Filtered modalities: {[m.value for m in filtered.keys()]}")

    state = manager.filter.get_attention_state()
    print(f"   Attention spent: {state['spent']:.2f}")
    print(f"   Focus: {state['focus_modality']}")

    print("\n2. With vision goal priority:")
    manager.set_modality_priority(ExtendedModality.VISION, 1.0)
    filtered2 = manager.allocate_attention(perceptions)
    print(f"   Filtered modalities: {[m.value for m in filtered2.keys()]}")

    state2 = manager.filter.get_attention_state()
    print(f"   Focus: {state2['focus_modality']}")

    print("\n3. Saliency computation:")
    saliency = SaliencyComputer()
    saliency_map = saliency.compute_saliency_map(perceptions)
    print("   Modality saliencies:")
    for m, s in saliency_map.get_top_k_modalities(4):
        print(f"      {m.value}: {s:.3f}")
    print(f"   Peak saliency: {saliency_map.peak_saliency:.3f}")

    print("\n4. Attention statistics:")
    stats = manager.get_attention_statistics()
    for key, value in stats["current_state"].items():
        print(f"   {key}: {value}")
