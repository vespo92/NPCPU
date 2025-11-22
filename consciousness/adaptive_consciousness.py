"""
Adaptive Consciousness

Consciousness that dynamically adapts based on environment and task requirements.
Enables agents to optimize their cognitive capabilities in real-time.

Based on Month 2 roadmap: Adaptive Consciousness
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Context Types
# ============================================================================

class EnvironmentType(Enum):
    """Types of environments for adaptation"""
    SAFE = "safe"
    THREATENING = "threatening"
    RESOURCE_RICH = "resource_rich"
    RESOURCE_SCARCE = "resource_scarce"
    SOCIAL = "social"
    ISOLATED = "isolated"
    COMPLEX = "complex"
    SIMPLE = "simple"


class TaskType(Enum):
    """Types of tasks for adaptation"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    SOCIAL = "social"
    CREATIVE = "creative"
    MOTOR = "motor"
    SURVIVAL = "survival"


@dataclass
class EnvironmentContext:
    """Context describing the current environment"""
    threat_level: float = 0.0  # 0.0 (safe) to 1.0 (dangerous)
    resource_availability: float = 0.5  # 0.0 (scarce) to 1.0 (abundant)
    social_density: float = 0.0  # 0.0 (isolated) to 1.0 (crowded)
    complexity: float = 0.5  # 0.0 (simple) to 1.0 (complex)
    novelty: float = 0.5  # 0.0 (familiar) to 1.0 (novel)
    time_pressure: float = 0.0  # 0.0 (relaxed) to 1.0 (urgent)
    energy_available: float = 1.0  # 0.0 (depleted) to 1.0 (full)
    additional_features: Dict[str, float] = field(default_factory=dict)

    def classify(self) -> List[EnvironmentType]:
        """Classify environment into types"""
        types = []

        if self.threat_level > 0.7:
            types.append(EnvironmentType.THREATENING)
        elif self.threat_level < 0.3:
            types.append(EnvironmentType.SAFE)

        if self.resource_availability > 0.7:
            types.append(EnvironmentType.RESOURCE_RICH)
        elif self.resource_availability < 0.3:
            types.append(EnvironmentType.RESOURCE_SCARCE)

        if self.social_density > 0.6:
            types.append(EnvironmentType.SOCIAL)
        elif self.social_density < 0.2:
            types.append(EnvironmentType.ISOLATED)

        if self.complexity > 0.7:
            types.append(EnvironmentType.COMPLEX)
        elif self.complexity < 0.3:
            types.append(EnvironmentType.SIMPLE)

        return types


@dataclass
class TaskContext:
    """Context describing the current task"""
    task_type: TaskType
    difficulty: float = 0.5  # 0.0 to 1.0
    duration: float = 0.5  # Expected duration (normalized)
    importance: float = 0.5  # 0.0 to 1.0
    required_capabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdaptationRecord:
    """Record of a consciousness adaptation"""
    timestamp: float
    environment: EnvironmentContext
    task: Optional[TaskContext]
    original_scores: Dict[str, float]
    adapted_scores: Dict[str, float]
    adaptation_reason: str
    effectiveness: Optional[float] = None  # Filled in later


# ============================================================================
# Adaptive Consciousness
# ============================================================================

class AdaptiveConsciousness:
    """
    Consciousness that adapts based on environment and task.

    Dynamically adjusts capability scores to optimize for current context,
    allowing agents to be more responsive in threats, more reflective
    in safe environments, etc.

    Features:
    - Environment-responsive adaptation
    - Task-specific optimization
    - Adaptation history tracking
    - Energy-aware adjustments
    - Smooth transitions (no sudden jumps)

    Example:
        base = GradedConsciousness(
            perception_fidelity=0.7,
            introspection_capacity=0.7
        )
        adaptive = AdaptiveConsciousness(base)

        # Adapt to threatening environment
        env = EnvironmentContext(threat_level=0.9)
        adapted = adaptive.adapt_to_environment(env)

        # Perception increased, introspection decreased
        print(adapted.perception_fidelity)  # Higher
        print(adapted.introspection_capacity)  # Lower
    """

    def __init__(
        self,
        base_consciousness: GradedConsciousness,
        adaptation_rate: float = 0.3,
        max_adaptation: float = 0.4,
        min_capability: float = 0.1
    ):
        """
        Initialize adaptive consciousness.

        Args:
            base_consciousness: Base consciousness to adapt from
            adaptation_rate: How quickly to adapt (0.0 to 1.0)
            max_adaptation: Maximum change per adaptation
            min_capability: Minimum capability score allowed
        """
        self.base = base_consciousness
        self.current = GradedConsciousness(
            **base_consciousness.get_capability_scores()
        )
        self.adaptation_rate = adaptation_rate
        self.max_adaptation = max_adaptation
        self.min_capability = min_capability

        self.adaptation_history: List[AdaptationRecord] = []
        self.total_energy_spent = 0.0

    def adapt_to_environment(
        self,
        environment: EnvironmentContext
    ) -> GradedConsciousness:
        """
        Adapt consciousness to environment.

        Args:
            environment: Current environment context

        Returns:
            Adapted consciousness state
        """
        adapted_scores = self.current.get_capability_scores().copy()
        original_scores = adapted_scores.copy()

        # High threat: boost perception and reaction, reduce reflection
        if environment.threat_level > 0.5:
            threat_factor = environment.threat_level
            adapted_scores["perception_fidelity"] = self._boost(
                adapted_scores["perception_fidelity"],
                0.3 * threat_factor
            )
            adapted_scores["reaction_speed"] = self._boost(
                adapted_scores["reaction_speed"],
                0.4 * threat_factor
            )
            adapted_scores["introspection_capacity"] = self._reduce(
                adapted_scores["introspection_capacity"],
                0.3 * threat_factor
            )
            adapted_scores["meta_cognitive_ability"] = self._reduce(
                adapted_scores["meta_cognitive_ability"],
                0.2 * threat_factor
            )

        # Resource rich: enhance learning and reflection
        if environment.resource_availability > 0.7:
            resource_factor = environment.resource_availability - 0.5
            adapted_scores["introspection_capacity"] = self._boost(
                adapted_scores["introspection_capacity"],
                0.2 * resource_factor
            )
            adapted_scores["meta_cognitive_ability"] = self._boost(
                adapted_scores["meta_cognitive_ability"],
                0.2 * resource_factor
            )
            adapted_scores["memory_depth"] = self._boost(
                adapted_scores["memory_depth"],
                0.15 * resource_factor
            )

        # Social environment: enhance communication capabilities
        if environment.social_density > 0.6:
            social_factor = environment.social_density - 0.3
            adapted_scores["intentional_coherence"] = self._boost(
                adapted_scores["intentional_coherence"],
                0.3 * social_factor
            )
            adapted_scores["qualia_richness"] = self._boost(
                adapted_scores["qualia_richness"],
                0.2 * social_factor
            )

        # High complexity: boost information integration
        if environment.complexity > 0.6:
            complexity_factor = environment.complexity - 0.3
            adapted_scores["information_integration"] = self._boost(
                adapted_scores["information_integration"],
                0.25 * complexity_factor
            )
            adapted_scores["memory_recall_accuracy"] = self._boost(
                adapted_scores["memory_recall_accuracy"],
                0.15 * complexity_factor
            )

        # Time pressure: favor speed over accuracy
        if environment.time_pressure > 0.7:
            pressure_factor = environment.time_pressure - 0.5
            adapted_scores["reaction_speed"] = self._boost(
                adapted_scores["reaction_speed"],
                0.3 * pressure_factor
            )
            adapted_scores["memory_recall_accuracy"] = self._reduce(
                adapted_scores["memory_recall_accuracy"],
                0.15 * pressure_factor
            )

        # Low energy: reduce all capabilities slightly
        if environment.energy_available < 0.3:
            energy_penalty = 0.3 - environment.energy_available
            for capability in adapted_scores:
                adapted_scores[capability] = self._reduce(
                    adapted_scores[capability],
                    0.1 * energy_penalty
                )

        # Novelty: boost perception and reduce automation
        if environment.novelty > 0.7:
            novelty_factor = environment.novelty - 0.5
            adapted_scores["perception_fidelity"] = self._boost(
                adapted_scores["perception_fidelity"],
                0.2 * novelty_factor
            )
            adapted_scores["introspection_capacity"] = self._boost(
                adapted_scores["introspection_capacity"],
                0.15 * novelty_factor
            )

        # Apply smooth transition
        adapted_scores = self._smooth_transition(
            self.current.get_capability_scores(),
            adapted_scores
        )

        # Update current consciousness
        self.current = GradedConsciousness(**adapted_scores)

        # Record adaptation
        self.adaptation_history.append(AdaptationRecord(
            timestamp=time.time(),
            environment=environment,
            task=None,
            original_scores=original_scores,
            adapted_scores=adapted_scores,
            adaptation_reason=f"Environment: {environment.classify()}"
        ))

        return self.current

    def adapt_to_task(
        self,
        task: TaskContext
    ) -> GradedConsciousness:
        """
        Adapt consciousness to task requirements.

        Args:
            task: Current task context

        Returns:
            Adapted consciousness state
        """
        adapted_scores = self.current.get_capability_scores().copy()
        original_scores = adapted_scores.copy()

        # Meet minimum required capabilities
        for capability, required in task.required_capabilities.items():
            if capability in adapted_scores:
                if adapted_scores[capability] < required:
                    # Boost to meet requirement (with limits)
                    boost_needed = required - adapted_scores[capability]
                    adapted_scores[capability] = self._boost(
                        adapted_scores[capability],
                        min(boost_needed, self.max_adaptation)
                    )

        # Task type specific adaptations
        if task.task_type == TaskType.PERCEPTION:
            adapted_scores["perception_fidelity"] = self._boost(
                adapted_scores["perception_fidelity"],
                0.2 * task.difficulty
            )
            adapted_scores["reaction_speed"] = self._boost(
                adapted_scores["reaction_speed"],
                0.1 * task.difficulty
            )

        elif task.task_type == TaskType.REASONING:
            adapted_scores["meta_cognitive_ability"] = self._boost(
                adapted_scores["meta_cognitive_ability"],
                0.25 * task.difficulty
            )
            adapted_scores["information_integration"] = self._boost(
                adapted_scores["information_integration"],
                0.2 * task.difficulty
            )
            adapted_scores["introspection_capacity"] = self._boost(
                adapted_scores["introspection_capacity"],
                0.15 * task.difficulty
            )

        elif task.task_type == TaskType.MEMORY:
            adapted_scores["memory_depth"] = self._boost(
                adapted_scores["memory_depth"],
                0.25 * task.difficulty
            )
            adapted_scores["memory_recall_accuracy"] = self._boost(
                adapted_scores["memory_recall_accuracy"],
                0.25 * task.difficulty
            )

        elif task.task_type == TaskType.SOCIAL:
            adapted_scores["intentional_coherence"] = self._boost(
                adapted_scores["intentional_coherence"],
                0.2 * task.difficulty
            )
            adapted_scores["qualia_richness"] = self._boost(
                adapted_scores["qualia_richness"],
                0.15 * task.difficulty
            )

        elif task.task_type == TaskType.CREATIVE:
            adapted_scores["qualia_richness"] = self._boost(
                adapted_scores["qualia_richness"],
                0.25 * task.difficulty
            )
            adapted_scores["information_integration"] = self._boost(
                adapted_scores["information_integration"],
                0.2 * task.difficulty
            )

        elif task.task_type == TaskType.SURVIVAL:
            adapted_scores["perception_fidelity"] = self._boost(
                adapted_scores["perception_fidelity"],
                0.3 * task.difficulty
            )
            adapted_scores["reaction_speed"] = self._boost(
                adapted_scores["reaction_speed"],
                0.3 * task.difficulty
            )

        # Importance scaling
        if task.importance > 0.8:
            # High importance tasks get extra boost
            importance_boost = (task.importance - 0.5) * 0.1
            for capability in adapted_scores:
                adapted_scores[capability] = self._boost(
                    adapted_scores[capability],
                    importance_boost
                )

        # Apply smooth transition
        adapted_scores = self._smooth_transition(
            self.current.get_capability_scores(),
            adapted_scores
        )

        # Update current consciousness
        self.current = GradedConsciousness(**adapted_scores)

        # Record adaptation
        self.adaptation_history.append(AdaptationRecord(
            timestamp=time.time(),
            environment=EnvironmentContext(),
            task=task,
            original_scores=original_scores,
            adapted_scores=adapted_scores,
            adaptation_reason=f"Task: {task.task_type.value}"
        ))

        return self.current

    def adapt_combined(
        self,
        environment: EnvironmentContext,
        task: TaskContext
    ) -> GradedConsciousness:
        """
        Adapt to both environment and task simultaneously.

        Args:
            environment: Current environment context
            task: Current task context

        Returns:
            Adapted consciousness state
        """
        # First adapt to environment
        env_adapted = self.adapt_to_environment(environment)

        # Then adapt to task (builds on environment adaptation)
        task_adapted = self.adapt_to_task(task)

        return task_adapted

    def reset_to_base(self):
        """Reset to base consciousness state"""
        self.current = GradedConsciousness(
            **self.base.get_capability_scores()
        )

    def _boost(self, value: float, amount: float) -> float:
        """Boost a capability score"""
        boosted = value + amount * self.adaptation_rate
        return min(1.0, boosted)

    def _reduce(self, value: float, amount: float) -> float:
        """Reduce a capability score"""
        reduced = value - amount * self.adaptation_rate
        return max(self.min_capability, reduced)

    def _smooth_transition(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply smooth transition between states"""
        smoothed = {}
        for capability in current:
            # Exponential moving average
            smoothed[capability] = (
                current[capability] * (1 - self.adaptation_rate) +
                target[capability] * self.adaptation_rate
            )
            # Clip to valid range
            smoothed[capability] = np.clip(
                smoothed[capability],
                self.min_capability,
                1.0
            )
        return smoothed

    def get_adaptation_trajectory(self) -> List[float]:
        """Get history of overall consciousness scores"""
        return [
            GradedConsciousness(**record.adapted_scores).overall_consciousness_score()
            for record in self.adaptation_history
        ]

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation history"""
        if not self.adaptation_history:
            return {"status": "No adaptations recorded"}

        trajectory = self.get_adaptation_trajectory()

        return {
            "total_adaptations": len(self.adaptation_history),
            "current_score": self.current.overall_consciousness_score(),
            "base_score": self.base.overall_consciousness_score(),
            "min_score": min(trajectory) if trajectory else 0,
            "max_score": max(trajectory) if trajectory else 0,
            "avg_score": np.mean(trajectory) if trajectory else 0,
            "current_state": self.current.describe_state(),
            "recent_reasons": [
                r.adaptation_reason
                for r in self.adaptation_history[-5:]
            ]
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Adaptive Consciousness Demo")
    print("=" * 50)

    # Create base consciousness
    base = GradedConsciousness(
        perception_fidelity=0.6,
        reaction_speed=0.5,
        memory_depth=0.7,
        memory_recall_accuracy=0.6,
        introspection_capacity=0.7,
        meta_cognitive_ability=0.6,
        information_integration=0.6,
        intentional_coherence=0.6,
        qualia_richness=0.5
    )

    adaptive = AdaptiveConsciousness(base)
    print(f"\nBase consciousness score: {base.overall_consciousness_score():.3f}")

    # Simulate different environments
    print("\n1. Adapting to threatening environment...")
    threat_env = EnvironmentContext(threat_level=0.9, time_pressure=0.8)
    adapted = adaptive.adapt_to_environment(threat_env)
    print(f"   Perception: {base.perception_fidelity:.2f} -> {adapted.perception_fidelity:.2f}")
    print(f"   Reaction speed: {base.reaction_speed:.2f} -> {adapted.reaction_speed:.2f}")
    print(f"   Introspection: {base.introspection_capacity:.2f} -> {adapted.introspection_capacity:.2f}")

    # Reset and try different environment
    adaptive.reset_to_base()

    print("\n2. Adapting to resource-rich social environment...")
    social_env = EnvironmentContext(
        resource_availability=0.9,
        social_density=0.8,
        threat_level=0.1
    )
    adapted = adaptive.adapt_to_environment(social_env)
    print(f"   Intentional coherence: {base.intentional_coherence:.2f} -> {adapted.intentional_coherence:.2f}")
    print(f"   Qualia richness: {base.qualia_richness:.2f} -> {adapted.qualia_richness:.2f}")
    print(f"   Meta-cognitive: {base.meta_cognitive_ability:.2f} -> {adapted.meta_cognitive_ability:.2f}")

    # Adapt to task
    print("\n3. Adapting to reasoning task...")
    reasoning_task = TaskContext(
        task_type=TaskType.REASONING,
        difficulty=0.8,
        importance=0.9
    )
    adapted = adaptive.adapt_to_task(reasoning_task)
    print(f"   Info integration: {adapted.information_integration:.2f}")
    print(f"   Meta-cognitive: {adapted.meta_cognitive_ability:.2f}")

    # Summary
    print("\n4. Adaptation Summary:")
    summary = adaptive.get_adaptation_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
