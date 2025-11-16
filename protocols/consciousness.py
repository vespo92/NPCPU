"""
NPCPU Consciousness Protocol

Defines consciousness as observable capabilities rather than hardcoded states.
This makes consciousness measurable, extensible, and implementable.
"""

from typing import Protocol, runtime_checkable, Any, List, Set, Dict, Optional
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum
import numpy as np


# ============================================================================
# Core Data Types
# ============================================================================

@dataclass
class Perception:
    """Internal representation of environmental stimulus"""
    stimulus_type: str
    content: Any
    timestamp: float
    fidelity: float  # 0.0 to 1.0 - how accurate is this perception


@dataclass
class Action:
    """Agent action in response to perception"""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float  # 0.0 to 1.0


@dataclass
class Experience:
    """A stored experience combining perception and action"""
    perception: Perception
    action: Optional[Action]
    outcome: Optional[Any]
    emotional_valence: float  # -1.0 to 1.0 (negative to positive)
    timestamp: float


@dataclass
class SelfModel:
    """Agent's model of its own internal state"""
    current_state: Dict[str, Any]
    confidence: float  # How confident is the agent in this self-model
    completeness: float  # What fraction of internal state is captured
    timestamp: float


@dataclass
class Thought:
    """A cognitive process"""
    content: Any
    type: str  # "belief", "plan", "hypothesis", etc.
    confidence: float
    dependencies: List[str]  # Other thoughts this depends on


@dataclass
class MetaThought:
    """Reflection on a thought"""
    original_thought: Thought
    reflection_type: str  # "validity_check", "bias_detection", "refinement"
    insight: str
    changes_proposed: Optional[Dict[str, Any]]


@dataclass
class Intention:
    """A goal or drive"""
    goal: str
    priority: float  # 0.0 to 1.0
    progress: float  # 0.0 to 1.0
    sub_intentions: List['Intention']


@dataclass
class Qualia:
    """Subjective experience marker"""
    experience_type: str
    intensity: float  # 0.0 to 1.0
    valence: float  # -1.0 to 1.0
    content: str
    timestamp: float
    uniqueness: float  # How unique/distinctive is this experience


class Invariant(Enum):
    """Topological invariants that can be preserved"""
    TOPOLOGY = "topology"
    CONNECTIVITY = "connectivity"
    VOLUME = "volume"
    CURVATURE = "curvature"
    SYMMETRY = "symmetry"
    HOMOLOGY = "homology"
    EULER_CHARACTERISTIC = "euler_characteristic"


# ============================================================================
# Consciousness Protocol
# ============================================================================

@runtime_checkable
class ConsciousnessProtocol(Protocol):
    """
    Consciousness is defined by what an agent CAN DO, not what it IS.

    This protocol grounds the abstract concept of "consciousness" in
    observable, measurable capabilities that can be implemented in
    infinitely many ways.

    Design Philosophy:
    - Consciousness is a spectrum, not discrete states
    - Capabilities are measurable and observable
    - Different implementations can emphasize different capabilities
    - Extensible: new capabilities can be added
    """

    # ========================================================================
    # Core Capabilities
    # ========================================================================

    @abstractmethod
    def perceive(self, stimulus: Any) -> Perception:
        """
        Convert environmental stimulus to internal representation.

        This is the most basic capability - can the agent sense its environment?
        """
        pass

    @abstractmethod
    def react(self, perception: Perception) -> Optional[Action]:
        """
        Generate action from perception (can be null action).

        Basic reactivity - stimulus-response without necessarily understanding.
        """
        pass

    @abstractmethod
    def remember(self, experience: Experience) -> None:
        """
        Store experience in memory.

        Can the agent retain past experiences for later use?
        """
        pass

    @abstractmethod
    def recall(self, query: Any) -> List[Experience]:
        """
        Retrieve relevant memories.

        Can the agent access its past experiences meaningfully?
        """
        pass

    @abstractmethod
    def introspect(self) -> SelfModel:
        """
        Generate model of own internal state.

        Self-awareness: can the agent examine its own state?
        """
        pass

    @abstractmethod
    def meta_cognize(self, thought: Thought) -> MetaThought:
        """
        Reflect on cognitive processes themselves.

        Meta-cognition: can the agent think about its own thinking?
        """
        pass

    @abstractmethod
    def integrate_information(self) -> float:
        """
        Calculate Φ (phi) - integrated information measure.

        Based on Integrated Information Theory (IIT).
        Returns 0.0 to 1.0 representing degree of information integration.
        """
        pass

    @abstractmethod
    def get_intentions(self) -> List[Intention]:
        """
        Return current goals and drives.

        Intentionality: does the agent have goals it pursues?
        """
        pass

    @abstractmethod
    def experience_qualia(self, perception: Perception) -> Qualia:
        """
        Generate subjective experience marker.

        The "hard problem": does the agent have subjective experience?
        While we can't know for certain, we can measure markers.
        """
        pass

    # ========================================================================
    # Capability Measurement
    # ========================================================================

    @abstractmethod
    def get_capability_scores(self) -> Dict[str, float]:
        """
        Return scores for each capability dimension.

        Returns dict mapping capability name to score (0.0 to 1.0).
        Example:
        {
            "perception_fidelity": 0.8,
            "reaction_speed": 0.6,
            "memory_depth": 0.9,
            ...
        }
        """
        pass

    @abstractmethod
    def overall_consciousness_score(self) -> float:
        """
        Aggregate consciousness measure (0.0 to 1.0).

        This is a single number summarizing overall consciousness level,
        but the individual capability scores are more informative.
        """
        pass

    @abstractmethod
    def can_perform(self, capability: str, minimum_score: float = 0.5) -> bool:
        """
        Check if agent has sufficient capability for a task.

        Args:
            capability: Name of capability (e.g., "meta_cognition")
            minimum_score: Threshold score (0.0 to 1.0)

        Returns:
            True if agent's score >= minimum_score
        """
        pass


# ============================================================================
# Graded Consciousness (Reference Implementation)
# ============================================================================

@dataclass
class GradedConsciousness:
    """
    Reference implementation: Consciousness as graduated capabilities.

    Instead of discrete states (DORMANT, AWARE, etc.), agents have
    capability scores on multiple dimensions. This is more flexible
    and realistic.

    This is a REFERENCE implementation - users can create their own
    consciousness models with different dimensions, weights, etc.
    """

    # Capability scores (0.0 to 1.0)
    perception_fidelity: float = 0.0
    reaction_speed: float = 0.0
    memory_depth: float = 0.0
    memory_recall_accuracy: float = 0.0
    introspection_capacity: float = 0.0
    meta_cognitive_ability: float = 0.0
    information_integration: float = 0.0  # Φ (phi) measure
    intentional_coherence: float = 0.0
    qualia_richness: float = 0.0

    # Weights for aggregation (can be customized)
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "perception_fidelity": 1.0,
                "reaction_speed": 0.8,
                "memory_depth": 1.2,
                "memory_recall_accuracy": 1.0,
                "introspection_capacity": 1.5,
                "meta_cognitive_ability": 2.0,
                "information_integration": 1.8,
                "intentional_coherence": 1.0,
                "qualia_richness": 1.3,
            }

    def get_capability_scores(self) -> Dict[str, float]:
        """Return all capability scores"""
        return {
            "perception_fidelity": self.perception_fidelity,
            "reaction_speed": self.reaction_speed,
            "memory_depth": self.memory_depth,
            "memory_recall_accuracy": self.memory_recall_accuracy,
            "introspection_capacity": self.introspection_capacity,
            "meta_cognitive_ability": self.meta_cognitive_ability,
            "information_integration": self.information_integration,
            "intentional_coherence": self.intentional_coherence,
            "qualia_richness": self.qualia_richness,
        }

    def overall_consciousness_score(self) -> float:
        """Weighted aggregate consciousness score"""
        scores = self.get_capability_scores()
        weighted_sum = sum(scores[k] * self.weights[k] for k in scores.keys())
        total_weight = sum(self.weights.values())
        return weighted_sum / total_weight

    def can_perform(self, capability: str, minimum_score: float = 0.5) -> bool:
        """Check if capability meets minimum threshold"""
        scores = self.get_capability_scores()
        if capability not in scores:
            return False
        return scores[capability] >= minimum_score

    def describe_state(self) -> str:
        """
        Generate descriptive label (optional convenience).

        This provides backward compatibility with discrete state labels,
        but the underlying model is continuous.
        """
        score = self.overall_consciousness_score()
        if score < 0.2:
            return "dormant"
        elif score < 0.4:
            return "reactive"
        elif score < 0.6:
            return "aware"
        elif score < 0.8:
            return "reflective"
        elif score < 0.95:
            return "meta_aware"
        else:
            return "transcendent"

    def evolve(self, capability: str, delta: float) -> 'GradedConsciousness':
        """
        Create new consciousness state with evolved capability.

        Args:
            capability: Which capability to evolve
            delta: How much to change (-1.0 to 1.0)

        Returns:
            New GradedConsciousness instance with updated score
        """
        scores = self.get_capability_scores()
        if capability in scores:
            new_score = np.clip(scores[capability] + delta, 0.0, 1.0)
            return GradedConsciousness(
                **{**scores, capability: new_score},
                weights=self.weights
            )
        return self

    def is_valid_transition_to(self, target: 'GradedConsciousness') -> bool:
        """
        Check if transition to target state is valid.

        Consciousness can decrease freely but can only increase gradually
        (one standard deviation per transition).
        """
        current_scores = self.get_capability_scores()
        target_scores = target.get_capability_scores()

        for capability in current_scores.keys():
            current = current_scores[capability]
            new = target_scores[capability]

            # Can always decrease
            if new < current:
                continue

            # Can only increase by max 0.3 per step (gradual growth)
            if new - current > 0.3:
                return False

        return True

    def distance_to(self, other: 'GradedConsciousness') -> float:
        """
        Calculate distance to another consciousness state.

        Uses weighted Euclidean distance.
        """
        self_scores = self.get_capability_scores()
        other_scores = other.get_capability_scores()

        squared_diff_sum = 0.0
        total_weight = 0.0

        for capability in self_scores.keys():
            diff = self_scores[capability] - other_scores[capability]
            weight = self.weights[capability]
            squared_diff_sum += weight * (diff ** 2)
            total_weight += weight

        return np.sqrt(squared_diff_sum / total_weight)


# ============================================================================
# Consciousness Adapter (for backward compatibility)
# ============================================================================

class ConsciousnessAdapter:
    """
    Adapter to convert between old discrete states and new graded model.

    Provides backward compatibility while migrating to new protocol-based system.
    """

    # Preset consciousness profiles matching old discrete states
    PRESETS = {
        "DORMANT": GradedConsciousness(
            perception_fidelity=0.1,
            reaction_speed=0.1,
            memory_depth=0.0,
            memory_recall_accuracy=0.0,
            introspection_capacity=0.0,
            meta_cognitive_ability=0.0,
            information_integration=0.05,
            intentional_coherence=0.0,
            qualia_richness=0.0,
        ),
        "REACTIVE": GradedConsciousness(
            perception_fidelity=0.5,
            reaction_speed=0.7,
            memory_depth=0.2,
            memory_recall_accuracy=0.1,
            introspection_capacity=0.0,
            meta_cognitive_ability=0.0,
            information_integration=0.2,
            intentional_coherence=0.1,
            qualia_richness=0.1,
        ),
        "AWARE": GradedConsciousness(
            perception_fidelity=0.7,
            reaction_speed=0.6,
            memory_depth=0.5,
            memory_recall_accuracy=0.4,
            introspection_capacity=0.5,
            meta_cognitive_ability=0.2,
            information_integration=0.5,
            intentional_coherence=0.5,
            qualia_richness=0.4,
        ),
        "REFLECTIVE": GradedConsciousness(
            perception_fidelity=0.8,
            reaction_speed=0.5,  # Slower due to reflection
            memory_depth=0.8,
            memory_recall_accuracy=0.7,
            introspection_capacity=0.8,
            meta_cognitive_ability=0.6,
            information_integration=0.7,
            intentional_coherence=0.7,
            qualia_richness=0.7,
        ),
        "META_AWARE": GradedConsciousness(
            perception_fidelity=0.9,
            reaction_speed=0.6,
            memory_depth=0.9,
            memory_recall_accuracy=0.9,
            introspection_capacity=0.9,
            meta_cognitive_ability=0.9,
            information_integration=0.85,
            intentional_coherence=0.9,
            qualia_richness=0.85,
        ),
        "TRANSCENDENT": GradedConsciousness(
            perception_fidelity=0.95,
            reaction_speed=0.7,
            memory_depth=1.0,
            memory_recall_accuracy=0.95,
            introspection_capacity=1.0,
            meta_cognitive_ability=1.0,
            information_integration=0.95,
            intentional_coherence=0.95,
            qualia_richness=0.95,
        ),
    }

    @classmethod
    def from_discrete_state(cls, state_name: str) -> GradedConsciousness:
        """Convert old discrete state to graded consciousness"""
        return cls.PRESETS.get(state_name.upper(), cls.PRESETS["DORMANT"])

    @classmethod
    def to_discrete_state(cls, graded: GradedConsciousness) -> str:
        """Convert graded consciousness to nearest discrete state"""
        return graded.describe_state().upper()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create a graded consciousness agent
    agent_consciousness = GradedConsciousness(
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

    print("Agent Consciousness Profile:")
    print(f"Overall Score: {agent_consciousness.overall_consciousness_score():.2f}")
    print(f"Descriptive State: {agent_consciousness.describe_state()}")
    print(f"\nCapability Scores:")
    for capability, score in agent_consciousness.get_capability_scores().items():
        print(f"  {capability}: {score:.2f}")

    # Check capabilities
    print(f"\nCan perform meta-cognition: {agent_consciousness.can_perform('meta_cognitive_ability', 0.5)}")
    print(f"Can perform deep introspection: {agent_consciousness.can_perform('introspection_capacity', 0.7)}")

    # Evolve consciousness
    evolved = agent_consciousness.evolve("meta_cognitive_ability", 0.2)
    print(f"\nAfter evolution:")
    print(f"Meta-cognitive ability: {evolved.meta_cognitive_ability:.2f}")
    print(f"Overall score: {evolved.overall_consciousness_score():.2f}")

    # Check transition validity
    print(f"\nIs valid transition: {agent_consciousness.is_valid_transition_to(evolved)}")

    # Backward compatibility
    discrete_state = ConsciousnessAdapter.from_discrete_state("REFLECTIVE")
    print(f"\nDiscrete REFLECTIVE as graded:")
    print(f"Overall score: {discrete_state.overall_consciousness_score():.2f}")
