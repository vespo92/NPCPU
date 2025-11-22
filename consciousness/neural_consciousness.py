"""
Neural Consciousness Implementation for NPCPU

A neural network-based consciousness model using attention mechanisms.
This module provides a sophisticated consciousness subsystem that can be
attached to any organism extending BaseOrganism.

Features:
- Attention-based perception weighting
- Working memory with capacity limits
- Emotional valence computation
- Self-model for introspection

Example:
    from core.abstractions import BaseOrganism
    from consciousness.neural_consciousness import NeuralConsciousness

    # Create consciousness and attach to organism
    consciousness = NeuralConsciousness(
        attention_dim=64,
        memory_capacity=7,
        emotional_sensitivity=0.5
    )
    organism.add_subsystem(consciousness)

    # Process perception through consciousness
    stimuli = {"visual": [0.1, 0.5, 0.3], "threat": 0.8, "food": 0.2}
    attended = consciousness.process_perception(stimuli)

    # Update working memory
    consciousness.update_working_memory({"event": "predator_spotted", "urgency": 0.9})

    # Get emotional state
    emotional_state = consciousness.compute_emotional_state()
    print(f"Valence: {emotional_state['valence']}, Arousal: {emotional_state['arousal']}")

    # Introspect
    self_report = consciousness.introspect()
    print(f"Consciousness level: {self_report['consciousness_level']}")
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import math
import uuid

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from core.abstractions import BaseSubsystem
from core.events import get_event_bus


# =============================================================================
# Data Structures
# =============================================================================

class EmotionalDimension(Enum):
    """Core emotional dimensions based on circumplex model"""
    VALENCE = auto()      # Positive/negative feeling
    AROUSAL = auto()      # Activation level
    DOMINANCE = auto()    # Control/power feeling


@dataclass
class MemoryItem:
    """A single item stored in working memory"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Get age of memory in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()

    @property
    def effective_strength(self) -> float:
        """Calculate current memory strength considering decay"""
        base_strength = self.importance * (1 + 0.1 * self.access_count)
        decay_factor = math.exp(-self.decay_rate * self.age_seconds / 60)
        return base_strength * decay_factor


@dataclass
class AttentionState:
    """Current state of the attention system"""
    focus_target: Optional[str] = None
    focus_strength: float = 0.0
    peripheral_items: List[str] = field(default_factory=list)
    attention_capacity: float = 1.0
    fatigue_level: float = 0.0


@dataclass
class EmotionalState:
    """Current emotional state of the consciousness"""
    valence: float = 0.0      # -1 (negative) to 1 (positive)
    arousal: float = 0.5      # 0 (calm) to 1 (excited)
    dominance: float = 0.5    # 0 (submissive) to 1 (dominant)
    mood_baseline: float = 0.0  # Long-term mood tendency

    def to_dict(self) -> Dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "mood_baseline": self.mood_baseline
        }


# =============================================================================
# Neural Components
# =============================================================================

class AttentionMechanism:
    """
    Scaled dot-product attention for perception weighting.

    Implements a simplified version of transformer-style attention
    that weights incoming stimuli based on relevance and salience.
    """

    def __init__(self, dim: int = 64, num_heads: int = 4):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        if HAS_NUMPY:
            # Initialize query, key, value projections
            self.W_q = np.random.randn(dim, dim) * 0.1
            self.W_k = np.random.randn(dim, dim) * 0.1
            self.W_v = np.random.randn(dim, dim) * 0.1
            self.W_o = np.random.randn(dim, dim) * 0.1
        else:
            self.W_q = self._random_matrix(dim, dim)
            self.W_k = self._random_matrix(dim, dim)
            self.W_v = self._random_matrix(dim, dim)
            self.W_o = self._random_matrix(dim, dim)

    def _random_matrix(self, rows: int, cols: int) -> List[List[float]]:
        """Create random matrix without numpy"""
        import random
        return [[random.gauss(0, 0.1) for _ in range(cols)] for _ in range(rows)]

    def _matmul(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication without numpy"""
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        result = [[0.0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax without numpy"""
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]

    def compute_attention(
        self,
        query: Any,
        keys: Any,
        values: Any
    ) -> Tuple[Any, List[float]]:
        """
        Compute attention-weighted values.

        Args:
            query: Query vector (what we're looking for)
            keys: Key vectors (what we have)
            values: Value vectors (content to retrieve)

        Returns:
            Tuple of (attended_values, attention_weights)
        """
        if HAS_NUMPY:
            query = np.array(query).reshape(1, -1)
            keys = np.array(keys)
            values = np.array(values)

            # Project to query/key/value spaces
            Q = query @ self.W_q
            K = keys @ self.W_k
            V = values @ self.W_v

            # Scaled dot-product attention
            scale = np.sqrt(self.dim)
            scores = (Q @ K.T) / scale
            weights = np.exp(scores - np.max(scores))
            weights = weights / weights.sum()

            # Weighted sum of values
            attended = weights @ V
            output = attended @ self.W_o

            return output.flatten().tolist(), weights.flatten().tolist()
        else:
            # Simplified attention without numpy
            if not keys:
                return query, [1.0]

            # Compute dot products as attention scores
            scores = []
            for key in keys:
                score = sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
                scores.append(score)

            # Softmax
            weights = self._softmax(scores)

            # Weighted sum
            result = [0.0] * len(values[0])
            for w, v in zip(weights, values):
                for i in range(len(v)):
                    result[i] += w * v[i]

            return result, weights


class SelfModel:
    """
    Internal model of self for introspection and metacognition.

    Tracks:
    - Current cognitive state
    - Recent performance
    - Internal goals and drives
    - Predicted future states
    """

    def __init__(self):
        self.cognitive_load: float = 0.0
        self.confidence: float = 0.5
        self.goals: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        self.predicted_state: Dict[str, float] = {}
        self.identity_features: Dict[str, float] = {}

    def update(
        self,
        cognitive_load: float,
        performance: Optional[float] = None,
        goals: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Update self-model with new information"""
        self.cognitive_load = cognitive_load

        if performance is not None:
            self.performance_history.append(performance)
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)

            # Update confidence based on recent performance
            recent = self.performance_history[-10:]
            if recent:
                self.confidence = sum(recent) / len(recent)

        if goals is not None:
            self.goals = goals

    def predict_next_state(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """Predict next internal state based on current trends"""
        predicted = {}
        for key, value in current_state.items():
            # Simple momentum-based prediction
            if key in self.predicted_state:
                delta = value - self.predicted_state.get(key, value)
                predicted[key] = value + delta * 0.5
            else:
                predicted[key] = value

        self.predicted_state = predicted
        return predicted

    def get_report(self) -> Dict[str, Any]:
        """Generate self-assessment report"""
        avg_performance = (
            sum(self.performance_history) / len(self.performance_history)
            if self.performance_history else 0.5
        )

        return {
            "cognitive_load": self.cognitive_load,
            "confidence": self.confidence,
            "average_performance": avg_performance,
            "active_goals": len(self.goals),
            "goals": self.goals[:5],  # Top 5 goals
            "identity": self.identity_features,
            "predicted_state": self.predicted_state
        }


# =============================================================================
# Main Consciousness Class
# =============================================================================

class NeuralConsciousness(BaseSubsystem):
    """
    Neural network-based consciousness using attention mechanisms.

    This consciousness model implements a sophisticated awareness system
    that processes perceptions, maintains working memory, computes
    emotional states, and provides introspection capabilities.

    Features:
    - Attention-based perception weighting
    - Working memory with capacity limits (Miller's 7±2)
    - Emotional valence computation using circumplex model
    - Self-model for introspection and metacognition

    Args:
        attention_dim: Dimension of attention vectors (default: 64)
        memory_capacity: Maximum working memory items (default: 7)
        emotional_sensitivity: How strongly emotions respond to stimuli (default: 0.5)
        consciousness_threshold: Minimum activation for awareness (default: 0.3)

    Example:
        consciousness = NeuralConsciousness(
            attention_dim=64,
            memory_capacity=7,
            emotional_sensitivity=0.5
        )

        # Attach to organism
        organism.add_subsystem(consciousness)

        # Process stimuli
        stimuli = {"visual": [0.5, 0.3], "auditory": [0.2], "threat": 0.8}
        attended = consciousness.process_perception(stimuli)

        # Check emotional state
        emotion = consciousness.compute_emotional_state()
        print(f"Feeling: {'positive' if emotion['valence'] > 0 else 'negative'}")
    """

    def __init__(
        self,
        attention_dim: int = 64,
        memory_capacity: int = 7,
        emotional_sensitivity: float = 0.5,
        consciousness_threshold: float = 0.3,
        name: str = "neural_consciousness"
    ):
        super().__init__(name)

        # Configuration
        self.attention_dim = attention_dim
        self.memory_capacity = memory_capacity
        self.emotional_sensitivity = emotional_sensitivity
        self.consciousness_threshold = consciousness_threshold

        # Core components
        self.attention = AttentionMechanism(dim=attention_dim)
        self.attention_state = AttentionState()
        self.self_model = SelfModel()

        # Working memory
        self.working_memory: List[MemoryItem] = []

        # Emotional system
        self.emotional_state = EmotionalState()
        self.emotion_history: List[EmotionalState] = []

        # Perception processing
        self.perception_buffer: Dict[str, Any] = {}
        self.salience_weights: Dict[str, float] = {
            "threat": 1.0,
            "food": 0.8,
            "social": 0.6,
            "novelty": 0.5,
            "default": 0.3
        }

        # Consciousness metrics
        self.consciousness_level: float = 0.5
        self.global_workspace: Dict[str, Any] = {}
        self.tick_count: int = 0

    # -------------------------------------------------------------------------
    # Perception Processing
    # -------------------------------------------------------------------------

    def process_perception(self, stimuli: Dict[str, Any]) -> Dict[str, Any]:
        """
        Weight and filter incoming stimuli through attention mechanism.

        Implements a global workspace theory approach where only the most
        salient and attended stimuli reach conscious awareness.

        Args:
            stimuli: Dictionary of sensory inputs
                     e.g., {"visual": [...], "threat": 0.8, "food": 0.2}

        Returns:
            Dictionary of attended (consciously processed) stimuli
        """
        self.perception_buffer = stimuli.copy()
        attended_stimuli = {}

        # Calculate salience for each stimulus
        salience_scores = {}
        for key, value in stimuli.items():
            base_salience = self.salience_weights.get(key, self.salience_weights["default"])

            # Modulate by emotional state (threats more salient when aroused)
            emotional_mod = 1.0
            if key == "threat":
                emotional_mod = 1.0 + self.emotional_state.arousal * 0.5
            elif key == "food":
                # Food more salient when in negative valence (hungry/stressed)
                emotional_mod = 1.0 - self.emotional_state.valence * 0.3

            # Calculate final salience
            if isinstance(value, (int, float)):
                salience = base_salience * float(value) * emotional_mod
            elif isinstance(value, (list, tuple)):
                salience = base_salience * (sum(abs(v) for v in value) / len(value)) * emotional_mod
            else:
                salience = base_salience * emotional_mod

            salience_scores[key] = salience

        # Apply attention - only process stimuli above threshold
        total_salience = sum(salience_scores.values()) or 1.0
        attention_capacity_used = 0.0

        # Sort by salience and process most salient first
        sorted_stimuli = sorted(
            salience_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for key, salience in sorted_stimuli:
            normalized_salience = salience / total_salience

            if normalized_salience >= self.consciousness_threshold:
                if attention_capacity_used + normalized_salience <= self.attention_state.attention_capacity:
                    attended_stimuli[key] = {
                        "value": stimuli[key],
                        "salience": salience,
                        "attention_weight": normalized_salience
                    }
                    attention_capacity_used += normalized_salience

                    # Update attention focus
                    if not self.attention_state.focus_target or salience > self.attention_state.focus_strength:
                        self.attention_state.focus_target = key
                        self.attention_state.focus_strength = salience

        # Store in global workspace
        self.global_workspace["attended_stimuli"] = attended_stimuli
        self.global_workspace["total_salience"] = total_salience

        # Emit perception event
        bus = get_event_bus()
        bus.emit("consciousness.perception", {
            "organism_id": self.owner.id if self.owner else None,
            "attended_count": len(attended_stimuli),
            "focus_target": self.attention_state.focus_target,
            "consciousness_level": self.consciousness_level
        }, source=self.name)

        return attended_stimuli

    # -------------------------------------------------------------------------
    # Working Memory
    # -------------------------------------------------------------------------

    def update_working_memory(
        self,
        info: Dict[str, Any],
        importance: float = 0.5,
        associations: Optional[List[str]] = None
    ) -> bool:
        """
        Manage limited-capacity working memory.

        Implements Miller's law (7±2 items) with importance-based
        replacement and decay over time.

        Args:
            info: Information to store
            importance: Priority of this memory (0-1)
            associations: IDs of related memories

        Returns:
            True if stored successfully, False if rejected
        """
        # Create memory item
        memory = MemoryItem(
            content=info,
            importance=importance,
            associations=associations or []
        )

        # Check if similar memory exists (consolidate)
        for existing in self.working_memory:
            if self._memories_similar(existing.content, info):
                existing.access_count += 1
                existing.importance = max(existing.importance, importance)
                existing.timestamp = datetime.now()
                return True

        # Check capacity
        if len(self.working_memory) >= self.memory_capacity:
            # Find weakest memory to replace
            weakest = min(self.working_memory, key=lambda m: m.effective_strength)

            if memory.importance > weakest.effective_strength:
                self.working_memory.remove(weakest)
                self._emit_memory_displaced(weakest)
            else:
                # New memory not important enough
                return False

        self.working_memory.append(memory)

        # Update self-model cognitive load
        load = len(self.working_memory) / self.memory_capacity
        self.self_model.update(cognitive_load=load)

        # Emit event
        bus = get_event_bus()
        bus.emit("consciousness.memory_stored", {
            "organism_id": self.owner.id if self.owner else None,
            "memory_id": memory.id,
            "importance": importance,
            "working_memory_usage": len(self.working_memory)
        }, source=self.name)

        return True

    def recall(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from working memory.

        Args:
            query: Optional search criteria
            limit: Maximum memories to return

        Returns:
            List of memory contents ordered by relevance
        """
        if not self.working_memory:
            return []

        if query is None:
            # Return most important memories
            sorted_memories = sorted(
                self.working_memory,
                key=lambda m: m.effective_strength,
                reverse=True
            )
        else:
            # Score memories by query relevance
            scored = []
            for memory in self.working_memory:
                score = self._compute_memory_relevance(memory.content, query)
                scored.append((memory, score))

            sorted_memories = [m for m, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

        # Update access counts
        results = []
        for memory in sorted_memories[:limit]:
            memory.access_count += 1
            results.append({
                "id": memory.id,
                "content": memory.content,
                "importance": memory.importance,
                "strength": memory.effective_strength,
                "age_seconds": memory.age_seconds
            })

        return results

    def _memories_similar(self, m1: Dict[str, Any], m2: Dict[str, Any]) -> bool:
        """Check if two memories are similar enough to consolidate"""
        # Simple key overlap check
        keys1, keys2 = set(m1.keys()), set(m2.keys())
        overlap = len(keys1 & keys2) / max(len(keys1 | keys2), 1)
        return overlap > 0.7

    def _compute_memory_relevance(self, memory: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Compute relevance score between memory and query"""
        score = 0.0
        for key, value in query.items():
            if key in memory:
                if memory[key] == value:
                    score += 1.0
                elif isinstance(value, (int, float)) and isinstance(memory[key], (int, float)):
                    score += 1.0 - abs(float(value) - float(memory[key]))
        return score / max(len(query), 1)

    def _emit_memory_displaced(self, memory: MemoryItem) -> None:
        """Emit event when memory is pushed out of working memory"""
        bus = get_event_bus()
        bus.emit("consciousness.memory_displaced", {
            "organism_id": self.owner.id if self.owner else None,
            "memory_id": memory.id,
            "final_strength": memory.effective_strength
        }, source=self.name)

    def consolidate_memory(self) -> None:
        """
        Consolidate working memory - strengthen important memories,
        let weak ones decay further.
        """
        for memory in self.working_memory:
            # Strengthen frequently accessed memories
            if memory.access_count > 3:
                memory.decay_rate *= 0.9
            # Increase decay for unused memories
            elif memory.access_count == 0 and memory.age_seconds > 60:
                memory.decay_rate *= 1.1

        # Remove very weak memories
        self.working_memory = [
            m for m in self.working_memory
            if m.effective_strength > 0.1
        ]

    # -------------------------------------------------------------------------
    # Emotional Processing
    # -------------------------------------------------------------------------

    def compute_emotional_state(self) -> Dict[str, float]:
        """
        Derive emotional valence from current state.

        Uses a circumplex model with valence (positive/negative),
        arousal (calm/excited), and dominance (submissive/dominant).

        Returns:
            Dictionary with emotional dimensions
        """
        # Start from mood baseline
        new_valence = self.emotional_state.mood_baseline
        new_arousal = 0.5
        new_dominance = 0.5

        # Process attended stimuli
        attended = self.global_workspace.get("attended_stimuli", {})

        for key, data in attended.items():
            value = data.get("value", 0)
            weight = data.get("attention_weight", 0.5)

            if isinstance(value, (list, tuple)):
                value = sum(value) / len(value) if value else 0

            value = float(value) if isinstance(value, (int, float)) else 0

            # Threat processing
            if key == "threat":
                new_valence -= value * self.emotional_sensitivity * weight
                new_arousal += value * self.emotional_sensitivity * weight
                new_dominance -= value * 0.5 * weight

            # Food/reward processing
            elif key == "food":
                new_valence += value * self.emotional_sensitivity * weight
                new_arousal += value * 0.3 * weight

            # Social processing
            elif key == "social":
                new_valence += value * self.emotional_sensitivity * 0.5 * weight
                new_dominance += (value - 0.5) * weight

            # Novelty processing
            elif key == "novelty":
                new_arousal += value * self.emotional_sensitivity * weight

        # Factor in cognitive load
        load = len(self.working_memory) / self.memory_capacity
        if load > 0.8:
            new_valence -= 0.1 * self.emotional_sensitivity
            new_arousal += 0.1

        # Smooth transition (emotional inertia)
        inertia = 0.7
        self.emotional_state.valence = (
            self.emotional_state.valence * inertia +
            max(-1, min(1, new_valence)) * (1 - inertia)
        )
        self.emotional_state.arousal = (
            self.emotional_state.arousal * inertia +
            max(0, min(1, new_arousal)) * (1 - inertia)
        )
        self.emotional_state.dominance = (
            self.emotional_state.dominance * inertia +
            max(0, min(1, new_dominance)) * (1 - inertia)
        )

        # Update mood baseline slowly
        self.emotional_state.mood_baseline = (
            self.emotional_state.mood_baseline * 0.99 +
            self.emotional_state.valence * 0.01
        )

        # Store in history
        self.emotion_history.append(EmotionalState(
            valence=self.emotional_state.valence,
            arousal=self.emotional_state.arousal,
            dominance=self.emotional_state.dominance,
            mood_baseline=self.emotional_state.mood_baseline
        ))
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)

        # Emit event
        bus = get_event_bus()
        bus.emit("consciousness.emotion_updated", {
            "organism_id": self.owner.id if self.owner else None,
            "valence": self.emotional_state.valence,
            "arousal": self.emotional_state.arousal,
            "dominance": self.emotional_state.dominance
        }, source=self.name)

        return self.emotional_state.to_dict()

    def get_dominant_emotion(self) -> str:
        """Get the name of the current dominant emotion"""
        v = self.emotional_state.valence
        a = self.emotional_state.arousal

        # Map to discrete emotions based on circumplex
        if v > 0.3:
            if a > 0.6:
                return "excited"
            elif a > 0.3:
                return "happy"
            else:
                return "content"
        elif v < -0.3:
            if a > 0.6:
                return "afraid"
            elif a > 0.3:
                return "sad"
            else:
                return "depressed"
        else:
            if a > 0.6:
                return "alert"
            elif a > 0.3:
                return "neutral"
            else:
                return "calm"

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def introspect(self) -> Dict[str, Any]:
        """
        Generate self-model report through metacognition.

        Returns comprehensive awareness of internal state including
        attention focus, memory load, emotional state, and predictions.

        Returns:
            Dictionary containing self-assessment
        """
        # Update consciousness level
        self._update_consciousness_level()

        # Get self-model report
        self_report = self.self_model.get_report()

        # Compile introspection report
        report = {
            "consciousness_level": self.consciousness_level,
            "attention": {
                "focus_target": self.attention_state.focus_target,
                "focus_strength": self.attention_state.focus_strength,
                "attention_fatigue": self.attention_state.fatigue_level
            },
            "working_memory": {
                "items_count": len(self.working_memory),
                "capacity": self.memory_capacity,
                "load_percentage": len(self.working_memory) / self.memory_capacity * 100,
                "recent_items": [m.content for m in self.working_memory[-3:]]
            },
            "emotional_state": {
                "current": self.emotional_state.to_dict(),
                "dominant_emotion": self.get_dominant_emotion(),
                "emotional_stability": self._compute_emotional_stability()
            },
            "self_model": self_report,
            "global_workspace": {
                "active_contents": list(self.global_workspace.keys()),
                "attended_stimuli_count": len(self.global_workspace.get("attended_stimuli", {}))
            },
            "tick_count": self.tick_count
        }

        # Emit introspection event
        bus = get_event_bus()
        bus.emit("consciousness.introspection", {
            "organism_id": self.owner.id if self.owner else None,
            "consciousness_level": self.consciousness_level,
            "dominant_emotion": report["emotional_state"]["dominant_emotion"]
        }, source=self.name)

        return report

    def _update_consciousness_level(self) -> None:
        """Update overall consciousness level based on various factors"""
        # Base level from attention focus
        attention_component = self.attention_state.focus_strength * 0.3

        # Memory engagement
        memory_component = (len(self.working_memory) / self.memory_capacity) * 0.2

        # Emotional arousal contribution
        arousal_component = self.emotional_state.arousal * 0.2

        # Self-model confidence
        confidence_component = self.self_model.confidence * 0.2

        # Integration (how much is in global workspace)
        integration_component = min(
            len(self.global_workspace.get("attended_stimuli", {})) / 5, 1.0
        ) * 0.1

        # Compute final level with smoothing
        new_level = (
            attention_component +
            memory_component +
            arousal_component +
            confidence_component +
            integration_component
        )

        # Apply fatigue penalty
        new_level *= (1.0 - self.attention_state.fatigue_level * 0.3)

        # Smooth transition
        self.consciousness_level = (
            self.consciousness_level * 0.8 +
            new_level * 0.2
        )

    def _compute_emotional_stability(self) -> float:
        """Compute emotional stability from recent emotion history"""
        if len(self.emotion_history) < 2:
            return 1.0

        # Compute variance in recent emotions
        recent = self.emotion_history[-20:]
        valences = [e.valence for e in recent]

        if len(valences) < 2:
            return 1.0

        mean = sum(valences) / len(valences)
        variance = sum((v - mean) ** 2 for v in valences) / len(valences)

        # Convert to stability score (lower variance = higher stability)
        return max(0, 1.0 - variance * 2)

    # -------------------------------------------------------------------------
    # BaseSubsystem Interface
    # -------------------------------------------------------------------------

    def tick(self) -> None:
        """Process one time step of consciousness"""
        if not self.enabled:
            return

        self.tick_count += 1

        # Consolidate memory periodically
        if self.tick_count % 10 == 0:
            self.consolidate_memory()

        # Update attention fatigue
        if self.attention_state.focus_strength > 0.7:
            self.attention_state.fatigue_level = min(
                1.0,
                self.attention_state.fatigue_level + 0.01
            )
        else:
            self.attention_state.fatigue_level = max(
                0.0,
                self.attention_state.fatigue_level - 0.02
            )

        # Natural emotional decay towards baseline
        decay = 0.01
        self.emotional_state.valence += (
            self.emotional_state.mood_baseline - self.emotional_state.valence
        ) * decay
        self.emotional_state.arousal += (0.5 - self.emotional_state.arousal) * decay

        # Update consciousness level
        self._update_consciousness_level()

        # Emit tick event
        bus = get_event_bus()
        bus.emit("consciousness.tick", {
            "organism_id": self.owner.id if self.owner else None,
            "tick": self.tick_count,
            "consciousness_level": self.consciousness_level
        }, source=self.name)

    def get_state(self) -> Dict[str, Any]:
        """Get current subsystem state for serialization"""
        return {
            "attention_dim": self.attention_dim,
            "memory_capacity": self.memory_capacity,
            "emotional_sensitivity": self.emotional_sensitivity,
            "consciousness_threshold": self.consciousness_threshold,
            "consciousness_level": self.consciousness_level,
            "tick_count": self.tick_count,
            "emotional_state": self.emotional_state.to_dict(),
            "attention_state": {
                "focus_target": self.attention_state.focus_target,
                "focus_strength": self.attention_state.focus_strength,
                "fatigue_level": self.attention_state.fatigue_level
            },
            "working_memory": [
                {
                    "id": m.id,
                    "content": m.content,
                    "importance": m.importance,
                    "access_count": m.access_count
                }
                for m in self.working_memory
            ],
            "self_model": self.self_model.get_report()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore subsystem state from serialized data"""
        self.consciousness_level = state.get("consciousness_level", 0.5)
        self.tick_count = state.get("tick_count", 0)

        # Restore emotional state
        es = state.get("emotional_state", {})
        self.emotional_state.valence = es.get("valence", 0.0)
        self.emotional_state.arousal = es.get("arousal", 0.5)
        self.emotional_state.dominance = es.get("dominance", 0.5)
        self.emotional_state.mood_baseline = es.get("mood_baseline", 0.0)

        # Restore attention state
        att = state.get("attention_state", {})
        self.attention_state.focus_target = att.get("focus_target")
        self.attention_state.focus_strength = att.get("focus_strength", 0.0)
        self.attention_state.fatigue_level = att.get("fatigue_level", 0.0)

        # Restore working memory
        self.working_memory = []
        for m in state.get("working_memory", []):
            memory = MemoryItem(
                id=m.get("id", str(uuid.uuid4())),
                content=m.get("content", {}),
                importance=m.get("importance", 0.5),
                access_count=m.get("access_count", 0)
            )
            self.working_memory.append(memory)

    def reset(self) -> None:
        """Reset consciousness to initial state"""
        super().reset()
        self.working_memory.clear()
        self.emotional_state = EmotionalState()
        self.emotion_history.clear()
        self.attention_state = AttentionState()
        self.self_model = SelfModel()
        self.global_workspace.clear()
        self.consciousness_level = 0.5
        self.tick_count = 0


# =============================================================================
# Integration with SimpleOrganism
# =============================================================================

def attach_neural_consciousness(
    organism,
    attention_dim: int = 64,
    memory_capacity: int = 7,
    emotional_sensitivity: float = 0.5
) -> NeuralConsciousness:
    """
    Convenience function to attach neural consciousness to an organism.

    Args:
        organism: Any organism extending BaseOrganism
        attention_dim: Dimension of attention vectors
        memory_capacity: Working memory capacity
        emotional_sensitivity: Emotional response strength

    Returns:
        The attached NeuralConsciousness instance

    Example:
        from core.simple_organism import SimpleOrganism
        from consciousness.neural_consciousness import attach_neural_consciousness

        org = SimpleOrganism("ConsciousOrganism")
        consciousness = attach_neural_consciousness(org)

        # The organism now has consciousness
        org.perceive({"threat": 0.8})
        consciousness.process_perception({"threat": 0.8})
        print(consciousness.introspect())
    """
    consciousness = NeuralConsciousness(
        attention_dim=attention_dim,
        memory_capacity=memory_capacity,
        emotional_sensitivity=emotional_sensitivity
    )
    organism.add_subsystem(consciousness)
    return consciousness
