"""
DENDRITE-X: Neuroplasticity Engine

Dynamic network restructuring based on experience, implementing biological
plasticity mechanisms for adaptive consciousness.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy


# ============================================================================
# Plasticity Types
# ============================================================================

class PlasticityType(Enum):
    """Types of neural plasticity"""
    HEBBIAN = "hebbian"  # "Neurons that fire together, wire together"
    ANTI_HEBBIAN = "anti_hebbian"  # Decorrelation
    HOMEOSTATIC = "homeostatic"  # Maintains stability
    STRUCTURAL = "structural"  # Synapse formation/elimination
    METAPLASTICITY = "metaplasticity"  # Plasticity of plasticity


class PlasticityPhase(Enum):
    """Phases of plasticity expression"""
    INDUCTION = "induction"  # Initial trigger
    EXPRESSION = "expression"  # Active change
    CONSOLIDATION = "consolidation"  # Stabilization
    MAINTENANCE = "maintenance"  # Long-term preservation


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PlasticityConfig:
    """Configuration for neuroplasticity engine"""
    learning_rate: float = 0.01
    decay_rate: float = 0.001
    homeostatic_target: float = 0.5
    structural_threshold: float = 0.1
    metaplasticity_rate: float = 0.005
    consolidation_threshold: float = 0.7
    max_weight: float = 1.0
    min_weight: float = -1.0


@dataclass
class SynapticConnection:
    """Represents a synaptic connection between neurons"""
    pre_idx: int
    post_idx: int
    weight: float
    age: int = 0
    activity_count: int = 0
    last_activity: float = 0.0
    plasticity_state: float = 1.0  # Metaplasticity modifier


# ============================================================================
# Neuroplasticity Engine
# ============================================================================

class NeuroplasticityEngine:
    """
    Engine for dynamic neural network adaptation.

    Implements multiple plasticity mechanisms:
    1. Hebbian learning - strengthen co-active connections
    2. Homeostatic plasticity - maintain stable activity
    3. Structural plasticity - create/remove connections
    4. Metaplasticity - adapt plasticity rules themselves

    Example:
        engine = NeuroplasticityEngine(num_neurons=100)

        # Apply Hebbian plasticity
        pre_activity = np.random.rand(100) > 0.5
        post_activity = np.random.rand(100) > 0.5
        engine.apply_hebbian(pre_activity, post_activity)

        # Get adapted weights
        weights = engine.get_weight_matrix()
    """

    def __init__(
        self,
        num_neurons: int,
        config: Optional[PlasticityConfig] = None
    ):
        self.num_neurons = num_neurons
        self.config = config or PlasticityConfig()

        # Initialize weight matrix
        self.weights = np.random.randn(num_neurons, num_neurons) * 0.1
        np.fill_diagonal(self.weights, 0)  # No self-connections

        # Connection metadata
        self.connections: Dict[Tuple[int, int], SynapticConnection] = {}
        self._init_connections()

        # Activity history for homeostasis
        self.activity_history: List[np.ndarray] = []
        self.max_history = 100

        # Plasticity state tracking
        self.current_phase = PlasticityPhase.MAINTENANCE
        self.plasticity_events: List[Dict[str, Any]] = []

        # Statistics
        self.total_potentiation = 0.0
        self.total_depression = 0.0
        self.connections_created = 0
        self.connections_pruned = 0

    def _init_connections(self):
        """Initialize connection metadata"""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and abs(self.weights[i, j]) > 0.01:
                    self.connections[(i, j)] = SynapticConnection(
                        pre_idx=i,
                        post_idx=j,
                        weight=self.weights[i, j]
                    )

    def apply_hebbian(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply Hebbian plasticity rule.

        "Neurons that fire together, wire together"

        Args:
            pre_activity: Presynaptic activity (num_neurons,)
            post_activity: Postsynaptic activity (num_neurons,)
            learning_rate: Override default learning rate

        Returns:
            Weight change matrix
        """
        lr = learning_rate or self.config.learning_rate

        # Compute outer product of activities
        delta_w = lr * np.outer(pre_activity, post_activity)

        # Apply metaplasticity modulation
        meta_matrix = self._get_metaplasticity_matrix()
        delta_w *= meta_matrix

        # Update weights
        self.weights += delta_w

        # Track potentiation
        self.total_potentiation += np.sum(np.maximum(delta_w, 0))
        self.total_depression += np.sum(np.maximum(-delta_w, 0))

        # Clip weights
        self.weights = np.clip(
            self.weights,
            self.config.min_weight,
            self.config.max_weight
        )

        # Log event
        self._log_plasticity_event(PlasticityType.HEBBIAN, {
            "mean_change": float(np.mean(np.abs(delta_w))),
            "learning_rate": lr
        })

        return delta_w

    def apply_anti_hebbian(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply anti-Hebbian plasticity for decorrelation.

        Weakens connections between co-active neurons.
        """
        lr = learning_rate or self.config.learning_rate

        # Anti-correlation: weaken co-active pairs
        delta_w = -lr * np.outer(pre_activity, post_activity)

        self.weights += delta_w
        self.weights = np.clip(
            self.weights,
            self.config.min_weight,
            self.config.max_weight
        )

        self._log_plasticity_event(PlasticityType.ANTI_HEBBIAN, {
            "mean_change": float(np.mean(np.abs(delta_w)))
        })

        return delta_w

    def apply_homeostatic(self, current_activity: np.ndarray) -> np.ndarray:
        """
        Apply homeostatic plasticity to maintain stable activity.

        Scales synaptic strengths to maintain target activity level.

        Args:
            current_activity: Current neural activity (num_neurons,)

        Returns:
            Scaling factors applied
        """
        # Store activity history
        self.activity_history.append(current_activity.copy())
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)

        # Compute mean activity over history
        mean_activity = np.mean(self.activity_history, axis=0)

        # Compute deviation from target
        deviation = self.config.homeostatic_target - mean_activity

        # Scale incoming weights
        scale_factors = 1.0 + deviation * self.config.learning_rate

        # Apply scaling to incoming weights (columns)
        for j in range(self.num_neurons):
            self.weights[:, j] *= scale_factors[j]

        # Clip weights
        self.weights = np.clip(
            self.weights,
            self.config.min_weight,
            self.config.max_weight
        )

        self._log_plasticity_event(PlasticityType.HOMEOSTATIC, {
            "mean_deviation": float(np.mean(np.abs(deviation))),
            "mean_scale": float(np.mean(scale_factors))
        })

        return scale_factors

    def apply_structural(
        self,
        activity: np.ndarray,
        growth_probability: float = 0.01,
        prune_threshold: float = 0.05
    ) -> Dict[str, int]:
        """
        Apply structural plasticity - create and remove connections.

        Args:
            activity: Current neural activity
            growth_probability: Probability of new synapse formation
            prune_threshold: Weight threshold below which synapses are pruned

        Returns:
            Dict with counts of created and pruned connections
        """
        created = 0
        pruned = 0

        # Prune weak connections
        for (i, j), conn in list(self.connections.items()):
            if abs(conn.weight) < prune_threshold:
                if conn.activity_count < 10 or conn.age > 100:
                    self.weights[i, j] = 0
                    del self.connections[(i, j)]
                    pruned += 1

        # Create new connections for highly active neurons
        active_neurons = np.where(activity > 0.7)[0]
        for i in active_neurons:
            if np.random.random() < growth_probability:
                # Find potential targets (currently weak/no connection)
                potential_targets = np.where(np.abs(self.weights[i, :]) < 0.01)[0]
                if len(potential_targets) > 0:
                    j = np.random.choice(potential_targets)
                    if i != j:
                        # Create new connection
                        new_weight = np.random.randn() * 0.1
                        self.weights[i, j] = new_weight
                        self.connections[(i, j)] = SynapticConnection(
                            pre_idx=i,
                            post_idx=j,
                            weight=new_weight
                        )
                        created += 1

        self.connections_created += created
        self.connections_pruned += pruned

        self._log_plasticity_event(PlasticityType.STRUCTURAL, {
            "created": created,
            "pruned": pruned,
            "total_connections": len(self.connections)
        })

        return {"created": created, "pruned": pruned}

    def apply_metaplasticity(self, activity_variance: np.ndarray):
        """
        Update metaplasticity state based on activity variance.

        High variance neurons have lower plasticity threshold.
        """
        for (i, j), conn in self.connections.items():
            # Combine pre and post variance
            combined_var = (activity_variance[i] + activity_variance[j]) / 2

            # Update plasticity state (BCM-like rule)
            if combined_var > 0.5:
                # High variance: reduce plasticity threshold
                conn.plasticity_state *= (1 - self.config.metaplasticity_rate)
            else:
                # Low variance: increase plasticity threshold
                conn.plasticity_state *= (1 + self.config.metaplasticity_rate)

            # Clip plasticity state
            conn.plasticity_state = np.clip(conn.plasticity_state, 0.1, 10.0)

        self._log_plasticity_event(PlasticityType.METAPLASTICITY, {
            "mean_variance": float(np.mean(activity_variance))
        })

    def _get_metaplasticity_matrix(self) -> np.ndarray:
        """Get matrix of metaplasticity modifiers"""
        meta_matrix = np.ones((self.num_neurons, self.num_neurons))

        for (i, j), conn in self.connections.items():
            meta_matrix[i, j] = conn.plasticity_state

        return meta_matrix

    def apply_decay(self):
        """Apply weight decay toward zero"""
        self.weights *= (1 - self.config.decay_rate)

        # Update connection metadata
        for conn in self.connections.values():
            conn.weight = self.weights[conn.pre_idx, conn.post_idx]
            conn.age += 1

    def consolidate(self) -> int:
        """
        Consolidate strong connections.

        Connections above threshold become resistant to decay.

        Returns:
            Number of consolidated connections
        """
        consolidated = 0

        for conn in self.connections.values():
            if abs(conn.weight) > self.config.consolidation_threshold:
                if conn.activity_count > 50:
                    # Mark as consolidated by increasing plasticity state
                    conn.plasticity_state = 0.5  # Reduced plasticity
                    consolidated += 1

        self.current_phase = PlasticityPhase.CONSOLIDATION

        return consolidated

    def _log_plasticity_event(
        self,
        plasticity_type: PlasticityType,
        details: Dict[str, Any]
    ):
        """Log a plasticity event"""
        self.plasticity_events.append({
            "type": plasticity_type.value,
            "phase": self.current_phase.value,
            **details
        })

        # Keep event log bounded
        if len(self.plasticity_events) > 1000:
            self.plasticity_events = self.plasticity_events[-500:]

    def get_weight_matrix(self) -> np.ndarray:
        """Get current weight matrix"""
        return self.weights.copy()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about connections"""
        weights_flat = self.weights.flatten()
        non_zero = weights_flat[np.abs(weights_flat) > 0.01]

        return {
            "total_connections": len(self.connections),
            "mean_weight": float(np.mean(non_zero)) if len(non_zero) > 0 else 0.0,
            "std_weight": float(np.std(non_zero)) if len(non_zero) > 0 else 0.0,
            "max_weight": float(np.max(np.abs(weights_flat))),
            "sparsity": 1 - (len(non_zero) / len(weights_flat)),
            "total_potentiation": self.total_potentiation,
            "total_depression": self.total_depression,
            "connections_created": self.connections_created,
            "connections_pruned": self.connections_pruned
        }

    def get_state(self) -> Dict[str, Any]:
        """Get complete engine state"""
        return {
            "num_neurons": self.num_neurons,
            "config": {
                "learning_rate": self.config.learning_rate,
                "decay_rate": self.config.decay_rate,
                "homeostatic_target": self.config.homeostatic_target
            },
            "phase": self.current_phase.value,
            "connection_stats": self.get_connection_stats(),
            "event_count": len(self.plasticity_events)
        }


# ============================================================================
# Adaptive Learning Rate
# ============================================================================

class AdaptivePlasticityRate:
    """
    Adaptive learning rate that adjusts based on network state.

    Implements:
    - Surprise-based modulation (higher LR for unexpected events)
    - Activity-dependent scaling
    - Temporal dynamics (fast initial, slow consolidation)
    """

    def __init__(
        self,
        base_rate: float = 0.01,
        surprise_scale: float = 2.0,
        activity_scale: float = 1.5,
        decay_half_life: int = 100
    ):
        self.base_rate = base_rate
        self.surprise_scale = surprise_scale
        self.activity_scale = activity_scale
        self.decay_half_life = decay_half_life

        self.prediction_error_history: List[float] = []
        self.step_count = 0

    def compute_rate(
        self,
        prediction_error: float,
        activity_level: float
    ) -> float:
        """
        Compute adaptive learning rate.

        Args:
            prediction_error: Error between predicted and actual outcome
            activity_level: Current network activity level (0-1)

        Returns:
            Adapted learning rate
        """
        self.step_count += 1

        # Temporal decay
        temporal_factor = 2 ** (-self.step_count / self.decay_half_life)

        # Surprise modulation
        self.prediction_error_history.append(prediction_error)
        if len(self.prediction_error_history) > 100:
            self.prediction_error_history.pop(0)

        mean_error = np.mean(self.prediction_error_history)
        if prediction_error > mean_error * 1.5:
            surprise_factor = self.surprise_scale
        else:
            surprise_factor = 1.0

        # Activity modulation
        activity_factor = 1.0 + (activity_level - 0.5) * (self.activity_scale - 1.0)

        # Combine factors
        rate = self.base_rate * temporal_factor * surprise_factor * activity_factor

        return float(np.clip(rate, self.base_rate * 0.1, self.base_rate * 10))


# ============================================================================
# Experience-Dependent Plasticity
# ============================================================================

class ExperienceDependentPlasticity:
    """
    Higher-level plasticity driven by experiences and outcomes.

    Implements reward-modulated learning and experience replay.
    """

    def __init__(
        self,
        engine: NeuroplasticityEngine,
        reward_scale: float = 1.0,
        replay_batch_size: int = 16
    ):
        self.engine = engine
        self.reward_scale = reward_scale
        self.replay_batch_size = replay_batch_size

        # Experience buffer
        self.experience_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 10000

        # Adaptive rate
        self.adaptive_rate = AdaptivePlasticityRate()

    def store_experience(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float,
        prediction_error: float = 0.0
    ):
        """Store an experience for later replay"""
        self.experience_buffer.append({
            "pre": pre_activity.copy(),
            "post": post_activity.copy(),
            "reward": reward,
            "prediction_error": prediction_error
        })

        if len(self.experience_buffer) > self.max_buffer_size:
            # Remove oldest, lowest reward experiences
            self.experience_buffer.sort(key=lambda x: abs(x["reward"]), reverse=True)
            self.experience_buffer = self.experience_buffer[:self.max_buffer_size]

    def apply_reward_modulated_plasticity(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float
    ) -> np.ndarray:
        """
        Apply plasticity modulated by reward signal.

        Positive reward: strengthen active connections
        Negative reward: weaken active connections (learning from mistakes)
        """
        # Compute adaptive learning rate
        activity_level = float(np.mean(post_activity))
        lr = self.adaptive_rate.compute_rate(abs(reward), activity_level)

        # Modulate by reward
        effective_lr = lr * reward * self.reward_scale

        # Apply Hebbian with modulated rate
        delta_w = self.engine.apply_hebbian(pre_activity, post_activity, effective_lr)

        # Store experience
        self.store_experience(pre_activity, post_activity, reward)

        return delta_w

    def replay_experiences(self, num_replays: int = 1) -> int:
        """
        Replay stored experiences for consolidation.

        Returns:
            Number of experiences replayed
        """
        if not self.experience_buffer:
            return 0

        total_replayed = 0

        for _ in range(num_replays):
            # Sample batch prioritized by reward magnitude
            weights = np.array([abs(e["reward"]) + 0.1 for e in self.experience_buffer])
            weights /= weights.sum()

            indices = np.random.choice(
                len(self.experience_buffer),
                size=min(self.replay_batch_size, len(self.experience_buffer)),
                replace=False,
                p=weights
            )

            for idx in indices:
                exp = self.experience_buffer[idx]

                # Replay with reduced learning rate
                self.engine.apply_hebbian(
                    exp["pre"],
                    exp["post"],
                    learning_rate=self.engine.config.learning_rate * 0.1 * (1 + exp["reward"])
                )
                total_replayed += 1

        return total_replayed


# ============================================================================
# Critical Period Plasticity
# ============================================================================

class CriticalPeriodManager:
    """
    Manages critical periods of heightened plasticity.

    Inspired by biological critical periods during development.
    """

    def __init__(self, engine: NeuroplasticityEngine):
        self.engine = engine
        self.original_config = copy.copy(engine.config)

        # Critical period state
        self.in_critical_period = False
        self.period_start_step = 0
        self.period_duration = 0
        self.plasticity_boost = 1.0

    def start_critical_period(
        self,
        duration: int,
        plasticity_boost: float = 5.0
    ):
        """
        Start a critical period of heightened plasticity.

        Args:
            duration: Number of steps for critical period
            plasticity_boost: Multiplier for learning rate
        """
        self.in_critical_period = True
        self.period_duration = duration
        self.plasticity_boost = plasticity_boost
        self.period_start_step = 0

        # Boost learning rate
        self.engine.config.learning_rate = (
            self.original_config.learning_rate * plasticity_boost
        )

        # Reduce consolidation threshold (easier to learn)
        self.engine.config.consolidation_threshold *= 0.5

    def end_critical_period(self):
        """End the critical period and restore normal plasticity"""
        self.in_critical_period = False

        # Restore original config
        self.engine.config.learning_rate = self.original_config.learning_rate
        self.engine.config.consolidation_threshold = self.original_config.consolidation_threshold

        # Consolidate learned connections
        self.engine.consolidate()

    def step(self) -> bool:
        """
        Advance critical period by one step.

        Returns:
            True if still in critical period
        """
        if not self.in_critical_period:
            return False

        self.period_start_step += 1

        if self.period_start_step >= self.period_duration:
            self.end_critical_period()
            return False

        # Gradually reduce plasticity boost
        progress = self.period_start_step / self.period_duration
        current_boost = self.plasticity_boost * (1 - progress * 0.5)
        self.engine.config.learning_rate = (
            self.original_config.learning_rate * current_boost
        )

        return True

    def get_state(self) -> Dict[str, Any]:
        """Get critical period state"""
        return {
            "in_critical_period": self.in_critical_period,
            "progress": self.period_start_step / self.period_duration if self.period_duration > 0 else 0,
            "current_boost": self.engine.config.learning_rate / self.original_config.learning_rate
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Neuroplasticity Engine Demo")
    print("=" * 50)

    # Create engine
    engine = NeuroplasticityEngine(num_neurons=50)

    print(f"\n1. Initial State:")
    stats = engine.get_connection_stats()
    print(f"   Total connections: {stats['total_connections']}")
    print(f"   Mean weight: {stats['mean_weight']:.4f}")

    # Simulate activity
    print("\n2. Applying Hebbian Plasticity:")
    for i in range(10):
        pre = (np.random.rand(50) > 0.7).astype(float)
        post = (np.random.rand(50) > 0.7).astype(float)
        engine.apply_hebbian(pre, post)

    stats = engine.get_connection_stats()
    print(f"   Potentiation: {stats['total_potentiation']:.4f}")
    print(f"   Depression: {stats['total_depression']:.4f}")

    # Apply homeostatic plasticity
    print("\n3. Applying Homeostatic Plasticity:")
    for i in range(20):
        activity = np.random.rand(50)
        engine.apply_homeostatic(activity)

    # Structural plasticity
    print("\n4. Applying Structural Plasticity:")
    activity = np.random.rand(50)
    result = engine.apply_structural(activity)
    print(f"   Created: {result['created']}")
    print(f"   Pruned: {result['pruned']}")

    # Experience-dependent plasticity
    print("\n5. Experience-Dependent Learning:")
    exp_plasticity = ExperienceDependentPlasticity(engine)

    for i in range(20):
        pre = (np.random.rand(50) > 0.5).astype(float)
        post = (np.random.rand(50) > 0.5).astype(float)
        reward = np.random.randn() * 0.5
        exp_plasticity.apply_reward_modulated_plasticity(pre, post, reward)

    print(f"   Experiences stored: {len(exp_plasticity.experience_buffer)}")

    # Replay experiences
    replayed = exp_plasticity.replay_experiences(num_replays=2)
    print(f"   Experiences replayed: {replayed}")

    # Critical period
    print("\n6. Critical Period:")
    critical_manager = CriticalPeriodManager(engine)
    critical_manager.start_critical_period(duration=10, plasticity_boost=3.0)

    for i in range(10):
        still_active = critical_manager.step()
        if i % 3 == 0:
            pre = (np.random.rand(50) > 0.5).astype(float)
            post = (np.random.rand(50) > 0.5).astype(float)
            engine.apply_hebbian(pre, post)

    print(f"   Critical period ended: {not critical_manager.in_critical_period}")

    # Final state
    print("\n7. Final State:")
    state = engine.get_state()
    print(f"   Phase: {state['phase']}")
    print(f"   Connections: {state['connection_stats']['total_connections']}")
    print(f"   Created: {state['connection_stats']['connections_created']}")
    print(f"   Pruned: {state['connection_stats']['connections_pruned']}")

    print("\n" + "=" * 50)
    print("Neuroplasticity Engine ready!")
