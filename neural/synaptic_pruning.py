"""
DENDRITE-X: Synaptic Pruning

Experience-based connection pruning for efficient neural networks.
Implements biological pruning mechanisms for network optimization.

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Pruning Strategies
# ============================================================================

class PruningStrategy(Enum):
    """Strategies for synaptic pruning"""
    ACTIVITY_BASED = "activity_based"  # Prune inactive connections
    MAGNITUDE_BASED = "magnitude_based"  # Prune weak connections
    GRADIENT_BASED = "gradient_based"  # Prune low-gradient connections
    COMPETITION_BASED = "competition_based"  # Winner-take-all pruning
    DEVELOPMENTAL = "developmental"  # Age-dependent pruning
    RANDOM = "random"  # Stochastic pruning


class PruningPhase(Enum):
    """Phases of development affecting pruning"""
    EARLY_DEVELOPMENT = "early_development"  # High synaptogenesis
    CRITICAL_PERIOD = "critical_period"  # Experience-dependent refinement
    ADOLESCENT = "adolescent"  # Major pruning phase
    ADULT = "adult"  # Maintenance pruning
    AGING = "aging"  # Gradual decline


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PruningConfig:
    """Configuration for synaptic pruning"""
    base_threshold: float = 0.1
    activity_window: int = 100
    min_activity_rate: float = 0.01
    competition_radius: int = 5
    age_factor: float = 0.001
    pruning_rate: float = 0.05  # Max fraction to prune per step
    protection_period: int = 50  # Steps before new connections can be pruned
    regrowth_probability: float = 0.01


@dataclass
class SynapseInfo:
    """Information about a synapse for pruning decisions"""
    pre_idx: int
    post_idx: int
    weight: float
    age: int = 0
    activity_count: int = 0
    last_active_step: int = 0
    protected: bool = False
    importance_score: float = 0.0


# ============================================================================
# Synaptic Pruning Engine
# ============================================================================

class SynapticPruningEngine:
    """
    Engine for experience-based synaptic pruning.

    Implements multiple pruning mechanisms:
    1. Activity-based: Remove unused connections
    2. Magnitude-based: Remove weak connections
    3. Competition-based: Winner-take-all among similar connections
    4. Developmental: Age-appropriate pruning schedules

    Inspired by:
    - Developmental synaptic pruning
    - Microglia-mediated elimination
    - Use-dependent selection

    Example:
        pruner = SynapticPruningEngine(num_neurons=100)

        # Update synapse activity
        pruner.record_activity(pre_idx=5, post_idx=10)

        # Run pruning
        removed = pruner.prune(strategy=PruningStrategy.ACTIVITY_BASED)
    """

    def __init__(
        self,
        num_neurons: int,
        weights: Optional[np.ndarray] = None,
        config: Optional[PruningConfig] = None
    ):
        self.num_neurons = num_neurons
        self.config = config or PruningConfig()

        # Initialize or use provided weights
        if weights is not None:
            self.weights = weights.copy()
        else:
            self.weights = np.random.randn(num_neurons, num_neurons) * 0.1
            np.fill_diagonal(self.weights, 0)

        # Synapse tracking
        self.synapses: Dict[Tuple[int, int], SynapseInfo] = {}
        self._init_synapses()

        # Activity history
        self.activity_history: List[Set[Tuple[int, int]]] = []
        self.current_step = 0

        # Pruning statistics
        self.total_pruned = 0
        self.total_regrown = 0
        self.pruning_history: List[Dict[str, Any]] = []

        # Development phase
        self.phase = PruningPhase.ADULT

    def _init_synapses(self):
        """Initialize synapse tracking"""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and abs(self.weights[i, j]) > 0.001:
                    self.synapses[(i, j)] = SynapseInfo(
                        pre_idx=i,
                        post_idx=j,
                        weight=self.weights[i, j],
                        protected=True  # Initially protected
                    )

    def set_phase(self, phase: PruningPhase):
        """Set developmental phase affecting pruning behavior"""
        self.phase = phase

    def record_activity(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        threshold: float = 0.5
    ):
        """
        Record synaptic activity based on pre/post neural activity.

        Args:
            pre_activity: Presynaptic activity levels
            post_activity: Postsynaptic activity levels
            threshold: Activity threshold for counting as active
        """
        self.current_step += 1
        active_synapses: Set[Tuple[int, int]] = set()

        # Find active pre and post neurons
        active_pre = np.where(pre_activity > threshold)[0]
        active_post = np.where(post_activity > threshold)[0]

        # Record activity for each active synapse
        for i in active_pre:
            for j in active_post:
                if (i, j) in self.synapses:
                    syn = self.synapses[(i, j)]
                    syn.activity_count += 1
                    syn.last_active_step = self.current_step
                    active_synapses.add((i, j))

        # Store activity history
        self.activity_history.append(active_synapses)
        if len(self.activity_history) > self.config.activity_window:
            self.activity_history.pop(0)

        # Update synapse ages and protection
        for syn in self.synapses.values():
            syn.age += 1
            if syn.age >= self.config.protection_period:
                syn.protected = False

    def compute_importance_scores(self, strategy: PruningStrategy):
        """Compute importance scores for all synapses"""
        for syn in self.synapses.values():
            if strategy == PruningStrategy.ACTIVITY_BASED:
                # Score based on recent activity
                activity_rate = syn.activity_count / max(1, self.current_step)
                recency = 1.0 / (1 + (self.current_step - syn.last_active_step) * 0.1)
                syn.importance_score = activity_rate * recency

            elif strategy == PruningStrategy.MAGNITUDE_BASED:
                # Score based on weight magnitude
                syn.importance_score = abs(syn.weight)

            elif strategy == PruningStrategy.DEVELOPMENTAL:
                # Combine activity and age
                activity_score = syn.activity_count / max(1, syn.age)
                age_penalty = 1.0 / (1 + syn.age * self.config.age_factor)
                syn.importance_score = activity_score * (1 - age_penalty * 0.5)

            elif strategy == PruningStrategy.COMPETITION_BASED:
                # Will be computed in competition phase
                syn.importance_score = abs(syn.weight)

            elif strategy == PruningStrategy.RANDOM:
                syn.importance_score = np.random.random()

    def _prune_activity_based(self) -> List[Tuple[int, int]]:
        """Prune inactive synapses"""
        to_prune = []

        for (i, j), syn in self.synapses.items():
            if syn.protected:
                continue

            # Check activity rate
            activity_rate = syn.activity_count / max(1, self.current_step)
            if activity_rate < self.config.min_activity_rate:
                # Check recency
                steps_since_active = self.current_step - syn.last_active_step
                if steps_since_active > self.config.activity_window:
                    to_prune.append((i, j))

        return to_prune

    def _prune_magnitude_based(self) -> List[Tuple[int, int]]:
        """Prune weak synapses"""
        to_prune = []

        for (i, j), syn in self.synapses.items():
            if syn.protected:
                continue

            if abs(syn.weight) < self.config.base_threshold:
                to_prune.append((i, j))

        return to_prune

    def _prune_competition_based(self) -> List[Tuple[int, int]]:
        """Prune based on competition among similar synapses"""
        to_prune = []

        # Group synapses by postsynaptic neuron
        post_groups: Dict[int, List[SynapseInfo]] = {}
        for syn in self.synapses.values():
            if syn.post_idx not in post_groups:
                post_groups[syn.post_idx] = []
            post_groups[syn.post_idx].append(syn)

        # Competition within each group
        for post_idx, synapses in post_groups.items():
            if len(synapses) <= 1:
                continue

            # Sort by importance
            synapses.sort(key=lambda s: s.importance_score, reverse=True)

            # Keep top fraction, prune rest
            keep_count = max(1, int(len(synapses) * (1 - self.config.pruning_rate)))
            for syn in synapses[keep_count:]:
                if not syn.protected:
                    to_prune.append((syn.pre_idx, syn.post_idx))

        return to_prune

    def _prune_developmental(self) -> List[Tuple[int, int]]:
        """Developmental phase-appropriate pruning"""
        to_prune = []

        # Adjust pruning rate by phase
        phase_rates = {
            PruningPhase.EARLY_DEVELOPMENT: 0.01,
            PruningPhase.CRITICAL_PERIOD: 0.03,
            PruningPhase.ADOLESCENT: 0.08,
            PruningPhase.ADULT: 0.02,
            PruningPhase.AGING: 0.05
        }

        prune_rate = phase_rates.get(self.phase, 0.02)

        # Score all synapses
        scored = [(syn.importance_score, (i, j))
                  for (i, j), syn in self.synapses.items()
                  if not syn.protected]

        if not scored:
            return to_prune

        # Sort by score (lowest first = prune first)
        scored.sort(key=lambda x: x[0])

        # Prune lowest scoring fraction
        prune_count = int(len(scored) * prune_rate)
        to_prune = [idx for _, idx in scored[:prune_count]]

        return to_prune

    def prune(
        self,
        strategy: PruningStrategy = PruningStrategy.ACTIVITY_BASED,
        max_prune: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Execute pruning with specified strategy.

        Args:
            strategy: Pruning strategy to use
            max_prune: Maximum number of synapses to prune

        Returns:
            List of pruned synapse indices
        """
        # Compute importance scores
        self.compute_importance_scores(strategy)

        # Get pruning candidates based on strategy
        if strategy == PruningStrategy.ACTIVITY_BASED:
            to_prune = self._prune_activity_based()
        elif strategy == PruningStrategy.MAGNITUDE_BASED:
            to_prune = self._prune_magnitude_based()
        elif strategy == PruningStrategy.COMPETITION_BASED:
            to_prune = self._prune_competition_based()
        elif strategy == PruningStrategy.DEVELOPMENTAL:
            to_prune = self._prune_developmental()
        elif strategy == PruningStrategy.RANDOM:
            candidates = [(i, j) for (i, j), syn in self.synapses.items()
                          if not syn.protected]
            prune_count = int(len(candidates) * self.config.pruning_rate)
            to_prune = list(np.random.choice(len(candidates), prune_count, replace=False))
            to_prune = [candidates[i] for i in to_prune]
        else:
            to_prune = []

        # Apply max prune limit
        if max_prune is not None:
            to_prune = to_prune[:max_prune]

        # Execute pruning
        for i, j in to_prune:
            self.weights[i, j] = 0
            if (i, j) in self.synapses:
                del self.synapses[(i, j)]

        self.total_pruned += len(to_prune)

        # Record history
        self.pruning_history.append({
            "step": self.current_step,
            "strategy": strategy.value,
            "count": len(to_prune),
            "remaining": len(self.synapses)
        })

        return to_prune

    def regrow(self, num_new: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Regrow synapses at empty locations.

        Args:
            num_new: Number of new synapses to create (default: based on config)

        Returns:
            List of new synapse indices
        """
        if num_new is None:
            # Default: regrow proportional to pruning rate
            num_new = int(self.num_neurons ** 2 * self.config.regrowth_probability)

        new_synapses = []

        # Find empty locations
        empty = []
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and (i, j) not in self.synapses:
                    empty.append((i, j))

        if not empty:
            return new_synapses

        # Sample new locations
        num_new = min(num_new, len(empty))
        indices = np.random.choice(len(empty), num_new, replace=False)

        for idx in indices:
            i, j = empty[idx]

            # Initialize new synapse
            weight = np.random.randn() * 0.05
            self.weights[i, j] = weight

            self.synapses[(i, j)] = SynapseInfo(
                pre_idx=i,
                post_idx=j,
                weight=weight,
                protected=True  # New synapses are protected
            )

            new_synapses.append((i, j))

        self.total_regrown += len(new_synapses)

        return new_synapses

    def get_statistics(self) -> Dict[str, Any]:
        """Get pruning statistics"""
        weights_flat = self.weights.flatten()
        non_zero = weights_flat[np.abs(weights_flat) > 0.001]

        return {
            "total_synapses": len(self.synapses),
            "total_pruned": self.total_pruned,
            "total_regrown": self.total_regrown,
            "net_change": self.total_regrown - self.total_pruned,
            "sparsity": 1 - (len(non_zero) / len(weights_flat)),
            "mean_weight": float(np.mean(np.abs(non_zero))) if len(non_zero) > 0 else 0,
            "phase": self.phase.value,
            "current_step": self.current_step
        }

    def get_weight_matrix(self) -> np.ndarray:
        """Get current weight matrix"""
        return self.weights.copy()


# ============================================================================
# Competitive Elimination
# ============================================================================

class CompetitiveElimination:
    """
    Implements competitive elimination where synapses compete
    for limited resources/validation signals.
    """

    def __init__(
        self,
        pruning_engine: SynapticPruningEngine,
        resource_per_neuron: float = 1.0
    ):
        self.engine = pruning_engine
        self.resource_per_neuron = resource_per_neuron

        # Resource allocation
        self.resources = np.ones(pruning_engine.num_neurons) * resource_per_neuron

    def allocate_resources(self, activity: np.ndarray):
        """Allocate resources based on activity"""
        # Active neurons get more resources
        self.resources = self.resource_per_neuron * (0.5 + 0.5 * activity)

    def compete_and_prune(self) -> List[Tuple[int, int]]:
        """
        Run competition and prune losers.

        Synapses on neurons with less resources are more likely to be pruned.
        """
        to_prune = []

        for (i, j), syn in self.engine.synapses.items():
            if syn.protected:
                continue

            # Competition based on pre and post resources
            resource_factor = self.resources[i] * self.resources[j]

            # Low resources + weak weight = prune
            survival_prob = resource_factor * abs(syn.weight) / self.engine.config.base_threshold

            if np.random.random() > survival_prob:
                to_prune.append((i, j))

        # Execute pruning
        for i, j in to_prune:
            self.engine.weights[i, j] = 0
            if (i, j) in self.engine.synapses:
                del self.engine.synapses[(i, j)]

        self.engine.total_pruned += len(to_prune)

        return to_prune


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("DENDRITE-X: Synaptic Pruning Demo")
    print("=" * 50)

    # Create pruning engine
    pruner = SynapticPruningEngine(num_neurons=50)

    print(f"\n1. Initial State:")
    stats = pruner.get_statistics()
    print(f"   Synapses: {stats['total_synapses']}")
    print(f"   Sparsity: {stats['sparsity']:.2%}")

    # Simulate activity
    print("\n2. Recording Activity:")
    for step in range(100):
        pre = (np.random.rand(50) > 0.7).astype(float)
        post = (np.random.rand(50) > 0.7).astype(float)
        pruner.record_activity(pre, post)

    print(f"   Steps recorded: {pruner.current_step}")

    # Activity-based pruning
    print("\n3. Activity-Based Pruning:")
    removed = pruner.prune(PruningStrategy.ACTIVITY_BASED)
    print(f"   Pruned: {len(removed)}")

    # Magnitude-based pruning
    print("\n4. Magnitude-Based Pruning:")
    removed = pruner.prune(PruningStrategy.MAGNITUDE_BASED)
    print(f"   Pruned: {len(removed)}")

    # Developmental pruning
    print("\n5. Developmental Pruning (Adolescent Phase):")
    pruner.set_phase(PruningPhase.ADOLESCENT)
    removed = pruner.prune(PruningStrategy.DEVELOPMENTAL)
    print(f"   Pruned: {len(removed)}")

    # Regrowth
    print("\n6. Synaptic Regrowth:")
    new_synapses = pruner.regrow(num_new=20)
    print(f"   Regrown: {len(new_synapses)}")

    # Competitive elimination
    print("\n7. Competitive Elimination:")
    competition = CompetitiveElimination(pruner)
    activity = np.random.rand(50)
    competition.allocate_resources(activity)
    eliminated = competition.compete_and_prune()
    print(f"   Eliminated: {len(eliminated)}")

    # Final statistics
    print("\n8. Final Statistics:")
    stats = pruner.get_statistics()
    print(f"   Total synapses: {stats['total_synapses']}")
    print(f"   Total pruned: {stats['total_pruned']}")
    print(f"   Total regrown: {stats['total_regrown']}")
    print(f"   Net change: {stats['net_change']}")
    print(f"   Sparsity: {stats['sparsity']:.2%}")

    print("\n" + "=" * 50)
    print("Synaptic Pruning Engine ready!")
