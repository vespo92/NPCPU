"""
Organism Lifecycle Management

Manages the complete lifecycle of a digital organism from birth through
death, including growth patterns, maturation, and graceful termination.

Like biological organisms, NPCPU entities have:
- Birth/spawning
- Growth and development
- Maturity and peak performance
- Senescence and decline
- Death and resource release
"""

import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Enums
# ============================================================================

class LifecycleStage(Enum):
    """Stages of organism lifecycle"""
    EMBRYONIC = "embryonic"      # Pre-birth, initializing
    NASCENT = "nascent"          # Just born, highly plastic
    JUVENILE = "juvenile"        # Growing, learning rapidly
    ADOLESCENT = "adolescent"    # Approaching maturity
    MATURE = "mature"            # Peak performance
    SENIOR = "senior"            # Beginning decline
    SENESCENT = "senescent"      # Significant decline
    DYING = "dying"              # Terminal stage
    DEAD = "dead"                # No longer active


class DeathCause(Enum):
    """Causes of organism death"""
    NATURAL = "natural"          # Old age
    STARVATION = "starvation"    # Resource depletion
    DAMAGE = "damage"            # Accumulated damage
    TERMINATED = "terminated"    # External termination
    SACRIFICE = "sacrifice"      # Self-termination for collective
    MERGED = "merged"            # Merged into another organism
    EVOLVED = "evolved"          # Transformed into new form


class GrowthPattern(Enum):
    """Patterns of growth"""
    LINEAR = "linear"            # Steady growth
    EXPONENTIAL = "exponential"  # Accelerating growth
    SIGMOIDAL = "sigmoidal"      # S-curve (slow-fast-slow)
    PUNCTUATED = "punctuated"    # Bursts of growth
    ADAPTIVE = "adaptive"        # Environment-responsive


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LifecycleEvent:
    """An event in the organism's lifecycle"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    stage_from: Optional[LifecycleStage] = None
    stage_to: Optional[LifecycleStage] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GrowthMetrics:
    """Metrics tracking organism growth"""
    size: float = 1.0
    complexity: float = 0.0
    capability_sum: float = 0.0
    connection_count: int = 0
    knowledge_units: int = 0
    experience_count: int = 0


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle behavior"""
    # Stage durations (in cycles)
    embryonic_duration: int = 10
    nascent_duration: int = 50
    juvenile_duration: int = 200
    adolescent_duration: int = 100
    mature_duration: int = 1000
    senior_duration: int = 200
    senescent_duration: int = 100

    # Growth parameters
    growth_pattern: GrowthPattern = GrowthPattern.SIGMOIDAL
    max_size: float = 100.0
    growth_rate: float = 0.1

    # Decline parameters
    decline_rate: float = 0.01
    death_threshold: float = 0.1

    # Reproduction
    reproduction_age_min: int = 300  # Minimum cycles before reproduction
    reproduction_cost: float = 0.3   # Energy cost of reproduction


# ============================================================================
# Organism Lifecycle
# ============================================================================

class OrganismLifecycle:
    """
    Manages the complete lifecycle of a digital organism.

    Features:
    - Stage transitions with callbacks
    - Growth tracking and patterns
    - Maturation of capabilities
    - Graceful aging and death
    - Reproduction and offspring

    Example:
        lifecycle = OrganismLifecycle(consciousness)

        # Register callbacks
        lifecycle.on_stage_change(lambda old, new: print(f"{old} -> {new}"))

        # Run lifecycle
        await lifecycle.tick()  # Call each cycle

        # Check status
        if lifecycle.can_reproduce():
            offspring = lifecycle.reproduce()
    """

    def __init__(
        self,
        consciousness: GradedConsciousness,
        config: Optional[LifecycleConfig] = None,
        organism_id: Optional[str] = None
    ):
        self.organism_id = organism_id or str(uuid.uuid4())
        self.consciousness = consciousness
        self.config = config or LifecycleConfig()

        # Lifecycle state
        self.stage = LifecycleStage.EMBRYONIC
        self.age = 0  # In cycles
        self.birth_time = time.time()
        self.death_time: Optional[float] = None
        self.death_cause: Optional[DeathCause] = None

        # Growth tracking
        self.metrics = GrowthMetrics()
        self.peak_metrics: Optional[GrowthMetrics] = None

        # Event history
        self.events: List[LifecycleEvent] = []
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Initial consciousness snapshot
        self._initial_consciousness = consciousness.get_capability_scores().copy()

        # Record birth event
        self._record_event("birth", details={"initial_scores": self._initial_consciousness})

    def on_stage_change(self, callback: Callable[[LifecycleStage, LifecycleStage], None]):
        """Register callback for stage changes"""
        self._callbacks["stage_change"].append(callback)

    def on_death(self, callback: Callable[['OrganismLifecycle', DeathCause], None]):
        """Register callback for death"""
        self._callbacks["death"].append(callback)

    def on_reproduction(self, callback: Callable[['OrganismLifecycle', 'OrganismLifecycle'], None]):
        """Register callback for reproduction"""
        self._callbacks["reproduction"].append(callback)

    async def tick(self) -> LifecycleStage:
        """
        Process one lifecycle cycle.

        Call this regularly to advance the organism's lifecycle.
        """
        if self.stage == LifecycleStage.DEAD:
            return self.stage

        self.age += 1

        # Update growth metrics
        self._update_metrics()

        # Check for stage transitions
        new_stage = self._determine_stage()
        if new_stage != self.stage:
            await self._transition_stage(new_stage)

        # Apply stage-specific effects
        await self._apply_stage_effects()

        return self.stage

    def _update_metrics(self):
        """Update growth metrics"""
        scores = self.consciousness.get_capability_scores()

        self.metrics.capability_sum = sum(scores.values())
        self.metrics.complexity = len([s for s in scores.values() if s > 0.5])

        # Update size based on growth pattern
        self.metrics.size = self._calculate_size()

        # Track peak
        if self.peak_metrics is None or self.metrics.capability_sum > self.peak_metrics.capability_sum:
            self.peak_metrics = GrowthMetrics(
                size=self.metrics.size,
                complexity=self.metrics.complexity,
                capability_sum=self.metrics.capability_sum,
                connection_count=self.metrics.connection_count,
                knowledge_units=self.metrics.knowledge_units,
                experience_count=self.metrics.experience_count
            )

    def _calculate_size(self) -> float:
        """Calculate organism size based on growth pattern"""
        config = self.config

        if config.growth_pattern == GrowthPattern.LINEAR:
            return min(config.max_size, 1.0 + self.age * config.growth_rate)

        elif config.growth_pattern == GrowthPattern.EXPONENTIAL:
            return min(config.max_size, np.exp(self.age * config.growth_rate * 0.01))

        elif config.growth_pattern == GrowthPattern.SIGMOIDAL:
            # S-curve growth
            midpoint = (config.juvenile_duration + config.adolescent_duration) / 2
            steepness = 0.05
            return config.max_size / (1 + np.exp(-steepness * (self.age - midpoint)))

        elif config.growth_pattern == GrowthPattern.PUNCTUATED:
            # Bursts at stage transitions
            burst_ages = [
                config.nascent_duration,
                config.nascent_duration + config.juvenile_duration,
                config.nascent_duration + config.juvenile_duration + config.adolescent_duration
            ]
            bursts = sum(1 for ba in burst_ages if self.age > ba)
            return min(config.max_size, 1.0 + bursts * 10 + self.age * config.growth_rate * 0.5)

        else:  # ADAPTIVE
            # Based on current capability sum
            return min(config.max_size, self.metrics.capability_sum * 2)

    def _determine_stage(self) -> LifecycleStage:
        """Determine what stage the organism should be in"""
        config = self.config

        cumulative = 0

        cumulative += config.embryonic_duration
        if self.age < cumulative:
            return LifecycleStage.EMBRYONIC

        cumulative += config.nascent_duration
        if self.age < cumulative:
            return LifecycleStage.NASCENT

        cumulative += config.juvenile_duration
        if self.age < cumulative:
            return LifecycleStage.JUVENILE

        cumulative += config.adolescent_duration
        if self.age < cumulative:
            return LifecycleStage.ADOLESCENT

        cumulative += config.mature_duration
        if self.age < cumulative:
            return LifecycleStage.MATURE

        cumulative += config.senior_duration
        if self.age < cumulative:
            return LifecycleStage.SENIOR

        cumulative += config.senescent_duration
        if self.age < cumulative:
            return LifecycleStage.SENESCENT

        return LifecycleStage.DYING

    async def _transition_stage(self, new_stage: LifecycleStage):
        """Handle stage transition"""
        old_stage = self.stage
        self.stage = new_stage

        # Record event
        self._record_event(
            "stage_transition",
            stage_from=old_stage,
            stage_to=new_stage
        )

        # Trigger callbacks
        for callback in self._callbacks["stage_change"]:
            callback(old_stage, new_stage)

        # Check for death
        if new_stage == LifecycleStage.DYING:
            await self._begin_dying()

    async def _apply_stage_effects(self):
        """Apply effects based on current stage"""
        scores = self.consciousness.get_capability_scores()

        if self.stage == LifecycleStage.EMBRYONIC:
            # Rapid initial development
            for cap in scores:
                scores[cap] = min(1.0, scores[cap] * 1.05)

        elif self.stage == LifecycleStage.NASCENT:
            # High plasticity, rapid learning
            for cap in scores:
                scores[cap] = min(1.0, scores[cap] * 1.02)

        elif self.stage == LifecycleStage.JUVENILE:
            # Continued growth, specialization beginning
            for cap in scores:
                if scores[cap] > 0.5:
                    scores[cap] = min(1.0, scores[cap] * 1.01)

        elif self.stage == LifecycleStage.ADOLESCENT:
            # Approaching peak
            pass  # No automatic changes

        elif self.stage == LifecycleStage.MATURE:
            # Peak performance, stable
            pass

        elif self.stage == LifecycleStage.SENIOR:
            # Beginning decline
            for cap in scores:
                scores[cap] = max(0.1, scores[cap] * (1 - self.config.decline_rate * 0.5))

        elif self.stage == LifecycleStage.SENESCENT:
            # Significant decline
            for cap in scores:
                scores[cap] = max(0.1, scores[cap] * (1 - self.config.decline_rate))

        elif self.stage == LifecycleStage.DYING:
            # Rapid decline
            for cap in scores:
                scores[cap] = max(0.0, scores[cap] * 0.9)

            # Check for death
            if self.consciousness.overall_consciousness_score() < self.config.death_threshold:
                await self._die(DeathCause.NATURAL)

        # Update consciousness
        self.consciousness = GradedConsciousness(**scores)

    async def _begin_dying(self):
        """Begin the dying process"""
        self._record_event("dying_began")

    async def _die(self, cause: DeathCause):
        """Process death"""
        self.stage = LifecycleStage.DEAD
        self.death_time = time.time()
        self.death_cause = cause

        self._record_event(
            "death",
            details={
                "cause": cause.value,
                "final_age": self.age,
                "peak_capability": self.peak_metrics.capability_sum if self.peak_metrics else 0
            }
        )

        # Trigger callbacks
        for callback in self._callbacks["death"]:
            callback(self, cause)

    def can_reproduce(self) -> bool:
        """Check if organism can reproduce"""
        return (
            self.stage in [LifecycleStage.MATURE, LifecycleStage.ADOLESCENT] and
            self.age >= self.config.reproduction_age_min and
            self.consciousness.overall_consciousness_score() > self.config.reproduction_cost
        )

    def reproduce(self) -> Optional['OrganismLifecycle']:
        """
        Create offspring organism.

        Returns new organism or None if cannot reproduce.
        """
        if not self.can_reproduce():
            return None

        # Get current scores
        scores = self.consciousness.get_capability_scores()

        # Create offspring with inherited traits + variation
        offspring_scores = {}
        for cap, score in scores.items():
            # Inherit with some variation
            variation = np.random.uniform(-0.1, 0.1)
            offspring_scores[cap] = max(0.0, min(1.0, score * 0.8 + variation))

        offspring_consciousness = GradedConsciousness(**offspring_scores)

        # Create offspring lifecycle
        offspring = OrganismLifecycle(
            consciousness=offspring_consciousness,
            config=self.config
        )

        # Cost to parent
        for cap in scores:
            scores[cap] = max(0.1, scores[cap] * (1 - self.config.reproduction_cost))
        self.consciousness = GradedConsciousness(**scores)

        # Record events
        self._record_event(
            "reproduction",
            details={"offspring_id": offspring.organism_id}
        )
        offspring._record_event(
            "born_from",
            details={"parent_id": self.organism_id}
        )

        # Trigger callbacks
        for callback in self._callbacks["reproduction"]:
            callback(self, offspring)

        return offspring

    def terminate(self, cause: DeathCause = DeathCause.TERMINATED):
        """Forcefully terminate the organism"""
        asyncio.create_task(self._die(cause))

    def _record_event(
        self,
        event_type: str,
        stage_from: Optional[LifecycleStage] = None,
        stage_to: Optional[LifecycleStage] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a lifecycle event"""
        event = LifecycleEvent(
            event_type=event_type,
            stage_from=stage_from,
            stage_to=stage_to,
            details=details or {}
        )
        self.events.append(event)

    def get_vitality(self) -> float:
        """
        Get current vitality score (0-1).

        Combines consciousness level, stage, and age factors.
        """
        base = self.consciousness.overall_consciousness_score()

        # Stage multiplier
        stage_multipliers = {
            LifecycleStage.EMBRYONIC: 0.5,
            LifecycleStage.NASCENT: 0.7,
            LifecycleStage.JUVENILE: 0.85,
            LifecycleStage.ADOLESCENT: 0.95,
            LifecycleStage.MATURE: 1.0,
            LifecycleStage.SENIOR: 0.85,
            LifecycleStage.SENESCENT: 0.6,
            LifecycleStage.DYING: 0.3,
            LifecycleStage.DEAD: 0.0
        }

        multiplier = stage_multipliers.get(self.stage, 0.5)

        return base * multiplier

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            "organism_id": self.organism_id,
            "stage": self.stage.value,
            "age": self.age,
            "vitality": self.get_vitality(),
            "consciousness_score": self.consciousness.overall_consciousness_score(),
            "size": self.metrics.size,
            "complexity": self.metrics.complexity,
            "can_reproduce": self.can_reproduce(),
            "is_alive": self.stage != LifecycleStage.DEAD,
            "lifespan": time.time() - self.birth_time if self.death_time is None else self.death_time - self.birth_time,
            "events_count": len(self.events)
        }


# ============================================================================
# Population Manager
# ============================================================================

class PopulationManager:
    """
    Manage a population of organisms.

    Handles births, deaths, and population dynamics.
    """

    def __init__(self, carrying_capacity: int = 100):
        self.organisms: Dict[str, OrganismLifecycle] = {}
        self.carrying_capacity = carrying_capacity
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0

    def add_organism(self, organism: OrganismLifecycle):
        """Add organism to population"""
        self.organisms[organism.organism_id] = organism
        self.total_births += 1

        # Register death callback
        organism.on_death(self._handle_death)

    def _handle_death(self, organism: OrganismLifecycle, cause: DeathCause):
        """Handle organism death"""
        if organism.organism_id in self.organisms:
            del self.organisms[organism.organism_id]
            self.total_deaths += 1

    async def tick(self):
        """Process one population cycle"""
        # Update all organisms
        for organism in list(self.organisms.values()):
            await organism.tick()

        # Handle reproduction
        if len(self.organisms) < self.carrying_capacity:
            for organism in list(self.organisms.values()):
                if organism.can_reproduce() and len(self.organisms) < self.carrying_capacity:
                    offspring = organism.reproduce()
                    if offspring:
                        self.add_organism(offspring)

        # Update generation counter
        self.generation += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics"""
        if not self.organisms:
            return {
                "population": 0,
                "generation": self.generation,
                "total_births": self.total_births,
                "total_deaths": self.total_deaths
            }

        stages = defaultdict(int)
        vitalities = []
        ages = []

        for organism in self.organisms.values():
            stages[organism.stage.value] += 1
            vitalities.append(organism.get_vitality())
            ages.append(organism.age)

        return {
            "population": len(self.organisms),
            "carrying_capacity": self.carrying_capacity,
            "generation": self.generation,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "stage_distribution": dict(stages),
            "average_vitality": np.mean(vitalities),
            "average_age": np.mean(ages),
            "oldest_age": max(ages),
            "youngest_age": min(ages)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("Organism Lifecycle Demo")
        print("=" * 50)

        # Create organism
        consciousness = GradedConsciousness(
            perception_fidelity=0.5,
            reaction_speed=0.5,
            memory_depth=0.5,
            introspection_capacity=0.5
        )

        config = LifecycleConfig(
            embryonic_duration=5,
            nascent_duration=10,
            juvenile_duration=20,
            adolescent_duration=15,
            mature_duration=50,
            senior_duration=15,
            senescent_duration=10,
            growth_pattern=GrowthPattern.SIGMOIDAL
        )

        lifecycle = OrganismLifecycle(consciousness, config)

        # Track stage changes
        stage_changes = []
        lifecycle.on_stage_change(lambda old, new: stage_changes.append((old, new)))

        print(f"\n1. Initial state: {lifecycle.stage.value}")
        print(f"   Vitality: {lifecycle.get_vitality():.3f}")

        # Run lifecycle
        print("\n2. Running lifecycle simulation...")
        for i in range(150):
            await lifecycle.tick()

            if i % 25 == 0:
                status = lifecycle.get_status()
                print(f"   Age {status['age']}: {status['stage']} "
                      f"(vitality: {status['vitality']:.3f}, size: {status['size']:.1f})")

        print(f"\n3. Stage transitions: {len(stage_changes)}")
        for old, new in stage_changes:
            print(f"   {old.value} -> {new.value}")

        # Test reproduction
        print("\n4. Testing reproduction...")
        if lifecycle.can_reproduce():
            offspring = lifecycle.reproduce()
            if offspring:
                print(f"   Created offspring: {offspring.organism_id[:8]}...")
                print(f"   Offspring vitality: {offspring.get_vitality():.3f}")
                print(f"   Parent vitality after: {lifecycle.get_vitality():.3f}")
        else:
            print(f"   Cannot reproduce (stage: {lifecycle.stage.value})")

        # Population simulation
        print("\n5. Population simulation...")
        population = PopulationManager(carrying_capacity=20)

        # Seed population
        for _ in range(5):
            c = GradedConsciousness(
                perception_fidelity=0.4 + np.random.uniform(0, 0.2),
                reaction_speed=0.4 + np.random.uniform(0, 0.2)
            )
            population.add_organism(OrganismLifecycle(c, config))

        for gen in range(100):
            await population.tick()

            if gen % 20 == 0:
                stats = population.get_statistics()
                print(f"   Gen {gen}: pop={stats['population']}, "
                      f"avg_vitality={stats.get('average_vitality', 0):.3f}")

        final_stats = population.get_statistics()
        print(f"\n6. Final population stats:")
        for key, value in final_stats.items():
            if not isinstance(value, dict):
                print(f"   {key}: {value}")

    asyncio.run(main())
