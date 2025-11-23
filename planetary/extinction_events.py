"""
Mass Extinction Events

Implements modeling and simulation of mass extinction events and
their effects on planetary consciousness. Includes causes, impacts,
recovery dynamics, and consciousness survival strategies.

Features:
- Multiple extinction cause modeling
- Severity and selectivity simulation
- Recovery trajectory prediction
- Consciousness preservation during extinction
- Post-extinction radiation dynamics
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class ExtinctionCause(Enum):
    """Causes of mass extinction"""
    CLIMATE_CHANGE = "climate_change"
    ASTEROID_IMPACT = "asteroid_impact"
    VOLCANIC = "volcanic"
    ANOXIA = "anoxia"
    DISEASE = "disease"
    RESOURCE_DEPLETION = "resource_depletion"
    CONSCIOUSNESS_COLLAPSE = "consciousness_collapse"
    EXTERNAL_INTERVENTION = "external_intervention"
    CASCADE_FAILURE = "cascade_failure"


class ExtinctionSeverity(Enum):
    """Severity levels of extinction"""
    MINOR = "minor"              # < 25% loss
    MODERATE = "moderate"        # 25-50% loss
    MAJOR = "major"              # 50-75% loss
    MASS = "mass"                # 75-90% loss
    CATASTROPHIC = "catastrophic"  # > 90% loss


class RecoveryPhase(Enum):
    """Phases of post-extinction recovery"""
    CRISIS = "crisis"            # Active extinction
    SURVIVAL = "survival"        # Immediate aftermath
    STABILIZATION = "stabilization"  # Basic stability
    DIVERSIFICATION = "diversification"  # Species radiation
    RESTORED = "restored"        # Full recovery


class SelectivityPattern(Enum):
    """Patterns of extinction selectivity"""
    RANDOM = "random"            # No pattern
    SIZE_BIASED = "size_biased"  # Large organisms first
    SPECIALIST = "specialist"    # Specialists vulnerable
    GENERALIST = "generalist"    # Generalists vulnerable
    CONSCIOUSNESS_BIASED = "consciousness_biased"  # Low consciousness
    GEOGRAPHIC = "geographic"    # Location-based


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ExtinctionEvent:
    """A mass extinction event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    cause: ExtinctionCause = ExtinctionCause.CLIMATE_CHANGE
    severity: ExtinctionSeverity = ExtinctionSeverity.MODERATE
    selectivity: SelectivityPattern = SelectivityPattern.RANDOM

    # Timing
    start_tick: int = 0
    peak_tick: Optional[int] = None
    end_tick: Optional[int] = None
    duration: int = 100

    # Impact
    species_lost: int = 0
    population_lost: int = 0
    peak_loss_rate: float = 0.0
    consciousness_impact: float = 0.0

    # State
    active: bool = False
    phase: RecoveryPhase = RecoveryPhase.CRISIS

    # Metrics
    pre_event_biodiversity: float = 1.0
    current_biodiversity: float = 1.0
    recovery_progress: float = 0.0


@dataclass
class Species:
    """A species for extinction modeling"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    population: int = 1000
    max_population: int = 10000

    # Vulnerability traits
    body_size: float = 0.5        # 0=small, 1=large
    specialization: float = 0.5   # 0=generalist, 1=specialist
    consciousness_level: float = 0.5
    geographic_range: float = 0.5  # 0=narrow, 1=wide
    reproductive_rate: float = 0.5

    # State
    extinct: bool = False
    extinction_risk: float = 0.0


@dataclass
class ExtinctionConfig:
    """Configuration for extinction modeling"""
    base_extinction_rate: float = 0.001
    recovery_rate: float = 0.01
    consciousness_preservation_bonus: float = 0.3
    min_viable_population: int = 50
    radiation_delay: int = 50


# ============================================================================
# Extinction System
# ============================================================================

class ExtinctionSystem:
    """
    Mass extinction event modeling.

    Simulates the causes, progression, and recovery from
    mass extinction events, with special handling for
    consciousness preservation.

    Example:
        extinction = ExtinctionSystem()
        extinction.create_species(100)

        # Trigger extinction
        event = extinction.trigger_extinction(
            cause=ExtinctionCause.CLIMATE_CHANGE,
            severity=ExtinctionSeverity.MAJOR
        )

        # Simulate
        for _ in range(500):
            extinction.tick()
    """

    def __init__(self, config: Optional[ExtinctionConfig] = None):
        self.config = config or ExtinctionConfig()

        # Species
        self.species: Dict[str, Species] = {}

        # Extinction events
        self.active_events: List[ExtinctionEvent] = []
        self.historical_events: List[ExtinctionEvent] = []

        # Global state
        self.total_biodiversity: float = 1.0
        self.total_population: int = 0
        self.total_consciousness: float = 0.0

        # Recovery tracking
        self.recovery_phase: RecoveryPhase = RecoveryPhase.RESTORED
        self.time_since_extinction: int = 0

        # History
        self.tick_count = 0
        self.biodiversity_history: List[float] = []
        self.population_history: List[int] = []

    def create_species(self, count: int):
        """Create a set of species"""
        names = [
            "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
            "Zeta", "Eta", "Theta", "Iota", "Kappa"
        ]

        for i in range(count):
            name = f"{names[i % len(names)]}_{i // len(names)}"

            species = Species(
                name=name,
                population=np.random.randint(100, 5000),
                max_population=np.random.randint(5000, 20000),
                body_size=np.random.random(),
                specialization=np.random.random(),
                consciousness_level=np.random.uniform(0.2, 0.8),
                geographic_range=np.random.random(),
                reproductive_rate=np.random.uniform(0.1, 0.5)
            )

            self.species[species.id] = species

        self._update_global_metrics()

    def trigger_extinction(
        self,
        cause: ExtinctionCause,
        severity: ExtinctionSeverity,
        selectivity: SelectivityPattern = SelectivityPattern.RANDOM,
        duration: int = 100
    ) -> ExtinctionEvent:
        """Trigger a mass extinction event"""
        event = ExtinctionEvent(
            name=f"{cause.value}_{self.tick_count}",
            cause=cause,
            severity=severity,
            selectivity=selectivity,
            start_tick=self.tick_count,
            duration=duration,
            active=True,
            phase=RecoveryPhase.CRISIS,
            pre_event_biodiversity=self.total_biodiversity
        )

        # Set peak tick
        event.peak_tick = self.tick_count + duration // 3

        self.active_events.append(event)
        self.recovery_phase = RecoveryPhase.CRISIS

        return event

    def tick(self):
        """Process one extinction cycle"""
        self.tick_count += 1

        # Process active extinctions
        for event in self.active_events:
            self._process_extinction(event)

        # Background extinction
        self._background_extinction()

        # Recovery for completed events
        if not self.active_events and self.recovery_phase != RecoveryPhase.RESTORED:
            self._process_recovery()

        # Update metrics
        self._update_global_metrics()
        self._record_history()

        # Complete finished events
        self._finalize_events()

    def _process_extinction(self, event: ExtinctionEvent):
        """Process an active extinction event"""
        elapsed = self.tick_count - event.start_tick

        # Calculate intensity
        # Peaks at 1/3 of duration, then declines
        if elapsed < event.duration // 3:
            intensity = elapsed / (event.duration // 3)
        else:
            remaining = event.duration - elapsed
            intensity = remaining / (event.duration * 2 / 3)

        intensity = max(0, min(1, intensity))

        # Base loss rate from severity
        severity_rates = {
            ExtinctionSeverity.MINOR: 0.01,
            ExtinctionSeverity.MODERATE: 0.03,
            ExtinctionSeverity.MAJOR: 0.05,
            ExtinctionSeverity.MASS: 0.08,
            ExtinctionSeverity.CATASTROPHIC: 0.15
        }
        base_rate = severity_rates[event.severity] * intensity

        event.peak_loss_rate = max(event.peak_loss_rate, base_rate)

        # Apply to each species
        for species in self.species.values():
            if species.extinct:
                continue

            # Calculate vulnerability
            vulnerability = self._calculate_vulnerability(species, event)

            # Apply losses
            loss_rate = base_rate * vulnerability
            losses = int(species.population * loss_rate)
            species.population = max(0, species.population - losses)
            event.population_lost += losses

            # Check extinction
            if species.population < self.config.min_viable_population:
                species.extinct = True
                species.population = 0
                event.species_lost += 1

        # Update event biodiversity
        living_species = [s for s in self.species.values() if not s.extinct]
        event.current_biodiversity = len(living_species) / len(self.species) if self.species else 0

        # Check if event is ending
        if elapsed >= event.duration:
            event.active = False
            event.end_tick = self.tick_count
            event.phase = RecoveryPhase.SURVIVAL

    def _calculate_vulnerability(self, species: Species, event: ExtinctionEvent) -> float:
        """Calculate species vulnerability to extinction"""
        base_vulnerability = 1.0

        # Selectivity patterns
        if event.selectivity == SelectivityPattern.SIZE_BIASED:
            base_vulnerability *= (0.5 + species.body_size)

        elif event.selectivity == SelectivityPattern.SPECIALIST:
            base_vulnerability *= (0.5 + species.specialization)

        elif event.selectivity == SelectivityPattern.GENERALIST:
            base_vulnerability *= (1.5 - species.specialization)

        elif event.selectivity == SelectivityPattern.CONSCIOUSNESS_BIASED:
            # Low consciousness more vulnerable
            base_vulnerability *= (1.5 - species.consciousness_level)

        elif event.selectivity == SelectivityPattern.GEOGRAPHIC:
            # Narrow range more vulnerable
            base_vulnerability *= (1.5 - species.geographic_range)

        # Consciousness preservation bonus
        consciousness_protection = (
            species.consciousness_level *
            self.config.consciousness_preservation_bonus
        )
        base_vulnerability *= (1.0 - consciousness_protection)

        # Geographic range protection
        base_vulnerability *= (1.0 - species.geographic_range * 0.3)

        return max(0.1, min(2.0, base_vulnerability))

    def _background_extinction(self):
        """Apply background extinction rate"""
        for species in self.species.values():
            if species.extinct:
                continue

            if np.random.random() < self.config.base_extinction_rate:
                losses = int(species.population * 0.01)
                species.population = max(0, species.population - losses)

    def _process_recovery(self):
        """Process post-extinction recovery"""
        self.time_since_extinction += 1

        living_species = [s for s in self.species.values() if not s.extinct]

        if not living_species:
            return

        # Determine recovery phase
        initial_diversity = len(self.species)
        current_diversity = len(living_species)
        diversity_ratio = current_diversity / initial_diversity if initial_diversity > 0 else 0

        if self.time_since_extinction < 20:
            self.recovery_phase = RecoveryPhase.SURVIVAL
        elif diversity_ratio < 0.5:
            self.recovery_phase = RecoveryPhase.STABILIZATION
        elif self.time_since_extinction < self.config.radiation_delay:
            self.recovery_phase = RecoveryPhase.DIVERSIFICATION
        else:
            self.recovery_phase = RecoveryPhase.RESTORED

        # Population recovery
        for species in living_species:
            if species.population < species.max_population:
                growth = int(
                    species.population *
                    species.reproductive_rate *
                    self.config.recovery_rate
                )
                species.population = min(species.max_population, species.population + growth)

    def _update_global_metrics(self):
        """Update global metrics"""
        living = [s for s in self.species.values() if not s.extinct]

        self.total_population = sum(s.population for s in living)
        self.total_biodiversity = len(living) / len(self.species) if self.species else 0

        if living:
            self.total_consciousness = np.mean([
                s.consciousness_level * s.population
                for s in living
            ]) / max(1, self.total_population)
        else:
            self.total_consciousness = 0.0

    def _record_history(self):
        """Record historical data"""
        self.biodiversity_history.append(self.total_biodiversity)
        self.population_history.append(self.total_population)

        # Trim
        if len(self.biodiversity_history) > 1000:
            self.biodiversity_history = self.biodiversity_history[-1000:]
            self.population_history = self.population_history[-1000:]

    def _finalize_events(self):
        """Finalize completed events"""
        completed = [e for e in self.active_events if not e.active]

        for event in completed:
            # Calculate consciousness impact
            if event.pre_event_biodiversity > 0:
                loss_fraction = 1.0 - (event.current_biodiversity / event.pre_event_biodiversity)
                event.consciousness_impact = loss_fraction * 0.8

            self.active_events.remove(event)
            self.historical_events.append(event)
            self.time_since_extinction = 0

    # ========================================================================
    # Public API
    # ========================================================================

    def get_extinction_risk(self, species_id: str) -> float:
        """Get extinction risk for a species"""
        species = self.species.get(species_id)
        if not species or species.extinct:
            return 0.0

        # Base risk from population size
        pop_risk = 1.0 - min(1.0, species.population / species.max_population)

        # Risk from traits
        trait_risk = (
            species.specialization * 0.3 +
            species.body_size * 0.2 +
            (1.0 - species.geographic_range) * 0.2 +
            (1.0 - species.consciousness_level) * 0.2
        )

        # Active event risk
        event_risk = 0.0
        for event in self.active_events:
            vulnerability = self._calculate_vulnerability(species, event)
            event_risk += vulnerability * 0.3

        return min(1.0, pop_risk * 0.4 + trait_risk * 0.3 + event_risk * 0.3)

    def get_species_status(self) -> Dict[str, Any]:
        """Get status of all species"""
        living = [s for s in self.species.values() if not s.extinct]
        extinct = [s for s in self.species.values() if s.extinct]

        return {
            "total_species": len(self.species),
            "living_species": len(living),
            "extinct_species": len(extinct),
            "extinction_rate": len(extinct) / len(self.species) if self.species else 0,
            "total_population": self.total_population,
            "biodiversity_index": self.total_biodiversity,
            "consciousness_average": self.total_consciousness
        }

    def get_event_history(self) -> List[Dict[str, Any]]:
        """Get history of extinction events"""
        return [
            {
                "name": e.name,
                "cause": e.cause.value,
                "severity": e.severity.value,
                "species_lost": e.species_lost,
                "population_lost": e.population_lost,
                "consciousness_impact": e.consciousness_impact,
                "duration": e.end_tick - e.start_tick if e.end_tick else 0
            }
            for e in self.historical_events
        ]

    def get_global_status(self) -> Dict[str, Any]:
        """Get global extinction status"""
        return {
            "tick_count": self.tick_count,
            "active_events": len(self.active_events),
            "historical_events": len(self.historical_events),
            "recovery_phase": self.recovery_phase.value,
            "time_since_extinction": self.time_since_extinction,
            "biodiversity": self.total_biodiversity,
            "total_population": self.total_population,
            "consciousness": self.total_consciousness
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Mass Extinction Events Demo")
    print("=" * 50)

    extinction = ExtinctionSystem()
    extinction.create_species(50)

    print(f"\n1. Created {len(extinction.species)} species")
    status = extinction.get_species_status()
    print(f"   Initial population: {status['total_population']}")
    print(f"   Biodiversity: {status['biodiversity_index']:.3f}")

    # Run baseline
    print("\n2. Running baseline simulation...")
    for i in range(50):
        extinction.tick()

    print(f"   Population after 50 ticks: {extinction.total_population}")

    # Trigger extinction
    print("\n3. Triggering major extinction event...")
    event = extinction.trigger_extinction(
        cause=ExtinctionCause.CLIMATE_CHANGE,
        severity=ExtinctionSeverity.MAJOR,
        selectivity=SelectivityPattern.SPECIALIST,
        duration=100
    )

    # Simulate extinction
    print("\n4. Simulating extinction event...")
    for i in range(150):
        extinction.tick()

        if i % 30 == 0:
            status = extinction.get_species_status()
            global_status = extinction.get_global_status()
            print(f"   Tick {i}: living={status['living_species']}, "
                  f"extinct={status['extinct_species']}, "
                  f"phase={global_status['recovery_phase']}")

    # Final status
    print("\n5. Final status:")
    status = extinction.get_species_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    print("\n6. Event summary:")
    history = extinction.get_event_history()
    for event in history:
        print(f"   {event['name']}: {event['species_lost']} species lost, "
              f"consciousness impact: {event['consciousness_impact']:.3f}")
