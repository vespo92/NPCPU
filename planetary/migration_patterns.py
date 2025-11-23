"""
Population Migration Patterns

Implements planetary-scale population migration modeling for
digital organisms. Tracks movement patterns, distribution,
and consciousness flow across regions.

Features:
- Multi-species migration modeling
- Resource-driven migration
- Climate-responsive movement
- Consciousness gradient following
- Population distribution optimization
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
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

class MigrationDriver(Enum):
    """Drivers of migration behavior"""
    RESOURCE_SEEKING = "resource_seeking"
    CLIMATE_ESCAPE = "climate_escape"
    CONSCIOUSNESS_GRADIENT = "consciousness_gradient"
    SOCIAL_ATTRACTION = "social_attraction"
    TERRITORIAL = "territorial"
    SEASONAL = "seasonal"
    REPRODUCTIVE = "reproductive"
    RANDOM_DISPERSAL = "random_dispersal"


class MigrationPattern(Enum):
    """Types of migration patterns"""
    DIFFUSION = "diffusion"              # Random spread
    DIRECTED = "directed"                 # Toward specific target
    WAVE = "wave"                         # Expanding wavefront
    SEASONAL_CIRCUIT = "seasonal_circuit" # Annual cycle
    CHAIN = "chain"                       # Sequential region movement
    LEAP_FROG = "leap_frog"              # Skip intermediate regions


class RegionAttractivness(Enum):
    """Factors affecting region attractiveness"""
    RESOURCES = "resources"
    CLIMATE = "climate"
    SAFETY = "safety"
    CONSCIOUSNESS_LEVEL = "consciousness_level"
    POPULATION_DENSITY = "population_density"
    SOCIAL_CONNECTIONS = "social_connections"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MigrationRegion:
    """A region for migration tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    position: Tuple[float, float] = (0.0, 0.0)  # x, y coordinates
    area: float = 1000.0
    capacity: int = 1000

    # Current state
    population: int = 0
    resource_level: float = 1.0
    climate_suitability: float = 1.0
    consciousness_level: float = 0.5
    danger_level: float = 0.0

    # Connectivity
    connected_regions: Set[str] = field(default_factory=set)
    migration_barriers: Dict[str, float] = field(default_factory=dict)


@dataclass
class MigrationFlow:
    """A flow of migrants between regions"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    volume: int = 0
    driver: MigrationDriver = MigrationDriver.RESOURCE_SEEKING
    started_at: int = 0
    duration: int = 1


@dataclass
class SpeciesMigration:
    """Migration characteristics for a species"""
    species_id: str = ""
    mobility: float = 0.5          # Movement capability
    range_size: float = 100.0      # Territory size
    social_tendency: float = 0.5    # Preference for groups
    resource_dependence: float = 0.7
    climate_sensitivity: float = 0.5
    preferred_pattern: MigrationPattern = MigrationPattern.DIFFUSION


@dataclass
class MigrationConfig:
    """Configuration for migration system"""
    base_migration_rate: float = 0.1
    distance_decay: float = 0.5      # Effect of distance on migration
    capacity_pressure: float = 0.3    # Push from overcrowding
    resource_attraction: float = 0.4  # Pull of resources
    consciousness_weight: float = 0.2 # Weight of consciousness gradient


# ============================================================================
# Migration System
# ============================================================================

class MigrationSystem:
    """
    Planetary-scale population migration system.

    Models organism movement across regions based on:
    - Resource availability
    - Climate conditions
    - Population pressure
    - Consciousness gradients
    - Social connections

    Example:
        migration = MigrationSystem()
        migration.create_region_network(regions)

        for _ in range(100):
            migration.simulate_tick()

        flows = migration.get_current_flows()
    """

    def __init__(self, config: Optional[MigrationConfig] = None):
        self.config = config or MigrationConfig()

        # Regions
        self.regions: Dict[str, MigrationRegion] = {}

        # Species
        self.species_migrations: Dict[str, SpeciesMigration] = {}

        # Population tracking
        self.populations: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Active flows
        self.active_flows: List[MigrationFlow] = []
        self.flow_history: List[MigrationFlow] = []

        # Tracking
        self.tick_count = 0
        self.total_migrations = 0

        # Statistics
        self.distribution_history: List[Dict[str, int]] = []

    def add_region(self, region: MigrationRegion):
        """Add a region to the network"""
        self.regions[region.id] = region

    def create_region_network(self, num_regions: int = 10):
        """Create a network of regions"""
        # Create regions in a grid-like pattern
        grid_size = int(np.ceil(np.sqrt(num_regions)))

        region_names = [
            "Northern Plains", "Eastern Forests", "Southern Coast",
            "Western Highlands", "Central Valley", "Mountain Range",
            "Desert Basin", "Tropical Zone", "Polar Region", "Island Chain"
        ]

        for i in range(min(num_regions, len(region_names))):
            x = (i % grid_size) * 100
            y = (i // grid_size) * 100

            region = MigrationRegion(
                name=region_names[i],
                position=(float(x), float(y)),
                capacity=500 + np.random.randint(0, 500),
                population=np.random.randint(50, 200),
                resource_level=np.random.uniform(0.5, 1.0),
                climate_suitability=np.random.uniform(0.4, 1.0),
                consciousness_level=np.random.uniform(0.2, 0.6)
            )

            self.add_region(region)

        # Connect nearby regions
        self._connect_regions()

    def _connect_regions(self):
        """Connect nearby regions"""
        regions = list(self.regions.values())

        for i, region in enumerate(regions):
            for j, other in enumerate(regions):
                if i >= j:
                    continue

                # Calculate distance
                dist = np.sqrt(
                    (region.position[0] - other.position[0]) ** 2 +
                    (region.position[1] - other.position[1]) ** 2
                )

                # Connect if within range
                if dist < 150:  # Connection threshold
                    region.connected_regions.add(other.id)
                    other.connected_regions.add(region.id)

                    # Some barriers
                    if np.random.random() < 0.2:
                        barrier = np.random.uniform(0.1, 0.3)
                        region.migration_barriers[other.id] = barrier
                        other.migration_barriers[region.id] = barrier

    def add_species(self, species: SpeciesMigration):
        """Add a species to track"""
        self.species_migrations[species.species_id] = species

    def set_population(self, region_id: str, species_id: str, count: int):
        """Set population of a species in a region"""
        if region_id in self.regions:
            self.populations[region_id][species_id] = count
            self.regions[region_id].population = sum(self.populations[region_id].values())

    def simulate_tick(self):
        """Simulate one migration tick"""
        self.tick_count += 1

        # Calculate attractiveness for each region
        attractiveness = self._calculate_attractiveness()

        # Generate migration flows
        new_flows = self._generate_flows(attractiveness)

        # Process active flows
        self._process_flows()

        # Add new flows
        self.active_flows.extend(new_flows)

        # Update region populations
        self._update_populations()

        # Record distribution
        self._record_distribution()

    def _calculate_attractiveness(self) -> Dict[str, float]:
        """Calculate attractiveness score for each region"""
        attractiveness = {}

        for region in self.regions.values():
            # Base attractiveness from resources and climate
            base = (
                region.resource_level * self.config.resource_attraction +
                region.climate_suitability * 0.3
            )

            # Population pressure (less attractive when crowded)
            density = region.population / region.capacity
            pressure = max(0, 1.0 - density * self.config.capacity_pressure)

            # Consciousness contribution
            consciousness = region.consciousness_level * self.config.consciousness_weight

            # Safety
            safety = 1.0 - region.danger_level

            attractiveness[region.id] = base * pressure * safety + consciousness

        return attractiveness

    def _generate_flows(self, attractiveness: Dict[str, float]) -> List[MigrationFlow]:
        """Generate migration flows based on conditions"""
        flows = []

        for region in self.regions.values():
            # Skip if no population
            if region.population == 0:
                continue

            # Calculate push factors
            density = region.population / region.capacity
            push = density * self.config.capacity_pressure

            # Low resources push migration
            push += (1.0 - region.resource_level) * 0.2

            # Consider each connected region
            for target_id in region.connected_regions:
                target = self.regions.get(target_id)
                if not target:
                    continue

                # Calculate pull
                pull = attractiveness.get(target_id, 0.0)

                # Apply distance decay
                dist = np.sqrt(
                    (region.position[0] - target.position[0]) ** 2 +
                    (region.position[1] - target.position[1]) ** 2
                )
                distance_factor = np.exp(-dist * self.config.distance_decay / 100)

                # Apply barriers
                barrier = region.migration_barriers.get(target_id, 0.0)
                barrier_factor = 1.0 - barrier

                # Calculate net migration probability
                net_pressure = (pull - attractiveness.get(region.id, 0.5)) * push
                migration_prob = max(0, net_pressure) * distance_factor * barrier_factor

                # Determine volume
                if migration_prob > 0 and np.random.random() < migration_prob:
                    volume = int(region.population * self.config.base_migration_rate * migration_prob)
                    volume = min(volume, region.population // 2)  # Cap at half population

                    if volume > 0:
                        # Determine driver
                        if region.resource_level < 0.3:
                            driver = MigrationDriver.RESOURCE_SEEKING
                        elif region.climate_suitability < 0.3:
                            driver = MigrationDriver.CLIMATE_ESCAPE
                        elif target.consciousness_level > region.consciousness_level + 0.2:
                            driver = MigrationDriver.CONSCIOUSNESS_GRADIENT
                        else:
                            driver = MigrationDriver.RANDOM_DISPERSAL

                        flow = MigrationFlow(
                            source_id=region.id,
                            target_id=target_id,
                            volume=volume,
                            driver=driver,
                            started_at=self.tick_count,
                            duration=max(1, int(dist / 50))
                        )
                        flows.append(flow)

        return flows

    def _process_flows(self):
        """Process active migration flows"""
        completed = []

        for flow in self.active_flows:
            elapsed = self.tick_count - flow.started_at
            if elapsed >= flow.duration:
                completed.append(flow)
                self._complete_flow(flow)

        for flow in completed:
            self.active_flows.remove(flow)
            self.flow_history.append(flow)

    def _complete_flow(self, flow: MigrationFlow):
        """Complete a migration flow"""
        source = self.regions.get(flow.source_id)
        target = self.regions.get(flow.target_id)

        if not source or not target:
            return

        # Transfer population
        actual_volume = min(flow.volume, source.population)
        source.population -= actual_volume
        target.population += actual_volume

        self.total_migrations += actual_volume

    def _update_populations(self):
        """Update population totals"""
        for region in self.regions.values():
            # Apply natural changes
            if region.population > region.capacity:
                # Overcrowding mortality
                excess = region.population - region.capacity
                mortality = int(excess * 0.1)
                region.population = max(0, region.population - mortality)

            # Resource regeneration
            if region.population < region.capacity:
                region.resource_level = min(1.0, region.resource_level + 0.01)
            else:
                region.resource_level = max(0.1, region.resource_level - 0.02)

    def _record_distribution(self):
        """Record population distribution"""
        distribution = {
            region.name: region.population
            for region in self.regions.values()
        }
        self.distribution_history.append(distribution)

        # Trim history
        if len(self.distribution_history) > 500:
            self.distribution_history = self.distribution_history[-500:]

    # ========================================================================
    # Public API
    # ========================================================================

    def get_region_status(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a region"""
        region = self.regions.get(region_id)
        if not region:
            return None

        return {
            "name": region.name,
            "population": region.population,
            "capacity": region.capacity,
            "density": region.population / region.capacity,
            "resources": region.resource_level,
            "climate": region.climate_suitability,
            "consciousness": region.consciousness_level,
            "connections": len(region.connected_regions)
        }

    def get_current_flows(self) -> List[Dict[str, Any]]:
        """Get current migration flows"""
        return [
            {
                "source": self.regions[f.source_id].name if f.source_id in self.regions else "",
                "target": self.regions[f.target_id].name if f.target_id in self.regions else "",
                "volume": f.volume,
                "driver": f.driver.value,
                "progress": (self.tick_count - f.started_at) / f.duration
            }
            for f in self.active_flows
        ]

    def get_global_distribution(self) -> Dict[str, Any]:
        """Get global population distribution"""
        total = sum(r.population for r in self.regions.values())
        total_capacity = sum(r.capacity for r in self.regions.values())

        return {
            "tick_count": self.tick_count,
            "total_population": total,
            "total_capacity": total_capacity,
            "global_density": total / total_capacity if total_capacity > 0 else 0,
            "regions": len(self.regions),
            "active_flows": len(self.active_flows),
            "total_migrations": self.total_migrations,
            "distribution": {
                r.name: r.population
                for r in self.regions.values()
            }
        }

    def get_migration_statistics(self) -> Dict[str, Any]:
        """Get migration statistics"""
        driver_counts = defaultdict(int)
        for flow in self.flow_history[-100:]:
            driver_counts[flow.driver.value] += flow.volume

        return {
            "total_migrations": self.total_migrations,
            "recent_flows": len(self.flow_history[-100:]),
            "driver_breakdown": dict(driver_counts),
            "average_flow_volume": (
                np.mean([f.volume for f in self.flow_history[-100:]])
                if self.flow_history else 0
            )
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Migration Patterns Demo")
    print("=" * 50)

    migration = MigrationSystem()
    migration.create_region_network(8)

    print(f"\n1. Created region network with {len(migration.regions)} regions")

    print("\n2. Regions:")
    for region in migration.regions.values():
        print(f"   - {region.name}: pop={region.population}, "
              f"capacity={region.capacity}, "
              f"connections={len(region.connected_regions)}")

    # Run simulation
    print("\n3. Running migration simulation...")
    for i in range(100):
        migration.simulate_tick()

        if i % 25 == 0:
            dist = migration.get_global_distribution()
            print(f"   Tick {i}: total_pop={dist['total_population']}, "
                  f"active_flows={dist['active_flows']}, "
                  f"migrations={dist['total_migrations']}")

    # Final distribution
    print("\n4. Final distribution:")
    dist = migration.get_global_distribution()
    for name, pop in sorted(dist["distribution"].items(), key=lambda x: -x[1]):
        print(f"   {name}: {pop}")

    # Statistics
    print("\n5. Migration statistics:")
    stats = migration.get_migration_statistics()
    print(f"   Total migrations: {stats['total_migrations']}")
    print(f"   Driver breakdown: {stats['driver_breakdown']}")
