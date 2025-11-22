"""
World Simulation Environment

A complete simulated world for digital organisms to inhabit:
- Multiple regions with different conditions
- Global resource pools
- World events and dynamics
- Day/night cycles
- Seasons and climate
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ecosystem.population import Population, InteractionType


# ============================================================================
# Enums
# ============================================================================

class RegionType(Enum):
    """Types of world regions"""
    CORE = "core"                    # Central, stable region
    FERTILE = "fertile"              # Resource-rich
    BARREN = "barren"                # Few resources
    FRONTIER = "frontier"            # Edge territory
    HOSTILE = "hostile"              # Dangerous
    SANCTUARY = "sanctuary"          # Protected area


class Climate(Enum):
    """Climate conditions"""
    OPTIMAL = "optimal"
    WARM = "warm"
    COLD = "cold"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    VOLATILE = "volatile"


class Season(Enum):
    """Seasonal cycles"""
    SPRING = "spring"                # Growth, renewal
    SUMMER = "summer"                # Abundance
    AUTUMN = "autumn"                # Harvest, decline
    WINTER = "winter"                # Scarcity, dormancy


class WorldEventType(Enum):
    """Types of world events"""
    RESOURCE_SURGE = "resource_surge"
    RESOURCE_DEPLETION = "resource_depletion"
    DISASTER = "disaster"
    MIGRATION = "migration"
    EVOLUTION_CATALYST = "evolution_catalyst"
    ENVIRONMENTAL_SHIFT = "environmental_shift"
    EXTINCTION_EVENT = "extinction_event"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ResourcePool:
    """A pool of resources in the world"""
    name: str
    amount: float = 1000.0
    max_capacity: float = 5000.0
    regeneration_rate: float = 10.0
    consumption_rate: float = 0.0
    quality: float = 1.0             # Resource quality modifier

    def consume(self, amount: float) -> float:
        """Consume resources from pool"""
        actual = min(amount, self.amount)
        self.amount -= actual
        self.consumption_rate += actual
        return actual * self.quality

    def regenerate(self):
        """Regenerate resources"""
        self.amount = min(
            self.max_capacity,
            self.amount + self.regeneration_rate
        )


@dataclass
class Region:
    """A region in the world"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: RegionType = RegionType.CORE
    climate: Climate = Climate.OPTIMAL
    resources: Dict[str, ResourcePool] = field(default_factory=dict)
    danger_level: float = 0.0        # 0-1
    habitability: float = 1.0        # 0-1
    population_ids: Set[str] = field(default_factory=set)
    neighbors: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldEvent:
    """A world-level event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: WorldEventType = WorldEventType.RESOURCE_SURGE
    name: str = ""
    magnitude: float = 0.5
    affected_regions: List[str] = field(default_factory=list)
    duration: int = 10               # Ticks
    started_at: int = 0
    effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldConfig:
    """Configuration for the world"""
    name: str = "Digital World"
    base_resource_rate: float = 10.0
    season_length: int = 100         # Ticks per season
    day_length: int = 24             # Ticks per day
    event_frequency: float = 0.05    # Probability per tick
    migration_enabled: bool = True
    evolution_rate: float = 0.01


# ============================================================================
# World
# ============================================================================

class World:
    """
    Complete world simulation for digital organisms.

    Features:
    - Multiple regions with unique conditions
    - Global resource management
    - Population hosting
    - World events
    - Time cycles (day/night, seasons)
    - Migration between regions

    Example:
        world = World()

        # Add populations
        world.add_population(pop, "core")

        # Run world
        for _ in range(1000):
            world.tick()

        # Get global stats
        stats = world.get_global_stats()
    """

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()

        # Regions
        self.regions: Dict[str, Region] = {}
        self._initialize_regions()

        # Populations (by region)
        self.populations: Dict[str, Population] = {}

        # Global resources
        self.global_resources: Dict[str, ResourcePool] = {
            "compute": ResourcePool("compute", 10000, 50000, 100),
            "memory": ResourcePool("memory", 10000, 50000, 80),
            "bandwidth": ResourcePool("bandwidth", 5000, 20000, 50),
            "energy": ResourcePool("energy", 20000, 100000, 200),
            "data": ResourcePool("data", 50000, 200000, 500)
        }

        # Time
        self.tick_count = 0
        self.day = 0
        self.time_of_day = 0         # 0-24
        self.season = Season.SPRING
        self.year = 1

        # Events
        self.active_events: List[WorldEvent] = []
        self.event_history: List[WorldEvent] = []

        # Statistics
        self.total_organisms_ever = 0
        self.total_extinctions = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "event_started": [],
            "season_changed": [],
            "extinction": []
        }

    def _initialize_regions(self):
        """Initialize default world regions"""
        regions = [
            Region(
                name="Central Core",
                type=RegionType.CORE,
                climate=Climate.OPTIMAL,
                resources={
                    "compute": ResourcePool("compute", 2000, 5000, 30),
                    "energy": ResourcePool("energy", 3000, 8000, 50)
                },
                danger_level=0.1,
                habitability=0.95
            ),
            Region(
                name="Fertile Plains",
                type=RegionType.FERTILE,
                climate=Climate.WARM,
                resources={
                    "compute": ResourcePool("compute", 3000, 8000, 50),
                    "data": ResourcePool("data", 5000, 15000, 100),
                    "energy": ResourcePool("energy", 4000, 10000, 60)
                },
                danger_level=0.15,
                habitability=0.9
            ),
            Region(
                name="Northern Wastes",
                type=RegionType.BARREN,
                climate=Climate.COLD,
                resources={
                    "compute": ResourcePool("compute", 500, 2000, 10),
                    "energy": ResourcePool("energy", 1000, 3000, 20)
                },
                danger_level=0.4,
                habitability=0.5
            ),
            Region(
                name="Eastern Frontier",
                type=RegionType.FRONTIER,
                climate=Climate.VOLATILE,
                resources={
                    "compute": ResourcePool("compute", 1500, 4000, 25),
                    "data": ResourcePool("data", 8000, 20000, 150),
                    "bandwidth": ResourcePool("bandwidth", 2000, 6000, 40)
                },
                danger_level=0.5,
                habitability=0.6,
                properties={"opportunity": 0.8, "discovery_bonus": 1.5}
            ),
            Region(
                name="Hostile Depths",
                type=RegionType.HOSTILE,
                climate=Climate.EXTREME_HEAT,
                resources={
                    "energy": ResourcePool("energy", 10000, 30000, 100, quality=1.5)
                },
                danger_level=0.8,
                habitability=0.2,
                properties={"risk_reward": 2.0}
            ),
            Region(
                name="Sanctuary",
                type=RegionType.SANCTUARY,
                climate=Climate.OPTIMAL,
                resources={
                    "compute": ResourcePool("compute", 1000, 3000, 20),
                    "energy": ResourcePool("energy", 2000, 5000, 30)
                },
                danger_level=0.0,
                habitability=1.0,
                properties={"protected": True, "healing_bonus": 1.5}
            )
        ]

        for region in regions:
            self.regions[region.id] = region

        # Connect neighbors
        region_list = list(self.regions.values())
        for i, region in enumerate(region_list):
            # Connect to adjacent regions (simple circular + one skip)
            next_idx = (i + 1) % len(region_list)
            prev_idx = (i - 1) % len(region_list)
            skip_idx = (i + 2) % len(region_list)

            region.neighbors.add(region_list[next_idx].id)
            region.neighbors.add(region_list[prev_idx].id)
            if np.random.random() < 0.5:
                region.neighbors.add(region_list[skip_idx].id)

    def add_population(self, population: Population, region_name: str = "Central Core"):
        """Add a population to a region"""
        # Find region by name
        region = None
        for r in self.regions.values():
            if r.name == region_name:
                region = r
                break

        if not region:
            region = list(self.regions.values())[0]

        self.populations[population.name] = population
        region.population_ids.add(population.name)

    def get_population(self, name: str) -> Optional[Population]:
        """Get population by name"""
        return self.populations.get(name)

    def get_region_by_name(self, name: str) -> Optional[Region]:
        """Get region by name"""
        for region in self.regions.values():
            if region.name == name:
                return region
        return None

    def migrate(self, population_name: str, from_region: str, to_region: str) -> bool:
        """Migrate a population between regions"""
        if not self.config.migration_enabled:
            return False

        from_r = self.regions.get(from_region)
        to_r = self.regions.get(to_region)

        if not from_r or not to_r:
            return False

        # Check if regions are neighbors
        if to_region not in from_r.neighbors:
            return False

        # Move population
        from_r.population_ids.discard(population_name)
        to_r.population_ids.add(population_name)

        return True

    def tick(self):
        """Process one world cycle"""
        self.tick_count += 1

        # Update time
        self._update_time()

        # Regenerate resources
        self._regenerate_resources()

        # Process events
        self._process_events()

        # Maybe trigger new event
        if np.random.random() < self.config.event_frequency:
            self._trigger_random_event()

        # Process populations
        self._process_populations()

        # Apply environmental effects
        self._apply_environmental_effects()

    def _update_time(self):
        """Update time cycles"""
        # Time of day
        self.time_of_day = self.tick_count % self.config.day_length

        # Day count
        new_day = self.tick_count // self.config.day_length
        if new_day > self.day:
            self.day = new_day

        # Season
        ticks_per_year = self.config.season_length * 4
        year_progress = (self.tick_count % ticks_per_year) / ticks_per_year

        old_season = self.season
        if year_progress < 0.25:
            self.season = Season.SPRING
        elif year_progress < 0.5:
            self.season = Season.SUMMER
        elif year_progress < 0.75:
            self.season = Season.AUTUMN
        else:
            self.season = Season.WINTER

        if self.season != old_season:
            self._on_season_change()

        # Year
        self.year = 1 + (self.tick_count // ticks_per_year)

    def _on_season_change(self):
        """Handle season change"""
        # Adjust regional resources based on season
        for region in self.regions.values():
            if self.season == Season.SUMMER:
                # Abundance
                for pool in region.resources.values():
                    pool.regeneration_rate *= 1.5
            elif self.season == Season.WINTER:
                # Scarcity
                for pool in region.resources.values():
                    pool.regeneration_rate *= 0.5
            elif self.season == Season.SPRING:
                # Recovery
                for pool in region.resources.values():
                    pool.regeneration_rate *= 1.2
            # Autumn is normal

        for callback in self._callbacks["season_changed"]:
            callback(self.season)

    def _regenerate_resources(self):
        """Regenerate all resources"""
        # Global resources
        for pool in self.global_resources.values():
            pool.regenerate()
            pool.consumption_rate = 0  # Reset tracking

        # Regional resources
        for region in self.regions.values():
            for pool in region.resources.values():
                pool.regenerate()
                pool.consumption_rate = 0

    def _process_events(self):
        """Process active events"""
        completed = []

        for event in self.active_events:
            elapsed = self.tick_count - event.started_at
            if elapsed >= event.duration:
                completed.append(event)
                self._complete_event(event)
            else:
                self._apply_event_tick(event)

        # Move completed events to history
        for event in completed:
            self.active_events.remove(event)
            self.event_history.append(event)

    def _apply_event_tick(self, event: WorldEvent):
        """Apply ongoing event effects"""
        for region_id in event.affected_regions:
            region = self.regions.get(region_id)
            if not region:
                continue

            if event.type == WorldEventType.RESOURCE_SURGE:
                for pool in region.resources.values():
                    pool.amount = min(
                        pool.max_capacity,
                        pool.amount + event.magnitude * 50
                    )

            elif event.type == WorldEventType.DISASTER:
                region.danger_level = min(1.0, region.danger_level + 0.1)
                region.habitability = max(0.1, region.habitability - 0.05)

    def _complete_event(self, event: WorldEvent):
        """Complete an event and cleanup"""
        for region_id in event.affected_regions:
            region = self.regions.get(region_id)
            if not region:
                continue

            # Restore from disaster
            if event.type == WorldEventType.DISASTER:
                region.danger_level = max(0, region.danger_level - 0.3)

    def _trigger_random_event(self):
        """Trigger a random world event"""
        event_type = np.random.choice([
            WorldEventType.RESOURCE_SURGE,
            WorldEventType.RESOURCE_DEPLETION,
            WorldEventType.DISASTER,
            WorldEventType.ENVIRONMENTAL_SHIFT
        ], p=[0.4, 0.2, 0.2, 0.2])

        # Select affected regions
        num_regions = np.random.randint(1, min(4, len(self.regions) + 1))
        affected = np.random.choice(
            list(self.regions.keys()),
            size=num_regions,
            replace=False
        ).tolist()

        event = WorldEvent(
            type=event_type,
            name=f"{event_type.value}_{self.tick_count}",
            magnitude=np.random.uniform(0.3, 0.8),
            affected_regions=affected,
            duration=np.random.randint(10, 50),
            started_at=self.tick_count
        )

        self.active_events.append(event)

        for callback in self._callbacks["event_started"]:
            callback(event)

    def trigger_event(
        self,
        event_type: WorldEventType,
        magnitude: float,
        regions: Optional[List[str]] = None,
        duration: int = 20
    ) -> WorldEvent:
        """Manually trigger an event"""
        if regions is None:
            regions = list(self.regions.keys())[:2]

        event = WorldEvent(
            type=event_type,
            name=f"{event_type.value}_{self.tick_count}",
            magnitude=magnitude,
            affected_regions=regions,
            duration=duration,
            started_at=self.tick_count
        )

        self.active_events.append(event)
        return event

    def _process_populations(self):
        """Process all populations"""
        for pop in self.populations.values():
            # Find region
            region = None
            for r in self.regions.values():
                if pop.name in r.population_ids:
                    region = r
                    break

            if region:
                # Consume regional resources
                self._population_consumes_resources(pop, region)

                # Apply regional modifiers
                self._apply_regional_modifiers(pop, region)

            # Run population tick
            pop.tick()

            # Track total organisms
            self.total_organisms_ever = max(
                self.total_organisms_ever,
                len(pop.organisms)
            )

            # Check extinction
            if len(pop.organisms) == 0:
                self.total_extinctions += 1
                for callback in self._callbacks["extinction"]:
                    callback(pop)

    def _population_consumes_resources(self, pop: Population, region: Region):
        """Have population consume regional resources"""
        pop_size = len(pop.organisms)
        if pop_size == 0:
            return

        # Consume resources based on population size
        for resource_name, pool in region.resources.items():
            consumption = pop_size * 0.5  # Base consumption per organism
            consumed = pool.consume(consumption)

            # Distribute to organisms
            per_organism = consumed / pop_size
            for organism in pop.organisms.values():
                if hasattr(organism, 'metabolism'):
                    organism.metabolism.energy = min(
                        organism.metabolism.max_energy,
                        organism.metabolism.energy + per_organism * 0.5
                    )

    def _apply_regional_modifiers(self, pop: Population, region: Region):
        """Apply regional effects to population"""
        for organism in pop.organisms.values():
            # Danger effects
            if region.danger_level > 0.5 and np.random.random() < region.danger_level * 0.1:
                if hasattr(organism, 'damage'):
                    organism.damage("TRAUMA", region.danger_level * 0.3, "environment")
                if hasattr(organism, 'endocrine'):
                    organism.endocrine.trigger_stress_response(region.danger_level * 0.2)

            # Sanctuary healing
            if region.properties.get("protected") and hasattr(organism, 'metabolism'):
                healing_bonus = region.properties.get("healing_bonus", 1.0)
                organism.metabolism.energy = min(
                    organism.metabolism.max_energy,
                    organism.metabolism.energy + healing_bonus
                )

    def _apply_environmental_effects(self):
        """Apply global environmental effects"""
        # Day/night effects
        is_night = self.time_of_day < 6 or self.time_of_day > 18

        for pop in self.populations.values():
            for organism in pop.organisms.values():
                if hasattr(organism, 'endocrine'):
                    if is_night:
                        organism.endocrine.trigger_rest_mode()

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global world statistics"""
        total_population = sum(
            len(pop.organisms) for pop in self.populations.values()
        )

        total_resources = sum(
            pool.amount for pool in self.global_resources.values()
        )

        return {
            "tick_count": self.tick_count,
            "day": self.day,
            "time_of_day": self.time_of_day,
            "season": self.season.value,
            "year": self.year,
            "regions": len(self.regions),
            "populations": len(self.populations),
            "total_organisms": total_population,
            "total_organisms_ever": self.total_organisms_ever,
            "extinctions": self.total_extinctions,
            "active_events": len(self.active_events),
            "total_global_resources": total_resources
        }

    def get_region_status(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a region"""
        region = self.regions.get(region_id)
        if not region:
            return None

        populations = [
            pop_name for pop_name in region.population_ids
            if pop_name in self.populations
        ]

        total_pop = sum(
            len(self.populations[p].organisms)
            for p in populations
        )

        return {
            "name": region.name,
            "type": region.type.value,
            "climate": region.climate.value,
            "danger_level": region.danger_level,
            "habitability": region.habitability,
            "populations": len(populations),
            "total_organisms": total_pop,
            "resources": {
                name: {"amount": pool.amount, "max": pool.max_capacity}
                for name, pool in region.resources.items()
            },
            "neighbors": len(region.neighbors)
        }

    def on_event_started(self, callback: Callable[[WorldEvent], None]):
        """Register callback for event start"""
        self._callbacks["event_started"].append(callback)

    def on_season_changed(self, callback: Callable[[Season], None]):
        """Register callback for season change"""
        self._callbacks["season_changed"].append(callback)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("World Simulation Demo")
    print("=" * 50)

    # Create world
    config = WorldConfig(
        name="Test World",
        season_length=25,
        day_length=12,
        event_frequency=0.1
    )
    world = World(config)

    print(f"\n1. World created: {config.name}")
    print(f"   Regions: {len(world.regions)}")

    # List regions
    print("\n2. Regions:")
    for region in world.regions.values():
        print(f"   - {region.name} ({region.type.value}): "
              f"habitability={region.habitability:.2f}, "
              f"danger={region.danger_level:.2f}")

    # Create mock population
    pop = Population("test_colony", carrying_capacity=50)
    world.add_population(pop, "Central Core")
    print(f"\n3. Added population '{pop.name}' to Central Core")

    # Run world simulation
    print("\n4. Running world simulation...")
    events_triggered = 0

    def on_event(event):
        nonlocal events_triggered
        events_triggered += 1
        print(f"   Event: {event.type.value} (magnitude={event.magnitude:.2f})")

    world.on_event_started(on_event)

    for tick in range(100):
        world.tick()

        if tick % 25 == 0:
            stats = world.get_global_stats()
            print(f"   Tick {tick}: day={stats['day']}, "
                  f"season={stats['season']}, "
                  f"events={stats['active_events']}")

    # Final stats
    print("\n5. Final world stats:")
    stats = world.get_global_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Region status
    print("\n6. Region status (Central Core):")
    core_region = world.get_region_by_name("Central Core")
    if core_region:
        status = world.get_region_status(core_region.id)
        if status:
            for key, value in status.items():
                if not isinstance(value, dict):
                    print(f"   {key}: {value}")

    print(f"\n   Total events triggered: {events_triggered}")
