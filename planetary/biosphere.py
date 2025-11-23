"""
Global Biosphere Simulation

Implements planetary-scale biosphere with climate zones, ecosystems,
and global biological processes. This is part of GAIA-7 workstream.

Features:
- Climate zone modeling (tropical, temperate, polar, etc.)
- Ecosystem health metrics
- Biodiversity tracking
- Carbon sequestration
- Atmospheric composition
- Ocean dynamics
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

class ClimateZone(Enum):
    """Global climate zones"""
    TROPICAL = "tropical"              # Hot and wet, high biodiversity
    SUBTROPICAL = "subtropical"        # Warm, seasonal rain
    TEMPERATE = "temperate"            # Moderate, four seasons
    CONTINENTAL = "continental"        # Extreme seasonal variation
    POLAR = "polar"                    # Cold, low biodiversity
    ARID = "arid"                      # Hot and dry, deserts
    MEDITERRANEAN = "mediterranean"   # Dry summer, wet winter
    OCEANIC = "oceanic"               # Mild, consistent precipitation


class BiomeType(Enum):
    """Major biome classifications"""
    TROPICAL_RAINFOREST = "tropical_rainforest"
    TEMPERATE_FOREST = "temperate_forest"
    BOREAL_FOREST = "boreal_forest"
    GRASSLAND = "grassland"
    SAVANNA = "savanna"
    DESERT = "desert"
    TUNDRA = "tundra"
    WETLAND = "wetland"
    CORAL_REEF = "coral_reef"
    DEEP_OCEAN = "deep_ocean"
    COASTAL = "coastal"


class EcosystemHealth(Enum):
    """Ecosystem health levels"""
    THRIVING = "thriving"            # > 0.8
    HEALTHY = "healthy"              # 0.6 - 0.8
    STRESSED = "stressed"            # 0.4 - 0.6
    DEGRADED = "degraded"            # 0.2 - 0.4
    COLLAPSED = "collapsed"          # < 0.2


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AtmosphericComposition:
    """Atmospheric gas concentrations (in ppm or percentages)"""
    co2: float = 400.0               # CO2 in ppm
    o2: float = 20.95                # O2 percentage
    n2: float = 78.08                # N2 percentage
    methane: float = 1.9             # Methane in ppm
    water_vapor: float = 2.5         # Water vapor percentage (average)
    other: float = 0.97              # Other gases

    def get_greenhouse_effect(self) -> float:
        """Calculate greenhouse warming potential"""
        # Simplified greenhouse effect calculation
        co2_effect = np.log(self.co2 / 280) * 5.35  # Radiative forcing
        methane_effect = (self.methane - 0.7) * 0.036
        return co2_effect + methane_effect

    def get_breathability(self) -> float:
        """Calculate atmospheric breathability (0-1)"""
        o2_optimal = abs(self.o2 - 21.0) < 5.0
        co2_safe = self.co2 < 5000  # Danger threshold
        return 1.0 if o2_optimal and co2_safe else 0.5


@dataclass
class OceanState:
    """Global ocean state"""
    temperature: float = 17.0         # Average surface temp (Celsius)
    acidity: float = 8.1              # pH level
    sea_level: float = 0.0            # Relative to baseline (meters)
    ice_coverage: float = 0.1         # Fraction of ocean covered by ice
    current_strength: float = 1.0     # Thermohaline circulation strength
    oxygen_content: float = 0.95      # Dissolved oxygen (relative)

    def get_health(self) -> float:
        """Calculate ocean health score (0-1)"""
        temp_score = 1.0 - abs(self.temperature - 17.0) / 10.0
        acidity_score = 1.0 - abs(self.acidity - 8.1) / 0.5
        oxygen_score = self.oxygen_content
        circulation_score = min(1.0, self.current_strength)

        return np.mean([temp_score, acidity_score, oxygen_score, circulation_score])


@dataclass
class BiomeRegion:
    """A biome region within the biosphere"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    biome_type: BiomeType = BiomeType.TEMPERATE_FOREST
    climate_zone: ClimateZone = ClimateZone.TEMPERATE
    area_km2: float = 100000.0        # Area in square kilometers
    latitude: float = 45.0            # Latitude (-90 to 90)

    # Ecosystem metrics
    biodiversity_index: float = 0.7   # 0-1 species richness
    biomass_density: float = 1.0      # Relative biomass
    primary_productivity: float = 1.0 # Net primary productivity
    ecosystem_health: float = 0.8     # Overall health 0-1

    # Carbon
    carbon_storage: float = 1000.0    # Tons of carbon stored
    carbon_flux: float = 0.0          # Net carbon exchange (+ = release)

    # Water
    precipitation: float = 1000.0     # Annual precipitation (mm)
    evapotranspiration: float = 800.0 # Annual ET (mm)

    # Population
    population_density: float = 0.0   # Organisms per km2
    species_count: int = 1000

    # Disturbance
    disturbance_level: float = 0.0    # 0-1, recent disturbance
    recovery_rate: float = 0.1        # Recovery speed

    def get_health_status(self) -> EcosystemHealth:
        """Get ecosystem health status"""
        if self.ecosystem_health > 0.8:
            return EcosystemHealth.THRIVING
        elif self.ecosystem_health > 0.6:
            return EcosystemHealth.HEALTHY
        elif self.ecosystem_health > 0.4:
            return EcosystemHealth.STRESSED
        elif self.ecosystem_health > 0.2:
            return EcosystemHealth.DEGRADED
        else:
            return EcosystemHealth.COLLAPSED


@dataclass
class BiosphereConfig:
    """Configuration for biosphere simulation"""
    enable_climate_dynamics: bool = True
    enable_carbon_cycle: bool = True
    enable_ocean_dynamics: bool = True
    base_temperature: float = 15.0    # Global average temp
    update_interval: int = 100        # Ticks between major updates
    disturbance_rate: float = 0.01    # Probability of disturbance per tick


# ============================================================================
# Biosphere
# ============================================================================

class Biosphere:
    """
    Global biosphere simulation.

    Manages planetary-scale biological systems including:
    - Climate zones and biomes
    - Carbon and nutrient cycles
    - Biodiversity metrics
    - Ecosystem health
    - Ocean dynamics

    Example:
        biosphere = Biosphere()
        biosphere.initialize_default_biomes()

        # Run simulation
        for _ in range(1000):
            biosphere.tick()

        # Get global metrics
        metrics = biosphere.get_global_metrics()
    """

    def __init__(self, config: Optional[BiosphereConfig] = None):
        self.config = config or BiosphereConfig()

        # Biome regions
        self.biomes: Dict[str, BiomeRegion] = {}

        # Atmospheric state
        self.atmosphere = AtmosphericComposition()

        # Ocean state
        self.ocean = OceanState()

        # Global metrics
        self.global_temperature = self.config.base_temperature
        self.total_biomass = 0.0
        self.total_carbon_storage = 0.0
        self.biodiversity_global = 0.0

        # Time tracking
        self.tick_count = 0
        self.epoch = 0  # Geological epochs

        # History
        self.temperature_history: List[float] = []
        self.co2_history: List[float] = []
        self.biodiversity_history: List[float] = []

        # Callbacks
        self._callbacks: Dict[str, List] = {
            "biome_collapsed": [],
            "mass_extinction": [],
            "climate_shift": []
        }

    def initialize_default_biomes(self):
        """Initialize Earth-like biome distribution"""
        default_biomes = [
            BiomeRegion(
                name="Amazon Basin",
                biome_type=BiomeType.TROPICAL_RAINFOREST,
                climate_zone=ClimateZone.TROPICAL,
                area_km2=5500000,
                latitude=-3.0,
                biodiversity_index=0.95,
                carbon_storage=150000,
                precipitation=2300
            ),
            BiomeRegion(
                name="Congo Rainforest",
                biome_type=BiomeType.TROPICAL_RAINFOREST,
                climate_zone=ClimateZone.TROPICAL,
                area_km2=2000000,
                latitude=0.0,
                biodiversity_index=0.90,
                carbon_storage=60000,
                precipitation=1800
            ),
            BiomeRegion(
                name="Siberian Taiga",
                biome_type=BiomeType.BOREAL_FOREST,
                climate_zone=ClimateZone.CONTINENTAL,
                area_km2=9000000,
                latitude=60.0,
                biodiversity_index=0.4,
                carbon_storage=200000,
                precipitation=500
            ),
            BiomeRegion(
                name="North American Prairies",
                biome_type=BiomeType.GRASSLAND,
                climate_zone=ClimateZone.TEMPERATE,
                area_km2=1300000,
                latitude=40.0,
                biodiversity_index=0.55,
                carbon_storage=15000,
                precipitation=600
            ),
            BiomeRegion(
                name="African Savanna",
                biome_type=BiomeType.SAVANNA,
                climate_zone=ClimateZone.SUBTROPICAL,
                area_km2=13000000,
                latitude=-5.0,
                biodiversity_index=0.75,
                carbon_storage=25000,
                precipitation=900
            ),
            BiomeRegion(
                name="Sahara Desert",
                biome_type=BiomeType.DESERT,
                climate_zone=ClimateZone.ARID,
                area_km2=9200000,
                latitude=23.0,
                biodiversity_index=0.15,
                carbon_storage=1000,
                precipitation=25
            ),
            BiomeRegion(
                name="Arctic Tundra",
                biome_type=BiomeType.TUNDRA,
                climate_zone=ClimateZone.POLAR,
                area_km2=8000000,
                latitude=70.0,
                biodiversity_index=0.25,
                carbon_storage=300000,  # Permafrost carbon
                precipitation=250
            ),
            BiomeRegion(
                name="European Mixed Forest",
                biome_type=BiomeType.TEMPERATE_FOREST,
                climate_zone=ClimateZone.OCEANIC,
                area_km2=2000000,
                latitude=50.0,
                biodiversity_index=0.60,
                carbon_storage=40000,
                precipitation=800
            ),
            BiomeRegion(
                name="Great Barrier Reef",
                biome_type=BiomeType.CORAL_REEF,
                climate_zone=ClimateZone.TROPICAL,
                area_km2=348000,
                latitude=-18.0,
                biodiversity_index=0.92,
                carbon_storage=5000,
                precipitation=2000
            ),
            BiomeRegion(
                name="Pacific Deep Ocean",
                biome_type=BiomeType.DEEP_OCEAN,
                climate_zone=ClimateZone.OCEANIC,
                area_km2=60000000,
                latitude=0.0,
                biodiversity_index=0.50,
                carbon_storage=1000000,
                precipitation=0
            )
        ]

        for biome in default_biomes:
            self.biomes[biome.id] = biome

        self._calculate_global_metrics()

    def add_biome(self, biome: BiomeRegion):
        """Add a biome region"""
        self.biomes[biome.id] = biome
        self._calculate_global_metrics()

    def tick(self):
        """Process one simulation cycle"""
        self.tick_count += 1

        # Update biomes
        for biome in self.biomes.values():
            self._update_biome(biome)

        # Update atmosphere
        if self.config.enable_carbon_cycle:
            self._update_carbon_cycle()

        # Update ocean
        if self.config.enable_ocean_dynamics:
            self._update_ocean()

        # Update climate
        if self.config.enable_climate_dynamics:
            self._update_climate()

        # Random disturbances
        if np.random.random() < self.config.disturbance_rate:
            self._trigger_disturbance()

        # Periodic major updates
        if self.tick_count % self.config.update_interval == 0:
            self._calculate_global_metrics()
            self._record_history()

    def _update_biome(self, biome: BiomeRegion):
        """Update a single biome"""
        # Recovery from disturbance
        if biome.disturbance_level > 0:
            biome.disturbance_level = max(0, biome.disturbance_level - biome.recovery_rate)

        # Calculate temperature effect on this biome
        temp_deviation = self.global_temperature - self.config.base_temperature
        temp_stress = abs(temp_deviation) / 5.0  # 5 degree tolerance

        # Update ecosystem health
        base_health = 1.0 - biome.disturbance_level
        climate_health = max(0, 1.0 - temp_stress * 0.2)

        biome.ecosystem_health = (
            base_health * 0.5 +
            climate_health * 0.3 +
            biome.biodiversity_index * 0.2
        )

        # Update primary productivity
        if biome.biome_type in [BiomeType.TROPICAL_RAINFOREST, BiomeType.TEMPERATE_FOREST]:
            co2_boost = min(0.2, (self.atmosphere.co2 - 280) / 1000)  # CO2 fertilization
            biome.primary_productivity = (
                biome.ecosystem_health *
                (1.0 + co2_boost) *
                max(0.5, 1.0 - temp_stress * 0.3)
            )
        else:
            biome.primary_productivity = biome.ecosystem_health

        # Carbon flux
        if biome.ecosystem_health > 0.5:
            # Healthy ecosystems sequester carbon
            biome.carbon_flux = -biome.primary_productivity * biome.area_km2 * 0.001
        else:
            # Stressed ecosystems release carbon
            biome.carbon_flux = (1.0 - biome.ecosystem_health) * biome.carbon_storage * 0.0001

        # Update carbon storage
        biome.carbon_storage = max(0, biome.carbon_storage - biome.carbon_flux)

        # Biodiversity responds to health
        biodiversity_delta = (biome.ecosystem_health - 0.5) * 0.01
        biome.biodiversity_index = np.clip(
            biome.biodiversity_index + biodiversity_delta,
            0.0, 1.0
        )

        # Check for collapse
        if biome.ecosystem_health < 0.1:
            self._on_biome_collapsed(biome)

    def _update_carbon_cycle(self):
        """Update global carbon cycle"""
        # Sum carbon flux from all biomes
        total_flux = sum(biome.carbon_flux for biome in self.biomes.values())

        # Ocean absorption (about 25% of emissions)
        ocean_absorption = total_flux * 0.25 if total_flux > 0 else 0

        # Atmospheric CO2 change
        atmospheric_change = (total_flux - ocean_absorption) * 0.1
        self.atmosphere.co2 = max(180, self.atmosphere.co2 + atmospheric_change)

        # Ocean acidification from CO2
        if total_flux > 0:
            self.ocean.acidity -= ocean_absorption * 0.0001

    def _update_ocean(self):
        """Update ocean dynamics"""
        # Temperature coupling with atmosphere
        temp_diff = self.global_temperature - self.ocean.temperature
        self.ocean.temperature += temp_diff * 0.01  # Slow thermal inertia

        # Ice coverage responds to temperature
        if self.ocean.temperature > 20:
            self.ocean.ice_coverage = max(0, self.ocean.ice_coverage - 0.001)
        elif self.ocean.temperature < 15:
            self.ocean.ice_coverage = min(0.3, self.ocean.ice_coverage + 0.0005)

        # Sea level from thermal expansion and ice melt
        thermal_expansion = (self.ocean.temperature - 17.0) * 0.01
        ice_contribution = (0.1 - self.ocean.ice_coverage) * 0.1
        self.ocean.sea_level = thermal_expansion + ice_contribution

        # Thermohaline circulation strength
        temp_gradient = abs(self.ocean.temperature - 5.0)  # Polar-tropical gradient
        self.ocean.current_strength = min(1.5, temp_gradient / 15.0)

        # Dissolved oxygen
        self.ocean.oxygen_content = min(1.0, 0.8 + (17.0 - self.ocean.temperature) * 0.02)

    def _update_climate(self):
        """Update global climate"""
        # Greenhouse effect
        greenhouse = self.atmosphere.get_greenhouse_effect()

        # Albedo from ice and vegetation
        ice_albedo = self.ocean.ice_coverage * 0.3
        vegetation_albedo = np.mean([
            b.biomass_density * 0.1
            for b in self.biomes.values()
        ])
        total_albedo = 0.3 + ice_albedo - vegetation_albedo

        # Temperature calculation
        solar_input = 1366  # W/m2 solar constant
        equilibrium_temp = (
            self.config.base_temperature +
            greenhouse * 0.5 -
            (total_albedo - 0.3) * 20
        )

        # Slow temperature adjustment
        self.global_temperature += (equilibrium_temp - self.global_temperature) * 0.01

        # Check for climate shift
        if abs(self.global_temperature - self.config.base_temperature) > 3.0:
            self._on_climate_shift()

    def _trigger_disturbance(self):
        """Trigger a random disturbance event"""
        if not self.biomes:
            return

        # Select random biome
        biome = np.random.choice(list(self.biomes.values()))

        # Disturbance magnitude
        magnitude = np.random.uniform(0.1, 0.5)
        biome.disturbance_level = min(1.0, biome.disturbance_level + magnitude)

        # Immediate effects
        biome.biodiversity_index = max(0.1, biome.biodiversity_index - magnitude * 0.2)
        biome.carbon_storage = max(0, biome.carbon_storage - magnitude * biome.carbon_storage * 0.1)

    def _calculate_global_metrics(self):
        """Calculate global biosphere metrics"""
        if not self.biomes:
            return

        total_area = sum(b.area_km2 for b in self.biomes.values())

        # Weighted averages by area
        self.biodiversity_global = sum(
            b.biodiversity_index * b.area_km2 / total_area
            for b in self.biomes.values()
        )

        self.total_biomass = sum(
            b.biomass_density * b.area_km2
            for b in self.biomes.values()
        )

        self.total_carbon_storage = sum(
            b.carbon_storage for b in self.biomes.values()
        )

    def _record_history(self):
        """Record historical data"""
        self.temperature_history.append(self.global_temperature)
        self.co2_history.append(self.atmosphere.co2)
        self.biodiversity_history.append(self.biodiversity_global)

        # Keep only last 1000 records
        max_history = 1000
        if len(self.temperature_history) > max_history:
            self.temperature_history = self.temperature_history[-max_history:]
            self.co2_history = self.co2_history[-max_history:]
            self.biodiversity_history = self.biodiversity_history[-max_history:]

    def _on_biome_collapsed(self, biome: BiomeRegion):
        """Handle biome collapse"""
        for callback in self._callbacks["biome_collapsed"]:
            callback(biome)

        # Check for mass extinction
        collapsed_count = sum(
            1 for b in self.biomes.values()
            if b.get_health_status() == EcosystemHealth.COLLAPSED
        )

        if collapsed_count > len(self.biomes) * 0.5:
            self._on_mass_extinction()

    def _on_mass_extinction(self):
        """Handle mass extinction event"""
        for callback in self._callbacks["mass_extinction"]:
            callback(self.tick_count)

    def _on_climate_shift(self):
        """Handle significant climate shift"""
        for callback in self._callbacks["climate_shift"]:
            callback(self.global_temperature)

    # ========================================================================
    # Public API
    # ========================================================================

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global biosphere metrics"""
        return {
            "tick_count": self.tick_count,
            "global_temperature": self.global_temperature,
            "temperature_anomaly": self.global_temperature - self.config.base_temperature,
            "atmosphere": {
                "co2_ppm": self.atmosphere.co2,
                "greenhouse_effect": self.atmosphere.get_greenhouse_effect(),
                "breathability": self.atmosphere.get_breathability()
            },
            "ocean": {
                "temperature": self.ocean.temperature,
                "acidity": self.ocean.acidity,
                "sea_level": self.ocean.sea_level,
                "ice_coverage": self.ocean.ice_coverage,
                "health": self.ocean.get_health()
            },
            "biodiversity": {
                "global_index": self.biodiversity_global,
                "total_biomass": self.total_biomass
            },
            "carbon": {
                "total_storage": self.total_carbon_storage,
                "atmospheric_co2": self.atmosphere.co2
            },
            "biome_count": len(self.biomes),
            "healthy_biomes": sum(
                1 for b in self.biomes.values()
                if b.get_health_status() in [EcosystemHealth.THRIVING, EcosystemHealth.HEALTHY]
            )
        }

    def get_biome_status(self, biome_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific biome"""
        biome = self.biomes.get(biome_id)
        if not biome:
            return None

        return {
            "name": biome.name,
            "type": biome.biome_type.value,
            "climate_zone": biome.climate_zone.value,
            "area_km2": biome.area_km2,
            "health": biome.ecosystem_health,
            "health_status": biome.get_health_status().value,
            "biodiversity": biome.biodiversity_index,
            "carbon_storage": biome.carbon_storage,
            "carbon_flux": biome.carbon_flux,
            "productivity": biome.primary_productivity,
            "disturbance": biome.disturbance_level
        }

    def get_all_biome_summaries(self) -> List[Dict[str, Any]]:
        """Get summary of all biomes"""
        return [
            {
                "id": biome.id,
                "name": biome.name,
                "type": biome.biome_type.value,
                "health": biome.ecosystem_health,
                "biodiversity": biome.biodiversity_index
            }
            for biome in self.biomes.values()
        ]

    def apply_stress(self, biome_id: str, stress_level: float):
        """Apply stress to a biome"""
        biome = self.biomes.get(biome_id)
        if biome:
            biome.disturbance_level = min(1.0, biome.disturbance_level + stress_level)

    def on_biome_collapsed(self, callback):
        """Register callback for biome collapse"""
        self._callbacks["biome_collapsed"].append(callback)

    def on_mass_extinction(self, callback):
        """Register callback for mass extinction"""
        self._callbacks["mass_extinction"].append(callback)

    def on_climate_shift(self, callback):
        """Register callback for climate shift"""
        self._callbacks["climate_shift"].append(callback)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Biosphere Simulation Demo")
    print("=" * 50)

    # Create biosphere
    config = BiosphereConfig(
        base_temperature=15.0,
        disturbance_rate=0.02
    )
    biosphere = Biosphere(config)
    biosphere.initialize_default_biomes()

    print(f"\n1. Initialized biosphere with {len(biosphere.biomes)} biomes")

    # List biomes
    print("\n2. Biomes:")
    for biome in biosphere.biomes.values():
        print(f"   - {biome.name} ({biome.biome_type.value}): "
              f"biodiversity={biome.biodiversity_index:.2f}")

    # Run simulation
    print("\n3. Running simulation for 500 ticks...")
    for i in range(500):
        biosphere.tick()

        if i % 100 == 0:
            metrics = biosphere.get_global_metrics()
            print(f"   Tick {i}: temp={metrics['global_temperature']:.2f}C, "
                  f"CO2={metrics['atmosphere']['co2_ppm']:.1f}ppm, "
                  f"biodiversity={metrics['biodiversity']['global_index']:.3f}")

    # Final metrics
    print("\n4. Final global metrics:")
    metrics = biosphere.get_global_metrics()
    print(f"   Temperature: {metrics['global_temperature']:.2f}C")
    print(f"   CO2: {metrics['atmosphere']['co2_ppm']:.1f} ppm")
    print(f"   Biodiversity index: {metrics['biodiversity']['global_index']:.3f}")
    print(f"   Ocean health: {metrics['ocean']['health']:.3f}")
    print(f"   Healthy biomes: {metrics['healthy_biomes']}/{metrics['biome_count']}")
