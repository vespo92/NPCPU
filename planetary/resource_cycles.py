"""
Global Resource Cycles

Implements planetary-scale biogeochemical cycles including carbon,
nitrogen, water, and phosphorus. These cycles are fundamental to
planetary homeostasis and consciousness emergence.

Features:
- Carbon cycle with atmospheric, oceanic, and biospheric reservoirs
- Nitrogen cycle with fixation and denitrification
- Water cycle with evaporation, precipitation, and runoff
- Phosphorus cycle with weathering and sedimentation
- Inter-cycle coupling and feedback
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

class CycleType(Enum):
    """Types of biogeochemical cycles"""
    CARBON = "carbon"
    NITROGEN = "nitrogen"
    WATER = "water"
    PHOSPHORUS = "phosphorus"
    OXYGEN = "oxygen"
    SULFUR = "sulfur"


class ReservoirType(Enum):
    """Types of element reservoirs"""
    ATMOSPHERE = "atmosphere"
    OCEAN = "ocean"
    TERRESTRIAL_BIOSPHERE = "terrestrial_biosphere"
    SOIL = "soil"
    SEDIMENT = "sediment"
    FOSSIL_FUEL = "fossil_fuel"
    LITHOSPHERE = "lithosphere"


class FluxType(Enum):
    """Types of element fluxes"""
    PHOTOSYNTHESIS = "photosynthesis"
    RESPIRATION = "respiration"
    DECOMPOSITION = "decomposition"
    COMBUSTION = "combustion"
    WEATHERING = "weathering"
    VOLCANIC = "volcanic"
    OCEANIC_UPTAKE = "oceanic_uptake"
    OCEANIC_RELEASE = "oceanic_release"
    PRECIPITATION = "precipitation"
    EVAPORATION = "evaporation"
    FIXATION = "fixation"
    DENITRIFICATION = "denitrification"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Reservoir:
    """A reservoir storing an element"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    reservoir_type: ReservoirType = ReservoirType.ATMOSPHERE
    element: str = "carbon"
    amount: float = 0.0           # Current amount (Gt or equivalent)
    capacity: float = float('inf')  # Maximum capacity
    residence_time: float = 100.0   # Average residence time (years)

    def add(self, amount: float) -> float:
        """Add to reservoir, return actual amount added"""
        space = self.capacity - self.amount
        actual = min(amount, space)
        self.amount += actual
        return actual

    def remove(self, amount: float) -> float:
        """Remove from reservoir, return actual amount removed"""
        actual = min(amount, self.amount)
        self.amount -= actual
        return actual


@dataclass
class Flux:
    """A flux transferring element between reservoirs"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    flux_type: FluxType = FluxType.RESPIRATION
    source_id: str = ""
    sink_id: str = ""
    base_rate: float = 1.0       # Base rate (Gt/year equivalent)
    current_rate: float = 1.0    # Current modified rate
    sensitivity: float = 1.0      # Sensitivity to environmental factors


@dataclass
class CycleConfig:
    """Configuration for resource cycles"""
    time_step: float = 1.0           # Years per tick
    enable_anthropogenic: bool = True  # Include human impacts
    enable_coupling: bool = True      # Inter-cycle coupling
    perturbation_rate: float = 0.001  # Random perturbation rate


# ============================================================================
# Resource Cycle Base Class
# ============================================================================

class ResourceCycle:
    """
    Base class for biogeochemical cycles.

    Provides common functionality for element cycling
    between reservoirs with configurable fluxes.
    """

    def __init__(self, cycle_type: CycleType, config: Optional[CycleConfig] = None):
        self.cycle_type = cycle_type
        self.config = config or CycleConfig()

        self.reservoirs: Dict[str, Reservoir] = {}
        self.fluxes: Dict[str, Flux] = {}

        self.tick_count = 0
        self.balance = 0.0  # Net flux balance

        # History tracking
        self.reservoir_history: Dict[str, List[float]] = defaultdict(list)
        self.flux_history: Dict[str, List[float]] = defaultdict(list)

    def add_reservoir(self, reservoir: Reservoir):
        """Add a reservoir to the cycle"""
        self.reservoirs[reservoir.id] = reservoir

    def add_flux(self, flux: Flux):
        """Add a flux to the cycle"""
        self.fluxes[flux.id] = flux

    def tick(self):
        """Process one cycle tick"""
        self.tick_count += 1

        # Process all fluxes
        total_flux = 0.0
        for flux in self.fluxes.values():
            amount = self._process_flux(flux)
            total_flux += amount

        self.balance = total_flux

        # Record history
        for res_id, reservoir in self.reservoirs.items():
            self.reservoir_history[res_id].append(reservoir.amount)

        for flux_id, flux in self.fluxes.items():
            self.flux_history[flux_id].append(flux.current_rate)

        # Trim history
        max_history = 1000
        for key in self.reservoir_history:
            if len(self.reservoir_history[key]) > max_history:
                self.reservoir_history[key] = self.reservoir_history[key][-max_history:]
        for key in self.flux_history:
            if len(self.flux_history[key]) > max_history:
                self.flux_history[key] = self.flux_history[key][-max_history:]

    def _process_flux(self, flux: Flux) -> float:
        """Process a single flux, return amount transferred"""
        source = self.reservoirs.get(flux.source_id)
        sink = self.reservoirs.get(flux.sink_id)

        if not source or not sink:
            return 0.0

        # Calculate transfer amount
        amount = flux.current_rate * self.config.time_step

        # Execute transfer
        removed = source.remove(amount)
        added = sink.add(removed)

        return removed

    def get_total_mass(self) -> float:
        """Get total mass in all reservoirs"""
        return sum(r.amount for r in self.reservoirs.values())

    def get_reservoir_fractions(self) -> Dict[str, float]:
        """Get fraction of total in each reservoir"""
        total = self.get_total_mass()
        if total == 0:
            return {}
        return {
            r.name: r.amount / total
            for r in self.reservoirs.values()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get cycle status"""
        return {
            "cycle_type": self.cycle_type.value,
            "tick_count": self.tick_count,
            "total_mass": self.get_total_mass(),
            "balance": self.balance,
            "reservoirs": {
                r.name: r.amount
                for r in self.reservoirs.values()
            },
            "fluxes": {
                f.name: f.current_rate
                for f in self.fluxes.values()
            }
        }


# ============================================================================
# Carbon Cycle
# ============================================================================

class CarbonCycle(ResourceCycle):
    """
    Global carbon cycle simulation.

    Models carbon exchange between atmosphere, ocean, biosphere,
    soils, and geological reservoirs.

    Example:
        cycle = CarbonCycle()
        cycle.initialize()

        for _ in range(100):
            cycle.tick()

        status = cycle.get_status()
    """

    def __init__(self, config: Optional[CycleConfig] = None):
        super().__init__(CycleType.CARBON, config)
        self.atmospheric_co2_ppm = 400.0

    def initialize(self):
        """Initialize carbon cycle with Earth-like values"""
        # Reservoirs (amounts in Gt C)
        atmosphere = Reservoir(
            name="atmosphere",
            reservoir_type=ReservoirType.ATMOSPHERE,
            element="carbon",
            amount=850,
            residence_time=4
        )

        ocean_surface = Reservoir(
            name="ocean_surface",
            reservoir_type=ReservoirType.OCEAN,
            element="carbon",
            amount=900,
            residence_time=10
        )

        ocean_deep = Reservoir(
            name="ocean_deep",
            reservoir_type=ReservoirType.OCEAN,
            element="carbon",
            amount=37000,
            residence_time=1000
        )

        terrestrial_biosphere = Reservoir(
            name="terrestrial_biosphere",
            reservoir_type=ReservoirType.TERRESTRIAL_BIOSPHERE,
            element="carbon",
            amount=550,
            residence_time=10
        )

        soil = Reservoir(
            name="soil",
            reservoir_type=ReservoirType.SOIL,
            element="carbon",
            amount=1500,
            residence_time=25
        )

        fossil_fuels = Reservoir(
            name="fossil_fuels",
            reservoir_type=ReservoirType.FOSSIL_FUEL,
            element="carbon",
            amount=10000,
            residence_time=float('inf')
        )

        self.add_reservoir(atmosphere)
        self.add_reservoir(ocean_surface)
        self.add_reservoir(ocean_deep)
        self.add_reservoir(terrestrial_biosphere)
        self.add_reservoir(soil)
        self.add_reservoir(fossil_fuels)

        # Fluxes (rates in Gt C/year)
        photosynthesis = Flux(
            name="terrestrial_photosynthesis",
            flux_type=FluxType.PHOTOSYNTHESIS,
            source_id=atmosphere.id,
            sink_id=terrestrial_biosphere.id,
            base_rate=120,
            current_rate=120
        )

        respiration = Flux(
            name="terrestrial_respiration",
            flux_type=FluxType.RESPIRATION,
            source_id=terrestrial_biosphere.id,
            sink_id=atmosphere.id,
            base_rate=60,
            current_rate=60
        )

        decomposition = Flux(
            name="decomposition",
            flux_type=FluxType.DECOMPOSITION,
            source_id=soil.id,
            sink_id=atmosphere.id,
            base_rate=60,
            current_rate=60
        )

        litter_fall = Flux(
            name="litter_fall",
            flux_type=FluxType.DECOMPOSITION,
            source_id=terrestrial_biosphere.id,
            sink_id=soil.id,
            base_rate=60,
            current_rate=60
        )

        ocean_uptake = Flux(
            name="ocean_uptake",
            flux_type=FluxType.OCEANIC_UPTAKE,
            source_id=atmosphere.id,
            sink_id=ocean_surface.id,
            base_rate=92,
            current_rate=92
        )

        ocean_release = Flux(
            name="ocean_release",
            flux_type=FluxType.OCEANIC_RELEASE,
            source_id=ocean_surface.id,
            sink_id=atmosphere.id,
            base_rate=90,
            current_rate=90
        )

        deep_mixing = Flux(
            name="deep_ocean_mixing",
            flux_type=FluxType.OCEANIC_UPTAKE,
            source_id=ocean_surface.id,
            sink_id=ocean_deep.id,
            base_rate=100,
            current_rate=100
        )

        upwelling = Flux(
            name="upwelling",
            flux_type=FluxType.OCEANIC_RELEASE,
            source_id=ocean_deep.id,
            sink_id=ocean_surface.id,
            base_rate=100,
            current_rate=100
        )

        self.add_flux(photosynthesis)
        self.add_flux(respiration)
        self.add_flux(decomposition)
        self.add_flux(litter_fall)
        self.add_flux(ocean_uptake)
        self.add_flux(ocean_release)
        self.add_flux(deep_mixing)
        self.add_flux(upwelling)

        # Store reservoir references
        self._atmosphere = atmosphere
        self._ocean_surface = ocean_surface
        self._terrestrial = terrestrial_biosphere
        self._fossil_fuels = fossil_fuels

    def tick(self):
        """Process one carbon cycle tick"""
        # Update CO2-dependent fluxes
        self._update_co2_effects()

        # Call parent tick
        super().tick()

        # Update atmospheric CO2 ppm
        self._update_atmospheric_co2()

    def _update_co2_effects(self):
        """Update fluxes based on CO2 levels"""
        # CO2 fertilization effect on photosynthesis
        co2_ratio = self.atmospheric_co2_ppm / 280  # Pre-industrial baseline
        fertilization_factor = 1.0 + 0.3 * np.log(co2_ratio)

        for flux in self.fluxes.values():
            if flux.flux_type == FluxType.PHOTOSYNTHESIS:
                flux.current_rate = flux.base_rate * fertilization_factor

        # Ocean uptake increases with higher atmospheric CO2
        for flux in self.fluxes.values():
            if flux.flux_type == FluxType.OCEANIC_UPTAKE:
                flux.current_rate = flux.base_rate * (co2_ratio ** 0.3)

    def _update_atmospheric_co2(self):
        """Calculate atmospheric CO2 in ppm"""
        # Approximate conversion: 2.13 Gt C = 1 ppm
        self.atmospheric_co2_ppm = self._atmosphere.amount / 2.13

    def emit_anthropogenic(self, amount: float):
        """Emit anthropogenic CO2 from fossil fuels"""
        if self.config.enable_anthropogenic:
            removed = self._fossil_fuels.remove(amount)
            self._atmosphere.add(removed)

    def sequester_carbon(self, amount: float, target: str = "terrestrial"):
        """Sequester carbon in a reservoir"""
        if target == "terrestrial":
            self._terrestrial.add(amount)
            self._atmosphere.remove(amount)
        elif target == "ocean":
            self._ocean_surface.add(amount)
            self._atmosphere.remove(amount)

    def get_atmospheric_co2(self) -> float:
        """Get atmospheric CO2 in ppm"""
        return self.atmospheric_co2_ppm


# ============================================================================
# Nitrogen Cycle
# ============================================================================

class NitrogenCycle(ResourceCycle):
    """
    Global nitrogen cycle simulation.

    Models nitrogen fixation, nitrification, denitrification,
    and nitrogen exchange between atmosphere, biosphere, and soils.
    """

    def __init__(self, config: Optional[CycleConfig] = None):
        super().__init__(CycleType.NITROGEN, config)

    def initialize(self):
        """Initialize nitrogen cycle with Earth-like values"""
        # Reservoirs (amounts in Tg N)
        atmosphere = Reservoir(
            name="atmosphere_n2",
            reservoir_type=ReservoirType.ATMOSPHERE,
            element="nitrogen",
            amount=3900000000,  # 3.9e9 Tg
            residence_time=float('inf')
        )

        biosphere = Reservoir(
            name="terrestrial_biosphere",
            reservoir_type=ReservoirType.TERRESTRIAL_BIOSPHERE,
            element="nitrogen",
            amount=3500,
            residence_time=20
        )

        soil_organic = Reservoir(
            name="soil_organic_n",
            reservoir_type=ReservoirType.SOIL,
            element="nitrogen",
            amount=95000,
            residence_time=1000
        )

        soil_inorganic = Reservoir(
            name="soil_inorganic_n",
            reservoir_type=ReservoirType.SOIL,
            element="nitrogen",
            amount=5000,
            residence_time=1
        )

        ocean = Reservoir(
            name="ocean_n",
            reservoir_type=ReservoirType.OCEAN,
            element="nitrogen",
            amount=600000,
            residence_time=500
        )

        self.add_reservoir(atmosphere)
        self.add_reservoir(biosphere)
        self.add_reservoir(soil_organic)
        self.add_reservoir(soil_inorganic)
        self.add_reservoir(ocean)

        # Fluxes (rates in Tg N/year)
        biological_fixation = Flux(
            name="biological_fixation",
            flux_type=FluxType.FIXATION,
            source_id=atmosphere.id,
            sink_id=biosphere.id,
            base_rate=140,
            current_rate=140
        )

        industrial_fixation = Flux(
            name="industrial_fixation",
            flux_type=FluxType.FIXATION,
            source_id=atmosphere.id,
            sink_id=soil_inorganic.id,
            base_rate=120,  # Anthropogenic
            current_rate=120
        )

        plant_uptake = Flux(
            name="plant_uptake",
            flux_type=FluxType.PHOTOSYNTHESIS,
            source_id=soil_inorganic.id,
            sink_id=biosphere.id,
            base_rate=200,
            current_rate=200
        )

        decomposition = Flux(
            name="n_decomposition",
            flux_type=FluxType.DECOMPOSITION,
            source_id=biosphere.id,
            sink_id=soil_organic.id,
            base_rate=200,
            current_rate=200
        )

        mineralization = Flux(
            name="mineralization",
            flux_type=FluxType.DECOMPOSITION,
            source_id=soil_organic.id,
            sink_id=soil_inorganic.id,
            base_rate=100,
            current_rate=100
        )

        denitrification = Flux(
            name="denitrification",
            flux_type=FluxType.DENITRIFICATION,
            source_id=soil_inorganic.id,
            sink_id=atmosphere.id,
            base_rate=130,
            current_rate=130
        )

        self.add_flux(biological_fixation)
        self.add_flux(industrial_fixation)
        self.add_flux(plant_uptake)
        self.add_flux(decomposition)
        self.add_flux(mineralization)
        self.add_flux(denitrification)


# ============================================================================
# Water Cycle
# ============================================================================

class WaterCycle(ResourceCycle):
    """
    Global water cycle simulation.

    Models evaporation, precipitation, runoff, and storage
    in atmosphere, oceans, ice, and groundwater.
    """

    def __init__(self, config: Optional[CycleConfig] = None):
        super().__init__(CycleType.WATER, config)
        self.global_temperature = 15.0

    def initialize(self):
        """Initialize water cycle with Earth-like values"""
        # Reservoirs (amounts in km3)
        oceans = Reservoir(
            name="oceans",
            reservoir_type=ReservoirType.OCEAN,
            element="water",
            amount=1370000000,
            residence_time=3000
        )

        atmosphere = Reservoir(
            name="atmosphere",
            reservoir_type=ReservoirType.ATMOSPHERE,
            element="water",
            amount=13000,
            residence_time=0.025  # ~9 days
        )

        ice_caps = Reservoir(
            name="ice_caps",
            reservoir_type=ReservoirType.LITHOSPHERE,
            element="water",
            amount=29000000,
            residence_time=20000
        )

        groundwater = Reservoir(
            name="groundwater",
            reservoir_type=ReservoirType.SOIL,
            element="water",
            amount=10500000,
            residence_time=5000
        )

        rivers_lakes = Reservoir(
            name="rivers_lakes",
            reservoir_type=ReservoirType.TERRESTRIAL_BIOSPHERE,
            element="water",
            amount=200000,
            residence_time=10
        )

        self.add_reservoir(oceans)
        self.add_reservoir(atmosphere)
        self.add_reservoir(ice_caps)
        self.add_reservoir(groundwater)
        self.add_reservoir(rivers_lakes)

        # Fluxes (rates in km3/year)
        ocean_evaporation = Flux(
            name="ocean_evaporation",
            flux_type=FluxType.EVAPORATION,
            source_id=oceans.id,
            sink_id=atmosphere.id,
            base_rate=425000,
            current_rate=425000
        )

        land_evaporation = Flux(
            name="land_evaporation",
            flux_type=FluxType.EVAPORATION,
            source_id=rivers_lakes.id,
            sink_id=atmosphere.id,
            base_rate=71000,
            current_rate=71000
        )

        ocean_precipitation = Flux(
            name="ocean_precipitation",
            flux_type=FluxType.PRECIPITATION,
            source_id=atmosphere.id,
            sink_id=oceans.id,
            base_rate=386000,
            current_rate=386000
        )

        land_precipitation = Flux(
            name="land_precipitation",
            flux_type=FluxType.PRECIPITATION,
            source_id=atmosphere.id,
            sink_id=rivers_lakes.id,
            base_rate=110000,
            current_rate=110000
        )

        runoff = Flux(
            name="runoff",
            flux_type=FluxType.WEATHERING,  # Using as transport
            source_id=rivers_lakes.id,
            sink_id=oceans.id,
            base_rate=39000,
            current_rate=39000
        )

        infiltration = Flux(
            name="infiltration",
            flux_type=FluxType.WEATHERING,
            source_id=rivers_lakes.id,
            sink_id=groundwater.id,
            base_rate=1000,
            current_rate=1000
        )

        self.add_flux(ocean_evaporation)
        self.add_flux(land_evaporation)
        self.add_flux(ocean_precipitation)
        self.add_flux(land_precipitation)
        self.add_flux(runoff)
        self.add_flux(infiltration)

        self._atmosphere = atmosphere
        self._ice_caps = ice_caps

    def tick(self):
        """Process one water cycle tick"""
        self._update_temperature_effects()
        super().tick()

    def _update_temperature_effects(self):
        """Update fluxes based on temperature"""
        # Warmer = more evaporation and precipitation
        temp_factor = 1.0 + (self.global_temperature - 15.0) * 0.03

        for flux in self.fluxes.values():
            if flux.flux_type == FluxType.EVAPORATION:
                flux.current_rate = flux.base_rate * temp_factor
            elif flux.flux_type == FluxType.PRECIPITATION:
                flux.current_rate = flux.base_rate * temp_factor

    def set_temperature(self, temperature: float):
        """Set global temperature for cycle effects"""
        self.global_temperature = temperature

    def get_ice_volume(self) -> float:
        """Get current ice volume"""
        return self._ice_caps.amount

    def get_atmospheric_water(self) -> float:
        """Get atmospheric water content"""
        return self._atmosphere.amount


# ============================================================================
# Integrated Planetary Cycles
# ============================================================================

class PlanetaryCycles:
    """
    Integrated planetary biogeochemical cycles.

    Couples carbon, nitrogen, water, and other cycles with
    feedback between them and the biosphere.

    Example:
        cycles = PlanetaryCycles()
        cycles.initialize()

        for _ in range(100):
            cycles.tick()

        status = cycles.get_global_status()
    """

    def __init__(self, config: Optional[CycleConfig] = None):
        self.config = config or CycleConfig()

        self.carbon_cycle = CarbonCycle(config)
        self.nitrogen_cycle = NitrogenCycle(config)
        self.water_cycle = WaterCycle(config)

        self.tick_count = 0
        self.global_temperature = 15.0

    def initialize(self):
        """Initialize all cycles"""
        self.carbon_cycle.initialize()
        self.nitrogen_cycle.initialize()
        self.water_cycle.initialize()

    def tick(self):
        """Process all cycles with coupling"""
        self.tick_count += 1

        # Update temperature effects
        if self.config.enable_coupling:
            self._update_coupled_effects()

        # Process each cycle
        self.carbon_cycle.tick()
        self.nitrogen_cycle.tick()
        self.water_cycle.tick()

    def _update_coupled_effects(self):
        """Update inter-cycle feedback"""
        # CO2 affects temperature (greenhouse effect)
        co2_ppm = self.carbon_cycle.get_atmospheric_co2()
        greenhouse_warming = 5.35 * np.log(co2_ppm / 280)
        self.global_temperature = 15.0 + greenhouse_warming * 0.5

        # Temperature affects water cycle
        self.water_cycle.set_temperature(self.global_temperature)

    def get_global_status(self) -> Dict[str, Any]:
        """Get status of all cycles"""
        return {
            "tick_count": self.tick_count,
            "global_temperature": self.global_temperature,
            "carbon": self.carbon_cycle.get_status(),
            "nitrogen": self.nitrogen_cycle.get_status(),
            "water": self.water_cycle.get_status()
        }

    def get_atmospheric_co2(self) -> float:
        """Get atmospheric CO2"""
        return self.carbon_cycle.get_atmospheric_co2()

    def emit_carbon(self, amount: float):
        """Emit carbon to atmosphere"""
        self.carbon_cycle.emit_anthropogenic(amount)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Planetary Resource Cycles Demo")
    print("=" * 50)

    # Create integrated cycles
    config = CycleConfig(
        time_step=1.0,
        enable_coupling=True
    )
    cycles = PlanetaryCycles(config)
    cycles.initialize()

    print(f"\n1. Initialized planetary cycles")
    print(f"   Initial CO2: {cycles.get_atmospheric_co2():.1f} ppm")

    # Run simulation
    print("\n2. Running simulation for 100 years...")
    for year in range(100):
        # Add some anthropogenic emissions
        cycles.emit_carbon(10)  # 10 Gt C/year
        cycles.tick()

        if year % 25 == 0:
            status = cycles.get_global_status()
            print(f"   Year {year}: CO2={status['carbon']['reservoirs']['atmosphere']:.0f} Gt, "
                  f"temp={status['global_temperature']:.2f}C")

    # Final status
    print("\n3. Final cycle status:")
    status = cycles.get_global_status()
    print(f"   Temperature: {status['global_temperature']:.2f}C")
    print(f"   CO2 (ppm): {cycles.get_atmospheric_co2():.1f}")

    print("\n4. Carbon reservoirs:")
    for name, amount in status["carbon"]["reservoirs"].items():
        print(f"   {name}: {amount:.0f} Gt")
