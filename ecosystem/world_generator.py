"""
Procedural World Generator for NPCPU

Generates varied environments using noise-based terrain generation with:
- Biome classification (forest, desert, ocean, etc.)
- Resource distribution based on biome rules
- Dynamic hazard placement
- Seed-based reproducibility

Example:
    >>> generator = WorldGenerator(seed=12345)
    >>> world = generator.generate(size=(100, 100))
    >>> biome = world.get_biome_at(50, 50)
    >>> print(f"Biome at center: {biome.name}")

    >>> # Or use ProceduralWorld directly
    >>> world = ProceduralWorld(seed=42, width=200, height=200)
    >>> world.spawn_resources()
    >>> stimuli = world.get_stimuli_at((100, 100))
"""

import math
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.abstractions import BaseWorld, BasePopulation
from core.events import get_event_bus, Event


# =============================================================================
# Enums
# =============================================================================

class BiomeType(Enum):
    """Types of biomes in the procedural world"""
    OCEAN = "ocean"
    BEACH = "beach"
    DESERT = "desert"
    SAVANNA = "savanna"
    GRASSLAND = "grassland"
    FOREST = "forest"
    RAINFOREST = "rainforest"
    TAIGA = "taiga"
    TUNDRA = "tundra"
    MOUNTAIN = "mountain"
    SNOW_PEAK = "snow_peak"
    SWAMP = "swamp"
    VOLCANIC = "volcanic"


class HazardType(Enum):
    """Types of environmental hazards"""
    NONE = "none"
    QUICKSAND = "quicksand"
    TOXIC_GAS = "toxic_gas"
    LAVA_FLOW = "lava_flow"
    DEEP_WATER = "deep_water"
    AVALANCHE_ZONE = "avalanche_zone"
    PREDATOR_DEN = "predator_den"
    RADIATION = "radiation"
    EXTREME_COLD = "extreme_cold"
    EXTREME_HEAT = "extreme_heat"


class ResourceType(Enum):
    """Types of resources that can be spawned"""
    FOOD = "food"
    WATER = "water"
    SHELTER = "shelter"
    ENERGY = "energy"
    MINERALS = "minerals"
    ORGANIC_MATTER = "organic_matter"
    RARE_ELEMENT = "rare_element"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Biome:
    """Represents a biome with its properties"""
    type: BiomeType
    name: str
    elevation_range: Tuple[float, float]  # (min, max)
    moisture_range: Tuple[float, float]   # (min, max)
    temperature_range: Tuple[float, float]  # (min, max)
    habitability: float = 0.5  # 0-1 how suitable for life
    movement_cost: float = 1.0  # Movement speed modifier
    resource_multiplier: float = 1.0  # Resource spawn rate modifier
    color: Tuple[int, int, int] = (128, 128, 128)  # RGB for visualization


@dataclass
class Resource:
    """A resource instance in the world"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ResourceType = ResourceType.FOOD
    x: float = 0.0
    y: float = 0.0
    amount: float = 100.0
    max_amount: float = 100.0
    regeneration_rate: float = 1.0
    quality: float = 1.0  # 0-1 resource quality

    def consume(self, amount: float) -> float:
        """Consume resources, returns actual amount consumed"""
        actual = min(amount, self.amount)
        self.amount -= actual
        return actual * self.quality

    def regenerate(self, delta_time: float = 1.0):
        """Regenerate resource over time"""
        self.amount = min(self.max_amount, self.amount + self.regeneration_rate * delta_time)

    @property
    def depleted(self) -> bool:
        return self.amount <= 0


@dataclass
class Hazard:
    """An environmental hazard in the world"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: HazardType = HazardType.NONE
    x: float = 0.0
    y: float = 0.0
    radius: float = 5.0
    intensity: float = 0.5  # 0-1 how dangerous
    active: bool = True
    periodic: bool = False  # Does it activate/deactivate?
    period: int = 100  # Ticks between state changes if periodic

    def affects_location(self, px: float, py: float) -> bool:
        """Check if hazard affects a location"""
        if not self.active:
            return False
        distance = math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        return distance <= self.radius

    def get_damage_at(self, px: float, py: float) -> float:
        """Get damage intensity at a location (falls off with distance)"""
        if not self.affects_location(px, py):
            return 0.0
        distance = math.sqrt((px - self.x) ** 2 + (py - self.y) ** 2)
        # Linear falloff from center
        falloff = 1.0 - (distance / self.radius)
        return self.intensity * falloff


@dataclass
class TerrainCell:
    """A single cell in the terrain grid"""
    x: int
    y: int
    elevation: float = 0.0  # -1 to 1
    moisture: float = 0.5   # 0 to 1
    temperature: float = 0.5  # 0 to 1 (cold to hot)
    biome: BiomeType = BiomeType.GRASSLAND
    resources: List[str] = field(default_factory=list)  # Resource IDs
    hazards: List[str] = field(default_factory=list)    # Hazard IDs


# =============================================================================
# Simplex-like Noise Implementation (no external dependency)
# =============================================================================

class SimplexNoise:
    """
    Simple noise generator for procedural terrain.
    Uses permutation tables for reproducible pseudo-random noise.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        random.seed(seed)

        # Create permutation table
        self.perm = list(range(256))
        random.shuffle(self.perm)
        self.perm = self.perm + self.perm  # Double it for overflow

        # Gradients for 2D
        self.gradients = [
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1)
        ]

    def _dot_grad(self, ix: int, iy: int, x: float, y: float) -> float:
        """Calculate dot product of distance and gradient vectors"""
        grad_idx = self.perm[(ix + self.perm[iy & 255]) & 255] % len(self.gradients)
        grad = self.gradients[grad_idx]
        return (x - ix) * grad[0] + (y - iy) * grad[1]

    def _fade(self, t: float) -> float:
        """Smoothstep interpolation"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation"""
        return a + t * (b - a)

    def noise2d(self, x: float, y: float) -> float:
        """
        Generate 2D Perlin-like noise at (x, y).
        Returns value in range [-1, 1].
        """
        # Grid cell coordinates
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        # Interpolation weights
        sx = self._fade(x - x0)
        sy = self._fade(y - y0)

        # Interpolate
        n00 = self._dot_grad(x0, y0, x, y)
        n10 = self._dot_grad(x1, y0, x, y)
        n01 = self._dot_grad(x0, y1, x, y)
        n11 = self._dot_grad(x1, y1, x, y)

        nx0 = self._lerp(n00, n10, sx)
        nx1 = self._lerp(n01, n11, sx)

        return self._lerp(nx0, nx1, sy)

    def octave_noise(
        self,
        x: float,
        y: float,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ) -> float:
        """
        Generate fractal noise using multiple octaves.
        Returns value normalized to approximately [-1, 1].
        """
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0

        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return total / max_value


# =============================================================================
# Biome Definitions
# =============================================================================

BIOME_DEFINITIONS: Dict[BiomeType, Biome] = {
    BiomeType.OCEAN: Biome(
        type=BiomeType.OCEAN,
        name="Ocean",
        elevation_range=(-1.0, -0.3),
        moisture_range=(0.0, 1.0),
        temperature_range=(0.0, 1.0),
        habitability=0.3,
        movement_cost=2.0,
        resource_multiplier=0.8,
        color=(30, 100, 180)
    ),
    BiomeType.BEACH: Biome(
        type=BiomeType.BEACH,
        name="Beach",
        elevation_range=(-0.3, -0.1),
        moisture_range=(0.3, 0.7),
        temperature_range=(0.3, 0.8),
        habitability=0.6,
        movement_cost=1.2,
        resource_multiplier=0.7,
        color=(238, 214, 175)
    ),
    BiomeType.DESERT: Biome(
        type=BiomeType.DESERT,
        name="Desert",
        elevation_range=(-0.1, 0.4),
        moisture_range=(0.0, 0.2),
        temperature_range=(0.6, 1.0),
        habitability=0.2,
        movement_cost=1.5,
        resource_multiplier=0.3,
        color=(237, 201, 175)
    ),
    BiomeType.SAVANNA: Biome(
        type=BiomeType.SAVANNA,
        name="Savanna",
        elevation_range=(-0.1, 0.3),
        moisture_range=(0.2, 0.4),
        temperature_range=(0.5, 0.9),
        habitability=0.6,
        movement_cost=1.0,
        resource_multiplier=0.9,
        color=(177, 209, 110)
    ),
    BiomeType.GRASSLAND: Biome(
        type=BiomeType.GRASSLAND,
        name="Grassland",
        elevation_range=(-0.1, 0.4),
        moisture_range=(0.3, 0.6),
        temperature_range=(0.3, 0.7),
        habitability=0.8,
        movement_cost=0.9,
        resource_multiplier=1.2,
        color=(140, 200, 100)
    ),
    BiomeType.FOREST: Biome(
        type=BiomeType.FOREST,
        name="Forest",
        elevation_range=(0.0, 0.5),
        moisture_range=(0.5, 0.8),
        temperature_range=(0.3, 0.7),
        habitability=0.85,
        movement_cost=1.3,
        resource_multiplier=1.5,
        color=(34, 139, 34)
    ),
    BiomeType.RAINFOREST: Biome(
        type=BiomeType.RAINFOREST,
        name="Rainforest",
        elevation_range=(-0.1, 0.4),
        moisture_range=(0.8, 1.0),
        temperature_range=(0.6, 1.0),
        habitability=0.9,
        movement_cost=1.5,
        resource_multiplier=2.0,
        color=(0, 100, 0)
    ),
    BiomeType.TAIGA: Biome(
        type=BiomeType.TAIGA,
        name="Taiga",
        elevation_range=(0.1, 0.5),
        moisture_range=(0.4, 0.7),
        temperature_range=(0.1, 0.4),
        habitability=0.5,
        movement_cost=1.2,
        resource_multiplier=0.8,
        color=(95, 130, 95)
    ),
    BiomeType.TUNDRA: Biome(
        type=BiomeType.TUNDRA,
        name="Tundra",
        elevation_range=(0.0, 0.5),
        moisture_range=(0.2, 0.5),
        temperature_range=(0.0, 0.2),
        habitability=0.3,
        movement_cost=1.4,
        resource_multiplier=0.4,
        color=(187, 206, 186)
    ),
    BiomeType.MOUNTAIN: Biome(
        type=BiomeType.MOUNTAIN,
        name="Mountain",
        elevation_range=(0.5, 0.8),
        moisture_range=(0.0, 1.0),
        temperature_range=(0.0, 1.0),
        habitability=0.4,
        movement_cost=2.0,
        resource_multiplier=0.6,
        color=(139, 137, 137)
    ),
    BiomeType.SNOW_PEAK: Biome(
        type=BiomeType.SNOW_PEAK,
        name="Snow Peak",
        elevation_range=(0.8, 1.0),
        moisture_range=(0.0, 1.0),
        temperature_range=(0.0, 0.2),
        habitability=0.1,
        movement_cost=2.5,
        resource_multiplier=0.2,
        color=(255, 250, 250)
    ),
    BiomeType.SWAMP: Biome(
        type=BiomeType.SWAMP,
        name="Swamp",
        elevation_range=(-0.2, 0.1),
        moisture_range=(0.7, 1.0),
        temperature_range=(0.4, 0.8),
        habitability=0.5,
        movement_cost=1.8,
        resource_multiplier=1.3,
        color=(70, 100, 70)
    ),
    BiomeType.VOLCANIC: Biome(
        type=BiomeType.VOLCANIC,
        name="Volcanic",
        elevation_range=(0.4, 0.9),
        moisture_range=(0.0, 0.3),
        temperature_range=(0.8, 1.0),
        habitability=0.1,
        movement_cost=2.0,
        resource_multiplier=0.5,
        color=(50, 30, 30)
    ),
}

# Resource spawn rules per biome
BIOME_RESOURCES: Dict[BiomeType, List[Tuple[ResourceType, float, float]]] = {
    # (resource_type, spawn_chance, quality_modifier)
    BiomeType.OCEAN: [
        (ResourceType.WATER, 1.0, 1.0),
        (ResourceType.FOOD, 0.6, 0.8),
    ],
    BiomeType.BEACH: [
        (ResourceType.WATER, 0.5, 0.6),
        (ResourceType.FOOD, 0.4, 0.7),
        (ResourceType.MINERALS, 0.3, 0.8),
    ],
    BiomeType.DESERT: [
        (ResourceType.MINERALS, 0.5, 1.2),
        (ResourceType.RARE_ELEMENT, 0.2, 1.5),
        (ResourceType.ENERGY, 0.4, 1.3),
    ],
    BiomeType.SAVANNA: [
        (ResourceType.FOOD, 0.6, 0.9),
        (ResourceType.WATER, 0.4, 0.7),
        (ResourceType.ORGANIC_MATTER, 0.5, 0.9),
    ],
    BiomeType.GRASSLAND: [
        (ResourceType.FOOD, 0.8, 1.0),
        (ResourceType.WATER, 0.6, 0.9),
        (ResourceType.SHELTER, 0.4, 0.8),
        (ResourceType.ORGANIC_MATTER, 0.6, 1.0),
    ],
    BiomeType.FOREST: [
        (ResourceType.FOOD, 0.9, 1.2),
        (ResourceType.WATER, 0.7, 1.0),
        (ResourceType.SHELTER, 0.8, 1.3),
        (ResourceType.ORGANIC_MATTER, 0.9, 1.2),
    ],
    BiomeType.RAINFOREST: [
        (ResourceType.FOOD, 1.0, 1.5),
        (ResourceType.WATER, 0.9, 1.2),
        (ResourceType.SHELTER, 0.7, 1.1),
        (ResourceType.ORGANIC_MATTER, 1.0, 1.4),
        (ResourceType.RARE_ELEMENT, 0.3, 1.2),
    ],
    BiomeType.TAIGA: [
        (ResourceType.FOOD, 0.5, 0.8),
        (ResourceType.WATER, 0.6, 0.9),
        (ResourceType.SHELTER, 0.6, 1.0),
        (ResourceType.ORGANIC_MATTER, 0.5, 0.8),
    ],
    BiomeType.TUNDRA: [
        (ResourceType.WATER, 0.7, 0.6),
        (ResourceType.MINERALS, 0.4, 1.0),
    ],
    BiomeType.MOUNTAIN: [
        (ResourceType.MINERALS, 0.8, 1.3),
        (ResourceType.RARE_ELEMENT, 0.4, 1.4),
        (ResourceType.SHELTER, 0.3, 0.9),
    ],
    BiomeType.SNOW_PEAK: [
        (ResourceType.WATER, 0.9, 0.5),
        (ResourceType.RARE_ELEMENT, 0.3, 1.6),
    ],
    BiomeType.SWAMP: [
        (ResourceType.WATER, 1.0, 0.5),
        (ResourceType.FOOD, 0.6, 0.7),
        (ResourceType.ORGANIC_MATTER, 0.8, 1.1),
    ],
    BiomeType.VOLCANIC: [
        (ResourceType.ENERGY, 0.9, 1.8),
        (ResourceType.MINERALS, 0.7, 1.5),
        (ResourceType.RARE_ELEMENT, 0.5, 2.0),
    ],
}

# Hazard spawn rules per biome
BIOME_HAZARDS: Dict[BiomeType, List[Tuple[HazardType, float, float]]] = {
    # (hazard_type, spawn_chance, intensity_modifier)
    BiomeType.OCEAN: [(HazardType.DEEP_WATER, 0.5, 0.7)],
    BiomeType.BEACH: [],
    BiomeType.DESERT: [
        (HazardType.EXTREME_HEAT, 0.6, 0.8),
        (HazardType.QUICKSAND, 0.2, 0.6),
    ],
    BiomeType.SAVANNA: [(HazardType.PREDATOR_DEN, 0.3, 0.5)],
    BiomeType.GRASSLAND: [(HazardType.PREDATOR_DEN, 0.1, 0.3)],
    BiomeType.FOREST: [(HazardType.PREDATOR_DEN, 0.2, 0.4)],
    BiomeType.RAINFOREST: [
        (HazardType.TOXIC_GAS, 0.3, 0.5),
        (HazardType.PREDATOR_DEN, 0.4, 0.6),
    ],
    BiomeType.TAIGA: [(HazardType.EXTREME_COLD, 0.3, 0.5)],
    BiomeType.TUNDRA: [(HazardType.EXTREME_COLD, 0.7, 0.8)],
    BiomeType.MOUNTAIN: [(HazardType.AVALANCHE_ZONE, 0.4, 0.7)],
    BiomeType.SNOW_PEAK: [
        (HazardType.EXTREME_COLD, 0.9, 1.0),
        (HazardType.AVALANCHE_ZONE, 0.6, 0.9),
    ],
    BiomeType.SWAMP: [
        (HazardType.TOXIC_GAS, 0.4, 0.6),
        (HazardType.QUICKSAND, 0.3, 0.7),
    ],
    BiomeType.VOLCANIC: [
        (HazardType.LAVA_FLOW, 0.7, 0.9),
        (HazardType.TOXIC_GAS, 0.5, 0.7),
        (HazardType.EXTREME_HEAT, 0.8, 1.0),
    ],
}


# =============================================================================
# World Generator
# =============================================================================

class WorldGenerator:
    """
    Procedural world generation with biomes, resources, and hazards.

    Features:
    - Noise-based terrain generation
    - Biome classification (forest, desert, ocean, etc.)
    - Resource distribution based on biome
    - Dynamic hazard placement

    Example:
        >>> generator = WorldGenerator(seed=12345)
        >>> terrain = generator.generate(size=(100, 100))
        >>> biome = generator.get_biome_at_cell(terrain, 50, 50)
        >>> print(f"Center biome: {biome.name}")
    """

    def __init__(
        self,
        seed: int = 0,
        elevation_scale: float = 0.02,
        moisture_scale: float = 0.03,
        temperature_scale: float = 0.025,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ):
        """
        Initialize the world generator.

        Args:
            seed: Random seed for reproducible generation
            elevation_scale: Scale factor for elevation noise
            moisture_scale: Scale factor for moisture noise
            temperature_scale: Scale factor for temperature noise
            octaves: Number of noise octaves for detail
            persistence: Amplitude falloff per octave
            lacunarity: Frequency increase per octave
        """
        self.seed = seed
        self.elevation_scale = elevation_scale
        self.moisture_scale = moisture_scale
        self.temperature_scale = temperature_scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        # Initialize noise generators with different seeds for each layer
        self._elevation_noise = SimplexNoise(seed)
        self._moisture_noise = SimplexNoise(seed + 1000)
        self._temperature_noise = SimplexNoise(seed + 2000)

        # Random generator for resource/hazard placement
        self._rng = random.Random(seed)

    def generate(
        self,
        size: Tuple[int, int] = (100, 100),
        latitude_effect: bool = True
    ) -> List[List[TerrainCell]]:
        """
        Generate world terrain from seed.

        Args:
            size: (width, height) of the world in cells
            latitude_effect: Apply temperature gradient based on y-position

        Returns:
            2D grid of TerrainCell objects
        """
        width, height = size
        terrain: List[List[TerrainCell]] = []

        for y in range(height):
            row: List[TerrainCell] = []
            for x in range(width):
                cell = self._generate_cell(x, y, width, height, latitude_effect)
                row.append(cell)
            terrain.append(row)

        return terrain

    def _generate_cell(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        latitude_effect: bool
    ) -> TerrainCell:
        """Generate a single terrain cell"""
        # Generate noise values
        elevation = self._elevation_noise.octave_noise(
            x * self.elevation_scale,
            y * self.elevation_scale,
            self.octaves,
            self.persistence,
            self.lacunarity
        )

        moisture = self._moisture_noise.octave_noise(
            x * self.moisture_scale,
            y * self.moisture_scale,
            self.octaves - 1,
            self.persistence,
            self.lacunarity
        )
        # Normalize moisture to 0-1
        moisture = (moisture + 1) / 2

        temperature = self._temperature_noise.octave_noise(
            x * self.temperature_scale,
            y * self.temperature_scale,
            self.octaves - 2,
            self.persistence,
            self.lacunarity
        )
        # Normalize temperature to 0-1
        temperature = (temperature + 1) / 2

        # Apply latitude effect (colder at top and bottom, warmer in middle)
        if latitude_effect and height > 0:
            latitude_factor = 1 - abs((y / height) - 0.5) * 2
            temperature = temperature * 0.6 + latitude_factor * 0.4

        # Lower temperature at high elevations
        if elevation > 0.3:
            temperature *= 1 - (elevation - 0.3) * 0.5

        # Increase moisture near oceans (low elevation)
        if elevation < 0:
            moisture = min(1.0, moisture + 0.2)

        # Classify biome
        biome = self._classify_biome(elevation, moisture, temperature)

        return TerrainCell(
            x=x,
            y=y,
            elevation=elevation,
            moisture=moisture,
            temperature=temperature,
            biome=biome
        )

    def _classify_biome(
        self,
        elevation: float,
        moisture: float,
        temperature: float
    ) -> BiomeType:
        """
        Classify biome based on elevation, moisture, and temperature.
        Uses Whittaker-inspired biome classification.
        """
        # Ocean for very low elevation
        if elevation < -0.3:
            return BiomeType.OCEAN

        # Beach at ocean edges
        if elevation < -0.1:
            return BiomeType.BEACH

        # Snow peaks at highest elevation
        if elevation > 0.8:
            return BiomeType.SNOW_PEAK

        # Mountains at high elevation
        if elevation > 0.5:
            return BiomeType.MOUNTAIN

        # Volcanic: high elevation + high temperature + low moisture
        if elevation > 0.4 and temperature > 0.8 and moisture < 0.3:
            return BiomeType.VOLCANIC

        # Swamp: low elevation + high moisture
        if elevation < 0.1 and moisture > 0.7:
            return BiomeType.SWAMP

        # Temperature and moisture-based biomes
        if temperature < 0.2:
            # Cold biomes
            if moisture < 0.5:
                return BiomeType.TUNDRA
            else:
                return BiomeType.TAIGA
        elif temperature < 0.4:
            # Cool biomes
            if moisture > 0.5:
                return BiomeType.FOREST
            elif moisture > 0.3:
                return BiomeType.GRASSLAND
            else:
                return BiomeType.TUNDRA
        elif temperature < 0.7:
            # Temperate biomes
            if moisture > 0.8:
                return BiomeType.RAINFOREST
            elif moisture > 0.5:
                return BiomeType.FOREST
            elif moisture > 0.3:
                return BiomeType.GRASSLAND
            else:
                return BiomeType.SAVANNA
        else:
            # Hot biomes
            if moisture > 0.8:
                return BiomeType.RAINFOREST
            elif moisture > 0.4:
                return BiomeType.SAVANNA
            elif moisture > 0.2:
                return BiomeType.SAVANNA
            else:
                return BiomeType.DESERT

    def get_biome_at_cell(
        self,
        terrain: List[List[TerrainCell]],
        x: int,
        y: int
    ) -> Biome:
        """
        Get biome definition at a cell location.

        Args:
            terrain: Generated terrain grid
            x: X coordinate
            y: Y coordinate

        Returns:
            Biome definition for the cell
        """
        if 0 <= y < len(terrain) and 0 <= x < len(terrain[0]):
            biome_type = terrain[y][x].biome
            return BIOME_DEFINITIONS[biome_type]
        return BIOME_DEFINITIONS[BiomeType.OCEAN]

    def spawn_resources_on_terrain(
        self,
        terrain: List[List[TerrainCell]],
        density: float = 0.1
    ) -> List[Resource]:
        """
        Spawn resources on terrain based on biome rules.

        Args:
            terrain: Generated terrain grid
            density: Base resource density (0-1)

        Returns:
            List of spawned Resource objects
        """
        resources: List[Resource] = []

        for row in terrain:
            for cell in row:
                biome_rules = BIOME_RESOURCES.get(cell.biome, [])
                biome_def = BIOME_DEFINITIONS[cell.biome]

                for resource_type, spawn_chance, quality_mod in biome_rules:
                    # Apply biome multiplier and density
                    effective_chance = spawn_chance * density * biome_def.resource_multiplier

                    if self._rng.random() < effective_chance:
                        resource = Resource(
                            type=resource_type,
                            x=cell.x + self._rng.random(),
                            y=cell.y + self._rng.random(),
                            amount=self._rng.uniform(50, 150),
                            max_amount=150.0,
                            regeneration_rate=self._rng.uniform(0.5, 2.0),
                            quality=quality_mod * self._rng.uniform(0.8, 1.2)
                        )
                        resources.append(resource)
                        cell.resources.append(resource.id)

        return resources

    def spawn_hazards_on_terrain(
        self,
        terrain: List[List[TerrainCell]],
        density: float = 0.05
    ) -> List[Hazard]:
        """
        Spawn hazards on terrain based on biome rules.

        Args:
            terrain: Generated terrain grid
            density: Base hazard density (0-1)

        Returns:
            List of spawned Hazard objects
        """
        hazards: List[Hazard] = []

        for row in terrain:
            for cell in row:
                hazard_rules = BIOME_HAZARDS.get(cell.biome, [])

                for hazard_type, spawn_chance, intensity_mod in hazard_rules:
                    effective_chance = spawn_chance * density

                    if self._rng.random() < effective_chance:
                        hazard = Hazard(
                            type=hazard_type,
                            x=cell.x + self._rng.random(),
                            y=cell.y + self._rng.random(),
                            radius=self._rng.uniform(2.0, 8.0),
                            intensity=intensity_mod * self._rng.uniform(0.3, 0.9),
                            periodic=self._rng.random() < 0.2,
                            period=self._rng.randint(50, 200)
                        )
                        hazards.append(hazard)
                        cell.hazards.append(hazard.id)

        return hazards


# =============================================================================
# Procedural World (BaseWorld Implementation)
# =============================================================================

class ProceduralWorld(BaseWorld):
    """
    World implementation using procedural generation.

    Extends BaseWorld with noise-based terrain, biome system,
    procedurally placed resources and hazards.

    Example:
        >>> world = ProceduralWorld(seed=42, width=100, height=100)
        >>> world.spawn_resources()
        >>> world.tick()
        >>> stimuli = world.get_stimuli_at((50, 50))
        >>> print(f"Biome: {stimuli['biome']}, Resources nearby: {len(stimuli['resources'])}")
    """

    def __init__(
        self,
        name: str = "Procedural World",
        seed: int = 0,
        width: int = 100,
        height: int = 100,
        resource_density: float = 0.1,
        hazard_density: float = 0.05,
        **kwargs
    ):
        """
        Initialize procedural world.

        Args:
            name: World name
            seed: Random seed for generation
            width: World width in cells
            height: World height in cells
            resource_density: Base resource spawn density
            hazard_density: Base hazard spawn density
        """
        super().__init__(name, **kwargs)

        self.seed = seed
        self.width = width
        self.height = height
        self.resource_density = resource_density
        self.hazard_density = hazard_density

        # Initialize generator
        self._generator = WorldGenerator(seed)

        # Generate terrain
        self._terrain: List[List[TerrainCell]] = self._generator.generate(
            size=(width, height)
        )

        # Storage for resources and hazards
        self._resources: Dict[str, Resource] = {}
        self._hazards: Dict[str, Hazard] = {}

        # Resource pools (for BaseWorld interface)
        self._resource_pools: Dict[str, float] = defaultdict(float)

        # Event bus
        self._event_bus = get_event_bus()

        # World time
        self._time = 0.0
        self._day_length = 100  # ticks per day

    @property
    def terrain(self) -> List[List[TerrainCell]]:
        """Get the terrain grid"""
        return self._terrain

    @property
    def resources(self) -> Dict[str, Resource]:
        """Get all resources"""
        return self._resources

    @property
    def hazards(self) -> Dict[str, Hazard]:
        """Get all hazards"""
        return self._hazards

    def get_biome_at(self, x: float, y: float) -> Biome:
        """
        Get biome type at location.

        Args:
            x: X coordinate (can be float)
            y: Y coordinate (can be float)

        Returns:
            Biome definition at the location
        """
        cell_x = int(x) % self.width
        cell_y = int(y) % self.height

        if 0 <= cell_y < len(self._terrain) and 0 <= cell_x < len(self._terrain[0]):
            biome_type = self._terrain[cell_y][cell_x].biome
            return BIOME_DEFINITIONS[biome_type]
        return BIOME_DEFINITIONS[BiomeType.OCEAN]

    def get_cell_at(self, x: float, y: float) -> Optional[TerrainCell]:
        """Get terrain cell at location"""
        cell_x = int(x) % self.width
        cell_y = int(y) % self.height

        if 0 <= cell_y < len(self._terrain) and 0 <= cell_x < len(self._terrain[0]):
            return self._terrain[cell_y][cell_x]
        return None

    def spawn_resources(self) -> int:
        """
        Distribute resources by biome rules.

        Returns:
            Number of resources spawned
        """
        resources = self._generator.spawn_resources_on_terrain(
            self._terrain,
            self.resource_density
        )

        for resource in resources:
            self._resources[resource.id] = resource
            self._resource_pools[resource.type.value] += resource.amount

        self._event_bus.emit("world.resources_spawned", {
            "world_id": self._id,
            "count": len(resources)
        })

        return len(resources)

    def spawn_hazards(self) -> int:
        """
        Place hazards based on biome rules.

        Returns:
            Number of hazards spawned
        """
        hazards = self._generator.spawn_hazards_on_terrain(
            self._terrain,
            self.hazard_density
        )

        for hazard in hazards:
            self._hazards[hazard.id] = hazard

        self._event_bus.emit("world.hazards_spawned", {
            "world_id": self._id,
            "count": len(hazards)
        })

        return len(hazards)

    def get_resources_near(
        self,
        x: float,
        y: float,
        radius: float = 5.0,
        resource_type: Optional[ResourceType] = None
    ) -> List[Resource]:
        """Get resources within radius of a location"""
        nearby: List[Resource] = []

        for resource in self._resources.values():
            if resource.depleted:
                continue

            if resource_type and resource.type != resource_type:
                continue

            distance = math.sqrt((resource.x - x) ** 2 + (resource.y - y) ** 2)
            if distance <= radius:
                nearby.append(resource)

        return nearby

    def get_hazards_at(self, x: float, y: float) -> List[Tuple[Hazard, float]]:
        """
        Get active hazards affecting a location.

        Returns:
            List of (hazard, damage_intensity) tuples
        """
        affecting: List[Tuple[Hazard, float]] = []

        for hazard in self._hazards.values():
            damage = hazard.get_damage_at(x, y)
            if damage > 0:
                affecting.append((hazard, damage))

        return affecting

    # =========================================================================
    # BaseWorld Interface Implementation
    # =========================================================================

    def tick(self) -> None:
        """Advance world by one time step"""
        self._tick_count += 1
        self._time += 1.0

        # Regenerate resources
        for resource in self._resources.values():
            if not resource.depleted:
                resource.regenerate()

        # Toggle periodic hazards
        for hazard in self._hazards.values():
            if hazard.periodic and self._tick_count % hazard.period == 0:
                hazard.active = not hazard.active

        # Process populations
        for population in self._populations.values():
            population.tick()

        # Emit tick event
        self._event_bus.emit("world.tick", {
            "world_id": self._id,
            "tick": self._tick_count,
            "time_of_day": self._time % self._day_length
        })

    def get_stimuli_at(self, location: Any) -> Dict[str, Any]:
        """
        Get environmental stimuli at a location.

        Args:
            location: (x, y) tuple or object with x, y attributes

        Returns:
            Dictionary of environmental stimuli
        """
        if isinstance(location, tuple):
            x, y = location
        elif hasattr(location, 'x') and hasattr(location, 'y'):
            x, y = location.x, location.y
        else:
            return {}

        cell = self.get_cell_at(x, y)
        if not cell:
            return {}

        biome = self.get_biome_at(x, y)
        nearby_resources = self.get_resources_near(x, y)
        active_hazards = self.get_hazards_at(x, y)

        # Calculate time of day (0-1)
        time_of_day = (self._time % self._day_length) / self._day_length
        is_day = 0.25 < time_of_day < 0.75

        return {
            "biome": biome.name,
            "biome_type": biome.type.value,
            "elevation": cell.elevation,
            "moisture": cell.moisture,
            "temperature": cell.temperature,
            "habitability": biome.habitability,
            "movement_cost": biome.movement_cost,
            "resources": [
                {"type": r.type.value, "amount": r.amount, "quality": r.quality, "x": r.x, "y": r.y}
                for r in nearby_resources
            ],
            "hazards": [
                {"type": h.type.value, "intensity": intensity}
                for h, intensity in active_hazards
            ],
            "total_hazard_damage": sum(d for _, d in active_hazards),
            "time_of_day": time_of_day,
            "is_day": is_day,
            "tick": self._tick_count
        }

    def get_resources(self, resource_type: str) -> float:
        """Get available amount of a resource type globally"""
        return self._resource_pools.get(resource_type, 0.0)

    def consume_resource(self, resource_type: str, amount: float) -> float:
        """
        Consume resources globally, returns actual amount consumed.
        Prefers consuming from depleted resources first.
        """
        remaining = amount
        consumed = 0.0

        for resource in self._resources.values():
            if resource.type.value == resource_type and not resource.depleted:
                taken = resource.consume(min(remaining, resource.amount))
                consumed += taken
                remaining -= taken

                if remaining <= 0:
                    break

        self._resource_pools[resource_type] -= consumed
        return consumed

    def consume_resource_at(
        self,
        x: float,
        y: float,
        resource_type: ResourceType,
        amount: float,
        radius: float = 3.0
    ) -> float:
        """
        Consume resources at a specific location.

        Args:
            x: X coordinate
            y: Y coordinate
            resource_type: Type of resource to consume
            amount: Amount to consume
            radius: Search radius

        Returns:
            Actual amount consumed
        """
        nearby = self.get_resources_near(x, y, radius, resource_type)

        if not nearby:
            return 0.0

        # Sort by distance, consume closest first
        nearby.sort(key=lambda r: (r.x - x) ** 2 + (r.y - y) ** 2)

        remaining = amount
        consumed = 0.0

        for resource in nearby:
            taken = resource.consume(min(remaining, resource.amount))
            consumed += taken
            remaining -= taken

            if remaining <= 0:
                break

        self._resource_pools[resource_type.value] -= consumed
        return consumed

    def trigger_event(self, event_type: str, **kwargs) -> None:
        """Trigger a world event"""
        self._event_bus.emit(f"world.event.{event_type}", {
            "world_id": self._id,
            "tick": self._tick_count,
            **kwargs
        })

        # Handle specific event types
        if event_type == "resource_bloom":
            self._handle_resource_bloom(kwargs)
        elif event_type == "disaster":
            self._handle_disaster(kwargs)
        elif event_type == "climate_shift":
            self._handle_climate_shift(kwargs)

    def _handle_resource_bloom(self, kwargs: Dict[str, Any]):
        """Handle resource bloom event - spawn extra resources in an area"""
        x = kwargs.get("x", self.width / 2)
        y = kwargs.get("y", self.height / 2)
        radius = kwargs.get("radius", 20)
        multiplier = kwargs.get("multiplier", 2.0)

        for resource in self._resources.values():
            distance = math.sqrt((resource.x - x) ** 2 + (resource.y - y) ** 2)
            if distance <= radius:
                resource.amount = min(
                    resource.max_amount,
                    resource.amount * multiplier
                )
                self._resource_pools[resource.type.value] += resource.amount * (multiplier - 1)

    def _handle_disaster(self, kwargs: Dict[str, Any]):
        """Handle disaster event - add temporary hazards"""
        x = kwargs.get("x", self.width / 2)
        y = kwargs.get("y", self.height / 2)
        hazard_type = kwargs.get("hazard_type", HazardType.TOXIC_GAS)

        hazard = Hazard(
            type=hazard_type,
            x=x,
            y=y,
            radius=kwargs.get("radius", 15),
            intensity=kwargs.get("intensity", 0.8),
            periodic=True,
            period=kwargs.get("duration", 50)
        )
        self._hazards[hazard.id] = hazard

    def _handle_climate_shift(self, kwargs: Dict[str, Any]):
        """Handle climate shift - modify temperature across the world"""
        delta = kwargs.get("temperature_delta", 0.1)

        for row in self._terrain:
            for cell in row:
                cell.temperature = max(0, min(1, cell.temperature + delta))

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize world state"""
        base = super().to_dict()
        base.update({
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "resource_count": len(self._resources),
            "hazard_count": len(self._hazards),
            "resource_pools": dict(self._resource_pools),
            "time": self._time
        })
        return base

    def get_map_data(self) -> Dict[str, Any]:
        """Get data for visualization"""
        biome_map: List[List[str]] = []
        elevation_map: List[List[float]] = []

        for row in self._terrain:
            biome_row: List[str] = []
            elevation_row: List[float] = []
            for cell in row:
                biome_row.append(cell.biome.value)
                elevation_row.append(cell.elevation)
            biome_map.append(biome_row)
            elevation_map.append(elevation_row)

        return {
            "width": self.width,
            "height": self.height,
            "biomes": biome_map,
            "elevation": elevation_map,
            "resources": [
                {"x": r.x, "y": r.y, "type": r.type.value, "amount": r.amount}
                for r in self._resources.values()
                if not r.depleted
            ],
            "hazards": [
                {"x": h.x, "y": h.y, "type": h.type.value, "radius": h.radius, "active": h.active}
                for h in self._hazards.values()
            ]
        }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Procedural World Generator Demo")
    print("=" * 60)

    # Create generator
    generator = WorldGenerator(seed=42)

    print("\n1. Generating terrain (50x50)...")
    terrain = generator.generate(size=(50, 50))

    # Count biomes
    biome_counts: Dict[BiomeType, int] = defaultdict(int)
    for row in terrain:
        for cell in row:
            biome_counts[cell.biome] += 1

    print("\n2. Biome distribution:")
    for biome, count in sorted(biome_counts.items(), key=lambda x: -x[1]):
        pct = (count / (50 * 50)) * 100
        print(f"   {biome.value:15} {count:4} cells ({pct:5.1f}%)")

    # Spawn resources
    print("\n3. Spawning resources...")
    resources = generator.spawn_resources_on_terrain(terrain, density=0.15)

    resource_counts: Dict[ResourceType, int] = defaultdict(int)
    for r in resources:
        resource_counts[r.type] += 1

    print(f"   Total resources: {len(resources)}")
    for rtype, count in sorted(resource_counts.items(), key=lambda x: -x[1]):
        print(f"   {rtype.value:15} {count:4}")

    # Spawn hazards
    print("\n4. Spawning hazards...")
    hazards = generator.spawn_hazards_on_terrain(terrain, density=0.08)

    hazard_counts: Dict[HazardType, int] = defaultdict(int)
    for h in hazards:
        hazard_counts[h.type] += 1

    print(f"   Total hazards: {len(hazards)}")
    for htype, count in sorted(hazard_counts.items(), key=lambda x: -x[1]):
        print(f"   {htype.value:15} {count:4}")

    # Create ProceduralWorld
    print("\n5. Creating ProceduralWorld...")
    world = ProceduralWorld(seed=12345, width=80, height=80)
    num_resources = world.spawn_resources()
    num_hazards = world.spawn_hazards()

    print(f"   World: {world.name}")
    print(f"   Size: {world.width}x{world.height}")
    print(f"   Resources: {num_resources}")
    print(f"   Hazards: {num_hazards}")

    # Sample stimuli at a location
    print("\n6. Environmental stimuli at (40, 40):")
    stimuli = world.get_stimuli_at((40, 40))
    for key, value in stimuli.items():
        if isinstance(value, list):
            print(f"   {key}: {len(value)} items")
        elif isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    # Run a few ticks
    print("\n7. Running 10 world ticks...")
    for _ in range(10):
        world.tick()

    print(f"   Tick count: {world.tick_count}")
    print(f"   World state: {world.to_dict()}")

    print("\nDone!")
