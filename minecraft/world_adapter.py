"""
Minecraft World Adapter

Maps Minecraft world structure to NPCPU ecosystem/world model.
Enables running full ecosystem simulations within Minecraft.

Mappings:
- Biomes → Regions with different conditions
- Resources → Resource pools
- Time/Weather → Environmental conditions
- Mobs → Threats and opportunities
- Players → Social entities
"""

import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minecraft.bridge import MinecraftBridge, Position, WorldState
from ecosystem.world import World, WorldConfig, Region, RegionType, Climate
from ecosystem.population import Population


# ============================================================================
# Minecraft Block/Entity Types
# ============================================================================

class BlockType(Enum):
    """Minecraft block categories"""
    # Resources
    WOOD = "wood"
    STONE = "stone"
    ORE = "ore"
    FOOD = "food"
    WATER = "water"
    LAVA = "lava"

    # Structures
    CRAFTING = "crafting"
    STORAGE = "storage"
    SHELTER = "shelter"

    # Terrain
    GROUND = "ground"
    AIR = "air"


class EntityType(Enum):
    """Minecraft entity categories"""
    PLAYER = "player"
    HOSTILE = "hostile"
    PASSIVE = "passive"
    NEUTRAL = "neutral"
    ITEM = "item"
    VEHICLE = "vehicle"


# Block type mappings
BLOCK_CATEGORIES = {
    BlockType.WOOD: ["oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log"],
    BlockType.STONE: ["stone", "cobblestone", "granite", "diorite", "andesite", "deepslate"],
    BlockType.ORE: ["coal_ore", "iron_ore", "gold_ore", "diamond_ore", "copper_ore", "redstone_ore", "lapis_ore", "emerald_ore"],
    BlockType.FOOD: ["wheat", "carrots", "potatoes", "beetroots", "melon", "pumpkin", "sweet_berry_bush"],
    BlockType.WATER: ["water"],
    BlockType.LAVA: ["lava"],
    BlockType.CRAFTING: ["crafting_table", "furnace", "blast_furnace", "smoker", "anvil"],
    BlockType.STORAGE: ["chest", "barrel", "shulker_box"],
}

# Entity type mappings
ENTITY_CATEGORIES = {
    EntityType.HOSTILE: ["zombie", "skeleton", "creeper", "spider", "enderman", "witch", "slime", "phantom", "drowned", "husk", "stray", "pillager", "vindicator"],
    EntityType.PASSIVE: ["cow", "pig", "sheep", "chicken", "rabbit", "horse", "donkey", "llama", "villager"],
    EntityType.NEUTRAL: ["wolf", "bee", "polar_bear", "iron_golem", "dolphin"],
}


# ============================================================================
# Biome to Region Mapping
# ============================================================================

BIOME_MAPPINGS = {
    # Fertile regions
    "plains": {"type": RegionType.FERTILE, "climate": Climate.OPTIMAL, "danger": 0.2, "resources": 0.7},
    "sunflower_plains": {"type": RegionType.FERTILE, "climate": Climate.OPTIMAL, "danger": 0.1, "resources": 0.8},
    "forest": {"type": RegionType.FERTILE, "climate": Climate.OPTIMAL, "danger": 0.3, "resources": 0.8},
    "flower_forest": {"type": RegionType.FERTILE, "climate": Climate.OPTIMAL, "danger": 0.2, "resources": 0.7},
    "meadow": {"type": RegionType.FERTILE, "climate": Climate.OPTIMAL, "danger": 0.1, "resources": 0.6},

    # Core regions (safe)
    "mushroom_fields": {"type": RegionType.SANCTUARY, "climate": Climate.OPTIMAL, "danger": 0.0, "resources": 0.5},

    # Harsh regions
    "desert": {"type": RegionType.BARREN, "climate": Climate.EXTREME_HEAT, "danger": 0.4, "resources": 0.3},
    "badlands": {"type": RegionType.BARREN, "climate": Climate.EXTREME_HEAT, "danger": 0.4, "resources": 0.4},
    "ice_spikes": {"type": RegionType.BARREN, "climate": Climate.EXTREME_COLD, "danger": 0.5, "resources": 0.2},
    "snowy_plains": {"type": RegionType.BARREN, "climate": Climate.COLD, "danger": 0.4, "resources": 0.3},

    # Frontier regions
    "jungle": {"type": RegionType.FRONTIER, "climate": Climate.WARM, "danger": 0.5, "resources": 0.9},
    "bamboo_jungle": {"type": RegionType.FRONTIER, "climate": Climate.WARM, "danger": 0.4, "resources": 0.8},
    "swamp": {"type": RegionType.FRONTIER, "climate": Climate.WARM, "danger": 0.6, "resources": 0.6},
    "mangrove_swamp": {"type": RegionType.FRONTIER, "climate": Climate.WARM, "danger": 0.5, "resources": 0.7},

    # Hostile regions
    "deep_dark": {"type": RegionType.HOSTILE, "climate": Climate.COLD, "danger": 0.9, "resources": 0.6},
    "nether_wastes": {"type": RegionType.HOSTILE, "climate": Climate.EXTREME_HEAT, "danger": 0.8, "resources": 0.5},
    "soul_sand_valley": {"type": RegionType.HOSTILE, "climate": Climate.EXTREME_HEAT, "danger": 0.9, "resources": 0.4},
    "basalt_deltas": {"type": RegionType.HOSTILE, "climate": Climate.EXTREME_HEAT, "danger": 0.85, "resources": 0.3},

    # Default
    "default": {"type": RegionType.CORE, "climate": Climate.NEUTRAL, "danger": 0.3, "resources": 0.5}
}


# ============================================================================
# Minecraft Region
# ============================================================================

@dataclass
class MinecraftRegion:
    """A region in the Minecraft world (based on biome)"""
    biome: str
    center: Position
    radius: float = 64.0
    type: RegionType = RegionType.CORE
    climate: Climate = Climate.NEUTRAL
    danger_level: float = 0.3
    resource_density: float = 0.5
    discovered_resources: Dict[BlockType, List[Position]] = field(default_factory=dict)
    known_entities: Dict[int, Any] = field(default_factory=dict)
    visit_count: int = 0
    last_visited: float = field(default_factory=time.time)


# ============================================================================
# Minecraft World Adapter
# ============================================================================

class MinecraftWorld:
    """
    Adapts Minecraft world to NPCPU ecosystem world model.

    Features:
    - Biome-based region mapping
    - Resource tracking per region
    - Danger assessment
    - Day/night cycles
    - Weather effects

    Example:
        mc_world = MinecraftWorld(bridge)

        # Scan current region
        region = await mc_world.get_current_region()

        # Get resources nearby
        resources = await mc_world.scan_resources()

        # Assess danger level
        danger = mc_world.get_danger_level()
    """

    def __init__(self, bridge: MinecraftBridge):
        self.bridge = bridge

        # Discovered regions
        self.regions: Dict[str, MinecraftRegion] = {}

        # Current region
        self.current_region: Optional[MinecraftRegion] = None

        # World state cache
        self.cached_world_state: Optional[WorldState] = None
        self.cache_time: float = 0.0
        self.cache_duration: float = 1.0  # Cache for 1 second

        # Resource scan results
        self.resource_cache: Dict[BlockType, List[Position]] = {}

    async def update(self):
        """Update world state from Minecraft"""
        current_time = time.time()
        if current_time - self.cache_time > self.cache_duration:
            self.cached_world_state = await self.bridge.get_world_state()
            self.cache_time = current_time

    async def get_current_region(self) -> MinecraftRegion:
        """Get the region at current position"""
        await self.update()

        biome = self.cached_world_state.biome if self.cached_world_state else "plains"
        pos = self.bridge.bot_state.position

        # Create region key
        region_key = f"{biome}_{int(pos.x // 64)}_{int(pos.z // 64)}"

        if region_key not in self.regions:
            # Create new region
            biome_config = BIOME_MAPPINGS.get(biome, BIOME_MAPPINGS["default"])

            self.regions[region_key] = MinecraftRegion(
                biome=biome,
                center=Position(pos.x, pos.y, pos.z),
                type=biome_config["type"],
                climate=biome_config["climate"],
                danger_level=biome_config["danger"],
                resource_density=biome_config["resources"]
            )

        region = self.regions[region_key]
        region.visit_count += 1
        region.last_visited = time.time()
        self.current_region = region

        return region

    async def scan_resources(self, radius: int = 32) -> Dict[BlockType, List[Position]]:
        """Scan for resources in current area"""
        resources = {}

        for block_type, block_names in BLOCK_CATEGORIES.items():
            positions = []
            for block_name in block_names[:2]:  # Limit scans
                found = await self.bridge.find_blocks(block_name, radius)
                positions.extend(found)
            if positions:
                resources[block_type] = positions

        self.resource_cache = resources

        # Update region if available
        if self.current_region:
            self.current_region.discovered_resources = resources

        return resources

    def get_danger_level(self) -> float:
        """Calculate current danger level"""
        base_danger = 0.0

        # Biome danger
        if self.current_region:
            base_danger = self.current_region.danger_level

        # Time-based danger (night is dangerous)
        if self.cached_world_state:
            time_of_day = self.cached_world_state.time_of_day
            if 12000 <= time_of_day <= 24000:  # Night
                base_danger += 0.3

            # Weather danger
            if self.cached_world_state.weather == "thunder":
                base_danger += 0.2
            elif self.cached_world_state.weather == "rain":
                base_danger += 0.05

        # Entity-based danger
        if self.cached_world_state:
            hostile_count = sum(
                1 for e in self.cached_world_state.nearby_entities
                if e.is_hostile
            )
            base_danger += min(0.4, hostile_count * 0.1)

        return min(1.0, base_danger)

    def get_resource_availability(self) -> float:
        """Calculate resource availability"""
        if not self.resource_cache:
            return 0.5

        total_resources = sum(len(positions) for positions in self.resource_cache.values())
        return min(1.0, total_resources / 50.0)

    def is_night(self) -> bool:
        """Check if it's night time"""
        if self.cached_world_state:
            return 12000 <= self.cached_world_state.time_of_day <= 24000
        return False

    def is_raining(self) -> bool:
        """Check if it's raining"""
        if self.cached_world_state:
            return self.cached_world_state.weather in ["rain", "thunder"]
        return False

    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of current environment"""
        return {
            "biome": self.cached_world_state.biome if self.cached_world_state else "unknown",
            "dimension": self.cached_world_state.dimension if self.cached_world_state else "overworld",
            "time_of_day": self.cached_world_state.time_of_day if self.cached_world_state else 0,
            "is_night": self.is_night(),
            "weather": self.cached_world_state.weather if self.cached_world_state else "clear",
            "danger_level": self.get_danger_level(),
            "resource_availability": self.get_resource_availability(),
            "region_type": self.current_region.type.value if self.current_region else "unknown",
            "regions_discovered": len(self.regions),
            "nearby_entities": len(self.cached_world_state.nearby_entities) if self.cached_world_state else 0
        }

    def to_npcpu_world(self) -> World:
        """Convert Minecraft world data to NPCPU World object"""
        config = WorldConfig(
            name="Minecraft World",
            season_length=1200,  # 1 Minecraft day per season
            day_length=24,
            event_frequency=0.02
        )

        world = World(config)

        # Clear default regions and add Minecraft ones
        world.regions.clear()

        for region_key, mc_region in self.regions.items():
            npcpu_region = Region(
                name=f"{mc_region.biome}_{region_key}",
                type=mc_region.type,
                climate=mc_region.climate,
                danger_level=mc_region.danger_level,
                habitability=1.0 - mc_region.danger_level,
                resources={
                    "wood": mc_region.resource_density * 1000,
                    "stone": mc_region.resource_density * 800,
                    "ore": mc_region.resource_density * 200,
                    "food": mc_region.resource_density * 500
                }
            )
            world.regions[region_key] = npcpu_region

        return world


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Minecraft World Adapter")
    print("=" * 50)
    print("\nThis module maps Minecraft world structure to NPCPU ecosystem.")
    print("\nBiome → Region mappings:")
    for biome, config in list(BIOME_MAPPINGS.items())[:5]:
        print(f"  {biome}: {config['type'].value}, danger={config['danger']}")
    print("\nBlock categories:")
    for block_type, blocks in list(BLOCK_CATEGORIES.items())[:3]:
        print(f"  {block_type.value}: {', '.join(blocks[:3])}...")
