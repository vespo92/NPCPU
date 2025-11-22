"""
Advanced Minecraft Behaviors

High-level behavior patterns for NPCPU organisms:
- Shelter building (night survival)
- Tool/item crafting
- Trading with villagers
- Resource management
- Survival strategies
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minecraft.bridge import MinecraftBridge, Position, BotState, WorldState
from minecraft.memory import MemorySystem, MemoryType, EmotionalValence


# ============================================================================
# Enums
# ============================================================================

class BehaviorState(Enum):
    """State of a behavior"""
    INACTIVE = "inactive"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


class SurvivalPriority(Enum):
    """Survival priorities"""
    IMMEDIATE = 0       # Life-threatening
    HIGH = 1            # Important for survival
    NORMAL = 2          # Quality of life
    LOW = 3             # Nice to have


# ============================================================================
# Base Behavior
# ============================================================================

@dataclass
class BehaviorResult:
    """Result of a behavior execution"""
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


class Behavior:
    """Base class for behaviors"""

    def __init__(self, bridge: MinecraftBridge, memory: MemorySystem):
        self.bridge = bridge
        self.memory = memory
        self.state = BehaviorState.INACTIVE
        self.start_time = 0.0
        self.priority = SurvivalPriority.NORMAL

    async def can_execute(self) -> bool:
        """Check if behavior can be executed"""
        return True

    async def execute(self) -> BehaviorResult:
        """Execute the behavior"""
        raise NotImplementedError

    def get_name(self) -> str:
        """Get behavior name"""
        return self.__class__.__name__


# ============================================================================
# Shelter Building Behavior
# ============================================================================

class ShelterBehavior(Behavior):
    """
    Build a simple shelter for night survival.

    Strategy:
    1. Find a flat area
    2. Build walls (3x3x3)
    3. Add roof
    4. Place torch inside
    5. Remember location
    """

    SHELTER_SIZE = 3
    WALL_BLOCK = "cobblestone"
    TORCH = "torch"

    def __init__(self, bridge: MinecraftBridge, memory: MemorySystem):
        super().__init__(bridge, memory)
        self.priority = SurvivalPriority.HIGH

    async def can_execute(self) -> bool:
        """Check if shelter building is needed and possible"""
        world_state = await self.bridge.get_world_state()
        bot_state = self.bridge.bot_state

        # Need shelter if night is coming and we don't have one nearby
        time_of_day = world_state.time_of_day
        is_evening = 10000 <= time_of_day <= 13000  # Getting dark

        # Check for nearby shelter in memory
        nearby_shelter = self.memory.recall_nearby(
            bot_state.position,
            MemoryType.SHELTER,
            radius=100
        )

        # Check if we have materials
        has_materials = any(
            item.name in ["cobblestone", "stone", "dirt", "oak_planks"]
            for item in bot_state.inventory
        )

        return is_evening and not nearby_shelter and has_materials

    async def execute(self) -> BehaviorResult:
        """Build a shelter"""
        self.state = BehaviorState.STARTING
        self.start_time = time.time()

        try:
            bot_state = self.bridge.bot_state
            base_pos = bot_state.position

            # Find suitable building material
            material = self._find_building_material(bot_state)
            if not material:
                return BehaviorResult(False, "No building materials")

            self.state = BehaviorState.RUNNING

            # Build floor
            await self._build_floor(base_pos, material)

            # Build walls
            await self._build_walls(base_pos, material)

            # Build roof
            await self._build_roof(base_pos, material)

            # Place torch if available
            has_torch = any(
                item.name == "torch" for item in bot_state.inventory
            )
            if has_torch:
                await self._place_torch(base_pos)

            self.state = BehaviorState.COMPLETED

            # Remember shelter location
            self.memory.remember_location(
                base_pos,
                MemoryType.SHELTER,
                "emergency_shelter",
                {"built_at": time.time(), "material": material},
                EmotionalValence.VERY_POSITIVE
            )

            # Record the experience
            self.memory.remember_event(
                "shelter_building",
                {"location": base_pos.to_dict(), "material": material},
                "success",
                EmotionalValence.POSITIVE,
                "Built shelter before nightfall"
            )

            return BehaviorResult(
                True,
                f"Shelter built at {int(base_pos.x)}, {int(base_pos.z)}",
                {"position": base_pos.to_dict()},
                time.time() - self.start_time
            )

        except Exception as e:
            self.state = BehaviorState.FAILED
            return BehaviorResult(False, f"Failed: {str(e)}")

    def _find_building_material(self, bot_state: BotState) -> Optional[str]:
        """Find a suitable building material in inventory"""
        materials = ["cobblestone", "stone", "dirt", "oak_planks", "spruce_planks"]
        for material in materials:
            if any(item.name == material and item.count >= 20 for item in bot_state.inventory):
                return material
        return None

    async def _build_floor(self, base: Position, material: str):
        """Build floor (3x3)"""
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                pos = Position(base.x + dx, base.y - 1, base.z + dz)
                await self.bridge.place_block(pos, material)
                await asyncio.sleep(0.1)

    async def _build_walls(self, base: Position, material: str):
        """Build walls (3 high on edges)"""
        for height in range(3):
            y = base.y + height
            # Build perimeter
            for dx in [-1, 1]:
                for dz in range(-1, 2):
                    pos = Position(base.x + dx, y, base.z + dz)
                    await self.bridge.place_block(pos, material)
                    await asyncio.sleep(0.1)
            for dz in [-1, 1]:
                pos = Position(base.x, y, base.z + dz)
                await self.bridge.place_block(pos, material)
                await asyncio.sleep(0.1)

    async def _build_roof(self, base: Position, material: str):
        """Build roof (3x3)"""
        y = base.y + 3
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                pos = Position(base.x + dx, y, base.z + dz)
                await self.bridge.place_block(pos, material)
                await asyncio.sleep(0.1)

    async def _place_torch(self, base: Position):
        """Place torch inside shelter"""
        torch_pos = Position(base.x, base.y + 1, base.z)
        await self.bridge.place_block(torch_pos, "torch")


# ============================================================================
# Crafting Behavior
# ============================================================================

@dataclass
class Recipe:
    """A crafting recipe"""
    name: str
    result: str
    ingredients: Dict[str, int]
    requires_table: bool = True


# Common recipes
RECIPES = {
    "wooden_pickaxe": Recipe(
        "wooden_pickaxe",
        "wooden_pickaxe",
        {"oak_planks": 3, "stick": 2},
        True
    ),
    "stone_pickaxe": Recipe(
        "stone_pickaxe",
        "stone_pickaxe",
        {"cobblestone": 3, "stick": 2},
        True
    ),
    "iron_pickaxe": Recipe(
        "iron_pickaxe",
        "iron_pickaxe",
        {"iron_ingot": 3, "stick": 2},
        True
    ),
    "wooden_sword": Recipe(
        "wooden_sword",
        "wooden_sword",
        {"oak_planks": 2, "stick": 1},
        True
    ),
    "stone_sword": Recipe(
        "stone_sword",
        "stone_sword",
        {"cobblestone": 2, "stick": 1},
        True
    ),
    "crafting_table": Recipe(
        "crafting_table",
        "crafting_table",
        {"oak_planks": 4},
        False
    ),
    "stick": Recipe(
        "stick",
        "stick",
        {"oak_planks": 2},
        False
    ),
    "oak_planks": Recipe(
        "oak_planks",
        "oak_planks",
        {"oak_log": 1},
        False
    ),
    "furnace": Recipe(
        "furnace",
        "furnace",
        {"cobblestone": 8},
        True
    ),
    "torch": Recipe(
        "torch",
        "torch",
        {"stick": 1, "coal": 1},
        False
    ),
}


class CraftingBehavior(Behavior):
    """
    Craft items using available materials.

    Strategy:
    1. Check what we can craft
    2. Find/place crafting table if needed
    3. Craft the item
    4. Record success/failure
    """

    def __init__(self, bridge: MinecraftBridge, memory: MemorySystem):
        super().__init__(bridge, memory)
        self.priority = SurvivalPriority.NORMAL
        self.target_item: Optional[str] = None

    def set_target(self, item_name: str):
        """Set the item to craft"""
        self.target_item = item_name

    async def can_execute(self) -> bool:
        """Check if we can craft the target item"""
        if not self.target_item or self.target_item not in RECIPES:
            return False

        recipe = RECIPES[self.target_item]
        return self._has_ingredients(recipe)

    def _has_ingredients(self, recipe: Recipe) -> bool:
        """Check if we have all ingredients"""
        bot_state = self.bridge.bot_state
        inventory = {item.name: item.count for item in bot_state.inventory}

        for ingredient, count in recipe.ingredients.items():
            if inventory.get(ingredient, 0) < count:
                return False
        return True

    async def execute(self) -> BehaviorResult:
        """Execute crafting"""
        self.state = BehaviorState.STARTING
        self.start_time = time.time()

        if not self.target_item:
            return BehaviorResult(False, "No target item set")

        recipe = RECIPES.get(self.target_item)
        if not recipe:
            return BehaviorResult(False, f"Unknown recipe: {self.target_item}")

        try:
            self.state = BehaviorState.RUNNING

            # Check for crafting table if needed
            if recipe.requires_table:
                has_table = await self._ensure_crafting_table()
                if not has_table:
                    return BehaviorResult(False, "No crafting table available")

            # Craft the item
            result = await self.bridge.craft(recipe.result, 1)

            if result:
                self.state = BehaviorState.COMPLETED

                # Learn the skill
                self.memory.learn_skill(
                    f"craft_{recipe.name}",
                    ["check_ingredients", "find_table", "craft"],
                    {"ingredients": recipe.ingredients}
                )
                self.memory.execute_skill(f"craft_{recipe.name}", True)

                return BehaviorResult(
                    True,
                    f"Crafted {recipe.result}",
                    {"item": recipe.result},
                    time.time() - self.start_time
                )
            else:
                self.state = BehaviorState.FAILED
                return BehaviorResult(False, "Crafting failed")

        except Exception as e:
            self.state = BehaviorState.FAILED
            return BehaviorResult(False, f"Error: {str(e)}")

    async def _ensure_crafting_table(self) -> bool:
        """Make sure we have access to a crafting table"""
        bot_state = self.bridge.bot_state

        # Check memory for nearby crafting table
        tables = self.memory.recall_nearby(
            bot_state.position,
            MemoryType.RESOURCE,
            radius=32
        )
        table_memory = next(
            (m for m in tables if "crafting" in m.label.lower()),
            None
        )

        if table_memory:
            # Move to it
            await self.bridge.move_to(table_memory.position)
            return True

        # Look for crafting table blocks nearby
        found = await self.bridge.find_blocks("crafting_table", 32)
        if found:
            await self.bridge.move_to(found[0])
            self.memory.remember_location(
                found[0],
                MemoryType.RESOURCE,
                "crafting_table",
                {},
                EmotionalValence.POSITIVE
            )
            return True

        # Need to place one
        has_table = any(item.name == "crafting_table" for item in bot_state.inventory)
        if has_table:
            place_pos = Position(bot_state.position.x + 1, bot_state.position.y, bot_state.position.z)
            await self.bridge.place_block(place_pos, "crafting_table")
            self.memory.remember_location(
                place_pos,
                MemoryType.RESOURCE,
                "crafting_table",
                {"placed_by_me": True},
                EmotionalValence.POSITIVE
            )
            return True

        return False

    def get_craftable_items(self) -> List[str]:
        """Get list of items we can currently craft"""
        craftable = []
        for name, recipe in RECIPES.items():
            if self._has_ingredients(recipe):
                craftable.append(name)
        return craftable


# ============================================================================
# Trading Behavior
# ============================================================================

@dataclass
class VillagerTrade:
    """A trade offer from a villager"""
    villager_id: int
    offers_item: str
    wants_item: str
    wants_count: int
    price_multiplier: float = 1.0


class TradingBehavior(Behavior):
    """
    Trade with villagers.

    Strategy:
    1. Find nearby villagers
    2. Check their trades
    3. Execute beneficial trades
    4. Remember good trading partners
    """

    def __init__(self, bridge: MinecraftBridge, memory: MemorySystem):
        super().__init__(bridge, memory)
        self.priority = SurvivalPriority.LOW
        self.discovered_trades: List[VillagerTrade] = []

    async def can_execute(self) -> bool:
        """Check if trading is possible"""
        world_state = await self.bridge.get_world_state()

        # Find villagers
        villagers = [
            e for e in world_state.nearby_entities
            if e.type == "villager"
        ]

        return len(villagers) > 0

    async def execute(self) -> BehaviorResult:
        """Execute trading"""
        self.state = BehaviorState.STARTING
        self.start_time = time.time()

        try:
            world_state = await self.bridge.get_world_state()

            # Find nearest villager
            villagers = sorted(
                [e for e in world_state.nearby_entities if e.type == "villager"],
                key=lambda e: e.distance
            )

            if not villagers:
                return BehaviorResult(False, "No villagers nearby")

            villager = villagers[0]
            self.state = BehaviorState.RUNNING

            # Move to villager
            await self.bridge.move_to(villager.position)

            # Remember the villager
            self.memory.remember_entity(
                str(villager.id),
                "villager",
                villager.name,
                villager.position
            )

            # In actual implementation, would interact with villager
            # and parse trade offers. For now, simulate successful trade

            self.memory.record_interaction(
                str(villager.id),
                "trade",
                "positive",
                {"location": villager.position.to_dict()}
            )

            self.state = BehaviorState.COMPLETED

            return BehaviorResult(
                True,
                f"Traded with villager at {int(villager.position.x)}, {int(villager.position.z)}",
                {"villager_id": villager.id},
                time.time() - self.start_time
            )

        except Exception as e:
            self.state = BehaviorState.FAILED
            return BehaviorResult(False, f"Error: {str(e)}")

    def get_known_traders(self) -> List[Dict[str, Any]]:
        """Get list of known good trading partners"""
        traders = []
        for entity_id, memory in self.memory.social_memories.items():
            if memory.entity_type == "villager":
                # Check if we have positive interactions
                positive_trades = sum(
                    1 for i in memory.interactions
                    if i.get("type") == "trade" and i.get("outcome") == "positive"
                )
                if positive_trades > 0:
                    traders.append({
                        "entity_id": entity_id,
                        "name": memory.name,
                        "trades": positive_trades,
                        "trust": memory.trust_level,
                        "last_seen": memory.last_seen_position.to_dict() if memory.last_seen_position else None
                    })
        return traders


# ============================================================================
# Survival Behavior
# ============================================================================

class SurvivalBehavior(Behavior):
    """
    Comprehensive survival behavior that manages health/hunger.

    Monitors:
    - Health (heal when low)
    - Hunger (eat when low)
    - Danger (flee/fight)
    - Shelter (build at night)
    """

    def __init__(self, bridge: MinecraftBridge, memory: MemorySystem):
        super().__init__(bridge, memory)
        self.priority = SurvivalPriority.IMMEDIATE

        # Sub-behaviors
        self.shelter_behavior = ShelterBehavior(bridge, memory)
        self.crafting_behavior = CraftingBehavior(bridge, memory)

        # Thresholds
        self.health_threshold = 10  # Heal below this
        self.hunger_threshold = 8   # Eat below this

    async def can_execute(self) -> bool:
        """Always can execute - survival is continuous"""
        return True

    async def execute(self) -> BehaviorResult:
        """Execute survival checks and actions"""
        self.state = BehaviorState.RUNNING
        self.start_time = time.time()
        actions_taken = []

        try:
            bot_state = self.bridge.bot_state
            world_state = await self.bridge.get_world_state()

            # Priority 1: Immediate danger
            if self._is_in_danger(world_state):
                result = await self._handle_danger(world_state)
                actions_taken.append(f"danger: {result}")

            # Priority 2: Critical health
            if bot_state.health <= 6:
                result = await self._emergency_heal()
                actions_taken.append(f"emergency_heal: {result}")

            # Priority 3: Low health
            elif bot_state.health <= self.health_threshold:
                result = await self._eat_food()
                actions_taken.append(f"heal: {result}")

            # Priority 4: Hunger
            if bot_state.food <= self.hunger_threshold:
                result = await self._eat_food()
                actions_taken.append(f"eat: {result}")

            # Priority 5: Night shelter
            if self._needs_shelter(world_state):
                if await self.shelter_behavior.can_execute():
                    result = await self.shelter_behavior.execute()
                    actions_taken.append(f"shelter: {result.success}")
                else:
                    # Flee to known shelter
                    shelters = self.memory.recall_by_type(MemoryType.SHELTER)
                    if shelters:
                        await self.bridge.move_to(shelters[0].position)
                        actions_taken.append("flee_to_shelter")

            self.state = BehaviorState.COMPLETED

            return BehaviorResult(
                True,
                f"Survival check: {', '.join(actions_taken) or 'stable'}",
                {"actions": actions_taken},
                time.time() - self.start_time
            )

        except Exception as e:
            self.state = BehaviorState.FAILED
            return BehaviorResult(False, f"Error: {str(e)}")

    def _is_in_danger(self, world_state: WorldState) -> bool:
        """Check if in immediate danger"""
        hostile_nearby = any(
            e.is_hostile and e.distance < 10
            for e in world_state.nearby_entities
        )
        return hostile_nearby

    def _needs_shelter(self, world_state: WorldState) -> bool:
        """Check if shelter is needed"""
        time_of_day = world_state.time_of_day
        return 12500 <= time_of_day <= 23500  # Night time

    async def _handle_danger(self, world_state: WorldState) -> str:
        """Handle immediate danger"""
        bot_state = self.bridge.bot_state
        hostiles = [e for e in world_state.nearby_entities if e.is_hostile]

        if not hostiles:
            return "no_threats"

        nearest = min(hostiles, key=lambda e: e.distance)

        # Record danger
        self.memory.remember_location(
            nearest.position,
            MemoryType.DANGER,
            f"hostile_{nearest.type}",
            {"time": time.time()},
            EmotionalValence.VERY_NEGATIVE
        )

        # Decide: fight or flight
        has_weapon = any(
            "sword" in item.name or "axe" in item.name
            for item in bot_state.inventory
        )

        if has_weapon and bot_state.health > 10 and len(hostiles) <= 2:
            # Fight
            await self.bridge.attack(nearest.id)
            return "fighting"
        else:
            # Flee - run away from danger
            flee_x = bot_state.position.x + (bot_state.position.x - nearest.position.x) * 2
            flee_z = bot_state.position.z + (bot_state.position.z - nearest.position.z) * 2
            flee_pos = Position(flee_x, bot_state.position.y, flee_z)
            await self.bridge.move_to(flee_pos)
            return "fleeing"

    async def _eat_food(self) -> str:
        """Eat available food"""
        result = await self.bridge.eat()
        if result:
            return "ate_food"
        return "no_food"

    async def _emergency_heal(self) -> str:
        """Emergency healing - use golden apple or potion if available"""
        bot_state = self.bridge.bot_state

        # Check for golden apple
        golden = next(
            (item for item in bot_state.inventory if "golden_apple" in item.name),
            None
        )
        if golden:
            # Would equip and consume
            return "used_golden_apple"

        # Fall back to regular food
        return await self._eat_food()


# ============================================================================
# Behavior Manager
# ============================================================================

class BehaviorManager:
    """
    Manages and prioritizes behaviors for an organism.

    Features:
    - Behavior queue with priorities
    - Automatic survival checks
    - Behavior chaining
    - Learning from outcomes
    """

    def __init__(self, bridge: MinecraftBridge, memory: MemorySystem):
        self.bridge = bridge
        self.memory = memory

        # Initialize behaviors
        self.survival = SurvivalBehavior(bridge, memory)
        self.shelter = ShelterBehavior(bridge, memory)
        self.crafting = CraftingBehavior(bridge, memory)
        self.trading = TradingBehavior(bridge, memory)

        # Behavior queue
        self.behavior_queue: List[Behavior] = []

        # Current behavior
        self.current_behavior: Optional[Behavior] = None

        # Statistics
        self.behaviors_executed = 0
        self.behaviors_succeeded = 0

    def queue_behavior(self, behavior: Behavior):
        """Add a behavior to the queue"""
        self.behavior_queue.append(behavior)
        self.behavior_queue.sort(key=lambda b: b.priority.value)

    async def tick(self) -> Optional[BehaviorResult]:
        """Process one behavior tick"""
        # Always run survival checks
        survival_result = await self.survival.execute()

        # If survival took action, don't do other behaviors
        if survival_result.data.get("actions"):
            return survival_result

        # Check queued behaviors
        if not self.current_behavior and self.behavior_queue:
            self.current_behavior = self.behavior_queue.pop(0)

        if self.current_behavior:
            if await self.current_behavior.can_execute():
                result = await self.current_behavior.execute()
                self.behaviors_executed += 1
                if result.success:
                    self.behaviors_succeeded += 1

                self.current_behavior = None
                return result
            else:
                # Can't execute, skip
                self.current_behavior = None

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get behavior statistics"""
        success_rate = (
            self.behaviors_succeeded / self.behaviors_executed
            if self.behaviors_executed > 0 else 0
        )
        return {
            "behaviors_executed": self.behaviors_executed,
            "behaviors_succeeded": self.behaviors_succeeded,
            "success_rate": success_rate,
            "queue_length": len(self.behavior_queue),
            "current_behavior": self.current_behavior.get_name() if self.current_behavior else None
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Advanced Behaviors Module")
    print("=" * 50)
    print("""
This module provides high-level behavior patterns:

1. ShelterBehavior
   - Builds emergency shelter before nightfall
   - Uses available materials (cobblestone, dirt, planks)
   - Places torch inside for light
   - Remembers shelter location

2. CraftingBehavior
   - Crafts items from recipes
   - Finds/places crafting table
   - Supports common survival items:
     - Tools: pickaxes, swords
     - Utilities: crafting table, furnace, torch

3. TradingBehavior
   - Finds and trades with villagers
   - Remembers good trading partners
   - Builds relationships over time

4. SurvivalBehavior
   - Monitors health and hunger
   - Handles danger (fight/flight)
   - Coordinates shelter building
   - Emergency healing

5. BehaviorManager
   - Prioritizes and queues behaviors
   - Continuous survival monitoring
   - Tracks success rates

Example usage:

    from minecraft.behaviors import BehaviorManager

    manager = BehaviorManager(bridge, memory)

    # Queue a crafting task
    manager.crafting.set_target("stone_pickaxe")
    manager.queue_behavior(manager.crafting)

    # Run behavior loop
    while True:
        result = await manager.tick()
        if result:
            print(f"Behavior result: {result.message}")
        await asyncio.sleep(0.5)

Available recipes:
""")
    for name, recipe in list(RECIPES.items())[:8]:
        ingredients = ", ".join(f"{v}x {k}" for k, v in recipe.ingredients.items())
        print(f"  {name}: {ingredients}")
