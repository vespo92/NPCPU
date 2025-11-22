"""
Minecraft Organism Adapter

Maps NPCPU organism systems to Minecraft interactions.
Each biological system gets a Minecraft-specific implementation.

Mappings:
- Sensory → World perception (entities, blocks, sounds)
- Motor → Movement, mining, building, combat
- Metabolism → Health, hunger, inventory management
- Endocrine → Mood based on game state (danger, resources, social)
- Nervous → Reflex responses to game events
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

from minecraft.bridge import (
    MinecraftBridge, MinecraftConfig, BotState, WorldState,
    Position, Entity, Block, BotActivity
)
from organism.digital_body import DigitalBody, OrganismState
from sensory.perception import SensorySystem, SensorType, SensoryInput, SensoryModality
from motor.action import MotorSystem, ActionType, Action, ActionResult
from endocrine.hormones import EndocrineSystem, HormoneType


# ============================================================================
# Minecraft Sensory Adapter
# ============================================================================

class MinecraftSensory:
    """
    Adapts Minecraft world perception to NPCPU sensory system.

    Mappings:
    - Nearby entities → Threat/social detection
    - Block scanning → Resource detection
    - Health/hunger → Interoception
    - Time of day → Temporal sensing
    - Player proximity → Social sensing
    """

    def __init__(self, bridge: MinecraftBridge, sensory_system: SensorySystem):
        self.bridge = bridge
        self.sensory = sensory_system

        # Perception settings
        self.scan_radius = 32
        self.entity_awareness_radius = 16
        self.threat_radius = 8

        # Hostile mob types
        self.hostile_mobs = {
            "zombie", "skeleton", "creeper", "spider", "enderman",
            "witch", "slime", "phantom", "drowned", "husk", "stray"
        }

        # Resource blocks
        self.resource_blocks = {
            "coal_ore", "iron_ore", "gold_ore", "diamond_ore",
            "oak_log", "birch_log", "spruce_log",
            "wheat", "carrots", "potatoes"
        }

    async def perceive(self) -> List[SensoryInput]:
        """Perceive the Minecraft world and generate sensory inputs"""
        inputs = []

        # Get current states
        bot_state = await self.bridge.get_state()
        world_state = await self.bridge.get_world_state()

        # Interoception - internal state
        inputs.extend(self._sense_internal(bot_state))

        # Exteroception - entities
        inputs.extend(self._sense_entities(world_state))

        # Exteroception - resources/blocks
        inputs.extend(await self._sense_resources())

        # Temporal - time of day
        inputs.extend(self._sense_time(world_state))

        # Nociception - threats
        inputs.extend(self._sense_threats(world_state, bot_state))

        return inputs

    def _sense_internal(self, bot_state: BotState) -> List[SensoryInput]:
        """Sense internal state (health, hunger)"""
        inputs = []

        # Health sensing
        health_ratio = bot_state.health / 20.0
        inputs.append(self.sensory.sense(
            SensorType.SELF_MONITOR,
            raw_data={"type": "health", "value": bot_state.health, "ratio": health_ratio},
            source="internal_health"
        ))

        # Hunger sensing
        food_ratio = bot_state.food / 20.0
        inputs.append(self.sensory.sense(
            SensorType.SELF_MONITOR,
            raw_data={"type": "hunger", "value": bot_state.food, "ratio": food_ratio},
            source="internal_hunger"
        ))

        # Filter None values
        return [i for i in inputs if i is not None]

    def _sense_entities(self, world_state: WorldState) -> List[SensoryInput]:
        """Sense nearby entities"""
        inputs = []

        for entity in world_state.nearby_entities:
            if entity.distance > self.entity_awareness_radius:
                continue

            # Determine sensor type based on entity
            if entity.is_hostile:
                sensor_type = SensorType.THREAT_DETECTOR
            elif entity.is_player:
                sensor_type = SensorType.SOCIAL_RECEPTOR
            else:
                sensor_type = SensorType.DATA_STREAM

            intensity = 1.0 - (entity.distance / self.entity_awareness_radius)

            input = self.sensory.sense(
                sensor_type,
                raw_data={
                    "type": "entity",
                    "entity_type": entity.type,
                    "name": entity.name,
                    "position": entity.position.to_dict(),
                    "distance": entity.distance,
                    "is_hostile": entity.is_hostile,
                    "is_player": entity.is_player
                },
                source=f"entity_{entity.id}"
            )
            if input:
                input.intensity = intensity
                inputs.append(input)

        return inputs

    async def _sense_resources(self) -> List[SensoryInput]:
        """Sense nearby resources"""
        inputs = []

        for block_type in list(self.resource_blocks)[:3]:  # Limit scans
            positions = await self.bridge.find_blocks(block_type, self.scan_radius)

            if positions:
                closest = min(positions, key=lambda p: p.distance_to(self.bridge.bot_state.position))
                distance = closest.distance_to(self.bridge.bot_state.position)
                intensity = max(0.1, 1.0 - (distance / self.scan_radius))

                input = self.sensory.sense(
                    SensorType.RESOURCE_MONITOR,
                    raw_data={
                        "type": "resource",
                        "block_type": block_type,
                        "count": len(positions),
                        "closest_position": closest.to_dict(),
                        "distance": distance
                    },
                    source=f"resource_{block_type}"
                )
                if input:
                    input.intensity = intensity
                    inputs.append(input)

        return inputs

    def _sense_time(self, world_state: WorldState) -> List[SensoryInput]:
        """Sense time of day"""
        # Minecraft day: 0-12000 day, 12000-24000 night
        time_ratio = world_state.time_of_day / 24000.0
        is_night = 12000 <= world_state.time_of_day <= 24000

        input = self.sensory.sense(
            SensorType.TIME_SENSE,
            raw_data={
                "type": "time",
                "ticks": world_state.time_of_day,
                "ratio": time_ratio,
                "is_night": is_night,
                "weather": world_state.weather
            },
            source="time_sense"
        )

        return [input] if input else []

    def _sense_threats(self, world_state: WorldState, bot_state: BotState) -> List[SensoryInput]:
        """Sense threats (hostile mobs, low health, night)"""
        inputs = []

        # Nearby hostiles
        hostile_count = sum(
            1 for e in world_state.nearby_entities
            if e.is_hostile and e.distance < self.threat_radius
        )

        if hostile_count > 0:
            input = self.sensory.sense(
                SensorType.THREAT_DETECTOR,
                raw_data={
                    "type": "hostile_nearby",
                    "count": hostile_count,
                    "threat_level": min(1.0, hostile_count * 0.3)
                },
                source="threat_detection"
            )
            if input:
                input.intensity = min(1.0, hostile_count * 0.3)
                inputs.append(input)

        # Low health threat
        if bot_state.health < 8:
            input = self.sensory.sense(
                SensorType.THREAT_DETECTOR,
                raw_data={
                    "type": "low_health",
                    "health": bot_state.health,
                    "threat_level": 1.0 - (bot_state.health / 8.0)
                },
                source="health_threat"
            )
            if input:
                inputs.append(input)

        return inputs


# ============================================================================
# Minecraft Motor Adapter
# ============================================================================

class MinecraftMotor:
    """
    Adapts NPCPU motor system to Minecraft actions.

    Mappings:
    - MOVE → pathfinder navigation
    - ACQUIRE → mining, pickup
    - CONSUME → eating
    - DEFEND → combat
    - COMMUNICATE → chat
    - EXPLORE → random movement
    - BUILD → block placement
    """

    def __init__(self, bridge: MinecraftBridge, motor_system: MotorSystem):
        self.bridge = bridge
        self.motor = motor_system

        # Register action handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup action handlers"""
        self.motor.register_handler(ActionType.MOVE, self._handle_move)
        self.motor.register_handler(ActionType.ACQUIRE, self._handle_acquire)
        self.motor.register_handler(ActionType.CONSUME, self._handle_consume)
        self.motor.register_handler(ActionType.DEFEND, self._handle_defend)
        self.motor.register_handler(ActionType.COMMUNICATE, self._handle_communicate)
        self.motor.register_handler(ActionType.EXPLORE, self._handle_explore)
        self.motor.register_handler(ActionType.REST, self._handle_rest)

    async def _handle_move(self, action: Action) -> Any:
        """Handle movement action"""
        target = action.parameters.get("target")
        if target:
            if isinstance(target, dict):
                pos = Position.from_dict(target)
            elif isinstance(target, Position):
                pos = target
            else:
                return None

            success = await self.bridge.move_to(pos)
            return {"moved": success, "destination": pos.to_dict()}
        return None

    async def _handle_acquire(self, action: Action) -> Any:
        """Handle resource acquisition (mining)"""
        target = action.parameters.get("target")
        block_type = action.parameters.get("block_type")

        if target:
            pos = Position.from_dict(target) if isinstance(target, dict) else target
            success = await self.bridge.dig(pos)
            return {"mined": success, "position": pos.to_dict()}

        elif block_type:
            # Find and mine nearest of type
            blocks = await self.bridge.find_blocks(block_type, 16)
            if blocks:
                nearest = blocks[0]
                await self.bridge.move_to(nearest)
                success = await self.bridge.dig(nearest)
                return {"mined": success, "block": block_type}

        return None

    async def _handle_consume(self, action: Action) -> Any:
        """Handle consumption (eating)"""
        success = await self.bridge.eat()
        return {"ate": success}

    async def _handle_defend(self, action: Action) -> Any:
        """Handle defense/combat"""
        target_id = action.parameters.get("entity_id")

        if target_id:
            success = await self.bridge.attack(target_id)
            return {"attacked": success, "target": target_id}
        else:
            # Attack nearest hostile
            success = await self.bridge.attack_nearest_hostile()
            return {"attacked_nearest": success}

    async def _handle_communicate(self, action: Action) -> Any:
        """Handle communication (chat)"""
        message = action.parameters.get("message", "Hello!")
        success = await self.bridge.chat(message)
        return {"messaged": success, "content": message}

    async def _handle_explore(self, action: Action) -> Any:
        """Handle exploration"""
        # Move to a random nearby location
        current = self.bridge.bot_state.position
        offset_x = np.random.uniform(-20, 20)
        offset_z = np.random.uniform(-20, 20)
        target = Position(current.x + offset_x, current.y, current.z + offset_z)

        success = await self.bridge.move_to(target)
        return {"explored": success, "destination": target.to_dict()}

    async def _handle_rest(self, action: Action) -> Any:
        """Handle resting (stay still, maybe find shelter)"""
        # Just wait and recover
        await asyncio.sleep(1.0)
        return {"rested": True}

    async def execute_queued_actions(self) -> List[ActionResult]:
        """Execute queued actions asynchronously"""
        results = []

        while self.motor.action_queue:
            action = self.motor.action_queue.pop(0)
            action.state = "executing"
            action.started_at = time.time()

            try:
                handler = {
                    ActionType.MOVE: self._handle_move,
                    ActionType.ACQUIRE: self._handle_acquire,
                    ActionType.CONSUME: self._handle_consume,
                    ActionType.DEFEND: self._handle_defend,
                    ActionType.COMMUNICATE: self._handle_communicate,
                    ActionType.EXPLORE: self._handle_explore,
                    ActionType.REST: self._handle_rest
                }.get(action.type)

                if handler:
                    output = await handler(action)
                    success = output is not None
                else:
                    output = None
                    success = False

                result = ActionResult(
                    action_id=action.id,
                    success=success,
                    output=output,
                    energy_consumed=action.energy_cost,
                    duration=time.time() - action.started_at
                )
                results.append(result)

            except Exception as e:
                results.append(ActionResult(
                    action_id=action.id,
                    success=False,
                    error=str(e)
                ))

        return results


# ============================================================================
# Minecraft Metabolism Adapter
# ============================================================================

class MinecraftMetabolism:
    """
    Adapts NPCPU metabolism to Minecraft health/hunger system.

    Mappings:
    - Energy → Food/saturation
    - Health → Health points
    - Resources → Inventory items
    """

    def __init__(self, bridge: MinecraftBridge):
        self.bridge = bridge

        # Resource mappings
        self.food_items = {
            "cooked_beef": 8,
            "cooked_porkchop": 8,
            "golden_apple": 4,
            "bread": 5,
            "cooked_chicken": 6,
            "apple": 4
        }

    def get_energy(self) -> float:
        """Get current energy (from food)"""
        return self.bridge.bot_state.food

    def get_health(self) -> float:
        """Get current health"""
        return self.bridge.bot_state.health

    def needs_food(self) -> bool:
        """Check if bot needs to eat"""
        return self.bridge.bot_state.food < 18

    def is_starving(self) -> bool:
        """Check if bot is starving"""
        return self.bridge.bot_state.food < 6

    def get_inventory_summary(self) -> Dict[str, int]:
        """Get summary of inventory"""
        summary = {}
        for item in self.bridge.bot_state.inventory:
            name = item.get("name", "unknown")
            count = item.get("count", 0)
            summary[name] = summary.get(name, 0) + count
        return summary

    def has_food(self) -> bool:
        """Check if bot has food in inventory"""
        inventory = self.get_inventory_summary()
        return any(food in inventory for food in self.food_items)


# ============================================================================
# Minecraft Organism (Complete Integration)
# ============================================================================

class MinecraftOrganism:
    """
    Complete NPCPU organism living in Minecraft.

    Integrates:
    - DigitalBody (core organism)
    - MinecraftBridge (Minecraft connection)
    - Adapted sensory, motor, metabolism

    Example:
        config = MinecraftConfig(host="localhost", port=25565)
        organism = MinecraftOrganism(name="Steve_AI", config=config)

        await organism.connect()

        while organism.is_alive:
            await organism.tick()
            await asyncio.sleep(0.5)
    """

    def __init__(
        self,
        name: str = "NPCPU_Bot",
        config: Optional[MinecraftConfig] = None
    ):
        # Minecraft connection
        self.config = config or MinecraftConfig(username=name)
        self.bridge = MinecraftBridge(self.config)

        # Core organism
        self.body = DigitalBody(name=name)

        # Minecraft adapters
        self.mc_sensory = MinecraftSensory(self.bridge, self.body.sensory)
        self.mc_motor = MinecraftMotor(self.bridge, self.body.motor)
        self.mc_metabolism = MinecraftMetabolism(self.bridge)

        # State
        self.connected = False
        self.tick_count = 0

    @property
    def is_alive(self) -> bool:
        return self.body.is_alive and self.bridge.bot_state.is_alive

    async def connect(self) -> bool:
        """Connect to Minecraft server"""
        success = await self.bridge.connect()
        self.connected = success
        return success

    async def disconnect(self):
        """Disconnect from Minecraft"""
        await self.bridge.disconnect()
        self.connected = False

    async def tick(self):
        """Process one organism cycle in Minecraft"""
        if not self.connected:
            return

        self.tick_count += 1

        # 1. Perceive Minecraft world
        sensory_inputs = await self.mc_sensory.perceive()

        # 2. Update body based on Minecraft state
        self._sync_body_state()

        # 3. Let body process (mood, hormones, etc.)
        self.body.tick()

        # 4. Make decisions based on state
        self._decide_actions()

        # 5. Execute actions in Minecraft
        await self.mc_motor.execute_queued_actions()

    def _sync_body_state(self):
        """Sync organism state with Minecraft state"""
        mc_health = self.bridge.bot_state.health / 20.0
        mc_food = self.bridge.bot_state.food / 20.0

        # Sync metabolism energy
        self.body.metabolism.energy = mc_food * self.body.metabolism.max_energy

        # Sync stress based on danger
        hostile_nearby = any(
            e.is_hostile and e.distance < 10
            for e in self.bridge.world_state.nearby_entities
        )
        if hostile_nearby:
            self.body.endocrine.trigger_stress_response(0.3)

        # Sync mood based on food/health
        if mc_food > 0.8 and mc_health > 0.8:
            self.body.endocrine.trigger_reward(0.2)
        elif mc_food < 0.3:
            self.body.endocrine.trigger_stress_response(0.2)

    def _decide_actions(self):
        """Decide what actions to take based on organism state"""
        # Priority: Survival > Resources > Exploration

        # 1. Low health - flee or heal
        if self.bridge.bot_state.health < 8:
            self.body.motor.queue_action(
                ActionType.REST,
                priority=self.body.motor.ActionPriority.URGENT
            )
            return

        # 2. Hunger - find and eat food
        if self.mc_metabolism.needs_food():
            if self.mc_metabolism.has_food():
                self.body.motor.queue_action(
                    ActionType.CONSUME,
                    priority=self.body.motor.ActionPriority.HIGH
                )
            else:
                # Look for food
                self.body.motor.queue_action(
                    ActionType.EXPLORE,
                    priority=self.body.motor.ActionPriority.HIGH
                )
            return

        # 3. Threats nearby - defend or flee
        hostiles = [
            e for e in self.bridge.world_state.nearby_entities
            if e.is_hostile and e.distance < 10
        ]
        if hostiles and self.body.identity.traits.get("aggression", 0.3) > 0.4:
            self.body.motor.queue_action(
                ActionType.DEFEND,
                priority=self.body.motor.ActionPriority.URGENT
            )
            return

        # 4. Default - explore or gather resources
        if np.random.random() < self.body.identity.traits.get("curiosity", 0.5):
            self.body.motor.queue_action(ActionType.EXPLORE)
        else:
            self.body.motor.queue_action(
                ActionType.ACQUIRE,
                parameters={"block_type": "oak_log"}
            )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Minecraft Organism Adapter")
    print("=" * 50)
    print("\nThis module adapts NPCPU organisms to live in Minecraft.")
    print("\nUsage:")
    print("""
    import asyncio
    from minecraft.organism_adapter import MinecraftOrganism, MinecraftConfig

    async def main():
        config = MinecraftConfig(
            host="localhost",
            port=25565,
            username="NPCPU_Bot"
        )

        organism = MinecraftOrganism("TestBot", config)
        await organism.connect()

        while organism.is_alive:
            await organism.tick()
            await asyncio.sleep(0.5)

    asyncio.run(main())
    """)
