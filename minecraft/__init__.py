"""
NPCPU Minecraft Integration

Enables NPCPU digital organisms to inhabit and interact within Minecraft worlds.
Uses Mineflayer (Node.js) for bot control with Python coordination layer.

Architecture:
- Python: NPCPU organism logic, decision making, learning
- Node.js/Mineflayer: Minecraft world interaction
- Communication: JSON over subprocess/WebSocket

Components:
- bridge: Low-level communication with Mineflayer bot
- organism_adapter: Maps NPCPU systems to Minecraft (sensory, motor, metabolism)
- world_adapter: Maps Minecraft world to NPCPU ecosystem model
- memory: Spatial/episodic/social/procedural memory systems
- behaviors: High-level behavior patterns (shelter, crafting, trading)
- colony: Multi-agent coordination and communication
- ecosystem_runner: Main entry point for simulations
"""

from .bridge import (
    MinecraftBridge,
    MinecraftConfig,
    BotState,
    WorldState,
    Position
)

from .organism_adapter import (
    MinecraftOrganism,
    MinecraftSensory,
    MinecraftMotor,
    MinecraftMetabolism
)

from .world_adapter import (
    MinecraftWorld,
    MinecraftRegion,
    BlockType,
    EntityType
)

from .memory import (
    MemorySystem,
    MemoryType,
    SpatialMemory,
    EpisodicMemory,
    SocialMemory,
    ProceduralMemory,
    EmotionalValence
)

from .behaviors import (
    BehaviorManager,
    ShelterBehavior,
    CraftingBehavior,
    TradingBehavior,
    SurvivalBehavior,
    BehaviorState,
    RECIPES
)

from .colony import (
    Colony,
    ColonyMember,
    ColonyRole,
    ColonyMessage,
    ColonyTask,
    MessageType,
    TaskPriority
)

from .ecosystem_runner import (
    MinecraftEcosystemRunner,
    EcosystemConfig
)

__all__ = [
    # Bridge
    "MinecraftBridge",
    "MinecraftConfig",
    "BotState",
    "WorldState",
    "Position",

    # Organism
    "MinecraftOrganism",
    "MinecraftSensory",
    "MinecraftMotor",
    "MinecraftMetabolism",

    # World
    "MinecraftWorld",
    "MinecraftRegion",
    "BlockType",
    "EntityType",

    # Memory
    "MemorySystem",
    "MemoryType",
    "SpatialMemory",
    "EpisodicMemory",
    "SocialMemory",
    "ProceduralMemory",
    "EmotionalValence",

    # Behaviors
    "BehaviorManager",
    "ShelterBehavior",
    "CraftingBehavior",
    "TradingBehavior",
    "SurvivalBehavior",
    "BehaviorState",
    "RECIPES",

    # Colony
    "Colony",
    "ColonyMember",
    "ColonyRole",
    "ColonyMessage",
    "ColonyTask",
    "MessageType",
    "TaskPriority",

    # Runner
    "MinecraftEcosystemRunner",
    "EcosystemConfig"
]
