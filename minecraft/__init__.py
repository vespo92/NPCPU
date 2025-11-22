"""
NPCPU Minecraft Integration

Enables NPCPU digital organisms to inhabit and interact within Minecraft worlds.
Uses Mineflayer (Node.js) for bot control with Python coordination layer.

Architecture:
- Python: NPCPU organism logic, decision making, learning
- Node.js/Mineflayer: Minecraft world interaction
- Communication: JSON over subprocess/WebSocket
"""

from .bridge import (
    MinecraftBridge,
    MinecraftConfig,
    BotState,
    WorldState
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

__all__ = [
    "MinecraftBridge",
    "MinecraftConfig",
    "BotState",
    "WorldState",
    "MinecraftOrganism",
    "MinecraftSensory",
    "MinecraftMotor",
    "MinecraftMetabolism",
    "MinecraftWorld",
    "MinecraftRegion",
    "BlockType",
    "EntityType"
]
