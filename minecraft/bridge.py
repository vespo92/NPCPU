"""
Minecraft Bridge

Handles communication between Python NPCPU logic and
Mineflayer bots running in Node.js.

Communication methods:
1. Subprocess with JSON stdio
2. WebSocket for real-time
3. HTTP API for simple commands
"""

import json
import time
import uuid
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import queue


# ============================================================================
# Enums
# ============================================================================

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class BotActivity(Enum):
    """Current bot activity"""
    IDLE = "idle"
    MOVING = "moving"
    MINING = "mining"
    BUILDING = "building"
    FIGHTING = "fighting"
    EATING = "eating"
    CRAFTING = "crafting"
    TRADING = "trading"
    EXPLORING = "exploring"
    FLEEING = "fleeing"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MinecraftConfig:
    """Configuration for Minecraft connection"""
    host: str = "localhost"
    port: int = 25565
    username: str = "NPCPU_Bot"
    version: str = "1.20.1"
    auth: str = "offline"            # offline, microsoft
    view_distance: int = 4
    chat_delay: float = 1.0
    node_path: str = "node"
    bot_script_path: str = "./minecraft/bot.js"


@dataclass
class Position:
    """3D position in Minecraft"""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Position') -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'Position':
        return Position(d["x"], d["y"], d["z"])


@dataclass
class BotState:
    """Current state of the Minecraft bot"""
    position: Position = field(default_factory=lambda: Position(0, 64, 0))
    health: float = 20.0
    food: float = 20.0
    saturation: float = 5.0
    experience: int = 0
    gamemode: str = "survival"
    is_alive: bool = True
    activity: BotActivity = BotActivity.IDLE
    inventory: List[Dict[str, Any]] = field(default_factory=list)
    held_item: Optional[Dict[str, Any]] = None
    looking_at: Optional[Position] = None
    on_ground: bool = True
    in_water: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class Entity:
    """An entity in the Minecraft world"""
    id: int
    type: str
    name: Optional[str]
    position: Position
    velocity: Optional[Position] = None
    health: Optional[float] = None
    is_hostile: bool = False
    is_player: bool = False
    distance: float = 0.0


@dataclass
class Block:
    """A block in the Minecraft world"""
    position: Position
    type: str
    biome: str = "plains"
    light_level: int = 15
    is_solid: bool = True


@dataclass
class WorldState:
    """State of the Minecraft world around the bot"""
    nearby_entities: List[Entity] = field(default_factory=list)
    nearby_blocks: List[Block] = field(default_factory=list)
    time_of_day: int = 0             # 0-24000 ticks
    weather: str = "clear"           # clear, rain, thunder
    biome: str = "plains"
    light_level: int = 15
    dimension: str = "overworld"     # overworld, nether, end


# ============================================================================
# Command Protocol
# ============================================================================

@dataclass
class BotCommand:
    """A command to send to the bot"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "action": self.action,
            "params": self.params,
            "timestamp": self.timestamp
        })


@dataclass
class BotResponse:
    """Response from the bot"""
    command_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def from_json(json_str: str) -> 'BotResponse':
        d = json.loads(json_str)
        return BotResponse(
            command_id=d.get("command_id", ""),
            success=d.get("success", False),
            data=d.get("data"),
            error=d.get("error"),
            timestamp=d.get("timestamp", time.time())
        )


# ============================================================================
# Minecraft Bridge
# ============================================================================

class MinecraftBridge:
    """
    Bridge between Python NPCPU and Minecraft via Mineflayer.

    Handles:
    - Bot process management
    - Command/response communication
    - State synchronization
    - Event handling

    Example:
        bridge = MinecraftBridge(config)
        await bridge.connect()

        # Move the bot
        await bridge.move_to(Position(100, 64, 100))

        # Get world state
        state = await bridge.get_world_state()

        # Combat
        await bridge.attack_nearest_hostile()
    """

    def __init__(self, config: Optional[MinecraftConfig] = None):
        self.config = config or MinecraftConfig()

        # State
        self.connection_state = ConnectionState.DISCONNECTED
        self.bot_state = BotState()
        self.world_state = WorldState()

        # Process management
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        # Command/response handling
        self._command_queue: queue.Queue = queue.Queue()
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._pending_commands: Dict[str, BotCommand] = {}

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "connected": [],
            "disconnected": [],
            "state_update": [],
            "chat": [],
            "death": [],
            "spawn": [],
            "damage": [],
            "entity_spawn": [],
            "entity_gone": []
        }

    async def connect(self) -> bool:
        """Connect to Minecraft server"""
        if self.connection_state == ConnectionState.CONNECTED:
            return True

        self.connection_state = ConnectionState.CONNECTING

        try:
            # Start the Mineflayer bot process
            self._start_bot_process()

            # Wait for connection confirmation
            response = await self._send_command("connect", {
                "host": self.config.host,
                "port": self.config.port,
                "username": self.config.username,
                "version": self.config.version,
                "auth": self.config.auth
            }, timeout=30.0)

            if response and response.success:
                self.connection_state = ConnectionState.CONNECTED
                for callback in self._callbacks["connected"]:
                    callback()
                return True
            else:
                self.connection_state = ConnectionState.ERROR
                return False

        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            print(f"Connection error: {e}")
            return False

    def _start_bot_process(self):
        """Start the Node.js bot process"""
        script_path = Path(self.config.bot_script_path)

        # Use bundled script or generate minimal one
        if not script_path.exists():
            self._generate_bot_script(script_path)

        self._process = subprocess.Popen(
            [self.config.node_path, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        self._running = True
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _generate_bot_script(self, path: Path):
        """Generate the Mineflayer bot script"""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Script generation handled separately - see bot.js

    def _read_output(self):
        """Read output from bot process"""
        while self._running and self._process:
            try:
                line = self._process.stdout.readline()
                if line:
                    self._handle_bot_message(line.strip())
            except Exception as e:
                if self._running:
                    print(f"Read error: {e}")
                break

    def _handle_bot_message(self, message: str):
        """Handle a message from the bot"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "response":
                # Command response
                response = BotResponse(
                    command_id=data.get("command_id", ""),
                    success=data.get("success", False),
                    data=data.get("data"),
                    error=data.get("error")
                )
                self._handle_response(response)

            elif msg_type == "state_update":
                # Bot state update
                self._update_bot_state(data.get("state", {}))

            elif msg_type == "world_update":
                # World state update
                self._update_world_state(data.get("world", {}))

            elif msg_type == "event":
                # Game event
                self._handle_event(data.get("event", ""), data.get("data", {}))

        except json.JSONDecodeError:
            print(f"Invalid JSON from bot: {message}")

    def _handle_response(self, response: BotResponse):
        """Handle command response"""
        if response.command_id in self._response_futures:
            future = self._response_futures.pop(response.command_id)
            if not future.done():
                future.set_result(response)

    def _update_bot_state(self, state_data: Dict[str, Any]):
        """Update bot state from data"""
        if "position" in state_data:
            self.bot_state.position = Position.from_dict(state_data["position"])
        if "health" in state_data:
            self.bot_state.health = state_data["health"]
        if "food" in state_data:
            self.bot_state.food = state_data["food"]
        if "is_alive" in state_data:
            self.bot_state.is_alive = state_data["is_alive"]
        if "inventory" in state_data:
            self.bot_state.inventory = state_data["inventory"]

        self.bot_state.timestamp = time.time()

        for callback in self._callbacks["state_update"]:
            callback(self.bot_state)

    def _update_world_state(self, world_data: Dict[str, Any]):
        """Update world state from data"""
        if "nearby_entities" in world_data:
            self.world_state.nearby_entities = [
                Entity(
                    id=e["id"],
                    type=e["type"],
                    name=e.get("name"),
                    position=Position.from_dict(e["position"]),
                    health=e.get("health"),
                    is_hostile=e.get("is_hostile", False),
                    is_player=e.get("is_player", False),
                    distance=e.get("distance", 0)
                )
                for e in world_data["nearby_entities"]
            ]

        if "time" in world_data:
            self.world_state.time_of_day = world_data["time"]
        if "weather" in world_data:
            self.world_state.weather = world_data["weather"]
        if "biome" in world_data:
            self.world_state.biome = world_data["biome"]

    def _handle_event(self, event_name: str, data: Dict[str, Any]):
        """Handle a game event"""
        for callback in self._callbacks.get(event_name, []):
            callback(data)

    async def _send_command(
        self,
        action: str,
        params: Dict[str, Any] = None,
        timeout: float = 10.0
    ) -> Optional[BotResponse]:
        """Send a command to the bot"""
        command = BotCommand(action=action, params=params or {})

        # Create future for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._response_futures[command.id] = future
        self._pending_commands[command.id] = command

        # Send command
        try:
            if self._process and self._process.stdin:
                self._process.stdin.write(command.to_json() + "\n")
                self._process.stdin.flush()
        except Exception as e:
            print(f"Send error: {e}")
            return None

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._response_futures.pop(command.id, None)
            return None

    # ========================================================================
    # High-Level Bot Commands
    # ========================================================================

    async def move_to(self, position: Position) -> bool:
        """Move bot to a position"""
        response = await self._send_command("move_to", position.to_dict())
        return response and response.success

    async def look_at(self, position: Position) -> bool:
        """Look at a position"""
        response = await self._send_command("look_at", position.to_dict())
        return response and response.success

    async def attack(self, entity_id: int) -> bool:
        """Attack an entity"""
        response = await self._send_command("attack", {"entity_id": entity_id})
        return response and response.success

    async def attack_nearest_hostile(self) -> bool:
        """Attack the nearest hostile entity"""
        response = await self._send_command("attack_nearest_hostile", {})
        return response and response.success

    async def dig(self, position: Position) -> bool:
        """Dig/mine a block"""
        response = await self._send_command("dig", position.to_dict())
        return response and response.success

    async def place_block(self, position: Position, block_type: str) -> bool:
        """Place a block"""
        response = await self._send_command("place", {
            "position": position.to_dict(),
            "block": block_type
        })
        return response and response.success

    async def eat(self) -> bool:
        """Eat food from inventory"""
        response = await self._send_command("eat", {})
        return response and response.success

    async def craft(self, recipe: str, count: int = 1) -> bool:
        """Craft an item"""
        response = await self._send_command("craft", {
            "recipe": recipe,
            "count": count
        })
        return response and response.success

    async def chat(self, message: str) -> bool:
        """Send a chat message"""
        response = await self._send_command("chat", {"message": message})
        return response and response.success

    async def get_state(self) -> BotState:
        """Get current bot state"""
        response = await self._send_command("get_state", {})
        if response and response.success and response.data:
            self._update_bot_state(response.data)
        return self.bot_state

    async def get_world_state(self) -> WorldState:
        """Get current world state"""
        response = await self._send_command("get_world_state", {})
        if response and response.success and response.data:
            self._update_world_state(response.data)
        return self.world_state

    async def find_blocks(self, block_type: str, radius: int = 32) -> List[Position]:
        """Find blocks of a type nearby"""
        response = await self._send_command("find_blocks", {
            "block": block_type,
            "radius": radius
        })
        if response and response.success and response.data:
            return [Position.from_dict(p) for p in response.data]
        return []

    async def find_entities(self, entity_type: str = None, hostile_only: bool = False) -> List[Entity]:
        """Find entities nearby"""
        response = await self._send_command("find_entities", {
            "type": entity_type,
            "hostile_only": hostile_only
        })
        if response and response.success and response.data:
            return [
                Entity(
                    id=e["id"],
                    type=e["type"],
                    name=e.get("name"),
                    position=Position.from_dict(e["position"]),
                    is_hostile=e.get("is_hostile", False),
                    distance=e.get("distance", 0)
                )
                for e in response.data
            ]
        return []

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def disconnect(self):
        """Disconnect from Minecraft"""
        await self._send_command("disconnect", {})
        self._running = False

        if self._process:
            self._process.terminate()
            self._process = None

        self.connection_state = ConnectionState.DISCONNECTED

        for callback in self._callbacks["disconnected"]:
            callback()

    def on_event(self, event_name: str, callback: Callable):
        """Register an event callback"""
        if event_name not in self._callbacks:
            self._callbacks[event_name] = []
        self._callbacks[event_name].append(callback)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Minecraft Bridge")
    print("=" * 50)
    print("\nThis module provides the bridge between NPCPU and Minecraft.")
    print("It requires a Minecraft server and Node.js with Mineflayer installed.")
    print("\nSetup:")
    print("  1. npm install mineflayer mineflayer-pathfinder")
    print("  2. Start a Minecraft server (offline mode for testing)")
    print("  3. Configure MinecraftConfig with server details")
    print("  4. Use MinecraftBridge to connect and control bots")
