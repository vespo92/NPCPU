"""
Multi-Agent Colony System

Enables multiple NPCPU organisms to coordinate as a colony:
- Shared knowledge/memory
- Role specialization
- Task distribution
- Communication via Minecraft chat
- Collective decision making
- Territory management
"""

import time
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from minecraft.bridge import MinecraftBridge, MinecraftConfig, Position
from minecraft.organism_adapter import MinecraftOrganism
from minecraft.memory import MemorySystem, MemoryType, EmotionalValence


# ============================================================================
# Enums
# ============================================================================

class ColonyRole(Enum):
    """Roles organisms can have in the colony"""
    LEADER = "leader"               # Makes group decisions
    GATHERER = "gatherer"           # Collects resources
    BUILDER = "builder"             # Constructs structures
    GUARD = "guard"                 # Protects the colony
    SCOUT = "scout"                 # Explores new areas
    HEALER = "healer"               # Helps injured members
    WORKER = "worker"               # General tasks


class MessageType(Enum):
    """Types of colony messages"""
    RESOURCE_FOUND = "resource_found"
    DANGER_ALERT = "danger_alert"
    HELP_NEEDED = "help_needed"
    TASK_COMPLETE = "task_complete"
    RALLY = "rally"
    STATUS = "status"
    GREETING = "greeting"


class TaskPriority(Enum):
    """Task priorities"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ColonyMessage:
    """A message between colony members"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    message_type: MessageType = MessageType.STATUS
    content: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Position] = None
    timestamp: float = field(default_factory=time.time)
    ttl: float = 30.0               # Time to live in seconds


@dataclass
class ColonyTask:
    """A task assigned within the colony"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: Optional[str] = None
    position: Optional[Position] = None
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed: bool = False


@dataclass
class Territory:
    """Territory claimed by the colony"""
    center: Position
    radius: float = 50.0
    name: str = "Base"
    structures: List[str] = field(default_factory=list)
    resource_deposits: List[Position] = field(default_factory=list)


# ============================================================================
# Colony Member
# ============================================================================

class ColonyMember:
    """
    Wrapper for organism with colony capabilities.

    Adds:
    - Role assignment
    - Message handling
    - Task acceptance
    - Memory sharing
    """

    def __init__(
        self,
        organism: MinecraftOrganism,
        colony: 'Colony',
        role: ColonyRole = ColonyRole.WORKER
    ):
        self.organism = organism
        self.colony = colony
        self.role = role
        self.id = organism.body.identity.id

        # Personal memory
        self.memory = MemorySystem()

        # Current task
        self.current_task: Optional[ColonyTask] = None

        # Message inbox
        self.inbox: List[ColonyMessage] = []

        # Stats
        self.tasks_completed = 0
        self.resources_gathered = 0

        # Setup chat handler
        self._setup_chat_handler()

    def _setup_chat_handler(self):
        """Setup handler for incoming chat messages"""
        def on_chat(data):
            username = data.get("username", "")
            message = data.get("message", "")

            # Parse colony protocol messages
            if message.startswith("[COLONY]"):
                self._handle_colony_message(username, message)

        self.organism.bridge.on_event("chat", on_chat)

    def _handle_colony_message(self, sender: str, message: str):
        """Handle a colony protocol message"""
        try:
            # Format: [COLONY] TYPE: content
            parts = message.replace("[COLONY] ", "").split(": ", 1)
            msg_type = MessageType(parts[0].lower())
            content = parts[1] if len(parts) > 1 else ""

            colony_msg = ColonyMessage(
                sender_id=sender,
                message_type=msg_type,
                content={"raw": content}
            )

            self.inbox.append(colony_msg)

            # Auto-respond to certain messages
            if msg_type == MessageType.DANGER_ALERT and self.role == ColonyRole.GUARD:
                # Guards respond to danger
                asyncio.create_task(self._respond_to_danger(content))

        except (ValueError, IndexError):
            pass

    async def _respond_to_danger(self, content: str):
        """Respond to a danger alert"""
        # Move toward the danger to help
        self.organism.body.motor.queue_action(
            self.organism.body.motor.ActionType.DEFEND,
            priority=self.organism.body.motor.ActionPriority.URGENT
        )

    async def broadcast(self, message_type: MessageType, content: str):
        """Broadcast a message to the colony"""
        message = f"[COLONY] {message_type.value.upper()}: {content}"
        await self.organism.bridge.chat(message)

    async def report_resource(self, position: Position, resource_type: str, amount: int):
        """Report a found resource to the colony"""
        # Store in personal memory
        self.memory.remember_location(
            position,
            MemoryType.RESOURCE,
            resource_type,
            {"amount": amount},
            EmotionalValence.POSITIVE
        )

        # Share with colony
        self.colony.shared_memory.remember_location(
            position,
            MemoryType.RESOURCE,
            resource_type,
            {"amount": amount, "reported_by": self.id},
            EmotionalValence.POSITIVE
        )

        # Broadcast to others
        await self.broadcast(
            MessageType.RESOURCE_FOUND,
            f"{resource_type} at {int(position.x)},{int(position.z)}"
        )

    async def report_danger(self, position: Position, danger_type: str):
        """Report a danger to the colony"""
        # Store in memory
        self.memory.remember_location(
            position,
            MemoryType.DANGER,
            danger_type,
            {"reported_at": time.time()},
            EmotionalValence.NEGATIVE
        )

        self.colony.shared_memory.remember_location(
            position,
            MemoryType.DANGER,
            danger_type,
            {"reported_by": self.id},
            EmotionalValence.NEGATIVE
        )

        # Alert colony
        await self.broadcast(
            MessageType.DANGER_ALERT,
            f"{danger_type} at {int(position.x)},{int(position.z)}!"
        )

    async def request_help(self, reason: str):
        """Request help from colony members"""
        pos = self.organism.bridge.bot_state.position
        await self.broadcast(
            MessageType.HELP_NEEDED,
            f"{reason} at {int(pos.x)},{int(pos.z)}"
        )

    def accept_task(self, task: ColonyTask) -> bool:
        """Accept a task assignment"""
        if self.current_task and not self.current_task.completed:
            return False  # Already busy

        self.current_task = task
        task.assigned_to = self.id
        return True

    def complete_task(self):
        """Mark current task as complete"""
        if self.current_task:
            self.current_task.completed = True
            self.tasks_completed += 1
            self.current_task = None

    async def tick(self):
        """Process one member tick"""
        # Process inbox
        self._process_inbox()

        # Role-specific behavior
        await self._role_behavior()

        # Update organism
        await self.organism.tick()

    def _process_inbox(self):
        """Process incoming messages"""
        current_time = time.time()

        # Remove expired messages
        self.inbox = [m for m in self.inbox if current_time - m.timestamp < m.ttl]

        # Process remaining
        for message in self.inbox[:5]:  # Process up to 5
            self._react_to_message(message)

        self.inbox = self.inbox[5:]

    def _react_to_message(self, message: ColonyMessage):
        """React to a message based on role"""
        if message.message_type == MessageType.RALLY:
            # Everyone responds to rally
            if message.position:
                self.organism.body.motor.queue_action(
                    self.organism.body.motor.ActionType.MOVE,
                    parameters={"target": message.position.to_dict()}
                )

    async def _role_behavior(self):
        """Execute role-specific behavior"""
        if self.role == ColonyRole.SCOUT:
            await self._scout_behavior()
        elif self.role == ColonyRole.GATHERER:
            await self._gatherer_behavior()
        elif self.role == ColonyRole.GUARD:
            await self._guard_behavior()
        elif self.role == ColonyRole.BUILDER:
            await self._builder_behavior()

    async def _scout_behavior(self):
        """Scout explores and reports"""
        # Check for unexplored areas
        if np.random.random() < 0.3:
            self.organism.body.motor.queue_action(
                self.organism.body.motor.ActionType.EXPLORE
            )

        # Report any resources found
        world_state = await self.organism.bridge.get_world_state()
        # Would scan for resources and report them

    async def _gatherer_behavior(self):
        """Gatherer collects resources"""
        # Check shared memory for known resources
        resources = self.colony.shared_memory.recall_by_type(MemoryType.RESOURCE, limit=5)

        if resources and not self.current_task:
            # Go to nearest resource
            nearest = resources[0]
            self.organism.body.motor.queue_action(
                self.organism.body.motor.ActionType.ACQUIRE,
                parameters={
                    "target": nearest.position.to_dict(),
                    "block_type": nearest.label
                }
            )

    async def _guard_behavior(self):
        """Guard patrols and defends"""
        # Check for nearby threats
        world_state = await self.organism.bridge.get_world_state()

        hostiles = [e for e in world_state.nearby_entities if e.is_hostile]
        if hostiles:
            # Alert and attack
            await self.report_danger(
                hostiles[0].position,
                hostiles[0].type
            )
            self.organism.body.motor.queue_action(
                self.organism.body.motor.ActionType.DEFEND
            )

    async def _builder_behavior(self):
        """Builder constructs structures"""
        # Would check for build tasks
        pass


# ============================================================================
# Colony
# ============================================================================

class Colony:
    """
    A colony of NPCPU organisms in Minecraft.

    Features:
    - Shared memory/knowledge
    - Role assignment
    - Task distribution
    - Territory management
    - Collective survival

    Example:
        colony = Colony("AlphaColony", config)

        # Add members
        await colony.add_member("Worker1", ColonyRole.GATHERER)
        await colony.add_member("Worker2", ColonyRole.SCOUT)
        await colony.add_member("Worker3", ColonyRole.GUARD)

        # Run colony
        await colony.run()
    """

    def __init__(
        self,
        name: str,
        config: MinecraftConfig,
        base_position: Optional[Position] = None
    ):
        self.name = name
        self.config = config
        self.id = str(uuid.uuid4())

        # Members
        self.members: Dict[str, ColonyMember] = {}
        self.leader: Optional[ColonyMember] = None

        # Shared resources
        self.shared_memory = MemorySystem(max_memories=5000)

        # Territory
        self.territory = Territory(
            center=base_position or Position(0, 64, 0),
            name=f"{name}_Base"
        )

        # Tasks
        self.task_queue: List[ColonyTask] = []

        # State
        self.founded_at = time.time()
        self.tick_count = 0
        self.running = False

    async def add_member(
        self,
        name: str,
        role: ColonyRole = ColonyRole.WORKER
    ) -> ColonyMember:
        """Add a new member to the colony"""
        # Create organism with unique username
        member_config = MinecraftConfig(
            host=self.config.host,
            port=self.config.port,
            username=f"{self.name}_{name}",
            version=self.config.version,
            auth=self.config.auth
        )

        organism = MinecraftOrganism(name=name, config=member_config)
        member = ColonyMember(organism, self, role)

        self.members[member.id] = member

        # First member with LEADER role becomes leader
        if role == ColonyRole.LEADER and not self.leader:
            self.leader = member

        return member

    async def connect_all(self) -> int:
        """Connect all members to the server"""
        connected = 0
        for member in self.members.values():
            try:
                if await member.organism.connect():
                    connected += 1
                    # Stagger connections
                    await asyncio.sleep(1.0)
            except Exception as e:
                print(f"Failed to connect {member.organism.config.username}: {e}")

        return connected

    async def disconnect_all(self):
        """Disconnect all members"""
        for member in self.members.values():
            try:
                await member.organism.disconnect()
            except Exception:
                pass

    def create_task(
        self,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        position: Optional[Position] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> ColonyTask:
        """Create a new task for the colony"""
        task = ColonyTask(
            task_type=task_type,
            priority=priority,
            position=position,
            details=details or {}
        )
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value)
        return task

    def assign_tasks(self):
        """Assign pending tasks to available members"""
        for task in self.task_queue:
            if task.assigned_to or task.completed:
                continue

            # Find best member for task
            best_member = self._find_best_member_for_task(task)
            if best_member:
                best_member.accept_task(task)

    def _find_best_member_for_task(self, task: ColonyTask) -> Optional[ColonyMember]:
        """Find the best member for a task based on role"""
        role_preferences = {
            "gather": [ColonyRole.GATHERER, ColonyRole.WORKER],
            "build": [ColonyRole.BUILDER, ColonyRole.WORKER],
            "defend": [ColonyRole.GUARD, ColonyRole.WORKER],
            "scout": [ColonyRole.SCOUT, ColonyRole.WORKER],
            "explore": [ColonyRole.SCOUT, ColonyRole.WORKER]
        }

        preferred_roles = role_preferences.get(task.task_type, [ColonyRole.WORKER])

        for role in preferred_roles:
            for member in self.members.values():
                if member.role == role and member.current_task is None:
                    return member

        return None

    async def rally(self, position: Position, reason: str = ""):
        """Rally all members to a position"""
        if self.leader:
            await self.leader.broadcast(
                MessageType.RALLY,
                f"Rally at {int(position.x)},{int(position.z)}! {reason}"
            )

        for member in self.members.values():
            member.organism.body.motor.queue_action(
                member.organism.body.motor.ActionType.MOVE,
                parameters={"target": position.to_dict()},
                priority=member.organism.body.motor.ActionPriority.HIGH
            )

    async def tick(self):
        """Process one colony tick"""
        self.tick_count += 1

        # Assign tasks
        self.assign_tasks()

        # Clean completed tasks
        self.task_queue = [t for t in self.task_queue if not t.completed]

        # Tick all members
        for member in self.members.values():
            try:
                await member.tick()
            except Exception as e:
                print(f"Member tick error: {e}")

        # Colony-level decisions
        await self._colony_decisions()

        # Decay memories
        if self.tick_count % 100 == 0:
            self.shared_memory.decay_memories()

    async def _colony_decisions(self):
        """Make colony-level decisions"""
        # Check for dangers
        dangers = self.shared_memory.recall_by_type(MemoryType.DANGER, limit=5)
        recent_dangers = [
            d for d in dangers
            if time.time() - d.created_at < 60  # Last minute
        ]

        if recent_dangers and self.leader:
            # Rally guards to danger
            for danger in recent_dangers[:1]:
                await self.leader.broadcast(
                    MessageType.DANGER_ALERT,
                    f"Threat reported at {int(danger.position.x)},{int(danger.position.z)}"
                )

        # Create gather tasks if low on resources
        if self.tick_count % 50 == 0:
            resources = self.shared_memory.recall_by_type(MemoryType.RESOURCE, limit=10)
            if resources:
                self.create_task(
                    "gather",
                    TaskPriority.NORMAL,
                    resources[0].position,
                    {"resource": resources[0].label}
                )

    async def run(self, max_ticks: int = 10000, tick_delay: float = 0.5):
        """Run the colony simulation"""
        self.running = True
        print(f"\n{'='*60}")
        print(f"Colony '{self.name}' starting with {len(self.members)} members")
        print(f"{'='*60}")

        # Connect all members
        connected = await self.connect_all()
        print(f"Connected: {connected}/{len(self.members)} members")

        if connected == 0:
            print("No members connected. Exiting.")
            return

        try:
            tick = 0
            while self.running and tick < max_ticks:
                await self.tick()
                tick += 1

                if tick % 20 == 0:
                    self._print_status(tick)

                await asyncio.sleep(tick_delay)

        except KeyboardInterrupt:
            print("\nColony shutting down...")
        finally:
            await self.disconnect_all()
            print("Colony disconnected.")

    def _print_status(self, tick: int):
        """Print colony status"""
        alive = sum(1 for m in self.members.values() if m.organism.is_alive)
        tasks_pending = len([t for t in self.task_queue if not t.completed])

        print(f"[Tick {tick:5d}] Members: {alive}/{len(self.members)} | "
              f"Tasks: {tasks_pending} | "
              f"Memories: {len(self.shared_memory.spatial_memories)}")

    def get_status(self) -> Dict[str, Any]:
        """Get colony status"""
        return {
            "name": self.name,
            "members": len(self.members),
            "members_alive": sum(1 for m in self.members.values() if m.organism.is_alive),
            "roles": {
                role.value: sum(1 for m in self.members.values() if m.role == role)
                for role in ColonyRole
            },
            "tasks_pending": len([t for t in self.task_queue if not t.completed]),
            "tasks_completed": sum(m.tasks_completed for m in self.members.values()),
            "territory": {
                "center": self.territory.center.to_dict(),
                "radius": self.territory.radius
            },
            "shared_memories": self.shared_memory.get_summary(),
            "uptime": time.time() - self.founded_at,
            "tick_count": self.tick_count
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Colony System Demo")
    print("=" * 50)
    print("""
This module enables multi-agent colonies in Minecraft.

Example usage:

    import asyncio
    from minecraft.colony import Colony, ColonyRole
    from minecraft.bridge import MinecraftConfig

    async def main():
        config = MinecraftConfig(host="localhost", port=25565)
        colony = Colony("AlphaColony", config)

        # Add diverse members
        await colony.add_member("Scout1", ColonyRole.SCOUT)
        await colony.add_member("Guard1", ColonyRole.GUARD)
        await colony.add_member("Gatherer1", ColonyRole.GATHERER)
        await colony.add_member("Gatherer2", ColonyRole.GATHERER)
        await colony.add_member("Leader", ColonyRole.LEADER)

        # Run colony
        await colony.run(max_ticks=5000)

    asyncio.run(main())

Roles:
  - LEADER: Makes group decisions, coordinates
  - SCOUT: Explores, finds resources
  - GATHERER: Collects resources
  - GUARD: Patrols, defends from threats
  - BUILDER: Constructs structures
  - WORKER: General tasks

Features:
  - Shared memory between all members
  - Communication via Minecraft chat
  - Automatic task assignment
  - Role-specific behaviors
  - Collective threat response
    """)
