"""
Swarm Coordination System for NPCPU

Implements swarm intelligence and multi-agent coordination using:
- Boid-like flocking behavior (separation, alignment, cohesion)
- Task allocation and distribution
- Emergent formation patterns
- Distributed decision making

Example:
    from swarm.coordination import SwarmCoordinator
    from core.simple_organism import SimpleOrganism

    # Create coordinator
    coordinator = SwarmCoordinator()

    # Add organisms to swarm
    swarm_id = coordinator.create_swarm("hunters")
    for i in range(10):
        org = SimpleOrganism(f"Hunter_{i}")
        coordinator.add_to_swarm(swarm_id, org, position=(i * 10, 0))

    # Calculate flocking behavior
    for org in coordinator.get_swarm_members(swarm_id):
        vectors = coordinator.calculate_flocking_vectors(org.id)
        print(f"{org.name}: {vectors}")

    # Allocate tasks
    tasks = [{"type": "patrol", "location": (100, 100)}, {"type": "gather", "resource": "food"}]
    assignments = coordinator.allocate_tasks(tasks, swarm_id)
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
import math
import random

from core.abstractions import BaseOrganism, BaseSubsystem
from core.events import get_event_bus, Event


# =============================================================================
# Data Types
# =============================================================================

class FormationPattern(Enum):
    """Standard formation patterns for swarms"""
    NONE = auto()        # No formation
    LINE = auto()         # Linear formation
    CIRCLE = auto()       # Circular formation
    WEDGE = auto()        # V-shaped wedge
    SQUARE = auto()       # Grid/square formation
    SCATTER = auto()      # Random scatter


@dataclass
class Position:
    """2D position with velocity"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def direction_to(self, other: 'Position') -> Tuple[float, float]:
        """Get normalized direction vector to another position"""
        dx = other.x - self.x
        dy = other.y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.0001:
            return (0.0, 0.0)
        return (dx / dist, dy / dist)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position':
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            vx=data.get("vx", 0.0),
            vy=data.get("vy", 0.0)
        )


@dataclass
class FlockingVectors:
    """Result of flocking calculations"""
    separation: Tuple[float, float] = (0.0, 0.0)  # Avoid crowding
    alignment: Tuple[float, float] = (0.0, 0.0)   # Match velocity
    cohesion: Tuple[float, float] = (0.0, 0.0)    # Move toward center
    combined: Tuple[float, float] = (0.0, 0.0)    # Weighted sum

    def to_dict(self) -> Dict[str, Any]:
        return {
            "separation": self.separation,
            "alignment": self.alignment,
            "cohesion": self.cohesion,
            "combined": self.combined
        }


@dataclass
class Task:
    """A task that can be allocated to organisms"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    priority: float = 1.0
    location: Optional[Tuple[float, float]] = None
    data: Dict[str, Any] = field(default_factory=dict)
    required_capability: Optional[str] = None
    assigned_to: Optional[str] = None
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority,
            "location": self.location,
            "data": self.data,
            "required_capability": self.required_capability,
            "assigned_to": self.assigned_to,
            "completed": self.completed
        }


@dataclass
class SwarmMember:
    """Wrapper for organism in swarm with spatial info"""
    organism: BaseOrganism
    position: Position
    role: str = "member"

    @property
    def id(self) -> str:
        return self.organism.id


# =============================================================================
# Swarm Coordinator
# =============================================================================

class SwarmCoordinator:
    """
    Coordinates multiple organisms for collective behavior.

    Features:
    - Flocking behavior (separation, alignment, cohesion)
    - Task allocation
    - Emergent formation patterns
    - Distributed decision making

    Example:
        coordinator = SwarmCoordinator()
        swarm_id = coordinator.create_swarm("pack")

        # Add organisms
        coordinator.add_to_swarm(swarm_id, organism1, position=(0, 0))
        coordinator.add_to_swarm(swarm_id, organism2, position=(10, 0))

        # Get flocking vectors
        vectors = coordinator.calculate_flocking_vectors(organism1.id)
        print(f"Move: {vectors.combined}")
    """

    def __init__(
        self,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        separation_radius: float = 25.0,
        neighbor_radius: float = 50.0,
        max_speed: float = 5.0
    ):
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_radius = separation_radius
        self.neighbor_radius = neighbor_radius
        self.max_speed = max_speed

        # Swarm storage: swarm_id -> {member_id -> SwarmMember}
        self._swarms: Dict[str, Dict[str, SwarmMember]] = {}
        # Organism to swarm mapping
        self._organism_swarm: Dict[str, str] = {}
        # Task storage
        self._tasks: Dict[str, Task] = {}
        # Formation settings per swarm
        self._formations: Dict[str, FormationPattern] = {}

        self._bus = get_event_bus()

    # -------------------------------------------------------------------------
    # Swarm Management
    # -------------------------------------------------------------------------

    def create_swarm(self, name: str = "") -> str:
        """Create a new swarm, returns swarm ID"""
        swarm_id = str(uuid.uuid4())
        self._swarms[swarm_id] = {}
        self._formations[swarm_id] = FormationPattern.NONE

        self._bus.emit("swarm.created", {
            "swarm_id": swarm_id,
            "name": name
        })

        return swarm_id

    def delete_swarm(self, swarm_id: str) -> bool:
        """Delete a swarm, returns True if existed"""
        if swarm_id not in self._swarms:
            return False

        # Remove organism mappings
        for member_id in self._swarms[swarm_id]:
            self._organism_swarm.pop(member_id, None)

        del self._swarms[swarm_id]
        self._formations.pop(swarm_id, None)

        self._bus.emit("swarm.deleted", {"swarm_id": swarm_id})
        return True

    def add_to_swarm(
        self,
        swarm_id: str,
        organism: BaseOrganism,
        position: Tuple[float, float] = (0.0, 0.0),
        role: str = "member"
    ) -> bool:
        """Add an organism to a swarm"""
        if swarm_id not in self._swarms:
            return False

        # Remove from previous swarm if any
        if organism.id in self._organism_swarm:
            self.remove_from_swarm(organism.id)

        member = SwarmMember(
            organism=organism,
            position=Position(x=position[0], y=position[1]),
            role=role
        )

        self._swarms[swarm_id][organism.id] = member
        self._organism_swarm[organism.id] = swarm_id

        self._bus.emit("swarm.member_added", {
            "swarm_id": swarm_id,
            "organism_id": organism.id,
            "position": position,
            "role": role
        })

        return True

    def remove_from_swarm(self, organism_id: str) -> bool:
        """Remove an organism from its swarm"""
        swarm_id = self._organism_swarm.pop(organism_id, None)
        if swarm_id and swarm_id in self._swarms:
            self._swarms[swarm_id].pop(organism_id, None)

            self._bus.emit("swarm.member_removed", {
                "swarm_id": swarm_id,
                "organism_id": organism_id
            })
            return True
        return False

    def get_swarm_members(self, swarm_id: str) -> List[BaseOrganism]:
        """Get all organisms in a swarm"""
        if swarm_id not in self._swarms:
            return []
        return [m.organism for m in self._swarms[swarm_id].values()]

    def get_member_position(self, organism_id: str) -> Optional[Position]:
        """Get position of organism in swarm"""
        swarm_id = self._organism_swarm.get(organism_id)
        if swarm_id and swarm_id in self._swarms:
            member = self._swarms[swarm_id].get(organism_id)
            if member:
                return member.position
        return None

    def set_member_position(
        self,
        organism_id: str,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0
    ) -> bool:
        """Update position and velocity of organism"""
        swarm_id = self._organism_swarm.get(organism_id)
        if swarm_id and swarm_id in self._swarms:
            member = self._swarms[swarm_id].get(organism_id)
            if member:
                member.position.x = x
                member.position.y = y
                member.position.vx = vx
                member.position.vy = vy
                return True
        return False

    def get_swarm_size(self, swarm_id: str) -> int:
        """Get number of members in swarm"""
        return len(self._swarms.get(swarm_id, {}))

    def get_swarm_center(self, swarm_id: str) -> Optional[Tuple[float, float]]:
        """Get center of mass of swarm"""
        if swarm_id not in self._swarms or not self._swarms[swarm_id]:
            return None

        members = self._swarms[swarm_id].values()
        cx = sum(m.position.x for m in members) / len(self._swarms[swarm_id])
        cy = sum(m.position.y for m in members) / len(self._swarms[swarm_id])
        return (cx, cy)

    # -------------------------------------------------------------------------
    # Flocking Behavior (Boids)
    # -------------------------------------------------------------------------

    def calculate_flocking_vectors(
        self,
        organism_id: str,
        separation_weight: Optional[float] = None,
        alignment_weight: Optional[float] = None,
        cohesion_weight: Optional[float] = None
    ) -> FlockingVectors:
        """
        Calculate boid-like flocking vectors for an organism.

        Returns vectors for:
        - Separation: steer away from nearby neighbors
        - Alignment: match velocity with neighbors
        - Cohesion: steer toward average position of neighbors
        - Combined: weighted sum of all three

        Example:
            vectors = coordinator.calculate_flocking_vectors(org.id)
            org.position.x += vectors.combined[0]
            org.position.y += vectors.combined[1]
        """
        sep_w = separation_weight if separation_weight is not None else self.separation_weight
        ali_w = alignment_weight if alignment_weight is not None else self.alignment_weight
        coh_w = cohesion_weight if cohesion_weight is not None else self.cohesion_weight

        # Find swarm and member
        swarm_id = self._organism_swarm.get(organism_id)
        if not swarm_id or swarm_id not in self._swarms:
            return FlockingVectors()

        member = self._swarms[swarm_id].get(organism_id)
        if not member:
            return FlockingVectors()

        # Get neighbors
        neighbors = self._get_neighbors(swarm_id, organism_id)
        if not neighbors:
            return FlockingVectors()

        pos = member.position

        # Calculate separation
        sep = self._calculate_separation(pos, neighbors)

        # Calculate alignment
        ali = self._calculate_alignment(neighbors)

        # Calculate cohesion
        coh = self._calculate_cohesion(pos, neighbors)

        # Combine vectors
        combined_x = sep[0] * sep_w + ali[0] * ali_w + coh[0] * coh_w
        combined_y = sep[1] * sep_w + ali[1] * ali_w + coh[1] * coh_w

        # Limit speed
        speed = math.sqrt(combined_x**2 + combined_y**2)
        if speed > self.max_speed:
            combined_x = (combined_x / speed) * self.max_speed
            combined_y = (combined_y / speed) * self.max_speed

        return FlockingVectors(
            separation=sep,
            alignment=ali,
            cohesion=coh,
            combined=(combined_x, combined_y)
        )

    def _get_neighbors(self, swarm_id: str, organism_id: str) -> List[SwarmMember]:
        """Get nearby swarm members"""
        if swarm_id not in self._swarms:
            return []

        member = self._swarms[swarm_id].get(organism_id)
        if not member:
            return []

        neighbors = []
        for other_id, other in self._swarms[swarm_id].items():
            if other_id == organism_id:
                continue
            dist = member.position.distance_to(other.position)
            if dist <= self.neighbor_radius:
                neighbors.append(other)

        return neighbors

    def _calculate_separation(
        self,
        pos: Position,
        neighbors: List[SwarmMember]
    ) -> Tuple[float, float]:
        """Calculate separation vector - steer away from close neighbors"""
        if not neighbors:
            return (0.0, 0.0)

        steer_x, steer_y = 0.0, 0.0
        count = 0

        for neighbor in neighbors:
            dist = pos.distance_to(neighbor.position)
            if dist < self.separation_radius and dist > 0:
                # Weight by inverse distance
                diff_x = pos.x - neighbor.position.x
                diff_y = pos.y - neighbor.position.y
                diff_x /= dist
                diff_y /= dist
                steer_x += diff_x
                steer_y += diff_y
                count += 1

        if count > 0:
            steer_x /= count
            steer_y /= count

        return (steer_x, steer_y)

    def _calculate_alignment(
        self,
        neighbors: List[SwarmMember]
    ) -> Tuple[float, float]:
        """Calculate alignment vector - match velocity with neighbors"""
        if not neighbors:
            return (0.0, 0.0)

        avg_vx = sum(n.position.vx for n in neighbors) / len(neighbors)
        avg_vy = sum(n.position.vy for n in neighbors) / len(neighbors)

        return (avg_vx, avg_vy)

    def _calculate_cohesion(
        self,
        pos: Position,
        neighbors: List[SwarmMember]
    ) -> Tuple[float, float]:
        """Calculate cohesion vector - steer toward center of neighbors"""
        if not neighbors:
            return (0.0, 0.0)

        center_x = sum(n.position.x for n in neighbors) / len(neighbors)
        center_y = sum(n.position.y for n in neighbors) / len(neighbors)

        return pos.direction_to(Position(x=center_x, y=center_y))

    # -------------------------------------------------------------------------
    # Task Allocation
    # -------------------------------------------------------------------------

    def allocate_tasks(
        self,
        tasks: List[Dict[str, Any]],
        swarm_id: str,
        strategy: str = "nearest"
    ) -> Dict[str, str]:
        """
        Allocate tasks to organisms in a swarm.

        Strategies:
        - "nearest": Assign to nearest available organism
        - "balanced": Distribute evenly
        - "capability": Match by required capability

        Returns: Dict mapping task_id -> organism_id

        Example:
            tasks = [
                {"type": "patrol", "location": (100, 100)},
                {"type": "gather", "location": (50, 50)}
            ]
            assignments = coordinator.allocate_tasks(tasks, swarm_id)
        """
        if swarm_id not in self._swarms:
            return {}

        members = list(self._swarms[swarm_id].values())
        if not members:
            return {}

        # Convert dict tasks to Task objects
        task_objects = []
        for t in tasks:
            task = Task(
                type=t.get("type", ""),
                priority=t.get("priority", 1.0),
                location=t.get("location"),
                data=t.get("data", {}),
                required_capability=t.get("required_capability")
            )
            task_objects.append(task)
            self._tasks[task.id] = task

        # Sort by priority (higher first)
        task_objects.sort(key=lambda t: -t.priority)

        assignments = {}
        assigned_organisms: Set[str] = set()

        for task in task_objects:
            organism_id = self._select_organism_for_task(
                task, members, assigned_organisms, strategy
            )
            if organism_id:
                task.assigned_to = organism_id
                assignments[task.id] = organism_id
                assigned_organisms.add(organism_id)

                self._bus.emit("swarm.task_assigned", {
                    "task_id": task.id,
                    "task_type": task.type,
                    "organism_id": organism_id,
                    "swarm_id": swarm_id
                })

        return assignments

    def _select_organism_for_task(
        self,
        task: Task,
        members: List[SwarmMember],
        assigned: Set[str],
        strategy: str
    ) -> Optional[str]:
        """Select best organism for a task"""
        available = [m for m in members if m.id not in assigned]
        if not available:
            return None

        if strategy == "nearest" and task.location:
            # Find nearest to task location
            target = Position(x=task.location[0], y=task.location[1])
            available.sort(key=lambda m: m.position.distance_to(target))
            return available[0].id

        elif strategy == "capability" and task.required_capability:
            # Find organism with best matching capability
            best_member = None
            best_score = -1
            cap_name = task.required_capability

            for member in available:
                # Check organism capabilities
                for cap in member.organism.capabilities:
                    if cap.value == cap_name:
                        score = member.organism.get_capability(cap)
                        if score > best_score:
                            best_score = score
                            best_member = member
                        break

            if best_member:
                return best_member.id

        # Default: balanced/random
        return random.choice(available).id

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed"""
        if task_id in self._tasks:
            self._tasks[task_id].completed = True
            self._bus.emit("swarm.task_completed", {
                "task_id": task_id,
                "organism_id": self._tasks[task_id].assigned_to
            })
            return True
        return False

    def get_organism_tasks(self, organism_id: str) -> List[Task]:
        """Get all tasks assigned to an organism"""
        return [t for t in self._tasks.values()
                if t.assigned_to == organism_id and not t.completed]

    # -------------------------------------------------------------------------
    # Formation Patterns
    # -------------------------------------------------------------------------

    def set_formation(self, swarm_id: str, pattern: FormationPattern) -> bool:
        """Set formation pattern for swarm"""
        if swarm_id not in self._swarms:
            return False
        self._formations[swarm_id] = pattern
        return True

    def calculate_formation_positions(
        self,
        swarm_id: str,
        center: Tuple[float, float] = (0.0, 0.0),
        spacing: float = 20.0,
        facing: float = 0.0  # Angle in radians
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate target positions for formation.

        Returns: Dict mapping organism_id -> (x, y) target position

        Example:
            positions = coordinator.calculate_formation_positions(
                swarm_id, center=(100, 100), spacing=25.0
            )
            for org_id, (x, y) in positions.items():
                coordinator.set_member_position(org_id, x, y)
        """
        if swarm_id not in self._swarms:
            return {}

        pattern = self._formations.get(swarm_id, FormationPattern.NONE)
        members = list(self._swarms[swarm_id].keys())
        n = len(members)

        if n == 0:
            return {}

        positions = {}

        if pattern == FormationPattern.LINE:
            positions = self._line_formation(members, center, spacing, facing)
        elif pattern == FormationPattern.CIRCLE:
            positions = self._circle_formation(members, center, spacing)
        elif pattern == FormationPattern.WEDGE:
            positions = self._wedge_formation(members, center, spacing, facing)
        elif pattern == FormationPattern.SQUARE:
            positions = self._square_formation(members, center, spacing)
        elif pattern == FormationPattern.SCATTER:
            positions = self._scatter_formation(members, center, spacing)
        else:
            # No formation - return current positions
            for org_id in members:
                member = self._swarms[swarm_id][org_id]
                positions[org_id] = (member.position.x, member.position.y)

        return positions

    def _line_formation(
        self,
        members: List[str],
        center: Tuple[float, float],
        spacing: float,
        facing: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate line formation positions"""
        n = len(members)
        positions = {}

        # Perpendicular to facing direction
        perp = facing + math.pi / 2

        for i, org_id in enumerate(members):
            offset = (i - (n - 1) / 2) * spacing
            x = center[0] + offset * math.cos(perp)
            y = center[1] + offset * math.sin(perp)
            positions[org_id] = (x, y)

        return positions

    def _circle_formation(
        self,
        members: List[str],
        center: Tuple[float, float],
        spacing: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate circle formation positions"""
        n = len(members)
        positions = {}

        # Radius based on spacing and number of members
        radius = (spacing * n) / (2 * math.pi) if n > 1 else 0

        for i, org_id in enumerate(members):
            angle = (2 * math.pi * i) / n
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions[org_id] = (x, y)

        return positions

    def _wedge_formation(
        self,
        members: List[str],
        center: Tuple[float, float],
        spacing: float,
        facing: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate V-wedge formation positions"""
        positions = {}
        wedge_angle = math.pi / 4  # 45 degree angle

        for i, org_id in enumerate(members):
            if i == 0:
                # Leader at front
                positions[org_id] = center
            else:
                # Alternate left/right
                row = (i + 1) // 2
                side = 1 if i % 2 == 1 else -1

                # Calculate position behind and to the side
                back = facing + math.pi  # Opposite of facing
                lateral = facing + (math.pi / 2) * side

                x = center[0] + row * spacing * math.cos(back + side * wedge_angle)
                y = center[1] + row * spacing * math.sin(back + side * wedge_angle)
                positions[org_id] = (x, y)

        return positions

    def _square_formation(
        self,
        members: List[str],
        center: Tuple[float, float],
        spacing: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate grid/square formation positions"""
        n = len(members)
        positions = {}

        # Calculate grid dimensions
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        # Center offset
        offset_x = (cols - 1) * spacing / 2
        offset_y = (rows - 1) * spacing / 2

        for i, org_id in enumerate(members):
            col = i % cols
            row = i // cols
            x = center[0] + col * spacing - offset_x
            y = center[1] + row * spacing - offset_y
            positions[org_id] = (x, y)

        return positions

    def _scatter_formation(
        self,
        members: List[str],
        center: Tuple[float, float],
        spacing: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate scattered/random formation positions"""
        n = len(members)
        positions = {}

        radius = spacing * math.sqrt(n) / 2

        for org_id in members:
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, radius)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            positions[org_id] = (x, y)

        return positions

    # -------------------------------------------------------------------------
    # Distributed Decision Making
    # -------------------------------------------------------------------------

    def vote(
        self,
        swarm_id: str,
        options: List[str],
        voter_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Conduct a vote among swarm members.

        Each organism votes randomly (can be overridden via vote_callback).
        Returns (winning_option, vote_tallies).

        Example:
            winner, tallies = coordinator.vote(
                swarm_id,
                options=["attack", "defend", "retreat"]
            )
            print(f"Swarm decided to: {winner}")
        """
        if swarm_id not in self._swarms or not options:
            return ("", {})

        members = self._swarms[swarm_id]
        tallies: Dict[str, float] = {opt: 0.0 for opt in options}

        for org_id, member in members.items():
            # Weight is 1.0 by default, can be customized
            weight = voter_weights.get(org_id, 1.0) if voter_weights else 1.0

            # Simple random vote (can be extended with organism preferences)
            vote = random.choice(options)
            tallies[vote] += weight

        winner = max(tallies, key=tallies.get)

        self._bus.emit("swarm.vote_completed", {
            "swarm_id": swarm_id,
            "options": options,
            "winner": winner,
            "tallies": tallies
        })

        return (winner, tallies)

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Get coordinator state for serialization"""
        swarms_state = {}
        for swarm_id, members in self._swarms.items():
            swarms_state[swarm_id] = {
                org_id: {
                    "position": m.position.to_dict(),
                    "role": m.role
                }
                for org_id, m in members.items()
            }

        return {
            "swarms": swarms_state,
            "formations": {sid: f.name for sid, f in self._formations.items()},
            "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
            "settings": {
                "separation_weight": self.separation_weight,
                "alignment_weight": self.alignment_weight,
                "cohesion_weight": self.cohesion_weight,
                "separation_radius": self.separation_radius,
                "neighbor_radius": self.neighbor_radius,
                "max_speed": self.max_speed
            }
        }

    def tick(self) -> None:
        """Update all swarms - apply flocking to all members"""
        for swarm_id in self._swarms:
            for org_id in list(self._swarms[swarm_id].keys()):
                member = self._swarms[swarm_id].get(org_id)
                if not member:
                    continue

                # Calculate flocking
                vectors = self.calculate_flocking_vectors(org_id)

                # Update velocity
                member.position.vx = vectors.combined[0]
                member.position.vy = vectors.combined[1]

                # Update position
                member.position.x += member.position.vx
                member.position.y += member.position.vy

        self._bus.emit("swarm.tick", {
            "swarm_count": len(self._swarms),
            "total_members": sum(len(s) for s in self._swarms.values())
        })


# =============================================================================
# Swarm Subsystem (for individual organisms)
# =============================================================================

class SwarmSubsystem(BaseSubsystem):
    """
    Subsystem for organisms participating in swarms.

    Provides interface between organism and SwarmCoordinator.

    Example:
        org = SimpleOrganism("Alpha")
        swarm_sub = SwarmSubsystem(coordinator)
        org.add_subsystem(swarm_sub)

        # Join swarm
        swarm_sub.join_swarm(swarm_id, position=(100, 100))

        # Get movement direction
        direction = swarm_sub.get_flocking_direction()
    """

    def __init__(self, coordinator: SwarmCoordinator):
        super().__init__("swarm")
        self._coordinator = coordinator
        self._swarm_id: Optional[str] = None

    def tick(self) -> None:
        """Update swarm participation"""
        if not self.enabled or not self._owner:
            return

        # Sync organism state with coordinator
        if self._swarm_id:
            pos = self._coordinator.get_member_position(self._owner.id)
            if pos:
                # Store position in state for easy access
                self._state["position"] = pos.to_dict()

    def join_swarm(
        self,
        swarm_id: str,
        position: Tuple[float, float] = (0.0, 0.0),
        role: str = "member"
    ) -> bool:
        """Join a swarm"""
        if not self._owner:
            return False

        if self._coordinator.add_to_swarm(swarm_id, self._owner, position, role):
            self._swarm_id = swarm_id
            return True
        return False

    def leave_swarm(self) -> bool:
        """Leave current swarm"""
        if not self._owner:
            return False

        if self._coordinator.remove_from_swarm(self._owner.id):
            self._swarm_id = None
            return True
        return False

    def get_flocking_direction(self) -> Tuple[float, float]:
        """Get recommended movement direction from flocking"""
        if not self._owner:
            return (0.0, 0.0)

        vectors = self._coordinator.calculate_flocking_vectors(self._owner.id)
        return vectors.combined

    def get_swarm_center(self) -> Optional[Tuple[float, float]]:
        """Get center of current swarm"""
        if self._swarm_id:
            return self._coordinator.get_swarm_center(self._swarm_id)
        return None

    def get_position(self) -> Optional[Position]:
        """Get current position in swarm"""
        if self._owner:
            return self._coordinator.get_member_position(self._owner.id)
        return None

    def set_position(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> bool:
        """Update position"""
        if self._owner:
            return self._coordinator.set_member_position(self._owner.id, x, y, vx, vy)
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "swarm_id": self._swarm_id,
            **self._state
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._swarm_id = state.get("swarm_id")
        self._state = {k: v for k, v in state.items() if k != "swarm_id"}
