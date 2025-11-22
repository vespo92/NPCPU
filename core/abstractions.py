"""
Core Abstractions for NPCPU

Defines the fundamental interfaces that all NPCPU components must implement.
These abstractions enable:
- Swappable implementations
- Testing with mocks
- Custom organism types
- Alternative world simulations
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, List, Optional, Set, Callable,
    TypeVar, Generic, Iterator, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar('T')
TState = TypeVar('TState')
TConfig = TypeVar('TConfig')


# =============================================================================
# Enums
# =============================================================================

class LifecyclePhase(Enum):
    """Universal lifecycle phases for any entity"""
    INITIALIZING = auto()
    NASCENT = auto()      # Just born/created
    DEVELOPING = auto()   # Growing/learning
    MATURE = auto()       # Fully developed
    DECLINING = auto()    # Aging/degrading
    TERMINAL = auto()     # Near end
    ENDED = auto()        # Dead/terminated


class OrganismCapability(Enum):
    """Standard capabilities that organisms may have"""
    PERCEPTION = "perception"
    LOCOMOTION = "locomotion"
    MANIPULATION = "manipulation"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    LEARNING = "learning"
    REASONING = "reasoning"
    PLANNING = "planning"
    SOCIAL = "social"
    REPRODUCTION = "reproduction"
    SELF_REPAIR = "self_repair"
    ADAPTATION = "adaptation"


# =============================================================================
# Core Protocols (Structural Typing)
# =============================================================================

@runtime_checkable
class Tickable(Protocol):
    """Any entity that processes time steps"""
    def tick(self) -> None: ...


@runtime_checkable
class Serializable(Protocol):
    """Any entity that can be serialized/deserialized"""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...


@runtime_checkable
class Identifiable(Protocol):
    """Any entity with a unique identity"""
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class EntityState:
    """Base state container for any entity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(self):
        """Mark state as updated"""
        self.updated_at = datetime.now()


class BaseSubsystem(ABC):
    """
    Abstract base for organism subsystems.

    Subsystems are modular components that provide specific functionality
    to an organism (e.g., metabolism, nervous system, immune system).
    """

    def __init__(self, name: str, owner: Optional['BaseOrganism'] = None):
        self._name = name
        self._owner = owner
        self._enabled = True
        self._state: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def owner(self) -> Optional['BaseOrganism']:
        return self._owner

    @owner.setter
    def owner(self, organism: 'BaseOrganism'):
        self._owner = organism

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @abstractmethod
    def tick(self) -> None:
        """Process one time step"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current subsystem state"""
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore subsystem state"""
        pass

    def reset(self) -> None:
        """Reset to initial state"""
        self._state.clear()


class BaseOrganism(ABC):
    """
    Abstract base class for all organism types.

    An organism is a self-contained entity that:
    - Has a lifecycle (birth, growth, death)
    - Contains subsystems that provide functionality
    - Interacts with the world and other organisms
    - Has measurable capabilities and traits

    Implementations can range from simple reactive agents
    to complex conscious entities.
    """

    def __init__(self, name: str = "", **kwargs):
        self._id = str(uuid.uuid4())
        self._name = name or f"Organism_{self._id[:8]}"
        self._phase = LifecyclePhase.INITIALIZING
        self._subsystems: Dict[str, BaseSubsystem] = {}
        self._capabilities: Dict[OrganismCapability, float] = {}
        self._traits: Dict[str, float] = {}
        self._age: int = 0
        self._alive = True

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def phase(self) -> LifecyclePhase:
        return self._phase

    @property
    def age(self) -> int:
        return self._age

    @property
    def is_alive(self) -> bool:
        return self._alive and self._phase != LifecyclePhase.ENDED

    # -------------------------------------------------------------------------
    # Subsystems
    # -------------------------------------------------------------------------

    def add_subsystem(self, subsystem: BaseSubsystem) -> None:
        """Add a subsystem to this organism"""
        subsystem.owner = self
        self._subsystems[subsystem.name] = subsystem

    def get_subsystem(self, name: str) -> Optional[BaseSubsystem]:
        """Get a subsystem by name"""
        return self._subsystems.get(name)

    def remove_subsystem(self, name: str) -> Optional[BaseSubsystem]:
        """Remove and return a subsystem"""
        return self._subsystems.pop(name, None)

    @property
    def subsystems(self) -> Dict[str, BaseSubsystem]:
        """Get all subsystems"""
        return self._subsystems.copy()

    # -------------------------------------------------------------------------
    # Capabilities & Traits
    # -------------------------------------------------------------------------

    def get_capability(self, cap: OrganismCapability) -> float:
        """Get capability level (0.0 to 1.0)"""
        return self._capabilities.get(cap, 0.0)

    def set_capability(self, cap: OrganismCapability, level: float) -> None:
        """Set capability level"""
        self._capabilities[cap] = max(0.0, min(1.0, level))

    @property
    def capabilities(self) -> Dict[OrganismCapability, float]:
        return self._capabilities.copy()

    def get_trait(self, name: str, default: float = 0.0) -> float:
        """Get trait value"""
        return self._traits.get(name, default)

    def set_trait(self, name: str, value: float) -> None:
        """Set trait value"""
        self._traits[name] = value

    @property
    def traits(self) -> Dict[str, float]:
        return self._traits.copy()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @abstractmethod
    def tick(self) -> None:
        """Process one time step"""
        pass

    @abstractmethod
    def perceive(self, stimuli: Dict[str, Any]) -> None:
        """Process environmental stimuli"""
        pass

    @abstractmethod
    def decide(self) -> Optional[str]:
        """Make a decision about what action to take"""
        pass

    @abstractmethod
    def act(self, action: str) -> Any:
        """Execute an action"""
        pass

    def die(self, cause: str = "unknown") -> None:
        """End the organism's life"""
        self._alive = False
        self._phase = LifecyclePhase.ENDED
        self.on_death(cause)

    def on_death(self, cause: str) -> None:
        """Hook for death handling - override in subclasses"""
        pass

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize organism state"""
        return {
            "id": self._id,
            "name": self._name,
            "phase": self._phase.name,
            "age": self._age,
            "alive": self._alive,
            "capabilities": {k.value: v for k, v in self._capabilities.items()},
            "traits": self._traits.copy(),
            "subsystems": {
                name: sub.get_state()
                for name, sub in self._subsystems.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseOrganism':
        """Deserialize organism state - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement from_dict")


class BaseWorld(ABC):
    """
    Abstract base class for world/environment implementations.

    A world provides:
    - Environment that organisms live in
    - Resources for organisms to consume
    - Events that affect the simulation
    - Spatial organization (regions, areas)
    """

    def __init__(self, name: str = "World", **kwargs):
        self._id = str(uuid.uuid4())
        self._name = name
        self._tick_count = 0
        self._populations: Dict[str, 'BasePopulation'] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tick_count(self) -> int:
        return self._tick_count

    # -------------------------------------------------------------------------
    # Population Management
    # -------------------------------------------------------------------------

    def add_population(self, population: 'BasePopulation', location: str = "") -> None:
        """Add a population to this world"""
        self._populations[population.name] = population

    def get_population(self, name: str) -> Optional['BasePopulation']:
        """Get a population by name"""
        return self._populations.get(name)

    def remove_population(self, name: str) -> Optional['BasePopulation']:
        """Remove a population"""
        return self._populations.pop(name, None)

    @property
    def populations(self) -> Dict[str, 'BasePopulation']:
        return self._populations.copy()

    # -------------------------------------------------------------------------
    # World Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def tick(self) -> None:
        """Advance world by one time step"""
        pass

    @abstractmethod
    def get_stimuli_at(self, location: Any) -> Dict[str, Any]:
        """Get environmental stimuli at a location"""
        pass

    @abstractmethod
    def get_resources(self, resource_type: str) -> float:
        """Get available amount of a resource"""
        pass

    @abstractmethod
    def consume_resource(self, resource_type: str, amount: float) -> float:
        """Consume resources, returns actual amount consumed"""
        pass

    @abstractmethod
    def trigger_event(self, event_type: str, **kwargs) -> None:
        """Trigger a world event"""
        pass

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize world state"""
        return {
            "id": self._id,
            "name": self._name,
            "tick_count": self._tick_count,
            "populations": list(self._populations.keys())
        }


class BasePopulation(ABC):
    """
    Abstract base class for population management.

    A population is a collection of organisms that:
    - Share an environment
    - Can interact with each other
    - Have collective dynamics (birth rate, death rate)
    - May form social structures
    """

    def __init__(self, name: str = "Population", **kwargs):
        self._id = str(uuid.uuid4())
        self._name = name
        self._organisms: Dict[str, BaseOrganism] = {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return len(self._organisms)

    @property
    def organisms(self) -> Dict[str, BaseOrganism]:
        return self._organisms

    # -------------------------------------------------------------------------
    # Organism Management
    # -------------------------------------------------------------------------

    def add_organism(self, organism: BaseOrganism) -> None:
        """Add an organism to the population"""
        self._organisms[organism.id] = organism
        self.on_organism_added(organism)

    def remove_organism(self, organism_id: str) -> Optional[BaseOrganism]:
        """Remove an organism from the population"""
        organism = self._organisms.pop(organism_id, None)
        if organism:
            self.on_organism_removed(organism)
        return organism

    def get_organism(self, organism_id: str) -> Optional[BaseOrganism]:
        """Get an organism by ID"""
        return self._organisms.get(organism_id)

    def iter_organisms(self) -> Iterator[BaseOrganism]:
        """Iterate over all organisms"""
        return iter(self._organisms.values())

    def iter_alive(self) -> Iterator[BaseOrganism]:
        """Iterate over living organisms"""
        return (o for o in self._organisms.values() if o.is_alive)

    # -------------------------------------------------------------------------
    # Population Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def tick(self) -> None:
        """Process one time step for the population"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics"""
        pass

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def on_organism_added(self, organism: BaseOrganism) -> None:
        """Hook called when organism is added"""
        pass

    def on_organism_removed(self, organism: BaseOrganism) -> None:
        """Hook called when organism is removed"""
        pass

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize population state"""
        return {
            "id": self._id,
            "name": self._name,
            "size": self.size,
            "organism_ids": list(self._organisms.keys())
        }


# =============================================================================
# Behavior Abstraction
# =============================================================================

class BehaviorNode(ABC):
    """
    Base class for behavior tree nodes.

    Provides a standard interface for creating complex behaviors
    from simple, composable building blocks.
    """

    class Status(Enum):
        SUCCESS = auto()
        FAILURE = auto()
        RUNNING = auto()

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> 'BehaviorNode.Status':
        """Execute this behavior node"""
        pass

    def reset(self) -> None:
        """Reset node state"""
        pass


class CompositeNode(BehaviorNode):
    """A behavior node that contains child nodes"""

    def __init__(self, name: str = "", children: Optional[List[BehaviorNode]] = None):
        super().__init__(name)
        self.children = children or []

    def add_child(self, child: BehaviorNode) -> 'CompositeNode':
        self.children.append(child)
        return self

    def reset(self) -> None:
        for child in self.children:
            child.reset()


class SequenceNode(CompositeNode):
    """Executes children in sequence until one fails"""

    def execute(self, context: Dict[str, Any]) -> BehaviorNode.Status:
        for child in self.children:
            status = child.execute(context)
            if status != BehaviorNode.Status.SUCCESS:
                return status
        return BehaviorNode.Status.SUCCESS


class SelectorNode(CompositeNode):
    """Executes children until one succeeds"""

    def execute(self, context: Dict[str, Any]) -> BehaviorNode.Status:
        for child in self.children:
            status = child.execute(context)
            if status != BehaviorNode.Status.FAILURE:
                return status
        return BehaviorNode.Status.FAILURE


class ActionNode(BehaviorNode):
    """A leaf node that executes an action"""

    def __init__(self, name: str, action: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.action = action

    def execute(self, context: Dict[str, Any]) -> BehaviorNode.Status:
        try:
            result = self.action(context)
            return BehaviorNode.Status.SUCCESS if result else BehaviorNode.Status.FAILURE
        except Exception:
            return BehaviorNode.Status.FAILURE


class ConditionNode(BehaviorNode):
    """A leaf node that checks a condition"""

    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool]):
        super().__init__(name)
        self.condition = condition

    def execute(self, context: Dict[str, Any]) -> BehaviorNode.Status:
        try:
            result = self.condition(context)
            return BehaviorNode.Status.SUCCESS if result else BehaviorNode.Status.FAILURE
        except Exception:
            return BehaviorNode.Status.FAILURE


class BehaviorTree:
    """
    A behavior tree for controlling organism behavior.

    Example:
        tree = BehaviorTree(
            SelectorNode("survive", [
                SequenceNode("flee", [
                    ConditionNode("is_threatened", lambda ctx: ctx.get("threat_level", 0) > 0.5),
                    ActionNode("run_away", lambda ctx: ctx["organism"].flee())
                ]),
                SequenceNode("eat", [
                    ConditionNode("is_hungry", lambda ctx: ctx.get("energy", 100) < 30),
                    ActionNode("find_food", lambda ctx: ctx["organism"].find_food())
                ]),
                ActionNode("wander", lambda ctx: ctx["organism"].wander())
            ])
        )
    """

    def __init__(self, root: BehaviorNode):
        self.root = root

    def execute(self, context: Dict[str, Any]) -> BehaviorNode.Status:
        """Execute the behavior tree"""
        return self.root.execute(context)

    def reset(self) -> None:
        """Reset the tree state"""
        self.root.reset()
