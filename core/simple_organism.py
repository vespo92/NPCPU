"""
Simple Organism Implementation

A straightforward organism implementation using the core abstractions.
Demonstrates how to create custom organisms with:
- Basic lifecycle (birth, growth, aging, death)
- Energy-based metabolism
- Simple decision making
- Behavior trees
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import random

from .abstractions import (
    BaseOrganism, BaseSubsystem, LifecyclePhase, OrganismCapability,
    BehaviorTree, SequenceNode, SelectorNode, ActionNode, ConditionNode
)
from .events import Event, get_event_bus


# =============================================================================
# Simple Subsystems
# =============================================================================

class EnergySubsystem(BaseSubsystem):
    """Simple energy/metabolism subsystem"""

    def __init__(self, max_energy: float = 100.0, consumption_rate: float = 1.0):
        super().__init__("energy")
        self.max_energy = max_energy
        self.energy = max_energy
        self.consumption_rate = consumption_rate

    def tick(self) -> None:
        if not self.enabled:
            return
        # Consume energy each tick
        self.energy = max(0, self.energy - self.consumption_rate)

    def consume(self, amount: float) -> float:
        """Consume energy, returns actual consumed"""
        actual = min(amount, self.energy)
        self.energy -= actual
        return actual

    def restore(self, amount: float) -> float:
        """Restore energy, returns actual restored"""
        space = self.max_energy - self.energy
        actual = min(amount, space)
        self.energy += actual
        return actual

    @property
    def percentage(self) -> float:
        return self.energy / self.max_energy if self.max_energy > 0 else 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "max_energy": self.max_energy,
            "consumption_rate": self.consumption_rate
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.energy = state.get("energy", self.energy)
        self.max_energy = state.get("max_energy", self.max_energy)
        self.consumption_rate = state.get("consumption_rate", self.consumption_rate)


class HealthSubsystem(BaseSubsystem):
    """Simple health tracking subsystem"""

    def __init__(self, max_health: float = 100.0, regen_rate: float = 0.1):
        super().__init__("health")
        self.max_health = max_health
        self.health = max_health
        self.regen_rate = regen_rate

    def tick(self) -> None:
        if not self.enabled:
            return
        # Regenerate health if organism has energy
        if self.owner:
            energy = self.owner.get_subsystem("energy")
            if energy and energy.percentage > 0.2:
                self.health = min(self.max_health, self.health + self.regen_rate)

    def damage(self, amount: float) -> float:
        """Apply damage, returns actual damage taken"""
        actual = min(amount, self.health)
        self.health -= actual
        return actual

    def heal(self, amount: float) -> float:
        """Heal damage, returns actual healed"""
        space = self.max_health - self.health
        actual = min(amount, space)
        self.health += actual
        return actual

    @property
    def percentage(self) -> float:
        return self.health / self.max_health if self.max_health > 0 else 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "health": self.health,
            "max_health": self.max_health,
            "regen_rate": self.regen_rate
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.health = state.get("health", self.health)
        self.max_health = state.get("max_health", self.max_health)
        self.regen_rate = state.get("regen_rate", self.regen_rate)


class PerceptionSubsystem(BaseSubsystem):
    """Simple perception/awareness subsystem"""

    def __init__(self, range: float = 10.0):
        super().__init__("perception")
        self.range = range
        self.current_stimuli: Dict[str, Any] = {}
        self.memory: List[Dict[str, Any]] = []
        self.memory_size = 10

    def tick(self) -> None:
        pass  # Perception is passive

    def perceive(self, stimuli: Dict[str, Any]) -> None:
        """Process environmental stimuli"""
        self.current_stimuli = stimuli.copy()

        # Remember significant events
        if stimuli.get("threat_level", 0) > 0.5 or stimuli.get("food_nearby", False):
            self.memory.append(stimuli.copy())
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "range": self.range,
            "current_stimuli": self.current_stimuli,
            "memory": self.memory
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.range = state.get("range", self.range)
        self.current_stimuli = state.get("current_stimuli", {})
        self.memory = state.get("memory", [])


# =============================================================================
# Simple Organism
# =============================================================================

class SimpleOrganism(BaseOrganism):
    """
    A simple but complete organism implementation.

    Features:
    - Energy-based survival
    - Health and regeneration
    - Basic perception
    - Behavior tree for decisions
    - Lifecycle progression

    Example:
        org = SimpleOrganism("Alpha")
        org.perceive({"food_nearby": True, "threat_level": 0.2})

        while org.is_alive:
            org.tick()
            action = org.decide()
            if action:
                org.act(action)
    """

    def __init__(
        self,
        name: str = "",
        traits: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # Initialize traits
        self._traits = traits or self._generate_default_traits()

        # Add subsystems
        self.add_subsystem(EnergySubsystem(
            max_energy=100.0 * self._traits.get("vitality", 1.0),
            consumption_rate=1.0 * self._traits.get("metabolism", 1.0)
        ))
        self.add_subsystem(HealthSubsystem(
            max_health=100.0 * self._traits.get("resilience", 1.0),
            regen_rate=0.1 * self._traits.get("recovery", 1.0)
        ))
        self.add_subsystem(PerceptionSubsystem(
            range=10.0 * self._traits.get("awareness", 1.0)
        ))

        # Set capabilities based on traits
        self._setup_capabilities()

        # Create behavior tree
        self._behavior_tree = self._create_behavior_tree()

        # State
        self._phase = LifecyclePhase.NASCENT
        self._current_action: Optional[str] = None
        self._action_result: Any = None

    def _generate_default_traits(self) -> Dict[str, float]:
        """Generate random default traits"""
        return {
            "vitality": random.uniform(0.8, 1.2),
            "metabolism": random.uniform(0.8, 1.2),
            "resilience": random.uniform(0.8, 1.2),
            "recovery": random.uniform(0.8, 1.2),
            "awareness": random.uniform(0.8, 1.2),
            "aggression": random.uniform(0.0, 1.0),
            "sociability": random.uniform(0.0, 1.0),
        }

    def _setup_capabilities(self) -> None:
        """Set capabilities based on traits"""
        self.set_capability(OrganismCapability.PERCEPTION, self._traits.get("awareness", 1.0))
        self.set_capability(OrganismCapability.LOCOMOTION, 0.8)
        self.set_capability(OrganismCapability.SELF_REPAIR, self._traits.get("recovery", 1.0))
        self.set_capability(OrganismCapability.ADAPTATION, 0.5)

    def _create_behavior_tree(self) -> BehaviorTree:
        """Create behavior tree for decision making"""
        return BehaviorTree(
            SelectorNode("survive", [
                # Priority 1: Flee from threats
                SequenceNode("flee", [
                    ConditionNode("is_threatened", self._is_threatened),
                    ActionNode("flee", self._action_flee)
                ]),
                # Priority 2: Find food when hungry
                SequenceNode("eat", [
                    ConditionNode("is_hungry", self._is_hungry),
                    ActionNode("find_food", self._action_find_food)
                ]),
                # Priority 3: Rest when tired
                SequenceNode("rest", [
                    ConditionNode("is_tired", self._is_tired),
                    ActionNode("rest", self._action_rest)
                ]),
                # Priority 4: Explore
                ActionNode("explore", self._action_explore)
            ])
        )

    # -------------------------------------------------------------------------
    # Behavior Conditions
    # -------------------------------------------------------------------------

    def _is_threatened(self, context: Dict[str, Any]) -> bool:
        perception = self.get_subsystem("perception")
        if perception:
            return perception.current_stimuli.get("threat_level", 0) > 0.5
        return False

    def _is_hungry(self, context: Dict[str, Any]) -> bool:
        energy = self.get_subsystem("energy")
        return energy and energy.percentage < 0.3

    def _is_tired(self, context: Dict[str, Any]) -> bool:
        energy = self.get_subsystem("energy")
        return energy and energy.percentage < 0.5

    # -------------------------------------------------------------------------
    # Behavior Actions
    # -------------------------------------------------------------------------

    def _action_flee(self, context: Dict[str, Any]) -> bool:
        self._current_action = "flee"
        energy = self.get_subsystem("energy")
        if energy:
            energy.consume(5)  # Fleeing costs energy
        return True

    def _action_find_food(self, context: Dict[str, Any]) -> bool:
        self._current_action = "find_food"
        perception = self.get_subsystem("perception")
        if perception and perception.current_stimuli.get("food_nearby"):
            # Found food!
            energy = self.get_subsystem("energy")
            if energy:
                energy.restore(30)
            return True
        return False

    def _action_rest(self, context: Dict[str, Any]) -> bool:
        self._current_action = "rest"
        energy = self.get_subsystem("energy")
        if energy:
            energy.restore(5)
        return True

    def _action_explore(self, context: Dict[str, Any]) -> bool:
        self._current_action = "explore"
        energy = self.get_subsystem("energy")
        if energy:
            energy.consume(2)
        return True

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def tick(self) -> None:
        """Process one time step"""
        if not self.is_alive:
            return

        self._age += 1

        # Update lifecycle phase
        self._update_phase()

        # Tick all subsystems
        for subsystem in self._subsystems.values():
            subsystem.tick()

        # Check death conditions
        self._check_death()

        # Emit tick event
        bus = get_event_bus()
        bus.emit("organism.tick", {
            "organism_id": self._id,
            "organism_name": self._name,
            "age": self._age,
            "phase": self._phase.name
        }, source=self._id)

    def _update_phase(self) -> None:
        """Update lifecycle phase based on age"""
        if self._phase == LifecyclePhase.NASCENT and self._age > 10:
            self._phase = LifecyclePhase.DEVELOPING
        elif self._phase == LifecyclePhase.DEVELOPING and self._age > 50:
            self._phase = LifecyclePhase.MATURE
        elif self._phase == LifecyclePhase.MATURE and self._age > 200:
            self._phase = LifecyclePhase.DECLINING
        elif self._phase == LifecyclePhase.DECLINING and self._age > 300:
            self._phase = LifecyclePhase.TERMINAL

    def _check_death(self) -> None:
        """Check if organism should die"""
        energy = self.get_subsystem("energy")
        health = self.get_subsystem("health")

        if energy and energy.energy <= 0:
            self.die("starvation")
        elif health and health.health <= 0:
            self.die("damage")
        elif self._age > 400:
            self.die("old_age")

    def perceive(self, stimuli: Dict[str, Any]) -> None:
        """Process environmental stimuli"""
        perception = self.get_subsystem("perception")
        if perception:
            perception.perceive(stimuli)

    def decide(self) -> Optional[str]:
        """Decide on an action using behavior tree"""
        if not self.is_alive:
            return None

        context = {
            "organism": self,
            "age": self._age,
            "phase": self._phase
        }

        self._current_action = None
        self._behavior_tree.execute(context)

        return self._current_action

    def act(self, action: str) -> Any:
        """Execute an action"""
        # Action was already executed in behavior tree
        # This method is for external action triggering
        if action == "flee":
            return self._action_flee({})
        elif action == "find_food":
            return self._action_find_food({})
        elif action == "rest":
            return self._action_rest({})
        elif action == "explore":
            return self._action_explore({})
        return None

    def on_death(self, cause: str) -> None:
        """Handle death"""
        bus = get_event_bus()
        bus.emit("organism.died", {
            "organism_id": self._id,
            "organism_name": self._name,
            "cause": cause,
            "age": self._age
        }, source=self._id)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize organism state"""
        data = super().to_dict()
        data["current_action"] = self._current_action
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleOrganism':
        """Deserialize organism state"""
        org = cls(
            name=data.get("name", ""),
            traits=data.get("traits", {})
        )
        org._id = data.get("id", org._id)
        org._age = data.get("age", 0)
        org._alive = data.get("alive", True)

        # Restore phase
        phase_name = data.get("phase", "NASCENT")
        try:
            org._phase = LifecyclePhase[phase_name]
        except KeyError:
            org._phase = LifecyclePhase.NASCENT

        # Restore subsystems
        for name, state in data.get("subsystems", {}).items():
            subsystem = org.get_subsystem(name)
            if subsystem:
                subsystem.set_state(state)

        return org


# =============================================================================
# Simple Population
# =============================================================================

class SimplePopulation:
    """
    A simple population manager for SimpleOrganisms.

    Handles:
    - Adding/removing organisms
    - Population-wide updates
    - Basic statistics
    """

    def __init__(self, name: str = "Population", carrying_capacity: int = 100):
        self.name = name
        self.carrying_capacity = carrying_capacity
        self.organisms: Dict[str, SimpleOrganism] = {}
        self.total_births = 0
        self.total_deaths = 0

    def add(self, organism: SimpleOrganism) -> None:
        """Add an organism"""
        self.organisms[organism.id] = organism
        self.total_births += 1

        bus = get_event_bus()
        bus.emit("population.add", {
            "population": self.name,
            "organism_id": organism.id,
            "organism_name": organism.name
        })

    def remove(self, organism_id: str) -> Optional[SimpleOrganism]:
        """Remove an organism"""
        organism = self.organisms.pop(organism_id, None)
        if organism:
            self.total_deaths += 1

            bus = get_event_bus()
            bus.emit("population.remove", {
                "population": self.name,
                "organism_id": organism.id,
                "organism_name": organism.name
            })

        return organism

    def tick(self, stimuli: Optional[Dict[str, Any]] = None) -> None:
        """Update all organisms"""
        stimuli = stimuli or {}

        # Process each organism
        dead_ids = []
        for org in list(self.organisms.values()):
            org.perceive(stimuli)
            org.tick()
            org.decide()

            if not org.is_alive:
                dead_ids.append(org.id)

        # Remove dead organisms
        for org_id in dead_ids:
            self.remove(org_id)

    @property
    def size(self) -> int:
        return len(self.organisms)

    @property
    def alive_count(self) -> int:
        return sum(1 for o in self.organisms.values() if o.is_alive)

    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics"""
        if not self.organisms:
            return {
                "size": 0,
                "alive": 0,
                "avg_age": 0,
                "avg_energy": 0,
                "avg_health": 0
            }

        ages = [o.age for o in self.organisms.values()]
        energies = []
        healths = []

        for o in self.organisms.values():
            energy = o.get_subsystem("energy")
            health = o.get_subsystem("health")
            if energy:
                energies.append(energy.percentage)
            if health:
                healths.append(health.percentage)

        return {
            "size": self.size,
            "alive": self.alive_count,
            "avg_age": sum(ages) / len(ages) if ages else 0,
            "avg_energy": sum(energies) / len(energies) if energies else 0,
            "avg_health": sum(healths) / len(healths) if healths else 0,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths
        }
