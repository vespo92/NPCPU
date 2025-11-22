"""
Motor Action System

Enables digital organisms to perform actions and behaviors.
Implements biological-inspired motor control including:
- Action planning and execution
- Behavior patterns
- Motor learning
- Action coordination
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class ActionType(Enum):
    """Types of actions"""
    MOVE = "move"                     # Change position/state
    CONSUME = "consume"               # Consume resources
    PRODUCE = "produce"               # Create outputs
    COMMUNICATE = "communicate"       # Send signals
    MANIPULATE = "manipulate"         # Modify environment
    ACQUIRE = "acquire"               # Gather resources
    DEFEND = "defend"                 # Defensive actions
    REPRODUCE = "reproduce"           # Reproduction
    REPAIR = "repair"                 # Self-repair
    LEARN = "learn"                   # Learning actions
    EXPLORE = "explore"               # Exploration
    REST = "rest"                     # Recovery/rest


class ActionPriority(Enum):
    """Action priorities"""
    CRITICAL = 0     # Survival-critical
    URGENT = 1       # Needs immediate attention
    HIGH = 2         # Important
    NORMAL = 3       # Standard priority
    LOW = 4          # Can be delayed
    BACKGROUND = 5   # Do when idle


class ActionState(Enum):
    """States of an action"""
    PENDING = "pending"
    PREPARING = "preparing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BehaviorMode(Enum):
    """Behavioral modes"""
    EXPLORATION = "exploration"       # Seeking new experiences
    EXPLOITATION = "exploitation"     # Utilizing known resources
    DEFENSIVE = "defensive"           # Self-protection
    SOCIAL = "social"                 # Interaction-focused
    GROWTH = "growth"                 # Development-focused
    CONSERVATION = "conservation"     # Energy-saving


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Action:
    """A single action to be performed"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ActionType = ActionType.MOVE
    priority: ActionPriority = ActionPriority.NORMAL
    state: ActionState = ActionState.PENDING
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    energy_cost: float = 1.0
    duration: float = 1.0             # Expected duration
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    success: Optional[bool] = None


@dataclass
class ActionResult:
    """Result of an action execution"""
    action_id: str
    success: bool
    output: Any = None
    side_effects: List[str] = field(default_factory=list)
    energy_consumed: float = 0.0
    duration: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class BehaviorPattern:
    """A pattern of related actions"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    actions: List[ActionType] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: ActionPriority = ActionPriority.NORMAL
    success_rate: float = 1.0
    execution_count: int = 0


@dataclass
class MotorSkill:
    """A learned motor skill"""
    name: str
    proficiency: float = 0.0         # 0-1 skill level
    practice_count: int = 0
    last_used: Optional[float] = None
    associated_actions: List[ActionType] = field(default_factory=list)


# ============================================================================
# Motor Controller
# ============================================================================

class MotorController:
    """
    Low-level motor control and coordination.

    Handles:
    - Action preparation
    - Execution timing
    - Motor learning
    - Skill development
    """

    def __init__(
        self,
        base_speed: float = 1.0,
        precision: float = 0.9
    ):
        self.base_speed = base_speed
        self.precision = precision

        # Skills
        self.skills: Dict[str, MotorSkill] = {}

        # Coordination
        self.coordination: float = 1.0
        self.fatigue: float = 0.0

        # History
        self.execution_history: deque = deque(maxlen=100)

    def prepare_action(self, action: Action) -> Tuple[float, float]:
        """
        Prepare an action for execution.

        Returns (preparation_time, estimated_duration).
        """
        # Base preparation time
        prep_time = 0.1

        # Adjust for skill level
        skill = self._get_skill_for_action(action.type)
        if skill:
            prep_time *= (1 - skill.proficiency * 0.5)

        # Adjust for fatigue
        prep_time *= (1 + self.fatigue * 0.5)

        # Estimate duration
        duration = action.duration / self.base_speed
        duration *= (1 - self.precision * 0.2)  # Better precision = faster

        if skill:
            duration *= (1 - skill.proficiency * 0.3)  # Skill reduces time

        duration *= (1 + self.fatigue * 0.3)  # Fatigue increases time

        return prep_time, duration

    def execute_action(self, action: Action) -> bool:
        """
        Execute an action.

        Returns success probability outcome.
        """
        # Base success rate
        success_rate = self.precision

        # Skill modifier
        skill = self._get_skill_for_action(action.type)
        if skill:
            success_rate = min(1.0, success_rate + skill.proficiency * 0.2)
            skill.practice_count += 1
            skill.last_used = time.time()
            self._improve_skill(skill)

        # Fatigue penalty
        success_rate *= (1 - self.fatigue * 0.3)

        # Random outcome
        success = np.random.random() < success_rate

        # Update fatigue
        self.fatigue = min(1.0, self.fatigue + action.energy_cost * 0.01)

        # Record
        self.execution_history.append({
            "action_type": action.type.value,
            "success": success,
            "time": time.time()
        })

        return success

    def _get_skill_for_action(self, action_type: ActionType) -> Optional[MotorSkill]:
        """Get relevant skill for an action type"""
        for skill in self.skills.values():
            if action_type in skill.associated_actions:
                return skill
        return None

    def _improve_skill(self, skill: MotorSkill):
        """Improve a skill through practice"""
        improvement = 0.01 * (1 - skill.proficiency)  # Diminishing returns
        skill.proficiency = min(1.0, skill.proficiency + improvement)

    def learn_skill(self, name: str, action_types: List[ActionType]):
        """Learn a new motor skill"""
        self.skills[name] = MotorSkill(
            name=name,
            associated_actions=action_types.copy()
        )

    def recover(self, amount: float = 0.1):
        """Recover from fatigue"""
        self.fatigue = max(0, self.fatigue - amount)

    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            "base_speed": self.base_speed,
            "precision": self.precision,
            "coordination": self.coordination,
            "fatigue": self.fatigue,
            "skills": {
                name: skill.proficiency
                for name, skill in self.skills.items()
            }
        }


# ============================================================================
# Motor System
# ============================================================================

class MotorSystem:
    """
    Complete motor system for digital organisms.

    Features:
    - Action queue management
    - Behavior patterns
    - Motor learning
    - Energy-aware execution
    - Action coordination

    Example:
        motor = MotorSystem()

        # Queue an action
        action = motor.queue_action(
            ActionType.ACQUIRE,
            target="resource_001",
            energy_cost=5.0
        )

        # Execute pending actions
        results = motor.execute_pending()

        # Learn behaviors
        motor.learn_behavior("foraging", [ActionType.EXPLORE, ActionType.ACQUIRE])
    """

    def __init__(
        self,
        max_energy: float = 100.0,
        max_queue_size: int = 50
    ):
        self.max_energy = max_energy
        self.max_queue_size = max_queue_size

        # Energy
        self.energy: float = max_energy

        # Action queue
        self.action_queue: List[Action] = []
        self.current_action: Optional[Action] = None

        # Motor controller
        self.controller = MotorController()

        # Behavior patterns
        self.behaviors: Dict[str, BehaviorPattern] = {}
        self._initialize_default_behaviors()

        # Current mode
        self.mode = BehaviorMode.EXPLOITATION

        # Statistics
        self.total_actions = 0
        self.successful_actions = 0
        self.failed_actions = 0
        self.total_energy_spent = 0.0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "action_started": [],
            "action_completed": [],
            "action_failed": [],
            "behavior_triggered": []
        }

        # Action handlers
        self._handlers: Dict[ActionType, Callable] = {}

    def _initialize_default_behaviors(self):
        """Initialize default behavior patterns"""
        defaults = [
            BehaviorPattern(
                name="foraging",
                description="Search for and acquire resources",
                actions=[ActionType.EXPLORE, ActionType.ACQUIRE],
                triggers=["low_energy", "resource_detected"],
                conditions={"min_energy": 10.0}
            ),
            BehaviorPattern(
                name="defensive",
                description="Respond to threats",
                actions=[ActionType.DEFEND, ActionType.MOVE],
                triggers=["threat_detected"],
                priority=ActionPriority.URGENT
            ),
            BehaviorPattern(
                name="social",
                description="Interact with other organisms",
                actions=[ActionType.COMMUNICATE, ActionType.MOVE],
                triggers=["organism_detected", "signal_received"],
                conditions={"min_energy": 20.0}
            ),
            BehaviorPattern(
                name="recovery",
                description="Rest and recover",
                actions=[ActionType.REST, ActionType.REPAIR],
                triggers=["high_fatigue", "damage_detected"],
                conditions={"max_energy": 30.0}
            ),
            BehaviorPattern(
                name="learning",
                description="Acquire new knowledge",
                actions=[ActionType.LEARN, ActionType.EXPLORE],
                triggers=["novelty_detected", "idle"],
                conditions={"min_energy": 40.0}
            )
        ]

        for behavior in defaults:
            self.behaviors[behavior.name] = behavior

    def queue_action(
        self,
        action_type: ActionType,
        target: Optional[str] = None,
        priority: ActionPriority = ActionPriority.NORMAL,
        energy_cost: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None,
        duration: float = 1.0
    ) -> Optional[Action]:
        """
        Queue an action for execution.

        Returns the queued Action or None if queue is full.
        """
        if len(self.action_queue) >= self.max_queue_size:
            return None

        # Default energy costs by type
        default_costs = {
            ActionType.MOVE: 2.0,
            ActionType.CONSUME: 1.0,
            ActionType.PRODUCE: 5.0,
            ActionType.COMMUNICATE: 1.5,
            ActionType.MANIPULATE: 4.0,
            ActionType.ACQUIRE: 3.0,
            ActionType.DEFEND: 6.0,
            ActionType.REPRODUCE: 20.0,
            ActionType.REPAIR: 8.0,
            ActionType.LEARN: 4.0,
            ActionType.EXPLORE: 3.0,
            ActionType.REST: 0.5
        }

        action = Action(
            type=action_type,
            target=target,
            priority=priority,
            energy_cost=energy_cost or default_costs.get(action_type, 2.0),
            parameters=parameters or {},
            duration=duration
        )

        self.action_queue.append(action)
        self._sort_queue()

        return action

    def _sort_queue(self):
        """Sort action queue by priority"""
        self.action_queue.sort(key=lambda a: (a.priority.value, a.created_at))

    def cancel_action(self, action_id: str) -> bool:
        """Cancel a queued action"""
        for action in self.action_queue:
            if action.id == action_id:
                action.state = ActionState.CANCELLED
                self.action_queue.remove(action)
                return True
        return False

    def execute_next(self) -> Optional[ActionResult]:
        """Execute the next action in the queue"""
        if not self.action_queue:
            return None

        action = self.action_queue.pop(0)

        # Check energy
        if self.energy < action.energy_cost:
            action.state = ActionState.FAILED
            action.success = False
            self.failed_actions += 1
            return ActionResult(
                action_id=action.id,
                success=False,
                error="Insufficient energy"
            )

        # Prepare action
        action.state = ActionState.PREPARING
        prep_time, duration = self.controller.prepare_action(action)

        # Execute
        action.state = ActionState.EXECUTING
        action.started_at = time.time()

        for callback in self._callbacks["action_started"]:
            callback(action)

        # Run handler if registered
        output = None
        if action.type in self._handlers:
            try:
                output = self._handlers[action.type](action)
            except Exception as e:
                action.state = ActionState.FAILED
                action.success = False
                self.failed_actions += 1
                return ActionResult(
                    action_id=action.id,
                    success=False,
                    error=str(e)
                )

        # Determine success
        success = self.controller.execute_action(action)

        # Consume energy
        actual_energy = action.energy_cost * (0.8 if success else 1.2)
        self.energy -= actual_energy
        self.total_energy_spent += actual_energy

        # Complete action
        action.completed_at = time.time()
        action.state = ActionState.COMPLETED if success else ActionState.FAILED
        action.success = success
        action.result = output

        self.total_actions += 1
        if success:
            self.successful_actions += 1
            for callback in self._callbacks["action_completed"]:
                callback(action)
        else:
            self.failed_actions += 1
            for callback in self._callbacks["action_failed"]:
                callback(action)

        return ActionResult(
            action_id=action.id,
            success=success,
            output=output,
            energy_consumed=actual_energy,
            duration=action.completed_at - action.started_at
        )

    def execute_pending(self, max_actions: int = 5) -> List[ActionResult]:
        """Execute multiple pending actions"""
        results = []
        for _ in range(max_actions):
            if not self.action_queue:
                break
            result = self.execute_next()
            if result:
                results.append(result)
        return results

    def trigger_behavior(self, trigger: str) -> List[Action]:
        """
        Trigger behavior patterns matching a trigger.

        Returns list of queued actions.
        """
        triggered_actions = []

        for behavior in self.behaviors.values():
            if trigger not in behavior.triggers:
                continue

            # Check conditions
            if not self._check_behavior_conditions(behavior):
                continue

            # Queue behavior actions
            for action_type in behavior.actions:
                action = self.queue_action(
                    action_type=action_type,
                    priority=behavior.priority,
                    parameters={"behavior": behavior.name}
                )
                if action:
                    triggered_actions.append(action)

            behavior.execution_count += 1

            for callback in self._callbacks["behavior_triggered"]:
                callback(behavior)

        return triggered_actions

    def _check_behavior_conditions(self, behavior: BehaviorPattern) -> bool:
        """Check if behavior conditions are met"""
        conditions = behavior.conditions

        if "min_energy" in conditions and self.energy < conditions["min_energy"]:
            return False

        if "max_energy" in conditions and self.energy > conditions["max_energy"]:
            return False

        return True

    def learn_behavior(
        self,
        name: str,
        actions: List[ActionType],
        triggers: Optional[List[str]] = None
    ):
        """Learn a new behavior pattern"""
        self.behaviors[name] = BehaviorPattern(
            name=name,
            actions=actions,
            triggers=triggers or []
        )

    def set_mode(self, mode: BehaviorMode):
        """Set behavioral mode"""
        self.mode = mode

        # Adjust controller based on mode
        if mode == BehaviorMode.EXPLORATION:
            self.controller.base_speed = 1.2
        elif mode == BehaviorMode.CONSERVATION:
            self.controller.base_speed = 0.7
        elif mode == BehaviorMode.DEFENSIVE:
            self.controller.precision = 0.95
        else:
            self.controller.base_speed = 1.0

    def register_handler(
        self,
        action_type: ActionType,
        handler: Callable[[Action], Any]
    ):
        """Register a handler for an action type"""
        self._handlers[action_type] = handler

    def add_energy(self, amount: float):
        """Add energy to the system"""
        self.energy = min(self.max_energy, self.energy + amount)

    def tick(self):
        """Process one motor cycle"""
        # Natural recovery
        self.controller.recover(0.01)

        # Execute pending actions
        if self.action_queue:
            self.execute_next()

    def on_action_completed(self, callback: Callable[[Action], None]):
        """Register callback for action completion"""
        self._callbacks["action_completed"].append(callback)

    def on_action_failed(self, callback: Callable[[Action], None]):
        """Register callback for action failure"""
        self._callbacks["action_failed"].append(callback)

    def get_queue_status(self) -> List[Dict[str, Any]]:
        """Get status of action queue"""
        return [
            {
                "id": a.id,
                "type": a.type.value,
                "priority": a.priority.name,
                "state": a.state.value,
                "energy_cost": a.energy_cost
            }
            for a in self.action_queue
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get motor system status"""
        return {
            "energy": self.energy,
            "max_energy": self.max_energy,
            "mode": self.mode.value,
            "queue_size": len(self.action_queue),
            "max_queue_size": self.max_queue_size,
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success_rate": self.successful_actions / max(1, self.total_actions),
            "total_energy_spent": self.total_energy_spent,
            "behaviors": len(self.behaviors),
            "controller": self.controller.get_status()
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Motor Action System Demo")
    print("=" * 50)

    # Create motor system
    motor = MotorSystem(max_energy=100.0)

    print(f"\n1. Initial state:")
    status = motor.get_status()
    print(f"   Energy: {status['energy']}/{status['max_energy']}")
    print(f"   Mode: {status['mode']}")
    print(f"   Behaviors: {status['behaviors']}")

    # Learn a skill
    print("\n2. Learning motor skills...")
    motor.controller.learn_skill(
        "resource_gathering",
        [ActionType.EXPLORE, ActionType.ACQUIRE]
    )
    print(f"   Learned: resource_gathering")

    # Queue actions
    print("\n3. Queueing actions...")
    actions = [
        (ActionType.EXPLORE, "zone_1", ActionPriority.NORMAL),
        (ActionType.ACQUIRE, "resource_a", ActionPriority.HIGH),
        (ActionType.COMMUNICATE, "peer_001", ActionPriority.LOW),
        (ActionType.LEARN, None, ActionPriority.NORMAL),
        (ActionType.REST, None, ActionPriority.LOW)
    ]

    for action_type, target, priority in actions:
        action = motor.queue_action(action_type, target=target, priority=priority)
        print(f"   Queued: {action_type.value} -> {target}")

    # Execute actions
    print("\n4. Executing actions...")
    results = motor.execute_pending(max_actions=5)

    for result in results:
        status_str = "SUCCESS" if result.success else "FAILED"
        print(f"   {status_str}: energy={result.energy_consumed:.1f}, "
              f"duration={result.duration:.3f}s")

    # Trigger behavior
    print("\n5. Triggering behavior...")
    triggered = motor.trigger_behavior("threat_detected")
    print(f"   Triggered {len(triggered)} actions from 'defensive' behavior")

    # Execute triggered actions
    results = motor.execute_pending()
    for result in results:
        print(f"   Executed: {'SUCCESS' if result.success else 'FAILED'}")

    # Check skill improvement
    print("\n6. Skill progress:")
    controller_status = motor.controller.get_status()
    for skill_name, proficiency in controller_status["skills"].items():
        print(f"   {skill_name}: {proficiency:.2%}")

    # Final status
    print("\n7. Final status:")
    status = motor.get_status()
    print(f"   Energy: {status['energy']:.1f}/{status['max_energy']}")
    print(f"   Total actions: {status['total_actions']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Fatigue: {status['controller']['fatigue']:.2%}")
