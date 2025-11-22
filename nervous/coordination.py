"""
Neural Coordination System

Central nervous system equivalent for digital organisms.
Coordinates all internal systems and handles signal processing.

Features:
- Signal routing and processing
- Reflex responses
- Inter-system coordination
- Central command processing
"""

import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Callable, Set
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

class SignalType(Enum):
    """Types of neural signals"""
    SENSORY = "sensory"           # Input from environment
    MOTOR = "motor"               # Output commands
    INTERNAL = "internal"         # Internal communication
    REFLEX = "reflex"             # Automatic responses
    PAIN = "pain"                 # Damage/warning signals
    REWARD = "reward"             # Positive feedback
    ATTENTION = "attention"       # Focus direction
    MEMORY = "memory"             # Memory access/storage


class SystemType(Enum):
    """Types of organism systems"""
    CONSCIOUSNESS = "consciousness"
    METABOLISM = "metabolism"
    HOMEOSTASIS = "homeostasis"
    IMMUNE = "immune"
    LIFECYCLE = "lifecycle"
    MEMORY = "memory"
    MOTOR = "motor"
    SENSORY = "sensory"


class ReflexTrigger(Enum):
    """Conditions that trigger reflexes"""
    PAIN = "pain"
    THREAT = "threat"
    OPPORTUNITY = "opportunity"
    THRESHOLD = "threshold"
    PATTERN = "pattern"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Signal:
    """A neural signal"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: SignalType = SignalType.INTERNAL
    source: str = ""
    target: str = ""
    payload: Any = None
    priority: int = 1  # 1-10
    timestamp: float = field(default_factory=time.time)
    processed: bool = False
    response: Any = None


@dataclass
class NeuralPathway:
    """Connection between systems"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_system: SystemType = SystemType.CONSCIOUSNESS
    target_system: SystemType = SystemType.MOTOR
    signal_types: List[SignalType] = field(default_factory=list)
    strength: float = 1.0  # Connection strength
    latency: float = 0.01  # Processing delay
    active: bool = True


@dataclass
class Reflex:
    """Automatic reflex response"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger: ReflexTrigger = ReflexTrigger.THRESHOLD
    trigger_condition: Dict[str, Any] = field(default_factory=dict)
    response_action: str = ""
    response_params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    cooldown: float = 1.0  # Seconds between activations
    last_activated: float = 0.0


@dataclass
class CoordinationCommand:
    """Command from central coordinator"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command_type: str = ""
    targets: List[SystemType] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    issued_at: float = field(default_factory=time.time)


# ============================================================================
# Central Coordinator
# ============================================================================

class CentralCoordinator:
    """
    Central command center for organism coordination.

    Like the brain's executive functions, coordinates all systems
    and makes high-level decisions.
    """

    def __init__(self):
        self.attention_focus: Optional[str] = None
        self.current_goal: Optional[str] = None
        self.command_history: List[CoordinationCommand] = []

    def set_attention(self, focus: str):
        """Set attention focus"""
        self.attention_focus = focus

    def set_goal(self, goal: str):
        """Set current goal"""
        self.current_goal = goal

    def issue_command(
        self,
        command_type: str,
        targets: List[SystemType],
        parameters: Optional[Dict[str, Any]] = None
    ) -> CoordinationCommand:
        """Issue a coordination command"""
        command = CoordinationCommand(
            command_type=command_type,
            targets=targets,
            parameters=parameters or {}
        )
        self.command_history.append(command)
        return command

    def prioritize_systems(
        self,
        system_states: Dict[SystemType, Dict[str, Any]]
    ) -> List[SystemType]:
        """Prioritize systems based on current states"""
        priorities = []

        for system, state in system_states.items():
            priority = 0

            # Critical states get highest priority
            if state.get("critical", False):
                priority = 10
            elif state.get("stressed", False):
                priority = 7
            elif state.get("needs_attention", False):
                priority = 5
            else:
                priority = 3

            priorities.append((system, priority))

        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)

        return [p[0] for p in priorities]


# ============================================================================
# Nervous System
# ============================================================================

class NervousSystem:
    """
    Central nervous system for digital organisms.

    Handles all coordination, signal processing, and reflex responses.

    Example:
        nervous = NervousSystem()

        # Add pathways
        nervous.add_pathway(
            "perception_to_consciousness",
            SystemType.SENSORY,
            SystemType.CONSCIOUSNESS
        )

        # Add reflexes
        nervous.add_reflex(
            "pain_response",
            ReflexTrigger.PAIN,
            {"threshold": 0.7},
            "withdraw"
        )

        # Send signal
        signal = nervous.send_signal(
            SignalType.SENSORY,
            "environment",
            "consciousness",
            sensor_data
        )

        # Process signals
        nervous.tick()
    """

    def __init__(self):
        # Coordinator
        self.coordinator = CentralCoordinator()

        # Neural pathways
        self.pathways: Dict[str, NeuralPathway] = {}

        # Reflexes
        self.reflexes: Dict[str, Reflex] = {}

        # Signal queue
        self.signal_queue: deque = deque(maxlen=1000)
        self.processed_signals: List[Signal] = []

        # System handlers
        self.system_handlers: Dict[SystemType, Callable] = {}

        # State
        self.cycle_count = 0
        self.signals_processed = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "signal_received": [],
            "signal_processed": [],
            "reflex_triggered": []
        }

        # Initialize default pathways
        self._initialize_pathways()
        self._initialize_reflexes()

    def _initialize_pathways(self):
        """Initialize default neural pathways"""
        default_pathways = [
            ("sensory_to_consciousness", SystemType.SENSORY, SystemType.CONSCIOUSNESS,
             [SignalType.SENSORY, SignalType.ATTENTION]),
            ("consciousness_to_motor", SystemType.CONSCIOUSNESS, SystemType.MOTOR,
             [SignalType.MOTOR]),
            ("immune_to_consciousness", SystemType.IMMUNE, SystemType.CONSCIOUSNESS,
             [SignalType.PAIN, SignalType.INTERNAL]),
            ("metabolism_to_homeostasis", SystemType.METABOLISM, SystemType.HOMEOSTASIS,
             [SignalType.INTERNAL]),
            ("homeostasis_to_metabolism", SystemType.HOMEOSTASIS, SystemType.METABOLISM,
             [SignalType.INTERNAL]),
            ("lifecycle_to_all", SystemType.LIFECYCLE, SystemType.CONSCIOUSNESS,
             [SignalType.INTERNAL])
        ]

        for name, source, target, signals in default_pathways:
            self.add_pathway(name, source, target, signals)

    def _initialize_reflexes(self):
        """Initialize default reflexes"""
        default_reflexes = [
            ("pain_withdraw", ReflexTrigger.PAIN, {"threshold": 0.5},
             "reduce_exposure", {"amount": 0.5}),
            ("threat_alert", ReflexTrigger.THREAT, {"threshold": 0.3},
             "heighten_awareness", {"duration": 10}),
            ("low_energy_conserve", ReflexTrigger.THRESHOLD, {"metric": "energy", "below": 0.2},
             "enter_conservation", {}),
            ("high_load_shed", ReflexTrigger.THRESHOLD, {"metric": "load", "above": 0.9},
             "shed_load", {"percentage": 0.3})
        ]

        for name, trigger, condition, action, params in default_reflexes:
            self.add_reflex(name, trigger, condition, action, params)

    def add_pathway(
        self,
        name: str,
        source: SystemType,
        target: SystemType,
        signal_types: Optional[List[SignalType]] = None
    ) -> NeuralPathway:
        """Add a neural pathway"""
        pathway = NeuralPathway(
            name=name,
            source_system=source,
            target_system=target,
            signal_types=signal_types or [SignalType.INTERNAL]
        )
        self.pathways[pathway.id] = pathway
        return pathway

    def add_reflex(
        self,
        name: str,
        trigger: ReflexTrigger,
        condition: Dict[str, Any],
        action: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Reflex:
        """Add a reflex response"""
        reflex = Reflex(
            name=name,
            trigger=trigger,
            trigger_condition=condition,
            response_action=action,
            response_params=params or {}
        )
        self.reflexes[reflex.id] = reflex
        return reflex

    def register_handler(
        self,
        system: SystemType,
        handler: Callable[[Signal], Any]
    ):
        """Register a handler for a system"""
        self.system_handlers[system] = handler

    def send_signal(
        self,
        signal_type: SignalType,
        source: str,
        target: str,
        payload: Any,
        priority: int = 5
    ) -> Signal:
        """Send a signal through the nervous system"""
        signal = Signal(
            type=signal_type,
            source=source,
            target=target,
            payload=payload,
            priority=priority
        )

        self.signal_queue.append(signal)

        # Trigger callbacks
        for callback in self._callbacks["signal_received"]:
            callback(signal)

        return signal

    def tick(self) -> List[Signal]:
        """
        Process signals for one cycle.

        Returns list of processed signals.
        """
        self.cycle_count += 1
        processed = []

        # Sort queue by priority
        sorted_signals = sorted(
            list(self.signal_queue),
            key=lambda s: s.priority,
            reverse=True
        )
        self.signal_queue.clear()

        for signal in sorted_signals:
            # Check for reflex triggers
            reflex_response = self._check_reflexes(signal)
            if reflex_response:
                signal.response = reflex_response

            # Route signal
            response = self._route_signal(signal)
            signal.response = response
            signal.processed = True

            processed.append(signal)
            self.processed_signals.append(signal)
            self.signals_processed += 1

            # Trigger callbacks
            for callback in self._callbacks["signal_processed"]:
                callback(signal)

        return processed

    def _check_reflexes(self, signal: Signal) -> Optional[Dict[str, Any]]:
        """Check if signal triggers any reflexes"""
        current_time = time.time()

        for reflex in self.reflexes.values():
            # Check cooldown
            if current_time - reflex.last_activated < reflex.cooldown:
                continue

            # Check trigger
            triggered = False

            if reflex.trigger == ReflexTrigger.PAIN:
                if signal.type == SignalType.PAIN:
                    threshold = reflex.trigger_condition.get("threshold", 0.5)
                    if isinstance(signal.payload, (int, float)):
                        triggered = signal.payload > threshold

            elif reflex.trigger == ReflexTrigger.THREAT:
                if signal.type in [SignalType.PAIN, SignalType.SENSORY]:
                    threshold = reflex.trigger_condition.get("threshold", 0.5)
                    if isinstance(signal.payload, dict):
                        triggered = signal.payload.get("threat_level", 0) > threshold

            elif reflex.trigger == ReflexTrigger.THRESHOLD:
                metric = reflex.trigger_condition.get("metric")
                if isinstance(signal.payload, dict) and metric in signal.payload:
                    value = signal.payload[metric]
                    if "below" in reflex.trigger_condition:
                        triggered = value < reflex.trigger_condition["below"]
                    elif "above" in reflex.trigger_condition:
                        triggered = value > reflex.trigger_condition["above"]

            if triggered:
                reflex.last_activated = current_time

                # Trigger callback
                for callback in self._callbacks["reflex_triggered"]:
                    callback(reflex, signal)

                return {
                    "reflex": reflex.name,
                    "action": reflex.response_action,
                    "params": reflex.response_params
                }

        return None

    def _route_signal(self, signal: Signal) -> Optional[Any]:
        """Route signal to appropriate handler"""
        # Find matching pathways
        for pathway in self.pathways.values():
            if not pathway.active:
                continue

            if signal.type in pathway.signal_types:
                # Check if we have a handler for target
                if pathway.target_system in self.system_handlers:
                    handler = self.system_handlers[pathway.target_system]
                    return handler(signal)

        return None

    def broadcast(
        self,
        signal_type: SignalType,
        payload: Any,
        priority: int = 5
    ) -> List[Signal]:
        """Broadcast signal to all systems"""
        signals = []

        for pathway in self.pathways.values():
            if signal_type in pathway.signal_types:
                signal = self.send_signal(
                    signal_type,
                    "broadcast",
                    pathway.target_system.value,
                    payload,
                    priority
                )
                signals.append(signal)

        return signals

    def get_pathway_stats(self) -> Dict[str, Any]:
        """Get pathway statistics"""
        active = len([p for p in self.pathways.values() if p.active])
        return {
            "total_pathways": len(self.pathways),
            "active_pathways": active,
            "pathway_types": {
                st.value: len([p for p in self.pathways.values() if st in p.signal_types])
                for st in SignalType
            }
        }

    def on_signal_received(self, callback: Callable[[Signal], None]):
        """Register callback for signal reception"""
        self._callbacks["signal_received"].append(callback)

    def on_reflex_triggered(self, callback: Callable[[Reflex, Signal], None]):
        """Register callback for reflex triggers"""
        self._callbacks["reflex_triggered"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get nervous system status"""
        return {
            "cycle_count": self.cycle_count,
            "signals_processed": self.signals_processed,
            "queue_size": len(self.signal_queue),
            "pathways": len(self.pathways),
            "active_pathways": len([p for p in self.pathways.values() if p.active]),
            "reflexes": len(self.reflexes),
            "attention_focus": self.coordinator.attention_focus,
            "current_goal": self.coordinator.current_goal
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Nervous System Demo")
    print("=" * 50)

    # Create nervous system
    nervous = NervousSystem()

    # Track signals and reflexes
    signals_received = []
    reflexes_triggered = []

    nervous.on_signal_received(lambda s: signals_received.append(s))
    nervous.on_reflex_triggered(lambda r, s: reflexes_triggered.append((r.name, s.type)))

    print(f"\n1. Initial state:")
    status = nervous.get_status()
    print(f"   Pathways: {status['pathways']}")
    print(f"   Reflexes: {status['reflexes']}")

    # Register a handler
    def consciousness_handler(signal):
        return {"received": True, "processed_at": time.time()}

    nervous.register_handler(SystemType.CONSCIOUSNESS, consciousness_handler)

    # Send signals
    print("\n2. Sending signals...")

    # Normal sensory signal
    nervous.send_signal(
        SignalType.SENSORY,
        "environment",
        "consciousness",
        {"visual": [0.5, 0.3, 0.8]},
        priority=3
    )

    # Pain signal (should trigger reflex)
    nervous.send_signal(
        SignalType.PAIN,
        "immune",
        "consciousness",
        0.7,  # Above threshold
        priority=8
    )

    # Internal communication
    nervous.send_signal(
        SignalType.INTERNAL,
        "metabolism",
        "homeostasis",
        {"energy": 0.15},  # Below threshold
        priority=5
    )

    # Process signals
    print("\n3. Processing signals...")
    processed = nervous.tick()
    print(f"   Processed: {len(processed)} signals")

    for signal in processed:
        print(f"   - {signal.type.value}: {signal.source} -> {signal.target}")
        if signal.response:
            print(f"     Response: {signal.response}")

    # Check reflexes
    print(f"\n4. Reflexes triggered: {len(reflexes_triggered)}")
    for reflex_name, signal_type in reflexes_triggered:
        print(f"   - {reflex_name} (from {signal_type.value})")

    # Broadcast
    print("\n5. Broadcasting signal...")
    broadcast_signals = nervous.broadcast(
        SignalType.ATTENTION,
        {"focus": "threat", "priority": "high"},
        priority=7
    )
    print(f"   Broadcast to {len(broadcast_signals)} targets")

    nervous.tick()

    # Set coordination
    print("\n6. Central coordination...")
    nervous.coordinator.set_goal("survive")
    nervous.coordinator.set_attention("environment")

    command = nervous.coordinator.issue_command(
        "heighten_awareness",
        [SystemType.SENSORY, SystemType.CONSCIOUSNESS],
        {"duration": 30}
    )
    print(f"   Goal: {nervous.coordinator.current_goal}")
    print(f"   Attention: {nervous.coordinator.attention_focus}")
    print(f"   Command issued: {command.command_type}")

    # Final status
    print("\n7. Final status:")
    status = nervous.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
