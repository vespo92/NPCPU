"""
Environment Simulation

Simulates the environment in which digital organisms exist.
Provides stimuli, resources, and environmental conditions.
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class StimulusType(Enum):
    """Types of environmental stimuli"""
    RESOURCE = "resource"           # Available resources
    THREAT = "threat"               # Dangers/threats
    OPPORTUNITY = "opportunity"     # Beneficial situations
    SIGNAL = "signal"               # Communication signals
    CHANGE = "change"               # Environmental changes
    NOISE = "noise"                 # Background noise
    INTERACTION = "interaction"     # Organism interactions


class EnvironmentCondition(Enum):
    """Environmental conditions"""
    OPTIMAL = "optimal"             # Perfect conditions
    FAVORABLE = "favorable"         # Good conditions
    NEUTRAL = "neutral"             # Normal conditions
    HARSH = "harsh"                 # Challenging conditions
    HOSTILE = "hostile"             # Dangerous conditions
    CATASTROPHIC = "catastrophic"   # Survival-threatening


class ResourceAvailability(Enum):
    """Resource availability levels"""
    ABUNDANT = "abundant"
    PLENTIFUL = "plentiful"
    SUFFICIENT = "sufficient"
    SCARCE = "scarce"
    DEPLETED = "depleted"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Stimulus:
    """An environmental stimulus"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: StimulusType = StimulusType.SIGNAL
    intensity: float = 0.5
    location: Optional[str] = None
    data: Any = None
    duration: float = 1.0           # How long it lasts
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    source: str = ""

    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.created_at + self.duration

    @property
    def is_active(self) -> bool:
        return time.time() < self.expires_at


@dataclass
class EnvironmentEvent:
    """An event occurring in the environment"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    event_type: str = "generic"
    magnitude: float = 0.5
    affected_area: str = "global"
    stimuli: List[Stimulus] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentZone:
    """A zone within the environment"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    condition: EnvironmentCondition = EnvironmentCondition.NEUTRAL
    resources: Dict[str, float] = field(default_factory=dict)
    hazards: List[str] = field(default_factory=list)
    inhabitants: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    """Current state of the environment"""
    condition: EnvironmentCondition = EnvironmentCondition.NEUTRAL
    resource_availability: ResourceAvailability = ResourceAvailability.SUFFICIENT
    temperature: float = 0.5        # Normalized 0-1 (0.5 = optimal)
    pressure: float = 0.5           # System load/pressure
    entropy: float = 0.3            # Disorder level
    stability: float = 0.8          # How stable the environment is
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Environment
# ============================================================================

class Environment:
    """
    Simulated environment for digital organisms.

    Features:
    - Multiple zones with different conditions
    - Stimulus generation
    - Resource management
    - Event system
    - Environmental dynamics

    Example:
        env = Environment()

        # Add organism to environment
        env.register_organism("org_001")

        # Generate stimuli
        stimuli = env.generate_stimuli()

        # Process environment
        env.tick()
    """

    def __init__(
        self,
        base_resource_rate: float = 1.0,
        volatility: float = 0.2
    ):
        self.base_resource_rate = base_resource_rate
        self.volatility = volatility

        # Environment state
        self.state = EnvironmentState()

        # Zones
        self.zones: Dict[str, EnvironmentZone] = {}
        self._initialize_default_zones()

        # Active stimuli
        self.active_stimuli: Dict[str, Stimulus] = {}

        # Events
        self.events: List[EnvironmentEvent] = []
        self.event_history: List[EnvironmentEvent] = []

        # Organisms in the environment
        self.organisms: Set[str] = set()

        # Resources
        self.global_resources: Dict[str, float] = {
            "compute": 1000.0,
            "memory": 1000.0,
            "bandwidth": 500.0,
            "energy": 2000.0,
            "data": 5000.0
        }

        # Statistics
        self.cycle_count = 0
        self.total_stimuli_generated = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "stimulus_created": [],
            "event_occurred": [],
            "condition_changed": []
        }

    def _initialize_default_zones(self):
        """Initialize default environment zones"""
        zones = [
            EnvironmentZone(
                name="core",
                condition=EnvironmentCondition.OPTIMAL,
                resources={"compute": 500, "memory": 500, "energy": 1000},
                properties={"protected": True, "stability": 0.95}
            ),
            EnvironmentZone(
                name="periphery",
                condition=EnvironmentCondition.NEUTRAL,
                resources={"compute": 300, "memory": 300, "bandwidth": 400},
                properties={"stability": 0.7}
            ),
            EnvironmentZone(
                name="frontier",
                condition=EnvironmentCondition.HARSH,
                resources={"data": 2000, "bandwidth": 200},
                hazards=["instability", "data_corruption"],
                properties={"stability": 0.4, "opportunity": 0.8}
            ),
            EnvironmentZone(
                name="commons",
                condition=EnvironmentCondition.FAVORABLE,
                resources={"compute": 200, "memory": 200, "energy": 500},
                properties={"social": True, "stability": 0.8}
            )
        ]

        for zone in zones:
            self.zones[zone.name] = zone

    def register_organism(self, organism_id: str, zone: str = "core"):
        """Register an organism in the environment"""
        self.organisms.add(organism_id)
        if zone in self.zones:
            self.zones[zone].inhabitants.add(organism_id)

    def unregister_organism(self, organism_id: str):
        """Remove an organism from the environment"""
        self.organisms.discard(organism_id)
        for zone in self.zones.values():
            zone.inhabitants.discard(organism_id)

    def create_stimulus(
        self,
        stimulus_type: StimulusType,
        intensity: float,
        data: Any = None,
        location: Optional[str] = None,
        duration: float = 1.0,
        source: str = ""
    ) -> Stimulus:
        """Create a new stimulus in the environment"""
        stimulus = Stimulus(
            type=stimulus_type,
            intensity=intensity,
            location=location,
            data=data,
            duration=duration,
            source=source
        )

        self.active_stimuli[stimulus.id] = stimulus
        self.total_stimuli_generated += 1

        # Trigger callbacks
        for callback in self._callbacks["stimulus_created"]:
            callback(stimulus)

        return stimulus

    def generate_stimuli(self) -> List[Stimulus]:
        """Generate environmental stimuli based on current conditions"""
        stimuli = []

        # Resource availability stimuli
        if np.random.random() < 0.3:
            resource_type = np.random.choice(list(self.global_resources.keys()))
            amount = self.global_resources[resource_type] * np.random.random() * 0.1

            stimulus = self.create_stimulus(
                stimulus_type=StimulusType.RESOURCE,
                intensity=min(1.0, amount / 100),
                data={"resource": resource_type, "amount": amount},
                location=np.random.choice(list(self.zones.keys())),
                source="environment"
            )
            stimuli.append(stimulus)

        # Threat stimuli based on conditions
        if self.state.condition in [EnvironmentCondition.HARSH, EnvironmentCondition.HOSTILE]:
            if np.random.random() < 0.2:
                stimulus = self.create_stimulus(
                    stimulus_type=StimulusType.THREAT,
                    intensity=0.3 + np.random.random() * 0.5,
                    data={"threat_type": "environmental_hazard"},
                    source="environment"
                )
                stimuli.append(stimulus)

        # Opportunity stimuli
        if np.random.random() < 0.15:
            stimulus = self.create_stimulus(
                stimulus_type=StimulusType.OPPORTUNITY,
                intensity=np.random.random() * 0.7,
                data={"opportunity_type": np.random.choice([
                    "resource_cache", "knowledge_source", "alliance"
                ])},
                location=np.random.choice(list(self.zones.keys())),
                source="environment"
            )
            stimuli.append(stimulus)

        # Background noise
        if np.random.random() < 0.4:
            stimulus = self.create_stimulus(
                stimulus_type=StimulusType.NOISE,
                intensity=np.random.random() * 0.3,
                duration=0.5,
                source="background"
            )
            stimuli.append(stimulus)

        return stimuli

    def trigger_event(
        self,
        name: str,
        event_type: str,
        magnitude: float,
        affected_area: str = "global"
    ) -> EnvironmentEvent:
        """Trigger an environmental event"""
        # Generate associated stimuli
        stimuli = []
        stimulus_count = max(1, int(magnitude * 5))

        for _ in range(stimulus_count):
            stimulus = self.create_stimulus(
                stimulus_type=StimulusType.CHANGE,
                intensity=magnitude * np.random.uniform(0.5, 1.0),
                data={"event": name, "type": event_type},
                location=affected_area if affected_area != "global" else None,
                duration=2.0 + magnitude * 3,
                source=f"event:{name}"
            )
            stimuli.append(stimulus)

        event = EnvironmentEvent(
            name=name,
            event_type=event_type,
            magnitude=magnitude,
            affected_area=affected_area,
            stimuli=stimuli
        )

        self.events.append(event)

        # Apply event effects
        self._apply_event_effects(event)

        # Trigger callbacks
        for callback in self._callbacks["event_occurred"]:
            callback(event)

        return event

    def _apply_event_effects(self, event: EnvironmentEvent):
        """Apply effects of an event"""
        if event.magnitude > 0.7:
            # Major event affects global conditions
            if event.event_type in ["disaster", "attack", "failure"]:
                self._degrade_condition()
            elif event.event_type in ["bonus", "windfall", "upgrade"]:
                self._improve_condition()

        # Zone-specific effects
        if event.affected_area in self.zones:
            zone = self.zones[event.affected_area]
            if event.event_type in ["disaster", "attack"]:
                zone.condition = EnvironmentCondition.HOSTILE
            elif event.event_type in ["bonus", "upgrade"]:
                zone.condition = EnvironmentCondition.FAVORABLE

    def _degrade_condition(self):
        """Degrade environmental conditions"""
        conditions = list(EnvironmentCondition)
        current_idx = conditions.index(self.state.condition)
        if current_idx < len(conditions) - 1:
            new_condition = conditions[current_idx + 1]
            self.state.condition = new_condition
            self.state.stability = max(0.1, self.state.stability - 0.2)

            for callback in self._callbacks["condition_changed"]:
                callback(new_condition)

    def _improve_condition(self):
        """Improve environmental conditions"""
        conditions = list(EnvironmentCondition)
        current_idx = conditions.index(self.state.condition)
        if current_idx > 0:
            new_condition = conditions[current_idx - 1]
            self.state.condition = new_condition
            self.state.stability = min(1.0, self.state.stability + 0.1)

            for callback in self._callbacks["condition_changed"]:
                callback(new_condition)

    def provide_resource(
        self,
        organism_id: str,
        resource_type: str,
        requested_amount: float
    ) -> float:
        """Provide resources to an organism"""
        if resource_type not in self.global_resources:
            return 0.0

        available = self.global_resources[resource_type]
        provided = min(requested_amount, available)

        self.global_resources[resource_type] -= provided

        return provided

    def return_resource(self, resource_type: str, amount: float):
        """Return resources to the environment"""
        if resource_type in self.global_resources:
            self.global_resources[resource_type] += amount

    def get_zone_info(self, zone_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a zone"""
        zone = self.zones.get(zone_name)
        if not zone:
            return None

        return {
            "name": zone.name,
            "condition": zone.condition.value,
            "resources": zone.resources.copy(),
            "hazards": zone.hazards.copy(),
            "inhabitants": len(zone.inhabitants),
            "properties": zone.properties.copy()
        }

    def get_active_stimuli(
        self,
        stimulus_type: Optional[StimulusType] = None,
        location: Optional[str] = None
    ) -> List[Stimulus]:
        """Get currently active stimuli"""
        stimuli = []

        for stimulus in self.active_stimuli.values():
            if not stimulus.is_active:
                continue

            if stimulus_type and stimulus.type != stimulus_type:
                continue

            if location and stimulus.location != location:
                continue

            stimuli.append(stimulus)

        return stimuli

    def tick(self):
        """Process one environment cycle"""
        self.cycle_count += 1

        # Remove expired stimuli
        expired = [
            sid for sid, s in self.active_stimuli.items()
            if not s.is_active
        ]
        for sid in expired:
            del self.active_stimuli[sid]

        # Move old events to history
        while len(self.events) > 100:
            self.event_history.append(self.events.pop(0))

        # Natural resource regeneration
        for resource in self.global_resources:
            self.global_resources[resource] = min(
                5000.0,
                self.global_resources[resource] + self.base_resource_rate
            )

        # Environmental dynamics
        self._apply_dynamics()

        # Generate background stimuli
        if np.random.random() < 0.5:
            self.generate_stimuli()

    def _apply_dynamics(self):
        """Apply environmental dynamics"""
        # Random fluctuations
        self.state.temperature += np.random.uniform(-0.05, 0.05) * self.volatility
        self.state.temperature = np.clip(self.state.temperature, 0, 1)

        self.state.pressure += np.random.uniform(-0.03, 0.03) * self.volatility
        self.state.pressure = np.clip(self.state.pressure, 0, 1)

        self.state.entropy += np.random.uniform(-0.02, 0.02) * self.volatility
        self.state.entropy = np.clip(self.state.entropy, 0, 1)

        # Stability recovery
        self.state.stability = min(1.0, self.state.stability + 0.01)

        # Condition may shift based on state
        if self.state.entropy > 0.7 and self.state.stability < 0.3:
            self._degrade_condition()

        self.state.timestamp = time.time()

    def on_stimulus_created(self, callback: Callable[[Stimulus], None]):
        """Register callback for stimulus creation"""
        self._callbacks["stimulus_created"].append(callback)

    def on_event_occurred(self, callback: Callable[[EnvironmentEvent], None]):
        """Register callback for events"""
        self._callbacks["event_occurred"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get environment status"""
        return {
            "condition": self.state.condition.value,
            "resource_availability": self.state.resource_availability.value,
            "temperature": self.state.temperature,
            "pressure": self.state.pressure,
            "entropy": self.state.entropy,
            "stability": self.state.stability,
            "zones": len(self.zones),
            "organisms": len(self.organisms),
            "active_stimuli": len(self.active_stimuli),
            "total_stimuli": self.total_stimuli_generated,
            "events": len(self.events),
            "global_resources": self.global_resources.copy(),
            "cycle_count": self.cycle_count
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Environment Simulation Demo")
    print("=" * 50)

    # Create environment
    env = Environment(base_resource_rate=5.0, volatility=0.3)

    print(f"\n1. Initial state:")
    status = env.get_status()
    print(f"   Condition: {status['condition']}")
    print(f"   Stability: {status['stability']:.2f}")
    print(f"   Zones: {status['zones']}")

    # Register organisms
    print("\n2. Registering organisms...")
    for i in range(5):
        org_id = f"organism_{i:03d}"
        zone = ["core", "periphery", "frontier", "commons"][i % 4]
        env.register_organism(org_id, zone)
    print(f"   Registered {len(env.organisms)} organisms")

    # Generate initial stimuli
    print("\n3. Generating stimuli...")
    stimuli = env.generate_stimuli()
    print(f"   Generated {len(stimuli)} stimuli")
    for s in stimuli:
        print(f"   - {s.type.value}: intensity={s.intensity:.2f}")

    # Trigger an event
    print("\n4. Triggering event...")
    event = env.trigger_event(
        name="resource_surge",
        event_type="bonus",
        magnitude=0.6,
        affected_area="periphery"
    )
    print(f"   Event: {event.name}")
    print(f"   Stimuli generated: {len(event.stimuli)}")

    # Run environment cycles
    print("\n5. Running environment cycles...")
    for cycle in range(20):
        env.tick()

        if cycle % 5 == 0:
            status = env.get_status()
            print(f"   Cycle {cycle}: condition={status['condition']}, "
                  f"stimuli={status['active_stimuli']}, "
                  f"stability={status['stability']:.2f}")

    # Get zone info
    print("\n6. Zone information:")
    for zone_name in ["core", "frontier"]:
        info = env.get_zone_info(zone_name)
        if info:
            print(f"   {zone_name}:")
            print(f"      Condition: {info['condition']}")
            print(f"      Inhabitants: {info['inhabitants']}")

    # Request resources
    print("\n7. Resource provision:")
    amount = env.provide_resource("organism_001", "compute", 100)
    print(f"   Provided {amount} compute to organism_001")

    # Final status
    print("\n8. Final status:")
    status = env.get_status()
    for key in ["condition", "stability", "entropy", "active_stimuli", "total_stimuli"]:
        print(f"   {key}: {status[key]}")
