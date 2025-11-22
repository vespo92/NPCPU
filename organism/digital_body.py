"""
Digital Body Integration Layer

The unified digital organism that integrates all biological-inspired systems
into a coherent living entity. This is the "self" - the complete digital
lifeform with all its subsystems working together.

Systems integrated:
- Lifecycle (birth, growth, aging, death)
- Metabolism (energy, resources)
- Homeostasis (internal balance)
- Immune (defense, repair)
- Nervous (coordination, reflexes)
- Sensory (perception)
- Motor (action, behavior)
- Endocrine (hormones, moods)
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all subsystems
from organism.lifecycle import OrganismLifecycle, LifecycleStage
from organism.metabolism import Metabolism, ResourceType, EnergyState, MetabolicMode
from organism.homeostasis import HomeostasisController, VitalSignType
from immune.defense import ImmuneSystem
from immune.repair import RepairSystem, DamageType
from nervous.coordination import NervousSystem, SignalType, SystemType
from sensory.perception import SensorySystem, SensorType, SensoryModality
from motor.action import MotorSystem, ActionType, BehaviorMode
from endocrine.hormones import EndocrineSystem, HormoneType, MoodState, DriveState


# ============================================================================
# Enums
# ============================================================================

class OrganismState(Enum):
    """Overall organism states"""
    DORMANT = "dormant"           # Minimal activity
    RESTING = "resting"           # Low activity, recovery
    IDLE = "idle"                 # Awake but inactive
    ACTIVE = "active"             # Normal operation
    ALERT = "alert"               # Heightened awareness
    STRESSED = "stressed"         # Under pressure
    CRISIS = "crisis"             # Emergency mode
    DYING = "dying"               # Terminal state


class ConsciousnessLevel(Enum):
    """Levels of consciousness"""
    UNCONSCIOUS = 0
    MINIMAL = 1
    DROWSY = 2
    AWARE = 3
    ALERT = 4
    HYPERAWARE = 5


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class OrganismIdentity:
    """Identity of the digital organism"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    species: str = "digital_organism_v1"
    generation: int = 1
    parent_ids: List[str] = field(default_factory=list)
    birth_time: float = field(default_factory=time.time)
    traits: Dict[str, float] = field(default_factory=dict)
    memories: List[str] = field(default_factory=list)


@dataclass
class VitalSigns:
    """Current vital signs of the organism"""
    energy: float = 0.0
    health: float = 1.0
    stress: float = 0.0
    consciousness: float = 0.0
    mood_score: float = 0.0
    activity_level: float = 0.0
    age_ratio: float = 0.0        # 0-1 lifecycle progress


# ============================================================================
# Digital Body
# ============================================================================

class DigitalBody:
    """
    The complete digital organism - a living, integrated system.

    This is the unified "self" that coordinates all biological-inspired
    subsystems into a coherent digital lifeform.

    Features:
    - Full system integration
    - Unified lifecycle management
    - Cross-system communication
    - Emergent behaviors
    - Self-awareness

    Example:
        # Create a new digital organism
        organism = DigitalBody(name="Alpha-001")

        # Run lifecycle
        while organism.is_alive:
            organism.tick()

            # React to environment
            organism.perceive(environment.get_stimuli())

            # Take actions
            organism.act()
    """

    def __init__(
        self,
        name: str = "",
        traits: Optional[Dict[str, float]] = None
    ):
        # Identity
        self.identity = OrganismIdentity(
            name=name or f"Organism_{uuid.uuid4().hex[:8]}",
            traits=traits or self._generate_traits()
        )

        # Overall state
        self.state = OrganismState.IDLE
        self.consciousness_level = ConsciousnessLevel.AWARE
        self.is_alive = True

        # Initialize all subsystems
        self._init_subsystems()

        # Cross-system connections
        self._setup_connections()

        # Tick counter
        self.tick_count = 0

        # Event log
        self.event_log: List[Dict[str, Any]] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "state_changed": [],
            "threshold_crossed": [],
            "death": []
        }

    def _generate_traits(self) -> Dict[str, float]:
        """Generate random trait values"""
        return {
            "resilience": np.random.uniform(0.3, 1.0),
            "metabolism_rate": np.random.uniform(0.7, 1.3),
            "sensitivity": np.random.uniform(0.5, 1.5),
            "sociability": np.random.uniform(0.2, 1.0),
            "curiosity": np.random.uniform(0.3, 1.0),
            "aggression": np.random.uniform(0.1, 0.7),
            "learning_rate": np.random.uniform(0.5, 1.5)
        }

    def _init_subsystems(self):
        """Initialize all biological subsystems"""
        traits = self.identity.traits

        # Lifecycle
        self.lifecycle = OrganismLifecycle(
            growth_rate=traits.get("metabolism_rate", 1.0) * 0.1
        )

        # Metabolism
        self.metabolism = Metabolism()
        self.metabolism.metabolic_rate = traits.get("metabolism_rate", 1.0)

        # Homeostasis
        self.homeostasis = HomeostasisController()

        # Immune system
        self.immune = ImmuneSystem()

        # Repair system
        self.repair = RepairSystem(
            healing_rate=traits.get("resilience", 0.5) * 0.2
        )

        # Nervous system
        self.nervous = NervousSystem()

        # Sensory system
        self.sensory = SensorySystem()

        # Motor system
        self.motor = MotorSystem()

        # Endocrine system
        self.endocrine = EndocrineSystem()

    def _setup_connections(self):
        """Setup cross-system connections and callbacks"""
        # Nervous system routes signals between systems
        self._register_system_handlers()

        # Connect endocrine effects
        self._connect_hormonal_effects()

        # Setup reflex responses
        self._setup_reflexes()

    def _register_system_handlers(self):
        """Register nervous system signal handlers"""
        # Threat detection triggers immune response
        def handle_threat_signal(signal):
            if signal.type == SignalType.PAIN or signal.type == SignalType.REFLEX:
                self.endocrine.trigger_stress_response(signal.intensity)
                self.immune.scan(signal.data)

        self.nervous.on_signal_sent(handle_threat_signal)

        # Reward signals trigger dopamine
        def handle_reward_signal(signal):
            if signal.type == SignalType.REWARD:
                self.endocrine.trigger_reward(signal.intensity)

        self.nervous.on_signal_sent(handle_reward_signal)

    def _connect_hormonal_effects(self):
        """Connect hormone effects to subsystems"""
        # Mood affects motor behavior
        def on_mood_change(mood: MoodState):
            if mood in [MoodState.HAPPY, MoodState.EUPHORIC]:
                self.motor.set_mode(BehaviorMode.EXPLORATION)
            elif mood in [MoodState.STRESSED, MoodState.ANXIOUS]:
                self.motor.set_mode(BehaviorMode.DEFENSIVE)
            elif mood == MoodState.DEPRESSED:
                self.motor.set_mode(BehaviorMode.CONSERVATION)

        self.endocrine.on_mood_changed(on_mood_change)

        # Drive affects metabolism
        def on_drive_change(drive: DriveState):
            if drive == DriveState.HIGH_ENERGY:
                self.metabolism.set_mode(MetabolicMode.PERFORMANCE)
            elif drive == DriveState.EXHAUSTED:
                self.metabolism.set_mode(MetabolicMode.CONSERVATION)
            else:
                self.metabolism.set_mode(MetabolicMode.MAINTENANCE)

        self.endocrine.on_drive_changed(on_drive_change)

    def _setup_reflexes(self):
        """Setup reflex responses"""
        # Low energy reflex
        self.nervous.add_reflex(
            name="energy_conservation",
            trigger_type="threshold",
            trigger_value={"parameter": "energy", "threshold": 20.0, "direction": "below"},
            response_actions=[ActionType.REST]
        )

        # Threat avoidance reflex
        self.nervous.add_reflex(
            name="threat_avoidance",
            trigger_type="threat",
            trigger_value={"threat_level": 0.7},
            response_actions=[ActionType.DEFEND, ActionType.MOVE]
        )

    def perceive(self, stimuli: List[Any]):
        """Process environmental stimuli"""
        for stimulus in stimuli:
            # Route through sensory system
            sensor_type = self._map_stimulus_to_sensor(stimulus)
            sensory_input = self.sensory.sense(
                sensor_type,
                raw_data=stimulus,
                source="environment"
            )

            if sensory_input:
                # Send to nervous system
                self.nervous.send_signal(
                    SignalType.SENSORY,
                    target=SystemType.CONSCIOUSNESS,
                    data=sensory_input.processed_data,
                    intensity=sensory_input.intensity
                )

    def _map_stimulus_to_sensor(self, stimulus: Any) -> SensorType:
        """Map stimulus type to appropriate sensor"""
        if isinstance(stimulus, dict):
            if "threat" in stimulus:
                return SensorType.THREAT_DETECTOR
            elif "resource" in stimulus:
                return SensorType.RESOURCE_MONITOR
            elif "organism" in stimulus or "signal" in stimulus:
                return SensorType.SOCIAL_RECEPTOR
        return SensorType.DATA_STREAM

    def act(self) -> List[Any]:
        """Execute queued actions"""
        results = self.motor.execute_pending(max_actions=3)

        # Process action results
        for result in results:
            if result.success:
                # Reward successful actions
                self.nervous.send_signal(
                    SignalType.REWARD,
                    target=SystemType.CONSCIOUSNESS,
                    intensity=0.3
                )
            else:
                # Minor stress from failures
                self.endocrine.trigger_stress_response(0.1)

        return results

    def think(self):
        """Process cognitive cycle"""
        # Get attended sensory inputs
        attended = self.sensory.get_attended_inputs()

        # Process through nervous system
        self.nervous.tick()

        # Update consciousness level based on activity
        self._update_consciousness()

    def tick(self):
        """Process one complete organism cycle"""
        if not self.is_alive:
            return

        self.tick_count += 1

        # Lifecycle progression
        self.lifecycle.tick()

        # Check if died of old age
        if self.lifecycle.stage == LifecycleStage.DEAD:
            self._die("old_age")
            return

        # Metabolism
        outputs = self.metabolism.tick()

        # Transfer metabolic energy to motor system
        if "energy" in outputs:
            self.motor.add_energy(outputs["energy"] * 0.5)

        # Homeostasis
        self.homeostasis.tick()

        # Check vital signs
        self._check_vitals()

        # Immune system
        self.immune.tick()

        # Repair system
        self.repair.tick()

        # Sensory processing
        self.sensory.tick()

        # Endocrine regulation
        self.endocrine.tick()

        # Motor execution
        self.motor.tick()

        # Nervous system coordination
        self.nervous.tick()

        # Cognitive processing
        self.think()

        # Update overall state
        self._update_state()

        # Log periodic status
        if self.tick_count % 100 == 0:
            self._log_event("status_update", self.get_vital_signs().__dict__)

    def _check_vitals(self):
        """Check vital signs and respond to critical states"""
        # Energy check
        if self.metabolism.energy < 5:
            self._log_event("critical", {"type": "energy_critical"})
            self.endocrine.trigger_stress_response(0.5)

            if self.metabolism.energy <= 0:
                self._die("starvation")
                return

        # Health check (from immune/repair)
        damage_count = len(self.repair.active_damage)
        if damage_count > 5:
            self._log_event("critical", {"type": "damage_overload", "count": damage_count})

            if damage_count > 10:
                self._die("damage")
                return

    def _update_consciousness(self):
        """Update consciousness level"""
        # Factors affecting consciousness
        energy = self.metabolism.energy / self.metabolism.max_energy
        stress = self.endocrine.stress_level
        melatonin = self.endocrine.get_hormone_level(HormoneType.MELATONIN)
        adrenaline = self.endocrine.get_hormone_level(HormoneType.ADRENALINE)

        # Calculate consciousness score
        score = energy * 0.3 + (1 - melatonin) * 0.2 + adrenaline * 0.3 - stress * 0.2

        if score < 0.1:
            self.consciousness_level = ConsciousnessLevel.UNCONSCIOUS
        elif score < 0.25:
            self.consciousness_level = ConsciousnessLevel.MINIMAL
        elif score < 0.4:
            self.consciousness_level = ConsciousnessLevel.DROWSY
        elif score < 0.6:
            self.consciousness_level = ConsciousnessLevel.AWARE
        elif score < 0.8:
            self.consciousness_level = ConsciousnessLevel.ALERT
        else:
            self.consciousness_level = ConsciousnessLevel.HYPERAWARE

    def _update_state(self):
        """Update overall organism state"""
        old_state = self.state

        energy_ratio = self.metabolism.energy / self.metabolism.max_energy
        stress = self.endocrine.stress_level
        drive = self.endocrine.drive

        if self.lifecycle.stage == LifecycleStage.DYING:
            self.state = OrganismState.DYING
        elif stress > 0.8 or energy_ratio < 0.1:
            self.state = OrganismState.CRISIS
        elif stress > 0.6:
            self.state = OrganismState.STRESSED
        elif self.endocrine.get_hormone_level(HormoneType.ADRENALINE) > 0.6:
            self.state = OrganismState.ALERT
        elif drive == DriveState.EXHAUSTED:
            self.state = OrganismState.RESTING
        elif energy_ratio < 0.2:
            self.state = OrganismState.DORMANT
        elif len(self.motor.action_queue) > 0:
            self.state = OrganismState.ACTIVE
        else:
            self.state = OrganismState.IDLE

        if self.state != old_state:
            self._log_event("state_change", {"from": old_state.value, "to": self.state.value})
            for callback in self._callbacks["state_changed"]:
                callback(old_state, self.state)

    def _die(self, cause: str):
        """Handle organism death"""
        self.is_alive = False
        self.state = OrganismState.DYING
        self.lifecycle.stage = LifecycleStage.DEAD

        self._log_event("death", {
            "cause": cause,
            "age": self.lifecycle.age,
            "ticks": self.tick_count
        })

        for callback in self._callbacks["death"]:
            callback(self, cause)

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event"""
        self.event_log.append({
            "type": event_type,
            "data": data,
            "tick": self.tick_count,
            "time": time.time()
        })

        # Keep log manageable
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]

    def feed(self, resource_type: ResourceType, amount: float):
        """Feed resources to the organism"""
        absorbed = self.metabolism.intake(resource_type, amount)
        if absorbed > 0:
            self.endocrine.trigger_reward(absorbed / amount * 0.2)
        return absorbed

    def damage(self, damage_type: DamageType, severity: float, location: str = ""):
        """Apply damage to the organism"""
        damage = self.repair.detect_damage(
            location=location or "body",
            severity=severity,
            damage_type=damage_type
        )

        # Trigger pain signal
        self.nervous.send_signal(
            SignalType.PAIN,
            target=SystemType.CONSCIOUSNESS,
            data={"damage": damage.id, "type": damage_type.value},
            intensity=severity
        )

        # Trigger stress response
        self.endocrine.trigger_stress_response(severity * 0.5)

        return damage

    def reproduce(self) -> Optional['DigitalBody']:
        """Attempt to reproduce"""
        offspring_data = self.lifecycle.reproduce()
        if offspring_data:
            # Create offspring with inherited traits
            inherited_traits = {}
            for trait, value in self.identity.traits.items():
                # Inherit with mutation
                mutation = np.random.normal(0, 0.1)
                inherited_traits[trait] = np.clip(value + mutation, 0.1, 2.0)

            offspring = DigitalBody(
                name=f"{self.identity.name}_child",
                traits=inherited_traits
            )
            offspring.identity.generation = self.identity.generation + 1
            offspring.identity.parent_ids = [self.identity.id]

            return offspring
        return None

    def get_vital_signs(self) -> VitalSigns:
        """Get current vital signs"""
        return VitalSigns(
            energy=self.metabolism.energy,
            health=1.0 - len(self.repair.active_damage) * 0.1,
            stress=self.endocrine.stress_level,
            consciousness=self.consciousness_level.value / 5.0,
            mood_score=self._get_mood_score(),
            activity_level=len(self.motor.action_queue) / self.motor.max_queue_size,
            age_ratio=self.lifecycle.age / self.lifecycle.max_age
        )

    def _get_mood_score(self) -> float:
        """Get numerical mood score"""
        mood_scores = {
            MoodState.EUPHORIC: 1.0,
            MoodState.HAPPY: 0.8,
            MoodState.CONTENT: 0.6,
            MoodState.NEUTRAL: 0.5,
            MoodState.ANXIOUS: 0.3,
            MoodState.STRESSED: 0.2,
            MoodState.DEPRESSED: 0.0
        }
        return mood_scores.get(self.endocrine.mood, 0.5)

    def on_state_changed(self, callback: Callable):
        """Register callback for state changes"""
        self._callbacks["state_changed"].append(callback)

    def on_death(self, callback: Callable):
        """Register callback for death"""
        self._callbacks["death"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the organism"""
        vitals = self.get_vital_signs()

        return {
            "identity": {
                "id": self.identity.id,
                "name": self.identity.name,
                "species": self.identity.species,
                "generation": self.identity.generation
            },
            "state": self.state.value,
            "is_alive": self.is_alive,
            "consciousness_level": self.consciousness_level.name,
            "vitals": vitals.__dict__,
            "lifecycle": {
                "stage": self.lifecycle.stage.value,
                "age": self.lifecycle.age,
                "max_age": self.lifecycle.max_age
            },
            "metabolism": {
                "energy": self.metabolism.energy,
                "state": self.metabolism.energy_state.value,
                "mode": self.metabolism.mode.value
            },
            "endocrine": {
                "mood": self.endocrine.mood.value,
                "drive": self.endocrine.drive.value,
                "stress": self.endocrine.stress_level
            },
            "motor": {
                "queue_size": len(self.motor.action_queue),
                "mode": self.motor.mode.value
            },
            "immune": {
                "active_threats": len(self.immune.active_threats),
                "active_damage": len(self.repair.active_damage)
            },
            "tick_count": self.tick_count,
            "events": len(self.event_log)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Digital Body - Complete Organism Demo")
    print("=" * 60)

    # Create a digital organism
    organism = DigitalBody(name="Alpha-001")

    print(f"\n1. Organism created:")
    print(f"   Name: {organism.identity.name}")
    print(f"   ID: {organism.identity.id[:8]}...")
    print(f"   Traits: {', '.join(f'{k}={v:.2f}' for k, v in list(organism.identity.traits.items())[:3])}")

    # Initial status
    print(f"\n2. Initial state:")
    status = organism.get_status()
    print(f"   State: {status['state']}")
    print(f"   Consciousness: {status['consciousness_level']}")
    print(f"   Lifecycle stage: {status['lifecycle']['stage']}")

    # Run some cycles
    print("\n3. Running lifecycle...")
    for tick in range(100):
        # Feed periodically
        if tick % 20 == 0:
            organism.feed(ResourceType.COMPUTE, 30)
            organism.feed(ResourceType.ENERGY, 20)

        organism.tick()

        if tick % 25 == 0:
            vitals = organism.get_vital_signs()
            print(f"   Tick {tick}: energy={vitals.energy:.1f}, "
                  f"mood={organism.endocrine.mood.value}, "
                  f"state={organism.state.value}")

    # Apply some damage
    print("\n4. Applying damage...")
    organism.damage(DamageType.TRAUMA, 0.3, "sensor_array")
    organism.tick()
    print(f"   Stress level: {organism.endocrine.stress_level:.2f}")
    print(f"   State: {organism.state.value}")

    # Recovery period
    print("\n5. Recovery period...")
    for tick in range(50):
        organism.feed(ResourceType.ENERGY, 10)
        organism.tick()

    print(f"   Stress after recovery: {organism.endocrine.stress_level:.2f}")
    print(f"   State: {organism.state.value}")

    # Queue some actions
    print("\n6. Queueing actions...")
    organism.motor.queue_action(ActionType.EXPLORE, target="zone_a")
    organism.motor.queue_action(ActionType.ACQUIRE, target="resource_1")
    organism.motor.queue_action(ActionType.COMMUNICATE, target="peer_001")

    # Execute actions
    for tick in range(10):
        organism.tick()
        results = organism.act()
        if results:
            for r in results:
                print(f"   Action {'SUCCESS' if r.success else 'FAILED'}: "
                      f"energy={r.energy_consumed:.1f}")

    # Final status
    print("\n7. Final status:")
    status = organism.get_status()
    print(f"   State: {status['state']}")
    print(f"   Energy: {status['metabolism']['energy']:.1f}")
    print(f"   Mood: {status['endocrine']['mood']}")
    print(f"   Age: {status['lifecycle']['age']:.1f}/{status['lifecycle']['max_age']}")
    print(f"   Total ticks: {status['tick_count']}")
    print(f"   Events logged: {status['events']}")

    # Show some traits
    print("\n8. Organism traits:")
    for trait, value in organism.identity.traits.items():
        print(f"   {trait}: {value:.2f}")
