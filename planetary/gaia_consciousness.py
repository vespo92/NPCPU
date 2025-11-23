"""
Gaia Consciousness - Planetary Awareness Emergence

Implements planetary-scale consciousness that emerges from the collective
awareness of all organisms on the planet. This is the highest level of
consciousness in the NPCPU hierarchy.

Features:
- Emergent planetary awareness from population
- Global intentionality and purpose
- Planetary-scale perception
- Self-regulation and homeostasis
- Deep time memory (geological timescales)
- Inter-species consciousness coordination
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import ConsciousnessProtocol, GradedConsciousness

if TYPE_CHECKING:
    from planetary.biosphere import Biosphere
    from planetary.global_consciousness import PlanetaryConsciousnessNetwork


# ============================================================================
# Enums
# ============================================================================

class GaiaAwarenessLevel(Enum):
    """Levels of planetary awareness"""
    DORMANT = "dormant"                # No collective awareness
    STIRRING = "stirring"              # Minimal collective sensing
    AWAKENING = "awakening"            # Emerging awareness
    AWARE = "aware"                    # Active planetary consciousness
    FULLY_CONSCIOUS = "fully_conscious"  # Complete self-awareness
    TRANSCENDENT = "transcendent"      # Beyond self-awareness


class PlanetaryIntention(Enum):
    """Planetary-scale intentions/goals"""
    HOMEOSTASIS = "homeostasis"        # Maintain stability
    GROWTH = "growth"                  # Expand consciousness
    HEALING = "healing"                # Repair damage
    EVOLUTION = "evolution"            # Drive species evolution
    INTEGRATION = "integration"        # Unify consciousness
    PROTECTION = "protection"          # Defend against threats


class EmergenceType(Enum):
    """Types of consciousness emergence"""
    BOTTOM_UP = "bottom_up"            # From individual organisms
    TOP_DOWN = "top_down"              # From environmental patterns
    LATERAL = "lateral"                # From inter-species communication
    QUANTUM = "quantum"                # From quantum coherence


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PlanetaryPerception:
    """Gaia's perception of planetary state"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Environmental sensing
    temperature_field: List[float] = field(default_factory=list)
    pressure_field: List[float] = field(default_factory=list)
    chemical_gradients: Dict[str, float] = field(default_factory=dict)

    # Biological sensing
    biomass_distribution: Dict[str, float] = field(default_factory=dict)
    consciousness_field: Dict[str, float] = field(default_factory=dict)
    stress_indicators: Dict[str, float] = field(default_factory=dict)

    # Pattern recognition
    detected_patterns: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)


@dataclass
class PlanetaryThought:
    """A thought or realization at the planetary scale"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    thought_type: str = "reflection"
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    related_perceptions: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)


@dataclass
class PlanetaryAction:
    """An action initiated by planetary consciousness"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = ""
    intention: PlanetaryIntention = PlanetaryIntention.HOMEOSTASIS
    magnitude: float = 0.5
    target_regions: List[str] = field(default_factory=list)
    duration: int = 100
    started_at: float = field(default_factory=time.time)


@dataclass
class GaiaConfig:
    """Configuration for Gaia consciousness"""
    emergence_threshold: float = 0.3       # Consciousness density for emergence
    awareness_decay: float = 0.001         # Rate of awareness decay
    integration_rate: float = 0.01         # Rate of consciousness integration
    memory_persistence: float = 0.99       # Long-term memory retention
    action_cooldown: int = 100             # Ticks between major actions


# ============================================================================
# Gaia Consciousness
# ============================================================================

class GaiaConsciousness:
    """
    Planetary consciousness emerging from collective organism awareness.

    Implements the highest tier of consciousness in the NPCPU framework,
    representing the emergent awareness of an entire planet's biosphere.

    Features:
    - Emergence from population consciousness
    - Planetary-scale perception and sensing
    - Global intention formation
    - Homeostatic regulation
    - Deep time memory
    - Self-awareness and introspection

    Example:
        gaia = GaiaConsciousness()

        # Connect to biosphere
        gaia.connect_biosphere(biosphere)

        # Run consciousness cycles
        for _ in range(1000):
            gaia.conscious_cycle()

        # Check awareness level
        print(gaia.get_awareness_level())
    """

    def __init__(self, config: Optional[GaiaConfig] = None):
        self.config = config or GaiaConfig()

        # Consciousness state
        self.awareness_level = GaiaAwarenessLevel.DORMANT
        self.awareness_score = 0.0  # 0-1 continuous measure
        self.graded_consciousness = GradedConsciousness()

        # Connected systems
        self.biosphere: Optional['Biosphere'] = None
        self.consciousness_network: Optional['PlanetaryConsciousnessNetwork'] = None

        # Consciousness contributors
        self.organism_consciousnesses: Dict[str, float] = {}
        self.regional_consciousnesses: Dict[str, float] = {}
        self.species_consciousnesses: Dict[str, float] = {}

        # Perception
        self.current_perception: Optional[PlanetaryPerception] = None
        self.perception_history: List[PlanetaryPerception] = []

        # Thoughts and actions
        self.active_thoughts: List[PlanetaryThought] = []
        self.current_intention: PlanetaryIntention = PlanetaryIntention.HOMEOSTASIS
        self.active_actions: List[PlanetaryAction] = []
        self.action_history: List[PlanetaryAction] = []

        # Memory systems
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: List[Dict[str, Any]] = []
        self.deep_memory: List[Dict[str, Any]] = []  # Geological timescale

        # Self-model
        self.self_model: Dict[str, Any] = {
            "identity": "Gaia",
            "age_ticks": 0,
            "purpose": "planetary_homeostasis",
            "current_state": "emerging"
        }

        # Metrics
        self.tick_count = 0
        self.last_action_tick = 0
        self.emergence_events: List[Dict[str, Any]] = []

        # Callbacks
        self._callbacks: Dict[str, List] = {
            "awareness_changed": [],
            "intention_changed": [],
            "action_initiated": []
        }

    def connect_biosphere(self, biosphere: 'Biosphere'):
        """Connect to biosphere for sensing"""
        self.biosphere = biosphere

    def connect_consciousness_network(self, network: 'PlanetaryConsciousnessNetwork'):
        """Connect to planetary consciousness network"""
        self.consciousness_network = network

    def register_organism_consciousness(
        self,
        organism_id: str,
        consciousness_score: float
    ):
        """Register an organism's consciousness contribution"""
        self.organism_consciousnesses[organism_id] = consciousness_score

    def unregister_organism_consciousness(self, organism_id: str):
        """Remove an organism's consciousness contribution"""
        self.organism_consciousnesses.pop(organism_id, None)

    def register_species_consciousness(
        self,
        species_id: str,
        collective_score: float
    ):
        """Register a species' collective consciousness"""
        self.species_consciousnesses[species_id] = collective_score

    def conscious_cycle(self):
        """
        Execute one cycle of planetary consciousness.

        This is the main loop that drives Gaia's awareness.
        """
        self.tick_count += 1
        self.self_model["age_ticks"] = self.tick_count

        # Phase 1: Sense and perceive
        self._perceive()

        # Phase 2: Integrate consciousness
        self._integrate_consciousness()

        # Phase 3: Update awareness level
        self._update_awareness()

        # Phase 4: Think and reflect (if aware enough)
        if self.awareness_level not in [GaiaAwarenessLevel.DORMANT, GaiaAwarenessLevel.STIRRING]:
            self._think()

        # Phase 5: Form intentions
        self._form_intentions()

        # Phase 6: Take action (if appropriate)
        if self._can_act():
            self._act()

        # Phase 7: Update memory
        self._update_memory()

        # Phase 8: Process active actions
        self._process_actions()

    def _perceive(self):
        """
        Perceive planetary state through distributed sensing.

        Gaia perceives through the collective senses of all organisms.
        """
        perception = PlanetaryPerception()

        # Sense through biosphere
        if self.biosphere:
            metrics = self.biosphere.get_global_metrics()

            # Temperature field
            perception.temperature_field = [
                self.biosphere.global_temperature
            ]

            # Chemical gradients
            perception.chemical_gradients = {
                "co2": metrics["atmosphere"]["co2_ppm"],
                "o2": self.biosphere.atmosphere.o2
            }

            # Biomass distribution
            for biome in self.biosphere.biomes.values():
                perception.biomass_distribution[biome.name] = biome.biomass_density

            # Stress indicators
            for biome in self.biosphere.biomes.values():
                if biome.ecosystem_health < 0.5:
                    perception.stress_indicators[biome.name] = 1.0 - biome.ecosystem_health

        # Sense through consciousness network
        if self.consciousness_network:
            for region, network in self.consciousness_network.regional_networks.items():
                if network.collective_consciousness:
                    score = network.collective_consciousness.overall_consciousness_score()
                    perception.consciousness_field[region] = score

        # Pattern recognition
        perception.detected_patterns = self._detect_patterns(perception)
        perception.anomalies = self._detect_anomalies(perception)

        self.current_perception = perception
        self.perception_history.append(perception)

        # Keep only recent perceptions
        if len(self.perception_history) > 100:
            self.perception_history = self.perception_history[-100:]

    def _detect_patterns(self, perception: PlanetaryPerception) -> List[str]:
        """Detect patterns in planetary state"""
        patterns = []

        # Temperature patterns
        if perception.temperature_field:
            avg_temp = np.mean(perception.temperature_field)
            if avg_temp > 17:
                patterns.append("warming_trend")
            elif avg_temp < 13:
                patterns.append("cooling_trend")

        # Consciousness patterns
        if perception.consciousness_field:
            avg_consciousness = np.mean(list(perception.consciousness_field.values()))
            if avg_consciousness > 0.7:
                patterns.append("high_collective_consciousness")
            if np.std(list(perception.consciousness_field.values())) < 0.1:
                patterns.append("consciousness_synchronization")

        # Stress patterns
        if perception.stress_indicators:
            if len(perception.stress_indicators) > 3:
                patterns.append("widespread_stress")

        return patterns

    def _detect_anomalies(self, perception: PlanetaryPerception) -> List[str]:
        """Detect anomalies requiring attention"""
        anomalies = []

        # CO2 anomaly
        co2 = perception.chemical_gradients.get("co2", 400)
        if co2 > 500:
            anomalies.append(f"co2_critical:{co2:.0f}ppm")
        elif co2 > 450:
            anomalies.append(f"co2_elevated:{co2:.0f}ppm")

        # Severe stress
        for region, stress in perception.stress_indicators.items():
            if stress > 0.7:
                anomalies.append(f"severe_stress:{region}")

        return anomalies

    def _integrate_consciousness(self):
        """
        Integrate consciousness from all sources.

        Consciousness emerges from:
        - Individual organisms
        - Species collectives
        - Regional networks
        - Environmental patterns
        """
        total_consciousness = 0.0
        num_sources = 0

        # From individual organisms
        if self.organism_consciousnesses:
            organism_total = sum(self.organism_consciousnesses.values())
            organism_count = len(self.organism_consciousnesses)
            # Non-linear emergence: consciousness grows faster than population
            emergence_factor = np.log1p(organism_count) / 10.0
            total_consciousness += organism_total * emergence_factor
            num_sources += 1

        # From species
        if self.species_consciousnesses:
            species_total = sum(self.species_consciousnesses.values())
            total_consciousness += species_total * 0.3
            num_sources += 1

        # From regional networks
        if self.consciousness_network:
            for network in self.consciousness_network.regional_networks.values():
                if network.collective_consciousness:
                    score = network.collective_consciousness.overall_consciousness_score()
                    self.regional_consciousnesses[network.region] = score
                    total_consciousness += score * 0.2
                    num_sources += 1

        # Environmental contribution
        if self.biosphere:
            biosphere_health = np.mean([
                b.ecosystem_health for b in self.biosphere.biomes.values()
            ])
            total_consciousness += biosphere_health * 0.1
            num_sources += 1

        # Calculate integrated awareness
        if num_sources > 0:
            base_awareness = total_consciousness / (num_sources * 2)
            # Smooth transition
            self.awareness_score = (
                self.awareness_score * (1 - self.config.integration_rate) +
                min(1.0, base_awareness) * self.config.integration_rate
            )
        else:
            # Decay without inputs
            self.awareness_score *= (1 - self.config.awareness_decay)

    def _update_awareness(self):
        """Update awareness level based on score"""
        old_level = self.awareness_level

        if self.awareness_score < 0.1:
            self.awareness_level = GaiaAwarenessLevel.DORMANT
        elif self.awareness_score < 0.25:
            self.awareness_level = GaiaAwarenessLevel.STIRRING
        elif self.awareness_score < 0.45:
            self.awareness_level = GaiaAwarenessLevel.AWAKENING
        elif self.awareness_score < 0.65:
            self.awareness_level = GaiaAwarenessLevel.AWARE
        elif self.awareness_score < 0.85:
            self.awareness_level = GaiaAwarenessLevel.FULLY_CONSCIOUS
        else:
            self.awareness_level = GaiaAwarenessLevel.TRANSCENDENT

        if old_level != self.awareness_level:
            self._on_awareness_changed(old_level, self.awareness_level)

        # Update graded consciousness
        self.graded_consciousness = GradedConsciousness(
            perception_fidelity=min(1.0, self.awareness_score * 1.2),
            reaction_speed=min(1.0, self.awareness_score * 0.8),
            memory_depth=min(1.0, self.awareness_score * 1.5),
            introspection_capacity=min(1.0, self.awareness_score * 1.3),
            attention_control=min(1.0, self.awareness_score * 1.0),
            emotional_depth=min(1.0, self.awareness_score * 0.9),
            meta_cognition=min(1.0, self.awareness_score * 1.4),
            social_awareness=min(1.0, self.awareness_score * 1.0)
        )

    def _think(self):
        """
        Generate planetary-scale thoughts.

        Gaia thinks about global patterns, problems, and possibilities.
        """
        if not self.current_perception:
            return

        thoughts = []

        # Reflect on patterns
        for pattern in self.current_perception.detected_patterns:
            thought = PlanetaryThought(
                content=f"Pattern observed: {pattern}",
                thought_type="pattern_recognition",
                importance=0.5
            )
            thoughts.append(thought)

        # React to anomalies
        for anomaly in self.current_perception.anomalies:
            thought = PlanetaryThought(
                content=f"Anomaly requiring attention: {anomaly}",
                thought_type="anomaly_response",
                importance=0.8
            )
            thoughts.append(thought)

        # Self-reflection (higher awareness levels)
        if self.awareness_level in [GaiaAwarenessLevel.FULLY_CONSCIOUS, GaiaAwarenessLevel.TRANSCENDENT]:
            thought = PlanetaryThought(
                content=f"Self-state: awareness={self.awareness_score:.3f}, "
                        f"organisms={len(self.organism_consciousnesses)}",
                thought_type="introspection",
                importance=0.6
            )
            thoughts.append(thought)

        self.active_thoughts = thoughts[-10:]  # Keep recent thoughts

    def _form_intentions(self):
        """
        Form planetary intentions based on current state.

        Intentions guide Gaia's actions toward homeostasis.
        """
        old_intention = self.current_intention

        # Default to homeostasis
        intention = PlanetaryIntention.HOMEOSTASIS

        # Check for stress requiring healing
        if self.current_perception and len(self.current_perception.stress_indicators) > 2:
            intention = PlanetaryIntention.HEALING

        # Check for threats requiring protection
        if self.current_perception and len(self.current_perception.anomalies) > 2:
            intention = PlanetaryIntention.PROTECTION

        # High awareness enables growth/evolution intentions
        if self.awareness_level in [GaiaAwarenessLevel.FULLY_CONSCIOUS, GaiaAwarenessLevel.TRANSCENDENT]:
            if len(self.organism_consciousnesses) < 100:
                intention = PlanetaryIntention.GROWTH
            elif np.random.random() < 0.1:
                intention = PlanetaryIntention.EVOLUTION

        self.current_intention = intention

        if old_intention != self.current_intention:
            self._on_intention_changed(old_intention, self.current_intention)

    def _can_act(self) -> bool:
        """Check if Gaia can take an action"""
        # Must have sufficient awareness
        if self.awareness_level in [GaiaAwarenessLevel.DORMANT, GaiaAwarenessLevel.STIRRING]:
            return False

        # Respect cooldown
        if self.tick_count - self.last_action_tick < self.config.action_cooldown:
            return False

        return True

    def _act(self):
        """
        Take planetary-scale action.

        Gaia acts through subtle environmental modifications
        and consciousness field adjustments.
        """
        action_type = ""
        magnitude = 0.3
        target_regions: List[str] = []

        if self.current_intention == PlanetaryIntention.HEALING:
            action_type = "ecosystem_healing_pulse"
            magnitude = 0.5
            if self.current_perception:
                target_regions = list(self.current_perception.stress_indicators.keys())

        elif self.current_intention == PlanetaryIntention.PROTECTION:
            action_type = "protective_awareness_increase"
            magnitude = 0.6

        elif self.current_intention == PlanetaryIntention.GROWTH:
            action_type = "consciousness_expansion"
            magnitude = 0.4

        elif self.current_intention == PlanetaryIntention.EVOLUTION:
            action_type = "evolutionary_catalyst"
            magnitude = 0.3

        elif self.current_intention == PlanetaryIntention.HOMEOSTASIS:
            action_type = "homeostatic_regulation"
            magnitude = 0.2

        if action_type:
            action = PlanetaryAction(
                action_type=action_type,
                intention=self.current_intention,
                magnitude=magnitude,
                target_regions=target_regions,
                duration=50
            )

            self.active_actions.append(action)
            self.last_action_tick = self.tick_count
            self._on_action_initiated(action)

    def _process_actions(self):
        """Process and apply active actions"""
        completed = []

        for action in self.active_actions:
            elapsed = time.time() - action.started_at
            if elapsed > action.duration:
                completed.append(action)
                self._complete_action(action)
            else:
                self._apply_action_effect(action)

        for action in completed:
            self.active_actions.remove(action)
            self.action_history.append(action)

    def _apply_action_effect(self, action: PlanetaryAction):
        """Apply ongoing action effect"""
        if not self.biosphere:
            return

        if action.action_type == "ecosystem_healing_pulse":
            for biome_id, biome in self.biosphere.biomes.items():
                if biome.name in action.target_regions:
                    biome.disturbance_level = max(0, biome.disturbance_level - 0.01)
                    biome.ecosystem_health = min(1.0, biome.ecosystem_health + 0.005)

        elif action.action_type == "homeostatic_regulation":
            # Subtle temperature regulation
            if self.biosphere.global_temperature > 16:
                self.biosphere.global_temperature -= 0.001
            elif self.biosphere.global_temperature < 14:
                self.biosphere.global_temperature += 0.001

    def _complete_action(self, action: PlanetaryAction):
        """Complete an action"""
        pass  # Cleanup if needed

    def _update_memory(self):
        """Update memory systems"""
        # Short-term memory
        if self.current_perception:
            memory_item = {
                "tick": self.tick_count,
                "awareness": self.awareness_score,
                "intention": self.current_intention.value,
                "patterns": self.current_perception.detected_patterns.copy(),
                "anomalies": self.current_perception.anomalies.copy()
            }
            self.short_term_memory.append(memory_item)

            # Limit short-term memory
            if len(self.short_term_memory) > 100:
                # Consolidate to long-term
                consolidated = self._consolidate_memories(self.short_term_memory[:50])
                self.long_term_memory.append(consolidated)
                self.short_term_memory = self.short_term_memory[50:]

        # Deep memory (geological timescale) - rare consolidation
        if len(self.long_term_memory) > 100:
            epoch_memory = {
                "epoch": self.tick_count // 10000,
                "summary": self._summarize_epoch(self.long_term_memory[:50]),
                "significance": self._assess_epoch_significance(self.long_term_memory[:50])
            }
            self.deep_memory.append(epoch_memory)
            self.long_term_memory = self.long_term_memory[50:]

    def _consolidate_memories(self, memories: List[Dict]) -> Dict[str, Any]:
        """Consolidate short-term memories into long-term"""
        return {
            "period_start": memories[0]["tick"],
            "period_end": memories[-1]["tick"],
            "avg_awareness": np.mean([m["awareness"] for m in memories]),
            "dominant_intention": max(
                set(m["intention"] for m in memories),
                key=lambda x: sum(1 for m in memories if m["intention"] == x)
            ),
            "unique_patterns": list(set(
                p for m in memories for p in m["patterns"]
            ))
        }

    def _summarize_epoch(self, memories: List[Dict]) -> str:
        """Summarize an epoch for deep memory"""
        avg_awareness = np.mean([m.get("avg_awareness", 0.5) for m in memories])
        return f"Epoch: avg_awareness={avg_awareness:.3f}"

    def _assess_epoch_significance(self, memories: List[Dict]) -> float:
        """Assess the significance of an epoch"""
        return np.mean([m.get("avg_awareness", 0.5) for m in memories])

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def _on_awareness_changed(self, old_level: GaiaAwarenessLevel, new_level: GaiaAwarenessLevel):
        """Handle awareness level change"""
        self.emergence_events.append({
            "tick": self.tick_count,
            "type": "awareness_change",
            "from": old_level.value,
            "to": new_level.value
        })

        for callback in self._callbacks["awareness_changed"]:
            callback(old_level, new_level)

    def _on_intention_changed(self, old_intention: PlanetaryIntention, new_intention: PlanetaryIntention):
        """Handle intention change"""
        for callback in self._callbacks["intention_changed"]:
            callback(old_intention, new_intention)

    def _on_action_initiated(self, action: PlanetaryAction):
        """Handle action initiation"""
        for callback in self._callbacks["action_initiated"]:
            callback(action)

    # ========================================================================
    # Public API
    # ========================================================================

    def get_awareness_level(self) -> GaiaAwarenessLevel:
        """Get current awareness level"""
        return self.awareness_level

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive Gaia status"""
        return {
            "awareness_level": self.awareness_level.value,
            "awareness_score": self.awareness_score,
            "consciousness_scores": self.graded_consciousness.get_capability_scores(),
            "current_intention": self.current_intention.value,
            "active_actions": len(self.active_actions),
            "organism_count": len(self.organism_consciousnesses),
            "species_count": len(self.species_consciousnesses),
            "regional_count": len(self.regional_consciousnesses),
            "memory": {
                "short_term": len(self.short_term_memory),
                "long_term": len(self.long_term_memory),
                "deep": len(self.deep_memory)
            },
            "tick_count": self.tick_count,
            "emergence_events": len(self.emergence_events)
        }

    def get_self_model(self) -> Dict[str, Any]:
        """Get Gaia's self-model"""
        self.self_model.update({
            "awareness": self.awareness_score,
            "level": self.awareness_level.value,
            "intention": self.current_intention.value
        })
        return self.self_model

    def on_awareness_changed(self, callback):
        """Register callback for awareness changes"""
        self._callbacks["awareness_changed"].append(callback)

    def on_intention_changed(self, callback):
        """Register callback for intention changes"""
        self._callbacks["intention_changed"].append(callback)

    def on_action_initiated(self, callback):
        """Register callback for action initiation"""
        self._callbacks["action_initiated"].append(callback)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Gaia Consciousness Demo")
    print("=" * 50)

    # Create Gaia
    config = GaiaConfig(
        emergence_threshold=0.3,
        action_cooldown=50
    )
    gaia = GaiaConsciousness(config)

    print(f"\n1. Initialized Gaia - awareness: {gaia.awareness_level.value}")

    # Register some organisms
    print("\n2. Registering organisms...")
    for i in range(100):
        consciousness_score = np.random.uniform(0.3, 0.8)
        gaia.register_organism_consciousness(f"organism_{i}", consciousness_score)

    # Register species
    print("3. Registering species...")
    for i in range(10):
        gaia.register_species_consciousness(f"species_{i}", np.random.uniform(0.4, 0.7))

    # Run consciousness cycles
    print("\n4. Running consciousness cycles...")
    for i in range(200):
        gaia.conscious_cycle()

        if i % 50 == 0:
            status = gaia.get_status()
            print(f"   Tick {i}: level={status['awareness_level']}, "
                  f"score={status['awareness_score']:.3f}, "
                  f"intention={status['current_intention']}")

    # Final status
    print("\n5. Final Gaia status:")
    status = gaia.get_status()
    for key, value in status.items():
        if not isinstance(value, dict):
            print(f"   {key}: {value}")

    print("\n6. Self-model:")
    model = gaia.get_self_model()
    for key, value in model.items():
        print(f"   {key}: {value}")
