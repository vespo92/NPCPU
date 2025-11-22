"""
Sensory Perception System

Enables digital organisms to perceive and interpret their environment.
Implements biological-inspired sensory processing including:
- Multiple sensory modalities
- Attention mechanisms
- Perception filtering
- Sensory integration
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Enums
# ============================================================================

class SensorType(Enum):
    """Types of sensors"""
    DATA_STREAM = "data_stream"           # Incoming data feeds
    RESOURCE_MONITOR = "resource_monitor" # System resources
    NETWORK_PROBE = "network_probe"       # Network conditions
    TIME_SENSE = "time_sense"             # Temporal awareness
    SELF_MONITOR = "self_monitor"         # Internal state monitoring
    THREAT_DETECTOR = "threat_detector"   # Security/threat sensing
    SOCIAL_RECEPTOR = "social_receptor"   # Other organism signals
    ENVIRONMENTAL = "environmental"        # Environment conditions


class SensoryModality(Enum):
    """Sensory modalities (analogous to biological senses)"""
    EXTEROCEPTION = "exteroception"   # External environment
    INTEROCEPTION = "interoception"   # Internal state
    PROPRIOCEPTION = "proprioception" # Self-position/configuration
    NOCICEPTION = "nociception"       # Damage/threat detection
    TEMPORAL = "temporal"             # Time perception
    SOCIAL = "social"                 # Social/communication


class SensoryPriority(Enum):
    """Priority levels for sensory inputs"""
    CRITICAL = 0    # Immediate attention required
    HIGH = 1        # Important, process soon
    NORMAL = 2      # Standard priority
    LOW = 3         # Background processing
    MINIMAL = 4     # Process when idle


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SensoryInput:
    """A sensory input from the environment"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sensor_type: SensorType = SensorType.DATA_STREAM
    modality: SensoryModality = SensoryModality.EXTEROCEPTION
    priority: SensoryPriority = SensoryPriority.NORMAL
    raw_data: Any = None
    processed_data: Any = None
    intensity: float = 0.5           # 0-1 strength
    confidence: float = 1.0          # Reliability of sensing
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Sensor:
    """A single sensor"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    sensor_type: SensorType = SensorType.DATA_STREAM
    modality: SensoryModality = SensoryModality.EXTEROCEPTION
    sensitivity: float = 1.0         # Detection sensitivity
    threshold: float = 0.1           # Minimum signal to detect
    noise_filter: float = 0.05       # Filter out noise below this
    active: bool = True
    adaptation_rate: float = 0.01    # Sensory adaptation
    current_adaptation: float = 0.0  # Current adaptation level


@dataclass
class PerceptionFilter:
    """Filter for processing sensory inputs"""
    name: str
    modalities: Set[SensoryModality] = field(default_factory=set)
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    transform: Optional[Callable[[Any], Any]] = None
    active: bool = True


@dataclass
class AttentionFocus:
    """Current focus of attention"""
    target: str = ""
    modality: Optional[SensoryModality] = None
    strength: float = 1.0            # Attention intensity
    duration: float = 0.0            # How long focused
    started_at: float = field(default_factory=time.time)


# ============================================================================
# Attention Mechanism
# ============================================================================

class AttentionMechanism:
    """
    Manages attention allocation across sensory inputs.

    Implements:
    - Bottom-up attention (salient stimuli capture attention)
    - Top-down attention (goal-directed focus)
    - Attention switching
    - Habituation
    """

    def __init__(
        self,
        attention_capacity: float = 1.0,
        switch_cost: float = 0.1,
        habituation_rate: float = 0.05
    ):
        self.attention_capacity = attention_capacity
        self.switch_cost = switch_cost
        self.habituation_rate = habituation_rate

        # Current attention state
        self.focus_points: Dict[str, AttentionFocus] = {}
        self.attention_spent: float = 0.0

        # Habituation tracking
        self.habituation: Dict[str, float] = defaultdict(float)

        # History
        self.attention_history: deque = deque(maxlen=100)

    def allocate_attention(
        self,
        target: str,
        modality: SensoryModality,
        requested_strength: float
    ) -> float:
        """
        Allocate attention to a target.

        Returns actual attention allocated.
        """
        # Check capacity
        available = self.attention_capacity - self.attention_spent

        # Apply habituation penalty
        habituation_penalty = self.habituation.get(target, 0.0)
        effective_strength = requested_strength * (1 - habituation_penalty)

        # Allocate what's available
        allocated = min(available, effective_strength)

        if allocated > 0:
            if target in self.focus_points:
                # Update existing focus
                self.focus_points[target].strength = allocated
            else:
                # New focus point
                self.focus_points[target] = AttentionFocus(
                    target=target,
                    modality=modality,
                    strength=allocated
                )

            self.attention_spent += allocated
            self.attention_history.append({
                "target": target,
                "strength": allocated,
                "time": time.time()
            })

        return allocated

    def release_attention(self, target: str):
        """Release attention from a target"""
        if target in self.focus_points:
            focus = self.focus_points[target]
            self.attention_spent -= focus.strength
            del self.focus_points[target]

    def switch_focus(
        self,
        from_target: str,
        to_target: str,
        modality: SensoryModality
    ) -> bool:
        """Switch attention from one target to another"""
        if from_target in self.focus_points:
            old_strength = self.focus_points[from_target].strength
            self.release_attention(from_target)

            # Apply switch cost
            new_strength = old_strength * (1 - self.switch_cost)
            self.allocate_attention(to_target, modality, new_strength)
            return True
        return False

    def tick(self):
        """Process one attention cycle"""
        # Update habituation
        for target in list(self.focus_points.keys()):
            focus = self.focus_points[target]
            focus.duration = time.time() - focus.started_at

            # Increase habituation over time
            self.habituation[target] = min(
                0.9,
                self.habituation[target] + self.habituation_rate
            )

        # Decay habituation for unfocused targets
        for target in list(self.habituation.keys()):
            if target not in self.focus_points:
                self.habituation[target] = max(
                    0,
                    self.habituation[target] - self.habituation_rate * 0.5
                )

    def get_salience_boost(self, input: SensoryInput) -> float:
        """Calculate bottom-up salience boost for an input"""
        boost = 0.0

        # High intensity is salient
        if input.intensity > 0.8:
            boost += 0.3

        # Critical priority demands attention
        if input.priority == SensoryPriority.CRITICAL:
            boost += 0.5

        # Novelty (low habituation) is salient
        habituation = self.habituation.get(input.source, 0.0)
        boost += (1 - habituation) * 0.2

        # Pain/threat modality is salient
        if input.modality == SensoryModality.NOCICEPTION:
            boost += 0.4

        return min(1.0, boost)


# ============================================================================
# Sensory System
# ============================================================================

class SensorySystem:
    """
    Complete sensory system for digital organisms.

    Features:
    - Multiple sensor types and modalities
    - Sensory filtering and processing
    - Attention-based prioritization
    - Sensory integration
    - Sensory adaptation

    Example:
        sensory = SensorySystem()

        # Add sensors
        sensory.add_sensor(Sensor(name="data_in", sensor_type=SensorType.DATA_STREAM))

        # Process input
        input = sensory.sense(SensorType.DATA_STREAM, raw_data={"value": 42})

        # Get attended inputs
        attended = sensory.get_attended_inputs()
    """

    def __init__(
        self,
        attention_capacity: float = 1.0,
        buffer_size: int = 100
    ):
        # Sensors
        self.sensors: Dict[str, Sensor] = {}

        # Attention mechanism
        self.attention = AttentionMechanism(attention_capacity)

        # Input buffers by modality
        self.input_buffers: Dict[SensoryModality, deque] = {
            modality: deque(maxlen=buffer_size)
            for modality in SensoryModality
        }

        # Perception filters
        self.filters: List[PerceptionFilter] = []

        # Processing callbacks
        self._processors: Dict[SensorType, List[Callable]] = defaultdict(list)

        # Statistics
        self.total_inputs = 0
        self.attended_count = 0
        self.filtered_count = 0

        # Initialize default sensors
        self._initialize_default_sensors()

    def _initialize_default_sensors(self):
        """Initialize default sensor suite"""
        defaults = [
            Sensor(
                name="data_stream_sensor",
                sensor_type=SensorType.DATA_STREAM,
                modality=SensoryModality.EXTEROCEPTION,
                sensitivity=1.0
            ),
            Sensor(
                name="resource_sensor",
                sensor_type=SensorType.RESOURCE_MONITOR,
                modality=SensoryModality.INTEROCEPTION,
                sensitivity=0.8
            ),
            Sensor(
                name="time_sensor",
                sensor_type=SensorType.TIME_SENSE,
                modality=SensoryModality.TEMPORAL,
                sensitivity=1.0
            ),
            Sensor(
                name="self_monitor",
                sensor_type=SensorType.SELF_MONITOR,
                modality=SensoryModality.PROPRIOCEPTION,
                sensitivity=0.9
            ),
            Sensor(
                name="threat_sensor",
                sensor_type=SensorType.THREAT_DETECTOR,
                modality=SensoryModality.NOCICEPTION,
                sensitivity=1.2  # Extra sensitive
            ),
            Sensor(
                name="social_sensor",
                sensor_type=SensorType.SOCIAL_RECEPTOR,
                modality=SensoryModality.SOCIAL,
                sensitivity=0.8
            )
        ]

        for sensor in defaults:
            self.add_sensor(sensor)

    def add_sensor(self, sensor: Sensor):
        """Add a sensor to the system"""
        self.sensors[sensor.id] = sensor

    def remove_sensor(self, sensor_id: str):
        """Remove a sensor"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]

    def add_filter(self, filter: PerceptionFilter):
        """Add a perception filter"""
        self.filters.append(filter)

    def register_processor(
        self,
        sensor_type: SensorType,
        processor: Callable[[SensoryInput], Any]
    ):
        """Register a processor for a sensor type"""
        self._processors[sensor_type].append(processor)

    def sense(
        self,
        sensor_type: SensorType,
        raw_data: Any,
        source: str = "",
        priority: Optional[SensoryPriority] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SensoryInput]:
        """
        Process a sensory input.

        Returns SensoryInput if detected, None if below threshold.
        """
        # Find active sensor for this type
        sensor = self._get_sensor(sensor_type)
        if not sensor or not sensor.active:
            return None

        # Calculate signal intensity
        intensity = self._calculate_intensity(raw_data, sensor)

        # Apply threshold and noise filter
        if intensity < sensor.threshold:
            return None
        if intensity < sensor.noise_filter:
            self.filtered_count += 1
            return None

        # Apply sensory adaptation
        adapted_intensity = intensity * (1 - sensor.current_adaptation)

        # Create sensory input
        input = SensoryInput(
            sensor_type=sensor_type,
            modality=sensor.modality,
            priority=priority or self._determine_priority(adapted_intensity),
            raw_data=raw_data,
            intensity=adapted_intensity,
            source=source or sensor.name,
            metadata=metadata or {}
        )

        # Process through filters
        input = self._apply_filters(input)
        if input is None:
            self.filtered_count += 1
            return None

        # Process data
        input.processed_data = self._process_input(input)

        # Buffer the input
        self.input_buffers[sensor.modality].append(input)
        self.total_inputs += 1

        # Update sensor adaptation
        sensor.current_adaptation = min(
            0.5,
            sensor.current_adaptation + sensor.adaptation_rate
        )

        # Allocate attention based on salience
        salience = self.attention.get_salience_boost(input)
        if salience > 0.3:
            self.attention.allocate_attention(
                input.id,
                input.modality,
                salience
            )
            self.attended_count += 1

        return input

    def _get_sensor(self, sensor_type: SensorType) -> Optional[Sensor]:
        """Get an active sensor of the given type"""
        for sensor in self.sensors.values():
            if sensor.sensor_type == sensor_type and sensor.active:
                return sensor
        return None

    def _calculate_intensity(self, raw_data: Any, sensor: Sensor) -> float:
        """Calculate signal intensity"""
        if isinstance(raw_data, (int, float)):
            base_intensity = min(1.0, abs(raw_data) / 100.0)
        elif isinstance(raw_data, dict):
            # Use dict size/complexity as proxy
            base_intensity = min(1.0, len(raw_data) / 10.0)
        elif isinstance(raw_data, (list, tuple)):
            base_intensity = min(1.0, len(raw_data) / 50.0)
        elif isinstance(raw_data, str):
            base_intensity = min(1.0, len(raw_data) / 500.0)
        else:
            base_intensity = 0.5

        return base_intensity * sensor.sensitivity

    def _determine_priority(self, intensity: float) -> SensoryPriority:
        """Determine priority based on intensity"""
        if intensity > 0.9:
            return SensoryPriority.CRITICAL
        elif intensity > 0.7:
            return SensoryPriority.HIGH
        elif intensity > 0.4:
            return SensoryPriority.NORMAL
        elif intensity > 0.2:
            return SensoryPriority.LOW
        return SensoryPriority.MINIMAL

    def _apply_filters(self, input: SensoryInput) -> Optional[SensoryInput]:
        """Apply perception filters"""
        for filter in self.filters:
            if not filter.active:
                continue

            # Check modality match
            if filter.modalities and input.modality not in filter.modalities:
                continue

            # Check intensity range
            if not (filter.min_intensity <= input.intensity <= filter.max_intensity):
                return None

            # Apply transform
            if filter.transform:
                input.processed_data = filter.transform(input.raw_data)

        return input

    def _process_input(self, input: SensoryInput) -> Any:
        """Process input through registered processors"""
        processed = input.raw_data

        for processor in self._processors.get(input.sensor_type, []):
            try:
                processed = processor(input)
            except Exception:
                pass

        return processed

    def get_attended_inputs(
        self,
        modality: Optional[SensoryModality] = None
    ) -> List[SensoryInput]:
        """Get currently attended sensory inputs"""
        attended_ids = set(self.attention.focus_points.keys())
        inputs = []

        buffers = [self.input_buffers[modality]] if modality else self.input_buffers.values()

        for buffer in buffers:
            for input in buffer:
                if input.id in attended_ids:
                    inputs.append(input)

        return sorted(inputs, key=lambda i: i.priority.value)

    def get_recent_inputs(
        self,
        modality: Optional[SensoryModality] = None,
        count: int = 10
    ) -> List[SensoryInput]:
        """Get most recent sensory inputs"""
        if modality:
            return list(self.input_buffers[modality])[-count:]

        all_inputs = []
        for buffer in self.input_buffers.values():
            all_inputs.extend(list(buffer))

        all_inputs.sort(key=lambda i: i.timestamp, reverse=True)
        return all_inputs[:count]

    def tick(self):
        """Process one sensory cycle"""
        # Update attention
        self.attention.tick()

        # Decay sensor adaptation
        for sensor in self.sensors.values():
            sensor.current_adaptation = max(
                0,
                sensor.current_adaptation - sensor.adaptation_rate * 0.5
            )

    def focus_on(self, modality: SensoryModality, strength: float = 1.0):
        """Direct attention to a sensory modality"""
        # Find inputs in that modality
        for input in self.input_buffers[modality]:
            self.attention.allocate_attention(input.id, modality, strength)

    def get_sensory_summary(self) -> Dict[str, Any]:
        """Get summary of sensory state"""
        return {
            "active_sensors": len([s for s in self.sensors.values() if s.active]),
            "total_sensors": len(self.sensors),
            "total_inputs": self.total_inputs,
            "attended_count": self.attended_count,
            "filtered_count": self.filtered_count,
            "attention_spent": self.attention.attention_spent,
            "attention_capacity": self.attention.attention_capacity,
            "focus_points": len(self.attention.focus_points),
            "buffer_sizes": {
                m.value: len(b) for m, b in self.input_buffers.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Sensory Perception System Demo")
    print("=" * 50)

    # Create sensory system
    sensory = SensorySystem(attention_capacity=1.0)

    print(f"\n1. Initial state:")
    summary = sensory.get_sensory_summary()
    print(f"   Active sensors: {summary['active_sensors']}")
    print(f"   Attention capacity: {summary['attention_capacity']}")

    # Process various inputs
    print("\n2. Processing sensory inputs...")

    # Data stream input
    input1 = sensory.sense(
        SensorType.DATA_STREAM,
        raw_data={"message": "Hello world", "count": 42},
        source="external_api"
    )
    if input1:
        print(f"   Data input: intensity={input1.intensity:.2f}, "
              f"priority={input1.priority.value}")

    # Resource monitor input
    input2 = sensory.sense(
        SensorType.RESOURCE_MONITOR,
        raw_data={"cpu": 75, "memory": 60},
        source="system"
    )
    if input2:
        print(f"   Resource input: intensity={input2.intensity:.2f}")

    # Threat detection (high intensity)
    input3 = sensory.sense(
        SensorType.THREAT_DETECTOR,
        raw_data={"threat_level": 0.9, "type": "intrusion_attempt"},
        source="security",
        priority=SensoryPriority.CRITICAL
    )
    if input3:
        print(f"   Threat input: intensity={input3.intensity:.2f}, "
              f"priority={input3.priority.name}")

    # Social input
    input4 = sensory.sense(
        SensorType.SOCIAL_RECEPTOR,
        raw_data={"from_organism": "peer_001", "signal": "greeting"},
        source="network"
    )
    if input4:
        print(f"   Social input: intensity={input4.intensity:.2f}")

    # Get attended inputs
    print("\n3. Attended inputs:")
    attended = sensory.get_attended_inputs()
    for inp in attended:
        print(f"   - {inp.sensor_type.value}: {inp.source}")

    # Simulate processing cycles
    print("\n4. Running sensory cycles...")
    for cycle in range(10):
        sensory.tick()

        # Add more inputs periodically
        if cycle % 3 == 0:
            sensory.sense(
                SensorType.DATA_STREAM,
                raw_data={"cycle": cycle, "value": np.random.random()},
                source="periodic"
            )

    # Final summary
    print("\n5. Final summary:")
    summary = sensory.get_sensory_summary()
    for key, value in summary.items():
        if not isinstance(value, dict):
            print(f"   {key}: {value}")

    print("\n   Buffer sizes by modality:")
    for modality, size in summary["buffer_sizes"].items():
        if size > 0:
            print(f"      {modality}: {size}")
