"""
SYNAPSE-M: Proprioception

Internal body state sensing for digital organisms.
Implements body position awareness, joint state sensing,
and movement detection.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque

from .modality_types import (
    ExtendedModality, ProcessedModality, ModalityInput,
    get_modality_characteristics
)


# ============================================================================
# Body State Types
# ============================================================================

class BodyPartType(Enum):
    """Types of body parts for proprioception"""
    HEAD = auto()
    TORSO = auto()
    LEFT_ARM = auto()
    RIGHT_ARM = auto()
    LEFT_LEG = auto()
    RIGHT_LEG = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    LEFT_FOOT = auto()
    RIGHT_FOOT = auto()
    SPINE = auto()
    CORE = auto()


class MovementType(Enum):
    """Types of detected movement"""
    STATIONARY = auto()
    TRANSLATING = auto()
    ROTATING = auto()
    ACCELERATING = auto()
    DECELERATING = auto()
    OSCILLATING = auto()


class PostureType(Enum):
    """Body posture classifications"""
    NEUTRAL = auto()
    EXTENDED = auto()
    FLEXED = auto()
    ROTATED = auto()
    ASYMMETRIC = auto()


# ============================================================================
# Body State Data Structures
# ============================================================================

@dataclass
class JointState:
    """State of a single joint"""
    name: str
    angle: float                          # Current angle (radians)
    angular_velocity: float = 0.0         # Rate of change
    torque: float = 0.0                   # Applied torque
    min_angle: float = -np.pi             # Joint limits
    max_angle: float = np.pi
    stiffness: float = 1.0                # Joint stiffness
    damping: float = 0.1                  # Joint damping

    def is_at_limit(self) -> bool:
        """Check if joint is at limit"""
        tolerance = 0.01
        return (self.angle <= self.min_angle + tolerance or
                self.angle >= self.max_angle - tolerance)

    def get_normalized_position(self) -> float:
        """Get position normalized to [0, 1]"""
        range_val = self.max_angle - self.min_angle
        return (self.angle - self.min_angle) / range_val if range_val > 0 else 0.5


@dataclass
class BodyPartState:
    """State of a body part"""
    part_type: BodyPartType
    position: np.ndarray                  # 3D position
    orientation: np.ndarray               # Quaternion or euler angles
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joints: List[JointState] = field(default_factory=list)
    load: float = 0.0                     # External load/force
    fatigue: float = 0.0                  # Fatigue level (0-1)

    def get_speed(self) -> float:
        """Get speed magnitude"""
        return float(np.linalg.norm(self.velocity))


@dataclass
class BodySchema:
    """
    Complete body schema representation.
    Internal model of body structure and current state.
    """
    parts: Dict[BodyPartType, BodyPartState] = field(default_factory=dict)
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    overall_posture: PostureType = PostureType.NEUTRAL
    balance: float = 1.0                  # Balance stability (0-1)
    total_energy: float = 1.0             # Energy/fatigue state
    timestamp: float = field(default_factory=time.time)

    def get_part(self, part_type: BodyPartType) -> Optional[BodyPartState]:
        """Get state of a body part"""
        return self.parts.get(part_type)

    def is_moving(self, threshold: float = 0.01) -> bool:
        """Check if body is in motion"""
        for part in self.parts.values():
            if part.get_speed() > threshold:
                return True
        return False


@dataclass
class ProprioceptivePerception:
    """
    Proprioceptive perception output.
    """
    body_schema: BodySchema
    movement_type: MovementType
    position_confidence: float
    velocity_confidence: float
    features: np.ndarray                  # Encoded proprioceptive features
    anomalies: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Proprioceptive Sensors
# ============================================================================

@dataclass
class MuscleSpindleSensor:
    """
    Muscle spindle - senses muscle length and velocity.
    """
    name: str
    body_part: BodyPartType
    sensitivity: float = 1.0
    adaptation_rate: float = 0.01

    # State
    current_length: float = 0.5           # Normalized (0-1)
    length_velocity: float = 0.0

    def sense(self, actual_length: float, dt: float) -> Tuple[float, float]:
        """
        Sense muscle length and velocity.

        Returns:
            Tuple of (length, velocity)
        """
        # Compute velocity
        if dt > 0:
            velocity = (actual_length - self.current_length) / dt
        else:
            velocity = 0.0

        # Apply sensitivity
        sensed_length = actual_length * self.sensitivity
        sensed_velocity = velocity * self.sensitivity

        # Update state
        self.current_length = actual_length
        self.length_velocity = velocity

        return sensed_length, sensed_velocity


@dataclass
class GolgiTendonOrgan:
    """
    Golgi tendon organ - senses muscle tension/force.
    """
    name: str
    body_part: BodyPartType
    sensitivity: float = 1.0
    threshold: float = 0.1

    # State
    current_tension: float = 0.0

    def sense(self, tension: float) -> float:
        """Sense muscle tension"""
        if tension < self.threshold:
            return 0.0

        sensed = (tension - self.threshold) * self.sensitivity
        self.current_tension = sensed
        return sensed


@dataclass
class JointReceptor:
    """
    Joint receptor - senses joint angle and movement.
    """
    name: str
    joint: str
    sensitivity: float = 1.0

    # Response curve (simplified)
    peak_response_angle: float = 0.0      # Angle of maximum sensitivity

    def sense(self, angle: float) -> float:
        """Sense joint angle with position-dependent sensitivity"""
        # Gaussian sensitivity curve
        sigma = 0.5
        sensitivity = np.exp(-((angle - self.peak_response_angle) ** 2) / (2 * sigma ** 2))
        return angle * sensitivity * self.sensitivity


# ============================================================================
# Proprioception System
# ============================================================================

class ProprioceptionSystem:
    """
    Complete proprioception system for body awareness.

    Provides:
    - Body part position and orientation sensing
    - Movement and velocity detection
    - Joint state monitoring
    - Posture classification
    - Body schema maintenance
    """

    def __init__(
        self,
        feature_dim: int = 64,
        update_rate_hz: float = 100.0
    ):
        self.feature_dim = feature_dim
        self.update_rate_hz = update_rate_hz
        self.dt = 1.0 / update_rate_hz

        # Body schema
        self.body_schema = self._initialize_body_schema()

        # Sensors
        self.muscle_spindles: Dict[str, MuscleSpindleSensor] = {}
        self.golgi_organs: Dict[str, GolgiTendonOrgan] = {}
        self.joint_receptors: Dict[str, JointReceptor] = {}

        self._initialize_sensors()

        # History for temporal processing
        self._position_history: deque = deque(maxlen=50)
        self._velocity_history: deque = deque(maxlen=50)

        # Feature encoder
        np.random.seed(51)
        self._encoder = np.random.randn(100, feature_dim) * 0.1

    def _initialize_body_schema(self) -> BodySchema:
        """Initialize default body schema"""
        parts = {}

        for part_type in BodyPartType:
            # Default positions (would be configured per organism)
            position = np.array([0.0, 0.0, 0.0])
            orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

            parts[part_type] = BodyPartState(
                part_type=part_type,
                position=position,
                orientation=orientation
            )

        return BodySchema(parts=parts)

    def _initialize_sensors(self):
        """Initialize proprioceptive sensors"""
        for part_type in BodyPartType:
            name = part_type.name.lower()

            # Muscle spindles
            self.muscle_spindles[name] = MuscleSpindleSensor(
                name=f"{name}_spindle",
                body_part=part_type
            )

            # Golgi tendon organs
            self.golgi_organs[name] = GolgiTendonOrgan(
                name=f"{name}_gto",
                body_part=part_type
            )

    def update(
        self,
        body_state: Dict[BodyPartType, Dict[str, Any]]
    ) -> ProprioceptivePerception:
        """
        Update proprioception with new body state.

        Args:
            body_state: Dict mapping body parts to their current states
                Each state should have: position, orientation, velocity, load

        Returns:
            ProprioceptivePerception
        """
        anomalies = []

        # Update each body part
        for part_type, state in body_state.items():
            if part_type not in self.body_schema.parts:
                continue

            part = self.body_schema.parts[part_type]

            # Update position
            new_position = np.array(state.get("position", part.position))
            old_position = part.position

            # Compute velocity
            velocity = (new_position - old_position) / self.dt
            acceleration = (velocity - part.velocity) / self.dt

            part.position = new_position
            part.velocity = velocity
            part.acceleration = acceleration

            # Update orientation
            if "orientation" in state:
                part.orientation = np.array(state["orientation"])

            # Update load/tension
            if "load" in state:
                part.load = state["load"]

                # Sense with Golgi tendon organ
                name = part_type.name.lower()
                if name in self.golgi_organs:
                    sensed_tension = self.golgi_organs[name].sense(part.load)
                    if sensed_tension > 0.8:
                        anomalies.append(f"High tension in {name}")

            # Update joints
            if "joints" in state:
                for joint_data in state["joints"]:
                    joint = JointState(
                        name=joint_data.get("name", "unnamed"),
                        angle=joint_data.get("angle", 0.0),
                        angular_velocity=joint_data.get("angular_velocity", 0.0),
                        torque=joint_data.get("torque", 0.0)
                    )
                    part.joints.append(joint)

                    if joint.is_at_limit():
                        anomalies.append(f"Joint {joint.name} at limit")

        # Update center of mass
        self._update_center_of_mass()

        # Classify posture
        self._classify_posture()

        # Detect movement type
        movement_type = self._detect_movement_type()

        # Encode features
        features = self._encode_features()

        # Store in history
        self._position_history.append(self.body_schema.center_of_mass.copy())

        # Compute confidences
        position_conf = 0.9  # Would be based on sensor reliability
        velocity_conf = 0.85

        return ProprioceptivePerception(
            body_schema=self.body_schema,
            movement_type=movement_type,
            position_confidence=position_conf,
            velocity_confidence=velocity_conf,
            features=features,
            anomalies=anomalies
        )

    def _update_center_of_mass(self):
        """Update center of mass estimate"""
        positions = [part.position for part in self.body_schema.parts.values()]
        if positions:
            self.body_schema.center_of_mass = np.mean(positions, axis=0)

    def _classify_posture(self):
        """Classify current body posture"""
        # Simplified posture classification
        torso = self.body_schema.parts.get(BodyPartType.TORSO)
        if torso is None:
            return

        # Check for asymmetry
        left_arm = self.body_schema.parts.get(BodyPartType.LEFT_ARM)
        right_arm = self.body_schema.parts.get(BodyPartType.RIGHT_ARM)

        if left_arm and right_arm:
            arm_diff = np.linalg.norm(left_arm.position - right_arm.position)
            if arm_diff > 0.5:
                self.body_schema.overall_posture = PostureType.ASYMMETRIC
                return

        # Check torso orientation for flexion/extension
        if torso.orientation is not None:
            pitch = torso.orientation[1] if len(torso.orientation) > 1 else 0
            if pitch > 0.3:
                self.body_schema.overall_posture = PostureType.FLEXED
            elif pitch < -0.3:
                self.body_schema.overall_posture = PostureType.EXTENDED
            else:
                self.body_schema.overall_posture = PostureType.NEUTRAL

    def _detect_movement_type(self) -> MovementType:
        """Detect type of current movement"""
        if not self._position_history:
            return MovementType.STATIONARY

        # Check for motion
        total_velocity = np.zeros(3)
        for part in self.body_schema.parts.values():
            total_velocity += part.velocity

        avg_speed = np.linalg.norm(total_velocity) / len(self.body_schema.parts)

        if avg_speed < 0.01:
            return MovementType.STATIONARY

        # Check for acceleration
        total_accel = np.zeros(3)
        for part in self.body_schema.parts.values():
            total_accel += part.acceleration

        avg_accel = np.linalg.norm(total_accel) / len(self.body_schema.parts)

        if avg_accel > 0.1:
            if np.dot(total_velocity, total_accel) > 0:
                return MovementType.ACCELERATING
            else:
                return MovementType.DECELERATING

        # Check for rotation
        angular_velocities = []
        for part in self.body_schema.parts.values():
            for joint in part.joints:
                angular_velocities.append(abs(joint.angular_velocity))

        if angular_velocities and np.mean(angular_velocities) > 0.1:
            return MovementType.ROTATING

        return MovementType.TRANSLATING

    def _encode_features(self) -> np.ndarray:
        """Encode body state into feature vector"""
        raw_features = []

        # Encode each body part
        for part in self.body_schema.parts.values():
            raw_features.extend(part.position)
            raw_features.extend(part.velocity)
            raw_features.append(part.load)
            raw_features.append(part.fatigue)

        # Add global features
        raw_features.extend(self.body_schema.center_of_mass)
        raw_features.append(self.body_schema.balance)
        raw_features.append(self.body_schema.total_energy)

        # Pad or truncate to expected size
        raw_array = np.array(raw_features[:100])
        if len(raw_array) < 100:
            raw_array = np.pad(raw_array, (0, 100 - len(raw_array)))

        # Encode
        features = np.tanh(raw_array @ self._encoder)
        return features

    def get_body_part_position(self, part_type: BodyPartType) -> Optional[np.ndarray]:
        """Get position of a body part"""
        part = self.body_schema.parts.get(part_type)
        return part.position.copy() if part else None

    def get_movement_velocity(self) -> np.ndarray:
        """Get overall movement velocity"""
        total = np.zeros(3)
        for part in self.body_schema.parts.values():
            total += part.velocity
        return total / len(self.body_schema.parts) if self.body_schema.parts else total

    def to_processed_modality(
        self,
        perception: ProprioceptivePerception
    ) -> ProcessedModality:
        """Convert to standard ProcessedModality format"""
        input_data = ModalityInput(
            modality=ExtendedModality.PROPRIOCEPTION,
            raw_data={
                "center_of_mass": self.body_schema.center_of_mass.tolist(),
                "posture": self.body_schema.overall_posture.name,
                "balance": self.body_schema.balance
            },
            intensity=self.body_schema.total_energy
        )

        chars = get_modality_characteristics(ExtendedModality.PROPRIOCEPTION)

        return ProcessedModality(
            modality=ExtendedModality.PROPRIOCEPTION,
            raw_input=input_data,
            features=perception.features,
            semantic_embedding=perception.features[:chars.semantic_dim],
            salience=0.5 if perception.movement_type == MovementType.STATIONARY else 0.7,
            confidence=perception.position_confidence
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get proprioception statistics"""
        return {
            "body_parts": len(self.body_schema.parts),
            "sensors": {
                "muscle_spindles": len(self.muscle_spindles),
                "golgi_organs": len(self.golgi_organs),
                "joint_receptors": len(self.joint_receptors)
            },
            "current_posture": self.body_schema.overall_posture.name,
            "is_moving": self.body_schema.is_moving(),
            "balance": self.body_schema.balance,
            "center_of_mass": self.body_schema.center_of_mass.tolist()
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Proprioception Demo")
    print("=" * 50)

    # Create proprioception system
    proprio = ProprioceptionSystem()

    np.random.seed(42)

    print("\n1. Initial body schema:")
    stats = proprio.get_statistics()
    print(f"   Body parts: {stats['body_parts']}")
    print(f"   Posture: {stats['current_posture']}")

    print("\n2. Updating with movement...")
    for i in range(5):
        # Simulate movement
        body_state = {}
        for part_type in [BodyPartType.LEFT_ARM, BodyPartType.RIGHT_ARM,
                         BodyPartType.TORSO, BodyPartType.HEAD]:
            body_state[part_type] = {
                "position": np.random.randn(3) * 0.1 + i * 0.05,
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                "load": np.random.uniform(0.0, 0.5)
            }

        perception = proprio.update(body_state)

        print(f"   Step {i}: movement={perception.movement_type.name}, "
              f"anomalies={len(perception.anomalies)}")

    print("\n3. Final state:")
    stats = proprio.get_statistics()
    print(f"   Posture: {stats['current_posture']}")
    print(f"   Is moving: {stats['is_moving']}")
    print(f"   Balance: {stats['balance']:.2f}")
    print(f"   Center of mass: {stats['center_of_mass']}")

    print("\n4. Feature encoding:")
    perception = proprio.update({})
    print(f"   Feature shape: {perception.features.shape}")
    print(f"   Feature range: [{perception.features.min():.3f}, {perception.features.max():.3f}]")

    print("\n5. Converting to ProcessedModality:")
    processed = proprio.to_processed_modality(perception)
    print(f"   Modality: {processed.modality.value}")
    print(f"   Salience: {processed.salience:.2f}")
    print(f"   Confidence: {processed.confidence:.2f}")
