"""
SYNAPSE-M: Sensory Memory

Iconic (visual) and echoic (auditory) sensory memory buffers.
Implements ultra-short-term sensory storage with decay and interference.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod

from .modality_types import (
    ExtendedModality, ProcessedModality, ModalityCharacteristics,
    get_modality_characteristics, ModalityDomain
)


# ============================================================================
# Memory Types
# ============================================================================

class MemoryType(Enum):
    """Types of sensory memory"""
    ICONIC = auto()       # Visual sensory memory (~100-500ms)
    ECHOIC = auto()       # Auditory sensory memory (~2-4s)
    HAPTIC = auto()       # Tactile sensory memory (~2s)
    OLFACTORY = auto()    # Olfactory sensory memory (longer)
    GENERIC = auto()      # Generic sensory buffer


class DecayModel(Enum):
    """Models of memory decay"""
    EXPONENTIAL = auto()  # e^(-t/tau)
    LINEAR = auto()       # 1 - t/tau
    POWER_LAW = auto()    # t^(-alpha)


# ============================================================================
# Memory Data Structures
# ============================================================================

@dataclass
class MemoryTrace:
    """
    A memory trace stored in sensory memory.
    """
    id: str = field(default_factory=lambda: str(id(object())))
    modality: ExtendedModality = ExtendedModality.VISION
    content: ProcessedModality = None

    # Timing
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Strength
    initial_strength: float = 1.0
    current_strength: float = 1.0
    decay_rate: float = 0.1  # Per second

    # Access
    access_count: int = 0
    attended: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_age_ms(self) -> float:
        """Get age in milliseconds"""
        return (time.time() - self.created_at) * 1000

    def decay(self, dt: float = None) -> float:
        """Apply decay and return new strength"""
        if dt is None:
            dt = time.time() - self.last_accessed

        self.current_strength = self.initial_strength * np.exp(-self.decay_rate * dt)
        return self.current_strength

    def access(self) -> float:
        """Access the memory (refreshes slightly)"""
        self.access_count += 1
        self.last_accessed = time.time()
        # Accessing can slow decay slightly
        self.current_strength = min(
            self.initial_strength,
            self.current_strength * 1.05
        )
        return self.current_strength

    def is_available(self, threshold: float = 0.1) -> bool:
        """Check if memory is still available"""
        self.decay()
        return self.current_strength > threshold


@dataclass
class SensorySnapshot:
    """
    A snapshot of the entire sensory field at a moment.
    """
    timestamp: float = field(default_factory=time.time)
    traces: Dict[ExtendedModality, MemoryTrace] = field(default_factory=dict)
    total_strength: float = 0.0
    modality_count: int = 0

    def get_trace(self, modality: ExtendedModality) -> Optional[MemoryTrace]:
        """Get trace for a modality"""
        return self.traces.get(modality)


# ============================================================================
# Sensory Memory Buffers
# ============================================================================

T = TypeVar('T')


class SensoryMemoryBuffer(ABC):
    """
    Abstract base for sensory memory buffers.
    """

    def __init__(
        self,
        modality: ExtendedModality,
        capacity: int = 10,
        decay_rate: float = 0.1,
        persistence_ms: float = 500.0
    ):
        self.modality = modality
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.persistence_ms = persistence_ms

        self._buffer: deque = deque(maxlen=capacity)
        self._access_threshold = 0.1

    @abstractmethod
    def store(self, perception: ProcessedModality) -> MemoryTrace:
        """Store a perception in memory"""
        pass

    @abstractmethod
    def retrieve(self, index: int = -1) -> Optional[MemoryTrace]:
        """Retrieve a memory trace"""
        pass

    def get_available_traces(self) -> List[MemoryTrace]:
        """Get all currently available traces"""
        available = []
        for trace in self._buffer:
            if trace.is_available(self._access_threshold):
                available.append(trace)
        return available

    def decay_all(self):
        """Apply decay to all traces"""
        for trace in self._buffer:
            trace.decay()

    def clear_weak(self, threshold: float = None):
        """Clear traces below threshold"""
        threshold = threshold or self._access_threshold
        self._buffer = deque(
            [t for t in self._buffer if t.is_available(threshold)],
            maxlen=self.capacity
        )

    def size(self) -> int:
        """Current number of traces"""
        return len(self._buffer)


class IconicMemory(SensoryMemoryBuffer):
    """
    Iconic (visual) sensory memory.

    Characteristics:
    - Very short duration (~100-500ms)
    - High capacity (full visual field)
    - Fast decay
    - Can be partially read out before decay
    """

    def __init__(
        self,
        capacity: int = 20,
        decay_rate: float = 5.0,  # Fast decay
        persistence_ms: float = 250.0
    ):
        super().__init__(
            modality=ExtendedModality.VISION,
            capacity=capacity,
            decay_rate=decay_rate,
            persistence_ms=persistence_ms
        )

        # Iconic memory has high initial resolution
        self._resolution_decay_rate = 0.3

    def store(self, perception: ProcessedModality) -> MemoryTrace:
        """Store visual perception"""
        trace = MemoryTrace(
            modality=ExtendedModality.VISION,
            content=perception,
            initial_strength=1.0,
            current_strength=1.0,
            decay_rate=self.decay_rate,
            metadata={
                "iconic_resolution": 1.0,  # Full resolution initially
                "visual_features": perception.features[:10].tolist() if len(perception.features) > 0 else []
            }
        )
        self._buffer.append(trace)
        return trace

    def retrieve(self, index: int = -1) -> Optional[MemoryTrace]:
        """Retrieve with partial readout"""
        if not self._buffer or abs(index) > len(self._buffer):
            return None

        trace = self._buffer[index]

        # Iconic memory: reading out causes interference
        trace.access()

        # Resolution degrades on access
        resolution = trace.metadata.get("iconic_resolution", 1.0)
        trace.metadata["iconic_resolution"] = resolution * 0.8

        return trace

    def get_full_field_snapshot(self) -> List[MemoryTrace]:
        """Get snapshot of full visual field (if still available)"""
        current_time = time.time()
        snapshots = []

        for trace in self._buffer:
            age_ms = trace.get_age_ms()
            if age_ms < self.persistence_ms:
                # Still available at some resolution
                resolution = trace.metadata.get("iconic_resolution", 1.0)
                resolution *= np.exp(-self._resolution_decay_rate * age_ms / 1000.0)
                trace.metadata["current_resolution"] = resolution
                snapshots.append(trace)

        return snapshots


class EchoicMemory(SensoryMemoryBuffer):
    """
    Echoic (auditory) sensory memory.

    Characteristics:
    - Longer duration (~2-4 seconds)
    - Temporal stream representation
    - Slower decay than iconic
    - Preserves temporal order
    """

    def __init__(
        self,
        capacity: int = 50,
        decay_rate: float = 0.5,  # Slower than iconic
        persistence_ms: float = 3000.0
    ):
        super().__init__(
            modality=ExtendedModality.AUDITION,
            capacity=capacity,
            decay_rate=decay_rate,
            persistence_ms=persistence_ms
        )

        # Temporal stream for preserving order
        self._temporal_stream: List[Tuple[float, MemoryTrace]] = []

    def store(self, perception: ProcessedModality) -> MemoryTrace:
        """Store auditory perception"""
        trace = MemoryTrace(
            modality=ExtendedModality.AUDITION,
            content=perception,
            initial_strength=1.0,
            current_strength=1.0,
            decay_rate=self.decay_rate,
            metadata={
                "temporal_position": len(self._temporal_stream),
                "duration_ms": 50.0,  # Assumed chunk duration
                "audio_features": perception.features[:5].tolist() if len(perception.features) > 0 else []
            }
        )

        self._buffer.append(trace)
        self._temporal_stream.append((time.time(), trace))

        # Clean old entries from stream
        self._clean_temporal_stream()

        return trace

    def _clean_temporal_stream(self):
        """Remove expired entries from temporal stream"""
        current_time = time.time()
        cutoff = current_time - self.persistence_ms / 1000.0

        self._temporal_stream = [
            (t, trace) for t, trace in self._temporal_stream
            if t > cutoff
        ]

    def retrieve(self, index: int = -1) -> Optional[MemoryTrace]:
        """Retrieve auditory trace"""
        if not self._buffer or abs(index) > len(self._buffer):
            return None

        trace = self._buffer[index]
        trace.access()
        return trace

    def get_temporal_window(
        self,
        window_ms: float = 1000.0
    ) -> List[MemoryTrace]:
        """Get traces within a temporal window"""
        self._clean_temporal_stream()

        current_time = time.time()
        window_start = current_time - window_ms / 1000.0

        traces = [
            trace for t, trace in self._temporal_stream
            if t >= window_start and trace.is_available()
        ]

        return traces

    def replay(self, duration_ms: float = None) -> List[MemoryTrace]:
        """
        Replay echoic memory in temporal order.

        Echoic memory allows "replay" of recent sounds.
        """
        self._clean_temporal_stream()

        if duration_ms:
            return self.get_temporal_window(duration_ms)

        return [trace for _, trace in self._temporal_stream if trace.is_available()]


class HapticMemory(SensoryMemoryBuffer):
    """
    Haptic (tactile) sensory memory.

    Characteristics:
    - Duration ~2 seconds
    - Spatial and temporal patterns
    - Interacts with proprioception
    """

    def __init__(
        self,
        capacity: int = 30,
        decay_rate: float = 1.0,
        persistence_ms: float = 2000.0
    ):
        super().__init__(
            modality=ExtendedModality.TACTILE,
            capacity=capacity,
            decay_rate=decay_rate,
            persistence_ms=persistence_ms
        )

        # Spatial map for touch location
        self._spatial_map: Dict[str, MemoryTrace] = {}

    def store(self, perception: ProcessedModality) -> MemoryTrace:
        """Store tactile perception"""
        # Extract location from perception if available
        location = perception.metadata.get("location", "unknown")

        trace = MemoryTrace(
            modality=ExtendedModality.TACTILE,
            content=perception,
            initial_strength=1.0,
            current_strength=1.0,
            decay_rate=self.decay_rate,
            metadata={
                "touch_location": location,
                "pressure": perception.raw_input.intensity,
                "tactile_features": perception.features[:5].tolist() if len(perception.features) > 0 else []
            }
        )

        self._buffer.append(trace)
        self._spatial_map[location] = trace

        return trace

    def retrieve(self, index: int = -1) -> Optional[MemoryTrace]:
        """Retrieve haptic trace"""
        if not self._buffer or abs(index) > len(self._buffer):
            return None

        trace = self._buffer[index]
        trace.access()
        return trace

    def get_by_location(self, location: str) -> Optional[MemoryTrace]:
        """Get trace by touch location"""
        trace = self._spatial_map.get(location)
        if trace and trace.is_available():
            trace.access()
            return trace
        return None


# ============================================================================
# Unified Sensory Memory System
# ============================================================================

class SensoryMemorySystem:
    """
    Unified system managing all sensory memory buffers.
    """

    def __init__(self):
        # Create specialized buffers
        self.iconic = IconicMemory()
        self.echoic = EchoicMemory()
        self.haptic = HapticMemory()

        # Generic buffers for other modalities
        self._generic_buffers: Dict[ExtendedModality, SensoryMemoryBuffer] = {}

        # Mapping of modalities to buffers
        self._modality_map: Dict[ExtendedModality, SensoryMemoryBuffer] = {
            ExtendedModality.VISION: self.iconic,
            ExtendedModality.AUDITION: self.echoic,
            ExtendedModality.TACTILE: self.haptic
        }

        # Snapshots for multi-modal moments
        self._snapshots: deque = deque(maxlen=20)

    def _get_buffer(self, modality: ExtendedModality) -> SensoryMemoryBuffer:
        """Get or create buffer for modality"""
        if modality in self._modality_map:
            return self._modality_map[modality]

        if modality not in self._generic_buffers:
            chars = get_modality_characteristics(modality)
            self._generic_buffers[modality] = GenericSensoryBuffer(
                modality=modality,
                persistence_ms=chars.persistence_ms
            )
            self._modality_map[modality] = self._generic_buffers[modality]

        return self._generic_buffers[modality]

    def store(self, perception: ProcessedModality) -> MemoryTrace:
        """Store perception in appropriate buffer"""
        buffer = self._get_buffer(perception.modality)
        return buffer.store(perception)

    def store_multimodal(
        self,
        perceptions: Dict[ExtendedModality, ProcessedModality]
    ) -> SensorySnapshot:
        """Store multi-modal perceptions as snapshot"""
        traces = {}
        total_strength = 0.0

        for modality, perception in perceptions.items():
            trace = self.store(perception)
            traces[modality] = trace
            total_strength += trace.current_strength

        snapshot = SensorySnapshot(
            traces=traces,
            total_strength=total_strength,
            modality_count=len(traces)
        )
        self._snapshots.append(snapshot)

        return snapshot

    def retrieve(
        self,
        modality: ExtendedModality,
        index: int = -1
    ) -> Optional[MemoryTrace]:
        """Retrieve from specific modality buffer"""
        buffer = self._get_buffer(modality)
        return buffer.retrieve(index)

    def get_recent(
        self,
        modality: ExtendedModality,
        count: int = 5
    ) -> List[MemoryTrace]:
        """Get recent traces from a modality"""
        buffer = self._get_buffer(modality)
        available = buffer.get_available_traces()
        return available[-count:] if len(available) >= count else available

    def get_snapshot(self, index: int = -1) -> Optional[SensorySnapshot]:
        """Get a multi-modal snapshot"""
        if not self._snapshots or abs(index) > len(self._snapshots):
            return None
        return self._snapshots[index]

    def decay_all(self):
        """Apply decay to all buffers"""
        for buffer in self._modality_map.values():
            buffer.decay_all()

    def clear_weak(self, threshold: float = 0.1):
        """Clear weak traces from all buffers"""
        for buffer in self._modality_map.values():
            buffer.clear_weak(threshold)

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            "buffers": {},
            "total_traces": 0,
            "snapshots": len(self._snapshots)
        }

        for modality, buffer in self._modality_map.items():
            available = buffer.get_available_traces()
            stats["buffers"][modality.value] = {
                "size": buffer.size(),
                "available": len(available),
                "persistence_ms": buffer.persistence_ms
            }
            stats["total_traces"] += buffer.size()

        return stats


class GenericSensoryBuffer(SensoryMemoryBuffer):
    """Generic buffer for modalities without specialized handling"""

    def store(self, perception: ProcessedModality) -> MemoryTrace:
        trace = MemoryTrace(
            modality=self.modality,
            content=perception,
            initial_strength=1.0,
            decay_rate=self.decay_rate
        )
        self._buffer.append(trace)
        return trace

    def retrieve(self, index: int = -1) -> Optional[MemoryTrace]:
        if not self._buffer or abs(index) > len(self._buffer):
            return None
        trace = self._buffer[index]
        trace.access()
        return trace


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from typing import Tuple
    print("SYNAPSE-M: Sensory Memory Demo")
    print("=" * 50)

    from .modality_types import ModalityInput

    # Create memory system
    memory = SensoryMemorySystem()

    np.random.seed(42)

    print("\n1. Storing visual perceptions (iconic memory)...")
    for i in range(5):
        visual_input = ModalityInput(
            modality=ExtendedModality.VISION,
            raw_data=np.random.randn(256),
            intensity=0.8
        )
        perception = ProcessedModality(
            modality=ExtendedModality.VISION,
            raw_input=visual_input,
            features=np.random.randn(256),
            semantic_embedding=np.random.randn(128),
            salience=0.7,
            confidence=0.9
        )
        trace = memory.store(perception)
        print(f"   Stored trace {i}: strength={trace.current_strength:.2f}")

    print("\n2. Retrieving iconic memory...")
    trace = memory.retrieve(ExtendedModality.VISION, -1)
    if trace:
        print(f"   Retrieved: age={trace.get_age_ms():.1f}ms, strength={trace.current_strength:.2f}")
        resolution = trace.metadata.get("current_resolution", trace.metadata.get("iconic_resolution", 1.0))
        print(f"   Resolution after access: {resolution:.2f}")

    print("\n3. Storing auditory perceptions (echoic memory)...")
    for i in range(10):
        audio_input = ModalityInput(
            modality=ExtendedModality.AUDITION,
            raw_data=np.random.randn(128),
            intensity=0.7
        )
        perception = ProcessedModality(
            modality=ExtendedModality.AUDITION,
            raw_input=audio_input,
            features=np.random.randn(128),
            semantic_embedding=np.random.randn(64),
            salience=0.6,
            confidence=0.85
        )
        memory.store(perception)

    print("   Stored 10 auditory traces")

    print("\n4. Replaying echoic memory...")
    replay = memory.echoic.replay()
    print(f"   Replay contains {len(replay)} traces")

    print("\n5. Memory system statistics:")
    stats = memory.get_statistics()
    print(f"   Total traces: {stats['total_traces']}")
    print(f"   Snapshots: {stats['snapshots']}")
    for modality, mstats in stats['buffers'].items():
        print(f"   {modality}: size={mstats['size']}, available={mstats['available']}")

    print("\n6. Decay simulation...")
    import time as time_module
    time_module.sleep(0.1)  # Wait 100ms
    memory.decay_all()
    trace = memory.retrieve(ExtendedModality.VISION, 0)
    if trace:
        print(f"   After 100ms: strength={trace.current_strength:.2f}")
