"""
Simulation Recorder for NPCPU

Records simulation states for later replay with:
- Delta compression (only store changes)
- Keyframe intervals for efficient seeking
- Event logging with timestamps
- Configurable detail levels

Example:
    from persistence import SimulationRecorder, DetailLevel
    from core.simple_organism import SimpleOrganism, SimplePopulation

    # Create recorder
    recorder = SimulationRecorder(
        keyframe_interval=100,
        detail_level=DetailLevel.STANDARD
    )

    # Start recording
    recorder.start_recording("my_simulation")

    # Record each tick
    population = SimplePopulation("test")
    for tick in range(1000):
        population.tick()
        recorder.record_tick(
            tick=tick,
            organisms={org.id: org.to_dict() for org in population.organisms.values()},
            world_state={"resources": 100}
        )

    # Stop and save
    recorder.stop_recording()
    recorder.save("simulation_recording.npcpu")
"""

from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import json
import gzip
import hashlib
import copy
from pathlib import Path

from core.events import Event, get_event_bus


# =============================================================================
# Enums and Configuration
# =============================================================================

class DetailLevel(Enum):
    """Level of detail for recordings"""
    MINIMAL = auto()    # Only organism IDs and positions
    STANDARD = auto()   # Standard state (subsystems, traits)
    FULL = auto()       # Full state including all metadata
    DEBUG = auto()      # Everything including internal state


@dataclass
class RecordingConfig:
    """Configuration for simulation recording"""
    keyframe_interval: int = 100      # Full snapshot every N ticks
    detail_level: DetailLevel = DetailLevel.STANDARD
    compress: bool = True             # Use gzip compression
    record_events: bool = True        # Also record events
    max_events_per_tick: int = 1000   # Limit events per tick

    # Filtering
    organism_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    event_filter: Optional[Callable[[Event], bool]] = None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Keyframe:
    """Full state snapshot at a point in time"""
    tick: int
    timestamp: datetime
    organisms: Dict[str, Dict[str, Any]]
    world_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp.isoformat(),
            "organisms": self.organisms,
            "world_state": self.world_state,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Keyframe':
        return cls(
            tick=data["tick"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            organisms=data.get("organisms", {}),
            world_state=data.get("world_state", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class Delta:
    """Changes between ticks"""
    tick: int
    timestamp: datetime

    # Organism changes
    organisms_added: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    organisms_removed: Set[str] = field(default_factory=set)
    organisms_modified: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # World changes
    world_changes: Dict[str, Any] = field(default_factory=dict)

    # Events that occurred
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp.isoformat(),
            "organisms_added": self.organisms_added,
            "organisms_removed": list(self.organisms_removed),
            "organisms_modified": self.organisms_modified,
            "world_changes": self.world_changes,
            "events": self.events
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Delta':
        return cls(
            tick=data["tick"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            organisms_added=data.get("organisms_added", {}),
            organisms_removed=set(data.get("organisms_removed", [])),
            organisms_modified=data.get("organisms_modified", {}),
            world_changes=data.get("world_changes", {}),
            events=data.get("events", [])
        )


@dataclass
class Recording:
    """Complete recording of a simulation"""
    name: str
    created_at: datetime
    config: RecordingConfig

    # Data
    keyframes: List[Keyframe] = field(default_factory=list)
    deltas: List[Delta] = field(default_factory=list)

    # Metadata
    total_ticks: int = 0
    start_tick: int = 0
    end_tick: int = 0
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "config": {
                "keyframe_interval": self.config.keyframe_interval,
                "detail_level": self.config.detail_level.name,
                "compress": self.config.compress,
                "record_events": self.config.record_events
            },
            "keyframes": [kf.to_dict() for kf in self.keyframes],
            "deltas": [d.to_dict() for d in self.deltas],
            "total_ticks": self.total_ticks,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recording':
        config_data = data.get("config", {})
        config = RecordingConfig(
            keyframe_interval=config_data.get("keyframe_interval", 100),
            detail_level=DetailLevel[config_data.get("detail_level", "STANDARD")],
            compress=config_data.get("compress", True),
            record_events=config_data.get("record_events", True)
        )

        recording = cls(
            name=data.get("name", "unnamed"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            config=config,
            total_ticks=data.get("total_ticks", 0),
            start_tick=data.get("start_tick", 0),
            end_tick=data.get("end_tick", 0),
            version=data.get("version", "1.0.0")
        )

        recording.keyframes = [Keyframe.from_dict(kf) for kf in data.get("keyframes", [])]
        recording.deltas = [Delta.from_dict(d) for d in data.get("deltas", [])]

        return recording


# =============================================================================
# Simulation Recorder
# =============================================================================

class SimulationRecorder:
    """
    Records simulation for later replay.

    Features:
    - Delta compression (only store changes)
    - Keyframe intervals
    - Event logging with timestamps
    - Configurable detail levels

    Example:
        recorder = SimulationRecorder(keyframe_interval=100)
        recorder.start_recording("my_sim")

        for tick in range(1000):
            simulation.tick()
            recorder.record_tick(
                tick=tick,
                organisms=get_organism_states(),
                world_state=get_world_state()
            )

        recorder.stop_recording()
        recorder.save("recording.npcpu")
    """

    def __init__(
        self,
        keyframe_interval: int = 100,
        detail_level: DetailLevel = DetailLevel.STANDARD,
        compress: bool = True,
        record_events: bool = True
    ):
        self.config = RecordingConfig(
            keyframe_interval=keyframe_interval,
            detail_level=detail_level,
            compress=compress,
            record_events=record_events
        )

        self._recording: Optional[Recording] = None
        self._is_recording = False
        self._last_state: Dict[str, Dict[str, Any]] = {}
        self._last_world_state: Dict[str, Any] = {}
        self._event_buffer: List[Dict[str, Any]] = []
        self._event_handler_id: Optional[str] = None

    # -------------------------------------------------------------------------
    # Recording Control
    # -------------------------------------------------------------------------

    def start_recording(self, name: str = "simulation") -> None:
        """Start a new recording session"""
        if self._is_recording:
            raise RuntimeError("Already recording. Stop current recording first.")

        self._recording = Recording(
            name=name,
            created_at=datetime.now(),
            config=self.config
        )

        self._is_recording = True
        self._last_state = {}
        self._last_world_state = {}
        self._event_buffer = []

        # Subscribe to events if enabled
        if self.config.record_events:
            bus = get_event_bus()
            self._event_handler_id = bus.subscribe("*", self._on_event)

        # Emit recording started event
        bus = get_event_bus()
        bus.emit("recording.started", {
            "name": name,
            "config": {
                "keyframe_interval": self.config.keyframe_interval,
                "detail_level": self.config.detail_level.name
            }
        })

    def stop_recording(self) -> Optional[Recording]:
        """Stop the current recording session"""
        if not self._is_recording:
            return None

        self._is_recording = False

        # Unsubscribe from events
        if self._event_handler_id:
            bus = get_event_bus()
            bus.unsubscribe(self._event_handler_id)
            self._event_handler_id = None

        if self._recording:
            # Set end tick
            if self._recording.deltas:
                self._recording.end_tick = self._recording.deltas[-1].tick
            elif self._recording.keyframes:
                self._recording.end_tick = self._recording.keyframes[-1].tick

            # Emit recording stopped event
            bus = get_event_bus()
            bus.emit("recording.stopped", {
                "name": self._recording.name,
                "total_ticks": self._recording.total_ticks,
                "keyframes": len(self._recording.keyframes),
                "deltas": len(self._recording.deltas)
            })

        return self._recording

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_recording(self) -> Optional[Recording]:
        return self._recording

    # -------------------------------------------------------------------------
    # Recording Methods
    # -------------------------------------------------------------------------

    def record_tick(
        self,
        tick: int,
        organisms: Dict[str, Dict[str, Any]],
        world_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Capture state changes for a tick.

        Args:
            tick: Current tick number
            organisms: Dictionary mapping organism ID to state dict
            world_state: Optional world state dictionary
        """
        if not self._is_recording or not self._recording:
            return

        world_state = world_state or {}

        # Apply detail level filtering
        filtered_organisms = self._filter_organism_states(organisms)

        # First tick or keyframe interval - create keyframe
        if tick == 0 or tick % self.config.keyframe_interval == 0:
            self.create_keyframe(tick, filtered_organisms, world_state)
        else:
            # Create delta
            self._create_delta(tick, filtered_organisms, world_state)

        # Update tracking state
        self._last_state = copy.deepcopy(filtered_organisms)
        self._last_world_state = copy.deepcopy(world_state)
        self._recording.total_ticks = tick + 1

        if tick == 0:
            self._recording.start_tick = tick

    def create_keyframe(
        self,
        tick: int,
        organisms: Dict[str, Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> Keyframe:
        """Create a full state snapshot"""
        keyframe = Keyframe(
            tick=tick,
            timestamp=datetime.now(),
            organisms=copy.deepcopy(organisms),
            world_state=copy.deepcopy(world_state),
            metadata={
                "organism_count": len(organisms),
                "detail_level": self.config.detail_level.name
            }
        )

        if self._recording:
            self._recording.keyframes.append(keyframe)

        # Also update last state for delta calculation
        self._last_state = copy.deepcopy(organisms)
        self._last_world_state = copy.deepcopy(world_state)

        return keyframe

    def _create_delta(
        self,
        tick: int,
        organisms: Dict[str, Dict[str, Any]],
        world_state: Dict[str, Any]
    ) -> Delta:
        """Create a delta recording changes from last state"""
        delta = Delta(
            tick=tick,
            timestamp=datetime.now()
        )

        current_ids = set(organisms.keys())
        previous_ids = set(self._last_state.keys())

        # Find added organisms
        added_ids = current_ids - previous_ids
        for org_id in added_ids:
            delta.organisms_added[org_id] = organisms[org_id]

        # Find removed organisms
        delta.organisms_removed = previous_ids - current_ids

        # Find modified organisms
        common_ids = current_ids & previous_ids
        for org_id in common_ids:
            diff = self._compute_diff(self._last_state[org_id], organisms[org_id])
            if diff:
                delta.organisms_modified[org_id] = diff

        # World state changes
        delta.world_changes = self._compute_diff(self._last_world_state, world_state)

        # Add buffered events
        if self._event_buffer:
            delta.events = self._event_buffer[:self.config.max_events_per_tick]
            self._event_buffer = []

        if self._recording:
            self._recording.deltas.append(delta)

        return delta

    def _compute_diff(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute differences between two state dictionaries"""
        diff = {}

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)

            if old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    # Recursively diff nested dicts
                    nested_diff = self._compute_diff(old_val, new_val)
                    if nested_diff:
                        diff[key] = nested_diff
                else:
                    diff[key] = new_val

        return diff

    def _filter_organism_states(
        self,
        organisms: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter organism states based on detail level"""
        if self.config.detail_level == DetailLevel.FULL or \
           self.config.detail_level == DetailLevel.DEBUG:
            return organisms

        filtered = {}
        for org_id, state in organisms.items():
            # Apply custom filter if provided
            if self.config.organism_filter and not self.config.organism_filter(state):
                continue

            if self.config.detail_level == DetailLevel.MINIMAL:
                # Only keep essential fields
                filtered[org_id] = {
                    "id": state.get("id"),
                    "name": state.get("name"),
                    "phase": state.get("phase"),
                    "age": state.get("age"),
                    "alive": state.get("alive")
                }
            else:  # STANDARD
                # Keep standard fields, exclude verbose metadata
                filtered[org_id] = {
                    "id": state.get("id"),
                    "name": state.get("name"),
                    "phase": state.get("phase"),
                    "age": state.get("age"),
                    "alive": state.get("alive"),
                    "capabilities": state.get("capabilities", {}),
                    "traits": state.get("traits", {}),
                    "subsystems": state.get("subsystems", {})
                }

        return filtered

    def _on_event(self, event: Event) -> None:
        """Handle events during recording"""
        if not self._is_recording:
            return

        # Apply filter if provided
        if self.config.event_filter and not self.config.event_filter(event):
            return

        # Skip internal recording events
        if event.type.startswith("recording."):
            return

        self._event_buffer.append({
            "type": event.type,
            "data": event.data,
            "source": event.source,
            "priority": event.priority.name,
            "timestamp": event.timestamp.isoformat()
        })

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Save recording to file.

        Args:
            filepath: Path to save recording (will add .npcpu extension if missing)
        """
        if not self._recording:
            raise RuntimeError("No recording to save")

        path = Path(filepath)
        if path.suffix not in ['.npcpu', '.json', '.gz']:
            path = path.with_suffix('.npcpu')

        data = self._recording.to_dict()
        json_data = json.dumps(data, indent=2, default=str)

        if self.config.compress or path.suffix == '.gz':
            # Compress with gzip
            compressed = gzip.compress(json_data.encode('utf-8'))
            with open(path, 'wb') as f:
                f.write(compressed)
        else:
            with open(path, 'w') as f:
                f.write(json_data)

        # Emit save event
        bus = get_event_bus()
        bus.emit("recording.saved", {
            "name": self._recording.name,
            "filepath": str(path),
            "size_bytes": path.stat().st_size
        })

    @classmethod
    def load(cls, filepath: str) -> Recording:
        """
        Load recording from file.

        Args:
            filepath: Path to recording file

        Returns:
            Recording object
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Recording file not found: {filepath}")

        # Try to read as compressed first
        try:
            with open(path, 'rb') as f:
                content = f.read()
            decompressed = gzip.decompress(content)
            data = json.loads(decompressed.decode('utf-8'))
        except (gzip.BadGzipFile, OSError):
            # Not compressed, read as plain JSON
            with open(path, 'r') as f:
                data = json.load(f)

        recording = Recording.from_dict(data)

        # Emit load event
        bus = get_event_bus()
        bus.emit("recording.loaded", {
            "name": recording.name,
            "filepath": str(path),
            "total_ticks": recording.total_ticks
        })

        return recording

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current recording"""
        if not self._recording:
            return {"status": "no_recording"}

        return {
            "name": self._recording.name,
            "is_recording": self._is_recording,
            "total_ticks": self._recording.total_ticks,
            "keyframes": len(self._recording.keyframes),
            "deltas": len(self._recording.deltas),
            "start_tick": self._recording.start_tick,
            "end_tick": self._recording.end_tick,
            "detail_level": self.config.detail_level.name,
            "keyframe_interval": self.config.keyframe_interval
        }

    def estimate_size(self) -> int:
        """Estimate size of current recording in bytes"""
        if not self._recording:
            return 0

        # Rough estimate based on JSON serialization
        data = self._recording.to_dict()
        json_str = json.dumps(data, default=str)

        if self.config.compress:
            # Estimate compression ratio ~0.3 for typical simulation data
            return int(len(json_str.encode('utf-8')) * 0.3)
        return len(json_str.encode('utf-8'))
