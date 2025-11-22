"""
Simulation Player for NPCPU

Replays recorded simulations with:
- Forward/backward playback
- Variable speed control
- Jump to any tick
- Branch from any point (what-if scenarios)

Example:
    from persistence import SimulationPlayer, SimulationRecorder

    # Load recording
    recording = SimulationRecorder.load("simulation.npcpu")

    # Create player
    player = SimulationPlayer(recording)

    # Play at 2x speed
    player.play(speed=2.0)

    # Or step through manually
    while not player.is_finished:
        state = player.step_forward()
        print(f"Tick {state['tick']}: {len(state['organisms'])} organisms")

    # Jump to specific tick
    player.seek(500)

    # Branch to create new simulation
    branch_state = player.branch(250)
"""

from typing import Dict, Any, List, Optional, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import copy
import time
import threading

from core.events import get_event_bus
from .recorder import Recording, Keyframe, Delta, SimulationRecorder


# =============================================================================
# Enums
# =============================================================================

class PlaybackState(Enum):
    """Current state of the player"""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    FINISHED = auto()


class PlaybackDirection(Enum):
    """Direction of playback"""
    FORWARD = auto()
    BACKWARD = auto()


# =============================================================================
# State Reconstruction
# =============================================================================

@dataclass
class SimulationState:
    """Reconstructed state at a point in time"""
    tick: int
    timestamp: datetime
    organisms: Dict[str, Dict[str, Any]]
    world_state: Dict[str, Any]
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp.isoformat(),
            "organisms": self.organisms,
            "world_state": self.world_state,
            "events": self.events
        }


# =============================================================================
# Simulation Player
# =============================================================================

class SimulationPlayer:
    """
    Replays recorded simulations.

    Features:
    - Forward/backward playback
    - Variable speed
    - Jump to tick
    - Branch from any point (what-if scenarios)

    Example:
        recording = SimulationRecorder.load("sim.npcpu")
        player = SimulationPlayer(recording)

        # Play forward
        player.play(speed=1.0)

        # Or manual stepping
        state = player.step_forward()
        print(f"Tick {state.tick}")

        # Jump to tick 500
        player.seek(500)

        # Branch to create editable state
        branch = player.branch(250)
    """

    def __init__(self, recording: Recording):
        self._recording = recording
        self._current_tick = 0
        self._state = PlaybackState.STOPPED
        self._direction = PlaybackDirection.FORWARD
        self._speed = 1.0

        # Reconstructed current state
        self._current_state: Optional[SimulationState] = None

        # Build index for efficient seeking
        self._keyframe_index: Dict[int, int] = {}  # tick -> keyframe list index
        self._delta_index: Dict[int, int] = {}     # tick -> delta list index
        self._build_indices()

        # Playback thread control
        self._play_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Callbacks
        self._on_tick: Optional[Callable[[SimulationState], None]] = None
        self._on_state_change: Optional[Callable[[PlaybackState], None]] = None

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def _build_indices(self) -> None:
        """Build indices for efficient tick lookup"""
        for i, kf in enumerate(self._recording.keyframes):
            self._keyframe_index[kf.tick] = i

        for i, delta in enumerate(self._recording.deltas):
            self._delta_index[delta.tick] = i

    def _find_nearest_keyframe(self, tick: int) -> Optional[Keyframe]:
        """Find the keyframe at or before the given tick"""
        candidate_tick = -1
        candidate_idx = -1

        for kf_tick, idx in self._keyframe_index.items():
            if kf_tick <= tick and kf_tick > candidate_tick:
                candidate_tick = kf_tick
                candidate_idx = idx

        if candidate_idx >= 0:
            return self._recording.keyframes[candidate_idx]
        return None

    def _get_deltas_in_range(
        self,
        start_tick: int,
        end_tick: int,
        inclusive: bool = True
    ) -> List[Delta]:
        """Get all deltas between start and end ticks"""
        deltas = []
        for delta in self._recording.deltas:
            if inclusive:
                if start_tick <= delta.tick <= end_tick:
                    deltas.append(delta)
            else:
                if start_tick < delta.tick <= end_tick:
                    deltas.append(delta)
        return sorted(deltas, key=lambda d: d.tick)

    # -------------------------------------------------------------------------
    # State Reconstruction
    # -------------------------------------------------------------------------

    def _reconstruct_state(self, tick: int) -> SimulationState:
        """Reconstruct simulation state at a specific tick"""
        # Find nearest keyframe
        keyframe = self._find_nearest_keyframe(tick)

        if not keyframe:
            # No keyframe found, return empty state
            return SimulationState(
                tick=tick,
                timestamp=datetime.now(),
                organisms={},
                world_state={}
            )

        # Start with keyframe state
        organisms = copy.deepcopy(keyframe.organisms)
        world_state = copy.deepcopy(keyframe.world_state)
        events = []

        # Apply deltas from keyframe to target tick
        if tick > keyframe.tick:
            deltas = self._get_deltas_in_range(keyframe.tick, tick, inclusive=False)

            for delta in deltas:
                # Apply organism changes
                for org_id in delta.organisms_removed:
                    organisms.pop(org_id, None)

                for org_id, org_state in delta.organisms_added.items():
                    organisms[org_id] = copy.deepcopy(org_state)

                for org_id, changes in delta.organisms_modified.items():
                    if org_id in organisms:
                        self._apply_diff(organisms[org_id], changes)

                # Apply world changes
                self._apply_diff(world_state, delta.world_changes)

                # Collect events
                events.extend(delta.events)

        return SimulationState(
            tick=tick,
            timestamp=datetime.now(),
            organisms=organisms,
            world_state=world_state,
            events=events
        )

    def _apply_diff(self, state: Dict[str, Any], diff: Dict[str, Any]) -> None:
        """Apply a diff to a state dictionary"""
        for key, value in diff.items():
            if isinstance(value, dict) and key in state and isinstance(state[key], dict):
                # Recursively apply nested diffs
                self._apply_diff(state[key], value)
            else:
                state[key] = copy.deepcopy(value)

    # -------------------------------------------------------------------------
    # Playback Control
    # -------------------------------------------------------------------------

    def play(self, speed: float = 1.0) -> None:
        """
        Start playback at the given speed.

        Args:
            speed: Playback speed multiplier (1.0 = normal, 2.0 = 2x, etc.)
        """
        if self._state == PlaybackState.PLAYING:
            return

        self._speed = max(0.1, speed)
        self._stop_flag.clear()
        self._set_state(PlaybackState.PLAYING)

        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

        bus = get_event_bus()
        bus.emit("playback.started", {
            "recording": self._recording.name,
            "tick": self._current_tick,
            "speed": self._speed
        })

    def pause(self) -> None:
        """Pause playback"""
        if self._state != PlaybackState.PLAYING:
            return

        self._stop_flag.set()
        if self._play_thread:
            self._play_thread.join(timeout=1.0)

        self._set_state(PlaybackState.PAUSED)

        bus = get_event_bus()
        bus.emit("playback.paused", {
            "recording": self._recording.name,
            "tick": self._current_tick
        })

    def stop(self) -> None:
        """Stop playback and reset to beginning"""
        if self._state == PlaybackState.PLAYING:
            self._stop_flag.set()
            if self._play_thread:
                self._play_thread.join(timeout=1.0)

        self._current_tick = self._recording.start_tick
        self._current_state = None
        self._set_state(PlaybackState.STOPPED)

        bus = get_event_bus()
        bus.emit("playback.stopped", {
            "recording": self._recording.name
        })

    def _playback_loop(self) -> None:
        """Background playback loop"""
        base_interval = 1.0 / 60.0  # Base 60 ticks per second

        while not self._stop_flag.is_set():
            if self._direction == PlaybackDirection.FORWARD:
                state = self.step_forward()
            else:
                state = self.step_backward()

            if state and self._on_tick:
                self._on_tick(state)

            if self.is_finished:
                self._set_state(PlaybackState.FINISHED)
                bus = get_event_bus()
                bus.emit("playback.finished", {
                    "recording": self._recording.name,
                    "tick": self._current_tick
                })
                break

            # Sleep based on speed
            time.sleep(base_interval / self._speed)

    def _set_state(self, state: PlaybackState) -> None:
        """Set playback state and notify callback"""
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)

    # -------------------------------------------------------------------------
    # Stepping
    # -------------------------------------------------------------------------

    def step_forward(self) -> Optional[SimulationState]:
        """
        Advance one tick forward.

        Returns:
            SimulationState at the new tick, or None if at end
        """
        if self._current_tick >= self._recording.end_tick:
            return None

        self._current_tick += 1
        self._current_state = self._reconstruct_state(self._current_tick)

        bus = get_event_bus()
        bus.emit("playback.tick", {
            "tick": self._current_tick,
            "organism_count": len(self._current_state.organisms)
        })

        return self._current_state

    def step_backward(self) -> Optional[SimulationState]:
        """
        Go back one tick.

        Returns:
            SimulationState at the new tick, or None if at start
        """
        if self._current_tick <= self._recording.start_tick:
            return None

        self._current_tick -= 1
        self._current_state = self._reconstruct_state(self._current_tick)

        bus = get_event_bus()
        bus.emit("playback.tick", {
            "tick": self._current_tick,
            "organism_count": len(self._current_state.organisms)
        })

        return self._current_state

    def step(self, count: int = 1) -> Optional[SimulationState]:
        """
        Step multiple ticks forward or backward.

        Args:
            count: Number of ticks (positive=forward, negative=backward)

        Returns:
            SimulationState at the new position
        """
        if count > 0:
            for _ in range(count - 1):
                if not self.step_forward():
                    break
            return self.step_forward()
        elif count < 0:
            for _ in range(abs(count) - 1):
                if not self.step_backward():
                    break
            return self.step_backward()
        return self._current_state

    # -------------------------------------------------------------------------
    # Seeking
    # -------------------------------------------------------------------------

    def seek(self, tick: int) -> SimulationState:
        """
        Jump to a specific tick.

        Args:
            tick: Target tick number

        Returns:
            SimulationState at the target tick
        """
        # Clamp to valid range
        tick = max(self._recording.start_tick, min(tick, self._recording.end_tick))

        self._current_tick = tick
        self._current_state = self._reconstruct_state(tick)

        bus = get_event_bus()
        bus.emit("playback.seek", {
            "tick": tick,
            "organism_count": len(self._current_state.organisms)
        })

        return self._current_state

    def seek_percent(self, percent: float) -> SimulationState:
        """
        Jump to a percentage through the recording.

        Args:
            percent: Percentage (0.0 to 1.0)

        Returns:
            SimulationState at the target position
        """
        percent = max(0.0, min(1.0, percent))
        total_ticks = self._recording.end_tick - self._recording.start_tick
        target_tick = self._recording.start_tick + int(total_ticks * percent)
        return self.seek(target_tick)

    # -------------------------------------------------------------------------
    # Branching
    # -------------------------------------------------------------------------

    def branch(self, tick: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a branch point for what-if scenarios.

        Returns the complete state at the branch point that can be used
        to initialize a new simulation.

        Args:
            tick: Tick to branch from (None = current tick)

        Returns:
            Dictionary with full state suitable for initializing a simulation
        """
        target_tick = tick if tick is not None else self._current_tick
        state = self._reconstruct_state(target_tick)

        branch_data = {
            "source_recording": self._recording.name,
            "branch_tick": target_tick,
            "branched_at": datetime.now().isoformat(),
            "organisms": state.organisms,
            "world_state": state.world_state,
            "metadata": {
                "original_start_tick": self._recording.start_tick,
                "original_end_tick": self._recording.end_tick
            }
        }

        bus = get_event_bus()
        bus.emit("playback.branched", {
            "recording": self._recording.name,
            "tick": target_tick,
            "organism_count": len(state.organisms)
        })

        return branch_data

    def create_branch_recording(self, tick: Optional[int] = None) -> SimulationRecorder:
        """
        Create a new recorder initialized at a branch point.

        This allows continuing recording from a specific point in time.

        Args:
            tick: Tick to branch from (None = current tick)

        Returns:
            SimulationRecorder ready to continue from the branch point
        """
        branch_data = self.branch(tick)

        recorder = SimulationRecorder(
            keyframe_interval=self._recording.config.keyframe_interval,
            detail_level=self._recording.config.detail_level,
            compress=self._recording.config.compress,
            record_events=self._recording.config.record_events
        )

        # Start recording with branch info
        recorder.start_recording(f"{self._recording.name}_branch_{branch_data['branch_tick']}")

        # Record initial state as keyframe
        recorder.record_tick(
            tick=0,  # Start new recording from tick 0
            organisms=branch_data["organisms"],
            world_state=branch_data["world_state"]
        )

        return recorder

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def current_state(self) -> Optional[SimulationState]:
        if self._current_state is None:
            self._current_state = self._reconstruct_state(self._current_tick)
        return self._current_state

    @property
    def playback_state(self) -> PlaybackState:
        return self._state

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed = max(0.1, value)

    @property
    def direction(self) -> PlaybackDirection:
        return self._direction

    @direction.setter
    def direction(self, value: PlaybackDirection) -> None:
        self._direction = value

    @property
    def is_playing(self) -> bool:
        return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        return self._state == PlaybackState.PAUSED

    @property
    def is_finished(self) -> bool:
        if self._direction == PlaybackDirection.FORWARD:
            return self._current_tick >= self._recording.end_tick
        return self._current_tick <= self._recording.start_tick

    @property
    def total_ticks(self) -> int:
        return self._recording.total_ticks

    @property
    def progress(self) -> float:
        """Get playback progress as a percentage (0.0 to 1.0)"""
        total = self._recording.end_tick - self._recording.start_tick
        if total == 0:
            return 1.0
        current = self._current_tick - self._recording.start_tick
        return current / total

    @property
    def recording(self) -> Recording:
        return self._recording

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_tick(self, callback: Callable[[SimulationState], None]) -> None:
        """Set callback for each tick during playback"""
        self._on_tick = callback

    def on_state_change(self, callback: Callable[[PlaybackState], None]) -> None:
        """Set callback for playback state changes"""
        self._on_state_change = callback

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def iter_states(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1
    ) -> Iterator[SimulationState]:
        """
        Iterate over states in the recording.

        Args:
            start: Starting tick (default: recording start)
            end: Ending tick (default: recording end)
            step: Step size between ticks

        Yields:
            SimulationState for each tick
        """
        start = start if start is not None else self._recording.start_tick
        end = end if end is not None else self._recording.end_tick

        for tick in range(start, end + 1, step):
            yield self._reconstruct_state(tick)

    def iter_keyframes(self) -> Iterator[SimulationState]:
        """Iterate over keyframe states only (more efficient)"""
        for keyframe in self._recording.keyframes:
            yield SimulationState(
                tick=keyframe.tick,
                timestamp=keyframe.timestamp,
                organisms=copy.deepcopy(keyframe.organisms),
                world_state=copy.deepcopy(keyframe.world_state)
            )

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def get_organism_history(self, organism_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of a specific organism across the recording.

        Args:
            organism_id: ID of the organism to track

        Returns:
            List of state snapshots for the organism
        """
        history = []

        for state in self.iter_states():
            if organism_id in state.organisms:
                history.append({
                    "tick": state.tick,
                    "state": copy.deepcopy(state.organisms[organism_id])
                })

        return history

    def get_population_stats(self) -> List[Dict[str, Any]]:
        """
        Get population statistics over time.

        Returns:
            List of stats dictionaries for each keyframe
        """
        stats = []

        for keyframe in self._recording.keyframes:
            alive_count = sum(
                1 for org in keyframe.organisms.values()
                if org.get("alive", True)
            )

            stats.append({
                "tick": keyframe.tick,
                "total_organisms": len(keyframe.organisms),
                "alive_organisms": alive_count,
                "world_state": keyframe.world_state
            })

        return stats

    def find_events(
        self,
        event_type: Optional[str] = None,
        start_tick: Optional[int] = None,
        end_tick: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for events in the recording.

        Args:
            event_type: Filter by event type (None = all events)
            start_tick: Start of search range
            end_tick: End of search range

        Returns:
            List of matching events with tick information
        """
        start = start_tick if start_tick is not None else self._recording.start_tick
        end = end_tick if end_tick is not None else self._recording.end_tick

        events = []

        for delta in self._recording.deltas:
            if start <= delta.tick <= end:
                for event in delta.events:
                    if event_type is None or event.get("type") == event_type:
                        events.append({
                            "tick": delta.tick,
                            **event
                        })

        return events
