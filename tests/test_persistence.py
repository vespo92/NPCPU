"""
Tests for NPCPU Persistence & Replay System

Tests the SimulationRecorder and SimulationPlayer classes including:
- Recording simulation states
- Delta compression
- Keyframe creation
- State reconstruction
- Playback controls
- Branching functionality
"""

import pytest
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from persistence.recorder import (
    SimulationRecorder,
    DetailLevel,
    RecordingConfig,
    Recording,
    Keyframe,
    Delta
)
from persistence.replay import (
    SimulationPlayer,
    PlaybackState,
    PlaybackDirection,
    SimulationState
)
from core.simple_organism import SimpleOrganism, SimplePopulation


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_organism():
    """Create a simple organism for testing"""
    return SimpleOrganism(
        name="TestOrganism",
        traits={
            "vitality": 1.0,
            "metabolism": 1.0,
            "resilience": 1.0,
            "recovery": 1.0,
            "awareness": 1.0,
            "aggression": 0.5,
            "sociability": 0.5
        }
    )


@pytest.fixture
def simple_population():
    """Create a population with multiple organisms"""
    pop = SimplePopulation("TestPopulation")
    for i in range(5):
        org = SimpleOrganism(name=f"Organism_{i}")
        pop.add(org)
    return pop


@pytest.fixture
def recorder():
    """Create a SimulationRecorder"""
    return SimulationRecorder(
        keyframe_interval=10,
        detail_level=DetailLevel.STANDARD,
        compress=False,  # Disable compression for easier testing
        record_events=True
    )


@pytest.fixture
def sample_recording(recorder, simple_population):
    """Create a sample recording for testing playback"""
    recorder.start_recording("test_recording")

    # Simulate 50 ticks
    for tick in range(50):
        simple_population.tick()
        organisms = {
            org.id: org.to_dict()
            for org in simple_population.organisms.values()
        }
        recorder.record_tick(
            tick=tick,
            organisms=organisms,
            world_state={"tick": tick, "resources": 100 - tick}
        )

    recorder.stop_recording()
    return recorder.current_recording


# =============================================================================
# SimulationRecorder Tests
# =============================================================================

class TestSimulationRecorder:
    """Tests for SimulationRecorder"""

    def test_recorder_creation(self):
        """Test creating a recorder with various configurations"""
        recorder = SimulationRecorder()
        assert recorder.config.keyframe_interval == 100
        assert recorder.config.detail_level == DetailLevel.STANDARD
        assert not recorder.is_recording

    def test_recorder_with_custom_config(self):
        """Test recorder with custom configuration"""
        recorder = SimulationRecorder(
            keyframe_interval=50,
            detail_level=DetailLevel.FULL,
            compress=True,
            record_events=False
        )
        assert recorder.config.keyframe_interval == 50
        assert recorder.config.detail_level == DetailLevel.FULL
        assert recorder.config.compress is True
        assert recorder.config.record_events is False

    def test_start_recording(self, recorder):
        """Test starting a recording session"""
        recorder.start_recording("test")
        assert recorder.is_recording
        assert recorder.current_recording is not None
        assert recorder.current_recording.name == "test"

    def test_stop_recording(self, recorder):
        """Test stopping a recording session"""
        recorder.start_recording("test")
        recording = recorder.stop_recording()

        assert not recorder.is_recording
        assert recording is not None
        assert recording.name == "test"

    def test_double_start_raises_error(self, recorder):
        """Test that starting while already recording raises error"""
        recorder.start_recording("test1")
        with pytest.raises(RuntimeError):
            recorder.start_recording("test2")

    def test_record_tick(self, recorder, simple_organism):
        """Test recording a single tick"""
        recorder.start_recording("test")

        organisms = {simple_organism.id: simple_organism.to_dict()}
        recorder.record_tick(tick=0, organisms=organisms, world_state={"x": 1})

        recording = recorder.stop_recording()
        assert recording.total_ticks == 1
        assert len(recording.keyframes) == 1  # First tick is always keyframe

    def test_keyframe_creation(self, recorder, simple_organism):
        """Test that keyframes are created at correct intervals"""
        recorder.start_recording("test")

        organisms = {simple_organism.id: simple_organism.to_dict()}

        # Record 25 ticks with keyframe interval of 10
        for tick in range(25):
            recorder.record_tick(tick=tick, organisms=organisms)

        recording = recorder.stop_recording()

        # Should have keyframes at ticks 0, 10, 20
        assert len(recording.keyframes) == 3
        assert recording.keyframes[0].tick == 0
        assert recording.keyframes[1].tick == 10
        assert recording.keyframes[2].tick == 20

    def test_delta_compression(self, recorder):
        """Test that deltas only store changes"""
        recorder.start_recording("test")

        # Initial state
        organisms = {
            "org1": {"id": "org1", "name": "Test", "health": 100},
            "org2": {"id": "org2", "name": "Test2", "health": 100}
        }
        recorder.record_tick(tick=0, organisms=organisms)

        # Modify one organism
        organisms["org1"]["health"] = 90
        recorder.record_tick(tick=1, organisms=organisms)

        recording = recorder.stop_recording()

        # Should have 1 delta with only the changed field
        assert len(recording.deltas) == 1
        delta = recording.deltas[0]
        assert "org1" in delta.organisms_modified
        assert delta.organisms_modified["org1"]["health"] == 90

    def test_organism_added_detection(self, recorder):
        """Test detection of newly added organisms"""
        recorder.start_recording("test")

        organisms = {"org1": {"id": "org1", "name": "Original"}}
        recorder.record_tick(tick=0, organisms=organisms)

        # Add new organism
        organisms["org2"] = {"id": "org2", "name": "New"}
        recorder.record_tick(tick=1, organisms=organisms)

        recording = recorder.stop_recording()

        assert len(recording.deltas) == 1
        delta = recording.deltas[0]
        assert "org2" in delta.organisms_added

    def test_organism_removed_detection(self, recorder):
        """Test detection of removed organisms"""
        recorder.start_recording("test")

        organisms = {
            "org1": {"id": "org1"},
            "org2": {"id": "org2"}
        }
        recorder.record_tick(tick=0, organisms=organisms)

        # Remove organism
        del organisms["org2"]
        recorder.record_tick(tick=1, organisms=organisms)

        recording = recorder.stop_recording()

        assert len(recording.deltas) == 1
        delta = recording.deltas[0]
        assert "org2" in delta.organisms_removed

    def test_detail_level_minimal(self):
        """Test minimal detail level filtering"""
        recorder = SimulationRecorder(
            keyframe_interval=100,
            detail_level=DetailLevel.MINIMAL
        )
        recorder.start_recording("test")

        organisms = {
            "org1": {
                "id": "org1",
                "name": "Test",
                "age": 10,
                "phase": "MATURE",
                "alive": True,
                "subsystems": {"energy": {"value": 100}},
                "traits": {"vitality": 1.0},
                "capabilities": {"perception": 0.5},
                "extra_field": "should_be_filtered"
            }
        }
        recorder.record_tick(tick=0, organisms=organisms)

        recording = recorder.stop_recording()

        # Minimal should only include essential fields
        recorded_org = recording.keyframes[0].organisms["org1"]
        assert "id" in recorded_org
        assert "name" in recorded_org
        assert "phase" in recorded_org
        assert "age" in recorded_org
        assert "alive" in recorded_org
        assert "extra_field" not in recorded_org

    def test_save_and_load(self, recorder, simple_organism):
        """Test saving and loading recordings"""
        recorder.start_recording("test_save")

        organisms = {simple_organism.id: simple_organism.to_dict()}
        for tick in range(20):
            simple_organism.tick()
            organisms[simple_organism.id] = simple_organism.to_dict()
            recorder.record_tick(tick=tick, organisms=organisms)

        recorder.stop_recording()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.npcpu', delete=False) as f:
            temp_path = f.name

        try:
            recorder.save(temp_path)

            # Load and verify
            loaded = SimulationRecorder.load(temp_path)
            assert loaded.name == "test_save"
            assert loaded.total_ticks == 20
            assert len(loaded.keyframes) == 2  # At ticks 0 and 10
        finally:
            os.unlink(temp_path)

    def test_get_stats(self, recorder, simple_organism):
        """Test getting recording statistics"""
        recorder.start_recording("stats_test")

        organisms = {simple_organism.id: simple_organism.to_dict()}
        for tick in range(15):
            recorder.record_tick(tick=tick, organisms=organisms)

        stats = recorder.get_stats()

        assert stats["name"] == "stats_test"
        assert stats["is_recording"] is True
        assert stats["total_ticks"] == 15
        assert stats["keyframes"] == 2  # At 0 and 10


# =============================================================================
# SimulationPlayer Tests
# =============================================================================

class TestSimulationPlayer:
    """Tests for SimulationPlayer"""

    def test_player_creation(self, sample_recording):
        """Test creating a player from recording"""
        player = SimulationPlayer(sample_recording)

        assert player.current_tick == 0
        assert player.total_ticks == 50
        assert player.playback_state == PlaybackState.STOPPED

    def test_step_forward(self, sample_recording):
        """Test stepping forward through recording"""
        player = SimulationPlayer(sample_recording)

        state = player.step_forward()
        assert state is not None
        assert player.current_tick == 1

        state = player.step_forward()
        assert player.current_tick == 2

    def test_step_backward(self, sample_recording):
        """Test stepping backward through recording"""
        player = SimulationPlayer(sample_recording)

        # Go forward first
        player.seek(10)

        state = player.step_backward()
        assert state is not None
        assert player.current_tick == 9

    def test_step_multiple(self, sample_recording):
        """Test stepping multiple ticks"""
        player = SimulationPlayer(sample_recording)

        state = player.step(5)
        assert player.current_tick == 5

        state = player.step(-3)
        assert player.current_tick == 2

    def test_seek(self, sample_recording):
        """Test seeking to specific tick"""
        player = SimulationPlayer(sample_recording)

        state = player.seek(25)
        assert player.current_tick == 25
        assert state is not None
        assert state.tick == 25

    def test_seek_clamps_to_range(self, sample_recording):
        """Test that seek clamps to valid range"""
        player = SimulationPlayer(sample_recording)

        # Seek past end
        state = player.seek(1000)
        assert player.current_tick == sample_recording.end_tick

        # Seek before start
        state = player.seek(-100)
        assert player.current_tick == sample_recording.start_tick

    def test_seek_percent(self, sample_recording):
        """Test seeking by percentage"""
        player = SimulationPlayer(sample_recording)

        state = player.seek_percent(0.5)
        # 50% of 0-49 should be around tick 24-25
        assert 20 <= player.current_tick <= 30

        state = player.seek_percent(0.0)
        assert player.current_tick == sample_recording.start_tick

        state = player.seek_percent(1.0)
        assert player.current_tick == sample_recording.end_tick

    def test_state_reconstruction(self, sample_recording):
        """Test that states are correctly reconstructed"""
        player = SimulationPlayer(sample_recording)

        # Get state at tick 25
        state = player.seek(25)

        assert state is not None
        assert state.tick == 25
        assert isinstance(state.organisms, dict)
        assert isinstance(state.world_state, dict)

    def test_state_consistency(self, sample_recording):
        """Test state consistency when seeking multiple times"""
        player = SimulationPlayer(sample_recording)

        # Seek to same tick from different directions
        state1 = player.seek(25)
        player.seek(0)
        state2 = player.seek(25)

        # States should be identical
        assert state1.tick == state2.tick
        assert state1.organisms.keys() == state2.organisms.keys()

    def test_branch(self, sample_recording):
        """Test branching from a point"""
        player = SimulationPlayer(sample_recording)

        player.seek(20)
        branch_data = player.branch()

        assert branch_data["source_recording"] == sample_recording.name
        assert branch_data["branch_tick"] == 20
        assert "organisms" in branch_data
        assert "world_state" in branch_data

    def test_branch_at_specific_tick(self, sample_recording):
        """Test branching from specific tick"""
        player = SimulationPlayer(sample_recording)

        branch_data = player.branch(tick=30)

        assert branch_data["branch_tick"] == 30

    def test_create_branch_recording(self, sample_recording):
        """Test creating a new recorder from branch point"""
        player = SimulationPlayer(sample_recording)

        player.seek(25)
        new_recorder = player.create_branch_recording()

        assert new_recorder.is_recording
        assert new_recorder.current_recording is not None
        assert "branch_25" in new_recorder.current_recording.name

    def test_progress_property(self, sample_recording):
        """Test progress calculation"""
        player = SimulationPlayer(sample_recording)

        assert player.progress == 0.0

        player.seek(sample_recording.end_tick // 2)
        assert 0.4 <= player.progress <= 0.6

        player.seek(sample_recording.end_tick)
        assert player.progress == 1.0

    def test_iter_states(self, sample_recording):
        """Test iterating over states"""
        player = SimulationPlayer(sample_recording)

        states = list(player.iter_states(start=0, end=5))

        assert len(states) == 6  # 0 through 5 inclusive
        assert states[0].tick == 0
        assert states[5].tick == 5

    def test_iter_states_with_step(self, sample_recording):
        """Test iterating with step size"""
        player = SimulationPlayer(sample_recording)

        states = list(player.iter_states(start=0, end=20, step=5))

        assert len(states) == 5  # 0, 5, 10, 15, 20
        ticks = [s.tick for s in states]
        assert ticks == [0, 5, 10, 15, 20]

    def test_iter_keyframes(self, sample_recording):
        """Test iterating over keyframes only"""
        player = SimulationPlayer(sample_recording)

        keyframes = list(player.iter_keyframes())

        # Should have keyframes at 0, 10, 20, 30, 40
        assert len(keyframes) >= 4
        for kf in keyframes:
            assert kf.tick % 10 == 0

    def test_is_finished(self, sample_recording):
        """Test is_finished detection"""
        player = SimulationPlayer(sample_recording)

        assert not player.is_finished

        player.seek(sample_recording.end_tick)
        assert player.is_finished

    def test_direction_property(self, sample_recording):
        """Test playback direction property"""
        player = SimulationPlayer(sample_recording)

        assert player.direction == PlaybackDirection.FORWARD

        player.direction = PlaybackDirection.BACKWARD
        assert player.direction == PlaybackDirection.BACKWARD

    def test_get_organism_history(self, sample_recording):
        """Test getting history for a specific organism"""
        player = SimulationPlayer(sample_recording)

        # Get any organism ID from the recording
        first_kf = sample_recording.keyframes[0]
        if first_kf.organisms:
            org_id = list(first_kf.organisms.keys())[0]

            history = player.get_organism_history(org_id)

            assert len(history) > 0
            for entry in history:
                assert "tick" in entry
                assert "state" in entry

    def test_get_population_stats(self, sample_recording):
        """Test getting population statistics"""
        player = SimulationPlayer(sample_recording)

        stats = player.get_population_stats()

        assert len(stats) >= 1
        for stat in stats:
            assert "tick" in stat
            assert "total_organisms" in stat
            assert "alive_organisms" in stat


# =============================================================================
# Integration Tests
# =============================================================================

class TestPersistenceIntegration:
    """Integration tests for the full recording/playback cycle"""

    def test_full_recording_playback_cycle(self, simple_population):
        """Test complete recording and playback cycle"""
        # Create recorder
        recorder = SimulationRecorder(
            keyframe_interval=10,
            compress=False
        )
        recorder.start_recording("integration_test")

        # Record simulation
        for tick in range(30):
            simple_population.tick()
            organisms = {
                org.id: org.to_dict()
                for org in simple_population.organisms.values()
            }
            recorder.record_tick(
                tick=tick,
                organisms=organisms,
                world_state={"tick": tick}
            )

        recording = recorder.stop_recording()

        # Create player and verify
        player = SimulationPlayer(recording)

        # Step through and verify consistency
        for tick in range(10):
            state = player.step_forward()
            assert state is not None
            assert state.tick == tick + 1

        # Seek and branch
        player.seek(15)
        branch = player.branch()

        assert branch["branch_tick"] == 15
        assert len(branch["organisms"]) > 0

    def test_save_load_playback(self, simple_population):
        """Test saving, loading, and playing back"""
        recorder = SimulationRecorder(
            keyframe_interval=10,
            compress=True  # Test with compression
        )
        recorder.start_recording("save_load_test")

        for tick in range(25):
            simple_population.tick()
            organisms = {
                org.id: org.to_dict()
                for org in simple_population.organisms.values()
            }
            recorder.record_tick(tick=tick, organisms=organisms)

        recorder.stop_recording()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.npcpu', delete=False) as f:
            temp_path = f.name

        try:
            recorder.save(temp_path)

            # Load and play
            loaded_recording = SimulationRecorder.load(temp_path)
            player = SimulationPlayer(loaded_recording)

            assert player.total_ticks == 25

            # Verify playback works
            state = player.seek(12)
            assert state.tick == 12

        finally:
            os.unlink(temp_path)

    def test_branch_and_continue_recording(self, simple_population):
        """Test branching and continuing with new recording"""
        # Create initial recording
        recorder = SimulationRecorder(keyframe_interval=10, compress=False)
        recorder.start_recording("original")

        for tick in range(20):
            simple_population.tick()
            organisms = {
                org.id: org.to_dict()
                for org in simple_population.organisms.values()
            }
            recorder.record_tick(tick=tick, organisms=organisms)

        original = recorder.stop_recording()

        # Play and branch
        player = SimulationPlayer(original)
        player.seek(10)

        branch_recorder = player.create_branch_recording()

        # Continue recording from branch point
        for tick in range(5):
            simple_population.tick()
            organisms = {
                org.id: org.to_dict()
                for org in simple_population.organisms.values()
            }
            branch_recorder.record_tick(
                tick=tick + 1,  # Continue from tick 1 (0 was initial keyframe)
                organisms=organisms
            )

        branch_recording = branch_recorder.stop_recording()

        assert branch_recording is not None
        assert branch_recording.total_ticks >= 1


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestDataStructures:
    """Tests for data structure serialization"""

    def test_keyframe_serialization(self):
        """Test Keyframe to_dict and from_dict"""
        kf = Keyframe(
            tick=100,
            timestamp=datetime.now(),
            organisms={"org1": {"id": "org1", "name": "Test"}},
            world_state={"resources": 50},
            metadata={"test": True}
        )

        data = kf.to_dict()
        restored = Keyframe.from_dict(data)

        assert restored.tick == kf.tick
        assert restored.organisms == kf.organisms
        assert restored.world_state == kf.world_state
        assert restored.metadata == kf.metadata

    def test_delta_serialization(self):
        """Test Delta to_dict and from_dict"""
        delta = Delta(
            tick=50,
            timestamp=datetime.now(),
            organisms_added={"org2": {"id": "org2"}},
            organisms_removed={"org1"},
            organisms_modified={"org3": {"health": 80}},
            world_changes={"resources": 30},
            events=[{"type": "test", "data": {}}]
        )

        data = delta.to_dict()
        restored = Delta.from_dict(data)

        assert restored.tick == delta.tick
        assert restored.organisms_added == delta.organisms_added
        assert restored.organisms_removed == delta.organisms_removed
        assert restored.organisms_modified == delta.organisms_modified
        assert restored.world_changes == delta.world_changes
        assert restored.events == delta.events

    def test_recording_serialization(self):
        """Test Recording to_dict and from_dict"""
        config = RecordingConfig(
            keyframe_interval=50,
            detail_level=DetailLevel.FULL
        )

        recording = Recording(
            name="test_recording",
            created_at=datetime.now(),
            config=config,
            total_ticks=100,
            start_tick=0,
            end_tick=99
        )

        # Add some keyframes and deltas
        recording.keyframes.append(Keyframe(
            tick=0,
            timestamp=datetime.now(),
            organisms={},
            world_state={}
        ))
        recording.deltas.append(Delta(
            tick=1,
            timestamp=datetime.now()
        ))

        data = recording.to_dict()
        restored = Recording.from_dict(data)

        assert restored.name == recording.name
        assert restored.total_ticks == recording.total_ticks
        assert len(restored.keyframes) == 1
        assert len(restored.deltas) == 1
        assert restored.config.keyframe_interval == 50
        assert restored.config.detail_level == DetailLevel.FULL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
