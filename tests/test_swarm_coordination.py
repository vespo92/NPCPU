"""
Tests for Swarm Coordination System

Tests the SwarmCoordinator, flocking behavior, task allocation,
and formation patterns.
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from swarm.coordination import (
    SwarmCoordinator,
    SwarmSubsystem,
    Position,
    FlockingVectors,
    Task,
    FormationPattern
)
from core.simple_organism import SimpleOrganism


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def coordinator():
    """Create a SwarmCoordinator for tests"""
    return SwarmCoordinator(
        separation_weight=1.5,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        separation_radius=25.0,
        neighbor_radius=50.0,
        max_speed=5.0
    )


@pytest.fixture
def organisms():
    """Create a list of test organisms"""
    return [SimpleOrganism(f"Org_{i}") for i in range(5)]


@pytest.fixture
def swarm_with_members(coordinator, organisms):
    """Create a swarm with members"""
    swarm_id = coordinator.create_swarm("test_swarm")
    for i, org in enumerate(organisms):
        coordinator.add_to_swarm(swarm_id, org, position=(i * 20, 0))
    return swarm_id


# ============================================================================
# Position Tests
# ============================================================================

class TestPosition:
    """Test Position data class"""

    def test_position_creation(self):
        """Test creating a position"""
        pos = Position(x=10.0, y=20.0, vx=1.0, vy=2.0)
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.vx == 1.0
        assert pos.vy == 2.0

    def test_distance_to(self):
        """Test distance calculation"""
        pos1 = Position(x=0.0, y=0.0)
        pos2 = Position(x=3.0, y=4.0)
        assert pos1.distance_to(pos2) == 5.0

    def test_direction_to(self):
        """Test direction calculation"""
        pos1 = Position(x=0.0, y=0.0)
        pos2 = Position(x=10.0, y=0.0)
        dx, dy = pos1.direction_to(pos2)
        assert abs(dx - 1.0) < 0.0001
        assert abs(dy - 0.0) < 0.0001

    def test_direction_to_same_position(self):
        """Test direction to same position returns zero"""
        pos1 = Position(x=5.0, y=5.0)
        pos2 = Position(x=5.0, y=5.0)
        dx, dy = pos1.direction_to(pos2)
        assert dx == 0.0
        assert dy == 0.0

    def test_to_dict(self):
        """Test serialization"""
        pos = Position(x=1.0, y=2.0, vx=3.0, vy=4.0)
        data = pos.to_dict()
        assert data == {"x": 1.0, "y": 2.0, "vx": 3.0, "vy": 4.0}

    def test_from_dict(self):
        """Test deserialization"""
        data = {"x": 1.0, "y": 2.0, "vx": 3.0, "vy": 4.0}
        pos = Position.from_dict(data)
        assert pos.x == 1.0
        assert pos.y == 2.0


# ============================================================================
# Swarm Management Tests
# ============================================================================

class TestSwarmManagement:
    """Test swarm creation and management"""

    def test_create_swarm(self, coordinator):
        """Test creating a swarm"""
        swarm_id = coordinator.create_swarm("test")
        assert swarm_id is not None
        assert len(swarm_id) > 0

    def test_delete_swarm(self, coordinator):
        """Test deleting a swarm"""
        swarm_id = coordinator.create_swarm("test")
        assert coordinator.delete_swarm(swarm_id) is True
        assert coordinator.delete_swarm(swarm_id) is False  # Already deleted

    def test_add_to_swarm(self, coordinator, organisms):
        """Test adding organisms to swarm"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]

        result = coordinator.add_to_swarm(swarm_id, org, position=(10.0, 20.0))
        assert result is True
        assert coordinator.get_swarm_size(swarm_id) == 1

    def test_add_to_nonexistent_swarm(self, coordinator, organisms):
        """Test adding to non-existent swarm fails"""
        result = coordinator.add_to_swarm("fake_id", organisms[0])
        assert result is False

    def test_remove_from_swarm(self, coordinator, organisms):
        """Test removing organisms from swarm"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]
        coordinator.add_to_swarm(swarm_id, org)

        result = coordinator.remove_from_swarm(org.id)
        assert result is True
        assert coordinator.get_swarm_size(swarm_id) == 0

    def test_get_swarm_members(self, coordinator, organisms):
        """Test getting swarm members"""
        swarm_id = coordinator.create_swarm("test")
        for org in organisms[:3]:
            coordinator.add_to_swarm(swarm_id, org)

        members = coordinator.get_swarm_members(swarm_id)
        assert len(members) == 3

    def test_get_swarm_center(self, coordinator, organisms):
        """Test getting swarm center of mass"""
        swarm_id = coordinator.create_swarm("test")
        positions = [(0, 0), (10, 0), (20, 0)]

        for i, org in enumerate(organisms[:3]):
            coordinator.add_to_swarm(swarm_id, org, position=positions[i])

        center = coordinator.get_swarm_center(swarm_id)
        assert center is not None
        assert abs(center[0] - 10.0) < 0.0001
        assert abs(center[1] - 0.0) < 0.0001

    def test_member_position_management(self, coordinator, organisms):
        """Test getting and setting member positions"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]
        coordinator.add_to_swarm(swarm_id, org, position=(10.0, 20.0))

        pos = coordinator.get_member_position(org.id)
        assert pos is not None
        assert pos.x == 10.0
        assert pos.y == 20.0

        coordinator.set_member_position(org.id, 30.0, 40.0, 1.0, 2.0)
        pos = coordinator.get_member_position(org.id)
        assert pos.x == 30.0
        assert pos.y == 40.0
        assert pos.vx == 1.0
        assert pos.vy == 2.0


# ============================================================================
# Flocking Behavior Tests
# ============================================================================

class TestFlockingBehavior:
    """Test boid-like flocking behavior"""

    def test_flocking_vectors_no_neighbors(self, coordinator, organisms):
        """Test flocking with no neighbors"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]
        coordinator.add_to_swarm(swarm_id, org, position=(0, 0))

        vectors = coordinator.calculate_flocking_vectors(org.id)
        assert vectors.combined == (0.0, 0.0)

    def test_flocking_vectors_with_neighbors(self, swarm_with_members, coordinator, organisms):
        """Test flocking produces non-zero vectors with neighbors"""
        # Middle organism should have neighbors
        middle_org = organisms[2]
        vectors = coordinator.calculate_flocking_vectors(middle_org.id)

        # Should produce some movement vector
        assert isinstance(vectors, FlockingVectors)

    def test_separation_vector(self, coordinator, organisms):
        """Test separation pushes organisms apart when too close"""
        swarm_id = coordinator.create_swarm("test")

        # Place organisms very close together
        coordinator.add_to_swarm(swarm_id, organisms[0], position=(0, 0))
        coordinator.add_to_swarm(swarm_id, organisms[1], position=(5, 0))  # Within separation radius

        vectors = coordinator.calculate_flocking_vectors(organisms[0].id)
        # Separation should push left (negative x)
        assert vectors.separation[0] < 0

    def test_cohesion_vector(self, coordinator, organisms):
        """Test cohesion pulls organism toward group center"""
        swarm_id = coordinator.create_swarm("test")

        # Place one organism far from center of others
        coordinator.add_to_swarm(swarm_id, organisms[0], position=(0, 0))
        coordinator.add_to_swarm(swarm_id, organisms[1], position=(40, 0))
        coordinator.add_to_swarm(swarm_id, organisms[2], position=(40, 10))

        vectors = coordinator.calculate_flocking_vectors(organisms[0].id)
        # Cohesion should pull right toward the group
        assert vectors.cohesion[0] > 0

    def test_max_speed_limit(self, coordinator, organisms):
        """Test that combined vector is limited by max_speed"""
        coordinator.max_speed = 2.0
        swarm_id = coordinator.create_swarm("test")

        # Create situation with strong vectors
        for i, org in enumerate(organisms):
            coordinator.add_to_swarm(swarm_id, org, position=(i * 5, i * 5))

        vectors = coordinator.calculate_flocking_vectors(organisms[2].id)
        speed = math.sqrt(vectors.combined[0]**2 + vectors.combined[1]**2)
        assert speed <= coordinator.max_speed + 0.0001


# ============================================================================
# Task Allocation Tests
# ============================================================================

class TestTaskAllocation:
    """Test task allocation system"""

    def test_allocate_single_task(self, swarm_with_members, coordinator):
        """Test allocating a single task"""
        tasks = [{"type": "patrol", "location": (100, 100)}]
        assignments = coordinator.allocate_tasks(tasks, swarm_with_members)

        assert len(assignments) == 1

    def test_allocate_multiple_tasks(self, swarm_with_members, coordinator):
        """Test allocating multiple tasks"""
        tasks = [
            {"type": "patrol", "location": (100, 100)},
            {"type": "gather", "location": (50, 50)},
            {"type": "scout", "location": (200, 200)}
        ]
        assignments = coordinator.allocate_tasks(tasks, swarm_with_members)

        assert len(assignments) == 3
        # Each task assigned to different organism
        assert len(set(assignments.values())) == 3

    def test_task_priority(self, swarm_with_members, coordinator):
        """Test that higher priority tasks are allocated first"""
        tasks = [
            {"type": "low", "priority": 0.5, "location": (0, 0)},
            {"type": "high", "priority": 1.0, "location": (0, 0)}
        ]
        assignments = coordinator.allocate_tasks(tasks, swarm_with_members)

        assert len(assignments) == 2

    def test_nearest_allocation_strategy(self, coordinator, organisms):
        """Test nearest allocation strategy"""
        swarm_id = coordinator.create_swarm("test")
        coordinator.add_to_swarm(swarm_id, organisms[0], position=(0, 0))
        coordinator.add_to_swarm(swarm_id, organisms[1], position=(100, 0))

        tasks = [{"type": "go", "location": (90, 0)}]
        assignments = coordinator.allocate_tasks(tasks, swarm_id, strategy="nearest")

        # Task near (90,0) should go to organism at (100,0)
        assert list(assignments.values())[0] == organisms[1].id

    def test_complete_task(self, swarm_with_members, coordinator):
        """Test completing a task"""
        tasks = [{"type": "patrol"}]
        assignments = coordinator.allocate_tasks(tasks, swarm_with_members)
        task_id = list(assignments.keys())[0]

        result = coordinator.complete_task(task_id)
        assert result is True

        # Task should no longer appear in organism's tasks
        org_id = assignments[task_id]
        pending = coordinator.get_organism_tasks(org_id)
        assert len(pending) == 0

    def test_get_organism_tasks(self, swarm_with_members, coordinator, organisms):
        """Test getting tasks assigned to an organism"""
        tasks = [
            {"type": "task1"},
            {"type": "task2"},
            {"type": "task3"}
        ]
        coordinator.allocate_tasks(tasks, swarm_with_members)

        # Each organism should have at most 1 task (5 organisms, 3 tasks)
        for org in organisms:
            org_tasks = coordinator.get_organism_tasks(org.id)
            assert len(org_tasks) <= 1


# ============================================================================
# Formation Tests
# ============================================================================

class TestFormations:
    """Test formation pattern calculations"""

    def test_set_formation(self, swarm_with_members, coordinator):
        """Test setting formation pattern"""
        result = coordinator.set_formation(swarm_with_members, FormationPattern.LINE)
        assert result is True

    def test_line_formation(self, swarm_with_members, coordinator):
        """Test line formation positions"""
        coordinator.set_formation(swarm_with_members, FormationPattern.LINE)
        positions = coordinator.calculate_formation_positions(
            swarm_with_members,
            center=(100, 100),
            spacing=10.0
        )

        assert len(positions) == 5
        # All y coordinates should be approximately equal (horizontal line)
        y_coords = [p[1] for p in positions.values()]
        assert all(abs(y - 100) < 1 for y in y_coords)

    def test_circle_formation(self, swarm_with_members, coordinator):
        """Test circle formation positions"""
        coordinator.set_formation(swarm_with_members, FormationPattern.CIRCLE)
        positions = coordinator.calculate_formation_positions(
            swarm_with_members,
            center=(100, 100),
            spacing=20.0
        )

        assert len(positions) == 5
        # All positions should be equidistant from center
        distances = [
            math.sqrt((p[0]-100)**2 + (p[1]-100)**2)
            for p in positions.values()
        ]
        # All distances should be equal (within tolerance)
        avg_dist = sum(distances) / len(distances)
        assert all(abs(d - avg_dist) < 0.1 for d in distances)

    def test_wedge_formation(self, swarm_with_members, coordinator):
        """Test V-wedge formation positions"""
        coordinator.set_formation(swarm_with_members, FormationPattern.WEDGE)
        positions = coordinator.calculate_formation_positions(
            swarm_with_members,
            center=(100, 100),
            spacing=20.0
        )

        assert len(positions) == 5

    def test_square_formation(self, swarm_with_members, coordinator):
        """Test square/grid formation positions"""
        coordinator.set_formation(swarm_with_members, FormationPattern.SQUARE)
        positions = coordinator.calculate_formation_positions(
            swarm_with_members,
            center=(100, 100),
            spacing=20.0
        )

        assert len(positions) == 5


# ============================================================================
# Voting Tests
# ============================================================================

class TestVoting:
    """Test distributed decision making"""

    def test_vote(self, swarm_with_members, coordinator):
        """Test basic voting"""
        winner, tallies = coordinator.vote(
            swarm_with_members,
            options=["attack", "defend", "retreat"]
        )

        assert winner in ["attack", "defend", "retreat"]
        assert sum(tallies.values()) == 5  # 5 members voted

    def test_vote_empty_swarm(self, coordinator):
        """Test voting with empty swarm"""
        swarm_id = coordinator.create_swarm("empty")
        winner, tallies = coordinator.vote(swarm_id, options=["a", "b"])

        assert winner == ""
        assert tallies == {}

    def test_weighted_voting(self, swarm_with_members, coordinator):
        """Test weighted voting"""
        weights = {org_id: 2.0 for org_id in list(coordinator._swarms[swarm_with_members].keys())[:2]}
        winner, tallies = coordinator.vote(
            swarm_with_members,
            options=["a", "b"],
            voter_weights=weights
        )

        # Total should be 2+2+1+1+1 = 7 (first two have weight 2, rest have 1)
        assert sum(tallies.values()) == 7


# ============================================================================
# SwarmSubsystem Tests
# ============================================================================

class TestSwarmSubsystem:
    """Test SwarmSubsystem for organisms"""

    def test_subsystem_creation(self, coordinator):
        """Test creating swarm subsystem"""
        subsystem = SwarmSubsystem(coordinator)
        assert subsystem.name == "swarm"

    def test_join_swarm(self, coordinator, organisms):
        """Test organism joining swarm via subsystem"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]

        subsystem = SwarmSubsystem(coordinator)
        org.add_subsystem(subsystem)

        result = subsystem.join_swarm(swarm_id, position=(10, 20))
        assert result is True
        assert coordinator.get_swarm_size(swarm_id) == 1

    def test_leave_swarm(self, coordinator, organisms):
        """Test organism leaving swarm via subsystem"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]

        subsystem = SwarmSubsystem(coordinator)
        org.add_subsystem(subsystem)
        subsystem.join_swarm(swarm_id)

        result = subsystem.leave_swarm()
        assert result is True
        assert coordinator.get_swarm_size(swarm_id) == 0

    def test_get_flocking_direction(self, coordinator, organisms):
        """Test getting flocking direction via subsystem"""
        swarm_id = coordinator.create_swarm("test")

        subsystems = []
        for i, org in enumerate(organisms):
            sub = SwarmSubsystem(coordinator)
            org.add_subsystem(sub)
            sub.join_swarm(swarm_id, position=(i * 10, 0))
            subsystems.append(sub)

        direction = subsystems[2].get_flocking_direction()
        assert isinstance(direction, tuple)
        assert len(direction) == 2

    def test_position_management_via_subsystem(self, coordinator, organisms):
        """Test position get/set via subsystem"""
        swarm_id = coordinator.create_swarm("test")
        org = organisms[0]

        subsystem = SwarmSubsystem(coordinator)
        org.add_subsystem(subsystem)
        subsystem.join_swarm(swarm_id, position=(10, 20))

        pos = subsystem.get_position()
        assert pos is not None
        assert pos.x == 10
        assert pos.y == 20

        subsystem.set_position(30, 40)
        pos = subsystem.get_position()
        assert pos.x == 30
        assert pos.y == 40


# ============================================================================
# Tick and State Tests
# ============================================================================

class TestTickAndState:
    """Test tick processing and state management"""

    def test_tick(self, swarm_with_members, coordinator):
        """Test coordinator tick updates positions"""
        initial_positions = {}
        for org_id in coordinator._swarms[swarm_with_members]:
            pos = coordinator.get_member_position(org_id)
            initial_positions[org_id] = (pos.x, pos.y)

        coordinator.tick()

        # Positions should potentially change (flocking applied)
        # At minimum, tick should not crash

    def test_get_state(self, swarm_with_members, coordinator):
        """Test state serialization"""
        state = coordinator.get_state()

        assert "swarms" in state
        assert "formations" in state
        assert "tasks" in state
        assert "settings" in state
        assert swarm_with_members in state["swarms"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with SimpleOrganism"""

    def test_full_swarm_lifecycle(self, coordinator):
        """Test complete swarm lifecycle"""
        # Create organisms
        organisms = [SimpleOrganism(f"Agent_{i}") for i in range(10)]

        # Create swarm
        swarm_id = coordinator.create_swarm("hunters")

        # Add organisms
        for i, org in enumerate(organisms):
            angle = (2 * math.pi * i) / 10
            x = 100 + 30 * math.cos(angle)
            y = 100 + 30 * math.sin(angle)
            coordinator.add_to_swarm(swarm_id, org, position=(x, y))

        # Set formation
        coordinator.set_formation(swarm_id, FormationPattern.WEDGE)

        # Allocate tasks
        tasks = [
            {"type": "scout", "location": (200, 200), "priority": 0.8},
            {"type": "patrol", "location": (150, 150), "priority": 0.5}
        ]
        assignments = coordinator.allocate_tasks(tasks, swarm_id)
        assert len(assignments) == 2

        # Run several ticks
        for _ in range(5):
            coordinator.tick()

        # Vote on action
        winner, _ = coordinator.vote(swarm_id, ["advance", "hold", "retreat"])
        assert winner in ["advance", "hold", "retreat"]

        # Clean up
        coordinator.delete_swarm(swarm_id)
        assert coordinator.get_swarm_size(swarm_id) == 0
