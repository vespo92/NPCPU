"""
Tests for Core Abstractions

Unit tests for all base classes and behavior tree components.
"""

import pytest
from datetime import datetime
from typing import Dict, Any, Optional

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from core.abstractions import (
    # Enums
    LifecyclePhase, OrganismCapability,
    # Protocols
    Tickable, Serializable, Identifiable,
    # Base classes
    EntityState, BaseSubsystem, BaseOrganism, BaseWorld, BasePopulation,
    # Behavior tree
    BehaviorNode, CompositeNode, SequenceNode, SelectorNode,
    ActionNode, ConditionNode, BehaviorTree
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockSubsystem(BaseSubsystem):
    """Mock subsystem for testing"""

    def __init__(self, name: str = "mock"):
        super().__init__(name)
        self.tick_count = 0
        self.custom_value = 0

    def tick(self) -> None:
        if self.enabled:
            self.tick_count += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "tick_count": self.tick_count,
            "custom_value": self.custom_value
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.tick_count = state.get("tick_count", 0)
        self.custom_value = state.get("custom_value", 0)


class MockOrganism(BaseOrganism):
    """Mock organism for testing"""

    def __init__(self, name: str = "", **kwargs):
        super().__init__(name, **kwargs)
        self.perceived_stimuli: Dict[str, Any] = {}
        self.decided_action: Optional[str] = None
        self.action_results: list = []

    def tick(self) -> None:
        self._age += 1
        for subsystem in self._subsystems.values():
            if subsystem.enabled:
                subsystem.tick()

    def perceive(self, stimuli: Dict[str, Any]) -> None:
        self.perceived_stimuli = stimuli.copy()

    def decide(self) -> Optional[str]:
        if self.perceived_stimuli.get("threat"):
            self.decided_action = "flee"
        elif self.perceived_stimuli.get("food"):
            self.decided_action = "eat"
        else:
            self.decided_action = "explore"
        return self.decided_action

    def act(self, action: str) -> Any:
        result = f"performed_{action}"
        self.action_results.append(result)
        return result


class MockWorld(BaseWorld):
    """Mock world for testing"""

    def __init__(self, name: str = "TestWorld", **kwargs):
        super().__init__(name, **kwargs)
        self.resources: Dict[str, float] = {"food": 100.0, "water": 100.0}
        self.events_triggered: list = []

    def tick(self) -> None:
        self._tick_count += 1
        for pop in self._populations.values():
            pop.tick()

    def get_stimuli_at(self, location: Any) -> Dict[str, Any]:
        return {
            "location": location,
            "light": 0.8,
            "temperature": 25.0
        }

    def get_resources(self, resource_type: str) -> float:
        return self.resources.get(resource_type, 0.0)

    def consume_resource(self, resource_type: str, amount: float) -> float:
        available = self.resources.get(resource_type, 0.0)
        consumed = min(amount, available)
        self.resources[resource_type] = available - consumed
        return consumed

    def trigger_event(self, event_type: str, **kwargs) -> None:
        self.events_triggered.append((event_type, kwargs))


class MockPopulation(BasePopulation):
    """Mock population for testing"""

    def __init__(self, name: str = "TestPopulation", **kwargs):
        super().__init__(name, **kwargs)
        self.tick_count = 0

    def tick(self) -> None:
        self.tick_count += 1
        for organism in list(self._organisms.values()):
            organism.tick()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "tick_count": self.tick_count,
            "alive": sum(1 for o in self._organisms.values() if o.is_alive)
        }


# =============================================================================
# Tests: Enums
# =============================================================================

class TestLifecyclePhase:
    """Tests for LifecyclePhase enum"""

    def test_all_phases_exist(self):
        """Verify all lifecycle phases are defined"""
        phases = [
            LifecyclePhase.INITIALIZING,
            LifecyclePhase.NASCENT,
            LifecyclePhase.DEVELOPING,
            LifecyclePhase.MATURE,
            LifecyclePhase.DECLINING,
            LifecyclePhase.TERMINAL,
            LifecyclePhase.ENDED
        ]
        assert len(phases) == 7

    def test_phases_are_unique(self):
        """Verify all phase values are unique"""
        values = [p.value for p in LifecyclePhase]
        assert len(values) == len(set(values))


class TestOrganismCapability:
    """Tests for OrganismCapability enum"""

    def test_all_capabilities_exist(self):
        """Verify all capabilities are defined"""
        expected = [
            "perception", "locomotion", "manipulation", "communication",
            "memory", "learning", "reasoning", "planning",
            "social", "reproduction", "self_repair", "adaptation"
        ]
        actual = [c.value for c in OrganismCapability]
        assert sorted(expected) == sorted(actual)

    def test_capability_values_are_strings(self):
        """Verify capability values are strings"""
        for cap in OrganismCapability:
            assert isinstance(cap.value, str)


# =============================================================================
# Tests: EntityState
# =============================================================================

class TestEntityState:
    """Tests for EntityState dataclass"""

    def test_default_id_generation(self):
        """Test that ID is auto-generated"""
        state = EntityState()
        assert state.id is not None
        assert len(state.id) == 36  # UUID format

    def test_unique_ids(self):
        """Test that each state gets unique ID"""
        state1 = EntityState()
        state2 = EntityState()
        assert state1.id != state2.id

    def test_timestamps(self):
        """Test timestamp fields"""
        state = EntityState()
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)

    def test_update_method(self):
        """Test update modifies updated_at"""
        state = EntityState()
        original = state.updated_at
        import time
        time.sleep(0.01)
        state.update()
        assert state.updated_at > original

    def test_metadata_default(self):
        """Test metadata defaults to empty dict"""
        state = EntityState()
        assert state.metadata == {}
        assert isinstance(state.metadata, dict)


# =============================================================================
# Tests: BaseSubsystem
# =============================================================================

class TestBaseSubsystem:
    """Tests for BaseSubsystem"""

    def test_initialization(self):
        """Test subsystem initialization"""
        sub = MockSubsystem("test_subsystem")
        assert sub.name == "test_subsystem"
        assert sub.owner is None
        assert sub.enabled is True

    def test_enable_disable(self):
        """Test enable/disable functionality"""
        sub = MockSubsystem()
        assert sub.enabled is True

        sub.disable()
        assert sub.enabled is False

        sub.enable()
        assert sub.enabled is True

    def test_tick_when_enabled(self):
        """Test tick increments when enabled"""
        sub = MockSubsystem()
        sub.tick()
        assert sub.tick_count == 1
        sub.tick()
        assert sub.tick_count == 2

    def test_tick_when_disabled(self):
        """Test tick does not increment when disabled"""
        sub = MockSubsystem()
        sub.disable()
        sub.tick()
        assert sub.tick_count == 0

    def test_state_serialization(self):
        """Test get_state returns correct state"""
        sub = MockSubsystem()
        sub.tick_count = 5
        sub.custom_value = 42

        state = sub.get_state()
        assert state["tick_count"] == 5
        assert state["custom_value"] == 42

    def test_state_restoration(self):
        """Test set_state restores state correctly"""
        sub = MockSubsystem()
        sub.set_state({"tick_count": 10, "custom_value": 99})

        assert sub.tick_count == 10
        assert sub.custom_value == 99

    def test_reset(self):
        """Test reset clears internal state"""
        sub = MockSubsystem()
        sub._state["test"] = "value"
        sub.reset()
        assert sub._state == {}


# =============================================================================
# Tests: BaseOrganism
# =============================================================================

class TestBaseOrganism:
    """Tests for BaseOrganism"""

    def test_initialization_with_name(self):
        """Test organism initialization with name"""
        org = MockOrganism("TestOrganism")
        assert org.name == "TestOrganism"
        assert org.id is not None
        assert org.age == 0
        assert org.is_alive is True

    def test_initialization_without_name(self):
        """Test organism gets auto-generated name"""
        org = MockOrganism()
        assert org.name.startswith("Organism_")

    def test_initial_phase(self):
        """Test initial lifecycle phase"""
        org = MockOrganism()
        assert org.phase == LifecyclePhase.INITIALIZING

    def test_subsystem_management(self):
        """Test adding/getting/removing subsystems"""
        org = MockOrganism()
        sub = MockSubsystem("test_sub")

        org.add_subsystem(sub)
        assert org.get_subsystem("test_sub") is sub
        assert sub.owner is org

        removed = org.remove_subsystem("test_sub")
        assert removed is sub
        assert org.get_subsystem("test_sub") is None

    def test_subsystems_property(self):
        """Test subsystems property returns copy"""
        org = MockOrganism()
        sub1 = MockSubsystem("sub1")
        sub2 = MockSubsystem("sub2")

        org.add_subsystem(sub1)
        org.add_subsystem(sub2)

        subs = org.subsystems
        assert len(subs) == 2
        assert "sub1" in subs
        assert "sub2" in subs

    def test_capability_management(self):
        """Test capability get/set"""
        org = MockOrganism()

        org.set_capability(OrganismCapability.PERCEPTION, 0.8)
        assert org.get_capability(OrganismCapability.PERCEPTION) == 0.8

        # Test clamping
        org.set_capability(OrganismCapability.PERCEPTION, 1.5)
        assert org.get_capability(OrganismCapability.PERCEPTION) == 1.0

        org.set_capability(OrganismCapability.PERCEPTION, -0.5)
        assert org.get_capability(OrganismCapability.PERCEPTION) == 0.0

    def test_capability_default(self):
        """Test unset capability returns 0"""
        org = MockOrganism()
        assert org.get_capability(OrganismCapability.REASONING) == 0.0

    def test_trait_management(self):
        """Test trait get/set"""
        org = MockOrganism()

        org.set_trait("aggression", 0.7)
        assert org.get_trait("aggression") == 0.7
        assert org.get_trait("unknown", 0.5) == 0.5

    def test_tick_increments_age(self):
        """Test tick increments organism age"""
        org = MockOrganism()
        org.tick()
        assert org.age == 1
        org.tick()
        assert org.age == 2

    def test_tick_updates_subsystems(self):
        """Test tick updates all subsystems"""
        org = MockOrganism()
        sub = MockSubsystem("test")
        org.add_subsystem(sub)

        org.tick()
        assert sub.tick_count == 1

    def test_perceive(self):
        """Test perceive stores stimuli"""
        org = MockOrganism()
        org.perceive({"light": 0.5, "sound": 0.3})
        assert org.perceived_stimuli["light"] == 0.5

    def test_decide(self):
        """Test decide returns action based on stimuli"""
        org = MockOrganism()

        org.perceive({"threat": True})
        assert org.decide() == "flee"

        org.perceive({"food": True})
        assert org.decide() == "eat"

        org.perceive({})
        assert org.decide() == "explore"

    def test_act(self):
        """Test act executes action"""
        org = MockOrganism()
        result = org.act("flee")
        assert result == "performed_flee"
        assert "performed_flee" in org.action_results

    def test_die(self):
        """Test die marks organism as dead"""
        org = MockOrganism()
        assert org.is_alive is True

        org.die("test_cause")
        assert org.is_alive is False
        assert org.phase == LifecyclePhase.ENDED

    def test_to_dict(self):
        """Test serialization to dict"""
        org = MockOrganism("SerializeTest")
        org.set_trait("speed", 1.5)
        org.set_capability(OrganismCapability.LOCOMOTION, 0.9)

        data = org.to_dict()
        assert data["name"] == "SerializeTest"
        assert data["alive"] is True
        assert "locomotion" in data["capabilities"]
        assert data["traits"]["speed"] == 1.5


# =============================================================================
# Tests: BaseWorld
# =============================================================================

class TestBaseWorld:
    """Tests for BaseWorld"""

    def test_initialization(self):
        """Test world initialization"""
        world = MockWorld("TestWorld")
        assert world.name == "TestWorld"
        assert world.id is not None
        assert world.tick_count == 0

    def test_population_management(self):
        """Test adding/getting/removing populations"""
        world = MockWorld()
        pop = MockPopulation("TestPop")

        world.add_population(pop)
        assert world.get_population("TestPop") is pop

        removed = world.remove_population("TestPop")
        assert removed is pop
        assert world.get_population("TestPop") is None

    def test_populations_property(self):
        """Test populations property returns copy"""
        world = MockWorld()
        pop = MockPopulation("Pop1")
        world.add_population(pop)

        pops = world.populations
        assert "Pop1" in pops

    def test_tick_increments_count(self):
        """Test tick increments tick_count"""
        world = MockWorld()
        world.tick()
        assert world.tick_count == 1

    def test_get_stimuli_at(self):
        """Test getting stimuli at location"""
        world = MockWorld()
        stimuli = world.get_stimuli_at((10, 20))
        assert stimuli["location"] == (10, 20)
        assert "light" in stimuli

    def test_resource_management(self):
        """Test resource get/consume"""
        world = MockWorld()
        assert world.get_resources("food") == 100.0

        consumed = world.consume_resource("food", 30.0)
        assert consumed == 30.0
        assert world.get_resources("food") == 70.0

    def test_resource_overconsumption(self):
        """Test consuming more than available"""
        world = MockWorld()
        world.resources["food"] = 10.0

        consumed = world.consume_resource("food", 50.0)
        assert consumed == 10.0
        assert world.get_resources("food") == 0.0

    def test_trigger_event(self):
        """Test event triggering"""
        world = MockWorld()
        world.trigger_event("disaster", severity=5)

        assert len(world.events_triggered) == 1
        assert world.events_triggered[0] == ("disaster", {"severity": 5})

    def test_to_dict(self):
        """Test world serialization"""
        world = MockWorld("SerializeWorld")
        pop = MockPopulation("Pop1")
        world.add_population(pop)

        data = world.to_dict()
        assert data["name"] == "SerializeWorld"
        assert "Pop1" in data["populations"]


# =============================================================================
# Tests: BasePopulation
# =============================================================================

class TestBasePopulation:
    """Tests for BasePopulation"""

    def test_initialization(self):
        """Test population initialization"""
        pop = MockPopulation("TestPop")
        assert pop.name == "TestPop"
        assert pop.id is not None
        assert pop.size == 0

    def test_organism_management(self):
        """Test adding/getting/removing organisms"""
        pop = MockPopulation()
        org = MockOrganism("Org1")

        pop.add_organism(org)
        assert pop.size == 1
        assert pop.get_organism(org.id) is org

        removed = pop.remove_organism(org.id)
        assert removed is org
        assert pop.size == 0

    def test_iter_organisms(self):
        """Test iterating over organisms"""
        pop = MockPopulation()
        orgs = [MockOrganism(f"Org{i}") for i in range(3)]
        for org in orgs:
            pop.add_organism(org)

        iterated = list(pop.iter_organisms())
        assert len(iterated) == 3

    def test_iter_alive(self):
        """Test iterating over alive organisms only"""
        pop = MockPopulation()
        org1 = MockOrganism("Alive")
        org2 = MockOrganism("Dead")
        org2.die("test")

        pop.add_organism(org1)
        pop.add_organism(org2)

        alive = list(pop.iter_alive())
        assert len(alive) == 1
        assert alive[0].name == "Alive"

    def test_tick_updates_organisms(self):
        """Test tick updates all organisms"""
        pop = MockPopulation()
        org = MockOrganism()
        pop.add_organism(org)

        pop.tick()
        assert org.age == 1
        assert pop.tick_count == 1

    def test_get_stats(self):
        """Test population statistics"""
        pop = MockPopulation()
        org1 = MockOrganism("Alive")
        org2 = MockOrganism("Dead")
        org2.die("test")

        pop.add_organism(org1)
        pop.add_organism(org2)

        stats = pop.get_stats()
        assert stats["size"] == 2
        assert stats["alive"] == 1

    def test_to_dict(self):
        """Test population serialization"""
        pop = MockPopulation("SerializePop")
        org = MockOrganism("Org1")
        pop.add_organism(org)

        data = pop.to_dict()
        assert data["name"] == "SerializePop"
        assert data["size"] == 1
        assert org.id in data["organism_ids"]


# =============================================================================
# Tests: Behavior Tree
# =============================================================================

class TestBehaviorNode:
    """Tests for BehaviorNode status"""

    def test_status_enum(self):
        """Test behavior node status values"""
        assert BehaviorNode.Status.SUCCESS is not None
        assert BehaviorNode.Status.FAILURE is not None
        assert BehaviorNode.Status.RUNNING is not None


class TestActionNode:
    """Tests for ActionNode"""

    def test_success_action(self):
        """Test action that returns True"""
        action = ActionNode("test", lambda ctx: True)
        status = action.execute({})
        assert status == BehaviorNode.Status.SUCCESS

    def test_failure_action(self):
        """Test action that returns False"""
        action = ActionNode("test", lambda ctx: False)
        status = action.execute({})
        assert status == BehaviorNode.Status.FAILURE

    def test_exception_action(self):
        """Test action that raises exception"""
        def failing_action(ctx):
            raise ValueError("Test error")

        action = ActionNode("test", failing_action)
        status = action.execute({})
        assert status == BehaviorNode.Status.FAILURE

    def test_context_passing(self):
        """Test context is passed to action"""
        received_ctx = {}

        def capture_action(ctx):
            received_ctx.update(ctx)
            return True

        action = ActionNode("test", capture_action)
        action.execute({"key": "value"})
        assert received_ctx["key"] == "value"


class TestConditionNode:
    """Tests for ConditionNode"""

    def test_true_condition(self):
        """Test condition that returns True"""
        cond = ConditionNode("test", lambda ctx: True)
        status = cond.execute({})
        assert status == BehaviorNode.Status.SUCCESS

    def test_false_condition(self):
        """Test condition that returns False"""
        cond = ConditionNode("test", lambda ctx: False)
        status = cond.execute({})
        assert status == BehaviorNode.Status.FAILURE

    def test_exception_condition(self):
        """Test condition that raises exception"""
        cond = ConditionNode("test", lambda ctx: 1/0)
        status = cond.execute({})
        assert status == BehaviorNode.Status.FAILURE


class TestSequenceNode:
    """Tests for SequenceNode"""

    def test_all_succeed(self):
        """Test sequence with all successful children"""
        children = [
            ActionNode("a", lambda ctx: True),
            ActionNode("b", lambda ctx: True),
            ActionNode("c", lambda ctx: True)
        ]
        seq = SequenceNode("test", children)
        status = seq.execute({})
        assert status == BehaviorNode.Status.SUCCESS

    def test_first_fails(self):
        """Test sequence stops at first failure"""
        execution_order = []

        def make_action(name, result):
            def action(ctx):
                execution_order.append(name)
                return result
            return action

        children = [
            ActionNode("a", make_action("a", False)),
            ActionNode("b", make_action("b", True))
        ]
        seq = SequenceNode("test", children)
        status = seq.execute({})

        assert status == BehaviorNode.Status.FAILURE
        assert execution_order == ["a"]  # b was not executed

    def test_empty_sequence(self):
        """Test empty sequence returns success"""
        seq = SequenceNode("empty", [])
        status = seq.execute({})
        assert status == BehaviorNode.Status.SUCCESS


class TestSelectorNode:
    """Tests for SelectorNode"""

    def test_first_succeeds(self):
        """Test selector stops at first success"""
        execution_order = []

        def make_action(name, result):
            def action(ctx):
                execution_order.append(name)
                return result
            return action

        children = [
            ActionNode("a", make_action("a", True)),
            ActionNode("b", make_action("b", True))
        ]
        sel = SelectorNode("test", children)
        status = sel.execute({})

        assert status == BehaviorNode.Status.SUCCESS
        assert execution_order == ["a"]  # b was not executed

    def test_all_fail(self):
        """Test selector fails when all children fail"""
        children = [
            ActionNode("a", lambda ctx: False),
            ActionNode("b", lambda ctx: False)
        ]
        sel = SelectorNode("test", children)
        status = sel.execute({})
        assert status == BehaviorNode.Status.FAILURE

    def test_empty_selector(self):
        """Test empty selector returns failure"""
        sel = SelectorNode("empty", [])
        status = sel.execute({})
        assert status == BehaviorNode.Status.FAILURE


class TestCompositeNode:
    """Tests for CompositeNode"""

    def test_add_child(self):
        """Test adding children"""
        seq = SequenceNode("test")
        seq.add_child(ActionNode("a", lambda ctx: True))
        seq.add_child(ActionNode("b", lambda ctx: True))

        assert len(seq.children) == 2

    def test_reset_propagates(self):
        """Test reset propagates to children"""
        reset_called = {"count": 0}

        class TrackingAction(ActionNode):
            def reset(self):
                reset_called["count"] += 1

        children = [
            TrackingAction("a", lambda ctx: True),
            TrackingAction("b", lambda ctx: True)
        ]
        seq = SequenceNode("test", children)
        seq.reset()

        assert reset_called["count"] == 2


class TestBehaviorTree:
    """Tests for BehaviorTree"""

    def test_execute_root(self):
        """Test tree executes root node"""
        action = ActionNode("root", lambda ctx: True)
        tree = BehaviorTree(action)
        status = tree.execute({})
        assert status == BehaviorNode.Status.SUCCESS

    def test_complex_tree(self):
        """Test complex behavior tree"""
        # Build a tree: Selector(Sequence(Condition, Action), Action)
        tree = BehaviorTree(
            SelectorNode("root", [
                SequenceNode("check_and_act", [
                    ConditionNode("check", lambda ctx: ctx.get("ready", False)),
                    ActionNode("do_work", lambda ctx: True)
                ]),
                ActionNode("default", lambda ctx: True)
            ])
        )

        # First case: condition fails, falls through to default
        status = tree.execute({"ready": False})
        assert status == BehaviorNode.Status.SUCCESS

        # Second case: condition passes, does work
        status = tree.execute({"ready": True})
        assert status == BehaviorNode.Status.SUCCESS

    def test_reset(self):
        """Test tree reset"""
        reset_called = {"value": False}

        class TrackingNode(ActionNode):
            def reset(self):
                reset_called["value"] = True

        root = TrackingNode("root", lambda ctx: True)
        tree = BehaviorTree(root)
        tree.reset()

        assert reset_called["value"] is True


# =============================================================================
# Tests: Protocol Compliance
# =============================================================================

class TestProtocolCompliance:
    """Tests for protocol compliance"""

    def test_tickable_protocol(self):
        """Test Tickable protocol implementation"""
        org = MockOrganism()
        assert isinstance(org, Tickable)

        sub = MockSubsystem()
        assert isinstance(sub, Tickable)

    def test_identifiable_protocol(self):
        """Test Identifiable protocol implementation"""
        org = MockOrganism("Test")
        assert isinstance(org, Identifiable)
        assert org.id is not None
        assert org.name == "Test"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
