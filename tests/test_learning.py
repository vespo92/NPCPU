"""
Tests for Learning & Memory Module

Tests for:
- MemorySystem multi-tier memory
- Memory storage, recall, and consolidation
- ReinforcementLearner Q-learning
- Experience replay and policy updates
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.memory_systems import (
    MemorySystem,
    MemoryType,
    MemoryTier,
    Memory,
)
from learning.reinforcement import (
    ReinforcementLearner,
    ExplorationStrategy,
    Experience,
)
from core.simple_organism import SimpleOrganism


# =============================================================================
# Memory System Tests
# =============================================================================

class TestMemory:
    """Tests for the Memory dataclass"""

    def test_memory_creation(self):
        """Test basic memory creation"""
        memory = Memory(
            content={"event": "found_food", "location": (10, 20)},
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
        )

        assert memory.content["event"] == "found_food"
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.importance == 0.8
        assert memory.tier == MemoryTier.SENSORY  # Default
        assert memory.id  # Should have generated ID

    def test_memory_strength_calculation(self):
        """Test memory strength decays over time"""
        memory = Memory(
            content={"test": "data"},
            importance=1.0,
            created_at=0,
            decay_rate=0.1,
        )

        # Initial strength
        strength_t0 = memory.strength(0)

        # Strength after time passes (without access)
        strength_t100 = memory.strength(100)

        assert strength_t0 > strength_t100  # Memory should decay

    def test_memory_access_updates_stats(self):
        """Test that accessing memory updates statistics"""
        memory = Memory(
            content={"test": "data"},
            created_at=0,
        )

        assert memory.access_count == 0

        memory.access(10)
        assert memory.access_count == 1
        assert memory.last_accessed == 10

        memory.access(20)
        assert memory.access_count == 2
        assert memory.last_accessed == 20

    def test_memory_serialization(self):
        """Test memory serialization/deserialization"""
        original = Memory(
            content={"event": "test", "value": 42},
            memory_type=MemoryType.SEMANTIC,
            tier=MemoryTier.WORKING,
            importance=0.7,
            emotional_valence=0.5,
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = Memory.from_dict(data)

        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.tier == original.tier
        assert restored.importance == original.importance
        assert restored.emotional_valence == original.emotional_valence


class TestMemorySystem:
    """Tests for the MemorySystem subsystem"""

    @pytest.fixture
    def memory_system(self):
        """Create a memory system for testing"""
        return MemorySystem(
            sensory_capacity=20,
            working_capacity=5,
            long_term_capacity=100,
            consolidation_threshold=0.6,
        )

    def test_memory_system_initialization(self, memory_system):
        """Test memory system initializes correctly"""
        assert memory_system.name == "memory"
        assert memory_system.sensory_capacity == 20
        assert memory_system.working_capacity == 5
        assert memory_system.enabled is True

    def test_store_memory(self, memory_system):
        """Test storing a memory"""
        memory = memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"event": "found_food", "location": (10, 20)},
            importance=0.5,
        )

        assert memory is not None
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.content["event"] == "found_food"

    def test_high_importance_goes_to_working(self, memory_system):
        """Test that high importance memories go to working memory"""
        memory = memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"important": "event"},
            importance=0.8,  # Above consolidation threshold
        )

        working = memory_system.get_working_memory()
        assert len(working) > 0
        assert any(m.id == memory.id for m in working)

    def test_very_high_importance_goes_to_long_term(self, memory_system):
        """Test that very high importance memories go to long-term"""
        memory = memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"critical": "event"},
            importance=0.95,
        )

        long_term = memory_system.get_long_term_memory()
        assert len(long_term) > 0
        assert any(m.id == memory.id for m in long_term)

    def test_recall_by_content(self, memory_system):
        """Test recalling memories by content matching"""
        # Store several memories
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"event": "found_food", "location": (10, 20)},
            importance=0.8,
        )
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"event": "predator_seen", "location": (30, 40)},
            importance=0.9,
        )
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"event": "found_food", "location": (50, 60)},
            importance=0.7,
        )

        # Recall food-related memories
        recalled = memory_system.recall({"event": "found_food"})

        # Should find the two food memories (may include others with lower relevance)
        food_memories = [m for m in recalled if m.content.get("event") == "found_food"]
        assert len(food_memories) >= 2

    def test_recall_by_type(self, memory_system):
        """Test filtering recall by memory type"""
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"event": "test1"},
            importance=0.8,
        )
        memory_system.store(
            memory_type=MemoryType.SEMANTIC,
            content={"fact": "test2"},
            importance=0.8,
        )

        # Recall only episodic
        episodic = memory_system.recall({}, memory_type=MemoryType.EPISODIC)
        semantic = memory_system.recall({}, memory_type=MemoryType.SEMANTIC)

        assert all(m.memory_type == MemoryType.EPISODIC for m in episodic)
        assert all(m.memory_type == MemoryType.SEMANTIC for m in semantic)

    def test_consolidation(self, memory_system):
        """Test memory consolidation from working to long-term"""
        # Store memories and access them multiple times
        for i in range(3):
            mem = memory_system.store(
                memory_type=MemoryType.EPISODIC,
                content={"iteration": i},
                importance=0.7,
            )
            # Access multiple times to increase consolidation chance
            for _ in range(3):
                memory_system.recall({"iteration": i})

        # Manually consolidate
        count = memory_system.consolidate()

        # Should have consolidated some memories
        long_term = memory_system.get_long_term_memory()
        assert len(long_term) > 0

    def test_forget(self, memory_system):
        """Test explicitly forgetting a memory"""
        memory = memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"to_forget": True},
            importance=0.95,  # Will go to long-term
        )

        # Verify it's stored
        long_term = memory_system.get_long_term_memory()
        assert any(m.id == memory.id for m in long_term)

        # Forget it
        result = memory_system.forget(memory.id)
        assert result is True

        # Verify it's gone
        long_term = memory_system.get_long_term_memory()
        assert not any(m.id == memory.id for m in long_term)

    def test_tick_decay(self, memory_system):
        """Test that tick causes memory decay"""
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"decay_test": True},
            importance=0.3,  # Low importance
        )

        initial_sensory = len(list(memory_system._sensory))

        # Run many ticks to cause decay
        for _ in range(100):
            memory_system.tick()

        # Some sensory memories should have decayed
        # (depending on decay rate and strength)

    def test_working_memory_capacity(self, memory_system):
        """Test working memory respects capacity limit"""
        # Store more than working capacity
        for i in range(10):
            memory_system.store(
                memory_type=MemoryType.EPISODIC,
                content={"item": i},
                importance=0.7,  # High enough to go to working
            )

        working = memory_system.get_working_memory()
        assert len(working) <= memory_system.working_capacity

    def test_state_serialization(self, memory_system):
        """Test full state serialization/deserialization"""
        # Store some memories
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"test": "memory1"},
            importance=0.9,
        )
        memory_system.store(
            memory_type=MemoryType.SEMANTIC,
            content={"fact": "memory2"},
            importance=0.8,
        )

        # Tick a few times
        for _ in range(5):
            memory_system.tick()

        # Get state
        state = memory_system.get_state()

        # Create new system and restore
        new_system = MemorySystem()
        new_system.set_state(state)

        assert new_system._current_tick == memory_system._current_tick
        assert len(new_system.get_long_term_memory()) == len(memory_system.get_long_term_memory())

    def test_integration_with_organism(self):
        """Test memory system integration with SimpleOrganism"""
        organism = SimpleOrganism("TestOrganism")
        memory_system = MemorySystem()
        organism.add_subsystem(memory_system)

        assert organism.get_subsystem("memory") is memory_system
        assert memory_system.owner is organism

        # Store a memory
        memory_system.store(
            memory_type=MemoryType.EPISODIC,
            content={"organism_memory": True},
            importance=0.8,
        )

        # Run organism tick (should also tick memory system)
        organism.tick()


# =============================================================================
# Reinforcement Learner Tests
# =============================================================================

class TestExperience:
    """Tests for the Experience dataclass"""

    def test_experience_creation(self):
        """Test basic experience creation"""
        exp = Experience(
            state="state1",
            action="eat",
            reward=10.0,
            next_state="state2",
        )

        assert exp.state == "state1"
        assert exp.action == "eat"
        assert exp.reward == 10.0
        assert exp.next_state == "state2"
        assert exp.done is False

    def test_experience_serialization(self):
        """Test experience serialization"""
        original = Experience(
            state="s1",
            action="a1",
            reward=5.0,
            next_state="s2",
            done=True,
            tick=100,
        )

        data = original.to_dict()
        restored = Experience.from_dict(data)

        assert restored.state == original.state
        assert restored.action == original.action
        assert restored.reward == original.reward
        assert restored.done == original.done
        assert restored.tick == original.tick


class TestReinforcementLearner:
    """Tests for the ReinforcementLearner subsystem"""

    @pytest.fixture
    def learner(self):
        """Create a learner for testing"""
        return ReinforcementLearner(
            learning_rate=0.1,
            discount_factor=0.95,
            exploration_rate=0.3,
            exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
            use_experience_replay=False,  # Disable for simpler tests
        )

    def test_learner_initialization(self, learner):
        """Test learner initializes correctly"""
        assert learner.name == "reinforcement_learner"
        assert learner.learning_rate == 0.1
        assert learner.discount_factor == 0.95
        assert learner.exploration_rate == 0.3
        assert learner.enabled is True

    def test_state_key_generation(self, learner):
        """Test state key generation from dict"""
        state1 = {"energy": 0.5, "threat": 0.2}
        state2 = {"energy": 0.5, "threat": 0.2}
        state3 = {"energy": 0.8, "threat": 0.1}

        key1 = learner.get_state_key(state1)
        key2 = learner.get_state_key(state2)
        key3 = learner.get_state_key(state3)

        # Same states should produce same key
        assert key1 == key2
        # Different states should produce different keys
        assert key1 != key3

    def test_q_value_operations(self, learner):
        """Test setting and getting Q-values"""
        state = "test_state"
        action = "test_action"

        # Initially zero
        assert learner.get_q_value(state, action) == 0.0

        # Set value
        learner.set_q_value(state, action, 5.0)
        assert learner.get_q_value(state, action) == 5.0

        # Update value
        learner.set_q_value(state, action, 10.0)
        assert learner.get_q_value(state, action) == 10.0

    def test_action_selection_greedy(self, learner):
        """Test greedy action selection"""
        state = "test_state"
        actions = ["eat", "flee", "explore"]

        # Set Q-values
        learner.set_q_value(state, "eat", 10.0)
        learner.set_q_value(state, "flee", 5.0)
        learner.set_q_value(state, "explore", 2.0)

        # Greedy should always choose highest
        for _ in range(10):
            action = learner.select_action(state, actions, greedy=True)
            assert action == "eat"

    def test_action_selection_explores(self, learner):
        """Test that non-greedy selection sometimes explores"""
        learner.exploration_rate = 1.0  # Always explore

        state = "test_state"
        actions = ["eat", "flee", "explore"]

        learner.set_q_value(state, "eat", 100.0)
        learner.set_q_value(state, "flee", 0.0)
        learner.set_q_value(state, "explore", 0.0)

        # With exploration=1.0, should see variety
        selected = set()
        for _ in range(100):
            action = learner.select_action(state, actions, greedy=False)
            selected.add(action)

        # Should have selected multiple different actions
        assert len(selected) > 1

    def test_learn_from_experience(self, learner):
        """Test Q-learning update"""
        state = "s1"
        action = "eat"
        reward = 10.0
        next_state = "s2"

        # Set up next state Q-values
        learner.set_q_value(next_state, "eat", 5.0)
        learner.set_q_value(next_state, "flee", 2.0)

        # Initial Q-value
        initial_q = learner.get_q_value(state, action)
        assert initial_q == 0.0

        # Learn
        td_error = learner.learn_from_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            available_actions=["eat", "flee"],
        )

        # Q-value should have updated
        new_q = learner.get_q_value(state, action)
        assert new_q > initial_q

        # TD error should be returned
        assert td_error > 0

    def test_learn_terminal_state(self, learner):
        """Test learning when episode ends"""
        state = "s1"
        action = "die"
        reward = -100.0
        next_state = "terminal"

        learner.learn_from_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=True,
        )

        # Should have negative Q-value
        q = learner.get_q_value(state, action)
        assert q < 0

    def test_exploration_decay(self):
        """Test exploration rate decay"""
        learner = ReinforcementLearner(
            exploration_rate=1.0,
            exploration_strategy=ExplorationStrategy.DECAYING_EPSILON,
            exploration_decay=0.9,
            min_exploration_rate=0.1,
        )

        initial_rate = learner.exploration_rate

        # Run ticks
        for _ in range(10):
            learner.tick()

        # Exploration should have decayed
        assert learner.exploration_rate < initial_rate
        assert learner.exploration_rate >= learner.min_exploration_rate

    def test_softmax_selection(self):
        """Test softmax action selection"""
        learner = ReinforcementLearner(
            exploration_strategy=ExplorationStrategy.SOFTMAX,
            exploration_rate=0.5,
        )

        state = "test_state"
        actions = ["a", "b", "c"]

        learner.set_q_value(state, "a", 10.0)
        learner.set_q_value(state, "b", 5.0)
        learner.set_q_value(state, "c", 0.0)

        # Sample many times
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(1000):
            action = learner.select_action(state, actions)
            counts[action] += 1

        # Higher Q-value actions should be selected more often
        assert counts["a"] > counts["c"]

    def test_ucb_selection(self):
        """Test UCB action selection"""
        learner = ReinforcementLearner(
            exploration_strategy=ExplorationStrategy.UCB,
        )

        state = "test_state"
        actions = ["a", "b", "c"]

        # First selections should try each action at least once
        selected = set()
        for _ in range(10):
            action = learner.select_action(state, actions)
            selected.add(action)
            learner.learn_from_experience(state, action, 1.0, state)

        assert len(selected) == 3  # All actions tried

    def test_get_policy(self, learner):
        """Test getting policy for a state"""
        state = "test_state"

        learner.set_q_value(state, "eat", 10.0)
        learner.set_q_value(state, "flee", 5.0)

        policy = learner.get_policy(state)

        assert "eat" in policy
        assert "flee" in policy
        assert policy["eat"] > policy["flee"]  # Higher Q = higher probability
        assert abs(sum(policy.values()) - 1.0) < 0.01  # Sums to 1

    def test_reset_learning(self, learner):
        """Test resetting learned values"""
        # Learn something
        learner.set_q_value("s1", "a1", 10.0)
        learner.learn_from_experience("s1", "a1", 5.0, "s2")

        # Reset
        learner.reset_learning()

        # Q-table should be empty
        assert learner.get_q_value("s1", "a1") == 0.0
        assert learner._stats["total_experiences"] == 0

    def test_state_serialization(self, learner):
        """Test full state serialization"""
        # Learn some things
        learner.set_q_value("s1", "a1", 10.0)
        learner.set_q_value("s1", "a2", 5.0)
        learner.learn_from_experience("s1", "a1", 5.0, "s2")

        # Tick
        for _ in range(5):
            learner.tick()

        # Save state
        state = learner.get_state()

        # Create new learner and restore
        new_learner = ReinforcementLearner()
        new_learner.set_state(state)

        assert new_learner.get_q_value("s1", "a1") == learner.get_q_value("s1", "a1")
        assert new_learner._current_tick == learner._current_tick

    def test_integration_with_organism(self):
        """Test learner integration with SimpleOrganism"""
        organism = SimpleOrganism("TestOrganism")
        learner = ReinforcementLearner()
        organism.add_subsystem(learner)

        assert organism.get_subsystem("reinforcement_learner") is learner
        assert learner.owner is organism

        # Run organism tick
        organism.tick()

    def test_experience_replay(self):
        """Test experience replay functionality"""
        learner = ReinforcementLearner(
            use_experience_replay=True,
            replay_buffer_size=100,
            batch_size=5,
        )

        # Generate experiences
        for i in range(20):
            learner.learn_from_experience(
                state=f"s{i}",
                action="a",
                reward=float(i),
                next_state=f"s{i+1}",
            )

        # Replay buffer should have experiences
        assert len(learner._replay_buffer) == 20

        # Run ticks to trigger replay
        initial_updates = learner._stats["total_updates"]
        for _ in range(20):
            learner.tick()

        # Should have done some replay updates
        # (replay happens every 10 ticks)


# =============================================================================
# Integration Tests
# =============================================================================

class TestLearningIntegration:
    """Integration tests for learning module with organism"""

    def test_full_learning_cycle(self):
        """Test a full learning cycle with organism"""
        # Create organism with learning subsystems
        organism = SimpleOrganism("LearningAgent")
        memory = MemorySystem()
        learner = ReinforcementLearner(
            learning_rate=0.1,
            exploration_rate=0.2,
        )
        organism.add_subsystem(memory)
        organism.add_subsystem(learner)

        actions = ["eat", "flee", "explore", "rest"]

        # Simulate several steps
        for step in range(50):
            # Get state
            energy_sub = organism.get_subsystem("energy")
            state_dict = {
                "energy": energy_sub.percentage if energy_sub else 1.0,
                "age": organism.age,
            }
            state_key = learner.get_state_key(state_dict)

            # Select action
            action = learner.select_action(state_key, actions)

            # Execute action and get reward (simplified)
            reward = 1.0 if action == "rest" else -0.5

            # Store memory of action
            memory.store(
                memory_type=MemoryType.EPISODIC,
                content={"action": action, "step": step},
                importance=0.5 + abs(reward) / 10,
            )

            # Tick organism
            organism.tick()

            # Get new state
            new_state_dict = {
                "energy": energy_sub.percentage if energy_sub else 1.0,
                "age": organism.age,
            }
            new_state_key = learner.get_state_key(new_state_dict)

            # Learn
            learner.learn_from_experience(
                state=state_key,
                action=action,
                reward=reward,
                next_state=new_state_key,
                available_actions=actions,
            )

        # Should have learned something
        assert learner._stats["total_experiences"] > 0
        assert memory._stats["total_stored"] > 0

    def test_memory_assisted_learning(self):
        """Test that memories can assist learning"""
        organism = SimpleOrganism("Agent")
        memory = MemorySystem(consolidation_threshold=0.5)
        learner = ReinforcementLearner()
        organism.add_subsystem(memory)
        organism.add_subsystem(learner)

        # Store a memory about good action
        memory.store(
            memory_type=MemoryType.SEMANTIC,
            content={"action": "eat", "result": "good"},
            importance=0.9,
        )

        # Recall memories
        recalled = memory.recall({"action": "eat"})
        assert len(recalled) > 0

        # Use recalled memory to inform learning
        for mem in recalled:
            if mem.content.get("result") == "good":
                # Reinforce the good action
                learner.set_q_value("hungry_state", mem.content["action"], 5.0)

        # Verify learning was applied
        assert learner.get_q_value("hungry_state", "eat") > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
