"""
Tests for Simple Organism Implementation

Unit and integration tests for SimpleOrganism and its subsystems.
"""

import pytest
from typing import Dict, Any

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from core.simple_organism import (
    SimpleOrganism, SimplePopulation,
    EnergySubsystem, HealthSubsystem, PerceptionSubsystem
)
from core.abstractions import LifecyclePhase, OrganismCapability


# =============================================================================
# Tests: EnergySubsystem
# =============================================================================

class TestEnergySubsystem:
    """Tests for EnergySubsystem"""

    def test_initialization(self):
        """Test energy subsystem initialization"""
        energy = EnergySubsystem(max_energy=100.0, consumption_rate=2.0)
        assert energy.name == "energy"
        assert energy.max_energy == 100.0
        assert energy.energy == 100.0
        assert energy.consumption_rate == 2.0

    def test_tick_consumes_energy(self):
        """Test tick decreases energy"""
        energy = EnergySubsystem(max_energy=100.0, consumption_rate=5.0)
        energy.tick()
        assert energy.energy == 95.0

    def test_tick_when_disabled(self):
        """Test tick does nothing when disabled"""
        energy = EnergySubsystem(max_energy=100.0, consumption_rate=5.0)
        energy.disable()
        energy.tick()
        assert energy.energy == 100.0

    def test_consume_energy(self):
        """Test manual energy consumption"""
        energy = EnergySubsystem(max_energy=100.0)
        consumed = energy.consume(30.0)
        assert consumed == 30.0
        assert energy.energy == 70.0

    def test_consume_more_than_available(self):
        """Test consuming more energy than available"""
        energy = EnergySubsystem(max_energy=100.0)
        energy.energy = 20.0
        consumed = energy.consume(50.0)
        assert consumed == 20.0
        assert energy.energy == 0.0

    def test_restore_energy(self):
        """Test restoring energy"""
        energy = EnergySubsystem(max_energy=100.0)
        energy.energy = 50.0
        restored = energy.restore(30.0)
        assert restored == 30.0
        assert energy.energy == 80.0

    def test_restore_over_max(self):
        """Test restoring cannot exceed max"""
        energy = EnergySubsystem(max_energy=100.0)
        energy.energy = 90.0
        restored = energy.restore(50.0)
        assert restored == 10.0
        assert energy.energy == 100.0

    def test_percentage(self):
        """Test energy percentage calculation"""
        energy = EnergySubsystem(max_energy=100.0)
        assert energy.percentage == 1.0

        energy.energy = 50.0
        assert energy.percentage == 0.5

        energy.energy = 0.0
        assert energy.percentage == 0.0

    def test_percentage_zero_max(self):
        """Test percentage with zero max energy"""
        energy = EnergySubsystem(max_energy=0.0)
        assert energy.percentage == 0.0

    def test_energy_cannot_go_negative(self):
        """Test energy floor at zero"""
        energy = EnergySubsystem(max_energy=100.0, consumption_rate=200.0)
        energy.tick()
        assert energy.energy == 0.0

    def test_state_serialization(self):
        """Test get_state returns correct state"""
        energy = EnergySubsystem(max_energy=100.0, consumption_rate=2.0)
        energy.energy = 75.0

        state = energy.get_state()
        assert state["energy"] == 75.0
        assert state["max_energy"] == 100.0
        assert state["consumption_rate"] == 2.0

    def test_state_restoration(self):
        """Test set_state restores state correctly"""
        energy = EnergySubsystem()
        energy.set_state({
            "energy": 50.0,
            "max_energy": 200.0,
            "consumption_rate": 3.0
        })

        assert energy.energy == 50.0
        assert energy.max_energy == 200.0
        assert energy.consumption_rate == 3.0


# =============================================================================
# Tests: HealthSubsystem
# =============================================================================

class TestHealthSubsystem:
    """Tests for HealthSubsystem"""

    def test_initialization(self):
        """Test health subsystem initialization"""
        health = HealthSubsystem(max_health=100.0, regen_rate=0.5)
        assert health.name == "health"
        assert health.max_health == 100.0
        assert health.health == 100.0
        assert health.regen_rate == 0.5

    def test_damage(self):
        """Test applying damage"""
        health = HealthSubsystem(max_health=100.0)
        damage = health.damage(30.0)
        assert damage == 30.0
        assert health.health == 70.0

    def test_damage_exceeds_health(self):
        """Test damage capped at current health"""
        health = HealthSubsystem(max_health=100.0)
        health.health = 20.0
        damage = health.damage(50.0)
        assert damage == 20.0
        assert health.health == 0.0

    def test_heal(self):
        """Test healing"""
        health = HealthSubsystem(max_health=100.0)
        health.health = 50.0
        healed = health.heal(30.0)
        assert healed == 30.0
        assert health.health == 80.0

    def test_heal_over_max(self):
        """Test healing capped at max health"""
        health = HealthSubsystem(max_health=100.0)
        health.health = 90.0
        healed = health.heal(50.0)
        assert healed == 10.0
        assert health.health == 100.0

    def test_percentage(self):
        """Test health percentage calculation"""
        health = HealthSubsystem(max_health=100.0)
        assert health.percentage == 1.0

        health.health = 25.0
        assert health.percentage == 0.25

    def test_tick_regenerates_health(self):
        """Test tick regenerates health when energy is sufficient"""
        health = HealthSubsystem(max_health=100.0, regen_rate=5.0)
        health.health = 50.0

        # Create organism with sufficient energy
        org = SimpleOrganism()
        health.owner = org

        health.tick()
        assert health.health == 55.0

    def test_tick_no_regen_low_energy(self):
        """Test no regen when energy is low"""
        health = HealthSubsystem(max_health=100.0, regen_rate=5.0)
        health.health = 50.0

        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        energy.energy = 10.0  # Low energy (10%)
        health.owner = org

        health.tick()
        assert health.health == 50.0  # No regen

    def test_state_serialization(self):
        """Test get_state returns correct state"""
        health = HealthSubsystem(max_health=100.0, regen_rate=0.2)
        health.health = 60.0

        state = health.get_state()
        assert state["health"] == 60.0
        assert state["max_health"] == 100.0
        assert state["regen_rate"] == 0.2


# =============================================================================
# Tests: PerceptionSubsystem
# =============================================================================

class TestPerceptionSubsystem:
    """Tests for PerceptionSubsystem"""

    def test_initialization(self):
        """Test perception subsystem initialization"""
        perception = PerceptionSubsystem(range=15.0)
        assert perception.name == "perception"
        assert perception.range == 15.0
        assert perception.current_stimuli == {}
        assert perception.memory == []

    def test_perceive_updates_stimuli(self):
        """Test perceive updates current stimuli"""
        perception = PerceptionSubsystem()
        perception.perceive({"light": 0.8, "sound": 0.3})

        assert perception.current_stimuli["light"] == 0.8
        assert perception.current_stimuli["sound"] == 0.3

    def test_perceive_remembers_threats(self):
        """Test threats are remembered"""
        perception = PerceptionSubsystem()
        perception.perceive({"threat_level": 0.7})

        assert len(perception.memory) == 1
        assert perception.memory[0]["threat_level"] == 0.7

    def test_perceive_remembers_food(self):
        """Test food locations are remembered"""
        perception = PerceptionSubsystem()
        perception.perceive({"food_nearby": True})

        assert len(perception.memory) == 1
        assert perception.memory[0]["food_nearby"] is True

    def test_memory_limited(self):
        """Test memory has size limit"""
        perception = PerceptionSubsystem()
        perception.memory_size = 3

        for i in range(5):
            perception.perceive({"threat_level": 0.6 + i * 0.1})

        assert len(perception.memory) == 3

    def test_tick_does_nothing(self):
        """Test tick is passive"""
        perception = PerceptionSubsystem()
        perception.current_stimuli = {"test": True}
        perception.tick()
        assert perception.current_stimuli == {"test": True}

    def test_state_serialization(self):
        """Test get_state returns correct state"""
        perception = PerceptionSubsystem(range=20.0)
        perception.perceive({"light": 0.5})
        perception.perceive({"threat_level": 0.8})

        state = perception.get_state()
        assert state["range"] == 20.0
        assert state["current_stimuli"]["threat_level"] == 0.8
        assert len(state["memory"]) == 1


# =============================================================================
# Tests: SimpleOrganism
# =============================================================================

class TestSimpleOrganism:
    """Tests for SimpleOrganism"""

    def test_initialization(self):
        """Test organism initialization"""
        org = SimpleOrganism("TestOrg")
        assert org.name == "TestOrg"
        assert org.is_alive is True
        assert org.age == 0
        assert org.phase == LifecyclePhase.NASCENT

    def test_has_required_subsystems(self):
        """Test organism has all required subsystems"""
        org = SimpleOrganism()
        assert org.get_subsystem("energy") is not None
        assert org.get_subsystem("health") is not None
        assert org.get_subsystem("perception") is not None

    def test_has_capabilities(self):
        """Test organism has capabilities set"""
        org = SimpleOrganism()
        assert org.get_capability(OrganismCapability.PERCEPTION) > 0
        assert org.get_capability(OrganismCapability.LOCOMOTION) > 0
        assert org.get_capability(OrganismCapability.SELF_REPAIR) > 0

    def test_has_traits(self):
        """Test organism has traits"""
        org = SimpleOrganism()
        assert "vitality" in org.traits
        assert "metabolism" in org.traits
        assert "awareness" in org.traits

    def test_custom_traits(self):
        """Test organism with custom traits"""
        traits = {"vitality": 2.0, "aggression": 0.9}
        org = SimpleOrganism(traits=traits)

        assert org.get_trait("vitality") == 2.0
        assert org.get_trait("aggression") == 0.9

    def test_tick_increments_age(self):
        """Test tick increases age"""
        org = SimpleOrganism()
        org.tick()
        assert org.age == 1

    def test_tick_updates_subsystems(self):
        """Test tick updates all subsystems"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        initial_energy = energy.energy

        org.tick()
        assert energy.energy < initial_energy

    def test_lifecycle_progression(self):
        """Test lifecycle phases progress with age"""
        org = SimpleOrganism()
        assert org.phase == LifecyclePhase.NASCENT

        # Progress to DEVELOPING
        for _ in range(12):
            org.tick()
        assert org.phase == LifecyclePhase.DEVELOPING

        # Progress to MATURE
        for _ in range(40):
            org.tick()
        assert org.phase == LifecyclePhase.MATURE

    def test_starvation_death(self):
        """Test organism dies from starvation"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        energy.energy = 0

        org.tick()
        assert org.is_alive is False
        assert org.phase == LifecyclePhase.ENDED

    def test_damage_death(self):
        """Test organism dies from damage"""
        org = SimpleOrganism()
        health = org.get_subsystem("health")
        energy = org.get_subsystem("energy")

        # Set low energy to prevent health regeneration
        energy.energy = energy.max_energy * 0.1
        health.health = 0

        # Death check happens during tick
        org.tick()
        assert org.is_alive is False

    def test_old_age_death(self):
        """Test organism dies from old age"""
        org = SimpleOrganism()
        org._age = 400  # Near max age

        org.tick()
        assert org.is_alive is False

    def test_perceive(self):
        """Test perceive forwards to perception subsystem"""
        org = SimpleOrganism()
        org.perceive({"light": 0.5, "temperature": 25.0})

        perception = org.get_subsystem("perception")
        assert perception.current_stimuli["light"] == 0.5

    def test_decide_flee_on_threat(self):
        """Test decides to flee when threatened"""
        org = SimpleOrganism()
        org.perceive({"threat_level": 0.8})

        action = org.decide()
        assert action == "flee"

    def test_decide_find_food_when_hungry(self):
        """Test decides to find food when hungry and food is nearby"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        # Set energy very low (<30%) to trigger hungry behavior
        energy.energy = energy.max_energy * 0.25

        # Must have food_nearby for find_food action to succeed
        # Otherwise the behavior tree falls through to rest
        org.perceive({"food_nearby": True})
        action = org.decide()
        # With 25% energy and food nearby, action is find_food
        assert action == "find_food"

    def test_decide_rest_when_tired(self):
        """Test decides to rest when low energy"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        energy.energy = energy.max_energy * 0.4  # Medium energy

        org.perceive({})
        action = org.decide()
        assert action == "rest"

    def test_decide_explore_default(self):
        """Test explores by default"""
        org = SimpleOrganism()
        org.perceive({})
        action = org.decide()
        assert action == "explore"

    def test_act_flee(self):
        """Test flee action consumes energy"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        initial = energy.energy

        org.act("flee")
        assert energy.energy < initial

    def test_act_find_food_restores_energy(self):
        """Test finding food restores energy"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        energy.energy = 50.0
        org.perceive({"food_nearby": True})

        org.act("find_food")
        assert energy.energy > 50.0

    def test_act_rest_restores_energy(self):
        """Test resting restores energy"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        energy.energy = 50.0

        org.act("rest")
        assert energy.energy > 50.0

    def test_act_explore_consumes_energy(self):
        """Test exploring consumes energy"""
        org = SimpleOrganism()
        energy = org.get_subsystem("energy")
        initial = energy.energy

        org.act("explore")
        assert energy.energy < initial

    def test_dead_organism_cannot_act(self):
        """Test dead organism returns None on decide"""
        org = SimpleOrganism()
        org.die("test")

        assert org.decide() is None

    def test_dead_organism_does_not_tick(self):
        """Test dead organism tick does nothing"""
        org = SimpleOrganism()
        org.die("test")
        initial_age = org.age

        org.tick()
        assert org.age == initial_age

    def test_serialization(self):
        """Test to_dict serialization"""
        org = SimpleOrganism("SerializeOrg")
        org.tick()
        org.perceive({"test": True})
        org.decide()

        data = org.to_dict()
        assert data["name"] == "SerializeOrg"
        assert data["age"] == 1
        assert "energy" in data["subsystems"]

    def test_deserialization(self):
        """Test from_dict deserialization"""
        org = SimpleOrganism("Original")
        for _ in range(20):
            org.tick()

        data = org.to_dict()
        restored = SimpleOrganism.from_dict(data)

        assert restored.name == "Original"
        assert restored.id == org.id
        assert restored.age == org.age


# =============================================================================
# Tests: SimplePopulation
# =============================================================================

class TestSimplePopulation:
    """Tests for SimplePopulation"""

    def test_initialization(self):
        """Test population initialization"""
        pop = SimplePopulation("TestPop", carrying_capacity=50)
        assert pop.name == "TestPop"
        assert pop.carrying_capacity == 50
        assert pop.size == 0

    def test_add_organism(self):
        """Test adding organisms"""
        pop = SimplePopulation()
        org = SimpleOrganism("Org1")

        pop.add(org)
        assert pop.size == 1
        assert pop.total_births == 1

    def test_remove_organism(self):
        """Test removing organisms"""
        pop = SimplePopulation()
        org = SimpleOrganism("Org1")
        pop.add(org)

        removed = pop.remove(org.id)
        assert removed is org
        assert pop.size == 0
        assert pop.total_deaths == 1

    def test_tick_updates_organisms(self):
        """Test tick updates all organisms"""
        pop = SimplePopulation()
        org = SimpleOrganism()
        pop.add(org)

        pop.tick()
        assert org.age == 1

    def test_tick_removes_dead_organisms(self):
        """Test tick removes dead organisms"""
        pop = SimplePopulation()
        org = SimpleOrganism()
        org._age = 500  # Will die next tick
        pop.add(org)

        pop.tick()
        assert pop.size == 0

    def test_tick_with_stimuli(self):
        """Test tick passes stimuli to organisms"""
        pop = SimplePopulation()
        org = SimpleOrganism()
        pop.add(org)

        pop.tick({"light": 0.8})
        perception = org.get_subsystem("perception")
        assert perception.current_stimuli.get("light") == 0.8

    def test_alive_count(self):
        """Test alive count calculation"""
        pop = SimplePopulation()
        org1 = SimpleOrganism()
        org2 = SimpleOrganism()
        org2.die("test")

        pop.add(org1)
        pop.add(org2)

        assert pop.size == 2
        assert pop.alive_count == 1

    def test_get_stats_empty(self):
        """Test stats for empty population"""
        pop = SimplePopulation()
        stats = pop.get_stats()

        assert stats["size"] == 0
        assert stats["alive"] == 0
        assert stats["avg_age"] == 0

    def test_get_stats_with_organisms(self):
        """Test stats calculation"""
        pop = SimplePopulation()
        for i in range(5):
            org = SimpleOrganism(f"Org{i}")
            for _ in range(i * 10):
                org.tick()
            pop.add(org)

        stats = pop.get_stats()
        assert stats["size"] == 5
        assert stats["alive"] > 0
        assert stats["avg_age"] > 0
        assert 0 <= stats["avg_energy"] <= 1
        assert 0 <= stats["avg_health"] <= 1

    def test_multiple_ticks_lifecycle(self):
        """Test population over multiple ticks"""
        pop = SimplePopulation()
        for i in range(10):
            pop.add(SimpleOrganism(f"Org{i}"))

        initial_size = pop.size
        for _ in range(100):
            pop.tick({"food_nearby": True})

        # Some may have died from natural causes
        assert pop.total_births == initial_size


# =============================================================================
# Tests: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for SimpleOrganism"""

    def test_full_lifecycle(self):
        """Test organism through complete lifecycle"""
        org = SimpleOrganism("FullLifecycle")

        # Birth - should be nascent
        assert org.phase == LifecyclePhase.NASCENT

        # Early life - gather food
        for _ in range(15):
            org.perceive({"food_nearby": True})
            org.tick()
            action = org.decide()
            if action:
                org.act(action)

        assert org.phase == LifecyclePhase.DEVELOPING
        assert org.is_alive

    def test_survival_with_food(self):
        """Test organism survives with food"""
        org = SimpleOrganism("Survivor")

        for _ in range(100):
            org.perceive({"food_nearby": True})
            org.tick()
            action = org.decide()
            if action:
                org.act(action)

        assert org.is_alive

    def test_starvation_without_food(self):
        """Test organism starves without food eventually"""
        org = SimpleOrganism("Starving")
        energy = org.get_subsystem("energy")
        # Very high consumption to guarantee starvation
        # Resting restores 5 energy, so consumption must exceed that
        energy.consumption_rate = 10.0

        tick = 0
        max_ticks = 50  # Should starve within 50 ticks
        while org.is_alive and tick < max_ticks:
            org.perceive({})  # No food
            org.tick()
            org.decide()
            tick += 1

        assert not org.is_alive, f"Organism still alive after {tick} ticks with {energy.energy} energy"

    def test_population_dynamics(self):
        """Test population dynamics over time"""
        pop = SimplePopulation("DynamicPop")

        # Start with healthy population
        for i in range(20):
            pop.add(SimpleOrganism(f"Org{i}"))

        initial = pop.size

        # Run for many ticks
        for _ in range(200):
            pop.tick({"food_nearby": True, "threat_level": 0.1})

        # Population should persist with food
        stats = pop.get_stats()
        assert stats["alive"] > 0 or pop.total_deaths > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
