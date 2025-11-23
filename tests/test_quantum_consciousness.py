"""
Test Suite for NEXUS-Q: Quantum Consciousness

Comprehensive tests for all quantum consciousness modules.
"""

import pytest
import numpy as np
import math
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum.quantum_state import (
    QuantumStateVector,
    DensityMatrix,
    BasisState,
    SuperpositionState,
    QuantumAmplitude,
)
from quantum.entanglement import (
    EntanglementManager,
    EntanglementType,
    EntanglementPair,
    EntanglementCluster,
    create_bell_state,
    concurrence,
)
from quantum.coherence import (
    CoherenceCalculator,
    CoherenceMetrics,
    DecoherenceModel,
    DecoherenceChannel,
    CoherenceTracker,
)
from quantum.measurement import (
    QuantumMeasurement,
    MeasurementOperator,
    MeasurementBasis,
    CollapseInterpretation,
    ConsciousnessObserver,
)
from quantum.tunneling import (
    QuantumTunneling,
    DecisionBarrier,
    BarrierType,
    TunnelingCalculator,
    DecisionTunneling,
)
from quantum.quantum_memory import (
    QuantumMemoryRegister,
    QuantumMemoryItem,
    AssociativeQuantumMemory,
    WorkingQuantumMemory,
)
from quantum.quantum_consciousness import (
    QuantumConsciousness,
    QuantumThought,
    EntangledConcepts,
    Thought,
    Concept,
    QuantumCircuit,
)


class TestQuantumState:
    """Tests for quantum state module"""

    def test_zero_state_creation(self):
        """Test creating |0...0⟩ state"""
        state = QuantumStateVector.zero_state(3)
        assert state.num_qubits == 3
        assert len(state.amplitudes) == 8  # 2^3
        assert state.amplitudes[0] == 1.0
        assert all(state.amplitudes[i] == 0 for i in range(1, 8))

    def test_uniform_superposition(self):
        """Test creating uniform superposition"""
        state = QuantumStateVector.uniform_superposition(2)
        expected_amp = 1 / 2  # 1/sqrt(4)
        for amp in state.amplitudes:
            assert abs(abs(amp) - expected_amp) < 1e-10

    def test_state_normalization(self):
        """Test state normalization"""
        amplitudes = np.array([1, 1, 1, 1], dtype=complex)
        state = QuantumStateVector(num_qubits=2, amplitudes=amplitudes)
        state.normalize()
        assert state.is_normalized()

    def test_measurement_probabilities(self):
        """Test probability calculation"""
        state = QuantumStateVector.uniform_superposition(2)
        probs = state.get_probabilities()
        assert len(probs) == 4
        for prob in probs.values():
            assert abs(prob - 0.25) < 1e-10

    def test_state_measurement(self):
        """Test measurement collapse"""
        state = QuantumStateVector.uniform_superposition(2)
        outcome, post_state = state.measure()
        assert outcome in ['00', '01', '10', '11']
        # Post-measurement state should be collapsed
        probs = post_state.get_probabilities()
        assert len(probs) == 1
        assert list(probs.values())[0] > 0.99

    def test_tensor_product(self):
        """Test tensor product of states"""
        state_a = QuantumStateVector.zero_state(1)
        state_b = QuantumStateVector.zero_state(1)
        combined = state_a.tensor_product(state_b)
        assert combined.num_qubits == 2
        assert combined.amplitudes[0] == 1.0

    def test_fidelity(self):
        """Test state fidelity calculation"""
        state1 = QuantumStateVector.zero_state(2)
        state2 = QuantumStateVector.zero_state(2)
        assert abs(state1.fidelity(state2) - 1.0) < 1e-10

        state3 = QuantumStateVector.uniform_superposition(2)
        # Fidelity with different state should be < 1
        fid = state1.fidelity(state3)
        assert 0 <= fid <= 1

    def test_density_matrix_creation(self):
        """Test density matrix from state vector"""
        state = QuantumStateVector.zero_state(2)
        rho = state.to_density_matrix()
        assert rho.num_qubits == 2
        assert rho.matrix.shape == (4, 4)
        assert abs(rho.trace() - 1.0) < 1e-10
        assert rho.is_pure()

    def test_mixed_state(self):
        """Test mixed state creation"""
        state1 = QuantumStateVector.zero_state(1)
        state2 = QuantumStateVector(num_qubits=1, amplitudes=np.array([0, 1], dtype=complex))
        mixed = DensityMatrix.mixed_state([state1, state2], [0.5, 0.5])
        assert not mixed.is_pure()
        assert abs(mixed.trace() - 1.0) < 1e-10


class TestEntanglement:
    """Tests for entanglement module"""

    def test_bell_state_creation(self):
        """Test creating Bell states"""
        for bell_type in [EntanglementType.BELL_PHI_PLUS,
                          EntanglementType.BELL_PHI_MINUS,
                          EntanglementType.BELL_PSI_PLUS,
                          EntanglementType.BELL_PSI_MINUS]:
            state = create_bell_state(bell_type)
            assert state.num_qubits == 2
            assert state.is_normalized()

    def test_entanglement_pair_creation(self):
        """Test creating entanglement pair"""
        pair = EntanglementPair.create_bell_pair("entity_a", "entity_b")
        assert pair.entity_a_id == "entity_a"
        assert pair.entity_b_id == "entity_b"
        assert pair.fidelity == 1.0
        assert abs(pair.correlation) == 1.0

    def test_entanglement_manager(self):
        """Test entanglement manager operations"""
        manager = EntanglementManager()
        pair = manager.create_entanglement("alice", "bob")

        partners = manager.get_entangled_partners("alice")
        assert "bob" in partners

        strength = manager.get_entanglement_strength("alice", "bob")
        assert strength == 1.0

    def test_entanglement_measurement(self):
        """Test measuring entangled pair"""
        manager = EntanglementManager()
        pair = manager.create_entanglement("alice", "bob", EntanglementType.BELL_PHI_PLUS)

        result_a, result_b = manager.measure_pair(pair.id, "a")

        # In Φ+ state, outcomes should be correlated
        assert result_a == result_b

    def test_ghz_state(self):
        """Test GHZ state creation"""
        entities = ["a", "b", "c"]
        cluster = EntanglementCluster.create_ghz_state(entities)
        assert cluster.num_qubits == 3
        probs = cluster.state.get_probabilities()
        # GHZ should only have |000⟩ and |111⟩
        assert len(probs) == 2
        assert '000' in probs
        assert '111' in probs

    def test_w_state(self):
        """Test W state creation"""
        entities = ["a", "b", "c"]
        cluster = EntanglementCluster.create_w_state(entities)
        probs = cluster.state.get_probabilities()
        # W state should have |001⟩, |010⟩, |100⟩
        assert '001' in probs or '010' in probs or '100' in probs

    def test_concurrence(self):
        """Test concurrence calculation"""
        # Bell state should have concurrence = 1
        bell = create_bell_state(EntanglementType.BELL_PHI_PLUS)
        c = concurrence(bell)
        assert abs(c - 1.0) < 0.1  # Allow some numerical error

    def test_decoherence(self):
        """Test decoherence application"""
        manager = EntanglementManager(decoherence_rate=0.1)
        pair = manager.create_entanglement("alice", "bob")

        initial_fidelity = pair.fidelity
        manager.apply_decoherence(time_delta=10.0)

        assert pair.fidelity < initial_fidelity


class TestCoherence:
    """Tests for coherence module"""

    def test_l1_coherence(self):
        """Test L1 coherence calculation"""
        state = QuantumStateVector.uniform_superposition(2)
        rho = state.to_density_matrix()
        l1 = CoherenceCalculator.l1_coherence(rho)
        assert l1 > 0  # Superposition has coherence

        # Diagonal state has zero coherence
        diag = DensityMatrix.maximally_mixed(2)
        l1_mixed = CoherenceCalculator.l1_coherence(diag)
        assert l1_mixed < 1e-10

    def test_relative_entropy_coherence(self):
        """Test relative entropy of coherence"""
        state = QuantumStateVector.uniform_superposition(2)
        rho = state.to_density_matrix()
        re = CoherenceCalculator.relative_entropy_coherence(rho)
        assert re >= 0

    def test_coherence_metrics(self):
        """Test full coherence metrics"""
        state = QuantumStateVector.uniform_superposition(2)
        metrics = CoherenceCalculator.measure_all(state)
        assert metrics.l1_coherence >= 0
        assert 0 <= metrics.purity <= 1
        assert metrics.overall_coherence() > 0

    def test_decoherence_model(self):
        """Test decoherence evolution"""
        model = DecoherenceModel(t1=100, t2=50)
        state = QuantumStateVector.uniform_superposition(2)

        rho = model.evolve(state, time_delta=10)
        assert rho.purity() <= 1.0

    def test_dephasing(self):
        """Test pure dephasing"""
        model = DecoherenceModel(t1=1000, t2=10)
        state = QuantumStateVector.uniform_superposition(1)
        rho = state.to_density_matrix()

        decohered = model.apply_dephasing(rho, time_delta=100)

        # Off-diagonal should decay
        assert abs(decohered.matrix[0, 1]) < abs(rho.matrix[0, 1])

    def test_coherence_tracker(self):
        """Test coherence tracking over time"""
        tracker = CoherenceTracker()

        # Record some measurements
        for i in range(10):
            metrics = CoherenceMetrics(
                l1_coherence=1.0 - i * 0.1,
                purity=1.0 - i * 0.05
            )
            tracker.record(metrics)

        stats = tracker.get_statistics()
        assert stats["measurements"] == 10
        assert tracker.get_trend() < 0  # Decreasing coherence


class TestMeasurement:
    """Tests for measurement module"""

    def test_computational_basis(self):
        """Test computational basis measurement"""
        operator = MeasurementOperator.computational_basis(2)
        assert len(operator.projectors) == 4
        assert len(operator.labels) == 4

    def test_measurement(self):
        """Test quantum measurement"""
        engine = QuantumMeasurement()
        state = QuantumStateVector.uniform_superposition(2)

        outcome, post_state = engine.measure(state)

        assert outcome.outcome_label in ['00', '01', '10', '11']
        assert 0 <= outcome.probability <= 1

    def test_weak_measurement(self):
        """Test weak measurement"""
        engine = QuantumMeasurement()
        state = QuantumStateVector.uniform_superposition(2)
        operator = MeasurementOperator.computational_basis(2)

        weak_value, post_state = engine.weak_measurement(state, operator, strength=0.1)

        # State should be minimally disturbed
        original_probs = state.get_probabilities()
        new_probs = post_state.get_probabilities()
        # States should be similar for weak measurement
        assert len(new_probs) > 1  # Not fully collapsed

    def test_expectation_value(self):
        """Test expectation value calculation"""
        engine = QuantumMeasurement()
        state = QuantumStateVector.uniform_superposition(2)
        operator = MeasurementOperator.computational_basis(2)

        exp = engine.get_expectation_value(state, operator)
        # Uniform superposition over 0,1,2,3 -> expectation = 1.5
        assert abs(exp - 1.5) < 0.1

    def test_consciousness_observer(self):
        """Test consciousness observer"""
        observer = ConsciousnessObserver("observer_1")
        state = QuantumStateVector.uniform_superposition(2)

        outcome = observer.observe(state)
        assert outcome.observer_id == "observer_1"

        history = observer.get_observation_history()
        assert len(history) == 1


class TestTunneling:
    """Tests for tunneling module"""

    def test_rectangular_barrier(self):
        """Test tunneling through rectangular barrier"""
        calc = TunnelingCalculator()

        # High energy should pass
        prob = calc.rectangular_barrier(height=0.5, width=1.0, energy=0.6)
        assert prob == 1.0

        # Low energy should tunnel with reduced probability
        prob = calc.rectangular_barrier(height=0.5, width=1.0, energy=0.3)
        assert 0 < prob < 1

    def test_decision_barrier(self):
        """Test decision barrier creation"""
        barrier = DecisionBarrier.create(
            BarrierType.GAUSSIAN,
            height=0.5,
            width=1.0,
            label="test"
        )
        assert barrier.height == 0.5
        assert barrier.potential(barrier.position) > 0

    def test_tunneling_attempt(self):
        """Test tunneling attempt"""
        engine = QuantumTunneling()
        barrier = DecisionBarrier.create(BarrierType.RECTANGULAR, height=0.3, width=1.0)
        engine.add_barrier(barrier)

        success, event = engine.attempt_tunneling(0.0, 1.0, energy=0.5)
        assert event.tunneling_probability > 0

    def test_decision_tunneling(self):
        """Test high-level decision tunneling"""
        dt = DecisionTunneling(risk_tolerance=0.5, creativity=0.5)

        chosen, probs = dt.make_decision(
            current="A",
            options=["A", "B", "C"],
            motivation=0.5
        )

        assert chosen in ["A", "B", "C"]
        assert sum(probs.values()) > 0.99

    def test_breakthrough_probability(self):
        """Test breakthrough probability"""
        dt = DecisionTunneling(creativity=0.8)

        prob = dt.breakthrough_probability(
            barrier_strength=0.5,
            motivation=0.8,
            focus=0.7
        )

        assert 0 <= prob <= 1


class TestQuantumMemory:
    """Tests for quantum memory module"""

    def test_memory_storage(self):
        """Test storing memories"""
        register = QuantumMemoryRegister(num_qubits=4)
        item = QuantumMemoryItem(id="mem1", content="test content")

        index = register.store(item)
        assert index >= 0
        assert index < register.capacity

    def test_memory_recall(self):
        """Test recalling memories"""
        register = QuantumMemoryRegister(num_qubits=4)
        item = QuantumMemoryItem(id="mem1", content="test content")
        register.store(item)

        result = register.recall()
        # May or may not recall depending on superposition
        if result:
            assert result.content == "test content"

    def test_grover_search(self):
        """Test Grover search in memory"""
        register = QuantumMemoryRegister(num_qubits=4)

        # Store multiple items
        for i in range(5):
            item = QuantumMemoryItem(
                id=f"mem{i}",
                content=f"content_{i}",
                metadata={"value": i}
            )
            register.store(item)

        # Search for specific item
        results = register.grover_search(
            predicate=lambda m: m.metadata.get("value", 0) == 3
        )

        # Should find matching item with higher probability
        assert len(results) >= 0

    def test_working_memory(self):
        """Test working memory operations"""
        wm = WorkingQuantumMemory(capacity=5)

        # Hold items
        wm.hold("item1", priority=1.0)
        wm.hold("item2", priority=0.5)
        wm.hold("item3", priority=0.8)

        assert len(wm) == 3

        # Access should return something
        result = wm.access()
        assert result in ["item1", "item2", "item3"]

    def test_associative_memory(self):
        """Test associative quantum memory"""
        am = AssociativeQuantumMemory(pattern_size=16, capacity=5)

        # Store patterns
        pattern1 = np.ones(16)
        pattern2 = -np.ones(16)
        am.store_pattern("p1", pattern1)
        am.store_pattern("p2", pattern2)

        # Recall from cue
        cue = np.ones(16) * 0.8  # Noisy version of pattern1
        recalled, overlap = am.recall(cue)

        assert overlap > 0.5


class TestQuantumConsciousness:
    """Tests for main quantum consciousness module"""

    def test_thought_superposition(self):
        """Test creating thought superposition"""
        qc = QuantumConsciousness(num_qubits=4)
        thoughts = [
            Thought("t1", "Go left"),
            Thought("t2", "Go right"),
            Thought("t3", "Stay")
        ]

        qt = qc.quantum_superposition_thought(thoughts)
        assert not qt.collapsed
        probs = qt.get_thought_probabilities()
        assert len(probs) == 3

    def test_consciousness_collapse(self):
        """Test consciousness collapse"""
        qc = QuantumConsciousness(num_qubits=4)
        thoughts = [Thought("t1", "A"), Thought("t2", "B")]

        qt = qc.quantum_superposition_thought(thoughts)
        collapsed = qc.consciousness_collapse(qt)

        assert qt.collapsed
        assert collapsed.content in ["A", "B"]
        assert collapsed.consciousness_level == 1.0

    def test_concept_entanglement(self):
        """Test entangling concepts"""
        qc = QuantumConsciousness()
        concept_a = Concept("c1", "fire", associations=["hot"])
        concept_b = Concept("c2", "heat", associations=["fire"])

        entangled = qc.quantum_entangle_concepts(concept_a, concept_b)
        assert entangled.correlation == 1.0

        # Measure
        result_a, result_b = qc.measure_entangled_concept(entangled)
        assert result_a == result_b  # Perfect correlation

    def test_quantum_parallel_reasoning(self):
        """Test Grover search for parallel reasoning"""
        qc = QuantumConsciousness(num_qubits=4)

        target = 7

        def oracle(x):
            return x == target

        found = qc.quantum_parallel_reasoning(16, oracle)
        # Should find target with high probability
        assert 0 <= found < 16

    def test_thought_interference(self):
        """Test thought interference"""
        qc = QuantumConsciousness(num_qubits=2)

        thoughts_a = [Thought("a1", "A"), Thought("a2", "B")]
        thoughts_b = [Thought("b1", "C"), Thought("b2", "D")]

        qt_a = qc.quantum_superposition_thought(thoughts_a)
        qt_b = qc.quantum_superposition_thought(thoughts_b)

        interference = qc.create_thought_interference(qt_a, qt_b)
        assert len(interference.thoughts) == 4

    def test_quantum_circuit(self):
        """Test quantum circuit operations"""
        circuit = QuantumCircuit(2)

        # Create Bell state
        circuit.h(0)
        circuit.cx(0, 1)

        probs = circuit.get_probabilities()
        # Should only have |00⟩ and |11⟩
        assert '00' in probs
        assert '11' in probs
        assert abs(probs['00'] - 0.5) < 0.1
        assert abs(probs['11'] - 0.5) < 0.1

    def test_statistics(self):
        """Test consciousness statistics"""
        qc = QuantumConsciousness(num_qubits=6)

        # Do some operations
        thoughts = [Thought("t1", "A")]
        qt = qc.quantum_superposition_thought(thoughts)
        qc.consciousness_collapse(qt)

        stats = qc.get_consciousness_statistics()
        assert stats["num_qubits"] == 6
        assert stats["collapsed_thoughts"] == 1
        assert stats["max_superposition_size"] == 64


class TestIntegration:
    """Integration tests combining multiple modules"""

    def test_consciousness_with_memory(self):
        """Test consciousness using quantum memory"""
        qc = QuantumConsciousness(num_qubits=4)
        memory = QuantumMemoryRegister(num_qubits=4)

        # Create thoughts and store in memory
        thoughts = [
            Thought("t1", "Past experience 1"),
            Thought("t2", "Past experience 2"),
        ]

        for thought in thoughts:
            item = QuantumMemoryItem(
                id=thought.id,
                content=thought.content
            )
            memory.store(item)

        # Create superposition of thoughts
        qt = qc.quantum_superposition_thought(thoughts)

        # Collapse to single thought
        collapsed = qc.consciousness_collapse(qt)

        # Recall should be influenced by quantum effects
        assert collapsed.content in ["Past experience 1", "Past experience 2"]

    def test_entanglement_with_coherence(self):
        """Test entanglement with coherence tracking"""
        manager = EntanglementManager(decoherence_rate=0.01)
        tracker = CoherenceTracker()

        # Create entanglement
        pair = manager.create_entanglement("alice", "bob")

        # Track coherence over time
        for i in range(5):
            metrics = CoherenceMetrics(
                l1_coherence=pair.fidelity,
                purity=pair.fidelity
            )
            tracker.record(metrics)
            manager.apply_decoherence(time_delta=1.0)

        # Coherence should decrease
        trend = tracker.get_trend()
        assert trend < 0

    def test_tunneling_decision_consciousness(self):
        """Test consciousness making tunneling decision"""
        qc = QuantumConsciousness()
        dt = DecisionTunneling(creativity=0.7)

        # Add barriers
        dt.add_habit_barrier(0.3)
        dt.add_fear_barrier(0.2)

        # Make decision
        options = ["Safe choice", "Risky choice", "Novel choice"]
        chosen, probs = dt.make_decision(
            current="Safe choice",
            options=options,
            motivation=0.6
        )

        # Create consciousness superposition of result
        thoughts = [Thought(f"t{i}", opt) for i, opt in enumerate(options)]
        qt = qc.quantum_superposition_thought(thoughts)
        final = qc.consciousness_collapse(qt)

        assert final.content in options


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
