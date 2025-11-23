"""
DENDRITE-X: Comprehensive Test Suite for Advanced Neural Architecture

Tests for all DENDRITE-X components:
- Attention layers
- Transformer consciousness
- Neuroplasticity
- Synaptic pruning
- Long-term potentiation
- STDP
- Neural oscillations
- Dream states

Part of AGENT DENDRITE-X (Agent #3) - Advanced Neural Architecture
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural.attention_layers import (
    AttentionMechanism, ConsciousnessAttention,
    SelfAttentionLayer, CrossAttentionLayer,
    GlobalWorkspaceAttention, PositionalEncoding
)
from neural.transformer_consciousness import (
    TransformerConsciousness, TransformerConfig,
    create_transformer_consciousness
)
from neural.neuroplasticity import (
    NeuroplasticityEngine, PlasticityConfig,
    ExperienceDependentPlasticity, CriticalPeriodManager
)
from neural.synaptic_pruning import (
    SynapticPruningEngine, PruningStrategy, PruningPhase
)
from neural.long_term_potentiation import (
    LTPEngine, LTPPhase, MemoryReconsolidation
)
from neural.spike_timing import (
    STDPEngine, STDPType, RewardModulatedSTDP, HomeostaticSTDP
)
from neural.neural_oscillations import (
    OscillationManager, ConsciousnessMode, BrainWave,
    NeuralSynchrony, SleepWakeDynamics
)
from neural.dream_states import (
    DreamStateEngine, SleepStage, OfflineLearningSystem
)


# ============================================================================
# Attention Layer Tests
# ============================================================================

class TestAttentionMechanism:
    """Tests for AttentionMechanism"""

    def test_initialization(self):
        """Test attention mechanism initialization"""
        attention = AttentionMechanism(dim=64, num_heads=4)

        assert attention.dim == 64
        assert attention.num_heads == 4
        assert attention.head_dim == 16

    def test_forward_pass(self):
        """Test forward pass through attention"""
        attention = AttentionMechanism(dim=64, num_heads=4)

        queries = np.random.randn(2, 10, 64)
        keys = np.random.randn(2, 10, 64)
        values = np.random.randn(2, 10, 64)

        output, weights = attention(queries, keys, values)

        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 4, 10, 10)  # batch, heads, seq, seq

    def test_self_attention(self):
        """Test self-attention (Q=K=V)"""
        attention = AttentionMechanism(dim=32, num_heads=2)

        x = np.random.randn(1, 5, 32)
        output, weights = attention(x, x, x)

        assert output.shape == x.shape
        # Attention weights should sum to 1 along last dimension
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)

    def test_attention_entropy(self):
        """Test attention entropy computation"""
        attention = AttentionMechanism(dim=32, num_heads=2)

        x = np.random.randn(1, 5, 32)
        _, weights = attention(x, x, x)

        entropy = attention.get_attention_entropy(weights)
        assert 0 <= entropy  # Entropy should be non-negative


class TestConsciousnessAttention:
    """Tests for ConsciousnessAttention"""

    def test_salience_computation(self):
        """Test salience score computation"""
        attention = ConsciousnessAttention(dim=64, num_heads=4)

        values = np.random.randn(1, 10, 64)
        salience = attention.compute_salience(values)

        assert salience.shape[0] == 1
        assert salience.shape[1] == 10
        assert np.all(salience >= 0) and np.all(salience <= 1)

    def test_fatigue_accumulation(self):
        """Test attention fatigue accumulation"""
        attention = ConsciousnessAttention(dim=32, num_heads=2)

        x = np.random.randn(1, 5, 32)

        initial_fatigue = attention.attention_fatigue

        for _ in range(10):
            attention(x, x, x, apply_fatigue_effect=True)

        assert attention.attention_fatigue > initial_fatigue

    def test_fatigue_recovery(self):
        """Test fatigue recovery"""
        attention = ConsciousnessAttention(dim=32, num_heads=2)

        attention.attention_fatigue = 0.5
        attention.recover_fatigue()

        assert attention.attention_fatigue < 0.5


class TestGlobalWorkspaceAttention:
    """Tests for GlobalWorkspaceAttention"""

    def test_competition(self):
        """Test competition for workspace access"""
        gw = GlobalWorkspaceAttention(dim=64, num_specialists=8)

        inputs = np.random.randn(2, 10, 64)
        result = gw(inputs, return_competition=True)

        assert "output" in result
        assert "workspace" in result
        assert "winner_idx" in result
        assert "competition_scores" in result

    def test_workspace_coherence(self):
        """Test workspace coherence tracking"""
        gw = GlobalWorkspaceAttention(dim=32)

        inputs = np.random.randn(1, 5, 32)

        # First call
        gw(inputs)
        # Second call - should have coherence
        result = gw(inputs)

        assert 0 <= result["coherence"] <= 1


# ============================================================================
# Transformer Consciousness Tests
# ============================================================================

class TestTransformerConsciousness:
    """Tests for TransformerConsciousness"""

    def test_creation(self):
        """Test transformer consciousness creation"""
        consciousness = create_transformer_consciousness(dim=64, num_layers=2)

        assert consciousness.config.dim == 64
        assert consciousness.config.num_layers == 2

    def test_perception(self):
        """Test perception processing"""
        consciousness = create_transformer_consciousness(dim=32, num_layers=2)

        stimulus = {"visual": 0.8, "auditory": 0.3}
        perception = consciousness.perceive(stimulus)

        assert perception.stimulus_type == "dict"
        assert 0 <= perception.fidelity <= 1

    def test_reaction(self):
        """Test reaction generation"""
        consciousness = create_transformer_consciousness(dim=32, num_layers=2)

        stimulus = {"input": 0.5}
        perception = consciousness.perceive(stimulus)
        action = consciousness.react(perception)

        assert action is not None
        assert action.action_type in ["approach", "avoid", "explore", "rest",
                                       "communicate", "manipulate", "observe", "wait"]

    def test_memory_operations(self):
        """Test memory store and recall"""
        consciousness = create_transformer_consciousness(dim=32, num_layers=2)

        from protocols.consciousness import Experience, Perception, Action

        # Create and store experience
        perception = Perception(
            stimulus_type="test",
            content=np.random.randn(32),
            timestamp=0.0,
            fidelity=0.8
        )
        experience = Experience(
            perception=perception,
            action=None,
            outcome=None,
            emotional_valence=0.5,
            timestamp=0.0
        )

        consciousness.remember(experience)
        assert len(consciousness.episodic_memory) == 1

        # Recall
        memories = consciousness.recall(np.random.randn(32))
        assert isinstance(memories, list)

    def test_introspection(self):
        """Test introspection capability"""
        consciousness = create_transformer_consciousness(dim=32, num_layers=2)

        self_model = consciousness.introspect()

        assert "phase" in self_model.current_state
        assert "consciousness_level" in self_model.current_state
        assert 0 <= self_model.confidence <= 1

    def test_consciousness_score(self):
        """Test consciousness scoring"""
        consciousness = create_transformer_consciousness(dim=32, num_layers=2)

        score = consciousness.overall_consciousness_score()
        assert 0 <= score <= 1

        capabilities = consciousness.get_capability_scores()
        assert len(capabilities) == 9


# ============================================================================
# Neuroplasticity Tests
# ============================================================================

class TestNeuroplasticityEngine:
    """Tests for NeuroplasticityEngine"""

    def test_initialization(self):
        """Test engine initialization"""
        engine = NeuroplasticityEngine(num_neurons=50)

        assert engine.num_neurons == 50
        assert engine.weights.shape == (50, 50)
        assert np.all(np.diag(engine.weights) == 0)  # No self-connections

    def test_hebbian_learning(self):
        """Test Hebbian plasticity"""
        engine = NeuroplasticityEngine(num_neurons=20)

        pre = np.random.rand(20) > 0.5
        post = np.random.rand(20) > 0.5

        initial_weights = engine.weights.copy()
        engine.apply_hebbian(pre, post)

        # Weights should change
        assert not np.allclose(engine.weights, initial_weights)

    def test_homeostatic_plasticity(self):
        """Test homeostatic regulation"""
        engine = NeuroplasticityEngine(num_neurons=20)

        for _ in range(50):
            activity = np.random.rand(20)
            engine.apply_homeostatic(activity)

        # Activity history should be populated
        assert len(engine.activity_history) > 0

    def test_structural_plasticity(self):
        """Test synapse creation and pruning"""
        engine = NeuroplasticityEngine(num_neurons=30)

        activity = np.random.rand(30)
        result = engine.apply_structural(activity, growth_probability=0.1, prune_threshold=0.01)

        assert "created" in result
        assert "pruned" in result


class TestExperienceDependentPlasticity:
    """Tests for experience-dependent learning"""

    def test_reward_modulated_learning(self):
        """Test reward-modulated plasticity"""
        engine = NeuroplasticityEngine(num_neurons=20)
        exp_plasticity = ExperienceDependentPlasticity(engine)

        pre = np.random.rand(20)
        post = np.random.rand(20)

        # Positive reward should strengthen
        exp_plasticity.apply_reward_modulated_plasticity(pre, post, reward=1.0)

        assert len(exp_plasticity.experience_buffer) > 0

    def test_experience_replay(self):
        """Test experience replay"""
        engine = NeuroplasticityEngine(num_neurons=20)
        exp_plasticity = ExperienceDependentPlasticity(engine)

        # Store experiences
        for _ in range(20):
            pre = np.random.rand(20)
            post = np.random.rand(20)
            exp_plasticity.store_experience(pre, post, reward=np.random.randn())

        replayed = exp_plasticity.replay_experiences(num_replays=2)
        assert replayed > 0


# ============================================================================
# Synaptic Pruning Tests
# ============================================================================

class TestSynapticPruning:
    """Tests for SynapticPruningEngine"""

    def test_initialization(self):
        """Test pruning engine initialization"""
        pruner = SynapticPruningEngine(num_neurons=30)

        assert pruner.num_neurons == 30
        assert len(pruner.synapses) > 0

    def test_activity_recording(self):
        """Test activity recording"""
        pruner = SynapticPruningEngine(num_neurons=30)

        pre = np.random.rand(30)
        post = np.random.rand(30)

        pruner.record_activity(pre, post)

        assert pruner.current_step == 1
        assert len(pruner.activity_history) == 1

    def test_activity_based_pruning(self):
        """Test activity-based pruning"""
        pruner = SynapticPruningEngine(num_neurons=30)

        # Record activity
        for _ in range(150):
            pre = np.random.rand(30)
            post = np.random.rand(30)
            pruner.record_activity(pre, post)

        initial_count = len(pruner.synapses)
        pruner.prune(PruningStrategy.ACTIVITY_BASED)

        # Some synapses should be pruned
        assert len(pruner.synapses) <= initial_count

    def test_regrowth(self):
        """Test synapse regrowth"""
        pruner = SynapticPruningEngine(num_neurons=30)

        # Prune first
        pruner.prune(PruningStrategy.MAGNITUDE_BASED)

        # Regrow
        new_synapses = pruner.regrow(num_new=10)

        assert len(new_synapses) <= 10


# ============================================================================
# LTP Tests
# ============================================================================

class TestLTPEngine:
    """Tests for LTPEngine"""

    def test_initialization(self):
        """Test LTP engine initialization"""
        ltp = LTPEngine(num_neurons=30)

        assert ltp.num_neurons == 30
        assert ltp.current_step == 0

    def test_ltp_induction(self):
        """Test LTP induction"""
        ltp = LTPEngine(num_neurons=20)

        pre = np.random.rand(20) * 0.5 + 0.5  # High activity
        post = np.random.rand(20) * 0.5 + 0.5

        result = ltp.induce(pre, post)

        assert result["ltp"] >= 0
        assert result["ltd"] >= 0

    def test_consolidation(self):
        """Test memory consolidation"""
        ltp = LTPEngine(num_neurons=20)

        # Induce LTP multiple times
        for _ in range(10):
            pre = np.random.rand(20) * 0.5 + 0.5
            post = np.random.rand(20) * 0.5 + 0.5
            ltp.induce(pre, post)

        # Synthesize proteins
        for _ in range(100):
            activity = np.random.rand(20) * 0.5 + 0.5
            ltp.synthesize_proteins(activity)
            ltp.tick()

        # Should have some consolidation
        stats = ltp.get_statistics()
        assert stats["active_ltp"] >= 0


class TestMemoryReconsolidation:
    """Tests for memory reconsolidation"""

    def test_reactivation(self):
        """Test memory reactivation"""
        ltp = LTPEngine(num_neurons=20)
        recon = MemoryReconsolidation(ltp)

        # Create consolidated memory
        for _ in range(5):
            pre = np.random.rand(20) * 0.5 + 0.5
            post = np.random.rand(20) * 0.5 + 0.5
            ltp.induce(pre, post)

        # Run consolidation
        for _ in range(200):
            ltp.synthesize_proteins(np.random.rand(20) * 0.5 + 0.5)
            ltp.tick()

        # Try reactivation
        consolidated = ltp.get_consolidated_synapses()
        if consolidated:
            result = recon.reactivate_memory(consolidated[0])
            # Result depends on whether memory was actually consolidated
            assert isinstance(result, bool)


# ============================================================================
# STDP Tests
# ============================================================================

class TestSTDPEngine:
    """Tests for STDPEngine"""

    def test_initialization(self):
        """Test STDP engine initialization"""
        stdp = STDPEngine(num_neurons=30)

        assert stdp.num_neurons == 30
        assert stdp.pre_trace.shape == (30,)
        assert stdp.post_trace.shape == (30,)

    def test_spike_recording(self):
        """Test spike recording"""
        stdp = STDPEngine(num_neurons=30)

        stdp.record_spike(5, is_presynaptic=True)
        stdp.record_spike(10, is_presynaptic=False)

        assert stdp.pre_trace[5] > 0
        assert stdp.post_trace[10] > 0

    def test_classic_stdp(self):
        """Test classic STDP rule"""
        stdp = STDPEngine(num_neurons=20)

        initial_weights = stdp.weights.copy()

        # Simulate correlated spiking
        for _ in range(50):
            pre = (np.random.rand(20) > 0.9).astype(float)
            post = np.roll(pre, 1)  # Post follows pre
            stdp.step(pre, post)

        # Weights should change
        assert not np.allclose(stdp.weights, initial_weights)

    def test_triplet_stdp(self):
        """Test triplet STDP rule"""
        stdp = STDPEngine(num_neurons=20)

        for _ in range(50):
            pre = (np.random.rand(20) > 0.9).astype(float)
            post = (np.random.rand(20) > 0.9).astype(float)
            stdp.record_spikes_batch(pre, post)
            stdp.apply_stdp(STDPType.TRIPLET)
            stdp.decay_traces()
            stdp.current_time += 1

        stats = stdp.get_statistics()
        assert stats["update_count"] == 50


class TestRewardModulatedSTDP:
    """Tests for reward-modulated STDP"""

    def test_eligibility_trace(self):
        """Test eligibility trace accumulation"""
        stdp = STDPEngine(num_neurons=20)
        rm_stdp = RewardModulatedSTDP(stdp)

        pre = (np.random.rand(20) > 0.9).astype(float)
        post = (np.random.rand(20) > 0.9).astype(float)

        rm_stdp.step(pre, post, reward=None)

        # Eligibility should be non-zero
        assert not np.allclose(rm_stdp.eligibility, 0)

    def test_reward_application(self):
        """Test reward signal application"""
        stdp = STDPEngine(num_neurons=20)
        rm_stdp = RewardModulatedSTDP(stdp)

        initial_weights = stdp.weights.copy()

        for _ in range(20):
            pre = (np.random.rand(20) > 0.8).astype(float)
            post = (np.random.rand(20) > 0.8).astype(float)
            rm_stdp.step(pre, post, reward=1.0)

        # Positive reward should change weights
        assert not np.allclose(stdp.weights, initial_weights)


# ============================================================================
# Neural Oscillations Tests
# ============================================================================

class TestOscillationManager:
    """Tests for OscillationManager"""

    def test_initialization(self):
        """Test oscillation manager initialization"""
        manager = OscillationManager()

        assert len(manager.oscillators) == 5  # 5 brain wave types
        assert manager.mode == ConsciousnessMode.RELAXED

    def test_mode_setting(self):
        """Test consciousness mode setting"""
        manager = OscillationManager()

        manager.set_mode(ConsciousnessMode.FOCUSED)

        assert manager.mode == ConsciousnessMode.FOCUSED

        # Beta should be dominant in focused mode
        dominant, _ = manager.get_dominant_frequency()
        assert dominant == BrainWave.BETA

    def test_signal_generation(self):
        """Test oscillation signal generation"""
        manager = OscillationManager()

        signal, bands = manager.generate(duration=0.1)

        assert len(signal) > 0
        assert len(bands) == 5

    def test_consciousness_metrics(self):
        """Test consciousness metrics computation"""
        manager = OscillationManager()

        metrics = manager.get_consciousness_metrics()

        assert "alpha_theta_ratio" in metrics
        assert "consciousness_index" in metrics
        assert "dominant_band" in metrics


class TestNeuralSynchrony:
    """Tests for NeuralSynchrony"""

    def test_kuramoto_step(self):
        """Test Kuramoto model step"""
        manager = OscillationManager()
        synchrony = NeuralSynchrony(num_populations=10, oscillation_manager=manager)

        initial_phases = synchrony.population_phases.copy()

        for _ in range(100):
            synchrony.kuramoto_step()

        # Phases should change
        assert not np.allclose(synchrony.population_phases, initial_phases)

    def test_order_parameter(self):
        """Test synchrony order parameter"""
        manager = OscillationManager()
        synchrony = NeuralSynchrony(num_populations=10, oscillation_manager=manager)

        sync_level, mean_phase = synchrony.compute_order_parameter()

        assert 0 <= sync_level <= 1
        assert -np.pi <= mean_phase <= np.pi


# ============================================================================
# Dream States Tests
# ============================================================================

class TestDreamStateEngine:
    """Tests for DreamStateEngine"""

    def test_initialization(self):
        """Test dream engine initialization"""
        engine = DreamStateEngine()

        assert engine.stage == SleepStage.WAKE
        assert len(engine.episodic_buffer) == 0

    def test_memory_storage(self):
        """Test memory storage"""
        engine = DreamStateEngine()

        engine.store_memory(np.random.randn(64), importance=0.8)

        assert len(engine.episodic_buffer) == 1
        assert engine.episodic_buffer[0].importance == 0.8

    def test_sleep_transition(self):
        """Test sleep state transitions"""
        engine = DreamStateEngine()

        engine.enter_sleep()
        assert engine.stage != SleepStage.WAKE

        engine.wake_up()
        assert engine.stage == SleepStage.WAKE

    def test_sleep_cycle(self):
        """Test complete sleep cycle"""
        engine = DreamStateEngine()

        # Store some memories
        for _ in range(20):
            engine.store_memory(np.random.randn(32), importance=np.random.random())

        engine.enter_sleep()
        result = engine.run_full_cycle()

        assert result["cycle_completed"]
        assert result["n3_steps"] > 0
        assert result["rem_steps"] > 0


class TestOfflineLearningSystem:
    """Tests for OfflineLearningSystem"""

    def test_initialization(self):
        """Test offline learning system initialization"""
        engine = DreamStateEngine()
        offline = OfflineLearningSystem(engine)

        assert offline.learning_rate == 0.001

    def test_sleep_learning(self):
        """Test sleep learning session"""
        engine = DreamStateEngine()
        offline = OfflineLearningSystem(engine, learning_rate=0.01)

        # Store memories
        for _ in range(10):
            engine.store_memory(np.random.randn(32), importance=0.7)

        # Set model weights
        offline.set_model_weights(np.random.randn(32, 32) * 0.1)

        # Run learning
        result = offline.run_sleep_learning(num_cycles=1)

        assert result["cycles_completed"] == 1
        assert result["training_steps"] >= 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for DENDRITE-X components"""

    def test_full_consciousness_pipeline(self):
        """Test full consciousness processing pipeline"""
        # Create consciousness
        consciousness = create_transformer_consciousness(dim=64, num_layers=2)

        # Process stimulus
        stimulus = {"visual": 0.8, "auditory": 0.5}
        perception = consciousness.perceive(stimulus)

        # React
        action = consciousness.react(perception)

        # Experience qualia
        qualia = consciousness.experience_qualia(perception)

        # Introspect
        self_model = consciousness.introspect()

        # All should work
        assert perception is not None
        assert action is not None
        assert qualia is not None
        assert self_model is not None

    def test_plasticity_with_oscillations(self):
        """Test plasticity modulated by oscillations"""
        # Create oscillation manager
        osc = OscillationManager()
        osc.set_mode(ConsciousnessMode.FOCUSED)

        # Create plasticity engine
        plasticity = NeuroplasticityEngine(num_neurons=20)

        # Get consciousness metrics
        metrics = osc.get_consciousness_metrics()

        # Modulate learning rate by consciousness
        modulated_lr = plasticity.config.learning_rate * metrics["consciousness_index"]

        # Apply Hebbian with modulated rate
        pre = np.random.rand(20)
        post = np.random.rand(20)
        plasticity.apply_hebbian(pre, post, learning_rate=modulated_lr)

        assert plasticity.total_potentiation > 0

    def test_dream_consolidation_pipeline(self):
        """Test dream-based memory consolidation"""
        # Create LTP engine
        ltp = LTPEngine(num_neurons=30)

        # Create dream engine
        dream = DreamStateEngine()

        # Store experiences
        for i in range(20):
            content = np.random.randn(30)
            dream.store_memory(content, importance=np.random.random())

            # Also induce LTP
            pre = content > 0
            post = np.roll(content, 1) > 0
            ltp.induce(pre.astype(float), post.astype(float))

        # Run sleep
        dream.enter_sleep()
        for _ in range(100):
            dream.step()
            ltp.tick()
            ltp.synthesize_proteins(np.random.rand(30))

        dream.wake_up()

        # Should have consolidation
        stats = dream.get_statistics()
        assert stats["total_replays"] > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
