"""
Tests for SYNAPSE-M Multi-Modal Perception Engine

Tests the core components of the multi-modal perception system including:
- Modality types and registry
- Cross-modal binding
- Attention filtering
- Predictive perception
- Sensory memory
- Gestalt processing
- Proprioception
- Interoception
"""

import pytest
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sensory.modality_types import (
    ExtendedModality, ModalityDomain, ModalityCharacteristics,
    ModalityRegistry, ModalityInput, ProcessedModality,
    get_modality_characteristics, ModalityGroup, get_group_modalities
)
from sensory.cross_modal import (
    CrossModalIntegrator, SensoryFusion, BoundPercept,
    BindingType, TemporalBindingMechanism, SemanticBindingMechanism
)
from sensory.attention_filter import (
    AttentionFilter, MultiModalAttentionManager, SaliencyComputer,
    AttentionType, AttentionPriority
)
from sensory.predictive_perception import (
    PredictivePerceptionSystem, Prediction, PredictionError,
    TemporalPredictiveModel, PredictionType
)
from sensory.sensory_memory import (
    SensoryMemorySystem, IconicMemory, EchoicMemory,
    MemoryTrace, MemoryType
)
from sensory.gestalt_processing import (
    GestaltProcessor, GestaltPrinciple, PerceptualGroup,
    ProximityProcessor, SimilarityProcessor, ClosureProcessor
)
from sensory.proprioception import (
    ProprioceptionSystem, BodyPartType, MovementType,
    PostureType, BodySchema
)
from sensory.interoception import (
    InteroceptionSystem, OrganSystem, HomeostasisState,
    ArousalLevel, InteroceptivePerception
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def modality_registry():
    """Create a modality registry for testing"""
    registry = ModalityRegistry()
    registry.activate(ExtendedModality.VISION)
    registry.activate(ExtendedModality.AUDITION)
    return registry


@pytest.fixture
def sample_perceptions():
    """Create sample processed perceptions for testing"""
    np.random.seed(42)

    perceptions = {}
    for modality in [ExtendedModality.VISION, ExtendedModality.AUDITION,
                     ExtendedModality.TACTILE]:
        chars = get_modality_characteristics(modality)
        input_data = ModalityInput(
            modality=modality,
            raw_data=np.random.randn(chars.feature_dim),
            intensity=np.random.uniform(0.5, 1.0)
        )
        perceptions[modality] = ProcessedModality(
            modality=modality,
            raw_input=input_data,
            features=np.random.randn(chars.feature_dim),
            semantic_embedding=np.random.randn(chars.semantic_dim),
            salience=np.random.uniform(0.3, 0.9),
            confidence=np.random.uniform(0.7, 1.0),
            timestamp=time.time()
        )

    return perceptions


# ============================================================================
# Modality Types Tests
# ============================================================================

class TestModalityTypes:
    """Tests for modality type definitions"""

    def test_extended_modality_enum(self):
        """Test that all modalities are defined"""
        assert len(ExtendedModality) >= 12
        assert ExtendedModality.VISION in ExtendedModality
        assert ExtendedModality.AUDITION in ExtendedModality
        assert ExtendedModality.INTEROCEPTION in ExtendedModality

    def test_modality_characteristics(self):
        """Test modality characteristics retrieval"""
        vision_chars = get_modality_characteristics(ExtendedModality.VISION)
        assert vision_chars.feature_dim == 256
        assert vision_chars.domain == ModalityDomain.EXTEROCEPTIVE

        audio_chars = get_modality_characteristics(ExtendedModality.AUDITION)
        assert audio_chars.feature_dim == 128
        assert audio_chars.latency_ms < vision_chars.latency_ms

    def test_modality_registry(self, modality_registry):
        """Test modality registry operations"""
        assert modality_registry.is_active(ExtendedModality.VISION)
        assert modality_registry.is_active(ExtendedModality.AUDITION)
        assert not modality_registry.is_active(ExtendedModality.OLFACTION)

        modality_registry.activate(ExtendedModality.OLFACTION)
        assert modality_registry.is_active(ExtendedModality.OLFACTION)

        modality_registry.deactivate(ExtendedModality.VISION)
        assert not modality_registry.is_active(ExtendedModality.VISION)

    def test_modality_groups(self):
        """Test modality grouping"""
        distance = get_group_modalities(ModalityGroup.DISTANCE_SENSES)
        assert ExtendedModality.VISION in distance
        assert ExtendedModality.AUDITION in distance

        body = get_group_modalities(ModalityGroup.BODY_SENSES)
        assert ExtendedModality.PROPRIOCEPTION in body

    def test_modality_input(self):
        """Test modality input creation"""
        input_data = ModalityInput(
            modality=ExtendedModality.VISION,
            raw_data=np.random.randn(256),
            intensity=0.8
        )
        assert input_data.get_age_ms() >= 0


# ============================================================================
# Cross-Modal Binding Tests
# ============================================================================

class TestCrossModal:
    """Tests for cross-modal integration"""

    def test_cross_modal_integrator_creation(self):
        """Test integrator creation"""
        integrator = CrossModalIntegrator(integration_dim=256)
        assert integrator.integration_dim == 256

    def test_cross_modal_binding(self, sample_perceptions):
        """Test cross-modal binding"""
        integrator = CrossModalIntegrator()
        bound = integrator.integrate(sample_perceptions)

        assert isinstance(bound, BoundPercept)
        assert len(bound.modalities) == len(sample_perceptions)
        assert bound.unified_representation.shape[0] == integrator.integration_dim
        assert 0 <= bound.overall_binding_strength <= 1

    def test_temporal_binding(self, sample_perceptions):
        """Test temporal binding mechanism"""
        binder = TemporalBindingMechanism(synchrony_window_ms=100.0)
        coherence, bindings = binder.compute_binding(sample_perceptions)

        assert 0 <= coherence <= 1
        assert isinstance(bindings, list)

    def test_semantic_binding(self, sample_perceptions):
        """Test semantic binding mechanism"""
        binder = SemanticBindingMechanism()
        coherence, bindings = binder.compute_binding(sample_perceptions)

        assert 0 <= coherence <= 1

    def test_sensory_fusion(self, sample_perceptions):
        """Test sensory fusion API"""
        fusion = SensoryFusion()

        for perception in sample_perceptions.values():
            fusion.add_perception(perception)

        result = fusion.fuse()
        assert result is not None
        assert isinstance(result, BoundPercept)

    def test_empty_integration(self):
        """Test integration with no perceptions"""
        integrator = CrossModalIntegrator()
        bound = integrator.integrate({})

        assert bound.overall_binding_strength == 0.0
        assert len(bound.modalities) == 0


# ============================================================================
# Attention Filter Tests
# ============================================================================

class TestAttentionFilter:
    """Tests for attention filtering"""

    def test_attention_filter_creation(self):
        """Test filter creation"""
        filter = AttentionFilter(attention_capacity=1.0)
        assert filter.attention_capacity == 1.0

    def test_attention_filtering(self, sample_perceptions):
        """Test attention-based filtering"""
        filter = AttentionFilter(attention_capacity=1.0)
        filtered = filter.filter_perceptions(sample_perceptions)

        assert len(filtered) <= len(sample_perceptions)
        assert filter.attention_spent <= filter.attention_capacity

    def test_goal_directed_attention(self, sample_perceptions):
        """Test top-down attention goals"""
        manager = MultiModalAttentionManager()
        manager.set_modality_priority(ExtendedModality.VISION, 1.0)

        filtered = manager.allocate_attention(sample_perceptions)
        assert ExtendedModality.VISION in filtered

    def test_saliency_computation(self, sample_perceptions):
        """Test saliency computation"""
        computer = SaliencyComputer()
        saliency_map = computer.compute_saliency_map(sample_perceptions)

        assert saliency_map.peak_saliency >= 0
        for modality, saliency in saliency_map.modality_saliency.items():
            assert 0 <= saliency <= 1

    def test_attention_state(self, sample_perceptions):
        """Test attention state tracking"""
        filter = AttentionFilter()
        filter.filter_perceptions(sample_perceptions)

        state = filter.get_attention_state()
        assert "capacity" in state
        assert "spent" in state
        assert "focus_modality" in state


# ============================================================================
# Predictive Perception Tests
# ============================================================================

class TestPredictivePerception:
    """Tests for predictive perception"""

    def test_predictive_system_creation(self):
        """Test system creation"""
        system = PredictivePerceptionSystem()
        system.register_modality(ExtendedModality.VISION)

        assert ExtendedModality.VISION in system.models

    def test_prediction_generation(self, sample_perceptions):
        """Test prediction generation"""
        system = PredictivePerceptionSystem()
        system.register_modality(ExtendedModality.VISION)

        # Add history
        vision = sample_perceptions[ExtendedModality.VISION]
        system.history[ExtendedModality.VISION].append(vision)

        predictions = system.generate_predictions(ExtendedModality.VISION)
        assert len(predictions) >= 1
        assert isinstance(predictions[0], Prediction)

    def test_perception_processing(self, sample_perceptions):
        """Test perception with prediction processing"""
        system = PredictivePerceptionSystem()
        system.register_modality(ExtendedModality.VISION)

        vision = sample_perceptions[ExtendedModality.VISION]

        # Process multiple perceptions
        for _ in range(5):
            system.generate_predictions(ExtendedModality.VISION)
            enhanced, errors = system.process_perception(vision)

        stats = system.get_statistics()
        assert stats["total_predictions"] > 0

    def test_cross_modal_prediction(self, sample_perceptions):
        """Test cross-modal prediction links"""
        system = PredictivePerceptionSystem()
        system.register_modality(ExtendedModality.VISION)
        system.register_modality(ExtendedModality.AUDITION)
        system.register_cross_modal_link(
            ExtendedModality.VISION,
            ExtendedModality.AUDITION
        )

        assert (ExtendedModality.VISION, ExtendedModality.AUDITION) in system.cross_modal_models


# ============================================================================
# Sensory Memory Tests
# ============================================================================

class TestSensoryMemory:
    """Tests for sensory memory"""

    def test_iconic_memory_creation(self):
        """Test iconic memory creation"""
        iconic = IconicMemory()
        assert iconic.modality == ExtendedModality.VISION
        assert iconic.persistence_ms < 500

    def test_echoic_memory_creation(self):
        """Test echoic memory creation"""
        echoic = EchoicMemory()
        assert echoic.modality == ExtendedModality.AUDITION
        assert echoic.persistence_ms > 1000

    def test_memory_storage(self, sample_perceptions):
        """Test memory storage and retrieval"""
        system = SensoryMemorySystem()

        vision = sample_perceptions[ExtendedModality.VISION]
        trace = system.store(vision)

        assert isinstance(trace, MemoryTrace)
        assert trace.current_strength == 1.0

        retrieved = system.retrieve(ExtendedModality.VISION)
        assert retrieved is not None

    def test_memory_decay(self, sample_perceptions):
        """Test memory decay"""
        system = SensoryMemorySystem()
        vision = sample_perceptions[ExtendedModality.VISION]
        trace = system.store(vision)

        initial_strength = trace.current_strength
        trace.decay(0.1)

        assert trace.current_strength < initial_strength

    def test_multi_modal_snapshot(self, sample_perceptions):
        """Test multi-modal snapshot"""
        system = SensoryMemorySystem()
        snapshot = system.store_multimodal(sample_perceptions)

        assert len(snapshot.traces) == len(sample_perceptions)
        assert snapshot.total_strength > 0

    def test_echoic_replay(self, sample_perceptions):
        """Test echoic memory replay"""
        echoic = EchoicMemory()

        audio = sample_perceptions.get(ExtendedModality.AUDITION)
        if audio:
            for _ in range(5):
                echoic.store(audio)

            replay = echoic.replay()
            assert len(replay) > 0


# ============================================================================
# Gestalt Processing Tests
# ============================================================================

class TestGestaltProcessing:
    """Tests for Gestalt processing"""

    def test_gestalt_processor_creation(self):
        """Test processor creation"""
        processor = GestaltProcessor()
        assert processor.proximity is not None
        assert processor.similarity is not None

    def test_proximity_grouping(self, sample_perceptions):
        """Test proximity-based grouping"""
        processor = ProximityProcessor()
        perceptions_list = list(sample_perceptions.values())

        groups = processor.apply(perceptions_list)
        assert len(groups) > 0
        assert all(isinstance(g, PerceptualGroup) for g in groups)

    def test_similarity_grouping(self, sample_perceptions):
        """Test similarity-based grouping"""
        processor = SimilarityProcessor()
        perceptions_list = list(sample_perceptions.values())

        groups = processor.apply(perceptions_list)
        assert len(groups) > 0

    def test_pattern_completion(self, sample_perceptions):
        """Test closure/pattern completion"""
        processor = ClosureProcessor()
        vision = sample_perceptions[ExtendedModality.VISION]

        completed = processor.complete_pattern(vision)
        assert completed.completion_confidence >= 0
        assert completed.completed_features.shape == vision.features.shape

    def test_figure_ground(self, sample_perceptions):
        """Test figure-ground segmentation"""
        processor = GestaltProcessor()
        perceptions_list = list(sample_perceptions.values())

        fg = processor.get_figure_ground(perceptions_list)
        assert len(fg.figures) > 0 or fg.ground is not None

    def test_full_gestalt_processing(self, sample_perceptions):
        """Test full Gestalt processing pipeline"""
        processor = GestaltProcessor()
        perceptions_list = list(sample_perceptions.values())

        results = processor.process(perceptions_list)
        assert "proximity_groups" in results
        assert "similarity_groups" in results
        assert "best_grouping" in results


# ============================================================================
# Proprioception Tests
# ============================================================================

class TestProprioception:
    """Tests for proprioception"""

    def test_proprioception_creation(self):
        """Test system creation"""
        proprio = ProprioceptionSystem()
        assert len(proprio.body_schema.parts) == len(BodyPartType)

    def test_body_part_update(self):
        """Test body part state update"""
        proprio = ProprioceptionSystem()

        body_state = {
            BodyPartType.LEFT_ARM: {
                "position": [0.5, 0.3, 0.1],
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "load": 0.2
            }
        }

        perception = proprio.update(body_state)
        assert perception.movement_type is not None
        assert perception.features.shape[0] == proprio.feature_dim

    def test_movement_detection(self):
        """Test movement type detection"""
        proprio = ProprioceptionSystem()

        # Stationary
        perception = proprio.update({})
        assert perception.movement_type == MovementType.STATIONARY

        # Moving
        for i in range(5):
            proprio.update({
                BodyPartType.TORSO: {
                    "position": [0.0, 0.0, i * 0.1],
                    "load": 0.0
                }
            })

        perception = proprio.update({
            BodyPartType.TORSO: {
                "position": [0.0, 0.0, 0.5],
                "load": 0.0
            }
        })

    def test_posture_classification(self):
        """Test posture classification"""
        proprio = ProprioceptionSystem()
        perception = proprio.update({})

        assert proprio.body_schema.overall_posture in PostureType

    def test_to_processed_modality(self):
        """Test conversion to ProcessedModality"""
        proprio = ProprioceptionSystem()
        perception = proprio.update({})

        processed = proprio.to_processed_modality(perception)
        assert processed.modality == ExtendedModality.PROPRIOCEPTION


# ============================================================================
# Interoception Tests
# ============================================================================

class TestInteroception:
    """Tests for interoception"""

    def test_interoception_creation(self):
        """Test system creation"""
        intero = InteroceptionSystem()
        assert len(intero.organ_states) == len(OrganSystem)

    def test_internal_state_update(self):
        """Test internal state updates"""
        intero = InteroceptionSystem()

        perception = intero.update({
            "energy": 0.8,
            "hunger": 0.3,
            "temperature": 37.0
        })

        assert isinstance(perception, InteroceptivePerception)
        assert intero.current_state.energy_level == 0.8
        assert intero.current_state.hunger == 0.3

    def test_stress_response(self):
        """Test stress and arousal response"""
        intero = InteroceptionSystem()

        intero.update({"stress": 0.7})
        assert intero.current_state.arousal in [ArousalLevel.STRESSED, ArousalLevel.AROUSED]

    def test_pain_sensing(self):
        """Test pain/damage sensing"""
        intero = InteroceptionSystem()

        perception = intero.update({"damage": 0.5})
        assert intero.current_state.pain > 0
        assert "pain_signal" in perception.signals

    def test_wellbeing_calculation(self):
        """Test wellbeing calculation"""
        intero = InteroceptionSystem()

        # Good state
        intero.update({"energy": 0.9, "hunger": 0.1, "thirst": 0.1})
        high_wellbeing = intero.get_wellbeing()

        # Bad state
        intero.update({"energy": 0.2, "hunger": 0.8, "stress": 0.8})
        low_wellbeing = intero.get_wellbeing()

        assert high_wellbeing > low_wellbeing

    def test_homeostasis_states(self):
        """Test homeostasis state transitions"""
        intero = InteroceptionSystem()

        intero.update({"temperature": 37.0})
        temp_state = intero.organ_states[OrganSystem.THERMOREGULATORY]
        assert temp_state.homeostasis in HomeostasisState

    def test_to_processed_modality(self):
        """Test conversion to ProcessedModality"""
        intero = InteroceptionSystem()
        perception = intero.update({})

        processed = intero.to_processed_modality(perception)
        assert processed.modality == ExtendedModality.INTEROCEPTION


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full SYNAPSE-M system"""

    def test_full_perception_pipeline(self, sample_perceptions):
        """Test complete perception pipeline"""
        # 1. Attention filtering
        attention = MultiModalAttentionManager()
        filtered = attention.allocate_attention(sample_perceptions)

        # 2. Cross-modal binding
        integrator = CrossModalIntegrator()
        bound = integrator.integrate(filtered)

        # 3. Gestalt processing
        gestalt = GestaltProcessor()
        gestalt_results = gestalt.process(list(filtered.values()))

        assert bound.overall_binding_strength >= 0
        assert len(gestalt_results["best_grouping"]) > 0

    def test_memory_and_prediction(self, sample_perceptions):
        """Test memory-prediction interaction"""
        memory = SensoryMemorySystem()
        prediction = PredictivePerceptionSystem()

        for modality, perception in sample_perceptions.items():
            # Store in memory
            memory.store(perception)

            # Process with prediction
            prediction.register_modality(modality)
            prediction.generate_predictions(modality)
            prediction.process_perception(perception)

        memory_stats = memory.get_statistics()
        prediction_stats = prediction.get_statistics()

        assert memory_stats["total_traces"] > 0
        assert prediction_stats["total_predictions"] > 0

    def test_body_awareness_integration(self):
        """Test proprioception-interoception integration"""
        proprio = ProprioceptionSystem()
        intero = InteroceptionSystem()

        # Update body state
        proprio_perception = proprio.update({
            BodyPartType.TORSO: {
                "position": [0.0, 0.0, 0.0],
                "load": 0.5
            }
        })

        # Update internal state based on body activity
        intero_perception = intero.update({
            "energy": 0.8 - proprio_perception.body_schema.total_energy * 0.1,
            "fatigue": 0.2
        })

        # Both should produce valid perceptions
        proprio_processed = proprio.to_processed_modality(proprio_perception)
        intero_processed = intero.to_processed_modality(intero_perception)

        assert proprio_processed.modality == ExtendedModality.PROPRIOCEPTION
        assert intero_processed.modality == ExtendedModality.INTEROCEPTION


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
