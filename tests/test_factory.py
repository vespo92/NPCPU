"""
Tests for Consciousness Factory

Tests YAML loading, model creation, comparison, blending,
and caching functionality.
"""

import pytest
import os
import tempfile
import shutil
from factory.consciousness_factory import (
    ConsciousnessFactory,
    load_consciousness_model,
    ConsciousnessModelConfig
)
from protocols.consciousness import GradedConsciousness


# ============================================================================
# Basic Factory Tests
# ============================================================================

@pytest.mark.unit
def test_factory_creation():
    """Test creating factory"""
    factory = ConsciousnessFactory()

    assert factory is not None
    assert factory.config_dir is not None


@pytest.mark.unit
def test_factory_with_custom_dir(temp_config_dir):
    """Test factory with custom config directory"""
    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    assert factory.config_dir == temp_config_dir


# ============================================================================
# YAML Loading Tests
# ============================================================================

@pytest.mark.unit
def test_from_yaml_basic(consciousness_factory):
    """Test loading consciousness from YAML"""
    consciousness = consciousness_factory.from_yaml("test_model.yaml")

    assert isinstance(consciousness, GradedConsciousness)
    assert hasattr(consciousness, 'model_name')
    assert consciousness.model_name == "Test Consciousness"


@pytest.mark.unit
def test_from_yaml_sets_capabilities(consciousness_factory):
    """Test YAML loading sets capability scores"""
    consciousness = consciousness_factory.from_yaml("test_model.yaml")

    scores = consciousness.get_capability_scores()

    # From test YAML: perception_fidelity: 0.8
    assert scores["perception_fidelity"] == 0.8
    assert scores["reaction_speed"] == 0.7
    assert scores["memory_depth"] == 0.6


@pytest.mark.unit
def test_from_yaml_with_overrides(consciousness_factory):
    """Test loading with capability overrides"""
    consciousness = consciousness_factory.from_yaml(
        "test_model.yaml",
        overrides={"perception_fidelity": 0.95}
    )

    scores = consciousness.get_capability_scores()

    # Override should be applied
    assert scores["perception_fidelity"] == 0.95

    # Other values should be unchanged
    assert scores["reaction_speed"] == 0.7


@pytest.mark.unit
def test_from_yaml_caching(consciousness_factory):
    """Test that models are cached"""
    # Load twice
    c1 = consciousness_factory.from_yaml("test_model.yaml")
    c2 = consciousness_factory.from_yaml("test_model.yaml")

    # Should be same cached instance
    assert c1 is c2


@pytest.mark.unit
def test_from_yaml_cache_different_overrides(consciousness_factory):
    """Test cache handles different overrides"""
    c1 = consciousness_factory.from_yaml(
        "test_model.yaml",
        overrides={"perception_fidelity": 0.9}
    )

    c2 = consciousness_factory.from_yaml(
        "test_model.yaml",
        overrides={"perception_fidelity": 0.5}
    )

    # Should be different instances (different cache keys)
    assert c1.perception_fidelity != c2.perception_fidelity


@pytest.mark.unit
def test_from_yaml_nonexistent_file(consciousness_factory):
    """Test loading nonexistent file raises error"""
    with pytest.raises(FileNotFoundError):
        consciousness_factory.from_yaml("nonexistent.yaml")


# ============================================================================
# Real Model Loading Tests (Integration)
# ============================================================================

@pytest.mark.integration
def test_load_real_default_model(real_consciousness_factory):
    """Test loading real default.yaml model"""
    try:
        consciousness = real_consciousness_factory.from_yaml("default.yaml")

        assert isinstance(consciousness, GradedConsciousness)
        assert hasattr(consciousness, 'model_name')

        # Should have all standard capabilities
        scores = consciousness.get_capability_scores()
        assert "perception_fidelity" in scores
        assert "meta_cognitive_ability" in scores

    except FileNotFoundError:
        pytest.skip("default.yaml not found in configs")


@pytest.mark.integration
def test_load_real_plant_model(real_consciousness_factory):
    """Test loading real plant_consciousness.yaml model"""
    try:
        consciousness = real_consciousness_factory.from_yaml("plant_consciousness.yaml")

        assert isinstance(consciousness, GradedConsciousness)

        # Plant should have high perception, slow reaction
        scores = consciousness.get_capability_scores()
        assert scores.get("perception_fidelity", 0) > 0.7
        assert scores.get("reaction_speed", 1) < 0.3

    except FileNotFoundError:
        pytest.skip("plant_consciousness.yaml not found in configs")


@pytest.mark.integration
def test_load_real_explorer_model(real_consciousness_factory):
    """Test loading real explorer.yaml model"""
    try:
        consciousness = real_consciousness_factory.from_yaml("explorer.yaml")

        assert isinstance(consciousness, GradedConsciousness)

        # Explorer should have high perception, fast reaction
        scores = consciousness.get_capability_scores()
        assert scores.get("perception_fidelity", 0) > 0.7
        assert scores.get("reaction_speed", 0) > 0.7

    except FileNotFoundError:
        pytest.skip("explorer.yaml not found in configs")


# ============================================================================
# Programmatic Creation Tests
# ============================================================================

@pytest.mark.unit
def test_create_consciousness(consciousness_factory):
    """Test creating consciousness programmatically"""
    consciousness = consciousness_factory.create(
        name="Test Agent",
        perception_fidelity=0.8,
        reaction_speed=0.7
    )

    assert isinstance(consciousness, GradedConsciousness)
    assert consciousness.model_name == "Test Agent"
    assert consciousness.perception_fidelity == 0.8
    assert consciousness.reaction_speed == 0.7


@pytest.mark.unit
def test_create_with_all_capabilities(consciousness_factory):
    """Test creating with all capabilities specified"""
    consciousness = consciousness_factory.create(
        name="Full Spec",
        perception_fidelity=0.9,
        reaction_speed=0.8,
        memory_depth=0.85,
        memory_recall_accuracy=0.8,
        introspection_capacity=0.7,
        meta_cognitive_ability=0.6,
        information_integration=0.75,
        intentional_coherence=0.8,
        qualia_richness=0.7
    )

    scores = consciousness.get_capability_scores()

    assert scores["perception_fidelity"] == 0.9
    assert scores["reaction_speed"] == 0.8
    assert scores["memory_depth"] == 0.85


@pytest.mark.unit
def test_create_minimal(consciousness_factory):
    """Test creating with minimal parameters"""
    consciousness = consciousness_factory.create()

    assert isinstance(consciousness, GradedConsciousness)
    assert consciousness.model_name == "Custom Consciousness"


# ============================================================================
# Model Listing and Info Tests
# ============================================================================

@pytest.mark.unit
def test_list_available_models(consciousness_factory):
    """Test listing available models"""
    models = consciousness_factory.list_available_models()

    assert isinstance(models, list)
    assert "test_model.yaml" in models


@pytest.mark.unit
def test_list_models_sorted(consciousness_factory):
    """Test models are returned sorted"""
    models = consciousness_factory.list_available_models()

    assert models == sorted(models)


@pytest.mark.integration
def test_list_real_models(real_consciousness_factory):
    """Test listing real models"""
    models = real_consciousness_factory.list_available_models()

    # Should have at least default, plant, explorer
    model_names = {m for m in models}

    if len(models) > 0:
        # If any models exist, check they're YAML files
        assert all(m.endswith((".yaml", ".yml")) for m in models)


@pytest.mark.unit
def test_get_model_info(consciousness_factory):
    """Test getting model metadata"""
    info = consciousness_factory.get_model_info("test_model.yaml")

    assert info["name"] == "Test Consciousness"
    assert info["description"] == "Test model for pytest"
    assert info["version"] == "1.0"
    assert "perception_fidelity" in info["dimensions"]


@pytest.mark.unit
def test_get_model_info_nonexistent(consciousness_factory):
    """Test getting info for nonexistent model raises error"""
    with pytest.raises(FileNotFoundError):
        consciousness_factory.get_model_info("nonexistent.yaml")


# ============================================================================
# Model Comparison Tests
# ============================================================================

@pytest.mark.unit
def test_compare_models_basic(temp_config_dir):
    """Test comparing two models"""
    # Create two test models
    model1_yaml = """
model_type: "graded"
name: "Model 1"
description: "First test model"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.8
    weight: 1.0
  reaction_speed:
    default: 0.6
    weight: 1.0

aggregation: "weighted_mean"
"""

    model2_yaml = """
model_type: "graded"
name: "Model 2"
description: "Second test model"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.5
    weight: 1.0
  reaction_speed:
    default: 0.9
    weight: 1.0

aggregation: "weighted_mean"
"""

    with open(os.path.join(temp_config_dir, "model1.yaml"), 'w') as f:
        f.write(model1_yaml)

    with open(os.path.join(temp_config_dir, "model2.yaml"), 'w') as f:
        f.write(model2_yaml)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    comparison = factory.compare_models("model1.yaml", "model2.yaml")

    assert "model1" in comparison
    assert "model2" in comparison
    assert "capability_differences" in comparison
    assert "overall_difference" in comparison


@pytest.mark.unit
def test_compare_models_differences(temp_config_dir):
    """Test comparison shows capability differences"""
    # Create two test models
    model1_yaml = """
model_type: "graded"
name: "Model 1"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.8
    weight: 1.0

aggregation: "weighted_mean"
"""

    model2_yaml = """
model_type: "graded"
name: "Model 2"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.3
    weight: 1.0

aggregation: "weighted_mean"
"""

    with open(os.path.join(temp_config_dir, "m1.yaml"), 'w') as f:
        f.write(model1_yaml)

    with open(os.path.join(temp_config_dir, "m2.yaml"), 'w') as f:
        f.write(model2_yaml)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    comparison = factory.compare_models("m1.yaml", "m2.yaml")

    # Perception should show difference of -0.5 (0.3 - 0.8)
    perception_diff = comparison["capability_differences"]["perception_fidelity"]

    assert perception_diff["model1"] == 0.8
    assert perception_diff["model2"] == 0.3
    assert perception_diff["difference"] == pytest.approx(-0.5)


# ============================================================================
# Model Blending Tests
# ============================================================================

@pytest.mark.unit
def test_blend_models_50_50(temp_config_dir):
    """Test 50/50 blend of two models"""
    # Create two models
    model1_yaml = """
model_type: "graded"
name: "Model 1"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.8
    weight: 1.0
  reaction_speed:
    default: 0.4
    weight: 1.0

aggregation: "weighted_mean"
"""

    model2_yaml = """
model_type: "graded"
name: "Model 2"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.4
    weight: 1.0
  reaction_speed:
    default: 0.8
    weight: 1.0

aggregation: "weighted_mean"
"""

    with open(os.path.join(temp_config_dir, "blend1.yaml"), 'w') as f:
        f.write(model1_yaml)

    with open(os.path.join(temp_config_dir, "blend2.yaml"), 'w') as f:
        f.write(model2_yaml)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    blended = factory.blend_models(
        "blend1.yaml",
        "blend2.yaml",
        blend_ratio=0.5,
        name="Blended"
    )

    scores = blended.get_capability_scores()

    # 50/50 blend: (0.8 + 0.4) / 2 = 0.6
    assert scores["perception_fidelity"] == pytest.approx(0.6)
    assert scores["reaction_speed"] == pytest.approx(0.6)


@pytest.mark.unit
def test_blend_models_80_20(temp_config_dir):
    """Test 80/20 blend favoring second model"""
    model1_yaml = """
model_type: "graded"
name: "Model 1"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.2
    weight: 1.0

aggregation: "weighted_mean"
"""

    model2_yaml = """
model_type: "graded"
name: "Model 2"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 1.0
    weight: 1.0

aggregation: "weighted_mean"
"""

    with open(os.path.join(temp_config_dir, "b1.yaml"), 'w') as f:
        f.write(model1_yaml)

    with open(os.path.join(temp_config_dir, "b2.yaml"), 'w') as f:
        f.write(model2_yaml)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    blended = factory.blend_models(
        "b1.yaml",
        "b2.yaml",
        blend_ratio=0.8,
        name="Mostly Model 2"
    )

    scores = blended.get_capability_scores()

    # 80% of model2, 20% of model1: 0.2 * 0.2 + 0.8 * 1.0 = 0.84
    assert scores["perception_fidelity"] == pytest.approx(0.84)


@pytest.mark.unit
def test_blend_models_blends_weights(temp_config_dir):
    """Test blending also blends weights"""
    model1_yaml = """
model_type: "graded"
name: "Model 1"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.5
    weight: 1.0

aggregation: "weighted_mean"
"""

    model2_yaml = """
model_type: "graded"
name: "Model 2"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.5
    weight: 3.0

aggregation: "weighted_mean"
"""

    with open(os.path.join(temp_config_dir, "w1.yaml"), 'w') as f:
        f.write(model1_yaml)

    with open(os.path.join(temp_config_dir, "w2.yaml"), 'w') as f:
        f.write(model2_yaml)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    blended = factory.blend_models(
        "w1.yaml",
        "w2.yaml",
        blend_ratio=0.5
    )

    # Weight should be blended: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
    assert blended.weights["perception_fidelity"] == pytest.approx(2.0)


# ============================================================================
# Cache Management Tests
# ============================================================================

@pytest.mark.unit
def test_clear_cache(consciousness_factory):
    """Test clearing model cache"""
    # Load model to cache it
    consciousness_factory.from_yaml("test_model.yaml")

    # Clear cache
    consciousness_factory.clear_cache()

    # Cache should be empty
    assert len(consciousness_factory._model_cache) == 0


@pytest.mark.unit
def test_cache_reload_after_clear(consciousness_factory):
    """Test reloading after cache clear loads fresh"""
    c1 = consciousness_factory.from_yaml("test_model.yaml")

    consciousness_factory.clear_cache()

    c2 = consciousness_factory.from_yaml("test_model.yaml")

    # Should be different instances (not cached)
    assert c1 is not c2


# ============================================================================
# Convenience Function Tests
# ============================================================================

@pytest.mark.unit
def test_load_consciousness_model_function():
    """Test convenience function"""
    try:
        consciousness = load_consciousness_model("default.yaml")
        assert isinstance(consciousness, GradedConsciousness)
    except FileNotFoundError:
        pytest.skip("default.yaml not found")


@pytest.mark.unit
def test_load_consciousness_model_with_overrides():
    """Test convenience function with overrides"""
    try:
        consciousness = load_consciousness_model(
            "default.yaml",
            overrides={"perception_fidelity": 0.99}
        )

        assert consciousness.perception_fidelity == 0.99
    except FileNotFoundError:
        pytest.skip("default.yaml not found")


@pytest.mark.unit
def test_load_consciousness_model_default_string():
    """Test loading 'default' string"""
    try:
        consciousness = load_consciousness_model("default")
        assert isinstance(consciousness, GradedConsciousness)
    except FileNotFoundError:
        pytest.skip("default.yaml not found")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

@pytest.mark.unit
def test_factory_with_nonexistent_dir():
    """Test factory with nonexistent config directory"""
    factory = ConsciousnessFactory(config_dir="/nonexistent/path")

    models = factory.list_available_models()

    # Should return empty list, not crash
    assert models == []


@pytest.mark.unit
def test_parse_minimal_yaml(temp_config_dir):
    """Test parsing minimal YAML configuration"""
    minimal_yaml = """
model_type: "graded"
"""

    with open(os.path.join(temp_config_dir, "minimal.yaml"), 'w') as f:
        f.write(minimal_yaml)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    # Should not crash, use defaults
    consciousness = factory.from_yaml("minimal.yaml")

    assert isinstance(consciousness, GradedConsciousness)


@pytest.mark.unit
def test_override_nonexistent_capability(consciousness_factory):
    """Test overriding capability that doesn't exist in YAML"""
    consciousness = consciousness_factory.from_yaml(
        "test_model.yaml",
        overrides={"nonexistent_capability": 0.9}
    )

    # Should not crash
    # Nonexistent capability will just be ignored
    assert isinstance(consciousness, GradedConsciousness)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_full_factory_workflow(temp_config_dir):
    """Test complete factory workflow"""
    # Create test model
    yaml_content = """
model_type: "graded"
name: "Workflow Test"
description: "Testing full workflow"
version: "1.0"
author: "Pytest"

dimensions:
  perception_fidelity:
    default: 0.75
    weight: 1.0
    min: 0.0
    max: 1.0
  reaction_speed:
    default: 0.65
    weight: 0.8

aggregation: "weighted_mean"

thresholds:
  dormant: 0.0
  aware: 0.5
  meta_aware: 0.8
"""

    with open(os.path.join(temp_config_dir, "workflow.yaml"), 'w') as f:
        f.write(yaml_content)

    factory = ConsciousnessFactory(config_dir=temp_config_dir)

    # List models
    models = factory.list_available_models()
    assert "workflow.yaml" in models

    # Get info
    info = factory.get_model_info("workflow.yaml")
    assert info["name"] == "Workflow Test"
    assert info["author"] == "Pytest"

    # Load model
    consciousness = factory.from_yaml("workflow.yaml")
    assert consciousness.model_name == "Workflow Test"

    # Load with overrides
    custom = factory.from_yaml(
        "workflow.yaml",
        overrides={"perception_fidelity": 0.95}
    )
    assert custom.perception_fidelity == 0.95

    # Create programmatically
    created = factory.create(
        name="Created",
        perception_fidelity=0.8
    )
    assert created.model_name == "Created"

    # Clear cache
    factory.clear_cache()
