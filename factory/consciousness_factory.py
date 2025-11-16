"""
Consciousness Factory

Creates consciousness models from YAML configuration files.
Enables non-programmers to design consciousness models through configuration.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Import protocols
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessModelConfig:
    """Parsed consciousness model configuration"""
    model_type: str
    name: str
    description: str
    version: str
    dimensions: Dict[str, Dict[str, Any]]
    custom_dimensions: Dict[str, Dict[str, Any]]
    aggregation: str
    thresholds: Dict[str, float]
    metadata: Dict[str, Any]


class ConsciousnessFactory:
    """
    Factory for creating consciousness models from configuration.

    Supports:
    - YAML configuration files
    - Programmatic creation
    - Model composition
    - Model inheritance
    - Custom dimension types

    Example:
        # From YAML
        factory = ConsciousnessFactory()
        consciousness = factory.from_yaml("configs/plant_consciousness.yaml")

        # Programmatic
        consciousness = factory.create(
            name="CustomAgent",
            perception_fidelity=0.8,
            meta_cognitive_ability=0.6
        )

        # With overrides
        consciousness = factory.from_yaml(
            "configs/default.yaml",
            overrides={"perception_fidelity": 0.9}
        )
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize factory.

        Args:
            config_dir: Directory containing consciousness model configs
        """
        self.config_dir = config_dir or self._find_config_dir()
        self._model_cache = {}

    def _find_config_dir(self) -> str:
        """Find the consciousness models configuration directory"""
        # Try relative to this file
        current_dir = os.path.dirname(__file__)
        config_dir = os.path.join(current_dir, "..", "configs", "consciousness_models")

        if os.path.exists(config_dir):
            return os.path.abspath(config_dir)

        # Try current working directory
        config_dir = os.path.join(os.getcwd(), "configs", "consciousness_models")

        if os.path.exists(config_dir):
            return config_dir

        # Default
        return "./configs/consciousness_models"

    def from_yaml(
        self,
        filepath: str,
        overrides: Optional[Dict[str, float]] = None
    ) -> GradedConsciousness:
        """
        Load consciousness model from YAML file.

        Args:
            filepath: Path to YAML file (absolute or relative to config_dir)
            overrides: Optional capability overrides

        Returns:
            GradedConsciousness instance

        Example:
            consciousness = factory.from_yaml("plant_consciousness.yaml")
            consciousness = factory.from_yaml(
                "default.yaml",
                overrides={"perception_fidelity": 0.95}
            )
        """
        # Resolve filepath
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.config_dir, filepath)

        # Check cache
        cache_key = f"{filepath}:{str(overrides)}"
        if cache_key in self._model_cache:
            logger.debug(f"Loading {filepath} from cache")
            return self._model_cache[cache_key]

        # Load YAML
        logger.info(f"Loading consciousness model from {filepath}")

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Parse configuration
        model_config = self._parse_config(config)

        # Build consciousness
        consciousness = self._build_consciousness(model_config, overrides)

        # Cache
        self._model_cache[cache_key] = consciousness

        logger.info(
            f"Created {model_config.name} "
            f"(score: {consciousness.overall_consciousness_score():.2f})"
        )

        return consciousness

    def _parse_config(self, config: Dict[str, Any]) -> ConsciousnessModelConfig:
        """Parse YAML config into structured model config"""
        return ConsciousnessModelConfig(
            model_type=config.get("model_type", "graded"),
            name=config.get("name", "Unnamed Model"),
            description=config.get("description", ""),
            version=config.get("version", "1.0"),
            dimensions=config.get("dimensions", {}),
            custom_dimensions=config.get("custom_dimensions", {}),
            aggregation=config.get("aggregation", "weighted_mean"),
            thresholds=config.get("thresholds", {}),
            metadata={
                "author": config.get("author"),
                "created": config.get("created"),
                "tags": config.get("tags", []),
                "references": config.get("references", [])
            }
        )

    def _build_consciousness(
        self,
        config: ConsciousnessModelConfig,
        overrides: Optional[Dict[str, float]] = None
    ) -> GradedConsciousness:
        """Build GradedConsciousness from parsed config"""

        # Standard capability names (all that GradedConsciousness supports)
        standard_capabilities = [
            "perception_fidelity",
            "reaction_speed",
            "memory_depth",
            "memory_recall_accuracy",
            "introspection_capacity",
            "meta_cognitive_ability",
            "information_integration",
            "intentional_coherence",
            "qualia_richness"
        ]

        # Collect capability scores
        capability_scores = {}
        weights = {}

        # Initialize all standard capabilities to 0.0 first
        for cap_name in standard_capabilities:
            capability_scores[cap_name] = 0.0
            weights[cap_name] = 1.0

        # Override with values from YAML config
        for dim_name, dim_config in config.dimensions.items():
            default_value = dim_config.get("default", 0.5)
            weight = dim_config.get("weight", 1.0)

            # Apply override if present
            if overrides and dim_name in overrides:
                value = overrides[dim_name]
            else:
                value = default_value

            # Clamp to min/max
            min_val = dim_config.get("min", 0.0)
            max_val = dim_config.get("max", 1.0)
            value = max(min_val, min(max_val, value))

            capability_scores[dim_name] = value
            weights[dim_name] = weight

        # Custom dimensions (beyond standard 9)
        for dim_name, dim_config in config.custom_dimensions.items():
            default_value = dim_config.get("default", 0.5)
            weight = dim_config.get("weight", 1.0)

            if overrides and dim_name in overrides:
                value = overrides[dim_name]
            else:
                value = default_value

            min_val = dim_config.get("min", 0.0)
            max_val = dim_config.get("max", 1.0)
            value = max(min_val, min(max_val, value))

            capability_scores[dim_name] = value
            weights[dim_name] = weight

        # Create consciousness
        consciousness = GradedConsciousness(**capability_scores)
        consciousness.weights = weights
        consciousness.model_name = config.name
        consciousness.model_version = config.version
        consciousness.model_description = config.description
        consciousness.thresholds = config.thresholds

        return consciousness

    def create(
        self,
        name: str = "Custom Consciousness",
        **capabilities: float
    ) -> GradedConsciousness:
        """
        Create consciousness programmatically.

        Args:
            name: Model name
            **capabilities: Capability scores (0.0 to 1.0)

        Returns:
            GradedConsciousness instance

        Example:
            consciousness = factory.create(
                name="FastReactive",
                perception_fidelity=0.9,
                reaction_speed=0.95,
                memory_depth=0.3
            )
        """
        consciousness = GradedConsciousness(**capabilities)
        consciousness.model_name = name

        logger.info(
            f"Created {name} "
            f"(score: {consciousness.overall_consciousness_score():.2f})"
        )

        return consciousness

    def list_available_models(self) -> List[str]:
        """List all available YAML models in config directory"""
        if not os.path.exists(self.config_dir):
            logger.warning(f"Config directory not found: {self.config_dir}")
            return []

        models = []

        for filename in os.listdir(self.config_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                models.append(filename)

        return sorted(models)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get metadata about a model without loading it.

        Args:
            model_name: Model filename (e.g., "plant_consciousness.yaml")

        Returns:
            Model metadata
        """
        filepath = os.path.join(self.config_dir, model_name)

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        return {
            "name": config.get("name"),
            "description": config.get("description"),
            "version": config.get("version"),
            "author": config.get("author"),
            "created": config.get("created"),
            "tags": config.get("tags", []),
            "dimensions": list(config.get("dimensions", {}).keys()),
            "custom_dimensions": list(config.get("custom_dimensions", {}).keys())
        }

    def compare_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """
        Compare two consciousness models.

        Args:
            model1: First model filename
            model2: Second model filename

        Returns:
            Comparison showing differences
        """
        c1 = self.from_yaml(model1)
        c2 = self.from_yaml(model2)

        scores1 = c1.get_capability_scores()
        scores2 = c2.get_capability_scores()

        differences = {}

        all_capabilities = set(scores1.keys()) | set(scores2.keys())

        for capability in all_capabilities:
            score1 = scores1.get(capability, 0.0)
            score2 = scores2.get(capability, 0.0)
            diff = score2 - score1

            differences[capability] = {
                "model1": score1,
                "model2": score2,
                "difference": diff,
                "percent_change": (diff / score1 * 100) if score1 > 0 else float('inf')
            }

        return {
            "model1": {
                "name": getattr(c1, 'model_name', model1),
                "overall_score": c1.overall_consciousness_score()
            },
            "model2": {
                "name": getattr(c2, 'model_name', model2),
                "overall_score": c2.overall_consciousness_score()
            },
            "overall_difference": c2.overall_consciousness_score() - c1.overall_consciousness_score(),
            "capability_differences": differences
        }

    def blend_models(
        self,
        model1: str,
        model2: str,
        blend_ratio: float = 0.5,
        name: str = "Blended Model"
    ) -> GradedConsciousness:
        """
        Blend two consciousness models.

        Args:
            model1: First model filename
            model2: Second model filename
            blend_ratio: How much of model2 to use (0.0 = all model1, 1.0 = all model2)
            name: Name for blended model

        Returns:
            Blended GradedConsciousness

        Example:
            # 50/50 blend of default and plant consciousness
            hybrid = factory.blend_models("default.yaml", "plant_consciousness.yaml")

            # 80% plant, 20% default
            plant_like = factory.blend_models(
                "default.yaml",
                "plant_consciousness.yaml",
                blend_ratio=0.8
            )
        """
        c1 = self.from_yaml(model1)
        c2 = self.from_yaml(model2)

        scores1 = c1.get_capability_scores()
        scores2 = c2.get_capability_scores()

        # Blend scores
        blended_scores = {}
        all_capabilities = set(scores1.keys()) | set(scores2.keys())

        for capability in all_capabilities:
            score1 = scores1.get(capability, 0.0)
            score2 = scores2.get(capability, 0.0)

            blended_scores[capability] = (
                score1 * (1 - blend_ratio) +
                score2 * blend_ratio
            )

        # Blend weights
        weights1 = getattr(c1, 'weights', {})
        weights2 = getattr(c2, 'weights', {})

        blended_weights = {}
        for capability in all_capabilities:
            w1 = weights1.get(capability, 1.0)
            w2 = weights2.get(capability, 1.0)
            blended_weights[capability] = w1 * (1 - blend_ratio) + w2 * blend_ratio

        # Create blended consciousness
        blended = GradedConsciousness(**blended_scores)
        blended.weights = blended_weights
        blended.model_name = name

        logger.info(
            f"Blended {getattr(c1, 'model_name', model1)} and "
            f"{getattr(c2, 'model_name', model2)} "
            f"(ratio: {blend_ratio:.2f}, score: {blended.overall_consciousness_score():.2f})"
        )

        return blended

    def clear_cache(self):
        """Clear model cache"""
        self._model_cache.clear()
        logger.info("Model cache cleared")


# Convenience function
def load_consciousness_model(
    model: str,
    overrides: Optional[Dict[str, float]] = None
) -> GradedConsciousness:
    """
    Convenience function to load a consciousness model.

    Args:
        model: Model filename or "default"
        overrides: Optional capability overrides

    Returns:
        GradedConsciousness instance

    Example:
        consciousness = load_consciousness_model("plant_consciousness.yaml")
        consciousness = load_consciousness_model(
            "default.yaml",
            overrides={"perception_fidelity": 0.95}
        )
    """
    factory = ConsciousnessFactory()

    if model == "default" or model == "":
        model = "default.yaml"

    return factory.from_yaml(model, overrides)
