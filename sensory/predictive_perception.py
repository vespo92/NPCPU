"""
SYNAPSE-M: Predictive Perception

Prediction-based perception using predictive coding principles.
Implements anticipatory perception, prediction error computation,
and model-based sensory processing.

Part of Agent 6: Multi-Modal Perception Engine
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod

from .modality_types import (
    ExtendedModality, ProcessedModality, ModalityCharacteristics,
    get_modality_characteristics, ModalityInput
)


# ============================================================================
# Prediction Types
# ============================================================================

class PredictionType(Enum):
    """Types of predictions"""
    TEMPORAL = auto()       # What will happen next
    SPATIAL = auto()        # Where something will be
    FEATURE = auto()        # What features to expect
    CROSS_MODAL = auto()    # Predict one modality from another
    SEMANTIC = auto()       # Predict meaning/content


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    HIGH = 0.9
    MODERATE = 0.7
    LOW = 0.5
    UNCERTAIN = 0.3


# ============================================================================
# Prediction Data Structures
# ============================================================================

@dataclass
class Prediction:
    """
    A prediction about future sensory input.
    """
    id: str = field(default_factory=lambda: str(id(object())))
    modality: ExtendedModality = ExtendedModality.VISION
    prediction_type: PredictionType = PredictionType.TEMPORAL

    # Predicted content
    predicted_features: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_semantic: Optional[np.ndarray] = None
    predicted_intensity: float = 0.5

    # Timing
    prediction_horizon_ms: float = 100.0  # How far ahead
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 0.5)

    # Confidence
    confidence: float = 0.5
    prior_strength: float = 1.0  # How much to weight this prior

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if prediction has expired"""
        return time.time() > self.expires_at


@dataclass
class PredictionError:
    """
    Error between prediction and actual perception.
    """
    modality: ExtendedModality
    prediction: Prediction
    actual: ProcessedModality

    # Error metrics
    feature_error: float = 0.0         # Feature space error
    semantic_error: float = 0.0        # Semantic space error
    intensity_error: float = 0.0       # Intensity error
    temporal_error_ms: float = 0.0     # Timing error

    # Aggregates
    total_error: float = 0.0
    surprise: float = 0.0              # How surprising was this?

    timestamp: float = field(default_factory=time.time)

    def is_significant(self, threshold: float = 0.3) -> bool:
        """Check if error is significant"""
        return self.total_error > threshold


@dataclass
class PredictiveState:
    """
    Current state of predictive perception for a modality.
    """
    modality: ExtendedModality
    active_predictions: List[Prediction] = field(default_factory=list)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    prediction_accuracy: float = 0.5
    model_confidence: float = 0.5
    last_update: float = field(default_factory=time.time)


# ============================================================================
# Predictive Models
# ============================================================================

class PredictiveModel(ABC):
    """
    Abstract base for predictive models.
    """

    def __init__(self, modality: ExtendedModality):
        self.modality = modality
        self.characteristics = get_modality_characteristics(modality)

    @abstractmethod
    def predict(
        self,
        history: List[ProcessedModality],
        horizon_ms: float
    ) -> Prediction:
        """Generate prediction from history"""
        pass

    @abstractmethod
    def update(self, error: PredictionError):
        """Update model based on prediction error"""
        pass


class TemporalPredictiveModel(PredictiveModel):
    """
    Temporal prediction using simple autoregressive model.
    """

    def __init__(
        self,
        modality: ExtendedModality,
        feature_dim: int = 128,
        learning_rate: float = 0.01
    ):
        super().__init__(modality)
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate

        # Simple linear prediction weights
        np.random.seed(48)
        self.weights = np.random.randn(feature_dim, feature_dim) * 0.1
        self.bias = np.zeros(feature_dim)

    def predict(
        self,
        history: List[ProcessedModality],
        horizon_ms: float = 100.0
    ) -> Prediction:
        """Generate temporal prediction"""
        if not history:
            return Prediction(
                modality=self.modality,
                prediction_type=PredictionType.TEMPORAL,
                predicted_features=np.zeros(self.feature_dim),
                confidence=0.1,
                prediction_horizon_ms=horizon_ms
            )

        # Use most recent perception
        latest = history[-1]

        # Align dimensions
        features = latest.features
        if len(features) != self.feature_dim:
            aligned = np.zeros(self.feature_dim)
            min_dim = min(len(features), self.feature_dim)
            aligned[:min_dim] = features[:min_dim]
            features = aligned

        # Linear prediction
        predicted = np.tanh(features @ self.weights + self.bias)

        # Confidence based on history length and variance
        if len(history) > 5:
            variance = np.var([h.features[:min(len(h.features), 10)] for h in history[-5:]])
            confidence = 1.0 / (1.0 + variance)
        else:
            confidence = 0.3 + 0.1 * len(history)

        return Prediction(
            modality=self.modality,
            prediction_type=PredictionType.TEMPORAL,
            predicted_features=predicted,
            predicted_intensity=latest.raw_input.intensity * 0.9,  # Slight decay
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            prediction_horizon_ms=horizon_ms,
            expires_at=time.time() + horizon_ms / 1000.0
        )

    def update(self, error: PredictionError):
        """Update weights based on prediction error"""
        if error.prediction.predicted_features.size == 0:
            return

        # Compute gradient (simplified)
        predicted = error.prediction.predicted_features
        actual = error.actual.features

        # Align dimensions
        min_dim = min(len(predicted), len(actual), self.feature_dim)
        predicted_aligned = predicted[:min_dim]
        actual_aligned = actual[:min_dim]

        # Error gradient
        gradient = (predicted_aligned - actual_aligned)

        # Update weights (gradient descent)
        if len(gradient) == self.feature_dim:
            update = np.outer(gradient, predicted_aligned)
            self.weights[:min_dim, :min_dim] -= self.learning_rate * update[:min_dim, :min_dim]
            self.bias[:min_dim] -= self.learning_rate * gradient


class CrossModalPredictiveModel(PredictiveModel):
    """
    Cross-modal prediction: predict one modality from another.
    """

    def __init__(
        self,
        source_modality: ExtendedModality,
        target_modality: ExtendedModality,
        source_dim: int = 128,
        target_dim: int = 128
    ):
        super().__init__(target_modality)
        self.source_modality = source_modality
        self.source_dim = source_dim
        self.target_dim = target_dim

        # Cross-modal mapping
        np.random.seed(49)
        self.cross_weights = np.random.randn(source_dim, target_dim) * 0.1

    def predict(
        self,
        history: List[ProcessedModality],
        horizon_ms: float = 50.0
    ) -> Prediction:
        """Predict target modality from source"""
        if not history:
            return Prediction(
                modality=self.modality,
                prediction_type=PredictionType.CROSS_MODAL,
                predicted_features=np.zeros(self.target_dim),
                confidence=0.1
            )

        # Get source features
        source_features = history[-1].features
        if len(source_features) > self.source_dim:
            source_features = source_features[:self.source_dim]
        else:
            padded = np.zeros(self.source_dim)
            padded[:len(source_features)] = source_features
            source_features = padded

        # Cross-modal prediction
        predicted = np.tanh(source_features @ self.cross_weights)

        return Prediction(
            modality=self.modality,
            prediction_type=PredictionType.CROSS_MODAL,
            predicted_features=predicted,
            confidence=0.5,
            prediction_horizon_ms=horizon_ms,
            metadata={"source_modality": self.source_modality.value}
        )

    def update(self, error: PredictionError):
        """Update cross-modal mapping"""
        # Simplified update - would use proper gradient in production
        pass


# ============================================================================
# Predictive Perception System
# ============================================================================

class PredictivePerceptionSystem:
    """
    System implementing predictive coding for perception.

    Key principles:
    1. Brain maintains model of world
    2. Perception = prediction + prediction error
    3. Minimize surprise (prediction error)
    4. Update model when predictions fail
    """

    def __init__(
        self,
        default_horizon_ms: float = 100.0,
        error_threshold: float = 0.3,
        history_size: int = 50
    ):
        self.default_horizon_ms = default_horizon_ms
        self.error_threshold = error_threshold
        self.history_size = history_size

        # Predictive models per modality
        self.models: Dict[ExtendedModality, PredictiveModel] = {}

        # State per modality
        self.states: Dict[ExtendedModality, PredictiveState] = {}

        # Perception history per modality
        self.history: Dict[ExtendedModality, deque] = {}

        # Cross-modal models
        self.cross_modal_models: Dict[Tuple[ExtendedModality, ExtendedModality], CrossModalPredictiveModel] = {}

        # Statistics
        self.total_predictions = 0
        self.accurate_predictions = 0

    def register_modality(self, modality: ExtendedModality):
        """Register a modality for predictive processing"""
        chars = get_modality_characteristics(modality)

        self.models[modality] = TemporalPredictiveModel(
            modality=modality,
            feature_dim=chars.feature_dim
        )

        self.states[modality] = PredictiveState(modality=modality)
        self.history[modality] = deque(maxlen=self.history_size)

    def register_cross_modal_link(
        self,
        source: ExtendedModality,
        target: ExtendedModality
    ):
        """Register cross-modal prediction link"""
        source_chars = get_modality_characteristics(source)
        target_chars = get_modality_characteristics(target)

        self.cross_modal_models[(source, target)] = CrossModalPredictiveModel(
            source_modality=source,
            target_modality=target,
            source_dim=source_chars.feature_dim,
            target_dim=target_chars.feature_dim
        )

    def generate_predictions(
        self,
        modality: ExtendedModality,
        horizon_ms: Optional[float] = None
    ) -> List[Prediction]:
        """Generate predictions for a modality"""
        if modality not in self.models:
            self.register_modality(modality)

        horizon = horizon_ms or self.default_horizon_ms
        predictions = []

        # Temporal prediction
        if modality in self.history and len(self.history[modality]) > 0:
            pred = self.models[modality].predict(
                list(self.history[modality]),
                horizon
            )
            predictions.append(pred)

        # Cross-modal predictions
        for (source, target), model in self.cross_modal_models.items():
            if target == modality and source in self.history:
                cross_pred = model.predict(
                    list(self.history[source]),
                    horizon
                )
                predictions.append(cross_pred)

        # Store active predictions
        self.states[modality].active_predictions = predictions
        self.total_predictions += len(predictions)

        return predictions

    def process_perception(
        self,
        perception: ProcessedModality
    ) -> Tuple[ProcessedModality, List[PredictionError]]:
        """
        Process perception with predictive coding.

        Returns:
            Tuple of (enhanced perception, prediction errors)
        """
        modality = perception.modality

        if modality not in self.models:
            self.register_modality(modality)

        # Get active predictions
        predictions = self.states[modality].active_predictions

        # Compute prediction errors
        errors = []
        for pred in predictions:
            if not pred.is_expired():
                error = self._compute_error(pred, perception)
                errors.append(error)

                # Update model
                self.models[modality].update(error)

                # Track accuracy
                if not error.is_significant(self.error_threshold):
                    self.accurate_predictions += 1

        # Store in history
        self.history[modality].append(perception)

        # Enhance perception with prediction context
        enhanced = self._enhance_perception(perception, predictions, errors)

        # Update state
        self._update_state(modality, errors)

        return enhanced, errors

    def _compute_error(
        self,
        prediction: Prediction,
        actual: ProcessedModality
    ) -> PredictionError:
        """Compute prediction error"""
        # Feature error
        pred_features = prediction.predicted_features
        actual_features = actual.features

        min_dim = min(len(pred_features), len(actual_features))
        if min_dim > 0:
            feature_diff = pred_features[:min_dim] - actual_features[:min_dim]
            feature_error = np.linalg.norm(feature_diff) / np.sqrt(min_dim)
        else:
            feature_error = 1.0

        # Semantic error
        if prediction.predicted_semantic is not None:
            semantic_diff = prediction.predicted_semantic[:min_dim] - actual.semantic_embedding[:min_dim]
            semantic_error = np.linalg.norm(semantic_diff) / np.sqrt(min_dim)
        else:
            semantic_error = 0.5

        # Intensity error
        intensity_error = abs(prediction.predicted_intensity - actual.raw_input.intensity)

        # Temporal error
        expected_time = prediction.created_at + prediction.prediction_horizon_ms / 1000.0
        temporal_error = abs(actual.timestamp - expected_time) * 1000.0

        # Total error (weighted)
        total_error = (
            0.5 * feature_error +
            0.3 * semantic_error +
            0.1 * intensity_error +
            0.1 * min(temporal_error / 100.0, 1.0)
        )

        # Surprise (how unexpected)
        surprise = total_error * (1 - prediction.confidence)

        return PredictionError(
            modality=prediction.modality,
            prediction=prediction,
            actual=actual,
            feature_error=float(feature_error),
            semantic_error=float(semantic_error),
            intensity_error=float(intensity_error),
            temporal_error_ms=float(temporal_error),
            total_error=float(np.clip(total_error, 0.0, 1.0)),
            surprise=float(np.clip(surprise, 0.0, 1.0))
        )

    def _enhance_perception(
        self,
        perception: ProcessedModality,
        predictions: List[Prediction],
        errors: List[PredictionError]
    ) -> ProcessedModality:
        """Enhance perception using predictions"""
        if not predictions:
            return perception

        # Combine prediction with actual (predictive coding)
        # Perception = weighted combination of prediction + error
        avg_confidence = np.mean([p.confidence for p in predictions])

        if errors:
            avg_error = np.mean([e.total_error for e in errors])
        else:
            avg_error = 0.5

        # More error = weight actual more, less error = weight prediction more
        actual_weight = 0.5 + 0.5 * avg_error
        pred_weight = 1.0 - actual_weight

        # Blend features
        pred_features = np.mean([p.predicted_features for p in predictions], axis=0)
        min_dim = min(len(pred_features), len(perception.features))

        blended_features = (
            actual_weight * perception.features[:min_dim] +
            pred_weight * pred_features[:min_dim]
        )

        # Create enhanced perception
        enhanced = ProcessedModality(
            modality=perception.modality,
            raw_input=perception.raw_input,
            features=np.concatenate([blended_features, perception.features[min_dim:]]),
            semantic_embedding=perception.semantic_embedding,
            salience=perception.salience + (1 - avg_error) * 0.1,  # Predicted = less salient
            confidence=perception.confidence * avg_confidence,
            processing_time_ms=perception.processing_time_ms,
            timestamp=perception.timestamp,
            metadata={
                **perception.metadata,
                "prediction_enhanced": True,
                "prediction_error": avg_error,
                "prediction_weight": pred_weight
            }
        )

        return enhanced

    def _update_state(self, modality: ExtendedModality, errors: List[PredictionError]):
        """Update predictive state"""
        state = self.states[modality]
        state.last_update = time.time()

        for error in errors:
            state.recent_errors.append(error)

        # Update accuracy estimate
        if state.recent_errors:
            recent_total = [e.total_error for e in state.recent_errors]
            state.prediction_accuracy = 1.0 - np.mean(recent_total)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        accuracy = (
            self.accurate_predictions / self.total_predictions
            if self.total_predictions > 0 else 0.5
        )

        modality_stats = {}
        for modality, state in self.states.items():
            modality_stats[modality.value] = {
                "accuracy": state.prediction_accuracy,
                "active_predictions": len(state.active_predictions),
                "recent_errors": len(state.recent_errors)
            }

        return {
            "total_predictions": self.total_predictions,
            "accurate_predictions": self.accurate_predictions,
            "overall_accuracy": accuracy,
            "registered_modalities": len(self.models),
            "cross_modal_links": len(self.cross_modal_models),
            "modality_stats": modality_stats
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("SYNAPSE-M: Predictive Perception Demo")
    print("=" * 50)

    # Create predictive system
    system = PredictivePerceptionSystem()

    # Register modalities
    system.register_modality(ExtendedModality.VISION)
    system.register_modality(ExtendedModality.AUDITION)

    # Add cross-modal link
    system.register_cross_modal_link(
        ExtendedModality.VISION,
        ExtendedModality.AUDITION
    )

    np.random.seed(42)

    print("\n1. Simulating perception sequence...")

    for i in range(10):
        # Generate perception
        visual_input = ModalityInput(
            modality=ExtendedModality.VISION,
            raw_data=np.random.randn(256),
            intensity=0.8 + np.random.uniform(-0.1, 0.1)
        )

        perception = ProcessedModality(
            modality=ExtendedModality.VISION,
            raw_input=visual_input,
            features=np.random.randn(256) + i * 0.1,  # Slowly changing
            semantic_embedding=np.random.randn(128),
            salience=0.7,
            confidence=0.9,
            timestamp=time.time()
        )

        # Generate predictions
        predictions = system.generate_predictions(ExtendedModality.VISION)

        # Process with predictive coding
        enhanced, errors = system.process_perception(perception)

        if errors:
            avg_error = np.mean([e.total_error for e in errors])
            print(f"   Step {i}: prediction error = {avg_error:.3f}")
        else:
            print(f"   Step {i}: no predictions yet")

    print("\n2. System statistics:")
    stats = system.get_statistics()
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Accuracy: {stats['overall_accuracy']:.3f}")
    print(f"   Registered modalities: {stats['registered_modalities']}")
    print(f"   Cross-modal links: {stats['cross_modal_links']}")

    print("\n3. Modality-specific stats:")
    for modality, mstats in stats['modality_stats'].items():
        print(f"   {modality}:")
        for key, value in mstats.items():
            print(f"      {key}: {value}")
