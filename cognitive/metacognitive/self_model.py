"""
Self-Model System for ORACLE-Z Metacognitive Agent

Implements internal self-representation capabilities that allow the system
to build, maintain, and update an accurate model of its own cognitive state,
capabilities, and limitations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import hashlib
from abc import ABC, abstractmethod


class ModelAccuracy(Enum):
    """Accuracy levels of self-model predictions"""
    UNCERTAIN = "uncertain"      # < 0.3
    LOW = "low"                  # 0.3-0.5
    MODERATE = "moderate"        # 0.5-0.7
    HIGH = "high"                # 0.7-0.9
    VERY_HIGH = "very_high"      # > 0.9


class CognitiveComponent(Enum):
    """Components of the cognitive system that can be modeled"""
    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    LEARNING = "learning"
    PLANNING = "planning"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    METACOGNITION = "metacognition"


@dataclass
class ComponentState:
    """State of a single cognitive component"""
    component: CognitiveComponent
    active: bool = True
    health: float = 1.0  # 0.0 to 1.0
    load: float = 0.0  # 0.0 to 1.0
    performance: float = 1.0  # 0.0 to 1.0
    last_update: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert state to vector representation"""
        return np.array([
            float(self.active),
            self.health,
            self.load,
            self.performance
        ])

    def update_history(self, max_entries: int = 100):
        """Record current state in history"""
        self.history.append({
            "health": self.health,
            "load": self.load,
            "performance": self.performance,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]


@dataclass
class SelfModelPrediction:
    """A prediction made by the self-model"""
    target: str  # What is being predicted
    predicted_value: Any
    confidence: float
    uncertainty_bounds: Tuple[float, float]
    timestamp: datetime = field(default_factory=datetime.now)
    actual_value: Optional[Any] = None
    prediction_error: Optional[float] = None

    def evaluate(self, actual: Any) -> float:
        """Evaluate prediction accuracy"""
        self.actual_value = actual
        if isinstance(actual, (int, float)) and isinstance(self.predicted_value, (int, float)):
            self.prediction_error = abs(actual - self.predicted_value)
            # Normalize error to 0-1 range
            return max(0, 1 - self.prediction_error / max(abs(actual), 1e-6))
        return 0.5  # Default for non-numeric


@dataclass
class Capability:
    """A specific capability of the system"""
    name: str
    level: float  # 0.0 to 1.0
    reliability: float  # How consistent is this capability
    dependencies: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    improvement_potential: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "reliability": self.reliability,
            "dependencies": self.dependencies,
            "limitations": self.limitations,
            "improvement_potential": self.improvement_potential
        }


@dataclass
class Limitation:
    """A recognized limitation of the system"""
    name: str
    severity: float  # 0.0 to 1.0
    scope: List[CognitiveComponent]
    workarounds: List[str] = field(default_factory=list)
    is_fundamental: bool = False
    can_improve: bool = True


class SelfModel:
    """
    Core self-model implementation for ORACLE-Z.

    Maintains an internal representation of the system's:
    - Cognitive components and their states
    - Capabilities and limitations
    - Performance predictions
    - Historical performance data
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or f"self_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Component states
        self.components: Dict[CognitiveComponent, ComponentState] = {}
        self._initialize_components()

        # Capabilities and limitations
        self.capabilities: Dict[str, Capability] = {}
        self.limitations: Dict[str, Limitation] = {}

        # Prediction tracking
        self.predictions: List[SelfModelPrediction] = []
        self.prediction_accuracy_history: List[float] = []

        # Model parameters
        self.model_confidence: float = 0.5
        self.model_completeness: float = 0.3
        self.last_update: datetime = datetime.now()

        # Internal representation
        self._state_vector: np.ndarray = np.zeros(64)
        self._update_state_vector()

        # Calibration data
        self._calibration_samples: List[Tuple[np.ndarray, np.ndarray]] = []

    def _initialize_components(self):
        """Initialize all cognitive component states"""
        for component in CognitiveComponent:
            self.components[component] = ComponentState(
                component=component,
                health=np.random.uniform(0.7, 1.0),
                performance=np.random.uniform(0.6, 0.9)
            )

    def _update_state_vector(self):
        """Update the internal state vector representation"""
        vectors = []
        for component in CognitiveComponent:
            if component in self.components:
                vectors.append(self.components[component].to_vector())
            else:
                vectors.append(np.zeros(4))

        # Concatenate and pad to fixed size
        full_vector = np.concatenate(vectors)
        self._state_vector[:len(full_vector)] = full_vector

    def update_component(self,
                        component: CognitiveComponent,
                        health: Optional[float] = None,
                        load: Optional[float] = None,
                        performance: Optional[float] = None,
                        parameters: Optional[Dict[str, Any]] = None):
        """Update a cognitive component's state"""
        if component not in self.components:
            self.components[component] = ComponentState(component=component)

        state = self.components[component]

        if health is not None:
            state.health = np.clip(health, 0.0, 1.0)
        if load is not None:
            state.load = np.clip(load, 0.0, 1.0)
        if performance is not None:
            state.performance = np.clip(performance, 0.0, 1.0)
        if parameters is not None:
            state.parameters.update(parameters)

        state.last_update = datetime.now()
        state.update_history()

        self._update_state_vector()
        self.last_update = datetime.now()

    def register_capability(self,
                           name: str,
                           level: float,
                           reliability: float = 0.8,
                           dependencies: Optional[List[str]] = None,
                           limitations: Optional[List[str]] = None):
        """Register a system capability"""
        self.capabilities[name] = Capability(
            name=name,
            level=np.clip(level, 0.0, 1.0),
            reliability=np.clip(reliability, 0.0, 1.0),
            dependencies=dependencies or [],
            limitations=limitations or []
        )

    def register_limitation(self,
                           name: str,
                           severity: float,
                           scope: List[CognitiveComponent],
                           workarounds: Optional[List[str]] = None,
                           is_fundamental: bool = False):
        """Register a system limitation"""
        self.limitations[name] = Limitation(
            name=name,
            severity=np.clip(severity, 0.0, 1.0),
            scope=scope,
            workarounds=workarounds or [],
            is_fundamental=is_fundamental,
            can_improve=not is_fundamental
        )

    def predict_performance(self,
                           task_type: str,
                           complexity: float = 0.5) -> SelfModelPrediction:
        """Predict performance on a given task type"""
        # Identify relevant components
        relevant_components = self._get_relevant_components(task_type)

        # Calculate predicted performance
        if relevant_components:
            component_performances = [
                self.components[c].performance
                for c in relevant_components
                if c in self.components
            ]
            base_prediction = np.mean(component_performances) if component_performances else 0.5
        else:
            base_prediction = 0.5

        # Adjust for task complexity
        complexity_factor = 1.0 - (complexity * 0.3)
        predicted_performance = base_prediction * complexity_factor

        # Calculate confidence and uncertainty
        confidence = self._calculate_prediction_confidence(task_type)
        uncertainty = (1 - confidence) * 0.2

        prediction = SelfModelPrediction(
            target=f"performance_{task_type}",
            predicted_value=predicted_performance,
            confidence=confidence,
            uncertainty_bounds=(
                max(0, predicted_performance - uncertainty),
                min(1, predicted_performance + uncertainty)
            )
        )

        self.predictions.append(prediction)
        return prediction

    def _get_relevant_components(self, task_type: str) -> List[CognitiveComponent]:
        """Determine which components are relevant for a task type"""
        task_component_map = {
            "learning": [CognitiveComponent.LEARNING, CognitiveComponent.MEMORY],
            "reasoning": [CognitiveComponent.REASONING, CognitiveComponent.MEMORY],
            "planning": [CognitiveComponent.PLANNING, CognitiveComponent.REASONING],
            "perception": [CognitiveComponent.PERCEPTION],
            "execution": [CognitiveComponent.EXECUTION, CognitiveComponent.MONITORING],
            "metacognition": [CognitiveComponent.METACOGNITION, CognitiveComponent.MONITORING]
        }

        return task_component_map.get(task_type, list(CognitiveComponent))

    def _calculate_prediction_confidence(self, task_type: str) -> float:
        """Calculate confidence in prediction based on historical accuracy"""
        # Base confidence on model completeness
        base_confidence = self.model_completeness

        # Adjust based on historical predictions
        relevant_predictions = [
            p for p in self.predictions
            if p.target.endswith(task_type) and p.actual_value is not None
        ]

        if relevant_predictions:
            accuracies = [
                1 - (p.prediction_error or 0)
                for p in relevant_predictions[-10:]
            ]
            historical_accuracy = np.mean(accuracies)
            confidence = base_confidence * 0.5 + historical_accuracy * 0.5
        else:
            confidence = base_confidence * 0.7

        return np.clip(confidence, 0.1, 0.95)

    def assess_accuracy(self) -> ModelAccuracy:
        """Assess overall model accuracy"""
        if len(self.prediction_accuracy_history) < 5:
            return ModelAccuracy.UNCERTAIN

        avg_accuracy = np.mean(self.prediction_accuracy_history[-20:])

        if avg_accuracy < 0.3:
            return ModelAccuracy.UNCERTAIN
        elif avg_accuracy < 0.5:
            return ModelAccuracy.LOW
        elif avg_accuracy < 0.7:
            return ModelAccuracy.MODERATE
        elif avg_accuracy < 0.9:
            return ModelAccuracy.HIGH
        else:
            return ModelAccuracy.VERY_HIGH

    def get_overall_health(self) -> float:
        """Get overall system health based on component states"""
        if not self.components:
            return 0.5
        return np.mean([c.health for c in self.components.values()])

    def get_overall_performance(self) -> float:
        """Get overall system performance"""
        if not self.components:
            return 0.5
        return np.mean([c.performance for c in self.components.values()])

    def get_bottlenecks(self) -> List[CognitiveComponent]:
        """Identify components that are bottlenecks"""
        bottlenecks = []
        avg_performance = self.get_overall_performance()

        for component, state in self.components.items():
            if state.performance < avg_performance * 0.8 or state.load > 0.8:
                bottlenecks.append(component)

        return bottlenecks

    def calibrate(self,
                  actual_states: Dict[CognitiveComponent, Dict[str, float]],
                  actual_performance: float):
        """Calibrate the self-model using actual measurements"""
        # Update components with actual values
        for component, values in actual_states.items():
            self.update_component(
                component,
                health=values.get("health"),
                load=values.get("load"),
                performance=values.get("performance")
            )

        # Store calibration sample
        current_vector = self._state_vector.copy()
        self._calibration_samples.append((current_vector, np.array([actual_performance])))

        # Update model confidence based on calibration
        if len(self._calibration_samples) > 10:
            predictions = [self.predict_performance("general").predicted_value for _ in range(5)]
            avg_prediction = np.mean(predictions)
            error = abs(avg_prediction - actual_performance)
            accuracy = 1 - error
            self.prediction_accuracy_history.append(accuracy)

            # Update model confidence
            self.model_confidence = np.mean(self.prediction_accuracy_history[-20:])

        # Update model completeness
        covered_components = len([c for c in self.components.values() if c.history])
        self.model_completeness = covered_components / len(CognitiveComponent)

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current self-model state"""
        return {
            "model_id": self.model_id,
            "model_confidence": self.model_confidence,
            "model_completeness": self.model_completeness,
            "model_accuracy": self.assess_accuracy().value,
            "overall_health": self.get_overall_health(),
            "overall_performance": self.get_overall_performance(),
            "bottlenecks": [c.value for c in self.get_bottlenecks()],
            "component_count": len(self.components),
            "capability_count": len(self.capabilities),
            "limitation_count": len(self.limitations),
            "prediction_count": len(self.predictions),
            "last_update": self.last_update.isoformat()
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed self-model report"""
        return {
            "summary": self.get_state_summary(),
            "components": {
                c.value: {
                    "active": self.components[c].active,
                    "health": self.components[c].health,
                    "load": self.components[c].load,
                    "performance": self.components[c].performance
                }
                for c in CognitiveComponent if c in self.components
            },
            "capabilities": {
                name: cap.to_dict()
                for name, cap in self.capabilities.items()
            },
            "limitations": {
                name: {
                    "severity": lim.severity,
                    "scope": [s.value for s in lim.scope],
                    "workarounds": lim.workarounds,
                    "is_fundamental": lim.is_fundamental
                }
                for name, lim in self.limitations.items()
            },
            "recent_predictions": [
                {
                    "target": p.target,
                    "predicted": p.predicted_value,
                    "confidence": p.confidence,
                    "actual": p.actual_value,
                    "error": p.prediction_error
                }
                for p in self.predictions[-10:]
            ]
        }

    def get_state_hash(self) -> str:
        """Get a hash of the current model state"""
        state_str = json.dumps({
            "components": {
                c.value: s.to_vector().tolist()
                for c, s in self.components.items()
            },
            "capabilities": list(self.capabilities.keys()),
            "limitations": list(self.limitations.keys())
        }, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def to_vector(self) -> np.ndarray:
        """Get the full state vector representation"""
        self._update_state_vector()
        return self._state_vector.copy()

    def export(self, filepath: str):
        """Export self-model to file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_detailed_report(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'SelfModel':
        """Load self-model from file"""
        model = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore components
        for comp_name, comp_data in data.get("components", {}).items():
            component = CognitiveComponent(comp_name)
            model.update_component(
                component,
                health=comp_data.get("health"),
                load=comp_data.get("load"),
                performance=comp_data.get("performance")
            )

        # Restore capabilities
        for name, cap_data in data.get("capabilities", {}).items():
            model.register_capability(
                name=name,
                level=cap_data.get("level", 0.5),
                reliability=cap_data.get("reliability", 0.5),
                dependencies=cap_data.get("dependencies"),
                limitations=cap_data.get("limitations")
            )

        return model


class SelfDiagnosis:
    """
    Self-diagnosis system that uses the self-model to identify issues.
    """

    def __init__(self, self_model: SelfModel):
        self.self_model = self_model
        self.diagnosis_history: List[Dict[str, Any]] = []

    def diagnose(self) -> Dict[str, Any]:
        """Run a comprehensive self-diagnosis"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "recommendations": []
        }

        # Check component health
        for component, state in self.self_model.components.items():
            if state.health < 0.5:
                diagnosis["issues"].append({
                    "type": "low_health",
                    "component": component.value,
                    "severity": "high" if state.health < 0.3 else "medium",
                    "value": state.health
                })
            elif state.health < 0.7:
                diagnosis["warnings"].append({
                    "type": "degraded_health",
                    "component": component.value,
                    "value": state.health
                })

            # Check load
            if state.load > 0.9:
                diagnosis["issues"].append({
                    "type": "overload",
                    "component": component.value,
                    "severity": "high",
                    "value": state.load
                })
            elif state.load > 0.7:
                diagnosis["warnings"].append({
                    "type": "high_load",
                    "component": component.value,
                    "value": state.load
                })

        # Check model confidence
        if self.self_model.model_confidence < 0.5:
            diagnosis["warnings"].append({
                "type": "low_model_confidence",
                "value": self.self_model.model_confidence
            })
            diagnosis["recommendations"].append(
                "Consider running calibration to improve self-model accuracy"
            )

        # Check for bottlenecks
        bottlenecks = self.self_model.get_bottlenecks()
        if bottlenecks:
            diagnosis["warnings"].append({
                "type": "performance_bottlenecks",
                "components": [b.value for b in bottlenecks]
            })
            diagnosis["recommendations"].append(
                f"Address bottlenecks in: {', '.join(b.value for b in bottlenecks)}"
            )

        # Check prediction accuracy
        accuracy = self.self_model.assess_accuracy()
        if accuracy in [ModelAccuracy.UNCERTAIN, ModelAccuracy.LOW]:
            diagnosis["warnings"].append({
                "type": "low_prediction_accuracy",
                "level": accuracy.value
            })

        # Store diagnosis
        self.diagnosis_history.append(diagnosis)

        return diagnosis

    def get_health_trend(self, component: CognitiveComponent, window: int = 10) -> str:
        """Analyze health trend for a component"""
        state = self.self_model.components.get(component)
        if not state or len(state.history) < 2:
            return "unknown"

        recent = [h["health"] for h in state.history[-window:]]
        if len(recent) < 2:
            return "stable"

        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        if trend > 0.05:
            return "improving"
        elif trend < -0.05:
            return "declining"
        else:
            return "stable"
