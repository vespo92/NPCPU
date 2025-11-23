"""
Planetary Tipping Point Detection

Implements detection and modeling of planetary-scale tipping points
that can trigger rapid, irreversible changes in Earth systems.

Features:
- Early warning signal detection
- Multiple tipping element tracking
- Cascade effects modeling
- Recovery trajectory analysis
- Consciousness-aware tipping point management
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class TippingElementType(Enum):
    """Types of planetary tipping elements"""
    AMAZON_RAINFOREST = "amazon_rainforest"
    ARCTIC_SEA_ICE = "arctic_sea_ice"
    GREENLAND_ICE_SHEET = "greenland_ice_sheet"
    WEST_ANTARCTIC_ICE = "west_antarctic_ice"
    ATLANTIC_CIRCULATION = "atlantic_circulation"
    CORAL_REEFS = "coral_reefs"
    PERMAFROST = "permafrost"
    BOREAL_FORESTS = "boreal_forests"
    MONSOON_SYSTEMS = "monsoon_systems"
    CONSCIOUSNESS_NETWORK = "consciousness_network"


class TippingState(Enum):
    """State relative to tipping point"""
    STABLE = "stable"                  # Far from threshold
    APPROACHING = "approaching"        # Moving toward threshold
    CRITICAL = "critical"              # Near threshold
    TIPPING = "tipping"                # Crossing threshold
    TIPPED = "tipped"                  # Past point of no return
    RECOVERING = "recovering"          # Moving back toward stable


class WarningSignal(Enum):
    """Early warning signals for tipping points"""
    CRITICAL_SLOWING = "critical_slowing"      # Recovery time increasing
    INCREASED_VARIANCE = "increased_variance"   # State fluctuations
    INCREASED_AUTOCORRELATION = "increased_autocorrelation"
    FLICKERING = "flickering"                   # Rapid alternation
    SPATIAL_PATTERN = "spatial_pattern"         # Pattern changes


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TippingElement:
    """A planetary tipping element"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    element_type: TippingElementType = TippingElementType.AMAZON_RAINFOREST

    # State
    current_value: float = 0.0          # Current state (0-1 normalized)
    threshold: float = 0.5              # Tipping threshold
    state: TippingState = TippingState.STABLE
    recovery_rate: float = 0.1          # Natural recovery speed

    # Dynamics
    inertia: float = 0.9                # Resistance to change
    hysteresis: float = 0.2             # Difference between tip and recovery thresholds
    cascade_risk: float = 0.3           # Risk of triggering cascades

    # Monitoring
    value_history: List[float] = field(default_factory=list)
    warning_signals: List[WarningSignal] = field(default_factory=list)
    time_to_threshold: Optional[float] = None

    def get_distance_to_threshold(self) -> float:
        """Get distance from threshold (negative = past threshold)"""
        return self.threshold - self.current_value


@dataclass
class CascadeEvent:
    """A cascade of tipping events"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_element: str = ""
    affected_elements: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed: bool = False
    total_impact: float = 0.0


@dataclass
class TippingPointConfig:
    """Configuration for tipping point detection"""
    detection_window: int = 50          # Window for signal detection
    critical_distance: float = 0.1      # Distance considered critical
    warning_threshold: float = 0.3      # Threshold for warnings
    cascade_probability: float = 0.3    # Base cascade probability
    enable_consciousness_stabilization: bool = True


# ============================================================================
# Tipping Point Detector
# ============================================================================

class TippingPointDetector:
    """
    Detects and tracks planetary tipping points.

    Monitors multiple tipping elements for early warning signals
    and cascade risks. Can integrate with consciousness for
    coordinated response.

    Example:
        detector = TippingPointDetector()
        detector.initialize_elements()

        for _ in range(100):
            detector.update_element("amazon", 0.6)
            detector.analyze()

        warnings = detector.get_active_warnings()
    """

    def __init__(self, config: Optional[TippingPointConfig] = None):
        self.config = config or TippingPointConfig()

        # Tipping elements
        self.elements: Dict[str, TippingElement] = {}

        # Cascade tracking
        self.active_cascades: List[CascadeEvent] = []
        self.cascade_history: List[CascadeEvent] = []

        # Analysis
        self.global_stability: float = 1.0
        self.cascade_risk: float = 0.0
        self.active_warnings: Dict[str, List[WarningSignal]] = {}

        # History
        self.tick_count = 0
        self.tipping_events: List[Dict[str, Any]] = []

    def initialize_elements(self):
        """Initialize default tipping elements"""
        elements = [
            TippingElement(
                name="Amazon Rainforest",
                element_type=TippingElementType.AMAZON_RAINFOREST,
                current_value=0.2,
                threshold=0.6,
                recovery_rate=0.05,
                cascade_risk=0.4
            ),
            TippingElement(
                name="Arctic Sea Ice",
                element_type=TippingElementType.ARCTIC_SEA_ICE,
                current_value=0.3,
                threshold=0.7,
                recovery_rate=0.02,
                cascade_risk=0.5
            ),
            TippingElement(
                name="Greenland Ice Sheet",
                element_type=TippingElementType.GREENLAND_ICE_SHEET,
                current_value=0.1,
                threshold=0.5,
                recovery_rate=0.01,
                cascade_risk=0.6
            ),
            TippingElement(
                name="Atlantic Circulation",
                element_type=TippingElementType.ATLANTIC_CIRCULATION,
                current_value=0.15,
                threshold=0.55,
                recovery_rate=0.03,
                cascade_risk=0.7
            ),
            TippingElement(
                name="Coral Reefs",
                element_type=TippingElementType.CORAL_REEFS,
                current_value=0.4,
                threshold=0.65,
                recovery_rate=0.08,
                cascade_risk=0.2
            ),
            TippingElement(
                name="Permafrost",
                element_type=TippingElementType.PERMAFROST,
                current_value=0.25,
                threshold=0.6,
                recovery_rate=0.02,
                cascade_risk=0.5
            ),
            TippingElement(
                name="Consciousness Network",
                element_type=TippingElementType.CONSCIOUSNESS_NETWORK,
                current_value=0.1,
                threshold=0.4,  # Threshold for collapse
                recovery_rate=0.1,
                cascade_risk=0.3
            )
        ]

        for element in elements:
            self.elements[element.id] = element

    def update_element(self, element_id: str, new_value: float):
        """Update a tipping element's value"""
        element = self.elements.get(element_id)
        if not element:
            return

        # Record history
        element.value_history.append(element.current_value)
        if len(element.value_history) > self.config.detection_window * 2:
            element.value_history = element.value_history[-self.config.detection_window * 2:]

        # Update value with inertia
        delta = new_value - element.current_value
        element.current_value += delta * (1 - element.inertia)

        # Update state
        self._update_element_state(element)

    def update_element_by_name(self, name: str, new_value: float):
        """Update element by name"""
        for element in self.elements.values():
            if element.name.lower() == name.lower():
                self.update_element(element.id, new_value)
                return

    def analyze(self):
        """Analyze all tipping elements"""
        self.tick_count += 1
        self.active_warnings.clear()

        for element in self.elements.values():
            # Detect warning signals
            signals = self._detect_warning_signals(element)
            element.warning_signals = signals

            if signals:
                self.active_warnings[element.id] = signals

            # Estimate time to threshold
            element.time_to_threshold = self._estimate_time_to_threshold(element)

        # Calculate global metrics
        self._calculate_global_stability()
        self._calculate_cascade_risk()

        # Process cascades
        self._process_cascades()

    def _update_element_state(self, element: TippingElement):
        """Update element state based on value"""
        distance = element.get_distance_to_threshold()
        old_state = element.state

        if element.state == TippingState.TIPPED:
            # Check for recovery
            recovery_threshold = element.threshold + element.hysteresis
            if element.current_value < recovery_threshold:
                element.state = TippingState.RECOVERING
        elif element.state == TippingState.RECOVERING:
            if element.current_value < element.threshold - 0.1:
                element.state = TippingState.STABLE
        else:
            if distance <= 0:
                element.state = TippingState.TIPPING
            elif distance < self.config.critical_distance:
                element.state = TippingState.CRITICAL
            elif distance < self.config.warning_threshold:
                element.state = TippingState.APPROACHING
            else:
                element.state = TippingState.STABLE

        # Check for tipping
        if element.state == TippingState.TIPPING and old_state != TippingState.TIPPING:
            self._on_element_tipped(element)

    def _detect_warning_signals(self, element: TippingElement) -> List[WarningSignal]:
        """Detect early warning signals"""
        signals = []

        if len(element.value_history) < self.config.detection_window:
            return signals

        recent = element.value_history[-self.config.detection_window:]

        # Critical slowing down - increasing autocorrelation
        autocorr = self._calculate_autocorrelation(recent)
        if autocorr > 0.8:
            signals.append(WarningSignal.INCREASED_AUTOCORRELATION)
            signals.append(WarningSignal.CRITICAL_SLOWING)

        # Increased variance
        variance = np.var(recent)
        if len(element.value_history) >= self.config.detection_window * 2:
            old = element.value_history[-self.config.detection_window * 2:-self.config.detection_window]
            old_variance = np.var(old)
            if variance > old_variance * 1.5:
                signals.append(WarningSignal.INCREASED_VARIANCE)

        # Flickering
        sign_changes = sum(
            1 for i in range(1, len(recent))
            if (recent[i] - recent[i - 1]) * (recent[i - 1] - recent[i - 2] if i > 1 else 1) < 0
        )
        if sign_changes > len(recent) * 0.4:
            signals.append(WarningSignal.FLICKERING)

        return signals

    def _calculate_autocorrelation(self, values: List[float]) -> float:
        """Calculate lag-1 autocorrelation"""
        if len(values) < 3:
            return 0.0

        values = np.array(values)
        mean = np.mean(values)
        variance = np.var(values)

        if variance == 0:
            return 0.0

        autocov = np.mean((values[:-1] - mean) * (values[1:] - mean))
        return autocov / variance

    def _estimate_time_to_threshold(self, element: TippingElement) -> Optional[float]:
        """Estimate time until threshold is reached"""
        if len(element.value_history) < 5:
            return None

        # Calculate trend
        recent = element.value_history[-10:]
        if len(recent) < 2:
            return None

        trend = (recent[-1] - recent[0]) / len(recent)

        if trend <= 0:
            return float('inf')  # Moving away from threshold

        distance = element.get_distance_to_threshold()
        if distance <= 0:
            return 0  # Already past threshold

        return distance / trend

    def _calculate_global_stability(self):
        """Calculate overall system stability"""
        if not self.elements:
            self.global_stability = 1.0
            return

        stabilities = []
        for element in self.elements.values():
            distance = element.get_distance_to_threshold()
            # Normalize to 0-1 stability score
            stability = max(0, min(1, distance / element.threshold))
            stabilities.append(stability)

        self.global_stability = np.mean(stabilities)

    def _calculate_cascade_risk(self):
        """Calculate risk of cascade events"""
        critical_elements = [
            e for e in self.elements.values()
            if e.state in [TippingState.CRITICAL, TippingState.TIPPING]
        ]

        if not critical_elements:
            self.cascade_risk = 0.0
            return

        # Risk increases with number and severity of critical elements
        risk_sum = sum(e.cascade_risk for e in critical_elements)
        max_risk = sum(e.cascade_risk for e in self.elements.values())

        self.cascade_risk = risk_sum / max_risk if max_risk > 0 else 0.0

    def _on_element_tipped(self, element: TippingElement):
        """Handle element tipping"""
        element.state = TippingState.TIPPED

        self.tipping_events.append({
            "tick": self.tick_count,
            "element": element.name,
            "type": element.element_type.value,
            "value": element.current_value,
            "threshold": element.threshold
        })

        # Check for cascade
        if np.random.random() < element.cascade_risk:
            self._trigger_cascade(element)

    def _trigger_cascade(self, trigger: TippingElement):
        """Trigger a cascade from a tipped element"""
        cascade = CascadeEvent(
            trigger_element=trigger.id,
            affected_elements=[],
            total_impact=trigger.cascade_risk
        )

        # Find elements that could be affected
        for element in self.elements.values():
            if element.id == trigger.id:
                continue

            if element.state in [TippingState.APPROACHING, TippingState.CRITICAL]:
                # Probability of being affected
                if np.random.random() < self.config.cascade_probability * trigger.cascade_risk:
                    cascade.affected_elements.append(element.id)
                    # Push element closer to threshold
                    push = trigger.cascade_risk * 0.2
                    element.current_value += push
                    cascade.total_impact += push

        if cascade.affected_elements:
            self.active_cascades.append(cascade)

    def _process_cascades(self):
        """Process active cascades"""
        completed = []

        for cascade in self.active_cascades:
            # Check if cascade has finished propagating
            all_processed = all(
                self.elements[eid].state in [TippingState.TIPPED, TippingState.STABLE]
                for eid in cascade.affected_elements
                if eid in self.elements
            )

            if all_processed:
                cascade.completed = True
                completed.append(cascade)

        for cascade in completed:
            self.active_cascades.remove(cascade)
            self.cascade_history.append(cascade)

    # ========================================================================
    # Public API
    # ========================================================================

    def get_element_status(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific element"""
        element = self.elements.get(element_id)
        if not element:
            return None

        return {
            "name": element.name,
            "type": element.element_type.value,
            "value": element.current_value,
            "threshold": element.threshold,
            "distance": element.get_distance_to_threshold(),
            "state": element.state.value,
            "warnings": [w.value for w in element.warning_signals],
            "time_to_threshold": element.time_to_threshold,
            "cascade_risk": element.cascade_risk
        }

    def get_all_elements_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all elements"""
        return [
            {
                "id": e.id,
                "name": e.name,
                "state": e.state.value,
                "distance": e.get_distance_to_threshold(),
                "warnings": len(e.warning_signals)
            }
            for e in self.elements.values()
        ]

    def get_active_warnings(self) -> Dict[str, List[str]]:
        """Get all active warnings"""
        return {
            eid: [w.value for w in warnings]
            for eid, warnings in self.active_warnings.items()
        }

    def get_global_status(self) -> Dict[str, Any]:
        """Get global tipping point status"""
        return {
            "tick_count": self.tick_count,
            "global_stability": self.global_stability,
            "cascade_risk": self.cascade_risk,
            "elements_count": len(self.elements),
            "critical_elements": sum(
                1 for e in self.elements.values()
                if e.state in [TippingState.CRITICAL, TippingState.TIPPING]
            ),
            "tipped_elements": sum(
                1 for e in self.elements.values()
                if e.state == TippingState.TIPPED
            ),
            "active_warnings": sum(len(w) for w in self.active_warnings.values()),
            "active_cascades": len(self.active_cascades),
            "total_tipping_events": len(self.tipping_events)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Tipping Point Detection Demo")
    print("=" * 50)

    detector = TippingPointDetector()
    detector.initialize_elements()

    print(f"\n1. Initialized {len(detector.elements)} tipping elements")

    # List elements
    print("\n2. Tipping elements:")
    for element in detector.elements.values():
        print(f"   - {element.name}: threshold={element.threshold:.2f}, "
              f"current={element.current_value:.2f}")

    # Simulate stress on elements
    print("\n3. Simulating increasing stress...")
    for i in range(100):
        # Gradually stress the Amazon
        for element in detector.elements.values():
            if "Amazon" in element.name:
                detector.update_element(element.id, element.current_value + 0.005)
            elif "Coral" in element.name:
                detector.update_element(element.id, element.current_value + 0.003)

        detector.analyze()

        if i % 25 == 0:
            status = detector.get_global_status()
            print(f"   Tick {i}: stability={status['global_stability']:.2f}, "
                  f"warnings={status['active_warnings']}, "
                  f"cascade_risk={status['cascade_risk']:.2f}")

    # Final status
    print("\n4. Final element status:")
    for element in detector.elements.values():
        status = detector.get_element_status(element.id)
        if status:
            print(f"   {status['name']}: {status['state']}, "
                  f"distance={status['distance']:.2f}")

    print("\n5. Active warnings:")
    warnings = detector.get_active_warnings()
    for eid, warns in warnings.items():
        element = detector.elements.get(eid)
        if element:
            print(f"   {element.name}: {warns}")
