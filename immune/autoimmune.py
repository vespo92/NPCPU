"""
Autoimmune Prevention System

Prevents the immune system from attacking the organism's own components.
Implements self-tolerance mechanisms:
- Self-antigen recognition
- Regulatory T-cell suppression
- Anergy induction
- Peripheral tolerance
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from immune.defense import Threat, ThreatType, ThreatSeverity


# ============================================================================
# Enums
# ============================================================================

class SelfMarkerType(Enum):
    """Types of self-markers"""
    CORE_COMPONENT = "core_component"
    SUBSYSTEM = "subsystem"
    MEMORY = "memory"
    PROCESS = "process"
    DATA_STRUCTURE = "data_structure"
    CONFIG = "config"


class ToleranceType(Enum):
    """Types of immune tolerance"""
    CENTRAL = "central"         # Developed during creation
    PERIPHERAL = "peripheral"   # Developed after creation
    INDUCED = "induced"         # Actively induced
    ACQUIRED = "acquired"       # Learned over time


class AutoimmuneRisk(Enum):
    """Risk levels for autoimmune reaction"""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class SuppressionMethod(Enum):
    """Methods for suppressing autoimmune response"""
    ANERGY = "anergy"           # Make self-reactive cells unresponsive
    DELETION = "deletion"       # Remove self-reactive cells
    SUPPRESSION = "suppression" # Active suppression by regulatory cells
    IGNORANCE = "ignorance"     # Ignore low-affinity self-reactions


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SelfMarker:
    """Marker identifying self-components"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    marker_type: SelfMarkerType = SelfMarkerType.CORE_COMPONENT
    signature: str = ""
    patterns: List[str] = field(default_factory=list)
    criticality: float = 0.5  # How critical is this component (0-1)
    registered_at: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)


@dataclass
class ToleranceRecord:
    """Record of established tolerance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    self_marker_id: str = ""
    tolerance_type: ToleranceType = ToleranceType.PERIPHERAL
    strength: float = 1.0
    established_at: float = field(default_factory=time.time)
    violations: int = 0


@dataclass
class AutoimmuneEvent:
    """Record of an autoimmune event"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    self_marker_id: str = ""
    threat_id: str = ""  # The threat that triggered false positive
    risk_level: AutoimmuneRisk = AutoimmuneRisk.LOW
    suppression_method: Optional[SuppressionMethod] = None
    suppressed: bool = False
    damage_prevented: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RegulatoryCell:
    """Regulatory T-cell for suppression"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_markers: Set[str] = field(default_factory=set)
    suppression_strength: float = 0.8
    active: bool = True
    suppressions_performed: int = 0
    created_at: float = field(default_factory=time.time)


# ============================================================================
# Autoimmune Prevention System
# ============================================================================

class AutoimmunePreventionSystem:
    """
    System to prevent autoimmune attacks on self.

    Features:
    - Self-marker registration
    - Tolerance establishment
    - Regulatory cell management
    - Autoimmune event detection and suppression

    Example:
        auto = AutoimmunePreventionSystem()

        # Register self components
        auto.register_self_marker("core_memory", SelfMarkerType.MEMORY,
                                  signature="mem_", patterns=["memory_", "store_"])

        # Check if threat targets self
        if auto.is_self_target(threat):
            event = auto.detect_autoimmune_event(threat)
            auto.suppress_autoimmune(event)
    """

    def __init__(
        self,
        tolerance_threshold: float = 0.7,
        regulatory_cell_count: int = 10,
        self_check_sensitivity: float = 0.8
    ):
        self.tolerance_threshold = tolerance_threshold
        self.self_check_sensitivity = self_check_sensitivity

        # Self markers
        self.self_markers: Dict[str, SelfMarker] = {}
        self.marker_index: Dict[str, List[str]] = {}  # pattern -> marker_ids

        # Tolerance records
        self.tolerance_records: Dict[str, ToleranceRecord] = {}

        # Regulatory cells
        self.regulatory_cells: Dict[str, RegulatoryCell] = {}

        # Event history
        self.autoimmune_events: List[AutoimmuneEvent] = []
        self.prevented_count = 0
        self.missed_count = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "self_marker_registered": [],
            "autoimmune_detected": [],
            "autoimmune_suppressed": [],
            "tolerance_violation": []
        }

        # Initialize regulatory cells
        self._initialize_regulatory_cells(regulatory_cell_count)

    def _initialize_regulatory_cells(self, count: int):
        """Initialize regulatory T-cells"""
        for i in range(count):
            reg = RegulatoryCell(
                suppression_strength=0.7 + np.random.random() * 0.3
            )
            self.regulatory_cells[reg.id] = reg

    def register_self_marker(
        self,
        name: str,
        marker_type: SelfMarkerType,
        signature: str = "",
        patterns: Optional[List[str]] = None,
        criticality: float = 0.5
    ) -> SelfMarker:
        """Register a self-component marker"""
        marker = SelfMarker(
            name=name,
            marker_type=marker_type,
            signature=signature,
            patterns=patterns or [],
            criticality=criticality
        )

        self.self_markers[marker.id] = marker

        # Index patterns for fast lookup
        for pattern in marker.patterns:
            if pattern not in self.marker_index:
                self.marker_index[pattern] = []
            self.marker_index[pattern].append(marker.id)

        # Establish central tolerance
        self._establish_tolerance(marker, ToleranceType.CENTRAL)

        for callback in self._callbacks["self_marker_registered"]:
            callback(marker)

        return marker

    def _establish_tolerance(
        self,
        marker: SelfMarker,
        tolerance_type: ToleranceType
    ) -> ToleranceRecord:
        """Establish tolerance for a self-marker"""
        record = ToleranceRecord(
            self_marker_id=marker.id,
            tolerance_type=tolerance_type,
            strength=1.0 if tolerance_type == ToleranceType.CENTRAL else 0.8
        )

        self.tolerance_records[record.id] = record

        # Assign regulatory cells
        available_regs = [r for r in self.regulatory_cells.values()
                         if len(r.target_markers) < 5]
        if available_regs:
            reg = available_regs[0]
            reg.target_markers.add(marker.id)

        return record

    def is_self_target(self, threat: Threat) -> bool:
        """Check if a threat targets self-components"""
        # Check threat signature against self-markers
        threat_sig = str(threat.signature).lower()
        threat_source = str(threat.source).lower()
        threat_details = str(threat.details).lower()

        combined = f"{threat_sig} {threat_source} {threat_details}"

        for marker in self.self_markers.values():
            # Check signature match
            if marker.signature and marker.signature.lower() in combined:
                return True

            # Check pattern matches
            for pattern in marker.patterns:
                if pattern.lower() in combined:
                    return True

        return False

    def get_matching_self_markers(
        self,
        threat: Threat
    ) -> List[SelfMarker]:
        """Get self-markers that match the threat target"""
        matches = []
        threat_sig = str(threat.signature).lower()
        threat_details = str(threat.details).lower()
        combined = f"{threat_sig} {threat_details}"

        for marker in self.self_markers.values():
            score = 0.0

            if marker.signature and marker.signature.lower() in combined:
                score += 0.5

            for pattern in marker.patterns:
                if pattern.lower() in combined:
                    score += 0.3

            if score >= self.self_check_sensitivity * 0.5:
                matches.append(marker)

        return matches

    def detect_autoimmune_event(
        self,
        threat: Threat,
        auto_suppress: bool = True
    ) -> Optional[AutoimmuneEvent]:
        """Detect potential autoimmune event"""
        matching_markers = self.get_matching_self_markers(threat)

        if not matching_markers:
            return None

        # Get most critical matching marker
        critical_marker = max(matching_markers, key=lambda m: m.criticality)

        # Calculate risk level
        risk = self._calculate_autoimmune_risk(critical_marker, threat)

        event = AutoimmuneEvent(
            self_marker_id=critical_marker.id,
            threat_id=threat.id,
            risk_level=risk,
            damage_prevented=critical_marker.criticality * 100
        )

        self.autoimmune_events.append(event)

        for callback in self._callbacks["autoimmune_detected"]:
            callback(event)

        # Auto-suppress if enabled
        if auto_suppress:
            self.suppress_autoimmune(event)

        return event

    def _calculate_autoimmune_risk(
        self,
        marker: SelfMarker,
        threat: Threat
    ) -> AutoimmuneRisk:
        """Calculate autoimmune risk level"""
        severity_weight = {
            ThreatSeverity.LOW: 0.25,
            ThreatSeverity.MEDIUM: 0.5,
            ThreatSeverity.HIGH: 0.75,
            ThreatSeverity.CRITICAL: 1.0
        }.get(threat.severity, 0.5)

        criticality_weight = marker.criticality
        combined_risk = (severity_weight + criticality_weight) / 2

        if combined_risk < 0.2:
            return AutoimmuneRisk.NONE
        elif combined_risk < 0.4:
            return AutoimmuneRisk.LOW
        elif combined_risk < 0.6:
            return AutoimmuneRisk.MODERATE
        elif combined_risk < 0.8:
            return AutoimmuneRisk.HIGH
        else:
            return AutoimmuneRisk.CRITICAL

    def suppress_autoimmune(
        self,
        event: AutoimmuneEvent
    ) -> bool:
        """Suppress an autoimmune event"""
        if event.suppressed:
            return True

        marker = self.self_markers.get(event.self_marker_id)
        if not marker:
            return False

        # Select suppression method based on risk
        method = self._select_suppression_method(event)
        event.suppression_method = method

        # Find appropriate regulatory cell
        reg_cell = self._find_regulatory_cell(event.self_marker_id)

        success = False
        if reg_cell:
            success = self._apply_suppression(method, event, reg_cell)

        if success:
            event.suppressed = True
            self.prevented_count += 1

            # Update tolerance record
            for record in self.tolerance_records.values():
                if record.self_marker_id == event.self_marker_id:
                    record.violations += 1
                    # Strengthen tolerance after violation
                    record.strength = min(1.0, record.strength * 1.05)

            for callback in self._callbacks["autoimmune_suppressed"]:
                callback(event)
        else:
            self.missed_count += 1

            for callback in self._callbacks["tolerance_violation"]:
                callback(event)

        return success

    def _select_suppression_method(
        self,
        event: AutoimmuneEvent
    ) -> SuppressionMethod:
        """Select appropriate suppression method"""
        if event.risk_level == AutoimmuneRisk.CRITICAL:
            return SuppressionMethod.DELETION
        elif event.risk_level == AutoimmuneRisk.HIGH:
            return SuppressionMethod.SUPPRESSION
        elif event.risk_level == AutoimmuneRisk.MODERATE:
            return SuppressionMethod.ANERGY
        else:
            return SuppressionMethod.IGNORANCE

    def _find_regulatory_cell(
        self,
        marker_id: str
    ) -> Optional[RegulatoryCell]:
        """Find regulatory cell for marker"""
        # First try to find one assigned to this marker
        for reg in self.regulatory_cells.values():
            if marker_id in reg.target_markers and reg.active:
                return reg

        # Fall back to any active regulatory cell
        for reg in self.regulatory_cells.values():
            if reg.active:
                return reg

        return None

    def _apply_suppression(
        self,
        method: SuppressionMethod,
        event: AutoimmuneEvent,
        reg_cell: RegulatoryCell
    ) -> bool:
        """Apply suppression method"""
        reg_cell.suppressions_performed += 1

        base_success = reg_cell.suppression_strength

        method_modifiers = {
            SuppressionMethod.IGNORANCE: 0.95,
            SuppressionMethod.ANERGY: 0.85,
            SuppressionMethod.SUPPRESSION: 0.8,
            SuppressionMethod.DELETION: 0.7
        }

        success_chance = base_success * method_modifiers.get(method, 0.8)
        return np.random.random() < success_chance

    def verify_self_markers(self) -> List[SelfMarker]:
        """Verify all self-markers are still valid"""
        invalid = []
        current_time = time.time()

        for marker in self.self_markers.values():
            # Check if marker is still valid
            age = current_time - marker.registered_at

            # Very old markers without recent verification
            if age > 86400 and current_time - marker.last_verified > 3600:
                invalid.append(marker)
            else:
                marker.last_verified = current_time

        return invalid

    def add_induced_tolerance(
        self,
        component_name: str,
        patterns: List[str]
    ) -> SelfMarker:
        """Add induced tolerance for a new component"""
        marker = self.register_self_marker(
            name=component_name,
            marker_type=SelfMarkerType.SUBSYSTEM,
            patterns=patterns,
            criticality=0.3
        )

        # Override tolerance type
        for record in self.tolerance_records.values():
            if record.self_marker_id == marker.id:
                record.tolerance_type = ToleranceType.INDUCED
                record.strength = 0.7

        return marker

    def remove_tolerance(self, marker_id: str) -> bool:
        """Remove tolerance for a marker (e.g., for removed component)"""
        if marker_id not in self.self_markers:
            return False

        marker = self.self_markers[marker_id]

        # Remove from indexes
        for pattern in marker.patterns:
            if pattern in self.marker_index:
                self.marker_index[pattern] = [
                    m for m in self.marker_index[pattern] if m != marker_id
                ]

        # Remove tolerance records
        to_remove = [
            r_id for r_id, r in self.tolerance_records.items()
            if r.self_marker_id == marker_id
        ]
        for r_id in to_remove:
            del self.tolerance_records[r_id]

        # Update regulatory cells
        for reg in self.regulatory_cells.values():
            reg.target_markers.discard(marker_id)

        del self.self_markers[marker_id]
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get autoimmune prevention statistics"""
        return {
            "self_markers": len(self.self_markers),
            "tolerance_records": len(self.tolerance_records),
            "regulatory_cells": len(self.regulatory_cells),
            "active_regulatory_cells": len([r for r in self.regulatory_cells.values() if r.active]),
            "total_events": len(self.autoimmune_events),
            "prevented": self.prevented_count,
            "missed": self.missed_count,
            "prevention_rate": self.prevented_count / max(1, self.prevented_count + self.missed_count),
            "markers_by_type": {
                mt.value: len([m for m in self.self_markers.values() if m.marker_type == mt])
                for mt in SelfMarkerType
            },
            "events_by_risk": {
                risk.name: len([e for e in self.autoimmune_events if e.risk_level == risk])
                for risk in AutoimmuneRisk
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Autoimmune Prevention System Demo")
    print("=" * 50)

    # Create system
    auto_system = AutoimmunePreventionSystem()

    print("\n1. Registering self markers...")

    # Register core components
    auto_system.register_self_marker(
        "core_memory",
        SelfMarkerType.MEMORY,
        signature="memory_store",
        patterns=["memory_", "store_", "recall_"],
        criticality=0.9
    )

    auto_system.register_self_marker(
        "consciousness",
        SelfMarkerType.CORE_COMPONENT,
        signature="consciousness_",
        patterns=["conscious_", "aware_", "perceive_"],
        criticality=1.0
    )

    auto_system.register_self_marker(
        "config_system",
        SelfMarkerType.CONFIG,
        signature="config_",
        patterns=["config_", "settings_"],
        criticality=0.5
    )

    stats = auto_system.get_statistics()
    print(f"   Registered {stats['self_markers']} self markers")
    print(f"   Created {stats['tolerance_records']} tolerance records")

    print("\n2. Testing self-target detection...")

    # Create a threat that targets self
    self_threat = Threat(
        type=ThreatType.CORRUPTION,
        severity=ThreatSeverity.HIGH,
        signature="memory_store_corruption",
        details={"target": "memory_recall_function"}
    )

    is_self = auto_system.is_self_target(self_threat)
    print(f"   Self-targeting threat detected: {is_self}")

    # Create a threat that doesn't target self
    external_threat = Threat(
        type=ThreatType.MALICIOUS_INPUT,
        severity=ThreatSeverity.MEDIUM,
        signature="external_attack",
        details={"source": "network"}
    )

    is_self = auto_system.is_self_target(external_threat)
    print(f"   External threat detected as self: {is_self}")

    print("\n3. Testing autoimmune detection and suppression...")

    event = auto_system.detect_autoimmune_event(self_threat)
    if event:
        print(f"   Autoimmune event detected!")
        print(f"     Risk level: {event.risk_level.name}")
        print(f"     Suppression method: {event.suppression_method.value if event.suppression_method else 'None'}")
        print(f"     Suppressed: {event.suppressed}")
        print(f"     Damage prevented: {event.damage_prevented:.0f}")

    print("\n4. Adding induced tolerance...")
    new_marker = auto_system.add_induced_tolerance(
        "new_plugin",
        patterns=["plugin_", "extension_"]
    )
    print(f"   Added tolerance for: {new_marker.name}")

    print("\n5. Final statistics:")
    stats = auto_system.get_statistics()
    print(f"   Total events: {stats['total_events']}")
    print(f"   Prevention rate: {stats['prevention_rate']:.2%}")
    print(f"   Markers by type:")
    for mtype, count in stats['markers_by_type'].items():
        if count > 0:
            print(f"     {mtype}: {count}")
