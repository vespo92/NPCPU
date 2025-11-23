"""
Inflammation System

System-wide alert propagation mechanism that coordinates response
to threats across all immune components. Implements:
- Cytokine signaling
- Inflammatory cascades
- Resolution mechanisms
- Chronic inflammation prevention
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from immune.defense import Threat, ThreatType, ThreatSeverity


# ============================================================================
# Enums
# ============================================================================

class InflammationLevel(Enum):
    """Levels of inflammation"""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4
    SYSTEMIC = 5  # Full system inflammation


class CytokineType(Enum):
    """Types of signaling cytokines"""
    # Pro-inflammatory
    IL_1 = "IL-1"       # Fever, activation
    IL_6 = "IL-6"       # Acute phase response
    TNF_ALPHA = "TNF-α" # Tumor necrosis, death signals
    IFN_GAMMA = "IFN-γ" # Interferon, antiviral

    # Anti-inflammatory
    IL_10 = "IL-10"     # Suppression
    IL_4 = "IL-4"       # Resolution
    TGF_BETA = "TGF-β"  # Tissue repair


class InflammationPhase(Enum):
    """Phases of inflammatory response"""
    INITIATION = "initiation"
    AMPLIFICATION = "amplification"
    PEAK = "peak"
    RESOLUTION = "resolution"
    REPAIR = "repair"
    RESOLVED = "resolved"


class AlertPriority(Enum):
    """Priority levels for alerts"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Cytokine:
    """A signaling molecule"""
    type: CytokineType
    concentration: float = 0.0
    source: str = ""
    released_at: float = field(default_factory=time.time)
    half_life: float = 60.0  # seconds


@dataclass
class InflammatorySignal:
    """A signal in the inflammatory cascade"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_location: str = ""
    target_locations: List[str] = field(default_factory=list)
    priority: AlertPriority = AlertPriority.NORMAL
    cytokines: List[Cytokine] = field(default_factory=list)
    threat_id: Optional[str] = None
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    acknowledged_by: List[str] = field(default_factory=list)


@dataclass
class InflammationZone:
    """A zone of active inflammation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    location: str = ""
    level: InflammationLevel = InflammationLevel.MILD
    phase: InflammationPhase = InflammationPhase.INITIATION
    cytokine_levels: Dict[str, float] = field(default_factory=dict)
    threat_ids: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    peak_level: InflammationLevel = InflammationLevel.MILD
    duration: float = 0.0


@dataclass
class ResolutionEvent:
    """Record of inflammation resolution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    zone_id: str = ""
    resolved_at: float = field(default_factory=time.time)
    peak_level: InflammationLevel = InflammationLevel.MILD
    duration: float = 0.0
    residual_damage: float = 0.0


# ============================================================================
# Inflammation System
# ============================================================================

class InflammationSystem:
    """
    System-wide inflammation and alert propagation.

    Features:
    - Cytokine-based signaling
    - Cascading alerts
    - Inflammation zone management
    - Resolution mechanisms
    - Chronic inflammation prevention

    Example:
        inflammation = InflammationSystem()

        # Trigger inflammation at a location
        zone = inflammation.trigger_inflammation("memory_subsystem", threat)

        # Propagate alert
        signal = inflammation.create_alert(threat, priority=AlertPriority.HIGH)
        inflammation.propagate_signal(signal)

        # Advance inflammation through phases
        inflammation.tick()

        # Resolve when threat is handled
        inflammation.begin_resolution(zone.id)
    """

    def __init__(
        self,
        inflammation_threshold: float = 0.5,
        max_inflammation_level: InflammationLevel = InflammationLevel.SEVERE,
        resolution_rate: float = 0.1,
        chronic_threshold: float = 3600  # 1 hour
    ):
        self.inflammation_threshold = inflammation_threshold
        self.max_inflammation_level = max_inflammation_level
        self.resolution_rate = resolution_rate
        self.chronic_threshold = chronic_threshold

        # Active inflammation zones
        self.zones: Dict[str, InflammationZone] = {}

        # Cytokine environment
        self.cytokines: Dict[str, List[Cytokine]] = defaultdict(list)

        # Signal queue
        self.active_signals: List[InflammatorySignal] = []
        self.signal_history: List[InflammatorySignal] = []

        # Resolution history
        self.resolutions: List[ResolutionEvent] = []

        # Statistics
        self.total_signals = 0
        self.total_zones_created = 0

        # Subscribed listeners
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "inflammation_triggered": [],
            "alert_propagated": [],
            "inflammation_resolved": [],
            "chronic_detected": []
        }

    def subscribe(
        self,
        location: str,
        callback: Callable[[InflammatorySignal], None]
    ):
        """Subscribe to signals at a location"""
        self.listeners[location].append(callback)

    def trigger_inflammation(
        self,
        location: str,
        threat: Threat,
        initial_level: Optional[InflammationLevel] = None
    ) -> InflammationZone:
        """Trigger inflammation at a location"""
        self.total_zones_created += 1

        # Calculate initial level from threat severity
        if initial_level is None:
            initial_level = self._severity_to_inflammation(threat.severity)

        # Check for existing zone at location
        existing = self._get_zone_at_location(location)
        if existing:
            # Escalate existing inflammation
            return self._escalate_zone(existing, threat, initial_level)

        # Create new zone
        zone = InflammationZone(
            location=location,
            level=initial_level,
            phase=InflammationPhase.INITIATION,
            threat_ids=[threat.id],
            peak_level=initial_level
        )

        # Initialize cytokines
        zone.cytokine_levels = self._initial_cytokines(initial_level)

        self.zones[zone.id] = zone

        # Release initial cytokines
        self._release_cytokines(location, initial_level)

        for callback in self._callbacks["inflammation_triggered"]:
            callback(zone, threat)

        return zone

    def _severity_to_inflammation(
        self,
        severity: ThreatSeverity
    ) -> InflammationLevel:
        """Convert threat severity to inflammation level"""
        mapping = {
            ThreatSeverity.LOW: InflammationLevel.MILD,
            ThreatSeverity.MEDIUM: InflammationLevel.MODERATE,
            ThreatSeverity.HIGH: InflammationLevel.SEVERE,
            ThreatSeverity.CRITICAL: InflammationLevel.CRITICAL
        }
        return mapping.get(severity, InflammationLevel.MODERATE)

    def _get_zone_at_location(self, location: str) -> Optional[InflammationZone]:
        """Get active inflammation zone at location"""
        for zone in self.zones.values():
            if zone.location == location:
                return zone
        return None

    def _escalate_zone(
        self,
        zone: InflammationZone,
        threat: Threat,
        level: InflammationLevel
    ) -> InflammationZone:
        """Escalate an existing inflammation zone"""
        zone.threat_ids.append(threat.id)

        # Increase level
        new_level = InflammationLevel(
            min(level.value + 1, self.max_inflammation_level.value)
        )

        if new_level.value > zone.level.value:
            zone.level = new_level
            zone.phase = InflammationPhase.AMPLIFICATION

        if zone.level.value > zone.peak_level.value:
            zone.peak_level = zone.level

        # Boost cytokines
        for ctype in zone.cytokine_levels:
            zone.cytokine_levels[ctype] *= 1.5

        return zone

    def _initial_cytokines(
        self,
        level: InflammationLevel
    ) -> Dict[str, float]:
        """Get initial cytokine levels for inflammation"""
        base = level.value * 0.2
        return {
            CytokineType.IL_1.value: base,
            CytokineType.IL_6.value: base * 0.8,
            CytokineType.TNF_ALPHA.value: base * 0.6 if level.value >= 3 else 0,
            CytokineType.IFN_GAMMA.value: base * 0.4,
            CytokineType.IL_10.value: 0.1,  # Low anti-inflammatory initially
            CytokineType.IL_4.value: 0.05
        }

    def _release_cytokines(
        self,
        location: str,
        level: InflammationLevel
    ):
        """Release cytokines at a location"""
        base_amount = level.value * 0.3

        # Pro-inflammatory cytokines
        pro_inflammatory = [
            CytokineType.IL_1, CytokineType.IL_6,
            CytokineType.TNF_ALPHA, CytokineType.IFN_GAMMA
        ]

        for ctype in pro_inflammatory:
            cytokine = Cytokine(
                type=ctype,
                concentration=base_amount * (0.5 + np.random.random() * 0.5),
                source=location
            )
            self.cytokines[location].append(cytokine)

    def create_alert(
        self,
        threat: Threat,
        source_location: str = "immune_system",
        target_locations: Optional[List[str]] = None,
        priority: AlertPriority = AlertPriority.NORMAL,
        message: str = ""
    ) -> InflammatorySignal:
        """Create an inflammatory alert signal"""
        self.total_signals += 1

        # Auto-determine priority from threat severity
        if priority == AlertPriority.NORMAL:
            priority = self._severity_to_priority(threat.severity)

        # Create cytokines for signal
        cytokines = [
            Cytokine(
                type=CytokineType.IL_1,
                concentration=priority.value * 0.2,
                source=source_location
            ),
            Cytokine(
                type=CytokineType.IL_6,
                concentration=priority.value * 0.15,
                source=source_location
            )
        ]

        signal = InflammatorySignal(
            source_location=source_location,
            target_locations=target_locations or ["all"],
            priority=priority,
            cytokines=cytokines,
            threat_id=threat.id,
            message=message or f"Alert: {threat.type.value} detected"
        )

        self.active_signals.append(signal)
        return signal

    def _severity_to_priority(
        self,
        severity: ThreatSeverity
    ) -> AlertPriority:
        """Convert threat severity to alert priority"""
        mapping = {
            ThreatSeverity.LOW: AlertPriority.LOW,
            ThreatSeverity.MEDIUM: AlertPriority.NORMAL,
            ThreatSeverity.HIGH: AlertPriority.HIGH,
            ThreatSeverity.CRITICAL: AlertPriority.CRITICAL
        }
        return mapping.get(severity, AlertPriority.NORMAL)

    def propagate_signal(self, signal: InflammatorySignal):
        """Propagate signal to all target locations"""
        targets = signal.target_locations

        if "all" in targets:
            targets = list(self.listeners.keys())

        for location in targets:
            if location in self.listeners:
                for listener in self.listeners[location]:
                    try:
                        listener(signal)
                        signal.acknowledged = True
                        signal.acknowledged_by.append(location)
                    except Exception:
                        pass

        for callback in self._callbacks["alert_propagated"]:
            callback(signal)

    def begin_resolution(self, zone_id: str) -> bool:
        """Begin resolution of an inflammation zone"""
        zone = self.zones.get(zone_id)
        if not zone:
            return False

        zone.phase = InflammationPhase.RESOLUTION

        # Release anti-inflammatory cytokines
        zone.cytokine_levels[CytokineType.IL_10.value] = 0.8
        zone.cytokine_levels[CytokineType.IL_4.value] = 0.6
        zone.cytokine_levels[CytokineType.TGF_BETA.value] = 0.5

        # Reduce pro-inflammatory
        for ctype in [CytokineType.IL_1, CytokineType.IL_6,
                     CytokineType.TNF_ALPHA, CytokineType.IFN_GAMMA]:
            if ctype.value in zone.cytokine_levels:
                zone.cytokine_levels[ctype.value] *= 0.5

        return True

    def advance_zones(self):
        """Advance all inflammation zones through phases"""
        current_time = time.time()
        to_remove = []

        for zone_id, zone in self.zones.items():
            zone.duration = current_time - zone.started_at

            # Check for chronic inflammation
            if zone.duration > self.chronic_threshold:
                if zone.phase not in [InflammationPhase.RESOLUTION,
                                       InflammationPhase.REPAIR,
                                       InflammationPhase.RESOLVED]:
                    for callback in self._callbacks["chronic_detected"]:
                        callback(zone)

            # Phase progression
            if zone.phase == InflammationPhase.INITIATION:
                zone.phase = InflammationPhase.AMPLIFICATION

            elif zone.phase == InflammationPhase.AMPLIFICATION:
                if zone.level.value >= InflammationLevel.SEVERE.value:
                    zone.phase = InflammationPhase.PEAK

            elif zone.phase == InflammationPhase.PEAK:
                # Auto-begin resolution after some time at peak
                if np.random.random() < self.resolution_rate:
                    self.begin_resolution(zone_id)

            elif zone.phase == InflammationPhase.RESOLUTION:
                # Decrease inflammation level
                if zone.level.value > 0:
                    zone.level = InflammationLevel(zone.level.value - 1)
                    if zone.level == InflammationLevel.NONE:
                        zone.phase = InflammationPhase.REPAIR

            elif zone.phase == InflammationPhase.REPAIR:
                # Complete resolution
                zone.phase = InflammationPhase.RESOLVED
                to_remove.append(zone_id)

                # Record resolution
                resolution = ResolutionEvent(
                    zone_id=zone_id,
                    peak_level=zone.peak_level,
                    duration=zone.duration,
                    residual_damage=zone.peak_level.value * 0.1
                )
                self.resolutions.append(resolution)

                for callback in self._callbacks["inflammation_resolved"]:
                    callback(zone, resolution)

        # Remove resolved zones
        for zone_id in to_remove:
            del self.zones[zone_id]

    def decay_cytokines(self):
        """Decay cytokines based on half-life"""
        current_time = time.time()

        for location in list(self.cytokines.keys()):
            active = []
            for cytokine in self.cytokines[location]:
                age = current_time - cytokine.released_at
                decay = np.exp(-np.log(2) * age / cytokine.half_life)
                cytokine.concentration *= decay

                if cytokine.concentration > 0.01:
                    active.append(cytokine)

            self.cytokines[location] = active

    def process_signals(self):
        """Process and propagate active signals"""
        while self.active_signals:
            signal = self.active_signals.pop(0)
            self.propagate_signal(signal)
            self.signal_history.append(signal)

    def get_system_inflammation_level(self) -> InflammationLevel:
        """Get overall system inflammation level"""
        if not self.zones:
            return InflammationLevel.NONE

        max_level = max(zone.level.value for zone in self.zones.values())

        # If many zones, increase level
        if len(self.zones) >= 5:
            max_level = min(max_level + 1, InflammationLevel.SYSTEMIC.value)

        return InflammationLevel(max_level)

    def get_cytokine_summary(self) -> Dict[str, float]:
        """Get summary of all cytokine levels"""
        summary = defaultdict(float)

        for location_cytokines in self.cytokines.values():
            for cytokine in location_cytokines:
                summary[cytokine.type.value] += cytokine.concentration

        return dict(summary)

    def tick(self):
        """Process one cycle"""
        self.process_signals()
        self.advance_zones()
        self.decay_cytokines()

    def get_statistics(self) -> Dict[str, Any]:
        """Get inflammation system statistics"""
        return {
            "active_zones": len(self.zones),
            "total_zones_created": self.total_zones_created,
            "total_signals": self.total_signals,
            "resolutions": len(self.resolutions),
            "system_level": self.get_system_inflammation_level().name,
            "cytokine_summary": self.get_cytokine_summary(),
            "zones_by_phase": {
                phase.value: len([z for z in self.zones.values() if z.phase == phase])
                for phase in InflammationPhase
            },
            "zones_by_level": {
                level.name: len([z for z in self.zones.values() if z.level == level])
                for level in InflammationLevel
            },
            "average_resolution_time": np.mean([r.duration for r in self.resolutions])
                if self.resolutions else 0.0
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Inflammation System Demo")
    print("=" * 50)

    # Create system
    inflammation = InflammationSystem()

    # Add some listeners
    alerts_received = []
    inflammation.subscribe("memory", lambda s: alerts_received.append(("memory", s)))
    inflammation.subscribe("consciousness", lambda s: alerts_received.append(("consciousness", s)))

    print("\n1. Triggering inflammation...")

    threat = Threat(
        type=ThreatType.MALICIOUS_INPUT,
        severity=ThreatSeverity.HIGH,
        signature="attack_pattern"
    )

    zone = inflammation.trigger_inflammation("memory_subsystem", threat)
    print(f"   Zone created: {zone.location}")
    print(f"   Level: {zone.level.name}")
    print(f"   Phase: {zone.phase.value}")

    print("\n2. Creating and propagating alert...")

    signal = inflammation.create_alert(
        threat,
        source_location="memory_subsystem",
        target_locations=["memory", "consciousness"],
        message="Critical threat detected in memory subsystem"
    )
    print(f"   Signal created with priority: {signal.priority.name}")

    inflammation.propagate_signal(signal)
    print(f"   Alerts received: {len(alerts_received)}")
    for loc, sig in alerts_received:
        print(f"     {loc}: {sig.message}")

    print("\n3. Advancing inflammation phases...")

    for i in range(5):
        inflammation.tick()
        stats = inflammation.get_statistics()
        print(f"   Tick {i+1}:")
        print(f"     System level: {stats['system_level']}")
        print(f"     Zones by phase: {stats['zones_by_phase']}")

    print("\n4. Beginning resolution...")

    inflammation.begin_resolution(zone.id)
    print(f"   Zone phase: {zone.phase.value}")

    for i in range(5):
        inflammation.tick()

    print("\n5. Final statistics:")
    stats = inflammation.get_statistics()
    print(f"   Active zones: {stats['active_zones']}")
    print(f"   Total signals: {stats['total_signals']}")
    print(f"   Resolutions: {stats['resolutions']}")
    if stats['resolutions'] > 0:
        print(f"   Avg resolution time: {stats['average_resolution_time']:.1f}s")
