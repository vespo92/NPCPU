"""
Digital Immune Defense System

Detects and responds to threats against the organism including:
- Corruption and data integrity threats
- Malicious inputs
- Resource exhaustion attacks
- Pattern anomalies
"""

import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Enums
# ============================================================================

class ThreatType(Enum):
    """Types of threats"""
    CORRUPTION = "corruption"           # Data/state corruption
    INTRUSION = "intrusion"             # Unauthorized access
    MALICIOUS_INPUT = "malicious_input" # Harmful data
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # DoS-style attacks
    ANOMALY = "anomaly"                 # Unusual patterns
    CASCADE_FAILURE = "cascade_failure" # Spreading failures
    POISONING = "poisoning"             # Training/data poisoning


class ThreatSeverity(Enum):
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefenseAction(Enum):
    """Possible defense actions"""
    IGNORE = "ignore"
    MONITOR = "monitor"
    QUARANTINE = "quarantine"
    NEUTRALIZE = "neutralize"
    ELIMINATE = "eliminate"
    ALERT = "alert"
    ISOLATE = "isolate"
    REPAIR = "repair"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Threat:
    """A detected threat"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ThreatType = ThreatType.ANOMALY
    severity: ThreatSeverity = ThreatSeverity.LOW
    source: str = ""
    target: str = ""
    signature: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)
    neutralized: bool = False
    neutralized_at: Optional[float] = None


@dataclass
class AntibodyPattern:
    """Pattern for recognizing threats (like biological antibodies)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    threat_type: ThreatType = ThreatType.ANOMALY
    signature_pattern: str = ""
    detection_score: float = 0.0
    false_positive_rate: float = 0.0
    created_at: float = field(default_factory=time.time)
    match_count: int = 0


@dataclass
class DefenseResponse:
    """Response to a threat"""
    threat_id: str
    actions: List[DefenseAction]
    success: bool = False
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuarantineZone:
    """Isolated zone for containing threats"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    capacity: int = 100
    contained_threats: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# ============================================================================
# Immune System
# ============================================================================

class ImmuneSystem:
    """
    Digital immune system for threat detection and response.

    Features:
    - Pattern-based threat detection (antibodies)
    - Anomaly detection
    - Adaptive learning
    - Quarantine capabilities
    - Response coordination

    Example:
        immune = ImmuneSystem()

        # Add antibody patterns
        immune.add_antibody_pattern("corruption_pattern", ThreatType.CORRUPTION, "corrupt_")

        # Scan for threats
        threats = immune.scan(data)

        # Respond to threats
        for threat in threats:
            response = immune.respond(threat)
    """

    def __init__(
        self,
        sensitivity: float = 0.7,
        auto_response: bool = True
    ):
        self.sensitivity = sensitivity
        self.auto_response = auto_response

        # Antibody patterns
        self.antibodies: Dict[str, AntibodyPattern] = {}

        # Threat tracking
        self.active_threats: Dict[str, Threat] = {}
        self.threat_history: List[Threat] = []

        # Quarantine zones
        self.quarantine_zones: Dict[str, QuarantineZone] = {}
        self.default_quarantine = QuarantineZone(name="default")
        self.quarantine_zones["default"] = self.default_quarantine

        # Detection statistics
        self.baseline_stats: Dict[str, float] = {}
        self.current_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Response history
        self.response_history: List[DefenseResponse] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "threat_detected": [],
            "threat_neutralized": [],
            "critical_threat": []
        }

        # Initialize default antibodies
        self._initialize_antibodies()

    def _initialize_antibodies(self):
        """Initialize default antibody patterns"""
        default_patterns = [
            ("corruption_detector", ThreatType.CORRUPTION, "corrupt|invalid|malformed"),
            ("intrusion_detector", ThreatType.INTRUSION, "unauthorized|access_denied|breach"),
            ("anomaly_detector", ThreatType.ANOMALY, "unexpected|abnormal"),
            ("resource_attack", ThreatType.RESOURCE_EXHAUSTION, "overflow|exhausted|denied"),
            ("cascade_detector", ThreatType.CASCADE_FAILURE, "cascade|propagate|spread"),
        ]

        for name, threat_type, pattern in default_patterns:
            self.add_antibody_pattern(name, threat_type, pattern)

    def add_antibody_pattern(
        self,
        name: str,
        threat_type: ThreatType,
        signature_pattern: str
    ) -> AntibodyPattern:
        """Add an antibody pattern for threat detection"""
        antibody = AntibodyPattern(
            name=name,
            threat_type=threat_type,
            signature_pattern=signature_pattern
        )
        self.antibodies[antibody.id] = antibody
        return antibody

    def scan(
        self,
        data: Any,
        context: str = "general"
    ) -> List[Threat]:
        """
        Scan data for threats.

        Returns list of detected threats.
        """
        threats = []

        # Convert data to scannable format
        data_str = str(data).lower()
        data_hash = hashlib.md5(data_str.encode()).hexdigest()

        # Pattern matching with antibodies
        for antibody in self.antibodies.values():
            if self._pattern_match(data_str, antibody.signature_pattern):
                antibody.match_count += 1

                threat = Threat(
                    type=antibody.threat_type,
                    severity=self._assess_severity(antibody, data_str),
                    source=context,
                    signature=data_hash,
                    details={
                        "antibody": antibody.name,
                        "pattern": antibody.signature_pattern
                    }
                )
                threats.append(threat)

        # Anomaly detection
        anomaly_threat = self._detect_anomaly(data_str, context)
        if anomaly_threat:
            threats.append(anomaly_threat)

        # Process detected threats
        for threat in threats:
            self._register_threat(threat)

            if self.auto_response:
                self.respond(threat)

        return threats

    def _pattern_match(self, data: str, pattern: str) -> bool:
        """Check if data matches pattern"""
        import re
        try:
            return bool(re.search(pattern, data, re.IGNORECASE))
        except re.error:
            return pattern.lower() in data.lower()

    def _assess_severity(
        self,
        antibody: AntibodyPattern,
        data: str
    ) -> ThreatSeverity:
        """Assess threat severity"""
        # Simple heuristic based on antibody and data
        score = antibody.detection_score

        # More matches = higher severity
        import re
        try:
            matches = len(re.findall(antibody.signature_pattern, data, re.IGNORECASE))
        except re.error:
            matches = data.lower().count(antibody.signature_pattern.lower())

        if matches > 5:
            return ThreatSeverity.CRITICAL
        elif matches > 3:
            return ThreatSeverity.HIGH
        elif matches > 1:
            return ThreatSeverity.MEDIUM
        return ThreatSeverity.LOW

    def _detect_anomaly(
        self,
        data: str,
        context: str
    ) -> Optional[Threat]:
        """Detect anomalies through statistical analysis"""
        # Calculate data characteristics
        data_len = len(data)
        unique_chars = len(set(data))
        entropy = self._calculate_entropy(data)

        # Store for baseline
        stat_key = f"{context}_entropy"
        self.current_stats[stat_key].append(entropy)

        # Check against baseline
        if stat_key in self.baseline_stats:
            baseline = self.baseline_stats[stat_key]
            if abs(entropy - baseline) > baseline * (1 - self.sensitivity):
                return Threat(
                    type=ThreatType.ANOMALY,
                    severity=ThreatSeverity.MEDIUM,
                    source=context,
                    details={
                        "expected_entropy": baseline,
                        "actual_entropy": entropy,
                        "deviation": abs(entropy - baseline)
                    }
                )
        else:
            # Establish baseline if we have enough samples
            if len(self.current_stats[stat_key]) >= 10:
                self.baseline_stats[stat_key] = np.mean(self.current_stats[stat_key])

        return None

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0

        freq = defaultdict(int)
        for char in data:
            freq[char] += 1

        entropy = 0.0
        for count in freq.values():
            p = count / len(data)
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _register_threat(self, threat: Threat):
        """Register a detected threat"""
        self.active_threats[threat.id] = threat
        self.threat_history.append(threat)

        # Trigger callbacks
        for callback in self._callbacks["threat_detected"]:
            callback(threat)

        if threat.severity == ThreatSeverity.CRITICAL:
            for callback in self._callbacks["critical_threat"]:
                callback(threat)

    def respond(self, threat: Threat) -> DefenseResponse:
        """
        Respond to a threat.

        Returns the response taken.
        """
        actions = self._determine_actions(threat)

        success = True
        details = {}

        for action in actions:
            action_success = self._execute_action(action, threat)
            if not action_success:
                success = False
            details[action.value] = action_success

        response = DefenseResponse(
            threat_id=threat.id,
            actions=actions,
            success=success,
            details=details
        )

        self.response_history.append(response)

        if success:
            threat.neutralized = True
            threat.neutralized_at = time.time()

            # Remove from active
            if threat.id in self.active_threats:
                del self.active_threats[threat.id]

            # Trigger callbacks
            for callback in self._callbacks["threat_neutralized"]:
                callback(threat, response)

        return response

    def _determine_actions(self, threat: Threat) -> List[DefenseAction]:
        """Determine appropriate response actions"""
        actions = []

        if threat.severity == ThreatSeverity.LOW:
            actions = [DefenseAction.MONITOR]

        elif threat.severity == ThreatSeverity.MEDIUM:
            actions = [DefenseAction.ALERT, DefenseAction.MONITOR]
            if threat.type in [ThreatType.CORRUPTION, ThreatType.MALICIOUS_INPUT]:
                actions.append(DefenseAction.QUARANTINE)

        elif threat.severity == ThreatSeverity.HIGH:
            actions = [DefenseAction.ALERT, DefenseAction.QUARANTINE, DefenseAction.ISOLATE]
            if threat.type == ThreatType.CORRUPTION:
                actions.append(DefenseAction.REPAIR)

        elif threat.severity == ThreatSeverity.CRITICAL:
            actions = [
                DefenseAction.ALERT,
                DefenseAction.QUARANTINE,
                DefenseAction.ISOLATE,
                DefenseAction.NEUTRALIZE
            ]

        return actions

    def _execute_action(self, action: DefenseAction, threat: Threat) -> bool:
        """Execute a defense action"""
        if action == DefenseAction.IGNORE:
            return True

        elif action == DefenseAction.MONITOR:
            # Add to monitoring (already tracked)
            return True

        elif action == DefenseAction.QUARANTINE:
            return self._quarantine_threat(threat)

        elif action == DefenseAction.NEUTRALIZE:
            return self._neutralize_threat(threat)

        elif action == DefenseAction.ELIMINATE:
            return self._eliminate_threat(threat)

        elif action == DefenseAction.ALERT:
            return self._send_alert(threat)

        elif action == DefenseAction.ISOLATE:
            return self._isolate_threat(threat)

        elif action == DefenseAction.REPAIR:
            return True  # Repair handled by RepairSystem

        return False

    def _quarantine_threat(self, threat: Threat) -> bool:
        """Quarantine a threat"""
        zone = self.default_quarantine

        if len(zone.contained_threats) < zone.capacity:
            zone.contained_threats.append(threat.id)
            return True

        return False

    def _neutralize_threat(self, threat: Threat) -> bool:
        """Neutralize a threat"""
        # Mark as neutralized
        threat.neutralized = True
        threat.neutralized_at = time.time()
        return True

    def _eliminate_threat(self, threat: Threat) -> bool:
        """Completely eliminate a threat"""
        threat.neutralized = True
        threat.neutralized_at = time.time()

        # Remove from quarantine if present
        for zone in self.quarantine_zones.values():
            if threat.id in zone.contained_threats:
                zone.contained_threats.remove(threat.id)

        return True

    def _send_alert(self, threat: Threat) -> bool:
        """Send alert about threat"""
        # In real implementation, would send to monitoring system
        return True

    def _isolate_threat(self, threat: Threat) -> bool:
        """Isolate threat source"""
        # Mark source for isolation
        return True

    def learn_from_threat(self, threat: Threat, was_real: bool):
        """
        Learn from threat detection.

        Helps improve detection accuracy.
        """
        # Find antibody that detected it
        if "antibody" in threat.details:
            antibody_name = threat.details["antibody"]
            for antibody in self.antibodies.values():
                if antibody.name == antibody_name:
                    if was_real:
                        antibody.detection_score += 0.1
                    else:
                        antibody.false_positive_rate += 0.05
                    break

    def on_threat_detected(self, callback: Callable[[Threat], None]):
        """Register callback for threat detection"""
        self._callbacks["threat_detected"].append(callback)

    def on_threat_neutralized(self, callback: Callable[[Threat, DefenseResponse], None]):
        """Register callback for threat neutralization"""
        self._callbacks["threat_neutralized"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get immune system status"""
        return {
            "active_threats": len(self.active_threats),
            "total_threats_detected": len(self.threat_history),
            "antibody_count": len(self.antibodies),
            "quarantined": sum(
                len(z.contained_threats) for z in self.quarantine_zones.values()
            ),
            "sensitivity": self.sensitivity,
            "threat_breakdown": {
                tt.value: len([t for t in self.threat_history if t.type == tt])
                for tt in ThreatType
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Immune Defense System Demo")
    print("=" * 50)

    # Create immune system
    immune = ImmuneSystem(sensitivity=0.7, auto_response=True)

    # Track detections
    detected_threats = []
    immune.on_threat_detected(lambda t: detected_threats.append(t))

    print(f"\n1. Initial state:")
    status = immune.get_status()
    print(f"   Antibodies: {status['antibody_count']}")
    print(f"   Sensitivity: {status['sensitivity']}")

    # Test with clean data
    print("\n2. Scanning clean data...")
    clean_data = "This is normal operational data with no issues"
    threats = immune.scan(clean_data, "test_clean")
    print(f"   Threats found: {len(threats)}")

    # Test with suspicious data
    print("\n3. Scanning suspicious data...")
    suspicious_data = "Warning: corrupt data detected, invalid checksum, malformed packet"
    threats = immune.scan(suspicious_data, "test_suspicious")
    print(f"   Threats found: {len(threats)}")
    for t in threats:
        print(f"     - {t.type.value}: {t.severity.value}")

    # Test with attack patterns
    print("\n4. Scanning attack patterns...")
    attack_data = "CASCADE failure detected! System overflow, unauthorized access breach!"
    threats = immune.scan(attack_data, "test_attack")
    print(f"   Threats found: {len(threats)}")
    for t in threats:
        print(f"     - {t.type.value}: {t.severity.value}")
        if t.id in immune.active_threats:
            print(f"       Status: Active")
        else:
            print(f"       Status: Neutralized")

    # Final status
    print("\n5. Final status:")
    status = immune.get_status()
    print(f"   Total threats detected: {status['total_threats_detected']}")
    print(f"   Active threats: {status['active_threats']}")
    print(f"   Quarantined: {status['quarantined']}")
    print("\n   Threat breakdown:")
    for tt, count in status['threat_breakdown'].items():
        if count > 0:
            print(f"     {tt}: {count}")
