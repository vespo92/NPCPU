"""
Pathogen Detection System

Advanced anomaly detection for identifying malicious inputs, corrupted data,
and potential threats to the organism. Uses multiple detection strategies:
- Statistical anomaly detection
- Pattern-based signature matching
- Behavioral analysis
- Entropy analysis
"""

import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
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

class PathogenType(Enum):
    """Types of pathogens/malicious inputs"""
    MALFORMED_DATA = "malformed_data"
    INJECTION_ATTACK = "injection_attack"
    OVERFLOW_ATTEMPT = "overflow_attempt"
    CORRUPTION = "corruption"
    POISONING = "poisoning"
    TROJAN = "trojan"
    ANOMALOUS_PATTERN = "anomalous_pattern"
    UNKNOWN = "unknown"


class DetectionMethod(Enum):
    """Methods for pathogen detection"""
    SIGNATURE = "signature"         # Known pattern matching
    HEURISTIC = "heuristic"         # Rule-based detection
    STATISTICAL = "statistical"     # Statistical anomaly
    BEHAVIORAL = "behavioral"       # Behavior deviation
    ENTROPY = "entropy"             # Information entropy analysis
    HYBRID = "hybrid"               # Multiple methods combined


class RiskLevel(Enum):
    """Risk levels for detected pathogens"""
    NEGLIGIBLE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PathogenSignature:
    """Signature for identifying known pathogens"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    pathogen_type: PathogenType = PathogenType.UNKNOWN
    pattern: str = ""
    hash_patterns: Set[str] = field(default_factory=set)
    behavioral_markers: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MODERATE
    confidence: float = 0.9
    created_at: float = field(default_factory=time.time)
    match_count: int = 0


@dataclass
class DetectedPathogen:
    """A detected pathogen instance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pathogen_type: PathogenType = PathogenType.UNKNOWN
    detection_method: DetectionMethod = DetectionMethod.HEURISTIC
    risk_level: RiskLevel = RiskLevel.MODERATE
    confidence: float = 0.5
    source: str = ""
    payload: Any = None
    signature_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)
    neutralized: bool = False


@dataclass
class ScanResult:
    """Result of a pathogen scan"""
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target: str = ""
    pathogens: List[DetectedPathogen] = field(default_factory=list)
    scan_duration: float = 0.0
    methods_used: List[DetectionMethod] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.NEGLIGIBLE
    clean: bool = True
    scanned_at: float = field(default_factory=time.time)


@dataclass
class AnomalyProfile:
    """Statistical profile for anomaly detection"""
    feature: str = ""
    mean: float = 0.0
    std: float = 1.0
    min_value: float = float('-inf')
    max_value: float = float('inf')
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)


# ============================================================================
# Pathogen Detector
# ============================================================================

class PathogenDetector:
    """
    Advanced pathogen detection system.

    Features:
    - Multi-method detection (signature, heuristic, statistical, behavioral)
    - Adaptive thresholds based on environment
    - Learning from confirmed threats
    - Low false-positive rate optimization

    Example:
        detector = PathogenDetector()

        # Add known pathogen signature
        detector.add_signature("sql_injection", PathogenType.INJECTION_ATTACK,
                               pattern="SELECT.*FROM.*WHERE")

        # Scan input
        result = detector.scan(user_input, "user_query")
        if not result.clean:
            for pathogen in result.pathogens:
                print(f"Detected: {pathogen.pathogen_type.value}")
    """

    def __init__(
        self,
        sensitivity: float = 0.7,
        entropy_threshold: float = 7.0,
        anomaly_zscore: float = 3.0,
        enable_learning: bool = True
    ):
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.entropy_threshold = entropy_threshold
        self.anomaly_zscore = anomaly_zscore
        self.enable_learning = enable_learning

        # Signature database
        self.signatures: Dict[str, PathogenSignature] = {}

        # Statistical profiles for anomaly detection
        self.profiles: Dict[str, AnomalyProfile] = {}

        # Feature history for learning
        self.feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Detection history
        self.detection_history: List[DetectedPathogen] = []
        self.scan_count = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "pathogen_detected": [],
            "high_risk_detected": [],
            "scan_complete": []
        }

        # Initialize default signatures
        self._initialize_default_signatures()

    def _initialize_default_signatures(self):
        """Initialize common pathogen signatures"""
        default_sigs = [
            ("sql_injection", PathogenType.INJECTION_ATTACK,
             r"(SELECT|INSERT|UPDATE|DELETE|DROP|UNION).*\s+(FROM|INTO|SET|TABLE)",
             RiskLevel.HIGH),
            ("xss_attack", PathogenType.INJECTION_ATTACK,
             r"<script[^>]*>.*</script>",
             RiskLevel.HIGH),
            ("command_injection", PathogenType.INJECTION_ATTACK,
             r"(;|\||`|\$\()\s*(rm|cat|ls|echo|wget|curl)",
             RiskLevel.CRITICAL),
            ("buffer_overflow", PathogenType.OVERFLOW_ATTEMPT,
             r"(%[0-9a-fA-F]{2}){50,}|A{100,}",
             RiskLevel.HIGH),
            ("format_string", PathogenType.INJECTION_ATTACK,
             r"%[nxsp]{1,2}",
             RiskLevel.MODERATE),
            ("path_traversal", PathogenType.INJECTION_ATTACK,
             r"\.\./|\.\.\x5c",
             RiskLevel.MODERATE),
            ("null_byte", PathogenType.MALFORMED_DATA,
             r"\x00",
             RiskLevel.MODERATE),
        ]

        for name, ptype, pattern, risk in default_sigs:
            self.add_signature(name, ptype, pattern, risk)

    def add_signature(
        self,
        name: str,
        pathogen_type: PathogenType,
        pattern: str,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        confidence: float = 0.9
    ) -> PathogenSignature:
        """Add a pathogen signature for detection"""
        sig = PathogenSignature(
            name=name,
            pathogen_type=pathogen_type,
            pattern=pattern,
            risk_level=risk_level,
            confidence=confidence
        )
        self.signatures[sig.id] = sig
        return sig

    def remove_signature(self, signature_id: str) -> bool:
        """Remove a signature"""
        if signature_id in self.signatures:
            del self.signatures[signature_id]
            return True
        return False

    def scan(
        self,
        data: Any,
        source: str = "unknown",
        methods: Optional[List[DetectionMethod]] = None
    ) -> ScanResult:
        """
        Scan data for pathogens.

        Args:
            data: Data to scan (will be converted to string)
            source: Source identifier for the data
            methods: Specific methods to use (default: all)

        Returns:
            ScanResult with detected pathogens
        """
        start_time = time.time()
        self.scan_count += 1

        if methods is None:
            methods = list(DetectionMethod)

        pathogens: List[DetectedPathogen] = []
        data_str = str(data)

        # Signature-based detection
        if DetectionMethod.SIGNATURE in methods:
            pathogens.extend(self._scan_signatures(data_str, source))

        # Heuristic detection
        if DetectionMethod.HEURISTIC in methods:
            pathogens.extend(self._scan_heuristics(data_str, source))

        # Statistical anomaly detection
        if DetectionMethod.STATISTICAL in methods:
            pathogens.extend(self._scan_statistical(data_str, source))

        # Entropy analysis
        if DetectionMethod.ENTROPY in methods:
            pathogens.extend(self._scan_entropy(data_str, source))

        # Behavioral analysis (if context available)
        if DetectionMethod.BEHAVIORAL in methods:
            pathogens.extend(self._scan_behavioral(data, source))

        # Remove duplicates and merge
        pathogens = self._deduplicate_pathogens(pathogens)

        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(pathogens)

        # Create result
        result = ScanResult(
            target=source,
            pathogens=pathogens,
            scan_duration=time.time() - start_time,
            methods_used=methods,
            overall_risk=overall_risk,
            clean=len(pathogens) == 0
        )

        # Record detections
        for pathogen in pathogens:
            self.detection_history.append(pathogen)
            self._trigger_callbacks(pathogen)

        # Trigger scan complete callback
        for callback in self._callbacks["scan_complete"]:
            callback(result)

        return result

    def _scan_signatures(self, data: str, source: str) -> List[DetectedPathogen]:
        """Scan using signature patterns"""
        import re
        pathogens = []

        for sig in self.signatures.values():
            try:
                if re.search(sig.pattern, data, re.IGNORECASE):
                    sig.match_count += 1
                    pathogens.append(DetectedPathogen(
                        pathogen_type=sig.pathogen_type,
                        detection_method=DetectionMethod.SIGNATURE,
                        risk_level=sig.risk_level,
                        confidence=sig.confidence * self.sensitivity,
                        source=source,
                        signature_id=sig.id,
                        details={
                            "signature_name": sig.name,
                            "pattern": sig.pattern
                        }
                    ))
            except re.error:
                # Invalid regex pattern
                if sig.pattern.lower() in data.lower():
                    pathogens.append(DetectedPathogen(
                        pathogen_type=sig.pathogen_type,
                        detection_method=DetectionMethod.SIGNATURE,
                        risk_level=sig.risk_level,
                        confidence=sig.confidence * 0.8,
                        source=source,
                        signature_id=sig.id,
                        details={"signature_name": sig.name}
                    ))

        return pathogens

    def _scan_heuristics(self, data: str, source: str) -> List[DetectedPathogen]:
        """Scan using heuristic rules"""
        pathogens = []

        # Check for suspicious patterns
        heuristic_rules = [
            # Extremely long strings
            (len(data) > 10000, PathogenType.OVERFLOW_ATTEMPT,
             RiskLevel.MODERATE, "excessive_length"),

            # High ratio of special characters
            (self._special_char_ratio(data) > 0.5, PathogenType.MALFORMED_DATA,
             RiskLevel.MODERATE, "high_special_char_ratio"),

            # Repeated patterns (potential padding attack)
            (self._detect_repetition(data) > 0.7, PathogenType.OVERFLOW_ATTEMPT,
             RiskLevel.MODERATE, "repeated_patterns"),

            # Null bytes in middle of data
            ('\x00' in data[1:-1] if len(data) > 2 else False, PathogenType.MALFORMED_DATA,
             RiskLevel.HIGH, "embedded_null_bytes"),

            # Non-printable characters
            (self._non_printable_ratio(data) > 0.1, PathogenType.MALFORMED_DATA,
             RiskLevel.MODERATE, "non_printable_chars"),
        ]

        for condition, ptype, risk, detail in heuristic_rules:
            if condition:
                pathogens.append(DetectedPathogen(
                    pathogen_type=ptype,
                    detection_method=DetectionMethod.HEURISTIC,
                    risk_level=risk,
                    confidence=0.7 * self.sensitivity,
                    source=source,
                    details={"rule": detail}
                ))

        return pathogens

    def _scan_statistical(self, data: str, source: str) -> List[DetectedPathogen]:
        """Scan using statistical anomaly detection"""
        pathogens = []

        # Extract features
        features = self._extract_features(data)

        for feature_name, value in features.items():
            profile_key = f"{source}_{feature_name}"

            # Update or create profile
            if profile_key in self.profiles:
                profile = self.profiles[profile_key]

                # Check for anomaly
                if profile.sample_count >= 10:
                    if profile.std > 0:
                        zscore = abs(value - profile.mean) / profile.std
                        if zscore > self.anomaly_zscore:
                            pathogens.append(DetectedPathogen(
                                pathogen_type=PathogenType.ANOMALOUS_PATTERN,
                                detection_method=DetectionMethod.STATISTICAL,
                                risk_level=RiskLevel.MODERATE if zscore < 5 else RiskLevel.HIGH,
                                confidence=min(0.95, 0.5 + (zscore - self.anomaly_zscore) * 0.1),
                                source=source,
                                details={
                                    "feature": feature_name,
                                    "value": value,
                                    "expected_mean": profile.mean,
                                    "zscore": zscore
                                }
                            ))

                # Update profile with learning
                if self.enable_learning:
                    self._update_profile(profile, value)
            else:
                # Create new profile
                self.profiles[profile_key] = AnomalyProfile(
                    feature=feature_name,
                    mean=value,
                    sample_count=1
                )

            # Store in history
            self.feature_history[profile_key].append(value)

        return pathogens

    def _scan_entropy(self, data: str, source: str) -> List[DetectedPathogen]:
        """Scan using entropy analysis"""
        pathogens = []

        if not data:
            return pathogens

        entropy = self._calculate_entropy(data)

        # High entropy can indicate encrypted/compressed malicious payloads
        if entropy > self.entropy_threshold:
            pathogens.append(DetectedPathogen(
                pathogen_type=PathogenType.UNKNOWN,
                detection_method=DetectionMethod.ENTROPY,
                risk_level=RiskLevel.MODERATE,
                confidence=min(0.9, (entropy - self.entropy_threshold) / 2),
                source=source,
                details={
                    "entropy": entropy,
                    "threshold": self.entropy_threshold,
                    "reason": "High entropy suggests encrypted or obfuscated content"
                }
            ))

        # Very low entropy with specific characters (potential padding)
        if entropy < 1.0 and len(data) > 100:
            pathogens.append(DetectedPathogen(
                pathogen_type=PathogenType.OVERFLOW_ATTEMPT,
                detection_method=DetectionMethod.ENTROPY,
                risk_level=RiskLevel.MODERATE,
                confidence=0.6 * self.sensitivity,
                source=source,
                details={
                    "entropy": entropy,
                    "reason": "Very low entropy with long data suggests padding attack"
                }
            ))

        return pathogens

    def _scan_behavioral(self, data: Any, source: str) -> List[DetectedPathogen]:
        """Scan using behavioral analysis"""
        pathogens = []

        # Check if data is a dict with behavioral indicators
        if isinstance(data, dict):
            # Check for suspicious keys
            suspicious_keys = ['eval', 'exec', '__import__', 'system', 'popen']
            for key in data.keys():
                if any(sus in str(key).lower() for sus in suspicious_keys):
                    pathogens.append(DetectedPathogen(
                        pathogen_type=PathogenType.TROJAN,
                        detection_method=DetectionMethod.BEHAVIORAL,
                        risk_level=RiskLevel.HIGH,
                        confidence=0.8 * self.sensitivity,
                        source=source,
                        details={"suspicious_key": str(key)}
                    ))

        return pathogens

    def _extract_features(self, data: str) -> Dict[str, float]:
        """Extract statistical features from data"""
        return {
            "length": float(len(data)),
            "unique_chars": float(len(set(data))),
            "entropy": self._calculate_entropy(data),
            "special_ratio": self._special_char_ratio(data),
            "digit_ratio": sum(c.isdigit() for c in data) / max(len(data), 1),
            "alpha_ratio": sum(c.isalpha() for c in data) / max(len(data), 1),
        }

    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy"""
        if not data:
            return 0.0

        freq = defaultdict(int)
        for char in data:
            freq[char] += 1

        entropy = 0.0
        length = len(data)
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _special_char_ratio(self, data: str) -> float:
        """Calculate ratio of special characters"""
        if not data:
            return 0.0
        special = sum(1 for c in data if not c.isalnum() and not c.isspace())
        return special / len(data)

    def _non_printable_ratio(self, data: str) -> float:
        """Calculate ratio of non-printable characters"""
        if not data:
            return 0.0
        non_print = sum(1 for c in data if ord(c) < 32 or ord(c) > 126)
        return non_print / len(data)

    def _detect_repetition(self, data: str) -> float:
        """Detect repeated patterns in data"""
        if len(data) < 10:
            return 0.0

        # Check for character repetition
        max_repeat = 0
        current_repeat = 1
        for i in range(1, len(data)):
            if data[i] == data[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        return max_repeat / len(data)

    def _update_profile(self, profile: AnomalyProfile, value: float):
        """Update statistical profile with new value (online learning)"""
        n = profile.sample_count
        new_n = n + 1

        # Welford's online algorithm for mean and variance
        delta = value - profile.mean
        profile.mean += delta / new_n
        delta2 = value - profile.mean

        # Update variance (M2)
        m2 = profile.std ** 2 * n if n > 0 else 0
        m2 += delta * delta2
        profile.std = np.sqrt(m2 / new_n) if new_n > 0 else 0

        profile.sample_count = new_n
        profile.min_value = min(profile.min_value, value)
        profile.max_value = max(profile.max_value, value)
        profile.last_updated = time.time()

    def _deduplicate_pathogens(
        self,
        pathogens: List[DetectedPathogen]
    ) -> List[DetectedPathogen]:
        """Remove duplicate detections, keeping highest confidence"""
        if not pathogens:
            return []

        # Group by type and source
        groups: Dict[Tuple[PathogenType, str], DetectedPathogen] = {}

        for p in pathogens:
            key = (p.pathogen_type, p.source)
            if key not in groups or groups[key].confidence < p.confidence:
                groups[key] = p

        return list(groups.values())

    def _calculate_overall_risk(
        self,
        pathogens: List[DetectedPathogen]
    ) -> RiskLevel:
        """Calculate overall risk from detected pathogens"""
        if not pathogens:
            return RiskLevel.NEGLIGIBLE

        # Get maximum risk level
        max_risk = max(p.risk_level.value for p in pathogens)

        # Increase risk if multiple high-risk pathogens
        high_risk_count = sum(1 for p in pathogens if p.risk_level.value >= RiskLevel.HIGH.value)
        if high_risk_count >= 3:
            max_risk = min(max_risk + 1, RiskLevel.CATASTROPHIC.value)

        return RiskLevel(max_risk)

    def _trigger_callbacks(self, pathogen: DetectedPathogen):
        """Trigger callbacks for detected pathogen"""
        for callback in self._callbacks["pathogen_detected"]:
            callback(pathogen)

        if pathogen.risk_level.value >= RiskLevel.HIGH.value:
            for callback in self._callbacks["high_risk_detected"]:
                callback(pathogen)

    def on_pathogen_detected(self, callback: Callable[[DetectedPathogen], None]):
        """Register callback for pathogen detection"""
        self._callbacks["pathogen_detected"].append(callback)

    def on_high_risk_detected(self, callback: Callable[[DetectedPathogen], None]):
        """Register callback for high-risk pathogen detection"""
        self._callbacks["high_risk_detected"].append(callback)

    def learn_from_confirmation(
        self,
        pathogen: DetectedPathogen,
        was_real_threat: bool
    ):
        """Learn from confirmed/denied threat detection"""
        if pathogen.signature_id and pathogen.signature_id in self.signatures:
            sig = self.signatures[pathogen.signature_id]
            if was_real_threat:
                # Increase confidence
                sig.confidence = min(1.0, sig.confidence + 0.05)
            else:
                # Decrease confidence (false positive)
                sig.confidence = max(0.1, sig.confidence - 0.1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            "total_scans": self.scan_count,
            "total_detections": len(self.detection_history),
            "signature_count": len(self.signatures),
            "profile_count": len(self.profiles),
            "sensitivity": self.sensitivity,
            "detections_by_type": {
                pt.value: len([d for d in self.detection_history
                              if d.pathogen_type == pt])
                for pt in PathogenType
            },
            "detections_by_method": {
                dm.value: len([d for d in self.detection_history
                              if d.detection_method == dm])
                for dm in DetectionMethod
            },
            "risk_distribution": {
                rl.name: len([d for d in self.detection_history
                             if d.risk_level == rl])
                for rl in RiskLevel
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Pathogen Detection System Demo")
    print("=" * 50)

    # Create detector
    detector = PathogenDetector(sensitivity=0.8)

    # Track detections
    detected = []
    detector.on_pathogen_detected(lambda p: detected.append(p))

    print(f"\n1. Initial state:")
    stats = detector.get_statistics()
    print(f"   Signatures loaded: {stats['signature_count']}")
    print(f"   Sensitivity: {stats['sensitivity']}")

    # Test with clean data
    print("\n2. Scanning clean data...")
    result = detector.scan("Hello, this is normal user input", "user_input")
    print(f"   Clean: {result.clean}")
    print(f"   Pathogens: {len(result.pathogens)}")

    # Test with SQL injection
    print("\n3. Scanning SQL injection attempt...")
    result = detector.scan("SELECT * FROM users WHERE id=1 OR 1=1", "db_query")
    print(f"   Clean: {result.clean}")
    print(f"   Overall risk: {result.overall_risk.name}")
    for p in result.pathogens:
        print(f"     - {p.pathogen_type.value}: {p.detection_method.value}")

    # Test with XSS
    print("\n4. Scanning XSS attempt...")
    result = detector.scan("<script>alert('XSS')</script>", "html_input")
    print(f"   Clean: {result.clean}")
    for p in result.pathogens:
        print(f"     - {p.pathogen_type.value}: {p.risk_level.name}")

    # Test with high-entropy data
    print("\n5. Scanning high-entropy (potentially encrypted) data...")
    import random
    random_data = ''.join(chr(random.randint(0, 255)) for _ in range(500))
    result = detector.scan(random_data, "file_upload")
    print(f"   Clean: {result.clean}")
    for p in result.pathogens:
        print(f"     - {p.pathogen_type.value}: entropy={p.details.get('entropy', 'N/A'):.2f}")

    # Test with padding attack
    print("\n6. Scanning potential buffer overflow...")
    result = detector.scan("A" * 500, "buffer_input")
    print(f"   Clean: {result.clean}")
    for p in result.pathogens:
        print(f"     - {p.pathogen_type.value}: {p.details.get('rule', p.details.get('reason', ''))}")

    # Final statistics
    print("\n7. Detection statistics:")
    stats = detector.get_statistics()
    print(f"   Total scans: {stats['total_scans']}")
    print(f"   Total detections: {stats['total_detections']}")
    print(f"   Detections by type:")
    for ptype, count in stats['detections_by_type'].items():
        if count > 0:
            print(f"     {ptype}: {count}")
