"""
Innate Immunity System

Fast, non-specific defense mechanisms that provide immediate protection.
Unlike adaptive immunity, innate immunity doesn't require prior exposure
and responds identically to all threats of a given type.

Components:
- Physical barriers (input validation, sandboxing)
- Pattern recognition receptors (PRRs)
- Natural killer cells
- Inflammatory response triggering
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

from immune.defense import Threat, ThreatType, ThreatSeverity, DefenseAction


# ============================================================================
# Enums
# ============================================================================

class BarrierType(Enum):
    """Types of physical barriers"""
    INPUT_VALIDATION = "input_validation"
    TYPE_CHECKING = "type_checking"
    SIZE_LIMIT = "size_limit"
    RATE_LIMIT = "rate_limit"
    SANDBOX = "sandbox"
    FIREWALL = "firewall"
    ENCRYPTION = "encryption"


class PRRType(Enum):
    """Pattern Recognition Receptor types"""
    TOLL_LIKE = "toll_like"         # Recognize common threat patterns
    NOD_LIKE = "nod_like"           # Cytosolic pattern recognition
    RIG_LIKE = "rig_like"           # Viral pattern detection
    C_TYPE_LECTIN = "c_type_lectin" # Carbohydrate pattern recognition


class InnateResponseType(Enum):
    """Types of innate immune responses"""
    BLOCK = "block"                 # Block the threat immediately
    QUARANTINE = "quarantine"       # Isolate the threat
    DESTROY = "destroy"             # Destroy without specificity
    ALERT = "alert"                 # Signal other systems
    PHAGOCYTOSE = "phagocytose"     # Engulf and process
    COMPLEMENT = "complement"        # Activate complement cascade


class NKCellState(Enum):
    """Natural Killer cell states"""
    RESTING = "resting"
    SCANNING = "scanning"
    ACTIVATED = "activated"
    KILLING = "killing"
    EXHAUSTED = "exhausted"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Barrier:
    """A physical/logical barrier for protection"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    barrier_type: BarrierType = BarrierType.INPUT_VALIDATION
    strength: float = 1.0  # 0-1
    enabled: bool = True
    rules: Dict[str, Any] = field(default_factory=dict)
    blocked_count: int = 0
    passed_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class PatternRecognitionReceptor:
    """Pattern Recognition Receptor for threat detection"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    prr_type: PRRType = PRRType.TOLL_LIKE
    patterns: List[str] = field(default_factory=list)
    threat_types: Set[ThreatType] = field(default_factory=set)
    sensitivity: float = 0.7
    activation_threshold: float = 0.5
    activation_count: int = 0


@dataclass
class NaturalKillerCell:
    """Natural Killer cell for destroying compromised elements"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: NKCellState = NKCellState.RESTING
    kill_capacity: int = 10
    kills_performed: int = 0
    energy: float = 1.0
    target_types: Set[ThreatType] = field(default_factory=set)
    last_kill: Optional[float] = None


@dataclass
class InnateResponse:
    """Response from innate immune system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_type: InnateResponseType = InnateResponseType.BLOCK
    threat_id: str = ""
    responder_id: str = ""  # Barrier, PRR, or NK cell
    responder_type: str = ""
    success: bool = False
    latency_ms: float = 0.0  # Response time
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Innate Immune System
# ============================================================================

class InnateImmuneSystem:
    """
    Innate immune system for fast, non-specific defense.

    Features:
    - Multiple barrier layers
    - Pattern recognition receptors
    - Natural killer cells
    - Immediate response without learning
    - Low latency threat handling

    Example:
        innate = InnateImmuneSystem()

        # Add barriers
        innate.add_barrier("input_size", BarrierType.SIZE_LIMIT,
                           rules={"max_size": 10000})

        # Check input against barriers
        passed, responses = innate.check_barriers(input_data)

        # Scan for threats with PRRs
        threats = innate.scan_with_prrs(data)

        # Deploy NK cells if needed
        if threats:
            innate.deploy_nk_cells(threats)
    """

    def __init__(
        self,
        num_nk_cells: int = 5,
        prr_sensitivity: float = 0.7,
        response_timeout_ms: float = 100.0
    ):
        self.prr_sensitivity = prr_sensitivity
        self.response_timeout_ms = response_timeout_ms

        # Barriers
        self.barriers: Dict[str, Barrier] = {}

        # Pattern Recognition Receptors
        self.prrs: Dict[str, PatternRecognitionReceptor] = {}

        # Natural Killer cells
        self.nk_cells: Dict[str, NaturalKillerCell] = {}

        # Response history
        self.response_history: List[InnateResponse] = []

        # Statistics
        self.total_threats_blocked = 0
        self.total_threats_passed = 0
        self.average_response_time = 0.0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "barrier_triggered": [],
            "prr_activated": [],
            "nk_kill": [],
            "threat_detected": []
        }

        # Initialize defaults
        self._initialize_defaults(num_nk_cells)

    def _initialize_defaults(self, num_nk_cells: int):
        """Initialize default barriers, PRRs, and NK cells"""
        # Default barriers
        self.add_barrier("size_limit", BarrierType.SIZE_LIMIT,
                        rules={"max_size": 1000000})
        self.add_barrier("rate_limit", BarrierType.RATE_LIMIT,
                        rules={"max_per_second": 100})
        self.add_barrier("type_check", BarrierType.TYPE_CHECKING,
                        rules={"allowed_types": ["str", "int", "float", "dict", "list"]})

        # Default PRRs
        self.add_prr("tlr_corruption", PRRType.TOLL_LIKE,
                    patterns=["corrupt", "invalid", "malformed", "error"],
                    threat_types={ThreatType.CORRUPTION})
        self.add_prr("tlr_intrusion", PRRType.TOLL_LIKE,
                    patterns=["unauthorized", "access", "breach", "hack"],
                    threat_types={ThreatType.INTRUSION})
        self.add_prr("nod_malicious", PRRType.NOD_LIKE,
                    patterns=["attack", "exploit", "injection", "overflow"],
                    threat_types={ThreatType.MALICIOUS_INPUT})
        self.add_prr("rig_cascade", PRRType.RIG_LIKE,
                    patterns=["cascade", "propagate", "spread", "chain"],
                    threat_types={ThreatType.CASCADE_FAILURE})

        # Create NK cells
        for i in range(num_nk_cells):
            self._create_nk_cell(f"nk_cell_{i}")

    def add_barrier(
        self,
        name: str,
        barrier_type: BarrierType,
        strength: float = 1.0,
        rules: Optional[Dict[str, Any]] = None
    ) -> Barrier:
        """Add a protective barrier"""
        barrier = Barrier(
            name=name,
            barrier_type=barrier_type,
            strength=strength,
            rules=rules or {}
        )
        self.barriers[barrier.id] = barrier
        return barrier

    def add_prr(
        self,
        name: str,
        prr_type: PRRType,
        patterns: List[str],
        threat_types: Set[ThreatType],
        sensitivity: Optional[float] = None
    ) -> PatternRecognitionReceptor:
        """Add a Pattern Recognition Receptor"""
        prr = PatternRecognitionReceptor(
            name=name,
            prr_type=prr_type,
            patterns=patterns,
            threat_types=threat_types,
            sensitivity=sensitivity or self.prr_sensitivity
        )
        self.prrs[prr.id] = prr
        return prr

    def _create_nk_cell(self, name: str = "") -> NaturalKillerCell:
        """Create a Natural Killer cell"""
        nk = NaturalKillerCell(
            target_types=set(ThreatType)  # Can target any threat
        )
        self.nk_cells[nk.id] = nk
        return nk

    def check_barriers(
        self,
        data: Any,
        source: str = "unknown"
    ) -> tuple[bool, List[InnateResponse]]:
        """
        Check data against all barriers.

        Returns:
            (passed, responses) - Whether data passed all barriers and responses
        """
        responses = []
        all_passed = True

        for barrier in self.barriers.values():
            if not barrier.enabled:
                continue

            start_time = time.time()
            passed, details = self._check_barrier(barrier, data)
            latency = (time.time() - start_time) * 1000

            response = InnateResponse(
                response_type=InnateResponseType.BLOCK if not passed else InnateResponseType.ALERT,
                responder_id=barrier.id,
                responder_type="barrier",
                success=not passed,  # Success means blocking threat
                latency_ms=latency,
                details=details
            )
            responses.append(response)

            if passed:
                barrier.passed_count += 1
            else:
                barrier.blocked_count += 1
                all_passed = False
                self.total_threats_blocked += 1

                for callback in self._callbacks["barrier_triggered"]:
                    callback(barrier, data, details)

        if all_passed:
            self.total_threats_passed += 1

        return all_passed, responses

    def _check_barrier(
        self,
        barrier: Barrier,
        data: Any
    ) -> tuple[bool, Dict[str, Any]]:
        """Check data against a specific barrier"""
        details = {"barrier": barrier.name}

        if barrier.barrier_type == BarrierType.SIZE_LIMIT:
            max_size = barrier.rules.get("max_size", 1000000)
            data_size = len(str(data))
            details["size"] = data_size
            details["max_size"] = max_size
            return data_size <= max_size, details

        elif barrier.barrier_type == BarrierType.TYPE_CHECKING:
            allowed = barrier.rules.get("allowed_types", [])
            data_type = type(data).__name__
            details["type"] = data_type
            details["allowed"] = allowed
            return data_type in allowed or not allowed, details

        elif barrier.barrier_type == BarrierType.RATE_LIMIT:
            # Simple rate check (would need state tracking for real impl)
            max_per_second = barrier.rules.get("max_per_second", 100)
            details["limit"] = max_per_second
            return True, details  # Placeholder

        elif barrier.barrier_type == BarrierType.INPUT_VALIDATION:
            # Check for null bytes and other invalid content
            data_str = str(data)
            has_null = '\x00' in data_str
            details["has_null"] = has_null
            return not has_null, details

        return True, details

    def scan_with_prrs(
        self,
        data: Any,
        source: str = "unknown"
    ) -> List[Threat]:
        """Scan data using Pattern Recognition Receptors"""
        threats = []
        data_str = str(data).lower()

        for prr in self.prrs.values():
            activation_score = self._calculate_prr_activation(prr, data_str)

            if activation_score >= prr.activation_threshold:
                prr.activation_count += 1

                # Create threat for each matching threat type
                for threat_type in prr.threat_types:
                    threat = Threat(
                        type=threat_type,
                        severity=self._score_to_severity(activation_score),
                        source=source,
                        details={
                            "detected_by": prr.name,
                            "prr_type": prr.prr_type.value,
                            "activation_score": activation_score
                        }
                    )
                    threats.append(threat)

                for callback in self._callbacks["prr_activated"]:
                    callback(prr, activation_score)

        if threats:
            for callback in self._callbacks["threat_detected"]:
                callback(threats)

        return threats

    def _calculate_prr_activation(
        self,
        prr: PatternRecognitionReceptor,
        data: str
    ) -> float:
        """Calculate PRR activation score"""
        if not prr.patterns:
            return 0.0

        matches = 0
        for pattern in prr.patterns:
            if pattern.lower() in data:
                matches += 1

        base_score = matches / len(prr.patterns)
        return base_score * prr.sensitivity

    def _score_to_severity(self, score: float) -> ThreatSeverity:
        """Convert activation score to threat severity"""
        if score < 0.3:
            return ThreatSeverity.LOW
        elif score < 0.5:
            return ThreatSeverity.MEDIUM
        elif score < 0.8:
            return ThreatSeverity.HIGH
        return ThreatSeverity.CRITICAL

    def deploy_nk_cells(
        self,
        threats: List[Threat]
    ) -> List[InnateResponse]:
        """Deploy Natural Killer cells against threats"""
        responses = []

        # Find available NK cells
        available_nks = [
            nk for nk in self.nk_cells.values()
            if nk.state in [NKCellState.RESTING, NKCellState.SCANNING]
            and nk.energy > 0.2
            and nk.kills_performed < nk.kill_capacity
        ]

        for threat in threats:
            if not available_nks:
                break

            # Select NK cell
            nk = available_nks.pop(0)

            # Perform kill
            response = self._nk_kill(nk, threat)
            responses.append(response)

            # Check if NK is still available
            if (nk.state not in [NKCellState.EXHAUSTED] and
                nk.kills_performed < nk.kill_capacity):
                available_nks.append(nk)

        return responses

    def _nk_kill(
        self,
        nk: NaturalKillerCell,
        threat: Threat
    ) -> InnateResponse:
        """Perform NK cell kill"""
        start_time = time.time()

        nk.state = NKCellState.KILLING
        nk.kills_performed += 1
        nk.energy -= 0.1
        nk.last_kill = time.time()

        # Determine success (NK cells have high success rate)
        success = np.random.random() < 0.85

        if success:
            threat.neutralized = True
            threat.neutralized_at = time.time()

        # Update NK state
        if nk.kills_performed >= nk.kill_capacity or nk.energy < 0.1:
            nk.state = NKCellState.EXHAUSTED
        else:
            nk.state = NKCellState.SCANNING

        latency = (time.time() - start_time) * 1000

        response = InnateResponse(
            response_type=InnateResponseType.DESTROY,
            threat_id=threat.id,
            responder_id=nk.id,
            responder_type="nk_cell",
            success=success,
            latency_ms=latency,
            details={
                "kills_performed": nk.kills_performed,
                "energy_remaining": nk.energy
            }
        )

        self.response_history.append(response)

        for callback in self._callbacks["nk_kill"]:
            callback(nk, threat, success)

        return response

    def quick_response(
        self,
        threat: Threat
    ) -> InnateResponse:
        """Perform quick innate response to a threat"""
        start_time = time.time()

        # Determine response type based on threat
        response_type = self._select_response_type(threat)

        # Execute response
        success = self._execute_response(response_type, threat)

        latency = (time.time() - start_time) * 1000
        self._update_average_response_time(latency)

        response = InnateResponse(
            response_type=response_type,
            threat_id=threat.id,
            responder_type="innate_system",
            success=success,
            latency_ms=latency
        )

        self.response_history.append(response)
        return response

    def _select_response_type(self, threat: Threat) -> InnateResponseType:
        """Select appropriate response type"""
        severity_responses = {
            ThreatSeverity.LOW: InnateResponseType.ALERT,
            ThreatSeverity.MEDIUM: InnateResponseType.QUARANTINE,
            ThreatSeverity.HIGH: InnateResponseType.DESTROY,
            ThreatSeverity.CRITICAL: InnateResponseType.COMPLEMENT
        }
        return severity_responses.get(threat.severity, InnateResponseType.BLOCK)

    def _execute_response(
        self,
        response_type: InnateResponseType,
        threat: Threat
    ) -> bool:
        """Execute a response action"""
        # All innate responses have high success rate
        base_success = 0.8

        if response_type == InnateResponseType.BLOCK:
            return np.random.random() < base_success
        elif response_type == InnateResponseType.QUARANTINE:
            return np.random.random() < base_success * 0.9
        elif response_type == InnateResponseType.DESTROY:
            return np.random.random() < base_success * 0.85
        elif response_type == InnateResponseType.COMPLEMENT:
            return np.random.random() < base_success * 0.95

        return True

    def _update_average_response_time(self, latency: float):
        """Update running average of response times"""
        n = len(self.response_history)
        if n == 0:
            self.average_response_time = latency
        else:
            self.average_response_time = (
                (self.average_response_time * n + latency) / (n + 1)
            )

    def regenerate_nk_cells(self):
        """Regenerate exhausted NK cells"""
        for nk in self.nk_cells.values():
            if nk.state == NKCellState.EXHAUSTED:
                # Slowly regenerate
                nk.energy = min(1.0, nk.energy + 0.1)
                if nk.energy >= 0.5:
                    nk.state = NKCellState.RESTING
                    nk.kills_performed = max(0, nk.kills_performed - 1)

    def tick(self):
        """Process one cycle of innate immunity"""
        self.regenerate_nk_cells()

    def get_statistics(self) -> Dict[str, Any]:
        """Get innate immune system statistics"""
        return {
            "barriers": len(self.barriers),
            "prrs": len(self.prrs),
            "nk_cells": len(self.nk_cells),
            "total_blocked": self.total_threats_blocked,
            "total_passed": self.total_threats_passed,
            "average_response_ms": self.average_response_time,
            "barrier_stats": {
                b.name: {"blocked": b.blocked_count, "passed": b.passed_count}
                for b in self.barriers.values()
            },
            "prr_activations": {
                p.name: p.activation_count
                for p in self.prrs.values()
            },
            "nk_states": {
                state.value: len([nk for nk in self.nk_cells.values()
                                 if nk.state == state])
                for state in NKCellState
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Innate Immunity System Demo")
    print("=" * 50)

    # Create system
    innate = InnateImmuneSystem(num_nk_cells=3)

    print(f"\n1. Initial state:")
    stats = innate.get_statistics()
    print(f"   Barriers: {stats['barriers']}")
    print(f"   PRRs: {stats['prrs']}")
    print(f"   NK cells: {stats['nk_cells']}")

    # Test barrier checking
    print("\n2. Testing barriers...")

    # Normal data
    passed, responses = innate.check_barriers("Hello, normal data")
    print(f"   Normal data passed: {passed}")

    # Large data
    large_data = "x" * 2000000
    passed, responses = innate.check_barriers(large_data)
    print(f"   Large data passed: {passed}")
    for r in responses:
        if not r.success:
            print(f"     Blocked by: {r.details.get('barrier')}")

    # Test PRR scanning
    print("\n3. Testing PRR scanning...")
    malicious = "This contains unauthorized access breach attempt"
    threats = innate.scan_with_prrs(malicious, "test_input")
    print(f"   Threats detected: {len(threats)}")
    for threat in threats:
        print(f"     - {threat.type.value}: {threat.severity.value}")

    # Test NK cell deployment
    print("\n4. Testing NK cell deployment...")
    if threats:
        responses = innate.deploy_nk_cells(threats)
        for r in responses:
            print(f"   NK kill success: {r.success}")
            print(f"     Latency: {r.latency_ms:.2f}ms")

    # Test quick response
    print("\n5. Testing quick response...")
    test_threat = Threat(
        type=ThreatType.MALICIOUS_INPUT,
        severity=ThreatSeverity.HIGH
    )
    response = innate.quick_response(test_threat)
    print(f"   Response: {response.response_type.value}")
    print(f"   Success: {response.success}")
    print(f"   Latency: {response.latency_ms:.2f}ms")

    # Final stats
    print("\n6. Final statistics:")
    stats = innate.get_statistics()
    print(f"   Total blocked: {stats['total_blocked']}")
    print(f"   Total passed: {stats['total_passed']}")
    print(f"   Avg response: {stats['average_response_ms']:.2f}ms")
