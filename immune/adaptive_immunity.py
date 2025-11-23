"""
Adaptive Immunity System

Learning-based threat response system that improves over time.
Coordinates the full adaptive immune response:
- T-cell like coordination and help
- B-cell like antibody production
- Memory formation
- Response orchestration
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from immune.defense import Threat, ThreatType, ThreatSeverity
from immune.antibody_system import (
    AntibodyGenerator, Antibody, AntibodyResponse,
    Antigen, ResponseType
)


# ============================================================================
# Enums
# ============================================================================

class TCellType(Enum):
    """Types of T-cells"""
    HELPER = "helper"           # Coordinate immune response
    CYTOTOXIC = "cytotoxic"     # Directly kill infected cells
    REGULATORY = "regulatory"   # Suppress immune response
    MEMORY = "memory"           # Long-term memory


class BCellType(Enum):
    """Types of B-cells"""
    NAIVE = "naive"
    ACTIVATED = "activated"
    PLASMA = "plasma"           # Antibody producing
    MEMORY = "memory"


class ImmunePhase(Enum):
    """Phases of adaptive immune response"""
    RECOGNITION = "recognition"
    ACTIVATION = "activation"
    CLONAL_EXPANSION = "clonal_expansion"
    EFFECTOR = "effector"
    CONTRACTION = "contraction"
    MEMORY = "memory"


class ResponseStrength(Enum):
    """Strength of immune response"""
    MINIMAL = 1
    MODERATE = 2
    STRONG = 3
    OVERWHELMING = 4


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TCell:
    """T-cell for immune coordination"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cell_type: TCellType = TCellType.HELPER
    specificity: Set[ThreatType] = field(default_factory=set)
    activation_level: float = 0.0
    memory_strength: float = 0.0
    cytokines: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_activated: Optional[float] = None


@dataclass
class BCell:
    """B-cell for antibody production"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cell_type: BCellType = BCellType.NAIVE
    antibody_id: Optional[str] = None
    specificity: Set[ThreatType] = field(default_factory=set)
    activation_level: float = 0.0
    production_rate: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class AdaptiveResponse:
    """Complete adaptive immune response"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_id: str = ""
    phase: ImmunePhase = ImmunePhase.RECOGNITION
    strength: ResponseStrength = ResponseStrength.MODERATE
    t_cells_involved: List[str] = field(default_factory=list)
    b_cells_involved: List[str] = field(default_factory=list)
    antibodies_produced: List[str] = field(default_factory=list)
    effectiveness: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    memory_formed: bool = False


@dataclass
class ImmuneMemory:
    """Long-term immune memory"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_signature: str = ""
    threat_type: ThreatType = ThreatType.ANOMALY
    response_pattern: Dict[str, Any] = field(default_factory=dict)
    effectiveness_history: List[float] = field(default_factory=list)
    last_encounter: float = field(default_factory=time.time)
    encounter_count: int = 1
    decay_rate: float = 0.01


# ============================================================================
# Adaptive Immune System
# ============================================================================

class AdaptiveImmuneSystem:
    """
    Adaptive immune system with learning and memory.

    Features:
    - T-cell coordination
    - B-cell antibody production
    - Response phasing (recognition -> memory)
    - Affinity maturation
    - Immune memory formation

    Example:
        adaptive = AdaptiveImmuneSystem()

        # Mount response to threat
        response = adaptive.mount_response(threat)

        # Advance response through phases
        while response.phase != ImmunePhase.MEMORY:
            adaptive.advance_response(response)

        # Future encounters use memory
        cached = adaptive.check_memory(similar_threat)
    """

    def __init__(
        self,
        initial_t_cells: int = 20,
        initial_b_cells: int = 20,
        memory_retention: float = 0.95,
        learning_rate: float = 0.1
    ):
        self.memory_retention = memory_retention
        self.learning_rate = learning_rate

        # Antibody generator
        self.antibody_generator = AntibodyGenerator()

        # Cell populations
        self.t_cells: Dict[str, TCell] = {}
        self.b_cells: Dict[str, BCell] = {}

        # Active responses
        self.active_responses: Dict[str, AdaptiveResponse] = {}
        self.response_history: List[AdaptiveResponse] = []

        # Immune memory
        self.immune_memory: Dict[str, ImmuneMemory] = {}

        # Cytokine environment
        self.cytokines: Dict[str, float] = defaultdict(float)

        # Statistics
        self.total_responses = 0
        self.successful_responses = 0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "response_started": [],
            "phase_changed": [],
            "memory_formed": [],
            "response_complete": []
        }

        # Initialize cells
        self._initialize_cells(initial_t_cells, initial_b_cells)

    def _initialize_cells(self, num_t: int, num_b: int):
        """Initialize immune cell populations"""
        # Create T cells
        for i in range(num_t):
            t_type = TCellType.HELPER if i % 3 != 0 else TCellType.CYTOTOXIC
            t_cell = TCell(
                cell_type=t_type,
                specificity=set(list(ThreatType)[:((i % 4) + 1)])
            )
            self.t_cells[t_cell.id] = t_cell

        # Create B cells
        for i in range(num_b):
            b_cell = BCell(
                specificity=set(list(ThreatType)[:((i % 3) + 1)])
            )
            self.b_cells[b_cell.id] = b_cell

    def check_memory(
        self,
        threat: Threat
    ) -> Optional[ImmuneMemory]:
        """Check if we have memory of this threat type"""
        for memory in self.immune_memory.values():
            if memory.threat_type == threat.type:
                # Check signature similarity
                if self._signature_match(threat.signature, memory.threat_signature):
                    memory.encounter_count += 1
                    memory.last_encounter = time.time()
                    return memory
        return None

    def _signature_match(self, sig1: str, sig2: str) -> bool:
        """Check if two signatures match sufficiently"""
        if not sig1 or not sig2:
            return False

        # Simple similarity check
        common = sum(1 for c1, c2 in zip(sig1, sig2) if c1 == c2)
        max_len = max(len(sig1), len(sig2))
        return common / max_len > 0.5 if max_len > 0 else False

    def mount_response(
        self,
        threat: Threat,
        strength: Optional[ResponseStrength] = None
    ) -> AdaptiveResponse:
        """Mount an adaptive immune response to a threat"""
        self.total_responses += 1

        # Check for memory
        memory = self.check_memory(threat)
        if memory:
            # Faster, stronger response with memory
            strength = strength or ResponseStrength.STRONG
            initial_effectiveness = np.mean(memory.effectiveness_history) if memory.effectiveness_history else 0.5
        else:
            strength = strength or self._determine_strength(threat)
            initial_effectiveness = 0.0

        # Create response
        response = AdaptiveResponse(
            threat_id=threat.id,
            phase=ImmunePhase.RECOGNITION,
            strength=strength,
            effectiveness=initial_effectiveness
        )

        self.active_responses[response.id] = response

        # Trigger callback
        for callback in self._callbacks["response_started"]:
            callback(response, threat)

        # Start recognition phase
        self._recognition_phase(response, threat)

        return response

    def _determine_strength(self, threat: Threat) -> ResponseStrength:
        """Determine response strength based on threat"""
        severity_map = {
            ThreatSeverity.LOW: ResponseStrength.MINIMAL,
            ThreatSeverity.MEDIUM: ResponseStrength.MODERATE,
            ThreatSeverity.HIGH: ResponseStrength.STRONG,
            ThreatSeverity.CRITICAL: ResponseStrength.OVERWHELMING
        }
        return severity_map.get(threat.severity, ResponseStrength.MODERATE)

    def _recognition_phase(
        self,
        response: AdaptiveResponse,
        threat: Threat
    ):
        """Execute recognition phase"""
        # Find matching T cells
        matching_t = [
            t for t in self.t_cells.values()
            if threat.type in t.specificity
            and t.cell_type == TCellType.HELPER
        ]

        # Activate top T cells
        num_to_activate = min(len(matching_t), response.strength.value * 2)
        for t_cell in matching_t[:num_to_activate]:
            t_cell.activation_level = 0.3
            t_cell.last_activated = time.time()
            response.t_cells_involved.append(t_cell.id)

        # Release cytokines
        self.cytokines["IL-2"] += 0.1 * num_to_activate
        self.cytokines["IFN-gamma"] += 0.05 * num_to_activate

    def advance_response(
        self,
        response: AdaptiveResponse,
        threat: Optional[Threat] = None
    ) -> bool:
        """
        Advance response to next phase.

        Returns True if response is complete.
        """
        phase_handlers = {
            ImmunePhase.RECOGNITION: self._to_activation,
            ImmunePhase.ACTIVATION: self._to_clonal_expansion,
            ImmunePhase.CLONAL_EXPANSION: self._to_effector,
            ImmunePhase.EFFECTOR: self._to_contraction,
            ImmunePhase.CONTRACTION: self._to_memory,
            ImmunePhase.MEMORY: lambda r, t: True
        }

        handler = phase_handlers.get(response.phase)
        if handler:
            complete = handler(response, threat)

            if response.phase != ImmunePhase.MEMORY:
                for callback in self._callbacks["phase_changed"]:
                    callback(response)

            return complete

        return False

    def _to_activation(
        self,
        response: AdaptiveResponse,
        threat: Optional[Threat]
    ) -> bool:
        """Transition to activation phase"""
        response.phase = ImmunePhase.ACTIVATION

        # Activate B cells
        matching_b = [
            b for b in self.b_cells.values()
            if not b.antibody_id
        ][:response.strength.value * 3]

        for b_cell in matching_b:
            b_cell.cell_type = BCellType.ACTIVATED
            b_cell.activation_level = 0.5
            response.b_cells_involved.append(b_cell.id)

        # Increase T cell activation
        for t_id in response.t_cells_involved:
            if t_id in self.t_cells:
                self.t_cells[t_id].activation_level = 0.7

        return False

    def _to_clonal_expansion(
        self,
        response: AdaptiveResponse,
        threat: Optional[Threat]
    ) -> bool:
        """Transition to clonal expansion phase"""
        response.phase = ImmunePhase.CLONAL_EXPANSION

        # Generate antibodies
        if threat:
            for _ in range(response.strength.value * 2):
                antibody = self.antibody_generator.generate_antibody(threat, force_new=True)
                response.antibodies_produced.append(antibody.id)

        # Clone B cells (simplified)
        for b_id in response.b_cells_involved.copy():
            if b_id in self.b_cells:
                b_cell = self.b_cells[b_id]
                b_cell.cell_type = BCellType.PLASMA
                b_cell.production_rate = 0.8

        # Update effectiveness
        response.effectiveness = min(1.0, response.effectiveness + 0.2)

        return False

    def _to_effector(
        self,
        response: AdaptiveResponse,
        threat: Optional[Threat]
    ) -> bool:
        """Transition to effector phase"""
        response.phase = ImmunePhase.EFFECTOR

        # Maximum antibody production
        for b_id in response.b_cells_involved:
            if b_id in self.b_cells:
                self.b_cells[b_id].production_rate = 1.0

        # Activate cytotoxic T cells
        for t_id in response.t_cells_involved:
            if t_id in self.t_cells:
                t_cell = self.t_cells[t_id]
                if t_cell.cell_type == TCellType.HELPER:
                    t_cell.activation_level = 1.0
                    # Release helper cytokines
                    t_cell.cytokines["IL-4"] = 0.8
                    t_cell.cytokines["IL-5"] = 0.6

        # Attack threat if present
        if threat and not threat.neutralized:
            success_chance = response.effectiveness + 0.3
            if np.random.random() < success_chance:
                threat.neutralized = True
                threat.neutralized_at = time.time()
                response.effectiveness = min(1.0, response.effectiveness + 0.3)
                self.successful_responses += 1

        return False

    def _to_contraction(
        self,
        response: AdaptiveResponse,
        threat: Optional[Threat]
    ) -> bool:
        """Transition to contraction phase"""
        response.phase = ImmunePhase.CONTRACTION

        # Reduce cytokines
        for key in list(self.cytokines.keys()):
            self.cytokines[key] *= 0.5

        # Deactivate most cells, keep some as memory precursors
        for t_id in response.t_cells_involved:
            if t_id in self.t_cells:
                t_cell = self.t_cells[t_id]
                if np.random.random() < 0.1:  # 10% become memory
                    t_cell.cell_type = TCellType.MEMORY
                    t_cell.memory_strength = response.effectiveness
                else:
                    t_cell.activation_level *= 0.3

        for b_id in response.b_cells_involved:
            if b_id in self.b_cells:
                b_cell = self.b_cells[b_id]
                if np.random.random() < 0.1:
                    b_cell.cell_type = BCellType.MEMORY
                else:
                    b_cell.cell_type = BCellType.NAIVE
                    b_cell.production_rate = 0.0

        return False

    def _to_memory(
        self,
        response: AdaptiveResponse,
        threat: Optional[Threat]
    ) -> bool:
        """Transition to memory phase"""
        response.phase = ImmunePhase.MEMORY
        response.completed_at = time.time()

        # Form immune memory
        if threat and response.effectiveness > 0.5:
            memory = ImmuneMemory(
                threat_signature=threat.signature,
                threat_type=threat.type,
                response_pattern={
                    "strength": response.strength.value,
                    "antibodies": response.antibodies_produced[:5],
                    "t_cell_types": [self.t_cells[t_id].cell_type.value
                                    for t_id in response.t_cells_involved
                                    if t_id in self.t_cells]
                },
                effectiveness_history=[response.effectiveness]
            )
            self.immune_memory[memory.id] = memory
            response.memory_formed = True

            for callback in self._callbacks["memory_formed"]:
                callback(memory)

        # Move to history
        if response.id in self.active_responses:
            del self.active_responses[response.id]
        self.response_history.append(response)

        for callback in self._callbacks["response_complete"]:
            callback(response)

        return True

    def run_full_response(
        self,
        threat: Threat
    ) -> AdaptiveResponse:
        """Run complete adaptive response through all phases"""
        response = self.mount_response(threat)

        while response.phase != ImmunePhase.MEMORY:
            self.advance_response(response, threat)

        return response

    def decay_memory(self):
        """Apply decay to immune memory"""
        to_remove = []

        for mem_id, memory in self.immune_memory.items():
            # Calculate time since last encounter
            age = time.time() - memory.last_encounter
            decay = age * memory.decay_rate

            # Remove if decayed significantly
            if decay > 0.9 or np.random.random() < memory.decay_rate:
                to_remove.append(mem_id)

        for mem_id in to_remove:
            del self.immune_memory[mem_id]

    def get_memory_recall(
        self,
        threat_type: ThreatType
    ) -> List[ImmuneMemory]:
        """Get all memories for a threat type"""
        return [
            mem for mem in self.immune_memory.values()
            if mem.threat_type == threat_type
        ]

    def tick(self):
        """Process one cycle"""
        # Decay cytokines
        for key in list(self.cytokines.keys()):
            self.cytokines[key] *= 0.95
            if self.cytokines[key] < 0.01:
                del self.cytokines[key]

        # Occasional memory decay
        if np.random.random() < 0.01:
            self.decay_memory()

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive immune statistics"""
        return {
            "total_responses": self.total_responses,
            "successful_responses": self.successful_responses,
            "success_rate": self.successful_responses / max(1, self.total_responses),
            "active_responses": len(self.active_responses),
            "immune_memories": len(self.immune_memory),
            "t_cells": len(self.t_cells),
            "b_cells": len(self.b_cells),
            "antibodies": len(self.antibody_generator.antibodies),
            "t_cell_types": {
                tt.value: len([t for t in self.t_cells.values() if t.cell_type == tt])
                for tt in TCellType
            },
            "b_cell_types": {
                bt.value: len([b for b in self.b_cells.values() if b.cell_type == bt])
                for bt in BCellType
            },
            "cytokine_levels": dict(self.cytokines)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Adaptive Immunity System Demo")
    print("=" * 50)

    # Create system
    adaptive = AdaptiveImmuneSystem()

    print(f"\n1. Initial state:")
    stats = adaptive.get_statistics()
    print(f"   T cells: {stats['t_cells']}")
    print(f"   B cells: {stats['b_cells']}")
    print(f"   Memories: {stats['immune_memories']}")

    # Create threat
    threat = Threat(
        type=ThreatType.MALICIOUS_INPUT,
        severity=ThreatSeverity.HIGH,
        signature="malicious_payload_001",
        details={"source": "external_api"}
    )

    print("\n2. Mounting primary response...")
    response = adaptive.run_full_response(threat)
    print(f"   Phases completed: {response.phase.value}")
    print(f"   Effectiveness: {response.effectiveness:.2f}")
    print(f"   Memory formed: {response.memory_formed}")
    print(f"   T cells involved: {len(response.t_cells_involved)}")
    print(f"   Antibodies produced: {len(response.antibodies_produced)}")

    # Second exposure (should be faster/stronger)
    print("\n3. Secondary exposure (same threat type)...")
    threat2 = Threat(
        type=ThreatType.MALICIOUS_INPUT,
        severity=ThreatSeverity.HIGH,
        signature="malicious_payload_002"
    )

    memory = adaptive.check_memory(threat2)
    if memory:
        print(f"   Memory found! Previous encounters: {memory.encounter_count}")

    response2 = adaptive.run_full_response(threat2)
    print(f"   Effectiveness: {response2.effectiveness:.2f}")
    print(f"   Response strength: {response2.strength.value}")

    # Final statistics
    print("\n4. Final statistics:")
    stats = adaptive.get_statistics()
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")
    print(f"   Immune memories: {stats['immune_memories']}")
    print(f"   Active antibodies: {stats['antibodies']}")
