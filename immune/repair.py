"""
Self-Repair System

Enables digital organisms to heal from damage and corruption.
Implements biological-inspired repair mechanisms including:
- Damage detection
- Redundancy-based repair
- Regeneration
- Integrity verification
- Regeneration cascades
- Healing factor management
- Tissue regeneration simulation
"""

import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Enums
# ============================================================================

class DamageType(Enum):
    """Types of damage"""
    CORRUPTION = "corruption"       # Data corruption
    DEGRADATION = "degradation"     # Gradual wear
    TRAUMA = "trauma"               # Sudden damage
    OVERLOAD = "overload"           # Capacity exceeded
    DISCONNECTION = "disconnection" # Lost connections
    INCONSISTENCY = "inconsistency" # Internal contradictions


class RepairStrategy(Enum):
    """Strategies for repair"""
    REDUNDANCY = "redundancy"       # Use backup copies
    REGENERATION = "regeneration"   # Regrow from template
    ISOLATION = "isolation"         # Cut off damaged area
    COMPENSATION = "compensation"   # Work around damage
    REPLACEMENT = "replacement"     # Replace damaged component
    HEALING = "healing"             # Gradual recovery


class HealingPhase(Enum):
    """Phases of healing"""
    DETECTION = "detection"
    ASSESSMENT = "assessment"
    STABILIZATION = "stabilization"
    REPAIR = "repair"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    COMPLETE = "complete"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Damage:
    """Damage to the organism"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: DamageType = DamageType.CORRUPTION
    location: str = ""
    severity: float = 0.0  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)
    repaired: bool = False
    repaired_at: Optional[float] = None


@dataclass
class IntegrityCheck:
    """Result of an integrity check"""
    target: str
    passed: bool
    expected_hash: str = ""
    actual_hash: str = ""
    discrepancies: List[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)


@dataclass
class HealingProcess:
    """An active healing process"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    damage_id: str = ""
    strategy: RepairStrategy = RepairStrategy.HEALING
    phase: HealingPhase = HealingPhase.DETECTION
    progress: float = 0.0
    started_at: float = field(default_factory=time.time)
    estimated_duration: float = 10.0
    success_probability: float = 0.9


@dataclass
class BackupState:
    """Backup of organism state for redundancy repair"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    state_hash: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    valid: bool = True


# ============================================================================
# Repair System
# ============================================================================

class RepairSystem:
    """
    Self-repair system for digital organisms.

    Features:
    - Damage detection and assessment
    - Multiple repair strategies
    - Integrity verification
    - Backup management
    - Gradual healing

    Example:
        repair = RepairSystem()

        # Create backup
        repair.create_backup("current_state", organism_state)

        # Check integrity
        check = repair.check_integrity(current_data)

        # Repair damage
        healing = repair.initiate_repair(damage)
        while not healing.phase == HealingPhase.COMPLETE:
            repair.advance_healing(healing)
    """

    def __init__(
        self,
        healing_rate: float = 0.1,
        auto_backup_interval: int = 100
    ):
        self.healing_rate = healing_rate
        self.auto_backup_interval = auto_backup_interval

        # Damage tracking
        self.active_damage: Dict[str, Damage] = {}
        self.damage_history: List[Damage] = []

        # Healing processes
        self.active_healing: Dict[str, HealingProcess] = {}
        self.healing_history: List[HealingProcess] = []

        # Backups for redundancy
        self.backups: Dict[str, BackupState] = {}
        self.max_backups = 10

        # Integrity checksums
        self.checksums: Dict[str, str] = {}

        # Statistics
        self.cycle_count = 0
        self.total_damage_repaired = 0
        self.repair_success_rate = 1.0

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "damage_detected": [],
            "repair_complete": [],
            "repair_failed": []
        }

    def register_checksum(self, target: str, data: Any):
        """Register checksum for integrity verification"""
        data_str = str(data)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()
        self.checksums[target] = checksum

    def check_integrity(
        self,
        target: str,
        current_data: Any
    ) -> IntegrityCheck:
        """
        Check integrity of data against stored checksum.

        Returns IntegrityCheck result.
        """
        current_str = str(current_data)
        actual_hash = hashlib.sha256(current_str.encode()).hexdigest()
        expected_hash = self.checksums.get(target, "")

        passed = (actual_hash == expected_hash) if expected_hash else True

        discrepancies = []
        if not passed:
            discrepancies.append(f"Hash mismatch for {target}")

        return IntegrityCheck(
            target=target,
            passed=passed,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            discrepancies=discrepancies
        )

    def detect_damage(
        self,
        location: str,
        severity: float,
        damage_type: DamageType = DamageType.CORRUPTION,
        details: Optional[Dict[str, Any]] = None
    ) -> Damage:
        """Detect and register damage"""
        damage = Damage(
            type=damage_type,
            location=location,
            severity=severity,
            details=details or {}
        )

        self.active_damage[damage.id] = damage
        self.damage_history.append(damage)

        # Trigger callbacks
        for callback in self._callbacks["damage_detected"]:
            callback(damage)

        return damage

    def assess_damage(self, damage: Damage) -> Dict[str, Any]:
        """Assess damage and determine repair approach"""
        assessment = {
            "severity_level": self._categorize_severity(damage.severity),
            "recommended_strategy": self._recommend_strategy(damage),
            "estimated_repair_time": self._estimate_repair_time(damage),
            "success_probability": self._estimate_success(damage),
            "resources_required": self._estimate_resources(damage)
        }

        return assessment

    def _categorize_severity(self, severity: float) -> str:
        """Categorize severity level"""
        if severity < 0.25:
            return "minor"
        elif severity < 0.5:
            return "moderate"
        elif severity < 0.75:
            return "severe"
        else:
            return "critical"

    def _recommend_strategy(self, damage: Damage) -> RepairStrategy:
        """Recommend repair strategy"""
        if damage.type == DamageType.CORRUPTION:
            if damage.location in self.backups:
                return RepairStrategy.REDUNDANCY
            return RepairStrategy.REGENERATION

        elif damage.type == DamageType.DEGRADATION:
            return RepairStrategy.HEALING

        elif damage.type == DamageType.TRAUMA:
            if damage.severity > 0.7:
                return RepairStrategy.ISOLATION
            return RepairStrategy.REPLACEMENT

        elif damage.type == DamageType.OVERLOAD:
            return RepairStrategy.COMPENSATION

        elif damage.type == DamageType.DISCONNECTION:
            return RepairStrategy.REGENERATION

        return RepairStrategy.HEALING

    def _estimate_repair_time(self, damage: Damage) -> float:
        """Estimate time to repair"""
        base_time = 10.0
        severity_multiplier = 1 + damage.severity * 3

        type_multipliers = {
            DamageType.CORRUPTION: 1.5,
            DamageType.DEGRADATION: 0.5,
            DamageType.TRAUMA: 2.0,
            DamageType.OVERLOAD: 0.8,
            DamageType.DISCONNECTION: 1.2,
            DamageType.INCONSISTENCY: 1.0
        }

        type_mult = type_multipliers.get(damage.type, 1.0)

        return base_time * severity_multiplier * type_mult / self.healing_rate

    def _estimate_success(self, damage: Damage) -> float:
        """Estimate probability of successful repair"""
        base_prob = 0.95

        # Reduce probability with severity
        severity_factor = 1 - damage.severity * 0.3

        # Backup availability helps
        backup_factor = 1.1 if damage.location in self.backups else 1.0

        return min(1.0, base_prob * severity_factor * backup_factor)

    def _estimate_resources(self, damage: Damage) -> Dict[str, float]:
        """Estimate resources required for repair"""
        base_energy = 10.0 * (1 + damage.severity)

        return {
            "energy": base_energy,
            "compute": base_energy * 0.5,
            "memory": base_energy * 0.2
        }

    def initiate_repair(
        self,
        damage: Damage,
        strategy: Optional[RepairStrategy] = None
    ) -> HealingProcess:
        """Initiate repair process for damage"""
        if strategy is None:
            strategy = self._recommend_strategy(damage)

        healing = HealingProcess(
            damage_id=damage.id,
            strategy=strategy,
            phase=HealingPhase.DETECTION,
            estimated_duration=self._estimate_repair_time(damage),
            success_probability=self._estimate_success(damage)
        )

        self.active_healing[healing.id] = healing

        return healing

    def advance_healing(self, healing: HealingProcess) -> bool:
        """
        Advance healing process by one step.

        Returns True if healing is complete.
        """
        phases = list(HealingPhase)
        current_idx = phases.index(healing.phase)

        if healing.phase == HealingPhase.COMPLETE:
            return True

        # Progress within current phase
        healing.progress += self.healing_rate

        if healing.progress >= 1.0:
            # Move to next phase
            if current_idx < len(phases) - 1:
                healing.phase = phases[current_idx + 1]
                healing.progress = 0.0

                # Special handling for verification phase
                if healing.phase == HealingPhase.VERIFICATION:
                    if not self._verify_repair(healing):
                        healing.phase = HealingPhase.REPAIR
                        healing.success_probability *= 0.9
            else:
                healing.phase = HealingPhase.COMPLETE

        # Check if complete
        if healing.phase == HealingPhase.COMPLETE:
            self._complete_healing(healing)
            return True

        return False

    def _verify_repair(self, healing: HealingProcess) -> bool:
        """Verify repair was successful"""
        # Probabilistic success
        return np.random.random() < healing.success_probability

    def _complete_healing(self, healing: HealingProcess):
        """Complete a healing process"""
        # Mark damage as repaired
        if healing.damage_id in self.active_damage:
            damage = self.active_damage[healing.damage_id]
            damage.repaired = True
            damage.repaired_at = time.time()
            del self.active_damage[healing.damage_id]

        # Move healing to history
        if healing.id in self.active_healing:
            self.healing_history.append(healing)
            del self.active_healing[healing.id]

        self.total_damage_repaired += 1

        # Update success rate
        self._update_success_rate()

        # Trigger callbacks
        for callback in self._callbacks["repair_complete"]:
            callback(healing)

    def _update_success_rate(self):
        """Update repair success rate"""
        if self.healing_history:
            successful = len([h for h in self.healing_history if h.phase == HealingPhase.COMPLETE])
            self.repair_success_rate = successful / len(self.healing_history)

    def create_backup(
        self,
        name: str,
        state: Dict[str, Any]
    ) -> BackupState:
        """Create backup for redundancy repair"""
        state_str = str(state)
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()

        backup = BackupState(
            name=name,
            state_hash=state_hash,
            data=state.copy()
        )

        self.backups[name] = backup

        # Limit backup count
        while len(self.backups) > self.max_backups:
            oldest = min(self.backups.values(), key=lambda b: b.created_at)
            del self.backups[oldest.name]

        return backup

    def restore_from_backup(
        self,
        name: str
    ) -> Optional[Dict[str, Any]]:
        """Restore state from backup"""
        backup = self.backups.get(name)
        if backup and backup.valid:
            return backup.data.copy()
        return None

    def tick(self):
        """Process one repair cycle"""
        self.cycle_count += 1

        # Advance all active healing processes
        completed = []
        for healing_id, healing in list(self.active_healing.items()):
            if self.advance_healing(healing):
                completed.append(healing_id)

        # Auto backup
        if self.cycle_count % self.auto_backup_interval == 0:
            # Would trigger backup in real implementation
            pass

    def on_damage_detected(self, callback: Callable[[Damage], None]):
        """Register callback for damage detection"""
        self._callbacks["damage_detected"].append(callback)

    def on_repair_complete(self, callback: Callable[[HealingProcess], None]):
        """Register callback for repair completion"""
        self._callbacks["repair_complete"].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get repair system status"""
        return {
            "active_damage": len(self.active_damage),
            "active_healing": len(self.active_healing),
            "total_repaired": self.total_damage_repaired,
            "success_rate": self.repair_success_rate,
            "backups": len(self.backups),
            "healing_rate": self.healing_rate,
            "damage_by_type": {
                dt.value: len([d for d in self.damage_history if d.type == dt])
                for dt in DamageType
            }
        }


# ============================================================================
# Regeneration Cascade
# ============================================================================

class RegenerationCascade:
    """
    Cascading regeneration system for coordinated healing.

    Features:
    - Multi-stage regeneration
    - Healing factor management
    - Resource-aware regeneration
    - Cascade propagation
    """

    def __init__(
        self,
        healing_factor: float = 1.0,
        max_cascade_depth: int = 5,
        energy_per_stage: float = 10.0
    ):
        self.healing_factor = healing_factor
        self.max_cascade_depth = max_cascade_depth
        self.energy_per_stage = energy_per_stage

        # Active cascades
        self.active_cascades: Dict[str, 'HealingCascade'] = {}

        # Resource pool
        self.energy_pool: float = 100.0
        self.max_energy: float = 100.0

        # Statistics
        self.cascades_completed = 0
        self.total_regenerated = 0.0

    def initiate_cascade(
        self,
        damage: Damage,
        repair_system: RepairSystem
    ) -> 'HealingCascade':
        """Initiate a regeneration cascade"""
        cascade = HealingCascade(
            damage_id=damage.id,
            origin_location=damage.location,
            severity=damage.severity,
            max_depth=self.max_cascade_depth
        )

        self.active_cascades[cascade.id] = cascade

        # Start first stage
        self._execute_stage(cascade, repair_system)

        return cascade

    def _execute_stage(
        self,
        cascade: 'HealingCascade',
        repair_system: RepairSystem
    ) -> bool:
        """Execute one stage of the cascade"""
        if cascade.current_depth >= cascade.max_depth:
            cascade.complete = True
            return True

        # Check energy
        required_energy = self.energy_per_stage * (cascade.current_depth + 1)
        if self.energy_pool < required_energy:
            cascade.blocked_by_resources = True
            return False

        # Consume energy
        self.energy_pool -= required_energy

        # Calculate healing amount
        stage_healing = (
            self.healing_factor *
            (1.0 - cascade.current_depth * 0.15) *
            cascade.severity
        )

        cascade.healed_amount += stage_healing
        cascade.current_depth += 1
        cascade.stages_completed.append({
            'depth': cascade.current_depth,
            'healed': stage_healing,
            'energy_used': required_energy,
            'timestamp': time.time()
        })

        self.total_regenerated += stage_healing

        return False

    def advance_cascades(self, repair_system: RepairSystem):
        """Advance all active cascades"""
        completed = []

        for cascade_id, cascade in self.active_cascades.items():
            if cascade.complete:
                completed.append(cascade_id)
                continue

            if self._execute_stage(cascade, repair_system):
                completed.append(cascade_id)
                self.cascades_completed += 1

        for cascade_id in completed:
            if cascade_id in self.active_cascades:
                del self.active_cascades[cascade_id]

    def regenerate_energy(self, amount: float = 5.0):
        """Regenerate energy pool"""
        self.energy_pool = min(self.max_energy, self.energy_pool + amount)

    def get_status(self) -> Dict[str, Any]:
        """Get cascade status"""
        return {
            'active_cascades': len(self.active_cascades),
            'cascades_completed': self.cascades_completed,
            'total_regenerated': self.total_regenerated,
            'energy_pool': self.energy_pool,
            'healing_factor': self.healing_factor
        }


@dataclass
class HealingCascade:
    """A single healing cascade instance"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    damage_id: str = ""
    origin_location: str = ""
    severity: float = 0.5
    max_depth: int = 5
    current_depth: int = 0
    healed_amount: float = 0.0
    stages_completed: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    complete: bool = False
    blocked_by_resources: bool = False


# ============================================================================
# Tissue Regeneration
# ============================================================================

class TissueRegenerator:
    """
    Tissue-level regeneration for complex damage.

    Simulates biological tissue regeneration:
    - Stem cell activation
    - Cell proliferation
    - Differentiation
    - Tissue remodeling
    """

    def __init__(self):
        self.stem_cell_pool: float = 1.0  # Available stem cells (0-1)
        self.proliferation_rate: float = 0.1
        self.differentiation_rate: float = 0.15
        self.remodeling_rate: float = 0.05

        # Regeneration states
        self.regenerating_tissues: Dict[str, 'TissueState'] = {}

    def begin_regeneration(
        self,
        tissue_id: str,
        damage_extent: float
    ) -> 'TissueState':
        """Begin tissue regeneration"""
        state = TissueState(
            tissue_id=tissue_id,
            damage_extent=damage_extent,
            stem_cells_allocated=min(self.stem_cell_pool, damage_extent * 0.5)
        )

        self.stem_cell_pool -= state.stem_cells_allocated
        self.regenerating_tissues[tissue_id] = state

        return state

    def advance_regeneration(self, tissue_id: str) -> Optional['TissueState']:
        """Advance regeneration of a tissue"""
        state = self.regenerating_tissues.get(tissue_id)
        if not state:
            return None

        if state.phase == 'activation':
            state.progress += self.proliferation_rate
            if state.progress >= 1.0:
                state.phase = 'proliferation'
                state.progress = 0.0

        elif state.phase == 'proliferation':
            state.progress += self.proliferation_rate * state.stem_cells_allocated
            state.cells_generated += state.stem_cells_allocated * 0.1
            if state.progress >= 1.0:
                state.phase = 'differentiation'
                state.progress = 0.0

        elif state.phase == 'differentiation':
            state.progress += self.differentiation_rate
            state.tissue_formed = state.cells_generated * state.progress
            if state.progress >= 1.0:
                state.phase = 'remodeling'
                state.progress = 0.0

        elif state.phase == 'remodeling':
            state.progress += self.remodeling_rate
            if state.progress >= 1.0:
                state.phase = 'complete'
                state.complete = True
                # Return stem cells
                self.stem_cell_pool = min(1.0, self.stem_cell_pool + state.stem_cells_allocated * 0.5)
                del self.regenerating_tissues[tissue_id]

        return state

    def regenerate_stem_cells(self, amount: float = 0.05):
        """Regenerate stem cell pool"""
        self.stem_cell_pool = min(1.0, self.stem_cell_pool + amount)


@dataclass
class TissueState:
    """State of tissue regeneration"""
    tissue_id: str = ""
    damage_extent: float = 0.0
    stem_cells_allocated: float = 0.0
    phase: str = "activation"
    progress: float = 0.0
    cells_generated: float = 0.0
    tissue_formed: float = 0.0
    complete: bool = False
    started_at: float = field(default_factory=time.time)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Self-Repair System Demo")
    print("=" * 50)

    # Create repair system
    repair = RepairSystem(healing_rate=0.2)

    print(f"\n1. Initial state:")
    print(f"   Healing rate: {repair.healing_rate}")

    # Create backup
    print("\n2. Creating backup...")
    state = {"consciousness": 0.7, "memory": ["a", "b", "c"]}
    backup = repair.create_backup("main_state", state)
    print(f"   Backup created: {backup.name}")
    print(f"   Hash: {backup.state_hash[:16]}...")

    # Register checksum
    repair.register_checksum("main_state", state)

    # Check integrity (should pass)
    print("\n3. Checking integrity (unchanged)...")
    check = repair.check_integrity("main_state", state)
    print(f"   Passed: {check.passed}")

    # Corrupt and check
    print("\n4. Checking integrity (corrupted)...")
    corrupted_state = {"consciousness": 0.3, "memory": ["x"]}
    check = repair.check_integrity("main_state", corrupted_state)
    print(f"   Passed: {check.passed}")
    print(f"   Discrepancies: {check.discrepancies}")

    # Detect damage
    print("\n5. Detecting and repairing damage...")
    damage = repair.detect_damage(
        location="main_state",
        severity=0.4,
        damage_type=DamageType.CORRUPTION
    )
    print(f"   Damage detected: {damage.type.value}, severity={damage.severity}")

    # Assess damage
    assessment = repair.assess_damage(damage)
    print(f"   Assessment: {assessment['severity_level']}")
    print(f"   Strategy: {assessment['recommended_strategy'].value}")
    print(f"   Success probability: {assessment['success_probability']:.2%}")

    # Initiate repair
    healing = repair.initiate_repair(damage)
    print(f"\n   Starting repair process...")

    # Run healing
    while healing.phase != HealingPhase.COMPLETE:
        repair.tick()
        if healing.progress == 0:  # New phase
            print(f"   Phase: {healing.phase.value}")

    print(f"   Repair complete!")

    # Restore from backup
    print("\n6. Restoring from backup...")
    restored = repair.restore_from_backup("main_state")
    print(f"   Restored: {restored}")

    # Final status
    print("\n7. Final status:")
    status = repair.get_status()
    for key, value in status.items():
        if not isinstance(value, dict):
            print(f"   {key}: {value}")
