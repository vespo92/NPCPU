"""
Organism Metabolism System

Manages energy and resource consumption, storage, and allocation
for digital organisms. Implements biologically-inspired metabolic
processes including:

- Energy generation from resources
- Resource storage and depletion
- Metabolic rate adaptation
- Starvation and recovery
"""

import time
import uuid
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

class ResourceType(Enum):
    """Types of resources organisms can consume"""
    COMPUTE = "compute"          # CPU/processing cycles
    MEMORY = "memory"            # RAM/storage
    BANDWIDTH = "bandwidth"      # Network capacity
    DATA = "data"                # Information/training data
    ATTENTION = "attention"      # Human oversight/feedback
    ENERGY = "energy"            # Raw energy/electricity


class EnergyState(Enum):
    """Energy states of the organism"""
    THRIVING = "thriving"        # Abundant energy
    HEALTHY = "healthy"          # Normal operation
    CONSERVING = "conserving"    # Reducing consumption
    STRESSED = "stressed"        # Low energy
    STARVING = "starving"        # Critical energy
    DORMANT = "dormant"          # Minimal consumption mode


class MetabolicMode(Enum):
    """Metabolic operation modes"""
    GROWTH = "growth"            # Prioritize growth
    MAINTENANCE = "maintenance"  # Maintain current state
    PERFORMANCE = "performance"  # Maximum output
    CONSERVATION = "conservation"# Minimize consumption
    HEALING = "healing"          # Prioritize repair


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Resource:
    """A consumable resource"""
    type: ResourceType
    amount: float = 0.0
    max_capacity: float = 100.0
    decay_rate: float = 0.01     # Natural decay per cycle
    regeneration_rate: float = 0.0  # Natural regeneration

    def consume(self, amount: float) -> float:
        """Consume resource, return actual amount consumed"""
        actual = min(amount, self.amount)
        self.amount -= actual
        return actual

    def add(self, amount: float) -> float:
        """Add resource, return actual amount added"""
        space = self.max_capacity - self.amount
        actual = min(amount, space)
        self.amount += actual
        return actual

    def tick(self):
        """Process one cycle of decay/regeneration"""
        self.amount = max(0, self.amount - self.decay_rate)
        self.amount = min(self.max_capacity, self.amount + self.regeneration_rate)

    @property
    def fullness(self) -> float:
        """Get resource fullness as percentage"""
        return self.amount / self.max_capacity if self.max_capacity > 0 else 0


@dataclass
class MetabolicProcess:
    """A metabolic process that converts resources to energy/products"""
    name: str
    inputs: Dict[ResourceType, float]   # Resource type -> amount needed
    outputs: Dict[str, float]           # Output type -> amount produced
    efficiency: float = 1.0
    active: bool = True


@dataclass
class MetabolicConfig:
    """Configuration for metabolism"""
    base_metabolic_rate: float = 1.0    # Base energy consumption per cycle
    efficiency_factor: float = 1.0       # Overall metabolic efficiency
    starvation_threshold: float = 0.1    # Energy level triggering starvation
    conservation_threshold: float = 0.3  # Energy level triggering conservation
    dormancy_threshold: float = 0.05     # Energy level triggering dormancy


# ============================================================================
# Metabolism
# ============================================================================

class Metabolism:
    """
    Metabolic system for digital organisms.

    Manages resource intake, energy production, and allocation.

    Example:
        metabolism = Metabolism(config)

        # Add resources
        metabolism.intake(ResourceType.COMPUTE, 50)

        # Process metabolism
        metabolism.tick()

        # Check status
        if metabolism.energy_state == EnergyState.STARVING:
            metabolism.enter_dormancy()
    """

    def __init__(
        self,
        config: Optional[MetabolicConfig] = None,
        organism_id: Optional[str] = None
    ):
        self.organism_id = organism_id or str(uuid.uuid4())
        self.config = config or MetabolicConfig()

        # Energy
        self.energy: float = 50.0
        self.max_energy: float = 100.0
        self.energy_state = EnergyState.HEALTHY

        # Resources
        self.resources: Dict[ResourceType, Resource] = {
            rt: Resource(type=rt, amount=50.0, max_capacity=100.0)
            for rt in ResourceType
        }

        # Metabolic processes
        self.processes: List[MetabolicProcess] = self._initialize_processes()

        # State
        self.mode = MetabolicMode.MAINTENANCE
        self.metabolic_rate: float = self.config.base_metabolic_rate
        self.cycle_count: int = 0

        # Tracking
        self.energy_history: List[float] = []
        self.consumption_history: Dict[ResourceType, List[float]] = defaultdict(list)

    def _initialize_processes(self) -> List[MetabolicProcess]:
        """Initialize default metabolic processes"""
        return [
            MetabolicProcess(
                name="compute_to_energy",
                inputs={ResourceType.COMPUTE: 1.0},
                outputs={"energy": 2.0},
                efficiency=0.9
            ),
            MetabolicProcess(
                name="data_processing",
                inputs={ResourceType.DATA: 1.0, ResourceType.COMPUTE: 0.5},
                outputs={"knowledge": 1.0, "energy": 0.5},
                efficiency=0.8
            ),
            MetabolicProcess(
                name="memory_consolidation",
                inputs={ResourceType.MEMORY: 0.5, ResourceType.ENERGY: 1.0},
                outputs={"long_term_memory": 1.0},
                efficiency=0.85
            ),
            MetabolicProcess(
                name="network_sync",
                inputs={ResourceType.BANDWIDTH: 1.0, ResourceType.ENERGY: 0.5},
                outputs={"sync_quality": 1.0},
                efficiency=0.9
            )
        ]

    def intake(self, resource_type: ResourceType, amount: float) -> float:
        """
        Intake resources from environment.

        Returns actual amount absorbed.
        """
        if resource_type not in self.resources:
            self.resources[resource_type] = Resource(type=resource_type)

        absorbed = self.resources[resource_type].add(amount)
        return absorbed

    def consume(self, resource_type: ResourceType, amount: float) -> float:
        """
        Consume resources for operations.

        Returns actual amount consumed.
        """
        if resource_type not in self.resources:
            return 0.0

        consumed = self.resources[resource_type].consume(amount)
        self.consumption_history[resource_type].append(consumed)
        return consumed

    def tick(self) -> Dict[str, float]:
        """
        Process one metabolic cycle.

        Returns production outputs.
        """
        self.cycle_count += 1
        outputs: Dict[str, float] = defaultdict(float)

        # Natural resource decay/regeneration
        for resource in self.resources.values():
            resource.tick()

        # Base energy consumption
        base_consumption = self.metabolic_rate * self._get_mode_multiplier()
        self.energy -= base_consumption

        # Run active metabolic processes
        for process in self.processes:
            if not process.active:
                continue

            # Check if we have required inputs
            can_run = all(
                self.resources.get(rt, Resource(type=rt)).amount >= amount
                for rt, amount in process.inputs.items()
            )

            if can_run:
                # Consume inputs
                for rt, amount in process.inputs.items():
                    self.consume(rt, amount * process.efficiency)

                # Produce outputs
                for output, amount in process.outputs.items():
                    produced = amount * process.efficiency * self.config.efficiency_factor
                    outputs[output] += produced

                    if output == "energy":
                        self.energy = min(self.max_energy, self.energy + produced)

        # Update energy state
        self._update_energy_state()

        # Record history
        self.energy_history.append(self.energy)

        return dict(outputs)

    def _get_mode_multiplier(self) -> float:
        """Get consumption multiplier based on metabolic mode"""
        multipliers = {
            MetabolicMode.GROWTH: 1.5,
            MetabolicMode.MAINTENANCE: 1.0,
            MetabolicMode.PERFORMANCE: 2.0,
            MetabolicMode.CONSERVATION: 0.5,
            MetabolicMode.HEALING: 1.2
        }
        return multipliers.get(self.mode, 1.0)

    def _update_energy_state(self):
        """Update energy state based on current energy level"""
        energy_ratio = self.energy / self.max_energy

        if energy_ratio <= self.config.dormancy_threshold:
            self.energy_state = EnergyState.DORMANT
            self.mode = MetabolicMode.CONSERVATION
        elif energy_ratio <= self.config.starvation_threshold:
            self.energy_state = EnergyState.STARVING
            self.mode = MetabolicMode.CONSERVATION
        elif energy_ratio <= self.config.conservation_threshold:
            self.energy_state = EnergyState.STRESSED
        elif energy_ratio <= 0.5:
            self.energy_state = EnergyState.CONSERVING
        elif energy_ratio <= 0.8:
            self.energy_state = EnergyState.HEALTHY
        else:
            self.energy_state = EnergyState.THRIVING

    def set_mode(self, mode: MetabolicMode):
        """Set metabolic mode"""
        self.mode = mode

    def enter_dormancy(self):
        """Enter dormancy mode to conserve energy"""
        self.mode = MetabolicMode.CONSERVATION
        self.metabolic_rate = self.config.base_metabolic_rate * 0.1
        self.energy_state = EnergyState.DORMANT

        # Disable non-essential processes
        for process in self.processes:
            if process.name not in ["compute_to_energy"]:
                process.active = False

    def exit_dormancy(self):
        """Exit dormancy mode"""
        self.mode = MetabolicMode.MAINTENANCE
        self.metabolic_rate = self.config.base_metabolic_rate

        # Re-enable processes
        for process in self.processes:
            process.active = True

    def allocate_energy(self, target: str, amount: float) -> float:
        """
        Allocate energy to a specific target.

        Returns actual amount allocated.
        """
        actual = min(amount, self.energy)
        self.energy -= actual
        return actual

    def get_resource_status(self) -> Dict[str, Dict[str, float]]:
        """Get status of all resources"""
        return {
            rt.value: {
                "amount": r.amount,
                "capacity": r.max_capacity,
                "fullness": r.fullness
            }
            for rt, r in self.resources.items()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive metabolic status"""
        return {
            "energy": self.energy,
            "max_energy": self.max_energy,
            "energy_ratio": self.energy / self.max_energy,
            "energy_state": self.energy_state.value,
            "mode": self.mode.value,
            "metabolic_rate": self.metabolic_rate,
            "cycle_count": self.cycle_count,
            "resources": self.get_resource_status(),
            "active_processes": len([p for p in self.processes if p.active])
        }

    def get_efficiency_report(self) -> Dict[str, float]:
        """Get metabolic efficiency report"""
        if len(self.energy_history) < 2:
            return {"efficiency": 0.0}

        # Calculate energy production vs consumption
        energy_changes = [
            self.energy_history[i] - self.energy_history[i-1]
            for i in range(1, len(self.energy_history))
        ]

        return {
            "average_energy_change": np.mean(energy_changes),
            "energy_stability": 1.0 - np.std(energy_changes) / max(1, np.mean(np.abs(energy_changes))),
            "current_efficiency": self.config.efficiency_factor,
            "mode_efficiency": self._get_mode_multiplier()
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Metabolism System Demo")
    print("=" * 50)

    # Create metabolism
    config = MetabolicConfig(
        base_metabolic_rate=2.0,
        efficiency_factor=0.9
    )
    metabolism = Metabolism(config)

    print(f"\n1. Initial state:")
    status = metabolism.get_status()
    print(f"   Energy: {status['energy']:.1f}/{status['max_energy']}")
    print(f"   State: {status['energy_state']}")

    # Simulate normal operation
    print("\n2. Normal operation (with resource intake)...")
    for cycle in range(20):
        # Periodic resource intake
        if cycle % 5 == 0:
            metabolism.intake(ResourceType.COMPUTE, 20)
            metabolism.intake(ResourceType.DATA, 15)

        outputs = metabolism.tick()

        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: energy={metabolism.energy:.1f}, "
                  f"state={metabolism.energy_state.value}")

    # Simulate starvation
    print("\n3. Simulating starvation (no resource intake)...")
    for cycle in range(30):
        outputs = metabolism.tick()

        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: energy={metabolism.energy:.1f}, "
                  f"state={metabolism.energy_state.value}, mode={metabolism.mode.value}")

        if metabolism.energy_state == EnergyState.STARVING:
            metabolism.enter_dormancy()

    # Recovery
    print("\n4. Recovery with resource intake...")
    metabolism.exit_dormancy()
    for cycle in range(20):
        # Heavy resource intake
        metabolism.intake(ResourceType.COMPUTE, 30)
        metabolism.intake(ResourceType.ENERGY, 10)

        outputs = metabolism.tick()

        if cycle % 5 == 0:
            print(f"   Cycle {cycle}: energy={metabolism.energy:.1f}, "
                  f"state={metabolism.energy_state.value}")

    # Final status
    print("\n5. Final status:")
    status = metabolism.get_status()
    for key, value in status.items():
        if not isinstance(value, dict):
            print(f"   {key}: {value}")

    print("\n6. Efficiency report:")
    efficiency = metabolism.get_efficiency_report()
    for key, value in efficiency.items():
        print(f"   {key}: {value:.3f}")
