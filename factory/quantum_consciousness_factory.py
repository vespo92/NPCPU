"""
Quantum Consciousness Factory

Factory for creating QuantumConsciousness instances from configuration.
Part of NEXUS-Q agent workstream.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum import (
    QuantumConsciousness,
    HybridQuantumConsciousness,
    EntanglementManager,
    CoherenceTracker,
    QuantumMemoryRegister,
    QuantumTunneling,
    DecoherenceModel,
)
from protocols.consciousness import GradedConsciousness


logger = logging.getLogger(__name__)


@dataclass
class QuantumConsciousnessConfig:
    """Configuration for quantum consciousness"""
    name: str
    num_qubits: int = 10
    t1_relaxation: float = 1000.0
    t2_dephasing: float = 500.0
    tunneling_enhancement: float = 1.0
    decoherence_rate: float = 0.01
    enable_entanglement: bool = True
    enable_memory: bool = True
    enable_tunneling: bool = True
    classical_base: Optional[Dict[str, float]] = None


class QuantumConsciousnessFactory:
    """
    Factory for creating quantum consciousness instances.

    Supports:
    - Creating from YAML configuration
    - Programmatic creation with parameters
    - Hybrid classical-quantum consciousness
    - Full quantum consciousness suites

    Example:
        factory = QuantumConsciousnessFactory()

        # From config file
        qc = factory.from_yaml("quantum_consciousness.yaml")

        # Programmatic
        qc = factory.create(num_qubits=8, enable_memory=True)

        # Full suite with all components
        suite = factory.create_full_suite("quantum_agent")
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize factory.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = config_dir or self._find_config_dir()

    def _find_config_dir(self) -> str:
        """Find configuration directory"""
        current_dir = os.path.dirname(__file__)
        config_dir = os.path.join(current_dir, "..", "configs", "consciousness_models")

        if os.path.exists(config_dir):
            return os.path.abspath(config_dir)

        return "./configs/consciousness_models"

    def from_yaml(
        self,
        filepath: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> QuantumConsciousness:
        """
        Create QuantumConsciousness from YAML file.

        Args:
            filepath: Path to YAML configuration
            overrides: Optional parameter overrides

        Returns:
            QuantumConsciousness instance
        """
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.config_dir, filepath)

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Apply overrides
        if overrides:
            config.update(overrides)

        return self._build_from_config(config)

    def _build_from_config(self, config: Dict[str, Any]) -> QuantumConsciousness:
        """Build quantum consciousness from config dict"""
        quantum_params = config.get("quantum_parameters", {})

        num_qubits = quantum_params.get("num_qubits", 10)

        qc = QuantumConsciousness(num_qubits=num_qubits)

        logger.info(
            f"Created {config.get('name', 'Quantum Consciousness')} "
            f"with {num_qubits} qubits"
        )

        return qc

    def create(
        self,
        name: str = "QuantumConsciousness",
        num_qubits: int = 10,
        **kwargs
    ) -> QuantumConsciousness:
        """
        Create quantum consciousness programmatically.

        Args:
            name: Instance name
            num_qubits: Number of qubits
            **kwargs: Additional parameters

        Returns:
            QuantumConsciousness instance
        """
        qc = QuantumConsciousness(num_qubits=num_qubits)

        logger.info(f"Created {name} with {num_qubits} qubits")

        return qc

    def create_hybrid(
        self,
        name: str = "HybridConsciousness",
        num_qubits: int = 8,
        classical_params: Optional[Dict[str, float]] = None
    ) -> HybridQuantumConsciousness:
        """
        Create hybrid classical-quantum consciousness.

        Args:
            name: Instance name
            num_qubits: Quantum component qubits
            classical_params: Classical consciousness parameters

        Returns:
            HybridQuantumConsciousness instance
        """
        if classical_params is None:
            classical_params = {
                "perception_fidelity": 0.8,
                "meta_cognitive_ability": 0.6,
                "memory_depth": 0.7
            }

        classical = GradedConsciousness(**classical_params)
        hybrid = HybridQuantumConsciousness(classical, num_qubits=num_qubits)

        logger.info(
            f"Created hybrid {name}: "
            f"{num_qubits} qubits + classical (score: {classical.overall_consciousness_score():.2f})"
        )

        return hybrid

    def create_full_suite(
        self,
        name: str,
        num_qubits: int = 10,
        config: Optional[QuantumConsciousnessConfig] = None
    ) -> Dict[str, Any]:
        """
        Create full quantum consciousness suite with all components.

        Returns dictionary with:
        - consciousness: QuantumConsciousness
        - entanglement: EntanglementManager
        - coherence_tracker: CoherenceTracker
        - memory: QuantumMemoryRegister
        - tunneling: QuantumTunneling
        - decoherence: DecoherenceModel

        Args:
            name: Suite name
            num_qubits: Number of qubits
            config: Optional configuration

        Returns:
            Dictionary of quantum consciousness components
        """
        if config is None:
            config = QuantumConsciousnessConfig(name=name, num_qubits=num_qubits)

        suite = {
            "name": name,
            "consciousness": QuantumConsciousness(num_qubits=config.num_qubits),
        }

        if config.enable_entanglement:
            suite["entanglement"] = EntanglementManager(
                decoherence_rate=config.decoherence_rate
            )

        suite["coherence_tracker"] = CoherenceTracker()

        if config.enable_memory:
            suite["memory"] = QuantumMemoryRegister(
                num_qubits=config.num_qubits,
                decoherence_time=config.t1_relaxation
            )

        if config.enable_tunneling:
            suite["tunneling"] = QuantumTunneling(
                tunneling_enhancement=config.tunneling_enhancement
            )

        suite["decoherence"] = DecoherenceModel(
            t1=config.t1_relaxation,
            t2=config.t2_dephasing
        )

        logger.info(
            f"Created full quantum suite '{name}': "
            f"{config.num_qubits} qubits, "
            f"components: {list(suite.keys())}"
        )

        return suite


# Convenience function
def create_quantum_consciousness(
    name: str = "default",
    num_qubits: int = 10,
    from_config: bool = False
) -> QuantumConsciousness:
    """
    Convenience function to create quantum consciousness.

    Args:
        name: Model name or config filename
        num_qubits: Number of qubits (if not from config)
        from_config: Whether to load from YAML

    Returns:
        QuantumConsciousness instance
    """
    factory = QuantumConsciousnessFactory()

    if from_config:
        if not name.endswith('.yaml'):
            name = f"{name}.yaml"
        return factory.from_yaml(name)
    else:
        return factory.create(name=name, num_qubits=num_qubits)


__all__ = [
    'QuantumConsciousnessConfig',
    'QuantumConsciousnessFactory',
    'create_quantum_consciousness',
]
