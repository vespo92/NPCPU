"""
NPCPU Swarm Module

Provides distributed consciousness coordination, network-based
consciousness sharing, swarm intelligence patterns, and multi-agent coordination.

Core components:
- SwarmCoordinator: Flocking, task allocation, formations
- CollectiveMind: Knowledge sharing, group decisions, hive mind
- DistributedConsciousnessNetwork: Network-based consciousness sharing
- SwarmIntelligenceEngine: Emergent swarm patterns
"""

__all__ = []

# Core coordination (no external dependencies)
from .coordination import (
    SwarmCoordinator,
    SwarmSubsystem,
    Position,
    FlockingVectors,
    Task,
    FormationPattern
)
__all__.extend([
    "SwarmCoordinator",
    "SwarmSubsystem",
    "Position",
    "FlockingVectors",
    "Task",
    "FormationPattern",
])

# Collective behavior (no external dependencies)
from .collective_behavior import (
    CollectiveMind,
    CollectiveMindSubsystem,
    Knowledge,
    Memory,
    KnowledgeType,
    DecisionMethod
)
__all__.extend([
    "CollectiveMind",
    "CollectiveMindSubsystem",
    "Knowledge",
    "Memory",
    "KnowledgeType",
    "DecisionMethod",
])

# Optional: Distributed consciousness (requires numpy)
try:
    from .distributed_consciousness import (
        DistributedConsciousnessNetwork,
        ConsciousnessNetworkConfig,
        NetworkAgent,
        ConsciousnessMessage,
        SwarmConsciousness
    )
    __all__.extend([
        "DistributedConsciousnessNetwork",
        "ConsciousnessNetworkConfig",
        "NetworkAgent",
        "ConsciousnessMessage",
        "SwarmConsciousness",
    ])
except ImportError:
    pass  # numpy not available

# Optional: Intelligence patterns (requires numpy)
try:
    from .intelligence_patterns import (
        SwarmIntelligenceEngine,
        SwarmPattern,
        SwarmAgent,
        SharedEnvironment
    )
    __all__.extend([
        "SwarmIntelligenceEngine",
        "SwarmPattern",
        "SwarmAgent",
        "SharedEnvironment",
    ])
except ImportError:
    pass  # numpy not available
