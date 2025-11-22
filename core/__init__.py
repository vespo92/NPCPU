"""
NPCPU Core Abstractions

This module provides the fundamental abstractions that define the NPCPU framework.
All implementations should derive from these base classes to ensure compatibility
and extensibility.

Key Abstractions:
- BaseOrganism: Interface for all organism types
- BaseWorld: Interface for world/environment implementations
- BasePopulation: Interface for population management
- EventBus: Decoupled event communication
- Plugin: Extension system
- ConfigProvider: Configuration abstraction
"""

from .abstractions import (
    BaseOrganism,
    BaseWorld,
    BasePopulation,
    BaseSubsystem,
    LifecyclePhase,
    OrganismCapability,
)

from .events import (
    Event,
    EventBus,
    EventHandler,
    EventFilter,
)

from .plugins import (
    Plugin,
    PluginManager,
    hook,
)

from .config import (
    ConfigProvider,
    ConfigSection,
    LayeredConfig,
)

__all__ = [
    # Abstractions
    'BaseOrganism',
    'BaseWorld',
    'BasePopulation',
    'BaseSubsystem',
    'LifecyclePhase',
    'OrganismCapability',
    # Events
    'Event',
    'EventBus',
    'EventHandler',
    'EventFilter',
    # Plugins
    'Plugin',
    'PluginManager',
    'hook',
    # Config
    'ConfigProvider',
    'ConfigSection',
    'LayeredConfig',
]
