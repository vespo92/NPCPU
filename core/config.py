"""
Configuration Management for NPCPU

A flexible configuration system that supports:
- Multiple configuration sources (files, env, defaults)
- Layered configuration with overrides
- Type validation
- Dynamic reloading
- Configuration sections
"""

from typing import (
    Dict, Any, List, Optional, Set, Callable, TypeVar,
    Type, Union, Generic, get_type_hints
)
from dataclasses import dataclass, field, fields, is_dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
import os
import json
import copy

# Try to import YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


T = TypeVar('T')


# =============================================================================
# Configuration Sources
# =============================================================================

class ConfigSource(ABC):
    """Abstract base for configuration sources"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Source identifier"""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority (higher = overrides lower)"""
        pass

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from this source"""
        pass

    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration (if supported)"""
        return False


class DictSource(ConfigSource):
    """Configuration from a dictionary"""

    def __init__(self, data: Dict[str, Any], name: str = "dict", priority: int = 0):
        self._data = data
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def load(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)


class FileSource(ConfigSource):
    """Configuration from a file (JSON or YAML)"""

    def __init__(
        self,
        path: Union[str, Path],
        priority: int = 10,
        required: bool = False
    ):
        self._path = Path(path)
        self._priority = priority
        self._required = required

    @property
    def name(self) -> str:
        return f"file:{self._path}"

    @property
    def priority(self) -> int:
        return self._priority

    def load(self) -> Dict[str, Any]:
        if not self._path.exists():
            if self._required:
                raise FileNotFoundError(f"Required config file not found: {self._path}")
            return {}

        with open(self._path, 'r') as f:
            if self._path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML required for YAML config files")
                return yaml.safe_load(f) or {}
            else:  # Assume JSON
                return json.load(f)

    def save(self, config: Dict[str, Any]) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, 'w') as f:
                if self._path.suffix in ['.yaml', '.yml']:
                    if not HAS_YAML:
                        return False
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2, default=str)
            return True
        except Exception:
            return False


class EnvSource(ConfigSource):
    """Configuration from environment variables"""

    def __init__(
        self,
        prefix: str = "NPCPU_",
        priority: int = 20,
        mapping: Optional[Dict[str, str]] = None
    ):
        self._prefix = prefix
        self._priority = priority
        self._mapping = mapping or {}

    @property
    def name(self) -> str:
        return f"env:{self._prefix}"

    @property
    def priority(self) -> int:
        return self._priority

    def load(self) -> Dict[str, Any]:
        config = {}

        # Load mapped variables
        for env_key, config_key in self._mapping.items():
            value = os.environ.get(env_key)
            if value is not None:
                self._set_nested(config, config_key, self._parse_value(value))

        # Load prefixed variables
        for key, value in os.environ.items():
            if key.startswith(self._prefix):
                config_key = key[len(self._prefix):].lower().replace('__', '.')
                self._set_nested(config, config_key, self._parse_value(value))

        return config

    def _set_nested(self, config: Dict, key: str, value: Any) -> None:
        """Set a nested configuration value using dot notation"""
        parts = key.split('.')
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value


# =============================================================================
# Configuration Section
# =============================================================================

class ConfigSection:
    """
    A typed configuration section.

    Provides type-safe access to configuration values with defaults.

    Example:
        class SimulationConfig(ConfigSection):
            population_size: int = 20
            max_ticks: int = 1000
            seed: Optional[int] = None

        config = LayeredConfig()
        sim_config = config.section("simulation", SimulationConfig)
        print(sim_config.population_size)  # Type-safe access
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        """Apply default values from class annotations"""
        for name, value in self._get_defaults().items():
            if name not in self._data:
                self._data[name] = value

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default values from class attributes"""
        defaults = {}
        for cls in type(self).__mro__:
            if cls is ConfigSection or cls is object:
                continue
            for name, value in vars(cls).items():
                if not name.startswith('_') and not callable(value):
                    defaults[name] = value
        return defaults

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        if name in self._data:
            return self._data[name]
        defaults = self._get_defaults()
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self._data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return copy.deepcopy(self._data)

    def update(self, data: Dict[str, Any]) -> None:
        """Update with new values"""
        self._data.update(data)


# =============================================================================
# Configuration Provider
# =============================================================================

class ConfigProvider(ABC):
    """Abstract interface for configuration access"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    def section(self, name: str, cls: Type[T] = None) -> T:
        """Get a configuration section"""
        pass


# =============================================================================
# Layered Configuration
# =============================================================================

class LayeredConfig(ConfigProvider):
    """
    A layered configuration system.

    Combines multiple configuration sources with priority-based overrides.

    Example:
        config = LayeredConfig()

        # Add sources (higher priority overrides lower)
        config.add_source(DictSource({
            "simulation": {"population": 20}
        }, priority=0))  # Defaults

        config.add_source(FileSource("config.yaml", priority=10))
        config.add_source(EnvSource("NPCPU_", priority=20))

        # Load configuration
        config.load()

        # Access values
        pop = config.get("simulation.population", 10)

        # Access typed sections
        class SimConfig(ConfigSection):
            population: int = 20

        sim = config.section("simulation", SimConfig)
    """

    def __init__(self):
        self._sources: List[ConfigSource] = []
        self._config: Dict[str, Any] = {}
        self._sections: Dict[str, ConfigSection] = {}
        self._watchers: Dict[str, List[Callable[[str, Any, Any], None]]] = {}

    # -------------------------------------------------------------------------
    # Source Management
    # -------------------------------------------------------------------------

    def add_source(self, source: ConfigSource) -> 'LayeredConfig':
        """Add a configuration source"""
        self._sources.append(source)
        self._sources.sort(key=lambda s: s.priority)
        return self

    def remove_source(self, name: str) -> bool:
        """Remove a configuration source by name"""
        for i, source in enumerate(self._sources):
            if source.name == name:
                self._sources.pop(i)
                return True
        return False

    def load(self) -> 'LayeredConfig':
        """Load configuration from all sources"""
        self._config = {}

        for source in self._sources:
            try:
                source_config = source.load()
                self._merge(self._config, source_config)
            except Exception as e:
                print(f"Warning: Failed to load config from {source.name}: {e}")

        # Update existing sections
        for name, section in self._sections.items():
            section_data = self._get_nested(self._config, name) or {}
            section.update(section_data)

        return self

    def _merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override into base"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    # -------------------------------------------------------------------------
    # Access
    # -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation"""
        return self._get_nested(self._config, key, default)

    def _get_nested(self, config: Dict, key: str, default: Any = None) -> Any:
        """Get a nested value using dot notation"""
        parts = key.split('.')
        current = config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation"""
        old_value = self.get(key)

        parts = key.split('.')
        current = self._config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        # Notify watchers
        self._notify_watchers(key, old_value, value)

    def has(self, key: str) -> bool:
        """Check if a key exists"""
        return self.get(key) is not None

    def section(self, name: str, cls: Type[T] = None) -> T:
        """
        Get a typed configuration section.

        Args:
            name: Section name (dot notation for nested)
            cls: ConfigSection subclass (optional)

        Returns:
            ConfigSection instance with typed access
        """
        if name in self._sections:
            return self._sections[name]

        section_data = self._get_nested(self._config, name) or {}
        section_cls = cls or ConfigSection
        section = section_cls(section_data)

        self._sections[name] = section
        return section

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return copy.deepcopy(self._config)

    # -------------------------------------------------------------------------
    # Watching
    # -------------------------------------------------------------------------

    def watch(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Watch for changes to a configuration key.

        Callback receives (key, old_value, new_value).
        """
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

    def unwatch(self, key: str, callback: Optional[Callable] = None) -> int:
        """Remove watchers for a key"""
        if key not in self._watchers:
            return 0

        if callback is None:
            count = len(self._watchers[key])
            del self._watchers[key]
            return count

        original = len(self._watchers[key])
        self._watchers[key] = [w for w in self._watchers[key] if w != callback]
        return original - len(self._watchers[key])

    def _notify_watchers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify watchers of a change"""
        if old_value == new_value:
            return

        # Exact key watchers
        if key in self._watchers:
            for callback in self._watchers[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception:
                    pass

        # Parent key watchers (e.g., "simulation" watches "simulation.population")
        parts = key.split('.')
        for i in range(len(parts)):
            parent_key = '.'.join(parts[:i])
            if parent_key in self._watchers:
                for callback in self._watchers[parent_key]:
                    try:
                        callback(key, old_value, new_value)
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, source_name: Optional[str] = None) -> bool:
        """
        Save configuration to a source.

        If source_name is None, saves to first saveable source.
        """
        for source in self._sources:
            if source_name is None or source.name == source_name:
                if source.save(self._config):
                    return True
        return False


# =============================================================================
# Default Configuration Sections
# =============================================================================

class SimulationConfigSection(ConfigSection):
    """Default simulation configuration"""
    name: str = "NPCPU Simulation"
    seed: Optional[int] = None
    max_ticks: int = 10000
    initial_population: int = 20
    carrying_capacity: int = 200
    tick_rate: float = 0.0
    auto_save_interval: int = 1000
    output_dir: str = "./simulation_output"
    verbose: bool = True


class WorldConfigSection(ConfigSection):
    """Default world configuration"""
    name: str = "Digital World"
    season_length: int = 250
    day_length: int = 24
    event_frequency: float = 0.03


class OrganismConfigSection(ConfigSection):
    """Default organism configuration"""
    initial_energy: float = 100.0
    max_energy: float = 100.0
    base_metabolic_rate: float = 1.0
    reproduction_threshold: float = 80.0
    mutation_rate: float = 0.1


# =============================================================================
# Global Configuration
# =============================================================================

_global_config: Optional[LayeredConfig] = None


def get_config() -> LayeredConfig:
    """Get the global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = LayeredConfig()
        # Add default sources
        _global_config.add_source(EnvSource("NPCPU_", priority=20))
    return _global_config


def set_config(config: LayeredConfig) -> None:
    """Set the global configuration instance"""
    global _global_config
    _global_config = config
