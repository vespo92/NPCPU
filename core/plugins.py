"""
Plugin System for NPCPU

A flexible plugin architecture that enables:
- Extension of core functionality
- Custom organism types
- New subsystems
- Behavior modifications
- Event hooks
"""

from typing import (
    Dict, Any, List, Optional, Set, Callable, TypeVar,
    Type, Union, Tuple
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
import importlib
import importlib.util
import inspect
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Hook System
# =============================================================================

class HookPoint(Enum):
    """Standard hook points in the system"""
    # Simulation lifecycle
    SIMULATION_INIT = "simulation.init"
    SIMULATION_START = "simulation.start"
    SIMULATION_TICK = "simulation.tick"
    SIMULATION_END = "simulation.end"

    # Organism lifecycle
    ORGANISM_CREATE = "organism.create"
    ORGANISM_TICK = "organism.tick"
    ORGANISM_DIE = "organism.die"

    # Population
    POPULATION_ADD = "population.add"
    POPULATION_REMOVE = "population.remove"
    POPULATION_INTERACT = "population.interact"

    # World
    WORLD_TICK = "world.tick"
    WORLD_EVENT = "world.event"
    WORLD_RESOURCE_CHANGE = "world.resource_change"

    # Custom hooks can be added as strings


@dataclass
class HookRegistration:
    """Registration for a hook callback"""
    callback: Callable
    priority: int = 0  # Higher = called first
    plugin_name: str = ""


class HookManager:
    """
    Manages hooks throughout the system.

    Allows plugins to intercept and modify behavior at key points.
    """

    def __init__(self):
        self._hooks: Dict[str, List[HookRegistration]] = {}

    def register(
        self,
        hook_point: Union[HookPoint, str],
        callback: Callable,
        priority: int = 0,
        plugin_name: str = ""
    ) -> None:
        """Register a callback for a hook point"""
        key = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if key not in self._hooks:
            self._hooks[key] = []

        self._hooks[key].append(HookRegistration(
            callback=callback,
            priority=priority,
            plugin_name=plugin_name
        ))

        # Sort by priority (highest first)
        self._hooks[key].sort(key=lambda r: -r.priority)

    def unregister(
        self,
        hook_point: Union[HookPoint, str],
        callback: Optional[Callable] = None,
        plugin_name: Optional[str] = None
    ) -> int:
        """
        Unregister hooks.

        Can unregister by callback, plugin name, or both.
        Returns number of hooks removed.
        """
        key = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if key not in self._hooks:
            return 0

        original_count = len(self._hooks[key])

        self._hooks[key] = [
            reg for reg in self._hooks[key]
            if not (
                (callback is None or reg.callback == callback) and
                (plugin_name is None or reg.plugin_name == plugin_name)
            )
        ]

        return original_count - len(self._hooks[key])

    def call(
        self,
        hook_point: Union[HookPoint, str],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Call all registered hooks for a point.

        Returns list of results from each hook.
        """
        key = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if key not in self._hooks:
            return []

        results = []
        for reg in self._hooks[key]:
            try:
                result = reg.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {key} from {reg.plugin_name}: {e}")

        return results

    def call_filter(
        self,
        hook_point: Union[HookPoint, str],
        value: Any,
        *args,
        **kwargs
    ) -> Any:
        """
        Call hooks as a filter chain.

        Each hook receives the value and can modify it.
        Returns the final filtered value.
        """
        key = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if key not in self._hooks:
            return value

        for reg in self._hooks[key]:
            try:
                value = reg.callback(value, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in filter hook {key} from {reg.plugin_name}: {e}")

        return value

    def call_until(
        self,
        hook_point: Union[HookPoint, str],
        *args,
        stop_value: Any = True,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Call hooks until one returns the stop value.

        Returns (result, stopped) tuple.
        """
        key = hook_point.value if isinstance(hook_point, HookPoint) else hook_point

        if key not in self._hooks:
            return (None, False)

        for reg in self._hooks[key]:
            try:
                result = reg.callback(*args, **kwargs)
                if result == stop_value:
                    return (result, True)
            except Exception as e:
                logger.error(f"Error in hook {key} from {reg.plugin_name}: {e}")

        return (None, False)


# Decorator for hooks
def hook(
    hook_point: Union[HookPoint, str],
    priority: int = 0,
    manager: Optional[HookManager] = None
):
    """
    Decorator to register a function as a hook.

    Example:
        @hook(HookPoint.ORGANISM_CREATE)
        def on_organism_create(organism):
            print(f"Organism created: {organism.name}")
    """
    def decorator(func: Callable) -> Callable:
        # Store hook info on function for later registration
        if not hasattr(func, '_npcpu_hooks'):
            func._npcpu_hooks = []
        func._npcpu_hooks.append((hook_point, priority))

        # If manager provided, register immediately
        if manager:
            manager.register(hook_point, func, priority)

        return func
    return decorator


# =============================================================================
# Plugin Base Class
# =============================================================================

@dataclass
class PluginInfo:
    """Metadata about a plugin"""
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    url: str = ""


class Plugin(ABC):
    """
    Base class for NPCPU plugins.

    Plugins can:
    - Register hooks to modify behavior
    - Add new organism types
    - Provide new subsystems
    - Add world events
    - Extend the configuration

    Example:
        class MyPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(
                    name="my-plugin",
                    version="1.0.0",
                    description="Adds custom behaviors"
                )

            def activate(self, context: PluginContext) -> None:
                context.hooks.register(
                    HookPoint.ORGANISM_TICK,
                    self.on_organism_tick
                )

            def on_organism_tick(self, organism):
                # Custom behavior
                pass
    """

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Get plugin metadata"""
        pass

    @abstractmethod
    def activate(self, context: 'PluginContext') -> None:
        """
        Called when plugin is activated.

        Use context to register hooks, access managers, etc.
        """
        pass

    def deactivate(self, context: 'PluginContext') -> None:
        """
        Called when plugin is deactivated.

        Override to clean up resources.
        """
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Called with plugin configuration.

        Override to handle plugin-specific settings.
        """
        pass


@dataclass
class PluginContext:
    """Context provided to plugins during activation"""
    hooks: HookManager
    event_bus: Any  # EventBus from events module
    config: Dict[str, Any] = field(default_factory=dict)
    registry: Dict[str, Any] = field(default_factory=dict)

    def register(self, key: str, value: Any) -> None:
        """Register a value in the plugin registry"""
        self.registry[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the plugin registry"""
        return self.registry.get(key, default)


# =============================================================================
# Plugin Manager
# =============================================================================

class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Example:
        manager = PluginManager()

        # Load plugins from directory
        manager.discover("./plugins")

        # Or register directly
        manager.register(MyPlugin())

        # Activate all plugins
        manager.activate_all()

        # Access plugin
        plugin = manager.get("my-plugin")
    """

    def __init__(self, hook_manager: Optional[HookManager] = None):
        self._plugins: Dict[str, Plugin] = {}
        self._active: Set[str] = set()
        self._hooks = hook_manager or HookManager()
        self._context: Optional[PluginContext] = None

    @property
    def hooks(self) -> HookManager:
        return self._hooks

    def set_context(self, context: PluginContext) -> None:
        """Set the plugin context"""
        self._context = context

    def get_context(self) -> PluginContext:
        """Get or create the plugin context"""
        if self._context is None:
            from .events import get_event_bus
            self._context = PluginContext(
                hooks=self._hooks,
                event_bus=get_event_bus()
            )
        return self._context

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, plugin: Plugin) -> bool:
        """
        Register a plugin instance.

        Returns True if registered successfully.
        """
        info = plugin.info
        if info.name in self._plugins:
            logger.warning(f"Plugin {info.name} already registered")
            return False

        self._plugins[info.name] = plugin
        logger.info(f"Registered plugin: {info.name} v{info.version}")
        return True

    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.

        Deactivates plugin first if active.
        """
        if name not in self._plugins:
            return False

        if name in self._active:
            self.deactivate(name)

        del self._plugins[name]
        logger.info(f"Unregistered plugin: {name}")
        return True

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover(self, path: Union[str, Path]) -> List[str]:
        """
        Discover and register plugins from a directory.

        Looks for Python files containing Plugin subclasses.
        Returns list of discovered plugin names.
        """
        path = Path(path)
        discovered = []

        if not path.exists():
            logger.warning(f"Plugin directory not found: {path}")
            return discovered

        for file in path.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                plugins = self._load_plugins_from_file(file)
                for plugin in plugins:
                    if self.register(plugin):
                        discovered.append(plugin.info.name)
            except Exception as e:
                logger.error(f"Error loading plugins from {file}: {e}")

        return discovered

    def _load_plugins_from_file(self, file: Path) -> List[Plugin]:
        """Load plugin classes from a Python file"""
        spec = importlib.util.spec_from_file_location(file.stem, file)
        if spec is None or spec.loader is None:
            return []

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        plugins = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Plugin) and
                obj is not Plugin and
                not inspect.isabstract(obj)
            ):
                try:
                    plugins.append(obj())
                except Exception as e:
                    logger.error(f"Error instantiating plugin {name}: {e}")

        return plugins

    # -------------------------------------------------------------------------
    # Activation
    # -------------------------------------------------------------------------

    def activate(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Activate a plugin by name.

        Returns True if activated successfully.
        """
        if name not in self._plugins:
            logger.error(f"Plugin not found: {name}")
            return False

        if name in self._active:
            logger.warning(f"Plugin already active: {name}")
            return True

        plugin = self._plugins[name]
        info = plugin.info

        # Check dependencies
        for dep in info.dependencies:
            if dep not in self._active:
                logger.info(f"Activating dependency {dep} for {name}")
                if not self.activate(dep):
                    logger.error(f"Failed to activate dependency {dep} for {name}")
                    return False

        # Configure and activate
        context = self.get_context()
        if config:
            plugin.configure(config)

        try:
            plugin.activate(context)
            self._active.add(name)
            logger.info(f"Activated plugin: {name}")

            # Register any decorated hooks
            self._register_decorated_hooks(plugin)

            return True
        except Exception as e:
            logger.error(f"Error activating plugin {name}: {e}")
            return False

    def _register_decorated_hooks(self, plugin: Plugin) -> None:
        """Register hooks from @hook decorated methods"""
        for name, method in inspect.getmembers(plugin, predicate=inspect.ismethod):
            if hasattr(method, '_npcpu_hooks'):
                for hook_point, priority in method._npcpu_hooks:
                    self._hooks.register(
                        hook_point,
                        method,
                        priority=priority,
                        plugin_name=plugin.info.name
                    )

    def deactivate(self, name: str) -> bool:
        """
        Deactivate a plugin by name.

        Returns True if deactivated successfully.
        """
        if name not in self._active:
            return True

        plugin = self._plugins.get(name)
        if plugin is None:
            self._active.discard(name)
            return True

        try:
            context = self.get_context()
            plugin.deactivate(context)

            # Unregister all hooks from this plugin
            for hook_list in self._hooks._hooks.values():
                hook_list[:] = [
                    reg for reg in hook_list
                    if reg.plugin_name != name
                ]

            self._active.discard(name)
            logger.info(f"Deactivated plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deactivating plugin {name}: {e}")
            return False

    def activate_all(self, configs: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
        """
        Activate all registered plugins.

        Returns number of plugins activated.
        """
        configs = configs or {}
        count = 0
        for name in self._plugins:
            if name not in self._active:
                if self.activate(name, configs.get(name)):
                    count += 1
        return count

    def deactivate_all(self) -> int:
        """
        Deactivate all active plugins.

        Returns number of plugins deactivated.
        """
        count = 0
        for name in list(self._active):
            if self.deactivate(name):
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Access
    # -------------------------------------------------------------------------

    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name"""
        return self._plugins.get(name)

    def is_active(self, name: str) -> bool:
        """Check if a plugin is active"""
        return name in self._active

    @property
    def plugins(self) -> Dict[str, Plugin]:
        """Get all registered plugins"""
        return self._plugins.copy()

    @property
    def active_plugins(self) -> Set[str]:
        """Get names of active plugins"""
        return self._active.copy()

    def list_plugins(self) -> List[PluginInfo]:
        """Get info for all registered plugins"""
        return [p.info for p in self._plugins.values()]


# =============================================================================
# Global Plugin Manager
# =============================================================================

_global_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = PluginManager()
    return _global_manager


def set_plugin_manager(manager: PluginManager) -> None:
    """Set the global plugin manager instance"""
    global _global_manager
    _global_manager = manager
