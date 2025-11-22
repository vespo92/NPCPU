"""
Tests for Plugin System

Unit tests for HookManager, Plugin, PluginManager, and related functionality.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from core.plugins import (
    HookPoint, HookRegistration, HookManager,
    Plugin, PluginInfo, PluginContext, PluginManager,
    hook, get_plugin_manager, set_plugin_manager
)
from core.events import EventBus


# =============================================================================
# Test Fixtures
# =============================================================================

class SimplePlugin(Plugin):
    """Simple plugin for testing"""

    def __init__(self, name: str = "simple-plugin", version: str = "1.0.0"):
        self._name = name
        self._version = version
        self.activated = False
        self.deactivated = False
        self.config: Dict[str, Any] = {}
        self.hooks_called: List[str] = []

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name=self._name,
            version=self._version,
            description="A simple test plugin"
        )

    def activate(self, context: PluginContext) -> None:
        self.activated = True
        context.hooks.register(
            HookPoint.ORGANISM_CREATE,
            self.on_organism_create,
            plugin_name=self.info.name
        )

    def deactivate(self, context: PluginContext) -> None:
        self.deactivated = True

    def configure(self, config: Dict[str, Any]) -> None:
        self.config = config

    def on_organism_create(self, organism):
        self.hooks_called.append("organism_create")
        return organism


class DependentPlugin(Plugin):
    """Plugin that depends on another"""

    def __init__(self, dependencies: List[str]):
        self._dependencies = dependencies

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="dependent-plugin",
            version="1.0.0",
            dependencies=self._dependencies
        )

    def activate(self, context: PluginContext) -> None:
        pass


class DecoratedPlugin(Plugin):
    """Plugin using @hook decorator"""

    def __init__(self):
        self.hook_results: List[str] = []

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(name="decorated-plugin", version="1.0.0")

    def activate(self, context: PluginContext) -> None:
        pass

    @hook(HookPoint.SIMULATION_TICK)
    def on_tick(self, simulation):
        self.hook_results.append("tick")

    @hook("custom.hook", priority=10)
    def on_custom(self, data):
        self.hook_results.append("custom")


# =============================================================================
# Tests: HookPoint
# =============================================================================

class TestHookPoint:
    """Tests for HookPoint enum"""

    def test_standard_hooks_exist(self):
        """Test standard hook points are defined"""
        standard = [
            HookPoint.SIMULATION_INIT,
            HookPoint.SIMULATION_START,
            HookPoint.SIMULATION_TICK,
            HookPoint.SIMULATION_END,
            HookPoint.ORGANISM_CREATE,
            HookPoint.ORGANISM_TICK,
            HookPoint.ORGANISM_DIE,
            HookPoint.WORLD_TICK,
            HookPoint.WORLD_EVENT
        ]
        assert len(standard) == 9

    def test_hook_values_are_strings(self):
        """Test hook values are dot-separated strings"""
        for hook in HookPoint:
            assert isinstance(hook.value, str)
            assert "." in hook.value


# =============================================================================
# Tests: HookManager
# =============================================================================

class TestHookManager:
    """Tests for HookManager"""

    def test_initialization(self):
        """Test hook manager initialization"""
        manager = HookManager()
        assert manager is not None

    def test_register_hook(self):
        """Test registering a hook"""
        manager = HookManager()
        results = []

        manager.register(HookPoint.ORGANISM_CREATE, lambda o: results.append(o))
        manager.call(HookPoint.ORGANISM_CREATE, "test_organism")

        assert results == ["test_organism"]

    def test_register_with_string(self):
        """Test registering with string hook name"""
        manager = HookManager()
        results = []

        manager.register("custom.hook", lambda d: results.append(d))
        manager.call("custom.hook", "data")

        assert results == ["data"]

    def test_priority_ordering(self):
        """Test hooks called in priority order"""
        manager = HookManager()
        order = []

        manager.register(HookPoint.ORGANISM_TICK, lambda: order.append("low"), priority=0)
        manager.register(HookPoint.ORGANISM_TICK, lambda: order.append("high"), priority=10)
        manager.register(HookPoint.ORGANISM_TICK, lambda: order.append("medium"), priority=5)

        manager.call(HookPoint.ORGANISM_TICK)
        assert order == ["high", "medium", "low"]

    def test_unregister_by_callback(self):
        """Test unregistering by callback"""
        manager = HookManager()
        results = []

        def callback():
            results.append("called")

        manager.register(HookPoint.SIMULATION_TICK, callback)
        manager.call(HookPoint.SIMULATION_TICK)
        assert len(results) == 1

        removed = manager.unregister(HookPoint.SIMULATION_TICK, callback=callback)
        assert removed == 1

        manager.call(HookPoint.SIMULATION_TICK)
        assert len(results) == 1  # No additional calls

    def test_unregister_by_plugin_name(self):
        """Test unregistering by plugin name"""
        manager = HookManager()

        manager.register(HookPoint.SIMULATION_TICK, lambda: None, plugin_name="plugin_a")
        manager.register(HookPoint.SIMULATION_TICK, lambda: None, plugin_name="plugin_a")
        manager.register(HookPoint.SIMULATION_TICK, lambda: None, plugin_name="plugin_b")

        removed = manager.unregister(HookPoint.SIMULATION_TICK, plugin_name="plugin_a")
        assert removed == 2

    def test_call_returns_results(self):
        """Test call returns list of results"""
        manager = HookManager()

        manager.register(HookPoint.ORGANISM_CREATE, lambda o: o * 2)
        manager.register(HookPoint.ORGANISM_CREATE, lambda o: o + 10)

        results = manager.call(HookPoint.ORGANISM_CREATE, 5)
        assert results == [10, 15]

    def test_call_nonexistent_hook(self):
        """Test calling unregistered hook returns empty list"""
        manager = HookManager()
        results = manager.call("nonexistent.hook")
        assert results == []

    def test_call_filter(self):
        """Test call_filter chains modifications"""
        manager = HookManager()

        manager.register("filter.test", lambda v: v * 2)
        manager.register("filter.test", lambda v: v + 1)

        # (5 * 2) + 1 = 11
        result = manager.call_filter("filter.test", 5)
        assert result == 11

    def test_call_filter_with_args(self):
        """Test call_filter with additional arguments"""
        manager = HookManager()

        manager.register("filter.test", lambda v, mult: v * mult)

        result = manager.call_filter("filter.test", 5, 3)
        assert result == 15

    def test_call_until(self):
        """Test call_until stops at stop value"""
        manager = HookManager()

        manager.register("check", lambda: False, priority=10)
        manager.register("check", lambda: True, priority=5)  # Stop here
        manager.register("check", lambda: False, priority=0)  # Not called

        result, stopped = manager.call_until("check", stop_value=True)
        assert result is True
        assert stopped is True

    def test_call_until_not_stopped(self):
        """Test call_until when stop value not reached"""
        manager = HookManager()

        manager.register("check", lambda: False)
        manager.register("check", lambda: False)

        result, stopped = manager.call_until("check", stop_value=True)
        assert stopped is False

    def test_hook_exception_handling(self):
        """Test exceptions in hooks are caught"""
        manager = HookManager()
        results = []

        def failing_hook():
            raise ValueError("Test error")

        def working_hook():
            results.append("worked")

        manager.register(HookPoint.SIMULATION_TICK, failing_hook, priority=10)
        manager.register(HookPoint.SIMULATION_TICK, working_hook, priority=5)

        manager.call(HookPoint.SIMULATION_TICK)
        assert results == ["worked"]


# =============================================================================
# Tests: hook Decorator
# =============================================================================

class TestHookDecorator:
    """Tests for @hook decorator"""

    def test_decorator_stores_info(self):
        """Test decorator stores hook info on function"""
        @hook(HookPoint.ORGANISM_CREATE)
        def my_hook(organism):
            pass

        assert hasattr(my_hook, '_npcpu_hooks')
        assert (HookPoint.ORGANISM_CREATE, 0) in my_hook._npcpu_hooks

    def test_decorator_with_priority(self):
        """Test decorator with priority"""
        @hook(HookPoint.SIMULATION_TICK, priority=10)
        def high_priority():
            pass

        assert (HookPoint.SIMULATION_TICK, 10) in high_priority._npcpu_hooks

    def test_multiple_decorators(self):
        """Test multiple hook decorators on same function"""
        @hook(HookPoint.ORGANISM_CREATE)
        @hook(HookPoint.ORGANISM_DIE)
        def multi_hook(organism):
            pass

        assert len(multi_hook._npcpu_hooks) == 2

    def test_decorator_with_manager(self):
        """Test decorator registers immediately with manager"""
        manager = HookManager()
        results = []

        @hook(HookPoint.SIMULATION_TICK, manager=manager)
        def immediate_hook():
            results.append("called")

        manager.call(HookPoint.SIMULATION_TICK)
        assert results == ["called"]


# =============================================================================
# Tests: PluginInfo
# =============================================================================

class TestPluginInfo:
    """Tests for PluginInfo dataclass"""

    def test_initialization(self):
        """Test plugin info initialization"""
        info = PluginInfo(
            name="test-plugin",
            version="2.0.0",
            author="Test Author",
            description="A test plugin"
        )
        assert info.name == "test-plugin"
        assert info.version == "2.0.0"
        assert info.author == "Test Author"
        assert info.description == "A test plugin"

    def test_defaults(self):
        """Test default values"""
        info = PluginInfo(name="minimal")
        assert info.version == "1.0.0"
        assert info.author == ""
        assert info.dependencies == []


# =============================================================================
# Tests: PluginContext
# =============================================================================

class TestPluginContext:
    """Tests for PluginContext"""

    def test_initialization(self):
        """Test context initialization"""
        hooks = HookManager()
        bus = EventBus()
        context = PluginContext(hooks=hooks, event_bus=bus)

        assert context.hooks is hooks
        assert context.event_bus is bus
        assert context.config == {}
        assert context.registry == {}

    def test_register_and_get(self):
        """Test registry operations"""
        context = PluginContext(hooks=HookManager(), event_bus=EventBus())

        context.register("key", "value")
        assert context.get("key") == "value"
        assert context.get("missing", "default") == "default"


# =============================================================================
# Tests: Plugin
# =============================================================================

class TestPlugin:
    """Tests for Plugin base class"""

    def test_simple_plugin_info(self):
        """Test plugin info property"""
        plugin = SimplePlugin()
        info = plugin.info

        assert info.name == "simple-plugin"
        assert info.version == "1.0.0"

    def test_activate(self):
        """Test plugin activation"""
        plugin = SimplePlugin()
        context = PluginContext(hooks=HookManager(), event_bus=EventBus())

        assert plugin.activated is False
        plugin.activate(context)
        assert plugin.activated is True

    def test_deactivate(self):
        """Test plugin deactivation"""
        plugin = SimplePlugin()
        context = PluginContext(hooks=HookManager(), event_bus=EventBus())

        plugin.deactivate(context)
        assert plugin.deactivated is True

    def test_configure(self):
        """Test plugin configuration"""
        plugin = SimplePlugin()
        plugin.configure({"setting": "value", "number": 42})

        assert plugin.config["setting"] == "value"
        assert plugin.config["number"] == 42


# =============================================================================
# Tests: PluginManager
# =============================================================================

class TestPluginManager:
    """Tests for PluginManager"""

    def test_initialization(self):
        """Test manager initialization"""
        manager = PluginManager()
        assert manager is not None
        assert manager.hooks is not None

    def test_register_plugin(self):
        """Test registering a plugin"""
        manager = PluginManager()
        plugin = SimplePlugin()

        result = manager.register(plugin)
        assert result is True
        assert manager.get("simple-plugin") is plugin

    def test_register_duplicate(self):
        """Test registering duplicate plugin fails"""
        manager = PluginManager()
        plugin1 = SimplePlugin()
        plugin2 = SimplePlugin()

        manager.register(plugin1)
        result = manager.register(plugin2)
        assert result is False

    def test_unregister_plugin(self):
        """Test unregistering a plugin"""
        manager = PluginManager()
        plugin = SimplePlugin()

        manager.register(plugin)
        result = manager.unregister("simple-plugin")
        assert result is True
        assert manager.get("simple-plugin") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent plugin"""
        manager = PluginManager()
        result = manager.unregister("nonexistent")
        assert result is False

    def test_activate_plugin(self):
        """Test activating a plugin"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)

        result = manager.activate("simple-plugin")
        assert result is True
        assert plugin.activated is True
        assert manager.is_active("simple-plugin") is True

    def test_activate_nonexistent(self):
        """Test activating nonexistent plugin fails"""
        manager = PluginManager()
        result = manager.activate("nonexistent")
        assert result is False

    def test_activate_already_active(self):
        """Test activating already active plugin"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)

        manager.activate("simple-plugin")
        result = manager.activate("simple-plugin")
        assert result is True  # Returns True (already active)

    def test_activate_with_config(self):
        """Test activating with configuration"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)

        manager.activate("simple-plugin", config={"key": "value"})
        assert plugin.config["key"] == "value"

    def test_deactivate_plugin(self):
        """Test deactivating a plugin"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)
        manager.activate("simple-plugin")

        result = manager.deactivate("simple-plugin")
        assert result is True
        assert plugin.deactivated is True
        assert manager.is_active("simple-plugin") is False

    def test_deactivate_not_active(self):
        """Test deactivating not-active plugin"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)

        result = manager.deactivate("simple-plugin")
        assert result is True

    def test_activate_all(self):
        """Test activating all plugins"""
        manager = PluginManager()
        p1 = SimplePlugin("plugin-1")
        p2 = SimplePlugin("plugin-2")

        manager.register(p1)
        manager.register(p2)

        count = manager.activate_all()
        assert count == 2
        assert p1.activated is True
        assert p2.activated is True

    def test_deactivate_all(self):
        """Test deactivating all plugins"""
        manager = PluginManager()
        p1 = SimplePlugin("plugin-1")
        p2 = SimplePlugin("plugin-2")

        manager.register(p1)
        manager.register(p2)
        manager.activate_all()

        count = manager.deactivate_all()
        assert count == 2
        assert manager.is_active("plugin-1") is False
        assert manager.is_active("plugin-2") is False

    def test_plugins_property(self):
        """Test plugins property returns copy"""
        manager = PluginManager()
        p1 = SimplePlugin("plugin-1")
        manager.register(p1)

        plugins = manager.plugins
        assert "plugin-1" in plugins

    def test_active_plugins_property(self):
        """Test active_plugins property"""
        manager = PluginManager()
        p1 = SimplePlugin("plugin-1")
        p2 = SimplePlugin("plugin-2")

        manager.register(p1)
        manager.register(p2)
        manager.activate("plugin-1")

        active = manager.active_plugins
        assert "plugin-1" in active
        assert "plugin-2" not in active

    def test_list_plugins(self):
        """Test listing all plugin info"""
        manager = PluginManager()
        p1 = SimplePlugin("plugin-1", "1.0.0")
        p2 = SimplePlugin("plugin-2", "2.0.0")

        manager.register(p1)
        manager.register(p2)

        infos = manager.list_plugins()
        names = [i.name for i in infos]
        assert "plugin-1" in names
        assert "plugin-2" in names

    def test_hook_registration_on_activate(self):
        """Test hooks registered when plugin activates"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)
        manager.activate("simple-plugin")

        # Call the hook
        manager.hooks.call(HookPoint.ORGANISM_CREATE, "test_organism")
        assert "organism_create" in plugin.hooks_called

    def test_hooks_unregistered_on_deactivate(self):
        """Test hooks unregistered when plugin deactivates"""
        manager = PluginManager()
        plugin = SimplePlugin()
        manager.register(plugin)
        manager.activate("simple-plugin")
        manager.deactivate("simple-plugin")

        # Clear and call
        plugin.hooks_called.clear()
        manager.hooks.call(HookPoint.ORGANISM_CREATE, "test_organism")
        assert len(plugin.hooks_called) == 0


class TestPluginDependencies:
    """Tests for plugin dependencies"""

    def test_dependency_activation(self):
        """Test dependencies activated before dependent"""
        manager = PluginManager()
        base = SimplePlugin("base-plugin")
        dependent = DependentPlugin(["base-plugin"])

        manager.register(base)
        manager.register(dependent)

        manager.activate("dependent-plugin")

        assert manager.is_active("base-plugin")
        assert manager.is_active("dependent-plugin")

    def test_missing_dependency_fails(self):
        """Test activation fails with missing dependency"""
        manager = PluginManager()
        dependent = DependentPlugin(["missing-plugin"])

        manager.register(dependent)
        result = manager.activate("dependent-plugin")

        assert result is False


class TestDecoratedHooks:
    """Tests for @hook decorated methods in plugins"""

    def test_decorated_hooks_registered(self):
        """Test decorated hooks are registered on activation"""
        manager = PluginManager()
        plugin = DecoratedPlugin()
        manager.register(plugin)
        manager.activate("decorated-plugin")

        # Call standard hook
        manager.hooks.call(HookPoint.SIMULATION_TICK, None)
        assert "tick" in plugin.hook_results

        # Call custom hook
        manager.hooks.call("custom.hook", None)
        assert "custom" in plugin.hook_results


# =============================================================================
# Tests: Plugin Discovery
# =============================================================================

class TestPluginDiscovery:
    """Tests for plugin discovery"""

    def test_discover_nonexistent_directory(self):
        """Test discovery with nonexistent directory"""
        manager = PluginManager()
        discovered = manager.discover("/nonexistent/path")
        assert discovered == []

    def test_discover_plugins_from_directory(self):
        """Test discovering plugins from directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin file
            plugin_code = '''
from core.plugins import Plugin, PluginInfo

class DiscoveredPlugin(Plugin):
    @property
    def info(self):
        return PluginInfo(name="discovered-plugin", version="1.0.0")

    def activate(self, context):
        pass
'''
            plugin_path = Path(tmpdir) / "test_plugin.py"
            plugin_path.write_text(plugin_code)

            manager = PluginManager()
            discovered = manager.discover(tmpdir)

            assert "discovered-plugin" in discovered


# =============================================================================
# Tests: Global Plugin Manager
# =============================================================================

class TestGlobalPluginManager:
    """Tests for global plugin manager singleton"""

    def test_get_plugin_manager(self):
        """Test getting global plugin manager"""
        manager = get_plugin_manager()
        assert manager is not None
        assert isinstance(manager, PluginManager)

    def test_same_instance(self):
        """Test same instance returned"""
        m1 = get_plugin_manager()
        m2 = get_plugin_manager()
        assert m1 is m2

    def test_set_plugin_manager(self):
        """Test setting custom global manager"""
        original = get_plugin_manager()
        custom = PluginManager()

        set_plugin_manager(custom)
        assert get_plugin_manager() is custom

        # Restore original
        set_plugin_manager(original)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
