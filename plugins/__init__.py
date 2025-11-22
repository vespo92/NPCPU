"""
NPCPU Plugins

This directory contains plugins that extend NPCPU functionality.
Plugins can:
- Add new organism types
- Modify existing behaviors
- Add new subsystems
- Provide analytics and monitoring
- Integrate with external systems

To create a plugin:
1. Create a Python file in this directory
2. Define a class that extends Plugin
3. Implement the required methods
4. Register hooks to modify behavior

Example:
    class MyPlugin(Plugin):
        @property
        def info(self):
            return PluginInfo(name="my-plugin", version="1.0.0")

        def activate(self, context):
            context.hooks.register(HookPoint.ORGANISM_TICK, self.on_tick)
"""

from core.plugins import Plugin, PluginInfo, PluginContext, HookPoint, hook
