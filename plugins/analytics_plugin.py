"""
Analytics Plugin

An example plugin that demonstrates the NPCPU plugin system.
Provides population analytics and statistics tracking.

Features:
- Tracks population metrics over time
- Records significant events
- Generates periodic reports
- Exports data to various formats
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.plugins import Plugin, PluginInfo, PluginContext, HookPoint, hook
from core.events import Event


@dataclass
class PopulationSnapshot:
    """A snapshot of population state"""
    tick: int
    timestamp: datetime
    population_size: int
    births: int
    deaths: int
    avg_age: float
    avg_energy: float
    avg_health: float
    events: List[str] = field(default_factory=list)


class AnalyticsPlugin(Plugin):
    """
    Population analytics and statistics tracking plugin.

    Usage:
        from plugins.analytics_plugin import AnalyticsPlugin
        from core.plugins import get_plugin_manager

        manager = get_plugin_manager()
        manager.register(AnalyticsPlugin())
        manager.activate("analytics")

        # Run simulation...

        # Get analytics
        plugin = manager.get("analytics")
        report = plugin.generate_report()
        plugin.export_json("analytics.json")
    """

    def __init__(self):
        self._snapshots: List[PopulationSnapshot] = []
        self._events: List[Dict[str, Any]] = []
        self._current_tick = 0
        self._snapshot_interval = 100
        self._births_this_interval = 0
        self._deaths_this_interval = 0
        self._context: Optional[PluginContext] = None

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="analytics",
            version="1.0.0",
            author="NPCPU Team",
            description="Population analytics and statistics tracking"
        )

    def activate(self, context: PluginContext) -> None:
        """Activate the plugin"""
        self._context = context

        # Register hooks
        context.hooks.register(
            HookPoint.SIMULATION_TICK,
            self._on_simulation_tick,
            plugin_name=self.info.name
        )

        context.hooks.register(
            HookPoint.ORGANISM_CREATE,
            self._on_organism_create,
            plugin_name=self.info.name
        )

        context.hooks.register(
            HookPoint.ORGANISM_DIE,
            self._on_organism_die,
            plugin_name=self.info.name
        )

        # Subscribe to events
        context.event_bus.subscribe(
            "organism.born",
            self._handle_birth_event,
            priority=0
        )

        context.event_bus.subscribe(
            "organism.died",
            self._handle_death_event,
            priority=0
        )

        # Register plugin in context
        context.register("analytics", self)

    def deactivate(self, context: PluginContext) -> None:
        """Deactivate the plugin"""
        # Hooks are automatically unregistered by PluginManager
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin"""
        self._snapshot_interval = config.get("snapshot_interval", 100)

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def _on_simulation_tick(self, simulation) -> None:
        """Called every simulation tick"""
        self._current_tick += 1

        # Take snapshot at intervals
        if self._current_tick % self._snapshot_interval == 0:
            self._take_snapshot(simulation)

    def _on_organism_create(self, organism) -> None:
        """Called when organism is created"""
        self._births_this_interval += 1
        self._events.append({
            "type": "birth",
            "tick": self._current_tick,
            "organism_id": organism.id if hasattr(organism, 'id') else str(id(organism)),
            "organism_name": organism.name if hasattr(organism, 'name') else "Unknown"
        })

    def _on_organism_die(self, organism, cause: str = "unknown") -> None:
        """Called when organism dies"""
        self._deaths_this_interval += 1
        self._events.append({
            "type": "death",
            "tick": self._current_tick,
            "organism_id": organism.id if hasattr(organism, 'id') else str(id(organism)),
            "organism_name": organism.name if hasattr(organism, 'name') else "Unknown",
            "cause": cause
        })

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _handle_birth_event(self, event: Event) -> None:
        """Handle birth event from event bus"""
        self._births_this_interval += 1

    def _handle_death_event(self, event: Event) -> None:
        """Handle death event from event bus"""
        self._deaths_this_interval += 1

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------

    def _take_snapshot(self, simulation) -> None:
        """Take a snapshot of current state"""
        # Try to get population stats
        pop_size = 0
        avg_age = 0.0
        avg_energy = 0.0
        avg_health = 0.0

        if hasattr(simulation, 'population'):
            pop = simulation.population
            if hasattr(pop, 'organisms'):
                organisms = list(pop.organisms.values())
                pop_size = len(organisms)

                if pop_size > 0:
                    ages = []
                    energies = []
                    healths = []

                    for org in organisms:
                        if hasattr(org, 'age'):
                            ages.append(org.age)
                        if hasattr(org, 'metabolism') and hasattr(org.metabolism, 'energy'):
                            energies.append(org.metabolism.energy)
                        if hasattr(org, 'lifecycle') and hasattr(org.lifecycle, 'health'):
                            healths.append(org.lifecycle.health)

                    avg_age = sum(ages) / len(ages) if ages else 0
                    avg_energy = sum(energies) / len(energies) if energies else 0
                    avg_health = sum(healths) / len(healths) if healths else 0

        snapshot = PopulationSnapshot(
            tick=self._current_tick,
            timestamp=datetime.now(),
            population_size=pop_size,
            births=self._births_this_interval,
            deaths=self._deaths_this_interval,
            avg_age=avg_age,
            avg_energy=avg_energy,
            avg_health=avg_health,
            events=[e["type"] for e in self._events[-10:]]
        )

        self._snapshots.append(snapshot)
        self._births_this_interval = 0
        self._deaths_this_interval = 0

    def record_snapshot(
        self,
        tick: int,
        population_size: int,
        births: int = 0,
        deaths: int = 0,
        avg_age: float = 0,
        avg_energy: float = 0,
        avg_health: float = 0
    ) -> None:
        """Manually record a snapshot"""
        self._snapshots.append(PopulationSnapshot(
            tick=tick,
            timestamp=datetime.now(),
            population_size=population_size,
            births=births,
            deaths=deaths,
            avg_age=avg_age,
            avg_energy=avg_energy,
            avg_health=avg_health
        ))

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    def get_snapshots(self) -> List[PopulationSnapshot]:
        """Get all recorded snapshots"""
        return self._snapshots.copy()

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all recorded events"""
        return self._events.copy()

    def generate_report(self) -> Dict[str, Any]:
        """Generate analytics report"""
        if not self._snapshots:
            return {"error": "No data collected"}

        pop_sizes = [s.population_size for s in self._snapshots]
        ages = [s.avg_age for s in self._snapshots if s.avg_age > 0]
        energies = [s.avg_energy for s in self._snapshots if s.avg_energy > 0]
        healths = [s.avg_health for s in self._snapshots if s.avg_health > 0]

        total_births = sum(s.births for s in self._snapshots)
        total_deaths = sum(s.deaths for s in self._snapshots)

        return {
            "summary": {
                "total_ticks": self._current_tick,
                "snapshots_recorded": len(self._snapshots),
                "events_recorded": len(self._events),
                "total_births": total_births,
                "total_deaths": total_deaths,
            },
            "population": {
                "min": min(pop_sizes) if pop_sizes else 0,
                "max": max(pop_sizes) if pop_sizes else 0,
                "mean": sum(pop_sizes) / len(pop_sizes) if pop_sizes else 0,
                "final": pop_sizes[-1] if pop_sizes else 0,
            },
            "age": {
                "min": min(ages) if ages else 0,
                "max": max(ages) if ages else 0,
                "mean": sum(ages) / len(ages) if ages else 0,
            },
            "energy": {
                "min": min(energies) if energies else 0,
                "max": max(energies) if energies else 0,
                "mean": sum(energies) / len(energies) if energies else 0,
            },
            "health": {
                "min": min(healths) if healths else 0,
                "max": max(healths) if healths else 0,
                "mean": sum(healths) / len(healths) if healths else 0,
            },
            "death_causes": self._analyze_death_causes(),
        }

    def _analyze_death_causes(self) -> Dict[str, int]:
        """Analyze causes of death"""
        causes = defaultdict(int)
        for event in self._events:
            if event["type"] == "death":
                cause = event.get("cause", "unknown")
                causes[cause] += 1
        return dict(causes)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_json(self, filepath: str) -> None:
        """Export analytics to JSON"""
        data = {
            "report": self.generate_report(),
            "snapshots": [
                {
                    "tick": s.tick,
                    "timestamp": s.timestamp.isoformat(),
                    "population_size": s.population_size,
                    "births": s.births,
                    "deaths": s.deaths,
                    "avg_age": s.avg_age,
                    "avg_energy": s.avg_energy,
                    "avg_health": s.avg_health,
                }
                for s in self._snapshots
            ],
            "events": self._events
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def export_csv(self, filepath: str) -> None:
        """Export snapshots to CSV"""
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick", "timestamp", "population_size", "births", "deaths",
                "avg_age", "avg_energy", "avg_health"
            ])

            for s in self._snapshots:
                writer.writerow([
                    s.tick, s.timestamp.isoformat(), s.population_size,
                    s.births, s.deaths, s.avg_age, s.avg_energy, s.avg_health
                ])

    def reset(self) -> None:
        """Reset all collected data"""
        self._snapshots.clear()
        self._events.clear()
        self._current_tick = 0
        self._births_this_interval = 0
        self._deaths_this_interval = 0


# =============================================================================
# Standalone Usage Example
# =============================================================================

if __name__ == "__main__":
    # Demonstrate plugin usage without full simulation
    from core.plugins import PluginManager, PluginContext, HookManager
    from core.events import EventBus

    # Create plugin infrastructure
    hooks = HookManager()
    event_bus = EventBus()
    context = PluginContext(hooks=hooks, event_bus=event_bus)

    # Create and activate plugin
    manager = PluginManager(hooks)
    manager.set_context(context)

    plugin = AnalyticsPlugin()
    manager.register(plugin)
    manager.activate("analytics", {"snapshot_interval": 10})

    print(f"Plugin activated: {plugin.info.name} v{plugin.info.version}")
    print(f"Description: {plugin.info.description}")

    # Simulate some events
    for i in range(50):
        # Emit events
        if i % 5 == 0:
            event_bus.emit("organism.born", {"name": f"Organism_{i}"})
        if i % 7 == 0:
            event_bus.emit("organism.died", {"name": f"Organism_{i}", "cause": "starvation"})

        # Manually record snapshots (normally done via hooks)
        if i % 10 == 0:
            plugin.record_snapshot(
                tick=i,
                population_size=20 + i // 5,
                births=i // 5,
                deaths=i // 7,
                avg_age=i * 0.5,
                avg_energy=80 - i * 0.3,
                avg_health=90 - i * 0.2
            )

    # Generate report
    report = plugin.generate_report()

    print("\n=== Analytics Report ===")
    print(f"Total ticks: {report['summary']['total_ticks']}")
    print(f"Snapshots: {report['summary']['snapshots_recorded']}")
    print(f"Total births: {report['summary']['total_births']}")
    print(f"Total deaths: {report['summary']['total_deaths']}")
    print(f"\nPopulation range: {report['population']['min']} - {report['population']['max']}")
    print(f"Average energy: {report['energy']['mean']:.2f}")
    print(f"\nDeath causes: {report['death_causes']}")
