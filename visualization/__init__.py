"""
Visualization Module for NPCPU

Provides web-based dashboard for real-time simulation monitoring.

Example:
    from visualization import DashboardServer, run_with_dashboard
    from core.simple_organism import SimplePopulation, SimpleOrganism

    # Option 1: Manual control
    population = SimplePopulation("Test")
    for i in range(10):
        population.add(SimpleOrganism(f"Org_{i}"))

    dashboard = DashboardServer(host="localhost", port=8080)
    dashboard.register_simulation("main", population)
    dashboard.start()

    while True:
        population.tick()
        dashboard.broadcast_state("main")

    # Option 2: Simple runner
    run_with_dashboard(population, ticks=500)
"""

from .dashboard import (
    DashboardServer,
    DashboardConfig,
    SimulationState,
    create_dashboard,
    run_with_dashboard
)

__all__ = [
    'DashboardServer',
    'DashboardConfig',
    'SimulationState',
    'create_dashboard',
    'run_with_dashboard'
]
