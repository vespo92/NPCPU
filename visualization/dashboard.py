"""
Visualization Dashboard for NPCPU

A web-based dashboard for real-time simulation visualization.

Features:
- Real-time population graphs
- Organism location map (2D)
- Fitness/trait distribution charts
- Event timeline
- Control panel (pause/resume/speed)

Example:
    from visualization.dashboard import DashboardServer
    from core.simple_organism import SimplePopulation, SimpleOrganism

    # Create simulation
    population = SimplePopulation("Test Population")
    for i in range(10):
        population.add(SimpleOrganism(f"Organism_{i}"))

    # Start dashboard
    dashboard = DashboardServer(host="localhost", port=8080)
    dashboard.register_simulation("main", population)
    dashboard.start()

    # In a loop:
    while True:
        population.tick()
        dashboard.broadcast_state("main")
"""

import asyncio
import json
import os
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from core.events import get_event_bus, Event


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SimulationState:
    """Snapshot of simulation state for dashboard"""
    tick: int = 0
    population_size: int = 0
    alive_count: int = 0
    avg_age: float = 0.0
    avg_energy: float = 0.0
    avg_health: float = 0.0
    total_births: int = 0
    total_deaths: int = 0
    organisms: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DashboardConfig:
    """Configuration for dashboard server"""
    host: str = "localhost"
    port: int = 8080
    static_dir: Optional[str] = None
    max_events: int = 100
    broadcast_interval: float = 0.1  # seconds


# =============================================================================
# Dashboard Server
# =============================================================================

class DashboardServer:
    """
    Web dashboard for simulation visualization.

    Features:
    - Real-time population graphs
    - Organism location map (2D)
    - Fitness/trait distribution charts
    - Event timeline
    - Control panel (pause/resume/speed)

    Uses aiohttp for WebSocket support and async HTTP serving.

    Example:
        dashboard = DashboardServer(host="localhost", port=8080)
        dashboard.register_simulation("main", population)
        dashboard.start()

        # In simulation loop:
        while running:
            population.tick()
            dashboard.broadcast_state("main")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        static_dir: Optional[str] = None,
        max_events: int = 100
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for DashboardServer. "
                "Install with: pip install aiohttp"
            )

        self.config = DashboardConfig(
            host=host,
            port=port,
            static_dir=static_dir or str(Path(__file__).parent / "static"),
            max_events=max_events
        )

        # Registered simulations
        self._simulations: Dict[str, Any] = {}
        self._simulation_states: Dict[str, SimulationState] = {}
        self._tick_counts: Dict[str, int] = {}

        # WebSocket connections
        self._websockets: List[web.WebSocketResponse] = []

        # Event history
        self._events: List[Dict[str, Any]] = []

        # Control state
        self._paused: Dict[str, bool] = {}
        self._speed: Dict[str, float] = {}

        # Server state
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Subscribe to events
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Subscribe to relevant events"""
        bus = get_event_bus()

        # Track organism events
        bus.subscribe("organism.tick", self._on_organism_event)
        bus.subscribe("organism.died", self._on_organism_event)
        bus.subscribe("population.add", self._on_organism_event)
        bus.subscribe("population.remove", self._on_organism_event)

    def _on_organism_event(self, event: Event) -> None:
        """Handle organism events for timeline"""
        event_data = {
            "type": event.type,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "id": event.id
        }

        self._events.append(event_data)

        # Trim old events
        if len(self._events) > self.config.max_events:
            self._events = self._events[-self.config.max_events:]

    # -------------------------------------------------------------------------
    # Simulation Management
    # -------------------------------------------------------------------------

    def register_simulation(self, sim_id: str, population: Any) -> None:
        """
        Register a simulation/population for monitoring.

        Args:
            sim_id: Unique identifier for this simulation
            population: Population object with get_stats() and organisms attribute
        """
        self._simulations[sim_id] = population
        self._simulation_states[sim_id] = SimulationState()
        self._tick_counts[sim_id] = 0
        self._paused[sim_id] = False
        self._speed[sim_id] = 1.0

    def unregister_simulation(self, sim_id: str) -> None:
        """Remove a simulation from monitoring"""
        self._simulations.pop(sim_id, None)
        self._simulation_states.pop(sim_id, None)
        self._tick_counts.pop(sim_id, None)
        self._paused.pop(sim_id, None)
        self._speed.pop(sim_id, None)

    def get_simulation_state(self, sim_id: str) -> Optional[SimulationState]:
        """Get current state snapshot for a simulation"""
        if sim_id not in self._simulations:
            return None

        population = self._simulations[sim_id]
        stats = population.get_stats() if hasattr(population, 'get_stats') else {}

        # Collect organism data
        organisms = []
        if hasattr(population, 'organisms'):
            for org in population.organisms.values():
                org_data = {
                    "id": org.id,
                    "name": org.name,
                    "age": org.age,
                    "phase": org.phase.name if hasattr(org.phase, 'name') else str(org.phase),
                    "alive": org.is_alive,
                    "traits": org.traits,
                    "position": getattr(org, 'position', {"x": 0, "y": 0})
                }

                # Add subsystem info
                energy = org.get_subsystem("energy")
                health = org.get_subsystem("health")
                if energy:
                    org_data["energy"] = energy.percentage
                if health:
                    org_data["health"] = health.percentage

                organisms.append(org_data)

        self._tick_counts[sim_id] = self._tick_counts.get(sim_id, 0) + 1

        state = SimulationState(
            tick=self._tick_counts[sim_id],
            population_size=stats.get("size", len(organisms)),
            alive_count=stats.get("alive", len([o for o in organisms if o.get("alive", True)])),
            avg_age=stats.get("avg_age", 0),
            avg_energy=stats.get("avg_energy", 0),
            avg_health=stats.get("avg_health", 0),
            total_births=stats.get("total_births", 0),
            total_deaths=stats.get("total_deaths", 0),
            organisms=organisms,
            events=self._events[-20:]  # Last 20 events
        )

        self._simulation_states[sim_id] = state
        return state

    # -------------------------------------------------------------------------
    # WebSocket Broadcasting
    # -------------------------------------------------------------------------

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        """Send message to all connected WebSocket clients"""
        if not self._websockets:
            return

        data = json.dumps(message)

        # Send to all clients, remove disconnected ones
        disconnected = []
        for ws in self._websockets:
            try:
                await ws.send_str(data)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self._websockets.remove(ws)

    def broadcast_state(self, sim_id: str) -> None:
        """
        Broadcast current simulation state to all clients.

        Call this in your simulation loop after each tick.
        """
        state = self.get_simulation_state(sim_id)
        if state and self._loop:
            message = {
                "type": "state_update",
                "sim_id": sim_id,
                "state": asdict(state)
            }
            asyncio.run_coroutine_threadsafe(
                self._broadcast(message),
                self._loop
            )

    # -------------------------------------------------------------------------
    # Control Methods
    # -------------------------------------------------------------------------

    def is_paused(self, sim_id: str) -> bool:
        """Check if simulation is paused"""
        return self._paused.get(sim_id, False)

    def get_speed(self, sim_id: str) -> float:
        """Get simulation speed multiplier"""
        return self._speed.get(sim_id, 1.0)

    def pause(self, sim_id: str) -> None:
        """Pause a simulation"""
        self._paused[sim_id] = True
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({
                    "type": "control_update",
                    "sim_id": sim_id,
                    "paused": True,
                    "speed": self._speed.get(sim_id, 1.0)
                }),
                self._loop
            )

    def resume(self, sim_id: str) -> None:
        """Resume a simulation"""
        self._paused[sim_id] = False
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({
                    "type": "control_update",
                    "sim_id": sim_id,
                    "paused": False,
                    "speed": self._speed.get(sim_id, 1.0)
                }),
                self._loop
            )

    def set_speed(self, sim_id: str, speed: float) -> None:
        """Set simulation speed multiplier"""
        self._speed[sim_id] = max(0.1, min(10.0, speed))
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({
                    "type": "control_update",
                    "sim_id": sim_id,
                    "paused": self._paused.get(sim_id, False),
                    "speed": self._speed[sim_id]
                }),
                self._loop
            )

    # -------------------------------------------------------------------------
    # HTTP Routes
    # -------------------------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the main dashboard page"""
        index_path = Path(self.config.static_dir) / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="Dashboard not found", status=404)

    async def _handle_api_simulations(self, request: web.Request) -> web.Response:
        """List all registered simulations"""
        sims = []
        for sim_id in self._simulations:
            state = self._simulation_states.get(sim_id)
            sims.append({
                "id": sim_id,
                "paused": self._paused.get(sim_id, False),
                "speed": self._speed.get(sim_id, 1.0),
                "tick": state.tick if state else 0,
                "population_size": state.population_size if state else 0
            })
        return web.json_response({"simulations": sims})

    async def _handle_api_state(self, request: web.Request) -> web.Response:
        """Get current state of a simulation"""
        sim_id = request.match_info.get("sim_id", "main")
        state = self.get_simulation_state(sim_id)
        if state:
            return web.json_response(asdict(state))
        return web.json_response({"error": "Simulation not found"}, status=404)

    async def _handle_api_control(self, request: web.Request) -> web.Response:
        """Handle control commands (pause/resume/speed)"""
        sim_id = request.match_info.get("sim_id", "main")

        if sim_id not in self._simulations:
            return web.json_response({"error": "Simulation not found"}, status=404)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if "paused" in data:
            if data["paused"]:
                self.pause(sim_id)
            else:
                self.resume(sim_id)

        if "speed" in data:
            self.set_speed(sim_id, float(data["speed"]))

        return web.json_response({
            "sim_id": sim_id,
            "paused": self._paused.get(sim_id, False),
            "speed": self._speed.get(sim_id, 1.0)
        })

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._websockets.append(ws)

        # Send initial state
        for sim_id in self._simulations:
            state = self.get_simulation_state(sim_id)
            if state:
                await ws.send_json({
                    "type": "state_update",
                    "sim_id": sim_id,
                    "state": asdict(state)
                })

        # Handle incoming messages
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, data)
                    except json.JSONDecodeError:
                        pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            if ws in self._websockets:
                self._websockets.remove(ws)

        return ws

    async def _handle_ws_message(
        self,
        ws: web.WebSocketResponse,
        data: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket messages"""
        msg_type = data.get("type")
        sim_id = data.get("sim_id", "main")

        if msg_type == "pause":
            self.pause(sim_id)
        elif msg_type == "resume":
            self.resume(sim_id)
        elif msg_type == "set_speed":
            self.set_speed(sim_id, data.get("speed", 1.0))
        elif msg_type == "get_state":
            state = self.get_simulation_state(sim_id)
            if state:
                await ws.send_json({
                    "type": "state_update",
                    "sim_id": sim_id,
                    "state": asdict(state)
                })

    # -------------------------------------------------------------------------
    # Server Lifecycle
    # -------------------------------------------------------------------------

    def _create_app(self) -> web.Application:
        """Create the aiohttp application"""
        app = web.Application()

        # API routes
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/simulations", self._handle_api_simulations)
        app.router.add_get("/api/simulations/{sim_id}", self._handle_api_state)
        app.router.add_post("/api/simulations/{sim_id}/control", self._handle_api_control)
        app.router.add_get("/ws", self._handle_websocket)

        # Static files
        static_dir = Path(self.config.static_dir)
        if static_dir.exists():
            app.router.add_static("/static", static_dir)

        return app

    async def _start_server(self) -> None:
        """Start the HTTP server"""
        self._app = self._create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            self.config.host,
            self.config.port
        )
        await self._site.start()

    async def _stop_server(self) -> None:
        """Stop the HTTP server"""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    def _run_server(self) -> None:
        """Run the server in a thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._loop.run_until_complete(self._start_server())
        self._running = True

        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._stop_server())
            self._loop.close()

    def start(self, blocking: bool = False) -> None:
        """
        Start the dashboard server.

        Args:
            blocking: If True, block the current thread.
                     If False, run in a background thread.
        """
        if blocking:
            self._run_server()
        else:
            self._thread = threading.Thread(target=self._run_server, daemon=True)
            self._thread.start()

            # Wait for server to start
            import time
            while not self._running:
                time.sleep(0.01)

            print(f"Dashboard running at http://{self.config.host}:{self.config.port}")

    def stop(self) -> None:
        """Stop the dashboard server"""
        self._running = False

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def url(self) -> str:
        """Get the dashboard URL"""
        return f"http://{self.config.host}:{self.config.port}"


# =============================================================================
# Standalone Utility Functions
# =============================================================================

def create_dashboard(
    host: str = "localhost",
    port: int = 8080
) -> DashboardServer:
    """
    Create a dashboard server instance.

    Example:
        dashboard = create_dashboard()
        dashboard.register_simulation("main", my_population)
        dashboard.start()
    """
    return DashboardServer(host=host, port=port)


def run_with_dashboard(
    population: Any,
    host: str = "localhost",
    port: int = 8080,
    ticks: int = 1000,
    tick_callback: Optional[Callable[[int], None]] = None
) -> None:
    """
    Run a simulation with dashboard visualization.

    Args:
        population: Population to simulate
        host: Dashboard host
        port: Dashboard port
        ticks: Number of ticks to run
        tick_callback: Optional callback after each tick

    Example:
        from core.simple_organism import SimplePopulation, SimpleOrganism

        pop = SimplePopulation()
        for i in range(20):
            pop.add(SimpleOrganism(f"Org_{i}"))

        run_with_dashboard(pop, ticks=500)
    """
    import time

    dashboard = DashboardServer(host=host, port=port)
    dashboard.register_simulation("main", population)
    dashboard.start()

    print(f"Dashboard: {dashboard.url}")
    print(f"Running {ticks} ticks...")

    try:
        for tick in range(ticks):
            if not dashboard.is_paused("main"):
                population.tick()
                dashboard.broadcast_state("main")

            if tick_callback:
                tick_callback(tick)

            # Respect speed setting
            speed = dashboard.get_speed("main")
            time.sleep(0.05 / speed)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dashboard.stop()
