"""
NPCPU Real-Time Monitoring Server

WebSocket-based server for real-time simulation monitoring.
Provides:
- Live simulation metrics
- Population statistics
- Organism tracking
- Event notifications
- REST API for queries

Usage:
    # Start server
    python -m monitoring.server --port 8765

    # Connect via WebSocket
    ws://localhost:8765/ws

    # REST API
    GET http://localhost:8765/api/status
    GET http://localhost:8765/api/organisms
    GET http://localhost:8765/api/metrics
"""

import asyncio
import json
import time
from typing import Dict, Any, Set, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import websockets, fall back to basic HTTP server
try:
    import websockets
    from websockets.server import serve as websocket_serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Warning: websockets not installed. Install with: pip install websockets")

# Try to import asyncio HTTP server components
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


class MessageType(str, Enum):
    """Types of WebSocket messages"""
    # Server -> Client
    STATUS = "status"
    METRICS = "metrics"
    TICK = "tick"
    ORGANISM_BORN = "organism_born"
    ORGANISM_DIED = "organism_died"
    POPULATION = "population"
    GENERATION = "generation"
    WORLD_EVENT = "world_event"
    ERROR = "error"

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    QUERY = "query"
    COMMAND = "command"


@dataclass
class MonitoringState:
    """Current state being monitored"""
    simulation_id: Optional[str] = None
    simulation_name: str = "No Simulation"
    tick: int = 0
    population_count: int = 0
    state: str = "idle"
    start_time: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)


class SimulationMonitor:
    """
    Monitors a running simulation and broadcasts updates.

    Attaches to a Simulation instance and publishes events via WebSocket.
    """

    def __init__(self, simulation=None):
        self.simulation = simulation
        self.state = MonitoringState()
        self.subscribers: Set = set()
        self.event_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 100

        if simulation:
            self.attach(simulation)

    def attach(self, simulation):
        """Attach to a simulation instance"""
        self.simulation = simulation
        self.state.simulation_id = simulation.id
        self.state.simulation_name = simulation.config.name

        # Register callbacks
        simulation.on_tick(self._on_tick)
        simulation.on_organism_born(self._on_organism_born)
        simulation.on_organism_died(self._on_organism_died)
        simulation.on_completed(self._on_completed)

    def _on_tick(self, tick: int):
        """Called every tick"""
        self.state.tick = tick
        self.state.population_count = len(self.simulation.population.organisms)
        self.state.state = self.simulation.state.value

        # Broadcast tick update (every 10 ticks to reduce noise)
        if tick % 10 == 0:
            self._broadcast({
                "type": MessageType.TICK.value,
                "tick": tick,
                "population": self.state.population_count,
                "fitness": self.simulation.metrics.avg_fitness
            })

    def _on_organism_born(self, organism):
        """Called when organism is born"""
        event = {
            "type": MessageType.ORGANISM_BORN.value,
            "tick": self.state.tick,
            "organism_id": organism.id,
            "name": organism.name,
            "traits": organism.identity.traits
        }
        self._add_event(event)
        self._broadcast(event)

    def _on_organism_died(self, organism, cause):
        """Called when organism dies"""
        event = {
            "type": MessageType.ORGANISM_DIED.value,
            "tick": self.state.tick,
            "organism_id": organism.id,
            "name": organism.name,
            "cause": cause,
            "age": organism.age
        }
        self._add_event(event)
        self._broadcast(event)

    def _on_completed(self, metrics):
        """Called when simulation completes"""
        self.state.state = "completed"
        self._broadcast({
            "type": MessageType.STATUS.value,
            "state": "completed",
            "metrics": metrics.__dict__
        })

    def _add_event(self, event: Dict[str, Any]):
        """Add event to buffer"""
        event["timestamp"] = datetime.now().isoformat()
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)
        self.state.recent_events = self.event_buffer[-10:]

    def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all subscribers"""
        # Will be overridden by WebSocket server
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        if self.simulation:
            return {
                "simulation_id": self.simulation.id,
                "name": self.simulation.config.name,
                "state": self.simulation.state.value,
                "tick": self.simulation.tick_count,
                "population": len(self.simulation.population.organisms),
                "metrics": self.simulation.metrics.__dict__
            }
        return {"state": "idle", "message": "No simulation attached"}

    def get_organisms(self) -> List[Dict[str, Any]]:
        """Get all organisms"""
        if not self.simulation:
            return []

        organisms = []
        for org in self.simulation.population.organisms.values():
            organisms.append({
                "id": org.id,
                "name": org.name,
                "age": org.age,
                "state": org.state.value if hasattr(org.state, 'value') else str(org.state),
                "energy": org.metabolism.energy,
                "health": org.lifecycle.health,
                "traits": org.identity.traits
            })
        return organisms

    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics"""
        if not self.simulation:
            return {}

        return {
            "tick": self.simulation.tick_count,
            "metrics": self.simulation.metrics.__dict__,
            "population_history": self.simulation.population_history[-100:],  # Last 100 points
            "fitness_history": self.simulation.fitness_history[-100:]
        }


class MonitoringServer:
    """
    WebSocket server for real-time monitoring.

    Example:
        server = MonitoringServer(port=8765)
        server.attach_simulation(simulation)
        asyncio.run(server.start())
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.monitor = SimulationMonitor()
        self.clients: Set = set()
        self.running = False

    def attach_simulation(self, simulation):
        """Attach a simulation to monitor"""
        self.monitor.attach(simulation)
        # Override broadcast method
        self.monitor._broadcast = self._broadcast_sync

    def _broadcast_sync(self, message: Dict[str, Any]):
        """Synchronous broadcast wrapper"""
        if self.clients:
            asyncio.create_task(self._broadcast(message))

    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if self.clients:
            message_str = json.dumps(message, default=str)
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients],
                return_exceptions=True
            )

    async def _handle_client(self, websocket):
        """Handle a WebSocket client connection"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address

        print(f"Client connected: {client_addr}")

        # Send initial status
        await websocket.send(json.dumps({
            "type": MessageType.STATUS.value,
            **self.monitor.get_status()
        }, default=str))

        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected: {client_addr}")

    async def _handle_message(self, websocket, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == MessageType.QUERY.value:
                query = data.get("query")
                response = await self._handle_query(query)
                await websocket.send(json.dumps(response, default=str))

            elif msg_type == MessageType.COMMAND.value:
                command = data.get("command")
                response = await self._handle_command(command, data)
                await websocket.send(json.dumps(response, default=str))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": MessageType.ERROR.value,
                "error": "Invalid JSON"
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                "type": MessageType.ERROR.value,
                "error": str(e)
            }))

    async def _handle_query(self, query: str) -> Dict[str, Any]:
        """Handle query requests"""
        if query == "status":
            return {"type": "status", **self.monitor.get_status()}
        elif query == "organisms":
            return {"type": "organisms", "data": self.monitor.get_organisms()}
        elif query == "metrics":
            return {"type": "metrics", **self.monitor.get_metrics()}
        elif query == "events":
            return {"type": "events", "data": self.monitor.event_buffer}
        else:
            return {"type": "error", "error": f"Unknown query: {query}"}

    async def _handle_command(self, command: str, data: Dict) -> Dict[str, Any]:
        """Handle command requests"""
        if not self.monitor.simulation:
            return {"type": "error", "error": "No simulation attached"}

        if command == "pause":
            self.monitor.simulation.pause()
            return {"type": "status", "state": "paused"}
        elif command == "resume":
            self.monitor.simulation.resume()
            return {"type": "status", "state": "running"}
        else:
            return {"type": "error", "error": f"Unknown command: {command}"}

    async def start(self):
        """Start the WebSocket server"""
        if not HAS_WEBSOCKETS:
            print("WebSocket support requires 'websockets' package")
            print("Install with: pip install websockets")
            return

        self.running = True
        print(f"Starting monitoring server on ws://{self.host}:{self.port}")

        async with websocket_serve(self._handle_client, self.host, self.port):
            print(f"Monitoring server running at ws://{self.host}:{self.port}")
            while self.running:
                await asyncio.sleep(1)

    def stop(self):
        """Stop the server"""
        self.running = False


class SimpleHTTPMonitor(BaseHTTPRequestHandler):
    """Simple HTTP endpoint for monitoring (fallback when websockets unavailable)"""

    monitor: Optional[SimulationMonitor] = None

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/api/status":
            self._send_json(self.monitor.get_status() if self.monitor else {})
        elif self.path == "/api/organisms":
            self._send_json({"organisms": self.monitor.get_organisms() if self.monitor else []})
        elif self.path == "/api/metrics":
            self._send_json(self.monitor.get_metrics() if self.monitor else {})
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404)

    def _send_json(self, data: Dict[str, Any]):
        """Send JSON response"""
        content = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Suppress logging"""
        pass


def start_http_monitor(monitor: SimulationMonitor, host: str = "localhost", port: int = 8080):
    """Start simple HTTP monitoring endpoint"""
    SimpleHTTPMonitor.monitor = monitor
    server = HTTPServer((host, port), SimpleHTTPMonitor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"HTTP monitoring available at http://{host}:{port}/api/status")
    return server


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NPCPU Monitoring Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    parser.add_argument("--http", action="store_true", help="Use HTTP instead of WebSocket")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP port")

    args = parser.parse_args()

    if args.http or not HAS_WEBSOCKETS:
        # Start HTTP server
        print("Starting HTTP monitoring server...")
        monitor = SimulationMonitor()
        server = start_http_monitor(monitor, args.host, args.http_port)

        print(f"\nEndpoints available:")
        print(f"  GET /api/status    - Simulation status")
        print(f"  GET /api/organisms - List all organisms")
        print(f"  GET /api/metrics   - Simulation metrics")
        print(f"  GET /health        - Health check")
        print(f"\nPress Ctrl+C to stop...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
    else:
        # Start WebSocket server
        print("Starting WebSocket monitoring server...")
        server = MonitoringServer(host=args.host, port=args.port)

        print(f"\nConnect via WebSocket: ws://{args.host}:{args.port}")
        print(f"Press Ctrl+C to stop...")

        try:
            asyncio.run(server.start())
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.stop()


if __name__ == "__main__":
    main()
