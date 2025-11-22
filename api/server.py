"""
NPCPU REST API Server

FastAPI-based server for external interaction with NPCPU simulations.

Features:
- REST API for simulation management
- WebSocket support for real-time updates
- Pydantic models for request/response validation
- Rate limiting middleware
- CORS support

Example:
    # Run the server
    python -m api.server

    # Or with uvicorn
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

    # Create a simulation
    curl -X POST http://localhost:8000/simulations \\
         -H "Content-Type: application/json" \\
         -d '{"name": "Test", "initial_organisms": 10}'

    # Get simulation status
    curl http://localhost:8000/simulations/{id}

    # Advance simulation
    curl -X POST http://localhost:8000/simulations/{id}/tick \\
         -H "Content-Type: application/json" \\
         -d '{"ticks": 10}'
"""

from typing import Dict, Set, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .routes import router, _simulations, SimulationInstance
from .models import WebSocketMessage, SubscriptionRequest, ErrorResponse

# Import core events
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.events import get_event_bus, Event


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("npcpu.api")


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time simulation updates.

    Handles:
    - Connection lifecycle
    - Message broadcasting
    - Subscription management
    - Per-simulation channels
    """

    def __init__(self):
        # Connections per simulation: simulation_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Subscription preferences per connection
        self.subscriptions: Dict[WebSocket, SubscriptionRequest] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        simulation_id: str,
        subscription: Optional[SubscriptionRequest] = None
    ) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()

        async with self._lock:
            if simulation_id not in self.active_connections:
                self.active_connections[simulation_id] = set()
            self.active_connections[simulation_id].add(websocket)
            self.subscriptions[websocket] = subscription or SubscriptionRequest()

        logger.info(f"WebSocket connected to simulation {simulation_id}")

    async def disconnect(self, websocket: WebSocket, simulation_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if simulation_id in self.active_connections:
                self.active_connections[simulation_id].discard(websocket)
                if not self.active_connections[simulation_id]:
                    del self.active_connections[simulation_id]
            self.subscriptions.pop(websocket, None)

        logger.info(f"WebSocket disconnected from simulation {simulation_id}")

    async def broadcast(
        self,
        simulation_id: str,
        message: WebSocketMessage
    ) -> None:
        """Broadcast a message to all connections for a simulation."""
        if simulation_id not in self.active_connections:
            return

        disconnected = set()
        message_data = message.model_dump_json()

        for websocket in self.active_connections[simulation_id]:
            # Check if this connection is subscribed to this event type
            subscription = self.subscriptions.get(websocket)
            if subscription and message.type not in subscription.event_types:
                continue

            try:
                await websocket.send_text(message_data)
            except Exception as e:
                logger.warning(f"Failed to send to websocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected sockets
        for ws in disconnected:
            await self.disconnect(ws, simulation_id)

    async def send_personal_message(
        self,
        websocket: WebSocket,
        message: WebSocketMessage
    ) -> None:
        """Send a message to a specific connection."""
        try:
            await websocket.send_text(message.model_dump_json())
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")

    def get_connection_count(self, simulation_id: str) -> int:
        """Get number of active connections for a simulation."""
        return len(self.active_connections.get(simulation_id, set()))


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter.

    Limits requests per client IP.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this client."""
        now = datetime.now()
        window_start = now.timestamp() - 60

        async with self._lock:
            if client_ip not in self.requests:
                self.requests[client_ip] = []

            # Clean old requests
            self.requests[client_ip] = [
                ts for ts in self.requests[client_ip]
                if ts > window_start
            ]

            if len(self.requests[client_ip]) >= self.requests_per_minute:
                return False

            self.requests[client_ip].append(now.timestamp())
            return True


rate_limiter = RateLimiter(requests_per_minute=120)


# =============================================================================
# Event Bus Integration
# =============================================================================

def setup_event_forwarding():
    """
    Set up forwarding of internal events to WebSocket clients.

    Subscribes to core events and broadcasts them to connected clients.
    """
    bus = get_event_bus()

    async def forward_event(event: Event):
        """Forward an event to WebSocket clients."""
        # Extract simulation_id from event
        sim_id = event.data.get("simulation_id")
        if not sim_id:
            # Try to extract from source
            if event.source and event.source.startswith("simulation:"):
                sim_id = event.source.split(":", 1)[1]

        if sim_id:
            message = WebSocketMessage(
                type=event.type,
                data=event.data,
                timestamp=event.timestamp
            )
            # Schedule the broadcast (non-blocking)
            asyncio.create_task(manager.broadcast(sim_id, message))

    def sync_handler(event: Event):
        """Sync wrapper to schedule async broadcast."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(forward_event(event))
        except RuntimeError:
            # No event loop running
            pass

    # Subscribe to relevant events
    event_types = [
        "simulation.tick",
        "simulation.created",
        "simulation.deleted",
        "organism.tick",
        "organism.died",
        "organism.spawned",
        "population.add",
        "population.remove",
        "world.*"
    ]

    for event_type in event_types:
        bus.subscribe(event_type, sync_handler)

    logger.info("Event forwarding configured")


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting NPCPU API server...")
    setup_event_forwarding()
    yield
    # Shutdown
    logger.info("Shutting down NPCPU API server...")
    # Clean up simulations
    for sim_id in list(_simulations.keys()):
        del _simulations[sim_id]


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="NPCPU API",
    description="""
    REST API for NPCPU (Non-Player Cognitive Processing Unit).

    Provides endpoints for:
    - Creating and managing simulations
    - Spawning and monitoring organisms
    - Querying world state
    - Triggering world events
    - Real-time updates via WebSocket

    ## Quick Start

    1. Create a simulation: `POST /simulations`
    2. Advance time: `POST /simulations/{id}/tick`
    3. Check organisms: `GET /simulations/{id}/organisms`
    4. Stream updates: `WebSocket /simulations/{id}/stream`
    """,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limited"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "code": "VALIDATION_ERROR",
            "details": exc.errors()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "code": f"HTTP_{exc.status_code}"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR"
        }
    )


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Apply rate limiting to requests."""
    client_ip = request.client.host if request.client else "unknown"

    if not await rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "code": "RATE_LIMITED",
                "details": {"retry_after": 60}
            }
        )

    response = await call_next(request)
    return response


# =============================================================================
# Include Routes
# =============================================================================

app.include_router(router)


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/simulations/{simulation_id}/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    simulation_id: str
):
    """
    WebSocket endpoint for real-time simulation updates.

    Connect to receive live events from a simulation.

    Message format:
    ```json
    {
        "type": "simulation.tick",
        "data": {"tick": 100, "organism_count": 50},
        "timestamp": "2024-01-01T00:00:00"
    }
    ```

    To subscribe to specific events, send a subscription message:
    ```json
    {
        "type": "subscribe",
        "event_types": ["simulation.tick", "organism.died"],
        "include_world_state": true
    }
    ```
    """
    # Verify simulation exists
    if simulation_id not in _simulations:
        await websocket.close(code=4004, reason="Simulation not found")
        return

    await manager.connect(websocket, simulation_id)

    # Send initial state
    simulation = _simulations[simulation_id]
    initial_message = WebSocketMessage(
        type="connected",
        data={
            "simulation_id": simulation_id,
            "simulation_name": simulation.name,
            "current_tick": simulation.tick_count,
            "organism_count": simulation.population.size
        }
    )
    await manager.send_personal_message(websocket, initial_message)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle subscription updates
                if message.get("type") == "subscribe":
                    subscription = SubscriptionRequest(
                        event_types=message.get("event_types", ["tick"]),
                        include_world_state=message.get("include_world_state", True),
                        include_organisms=message.get("include_organisms", False)
                    )
                    manager.subscriptions[websocket] = subscription

                    await manager.send_personal_message(
                        websocket,
                        WebSocketMessage(
                            type="subscribed",
                            data={"event_types": subscription.event_types}
                        )
                    )

                # Handle ping
                elif message.get("type") == "ping":
                    await manager.send_personal_message(
                        websocket,
                        WebSocketMessage(type="pong", data={})
                    )

                # Handle state request
                elif message.get("type") == "get_state":
                    if simulation_id in _simulations:
                        sim = _simulations[simulation_id]
                        await manager.send_personal_message(
                            websocket,
                            WebSocketMessage(
                                type="state",
                                data={
                                    "tick": sim.tick_count,
                                    "organism_count": sim.population.size,
                                    "status": sim.status.value
                                }
                            )
                        )

            except json.JSONDecodeError:
                await manager.send_personal_message(
                    websocket,
                    WebSocketMessage(
                        type="error",
                        data={"message": "Invalid JSON"}
                    )
                )

    except WebSocketDisconnect:
        await manager.disconnect(websocket, simulation_id)


# =============================================================================
# Entry Point
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NPCPU API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
