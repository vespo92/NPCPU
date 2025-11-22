"""
NPCPU REST API Package

Provides a FastAPI-based REST API for external interaction with NPCPU simulations.

Components:
- server.py: FastAPI application with WebSocket support
- routes.py: API endpoint definitions
- models.py: Pydantic request/response models

Quick Start:
    # Run the server
    python -m api.server --port 8000

    # Or import and use programmatically
    from api.server import app, run_server
    run_server(port=8000)

API Endpoints:
    GET  /simulations              - List running simulations
    POST /simulations              - Create new simulation
    GET  /simulations/{id}         - Get simulation status
    POST /simulations/{id}/tick    - Advance simulation
    GET  /simulations/{id}/organisms    - List organisms
    GET  /simulations/{id}/organisms/{org_id} - Get organism details
    POST /simulations/{id}/organisms    - Spawn organism
    GET  /simulations/{id}/world   - Get world state
    POST /simulations/{id}/events  - Trigger world event
    WS   /simulations/{id}/stream  - Real-time updates
"""

from .models import (
    SimulationCreate,
    SimulationResponse,
    OrganismCreate,
    OrganismResponse,
    WorldStateResponse,
    WorldEventCreate,
    WorldEventResponse,
)

from .routes import router

__all__ = [
    "router",
    "SimulationCreate",
    "SimulationResponse",
    "OrganismCreate",
    "OrganismResponse",
    "WorldStateResponse",
    "WorldEventCreate",
    "WorldEventResponse",
]
