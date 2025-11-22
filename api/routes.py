"""
API Routes for NPCPU REST API

Defines all REST endpoints for:
- Simulation management
- Organism operations
- World state queries
- Event triggering

Example:
    from fastapi import FastAPI
    from api.routes import router

    app = FastAPI()
    app.include_router(router)
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from datetime import datetime
import uuid

from .models import (
    SimulationCreate, SimulationResponse, SimulationListResponse,
    SimulationTickRequest, SimulationTickResponse,
    OrganismCreate, OrganismResponse, OrganismListResponse,
    WorldStateResponse, WorldEventCreate, WorldEventResponse,
    SimulationStats, PopulationStats, ErrorResponse,
    SimulationStatus
)

# Import core components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simple_organism import SimpleOrganism, SimplePopulation
from core.events import get_event_bus, Event


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(tags=["NPCPU API"])


# =============================================================================
# Simulation Storage (In-Memory)
# =============================================================================

class SimulationInstance:
    """Container for a running simulation"""

    def __init__(
        self,
        sim_id: str,
        name: str,
        world_size: int,
        config: Dict[str, Any]
    ):
        self.id = sim_id
        self.name = name
        self.status = SimulationStatus.CREATED
        self.created_at = datetime.now()
        self.tick_count = 0
        self.world_size = world_size
        self.config = config
        self.population = SimplePopulation(name=f"{name}_population")
        self.events_generated = 0
        self.resources: Dict[str, float] = {
            "food": 1000.0,
            "water": 1000.0,
            "energy": 500.0
        }

    def tick(self, count: int = 1) -> int:
        """Advance simulation by count ticks"""
        events_before = self.events_generated
        for _ in range(count):
            stimuli = self._generate_stimuli()
            self.population.tick(stimuli)
            self.tick_count += 1
            self._update_resources()
            self._emit_tick_event()
        return self.events_generated - events_before

    def _generate_stimuli(self) -> Dict[str, Any]:
        """Generate environmental stimuli for organisms"""
        import random
        return {
            "food_nearby": random.random() < 0.3,
            "threat_level": random.random() * 0.5,
            "temperature": 20 + random.random() * 10,
            "light_level": 0.8
        }

    def _update_resources(self) -> None:
        """Update world resources each tick"""
        # Resources regenerate slowly
        for resource in self.resources:
            self.resources[resource] = min(
                1000.0,
                self.resources[resource] + 0.5
            )

    def _emit_tick_event(self) -> None:
        """Emit simulation tick event"""
        bus = get_event_bus()
        bus.emit("simulation.tick", {
            "simulation_id": self.id,
            "tick": self.tick_count,
            "organism_count": self.population.size
        }, source=f"simulation:{self.id}")
        self.events_generated += 1

    def to_response(self) -> SimulationResponse:
        """Convert to API response model"""
        return SimulationResponse(
            id=self.id,
            name=self.name,
            status=self.status,
            tick_count=self.tick_count,
            organism_count=self.population.size,
            created_at=self.created_at,
            world_size=self.world_size
        )


# In-memory simulation storage
_simulations: Dict[str, SimulationInstance] = {}


def get_simulation(sim_id: str) -> SimulationInstance:
    """Dependency to get simulation by ID"""
    if sim_id not in _simulations:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation '{sim_id}' not found"
        )
    return _simulations[sim_id]


# =============================================================================
# Simulation Endpoints
# =============================================================================

@router.get(
    "/simulations",
    response_model=SimulationListResponse,
    summary="List all simulations",
    description="Returns a list of all running simulations with their current status."
)
async def list_simulations(
    status: Optional[SimulationStatus] = Query(
        default=None,
        description="Filter by status"
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """
    List all running simulations.

    - **status**: Optional filter by simulation status
    - **limit**: Maximum number of results (default 100)
    - **offset**: Number of results to skip (default 0)
    """
    sims = list(_simulations.values())

    if status:
        sims = [s for s in sims if s.status == status]

    total = len(sims)
    sims = sims[offset:offset + limit]

    return SimulationListResponse(
        simulations=[s.to_response() for s in sims],
        total=total
    )


@router.post(
    "/simulations",
    response_model=SimulationResponse,
    status_code=201,
    summary="Create new simulation",
    description="Create and initialize a new simulation with specified parameters."
)
async def create_simulation(request: SimulationCreate):
    """
    Create a new simulation.

    - **name**: Name for the simulation
    - **world_size**: Size of the world grid (10-10000)
    - **initial_organisms**: Number of organisms to spawn initially
    - **config**: Additional configuration options
    """
    sim_id = str(uuid.uuid4())

    simulation = SimulationInstance(
        sim_id=sim_id,
        name=request.name,
        world_size=request.world_size,
        config=request.config
    )

    # Spawn initial organisms
    for i in range(request.initial_organisms):
        organism = SimpleOrganism(name=f"Organism_{i+1}")
        simulation.population.add(organism)

    simulation.status = SimulationStatus.RUNNING
    _simulations[sim_id] = simulation

    # Emit creation event
    bus = get_event_bus()
    bus.emit("simulation.created", {
        "simulation_id": sim_id,
        "name": request.name,
        "initial_organisms": request.initial_organisms
    })

    return simulation.to_response()


@router.get(
    "/simulations/{simulation_id}",
    response_model=SimulationResponse,
    summary="Get simulation status",
    description="Get detailed status of a specific simulation."
)
async def get_simulation_status(
    simulation_id: str = Path(..., description="Simulation ID"),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Get current status of a simulation."""
    return simulation.to_response()


@router.post(
    "/simulations/{simulation_id}/tick",
    response_model=SimulationTickResponse,
    summary="Advance simulation",
    description="Advance the simulation by the specified number of ticks."
)
async def tick_simulation(
    request: SimulationTickRequest,
    simulation_id: str = Path(..., description="Simulation ID"),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """
    Advance simulation by specified ticks.

    - **ticks**: Number of ticks to advance (1-1000)
    """
    if simulation.status == SimulationStatus.STOPPED:
        raise HTTPException(
            status_code=400,
            detail="Cannot advance stopped simulation"
        )

    events = simulation.tick(request.ticks)

    return SimulationTickResponse(
        simulation_id=simulation.id,
        ticks_advanced=request.ticks,
        current_tick=simulation.tick_count,
        events_generated=events
    )


@router.post(
    "/simulations/{simulation_id}/pause",
    response_model=SimulationResponse,
    summary="Pause simulation"
)
async def pause_simulation(
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Pause a running simulation."""
    if simulation.status != SimulationStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Can only pause running simulations"
        )
    simulation.status = SimulationStatus.PAUSED
    return simulation.to_response()


@router.post(
    "/simulations/{simulation_id}/resume",
    response_model=SimulationResponse,
    summary="Resume simulation"
)
async def resume_simulation(
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Resume a paused simulation."""
    if simulation.status != SimulationStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail="Can only resume paused simulations"
        )
    simulation.status = SimulationStatus.RUNNING
    return simulation.to_response()


@router.delete(
    "/simulations/{simulation_id}",
    status_code=204,
    summary="Stop and delete simulation"
)
async def delete_simulation(
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Stop and delete a simulation."""
    simulation.status = SimulationStatus.STOPPED

    bus = get_event_bus()
    bus.emit("simulation.deleted", {"simulation_id": simulation_id})

    del _simulations[simulation_id]
    return None


# =============================================================================
# Organism Endpoints
# =============================================================================

@router.get(
    "/simulations/{simulation_id}/organisms",
    response_model=OrganismListResponse,
    summary="List organisms",
    description="List all organisms in a simulation."
)
async def list_organisms(
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation),
    alive_only: bool = Query(default=False, description="Only return alive organisms"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """
    List organisms in a simulation.

    - **alive_only**: Filter to only living organisms
    - **limit**: Maximum results to return
    - **offset**: Results to skip
    """
    organisms = list(simulation.population.organisms.values())

    if alive_only:
        organisms = [o for o in organisms if o.is_alive]

    total = len(organisms)
    alive = sum(1 for o in organisms if o.is_alive)
    organisms = organisms[offset:offset + limit]

    return OrganismListResponse(
        organisms=[OrganismResponse.from_organism(o) for o in organisms],
        total=total,
        alive=alive
    )


@router.get(
    "/simulations/{simulation_id}/organisms/{organism_id}",
    response_model=OrganismResponse,
    summary="Get organism details",
    description="Get detailed information about a specific organism."
)
async def get_organism(
    organism_id: str = Path(..., description="Organism ID"),
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Get detailed information about an organism."""
    organism = simulation.population.organisms.get(organism_id)

    if not organism:
        raise HTTPException(
            status_code=404,
            detail=f"Organism '{organism_id}' not found"
        )

    return OrganismResponse.from_organism(organism)


@router.post(
    "/simulations/{simulation_id}/organisms",
    response_model=OrganismResponse,
    status_code=201,
    summary="Spawn organism",
    description="Spawn a new organism in the simulation."
)
async def spawn_organism(
    request: OrganismCreate,
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """
    Spawn a new organism.

    - **name**: Optional name for the organism
    - **traits**: Initial trait values
    - **position**: Spawn position (optional)
    """
    organism = SimpleOrganism(
        name=request.name or f"Organism_{simulation.population.size + 1}",
        traits=request.traits if request.traits else None
    )

    simulation.population.add(organism)

    bus = get_event_bus()
    bus.emit("organism.spawned", {
        "simulation_id": simulation_id,
        "organism_id": organism.id,
        "name": organism.name
    })

    return OrganismResponse.from_organism(organism)


@router.delete(
    "/simulations/{simulation_id}/organisms/{organism_id}",
    status_code=204,
    summary="Remove organism"
)
async def remove_organism(
    organism_id: str = Path(...),
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Remove an organism from the simulation."""
    organism = simulation.population.organisms.get(organism_id)

    if not organism:
        raise HTTPException(
            status_code=404,
            detail=f"Organism '{organism_id}' not found"
        )

    simulation.population.remove(organism_id)
    return None


# =============================================================================
# World Endpoints
# =============================================================================

@router.get(
    "/simulations/{simulation_id}/world",
    response_model=WorldStateResponse,
    summary="Get world state",
    description="Get the current state of the simulation world."
)
async def get_world_state(
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Get current world state including resources and populations."""
    return WorldStateResponse(
        id=simulation.id,
        name=f"{simulation.name}_world",
        tick_count=simulation.tick_count,
        size=simulation.world_size,
        resources=simulation.resources,
        populations=[simulation.population.name],
        metadata={
            "organism_count": simulation.population.size,
            "alive_count": simulation.population.alive_count
        }
    )


@router.post(
    "/simulations/{simulation_id}/events",
    response_model=WorldEventResponse,
    summary="Trigger world event",
    description="Trigger an environmental event in the simulation world."
)
async def trigger_world_event(
    request: WorldEventCreate,
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """
    Trigger a world event.

    - **event_type**: Type of event (e.g., 'natural_disaster', 'resource_spawn')
    - **data**: Event-specific data
    - **target_region**: Optional region to target
    """
    event_id = str(uuid.uuid4())
    affected = 0
    message = ""

    # Process different event types
    if request.event_type == "resource_spawn":
        resource_type = request.data.get("resource_type", "food")
        amount = request.data.get("amount", 100.0)
        simulation.resources[resource_type] = simulation.resources.get(resource_type, 0) + amount
        message = f"Spawned {amount} {resource_type}"

    elif request.event_type == "natural_disaster":
        severity = request.data.get("severity", 0.5)
        # Apply damage to organisms
        for organism in simulation.population.organisms.values():
            if organism.is_alive:
                health = organism.get_subsystem("health")
                if health:
                    damage = severity * 30
                    health.damage(damage)
                    affected += 1
        message = f"Disaster affected {affected} organisms"

    elif request.event_type == "food_abundance":
        # Restore energy to all organisms
        for organism in simulation.population.organisms.values():
            if organism.is_alive:
                energy = organism.get_subsystem("energy")
                if energy:
                    energy.restore(50)
                    affected += 1
        message = f"Fed {affected} organisms"

    else:
        message = f"Unknown event type: {request.event_type}"

    # Emit the event
    bus = get_event_bus()
    bus.emit(f"world.{request.event_type}", {
        "simulation_id": simulation_id,
        "event_id": event_id,
        **request.data
    })

    return WorldEventResponse(
        event_id=event_id,
        event_type=request.event_type,
        success=True,
        affected_organisms=affected,
        message=message
    )


# =============================================================================
# Statistics Endpoints
# =============================================================================

@router.get(
    "/simulations/{simulation_id}/stats",
    response_model=SimulationStats,
    summary="Get simulation statistics",
    description="Get comprehensive statistics for a simulation."
)
async def get_simulation_stats(
    simulation_id: str = Path(...),
    simulation: SimulationInstance = Depends(get_simulation)
):
    """Get detailed statistics for a simulation."""
    pop_stats = simulation.population.get_stats()

    uptime = (datetime.now() - simulation.created_at).total_seconds()

    return SimulationStats(
        simulation_id=simulation.id,
        tick_count=simulation.tick_count,
        uptime_seconds=uptime,
        population=PopulationStats(**pop_stats),
        events_processed=simulation.events_generated,
        world_resources=simulation.resources
    )


# =============================================================================
# Health Check
# =============================================================================

@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running."
)
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "simulations_count": len(_simulations)
    }
