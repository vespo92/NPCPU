"""
Pydantic Models for NPCPU REST API

Defines request/response schemas for:
- Simulations
- Organisms
- World state
- Events

Example:
    from api.models import SimulationCreate, OrganismResponse

    # Create a simulation
    request = SimulationCreate(name="Test Sim", world_size=100)

    # Serialize an organism
    response = OrganismResponse.from_organism(some_organism)
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class SimulationStatus(str, Enum):
    """Simulation status states"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class LifecyclePhaseEnum(str, Enum):
    """Organism lifecycle phases"""
    INITIALIZING = "INITIALIZING"
    NASCENT = "NASCENT"
    DEVELOPING = "DEVELOPING"
    MATURE = "MATURE"
    DECLINING = "DECLINING"
    TERMINAL = "TERMINAL"
    ENDED = "ENDED"


# =============================================================================
# Simulation Models
# =============================================================================

class SimulationCreate(BaseModel):
    """Request model for creating a new simulation"""
    name: str = Field(..., min_length=1, max_length=100, description="Simulation name")
    world_size: int = Field(default=100, ge=10, le=10000, description="World grid size")
    initial_organisms: int = Field(default=10, ge=0, le=1000, description="Initial organism count")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "My Simulation",
                "world_size": 100,
                "initial_organisms": 10,
                "config": {"tick_rate": 1.0}
            }]
        }
    }


class SimulationResponse(BaseModel):
    """Response model for simulation data"""
    id: str = Field(..., description="Unique simulation ID")
    name: str = Field(..., description="Simulation name")
    status: SimulationStatus = Field(..., description="Current status")
    tick_count: int = Field(default=0, description="Number of ticks elapsed")
    organism_count: int = Field(default=0, description="Current organism count")
    created_at: datetime = Field(..., description="Creation timestamp")
    world_size: int = Field(..., description="World grid size")

    model_config = {"from_attributes": True}


class SimulationListResponse(BaseModel):
    """Response model for listing simulations"""
    simulations: List[SimulationResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of simulations")


class SimulationTickRequest(BaseModel):
    """Request model for advancing simulation"""
    ticks: int = Field(default=1, ge=1, le=1000, description="Number of ticks to advance")


class SimulationTickResponse(BaseModel):
    """Response model after advancing simulation"""
    simulation_id: str
    ticks_advanced: int
    current_tick: int
    events_generated: int = 0


# =============================================================================
# Organism Models
# =============================================================================

class OrganismCreate(BaseModel):
    """Request model for spawning a new organism"""
    name: Optional[str] = Field(default=None, max_length=100, description="Organism name")
    traits: Dict[str, float] = Field(default_factory=dict, description="Initial traits")
    position: Optional[Dict[str, float]] = Field(
        default=None,
        description="Spawn position {x, y}"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "Alpha",
                "traits": {"vitality": 1.0, "aggression": 0.5},
                "position": {"x": 50.0, "y": 50.0}
            }]
        }
    }


class SubsystemResponse(BaseModel):
    """Response model for organism subsystem"""
    name: str
    enabled: bool
    state: Dict[str, Any]


class OrganismResponse(BaseModel):
    """Response model for organism data"""
    id: str = Field(..., description="Unique organism ID")
    name: str = Field(..., description="Organism name")
    phase: LifecyclePhaseEnum = Field(..., description="Lifecycle phase")
    age: int = Field(..., description="Age in ticks")
    is_alive: bool = Field(..., description="Whether organism is alive")
    traits: Dict[str, float] = Field(default_factory=dict, description="Organism traits")
    capabilities: Dict[str, float] = Field(default_factory=dict, description="Capabilities")
    subsystems: Dict[str, SubsystemResponse] = Field(
        default_factory=dict,
        description="Subsystem states"
    )

    model_config = {"from_attributes": True}

    @classmethod
    def from_organism(cls, organism) -> "OrganismResponse":
        """Create response from organism instance"""
        subsystems = {}
        for name, sub in organism.subsystems.items():
            subsystems[name] = SubsystemResponse(
                name=name,
                enabled=sub.enabled,
                state=sub.get_state()
            )

        return cls(
            id=organism.id,
            name=organism.name,
            phase=LifecyclePhaseEnum(organism.phase.name),
            age=organism.age,
            is_alive=organism.is_alive,
            traits=organism.traits,
            capabilities={k.value: v for k, v in organism.capabilities.items()},
            subsystems=subsystems
        )


class OrganismListResponse(BaseModel):
    """Response model for listing organisms"""
    organisms: List[OrganismResponse] = Field(default_factory=list)
    total: int = Field(default=0)
    alive: int = Field(default=0)


# =============================================================================
# World Models
# =============================================================================

class WorldStateResponse(BaseModel):
    """Response model for world state"""
    id: str
    name: str
    tick_count: int
    size: int
    resources: Dict[str, float] = Field(default_factory=dict)
    populations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Event Models
# =============================================================================

class WorldEventCreate(BaseModel):
    """Request model for triggering a world event"""
    event_type: str = Field(..., min_length=1, description="Type of event to trigger")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    target_region: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Target region for the event"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "event_type": "natural_disaster",
                "data": {"severity": 0.8, "type": "drought"},
                "target_region": {"x": 50, "y": 50, "radius": 20}
            }]
        }
    }


class WorldEventResponse(BaseModel):
    """Response model for world event"""
    event_id: str
    event_type: str
    success: bool
    affected_organisms: int = 0
    message: str = ""


# =============================================================================
# Statistics Models
# =============================================================================

class PopulationStats(BaseModel):
    """Population statistics"""
    size: int
    alive: int
    avg_age: float
    avg_energy: float
    avg_health: float
    total_births: int
    total_deaths: int


class SimulationStats(BaseModel):
    """Comprehensive simulation statistics"""
    simulation_id: str
    tick_count: int
    uptime_seconds: float
    population: PopulationStats
    events_processed: int
    world_resources: Dict[str, float]


# =============================================================================
# WebSocket Models
# =============================================================================

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)


class SubscriptionRequest(BaseModel):
    """Request to subscribe to simulation updates"""
    event_types: List[str] = Field(
        default=["tick", "organism.created", "organism.died"],
        description="Event types to subscribe to"
    )
    include_world_state: bool = Field(default=True)
    include_organisms: bool = Field(default=False)


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    loc: List[str]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "Validation error"
    code: str = "VALIDATION_ERROR"
    details: List[ValidationErrorDetail]
