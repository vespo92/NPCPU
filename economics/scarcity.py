"""
Scarcity Modeling and Competition for NPCPU Economic System

Implements resource scarcity, competition dynamics, and
allocation mechanisms under constrained supply.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ScarcityLevel(Enum):
    """Levels of resource scarcity."""
    ABUNDANT = "abundant"       # Supply >> Demand
    SUFFICIENT = "sufficient"   # Supply > Demand
    BALANCED = "balanced"       # Supply ≈ Demand
    SCARCE = "scarce"           # Supply < Demand
    CRITICAL = "critical"       # Supply << Demand
    DEPLETED = "depleted"       # No supply


class CompetitionMode(Enum):
    """Modes of competition for scarce resources."""
    FIRST_COME = "first_come"           # First to request gets it
    LOTTERY = "lottery"                 # Random allocation
    AUCTION = "auction"                 # Highest bidder wins
    PRIORITY = "priority"               # Based on priority scores
    PROPORTIONAL = "proportional"       # Divided by demand ratio
    NEED_BASED = "need_based"           # Based on necessity


class AllocationStrategy(Enum):
    """Strategies for resource allocation."""
    GREEDY = "greedy"           # Maximize immediate utility
    FAIR = "fair"               # Equal distribution
    OPTIMAL = "optimal"         # System-wide optimization
    ADAPTIVE = "adaptive"       # Learn from outcomes


@dataclass
class ResourceSupply:
    """
    Tracks supply of a resource type.
    """
    resource_type: str
    total_supply: float = 0.0
    available_supply: float = 0.0
    reserved_supply: float = 0.0
    regeneration_rate: float = 0.0
    max_capacity: float = float('inf')
    last_updated: float = field(default_factory=time.time)

    @property
    def utilization(self) -> float:
        """Current utilization ratio."""
        if self.total_supply == 0:
            return 0.0
        return (self.total_supply - self.available_supply) / self.total_supply

    def add(self, amount: float) -> float:
        """Add to supply. Returns actual amount added."""
        can_add = min(amount, self.max_capacity - self.total_supply)
        self.total_supply += can_add
        self.available_supply += can_add
        self.last_updated = time.time()
        return can_add

    def consume(self, amount: float) -> float:
        """Consume from supply. Returns actual amount consumed."""
        can_consume = min(amount, self.available_supply)
        self.available_supply -= can_consume
        self.total_supply -= can_consume
        self.last_updated = time.time()
        return can_consume

    def reserve(self, amount: float) -> float:
        """Reserve supply. Returns actual amount reserved."""
        can_reserve = min(amount, self.available_supply)
        self.available_supply -= can_reserve
        self.reserved_supply += can_reserve
        self.last_updated = time.time()
        return can_reserve

    def release_reservation(self, amount: float) -> float:
        """Release reserved supply. Returns actual amount released."""
        can_release = min(amount, self.reserved_supply)
        self.reserved_supply -= can_release
        self.available_supply += can_release
        self.last_updated = time.time()
        return can_release

    def regenerate(self) -> float:
        """Apply regeneration. Returns amount regenerated."""
        if self.regeneration_rate <= 0:
            return 0.0
        can_regen = min(self.regeneration_rate, self.max_capacity - self.total_supply)
        self.total_supply += can_regen
        self.available_supply += can_regen
        self.last_updated = time.time()
        return can_regen


@dataclass
class ResourceDemand:
    """
    Tracks demand for a resource type.
    """
    resource_type: str
    total_demand: float = 0.0
    satisfied_demand: float = 0.0
    pending_requests: List[Dict[str, Any]] = field(default_factory=list)
    demand_history: List[float] = field(default_factory=list)

    @property
    def satisfaction_ratio(self) -> float:
        """Demand satisfaction ratio."""
        if self.total_demand == 0:
            return 1.0
        return self.satisfied_demand / self.total_demand

    @property
    def unmet_demand(self) -> float:
        """Amount of unmet demand."""
        return self.total_demand - self.satisfied_demand

    def add_request(
        self,
        requester_id: str,
        amount: float,
        priority: float = 0.5,
        necessity: float = 0.5
    ) -> str:
        """Add a demand request. Returns request ID."""
        request_id = str(uuid.uuid4())
        self.pending_requests.append({
            "id": request_id,
            "requester_id": requester_id,
            "amount": amount,
            "priority": priority,
            "necessity": necessity,
            "timestamp": time.time()
        })
        self.total_demand += amount
        return request_id

    def fulfill_request(self, request_id: str, amount: float) -> bool:
        """Mark a request as fulfilled."""
        for req in self.pending_requests:
            if req["id"] == request_id:
                fulfilled = min(amount, req["amount"])
                self.satisfied_demand += fulfilled
                req["amount"] -= fulfilled
                if req["amount"] <= 0:
                    self.pending_requests.remove(req)
                return True
        return False

    def record_demand(self) -> None:
        """Record current demand for history."""
        self.demand_history.append(self.total_demand)
        if len(self.demand_history) > 1000:
            self.demand_history = self.demand_history[-1000:]

    def average_demand(self, periods: int = 100) -> float:
        """Get average demand over periods."""
        if not self.demand_history:
            return self.total_demand
        history = self.demand_history[-periods:]
        return sum(history) / len(history)


@dataclass
class ScarcityState:
    """
    Scarcity state for a resource type.
    """
    resource_type: str
    supply: ResourceSupply
    demand: ResourceDemand
    level: ScarcityLevel = ScarcityLevel.BALANCED
    price_multiplier: float = 1.0
    competition_intensity: float = 0.0

    def update_level(self) -> ScarcityLevel:
        """Update scarcity level based on supply/demand."""
        if self.supply.available_supply <= 0:
            self.level = ScarcityLevel.DEPLETED
        elif self.demand.total_demand == 0:
            self.level = ScarcityLevel.ABUNDANT
        else:
            ratio = self.supply.available_supply / self.demand.total_demand
            if ratio > 2.0:
                self.level = ScarcityLevel.ABUNDANT
            elif ratio > 1.2:
                self.level = ScarcityLevel.SUFFICIENT
            elif ratio > 0.8:
                self.level = ScarcityLevel.BALANCED
            elif ratio > 0.3:
                self.level = ScarcityLevel.SCARCE
            else:
                self.level = ScarcityLevel.CRITICAL

        # Update competition intensity
        if self.level in (ScarcityLevel.SCARCE, ScarcityLevel.CRITICAL):
            self.competition_intensity = min(1.0, 1.0 / max(0.1, ratio))
        else:
            self.competition_intensity = 0.0

        # Update price multiplier
        scarcity_multipliers = {
            ScarcityLevel.ABUNDANT: 0.5,
            ScarcityLevel.SUFFICIENT: 0.8,
            ScarcityLevel.BALANCED: 1.0,
            ScarcityLevel.SCARCE: 1.5,
            ScarcityLevel.CRITICAL: 3.0,
            ScarcityLevel.DEPLETED: 10.0
        }
        self.price_multiplier = scarcity_multipliers[self.level]

        return self.level

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "resource_type": self.resource_type,
            "level": self.level.value,
            "available_supply": self.supply.available_supply,
            "total_demand": self.demand.total_demand,
            "satisfaction_ratio": self.demand.satisfaction_ratio,
            "price_multiplier": self.price_multiplier,
            "competition_intensity": self.competition_intensity
        }


class ResourceAllocator:
    """
    Allocates scarce resources among competing requesters.
    """

    def __init__(
        self,
        mode: CompetitionMode = CompetitionMode.PROPORTIONAL,
        strategy: AllocationStrategy = AllocationStrategy.FAIR
    ):
        self.mode = mode
        self.strategy = strategy
        self._allocation_history: List[Dict[str, Any]] = []

    def allocate(
        self,
        available: float,
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Allocate available resources among requests.

        Returns list of (request_id, allocated_amount) tuples.
        """
        if not requests or available <= 0:
            return []

        if self.mode == CompetitionMode.FIRST_COME:
            return self._allocate_first_come(available, requests)
        elif self.mode == CompetitionMode.LOTTERY:
            return self._allocate_lottery(available, requests)
        elif self.mode == CompetitionMode.PRIORITY:
            return self._allocate_priority(available, requests)
        elif self.mode == CompetitionMode.PROPORTIONAL:
            return self._allocate_proportional(available, requests)
        elif self.mode == CompetitionMode.NEED_BASED:
            return self._allocate_need_based(available, requests)
        else:
            return self._allocate_proportional(available, requests)

    def _allocate_first_come(
        self,
        available: float,
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Allocate by request timestamp."""
        sorted_requests = sorted(requests, key=lambda r: r.get("timestamp", 0))
        allocations = []
        remaining = available

        for req in sorted_requests:
            if remaining <= 0:
                break
            amount = min(req["amount"], remaining)
            allocations.append((req["id"], amount))
            remaining -= amount

        return allocations

    def _allocate_lottery(
        self,
        available: float,
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Allocate randomly."""
        shuffled = requests.copy()
        np.random.shuffle(shuffled)
        return self._allocate_first_come(available, shuffled)

    def _allocate_priority(
        self,
        available: float,
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Allocate by priority score."""
        sorted_requests = sorted(
            requests,
            key=lambda r: r.get("priority", 0.5),
            reverse=True
        )
        return self._allocate_first_come(available, sorted_requests)

    def _allocate_proportional(
        self,
        available: float,
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Allocate proportionally to requested amounts."""
        total_requested = sum(r["amount"] for r in requests)
        if total_requested <= available:
            # Enough for everyone
            return [(r["id"], r["amount"]) for r in requests]

        # Allocate proportionally
        ratio = available / total_requested
        return [(r["id"], r["amount"] * ratio) for r in requests]

    def _allocate_need_based(
        self,
        available: float,
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """Allocate based on necessity scores."""
        # Weight by necessity
        total_need = sum(r["amount"] * r.get("necessity", 0.5) for r in requests)
        if total_need <= 0:
            return self._allocate_proportional(available, requests)

        allocations = []
        remaining = available

        # Sort by necessity
        sorted_requests = sorted(
            requests,
            key=lambda r: r.get("necessity", 0.5),
            reverse=True
        )

        for req in sorted_requests:
            if remaining <= 0:
                break
            need_weight = req.get("necessity", 0.5)
            share = (req["amount"] * need_weight / total_need) * available
            amount = min(share, req["amount"], remaining)
            allocations.append((req["id"], amount))
            remaining -= amount

        return allocations


class ScarcityModel:
    """
    Models scarcity dynamics across the system.

    Tracks supply/demand for all resource types and
    manages allocation under scarcity conditions.
    """

    def __init__(self):
        self._states: Dict[str, ScarcityState] = {}
        self._allocators: Dict[str, ResourceAllocator] = {}
        self._scarcity_callbacks: List[Callable[[str, ScarcityLevel], None]] = []
        self._competition_events: List[Dict[str, Any]] = []

    def register_resource(
        self,
        resource_type: str,
        initial_supply: float = 0.0,
        max_capacity: float = float('inf'),
        regeneration_rate: float = 0.0,
        competition_mode: CompetitionMode = CompetitionMode.PROPORTIONAL
    ) -> ScarcityState:
        """Register a resource for scarcity tracking."""
        supply = ResourceSupply(
            resource_type=resource_type,
            total_supply=initial_supply,
            available_supply=initial_supply,
            max_capacity=max_capacity,
            regeneration_rate=regeneration_rate
        )
        demand = ResourceDemand(resource_type=resource_type)
        state = ScarcityState(
            resource_type=resource_type,
            supply=supply,
            demand=demand
        )
        self._states[resource_type] = state
        self._allocators[resource_type] = ResourceAllocator(mode=competition_mode)
        return state

    def get_state(self, resource_type: str) -> Optional[ScarcityState]:
        """Get scarcity state for a resource."""
        return self._states.get(resource_type)

    def add_supply(self, resource_type: str, amount: float) -> float:
        """Add supply of a resource. Returns amount actually added."""
        state = self._states.get(resource_type)
        if not state:
            return 0.0
        added = state.supply.add(amount)
        state.update_level()
        return added

    def consume_supply(self, resource_type: str, amount: float) -> float:
        """Consume supply. Returns amount actually consumed."""
        state = self._states.get(resource_type)
        if not state:
            return 0.0
        consumed = state.supply.consume(amount)
        state.update_level()
        return consumed

    def request_resource(
        self,
        resource_type: str,
        requester_id: str,
        amount: float,
        priority: float = 0.5,
        necessity: float = 0.5
    ) -> Optional[str]:
        """
        Request a resource allocation.

        Returns request ID or None if resource not tracked.
        """
        state = self._states.get(resource_type)
        if not state:
            return None
        return state.demand.add_request(requester_id, amount, priority, necessity)

    def process_allocations(self, resource_type: str) -> List[Tuple[str, float]]:
        """
        Process pending allocation requests.

        Returns list of (request_id, allocated_amount).
        """
        state = self._states.get(resource_type)
        allocator = self._allocators.get(resource_type)
        if not state or not allocator:
            return []

        # Get pending requests
        requests = state.demand.pending_requests.copy()
        if not requests:
            return []

        # Allocate available supply
        available = state.supply.available_supply
        allocations = allocator.allocate(available, requests)

        # Apply allocations
        for request_id, amount in allocations:
            state.supply.consume(amount)
            state.demand.fulfill_request(request_id, amount)

        state.update_level()

        # Check for competition events
        if state.level in (ScarcityLevel.SCARCE, ScarcityLevel.CRITICAL):
            self._record_competition_event(resource_type, state, allocations)

        return allocations

    def _record_competition_event(
        self,
        resource_type: str,
        state: ScarcityState,
        allocations: List[Tuple[str, float]]
    ) -> None:
        """Record a competition event."""
        event = {
            "resource_type": resource_type,
            "timestamp": time.time(),
            "scarcity_level": state.level.value,
            "competition_intensity": state.competition_intensity,
            "participants": len(allocations),
            "total_allocated": sum(a[1] for a in allocations)
        }
        self._competition_events.append(event)

        # Keep last 1000 events
        if len(self._competition_events) > 1000:
            self._competition_events = self._competition_events[-1000:]

    def on_scarcity_change(
        self,
        callback: Callable[[str, ScarcityLevel], None]
    ) -> None:
        """Register callback for scarcity level changes."""
        self._scarcity_callbacks.append(callback)

    def get_price_multiplier(self, resource_type: str) -> float:
        """Get price multiplier for a resource based on scarcity."""
        state = self._states.get(resource_type)
        if not state:
            return 1.0
        return state.price_multiplier

    def get_most_scarce(self, n: int = 5) -> List[Tuple[str, ScarcityLevel]]:
        """Get the n most scarce resources."""
        scarcity_order = [
            ScarcityLevel.DEPLETED,
            ScarcityLevel.CRITICAL,
            ScarcityLevel.SCARCE,
            ScarcityLevel.BALANCED,
            ScarcityLevel.SUFFICIENT,
            ScarcityLevel.ABUNDANT
        ]

        sorted_states = sorted(
            self._states.values(),
            key=lambda s: scarcity_order.index(s.level)
        )

        return [(s.resource_type, s.level) for s in sorted_states[:n]]

    def tick(self) -> Dict[str, Any]:
        """Process one time step."""
        changes = []

        for state in self._states.values():
            old_level = state.level

            # Regenerate supply
            state.supply.regenerate()

            # Record demand history
            state.demand.record_demand()

            # Update scarcity level
            new_level = state.update_level()

            if old_level != new_level:
                changes.append((state.resource_type, new_level))
                for callback in self._scarcity_callbacks:
                    callback(state.resource_type, new_level)

        return {
            "level_changes": changes,
            "total_resources": len(self._states),
            "competition_events": len(self._competition_events)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall scarcity statistics."""
        level_counts = {}
        for level in ScarcityLevel:
            level_counts[level.value] = sum(
                1 for s in self._states.values() if s.level == level
            )

        return {
            "resource_count": len(self._states),
            "level_distribution": level_counts,
            "most_scarce": self.get_most_scarce(),
            "recent_competition_events": self._competition_events[-10:]
        }


# Default global scarcity model
_default_scarcity_model = ScarcityModel()


def get_scarcity_model() -> ScarcityModel:
    """Get the default scarcity model."""
    return _default_scarcity_model
