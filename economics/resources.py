"""
Resource Types and Properties for NPCPU Economic System

Defines the fundamental resource types that can be traded, consumed,
and accumulated within the organism economy.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ResourceCategory(Enum):
    """High-level categories of resources."""
    ENERGY = "energy"           # Power, computation, metabolism
    MATERIAL = "material"       # Physical components, building blocks
    INFORMATION = "information"  # Data, knowledge, patterns
    ATTENTION = "attention"     # Focus, processing priority
    SOCIAL = "social"           # Relationships, reputation, trust


class ResourceState(Enum):
    """Current state of a resource."""
    AVAILABLE = "available"     # Ready for use
    RESERVED = "reserved"       # Allocated but not consumed
    DEPLETED = "depleted"       # Exhausted
    REGENERATING = "regenerating"  # Recovering over time
    LOCKED = "locked"           # Cannot be accessed


class ResourceQuality(Enum):
    """Quality tiers for resources."""
    CRUDE = "crude"             # Low quality, inefficient
    STANDARD = "standard"       # Normal quality
    REFINED = "refined"         # High quality, efficient
    PURE = "pure"               # Highest quality
    SYNTHETIC = "synthetic"     # Artificially enhanced


@dataclass
class ResourceType:
    """
    Definition of a resource type.

    Each resource type has intrinsic properties that affect
    how it can be used, stored, and traded.
    """
    type_id: str
    name: str
    category: ResourceCategory
    base_value: float = 1.0
    decay_rate: float = 0.0         # How fast it degrades (0 = stable)
    max_stack: float = float('inf')  # Maximum per holder
    transferable: bool = True        # Can be traded
    divisible: bool = True           # Can be split
    renewable: bool = False          # Can regenerate
    regen_rate: float = 0.0          # Regeneration rate per tick
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.type_id)

    def __eq__(self, other):
        if isinstance(other, ResourceType):
            return self.type_id == other.type_id
        return False


@dataclass
class Resource:
    """
    A specific instance of a resource with quantity and quality.

    Resources are the fundamental units of economic activity.
    They can be produced, consumed, traded, and stored.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = None
    quantity: float = 0.0
    quality: ResourceQuality = ResourceQuality.STANDARD
    state: ResourceState = ResourceState.AVAILABLE
    owner_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Calculate the current value of this resource."""
        if self.resource_type is None:
            return 0.0

        quality_multipliers = {
            ResourceQuality.CRUDE: 0.5,
            ResourceQuality.STANDARD: 1.0,
            ResourceQuality.REFINED: 1.5,
            ResourceQuality.PURE: 2.0,
            ResourceQuality.SYNTHETIC: 2.5
        }
        multiplier = quality_multipliers.get(self.quality, 1.0)
        return self.resource_type.base_value * self.quantity * multiplier

    def decay(self, ticks: int = 1) -> float:
        """Apply decay to the resource. Returns amount decayed."""
        if self.resource_type is None or self.resource_type.decay_rate == 0:
            return 0.0

        decay_amount = self.quantity * self.resource_type.decay_rate * ticks
        self.quantity = max(0, self.quantity - decay_amount)
        self.last_updated = time.time()

        if self.quantity <= 0:
            self.state = ResourceState.DEPLETED

        return decay_amount

    def regenerate(self, ticks: int = 1) -> float:
        """Regenerate resource if renewable. Returns amount regenerated."""
        if self.resource_type is None:
            return 0.0
        if not self.resource_type.renewable:
            return 0.0

        regen_amount = self.resource_type.regen_rate * ticks
        max_stack = self.resource_type.max_stack

        old_quantity = self.quantity
        self.quantity = min(max_stack, self.quantity + regen_amount)
        self.last_updated = time.time()

        if self.state == ResourceState.DEPLETED and self.quantity > 0:
            self.state = ResourceState.REGENERATING
        if self.quantity >= max_stack * 0.5:
            self.state = ResourceState.AVAILABLE

        return self.quantity - old_quantity

    def split(self, amount: float) -> Optional['Resource']:
        """Split off a portion of this resource. Returns new resource or None."""
        if self.resource_type is None:
            return None
        if not self.resource_type.divisible:
            return None
        if amount <= 0 or amount >= self.quantity:
            return None

        self.quantity -= amount
        self.last_updated = time.time()

        return Resource(
            resource_type=self.resource_type,
            quantity=amount,
            quality=self.quality,
            state=self.state,
            owner_id=self.owner_id
        )

    def merge(self, other: 'Resource') -> bool:
        """Merge another resource into this one. Returns success."""
        if self.resource_type is None or other.resource_type is None:
            return False
        if self.resource_type.type_id != other.resource_type.type_id:
            return False

        max_stack = self.resource_type.max_stack
        total = self.quantity + other.quantity

        if total > max_stack:
            self.quantity = max_stack
            other.quantity = total - max_stack
        else:
            self.quantity = total
            other.quantity = 0
            other.state = ResourceState.DEPLETED

        # Quality becomes weighted average
        if self.quantity + other.quantity > 0:
            self_weight = self.quantity / (self.quantity + other.quantity)
            other_weight = 1 - self_weight
            quality_order = list(ResourceQuality)
            self_idx = quality_order.index(self.quality)
            other_idx = quality_order.index(other.quality)
            avg_idx = int(self_weight * self_idx + other_weight * other_idx)
            self.quality = quality_order[min(avg_idx, len(quality_order) - 1)]

        self.last_updated = time.time()
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize resource to dictionary."""
        return {
            "id": self.id,
            "type_id": self.resource_type.type_id if self.resource_type else None,
            "quantity": self.quantity,
            "quality": self.quality.value,
            "state": self.state.value,
            "owner_id": self.owner_id,
            "value": self.value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }


class ResourceRegistry:
    """
    Central registry for all resource types in the system.

    Provides factory methods and lookup for resource types.
    """

    def __init__(self):
        self._types: Dict[str, ResourceType] = {}
        self._register_default_types()

    def _register_default_types(self):
        """Register the standard resource types."""
        # Energy resources
        self.register(ResourceType(
            type_id="energy_compute",
            name="Computational Energy",
            category=ResourceCategory.ENERGY,
            base_value=1.0,
            decay_rate=0.01,
            renewable=True,
            regen_rate=0.1
        ))

        self.register(ResourceType(
            type_id="energy_metabolic",
            name="Metabolic Energy",
            category=ResourceCategory.ENERGY,
            base_value=1.2,
            decay_rate=0.02,
            renewable=True,
            regen_rate=0.05
        ))

        # Material resources
        self.register(ResourceType(
            type_id="material_raw",
            name="Raw Materials",
            category=ResourceCategory.MATERIAL,
            base_value=0.5,
            decay_rate=0.0,
            renewable=False
        ))

        self.register(ResourceType(
            type_id="material_components",
            name="Processed Components",
            category=ResourceCategory.MATERIAL,
            base_value=2.0,
            decay_rate=0.001,
            renewable=False
        ))

        # Information resources
        self.register(ResourceType(
            type_id="info_data",
            name="Raw Data",
            category=ResourceCategory.INFORMATION,
            base_value=0.3,
            decay_rate=0.0,
            renewable=False
        ))

        self.register(ResourceType(
            type_id="info_knowledge",
            name="Processed Knowledge",
            category=ResourceCategory.INFORMATION,
            base_value=3.0,
            decay_rate=0.005,  # Knowledge can become stale
            renewable=False
        ))

        self.register(ResourceType(
            type_id="info_insight",
            name="Derived Insights",
            category=ResourceCategory.INFORMATION,
            base_value=5.0,
            decay_rate=0.01,
            renewable=False
        ))

        # Attention resources
        self.register(ResourceType(
            type_id="attention_focus",
            name="Focused Attention",
            category=ResourceCategory.ATTENTION,
            base_value=2.0,
            decay_rate=0.1,
            max_stack=100.0,
            renewable=True,
            regen_rate=0.2
        ))

        # Social resources
        self.register(ResourceType(
            type_id="social_reputation",
            name="Reputation",
            category=ResourceCategory.SOCIAL,
            base_value=1.0,
            decay_rate=0.001,
            transferable=False,
            renewable=True,
            regen_rate=0.01
        ))

        self.register(ResourceType(
            type_id="social_trust",
            name="Trust Credits",
            category=ResourceCategory.SOCIAL,
            base_value=2.0,
            decay_rate=0.01,
            transferable=True,
            renewable=False
        ))

    def register(self, resource_type: ResourceType) -> None:
        """Register a new resource type."""
        self._types[resource_type.type_id] = resource_type

    def get(self, type_id: str) -> Optional[ResourceType]:
        """Get a resource type by ID."""
        return self._types.get(type_id)

    def get_by_category(self, category: ResourceCategory) -> List[ResourceType]:
        """Get all resource types in a category."""
        return [t for t in self._types.values() if t.category == category]

    def list_all(self) -> List[ResourceType]:
        """List all registered resource types."""
        return list(self._types.values())

    def create_resource(
        self,
        type_id: str,
        quantity: float = 1.0,
        quality: ResourceQuality = ResourceQuality.STANDARD,
        owner_id: Optional[str] = None
    ) -> Optional[Resource]:
        """Create a new resource instance of the given type."""
        resource_type = self.get(type_id)
        if resource_type is None:
            return None

        return Resource(
            resource_type=resource_type,
            quantity=min(quantity, resource_type.max_stack),
            quality=quality,
            owner_id=owner_id
        )


@dataclass
class ResourcePool:
    """
    A collection of resources held by an entity.

    Manages storage, access, and lifecycle of multiple resources.
    """
    owner_id: str
    resources: Dict[str, Resource] = field(default_factory=dict)
    capacity: Dict[ResourceCategory, float] = field(default_factory=dict)
    registry: ResourceRegistry = field(default_factory=ResourceRegistry)

    def add(self, resource: Resource) -> bool:
        """Add a resource to the pool. Returns success."""
        if resource.resource_type is None:
            return False

        category = resource.resource_type.category
        current_amount = self.get_total_by_category(category)
        capacity = self.capacity.get(category, float('inf'))

        if current_amount + resource.quantity > capacity:
            return False

        resource.owner_id = self.owner_id

        # Try to merge with existing resources of same type
        for existing in self.resources.values():
            if (existing.resource_type and
                existing.resource_type.type_id == resource.resource_type.type_id and
                existing.quality == resource.quality):
                existing.merge(resource)
                if resource.quantity <= 0:
                    return True

        # Add as new resource
        self.resources[resource.id] = resource
        return True

    def remove(self, resource_id: str) -> Optional[Resource]:
        """Remove a resource from the pool."""
        resource = self.resources.pop(resource_id, None)
        if resource:
            resource.owner_id = None
        return resource

    def get(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        return self.resources.get(resource_id)

    def get_by_type(self, type_id: str) -> List[Resource]:
        """Get all resources of a given type."""
        return [
            r for r in self.resources.values()
            if r.resource_type and r.resource_type.type_id == type_id
        ]

    def get_total_by_type(self, type_id: str) -> float:
        """Get total quantity of a resource type."""
        return sum(r.quantity for r in self.get_by_type(type_id))

    def get_total_by_category(self, category: ResourceCategory) -> float:
        """Get total quantity in a category."""
        return sum(
            r.quantity for r in self.resources.values()
            if r.resource_type and r.resource_type.category == category
        )

    def get_total_value(self) -> float:
        """Get total value of all resources."""
        return sum(r.value for r in self.resources.values())

    def withdraw(self, type_id: str, amount: float) -> Optional[Resource]:
        """Withdraw a specific amount of a resource type."""
        available = self.get_by_type(type_id)
        if not available:
            return None

        total_available = sum(r.quantity for r in available)
        if total_available < amount:
            return None

        # Collect from resources until we have enough
        withdrawn = None
        remaining = amount

        for resource in sorted(available, key=lambda r: r.quantity):
            if remaining <= 0:
                break

            if resource.quantity <= remaining:
                # Take entire resource
                remaining -= resource.quantity
                self.remove(resource.id)
                if withdrawn is None:
                    withdrawn = resource
                else:
                    withdrawn.merge(resource)
            else:
                # Split off needed amount
                split = resource.split(remaining)
                if split:
                    if withdrawn is None:
                        withdrawn = split
                    else:
                        withdrawn.merge(split)
                remaining = 0

        return withdrawn

    def tick(self) -> Dict[str, float]:
        """Process one time step for all resources."""
        decayed = {}
        regenerated = {}

        for resource in list(self.resources.values()):
            if resource.resource_type:
                decay_amount = resource.decay()
                if decay_amount > 0:
                    decayed[resource.resource_type.type_id] = (
                        decayed.get(resource.resource_type.type_id, 0) + decay_amount
                    )

                regen_amount = resource.regenerate()
                if regen_amount > 0:
                    regenerated[resource.resource_type.type_id] = (
                        regenerated.get(resource.resource_type.type_id, 0) + regen_amount
                    )

                # Remove depleted resources
                if resource.state == ResourceState.DEPLETED and resource.quantity <= 0:
                    self.remove(resource.id)

        return {"decayed": decayed, "regenerated": regenerated}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pool to dictionary."""
        return {
            "owner_id": self.owner_id,
            "resources": {rid: r.to_dict() for rid, r in self.resources.items()},
            "total_value": self.get_total_value(),
            "capacity": {k.value: v for k, v in self.capacity.items()}
        }


# Default global registry
_default_registry = ResourceRegistry()


def get_resource_registry() -> ResourceRegistry:
    """Get the default resource registry."""
    return _default_registry
