"""
Event System for NPCPU

A decoupled publish-subscribe event system that enables:
- Loose coupling between components
- Asynchronous communication
- Event filtering and prioritization
- Event history and replay
"""

from typing import (
    Dict, Any, List, Optional, Set, Callable, TypeVar,
    Generic, Union, Type
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict
import uuid
import threading
import queue
from abc import ABC, abstractmethod


# =============================================================================
# Event Types
# =============================================================================

class EventPriority(Enum):
    """Priority levels for events"""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 15


@dataclass
class Event:
    """
    Base event class.

    Events are the fundamental unit of communication in the system.
    They carry information about something that happened.
    """
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    priority: EventPriority = EventPriority.NORMAL
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    handled: bool = False
    propagate: bool = True  # Whether to continue to other handlers

    def __post_init__(self):
        if isinstance(self.priority, int):
            for p in EventPriority:
                if p.value == self.priority:
                    self.priority = p
                    break

    def stop_propagation(self):
        """Stop event from being handled by more handlers"""
        self.propagate = False

    def mark_handled(self):
        """Mark event as handled"""
        self.handled = True


# =============================================================================
# Event Filters
# =============================================================================

class EventFilter(ABC):
    """Abstract base for event filters"""

    @abstractmethod
    def matches(self, event: Event) -> bool:
        """Check if event matches this filter"""
        pass

    def __and__(self, other: 'EventFilter') -> 'AndFilter':
        return AndFilter(self, other)

    def __or__(self, other: 'EventFilter') -> 'OrFilter':
        return OrFilter(self, other)

    def __invert__(self) -> 'NotFilter':
        return NotFilter(self)


class TypeFilter(EventFilter):
    """Filter events by type"""

    def __init__(self, *event_types: str):
        self.event_types = set(event_types)

    def matches(self, event: Event) -> bool:
        return event.type in self.event_types


class SourceFilter(EventFilter):
    """Filter events by source"""

    def __init__(self, *sources: str):
        self.sources = set(sources)

    def matches(self, event: Event) -> bool:
        return event.source in self.sources


class PriorityFilter(EventFilter):
    """Filter events by minimum priority"""

    def __init__(self, min_priority: EventPriority):
        self.min_priority = min_priority

    def matches(self, event: Event) -> bool:
        return event.priority.value >= self.min_priority.value


class DataFilter(EventFilter):
    """Filter events by data content"""

    def __init__(self, **criteria):
        self.criteria = criteria

    def matches(self, event: Event) -> bool:
        for key, value in self.criteria.items():
            if key not in event.data:
                return False
            if callable(value):
                if not value(event.data[key]):
                    return False
            elif event.data[key] != value:
                return False
        return True


class AndFilter(EventFilter):
    """Combine filters with AND logic"""

    def __init__(self, *filters: EventFilter):
        self.filters = filters

    def matches(self, event: Event) -> bool:
        return all(f.matches(event) for f in self.filters)


class OrFilter(EventFilter):
    """Combine filters with OR logic"""

    def __init__(self, *filters: EventFilter):
        self.filters = filters

    def matches(self, event: Event) -> bool:
        return any(f.matches(event) for f in self.filters)


class NotFilter(EventFilter):
    """Negate a filter"""

    def __init__(self, filter: EventFilter):
        self.filter = filter

    def matches(self, event: Event) -> bool:
        return not self.filter.matches(event)


# =============================================================================
# Event Handlers
# =============================================================================

EventHandler = Callable[[Event], None]


@dataclass
class HandlerRegistration:
    """Registration info for an event handler"""
    handler: EventHandler
    filter: Optional[EventFilter] = None
    priority: int = 0  # Higher = called first
    once: bool = False  # Remove after first call
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


# =============================================================================
# Event Bus
# =============================================================================

class EventBus:
    """
    Central event bus for publish-subscribe communication.

    Features:
    - Type-based event routing
    - Event filtering
    - Handler prioritization
    - Event history
    - Synchronous and asynchronous dispatch

    Example:
        bus = EventBus()

        # Subscribe to events
        bus.subscribe("organism.born", lambda e: print(f"New organism: {e.data['name']}"))

        # Subscribe with filter
        bus.subscribe(
            "organism.died",
            lambda e: print(f"Elder died: {e.data['name']}"),
            filter=DataFilter(age=lambda a: a > 100)
        )

        # Publish events
        bus.publish(Event("organism.born", {"name": "Alpha", "traits": {...}}))
    """

    def __init__(self, async_mode: bool = False, history_size: int = 1000):
        self._handlers: Dict[str, List[HandlerRegistration]] = defaultdict(list)
        self._global_handlers: List[HandlerRegistration] = []
        self._history: List[Event] = []
        self._history_size = history_size
        self._async_mode = async_mode
        self._lock = threading.RLock()

        if async_mode:
            self._queue: queue.Queue = queue.Queue()
            self._worker = threading.Thread(target=self._process_queue, daemon=True)
            self._running = True
            self._worker.start()

    # -------------------------------------------------------------------------
    # Subscription
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        filter: Optional[EventFilter] = None,
        priority: int = 0,
        once: bool = False
    ) -> str:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to (use "*" for all)
            handler: Function to call when event occurs
            filter: Optional filter to further refine which events to handle
            priority: Higher priority handlers are called first
            once: If True, handler is removed after first call

        Returns:
            Handler ID for later unsubscription
        """
        registration = HandlerRegistration(
            handler=handler,
            filter=filter,
            priority=priority,
            once=once
        )

        with self._lock:
            if event_type == "*":
                self._global_handlers.append(registration)
                self._global_handlers.sort(key=lambda r: -r.priority)
            else:
                self._handlers[event_type].append(registration)
                self._handlers[event_type].sort(key=lambda r: -r.priority)

        return registration.id

    def unsubscribe(self, handler_id: str) -> bool:
        """
        Unsubscribe a handler by ID.

        Returns True if handler was found and removed.
        """
        with self._lock:
            # Check global handlers
            for i, reg in enumerate(self._global_handlers):
                if reg.id == handler_id:
                    self._global_handlers.pop(i)
                    return True

            # Check type-specific handlers
            for handlers in self._handlers.values():
                for i, reg in enumerate(handlers):
                    if reg.id == handler_id:
                        handlers.pop(i)
                        return True

        return False

    def unsubscribe_all(self, event_type: Optional[str] = None) -> int:
        """
        Remove all handlers, optionally for a specific event type.

        Returns number of handlers removed.
        """
        with self._lock:
            if event_type is None:
                count = len(self._global_handlers)
                for handlers in self._handlers.values():
                    count += len(handlers)
                self._global_handlers.clear()
                self._handlers.clear()
                return count
            elif event_type in self._handlers:
                count = len(self._handlers[event_type])
                del self._handlers[event_type]
                return count
        return 0

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """
        Publish an event to all matching subscribers.

        In async mode, event is queued for processing.
        In sync mode, handlers are called immediately.
        """
        if self._async_mode:
            self._queue.put(event)
        else:
            self._dispatch(event)

    def emit(self, event_type: str, data: Optional[Dict[str, Any]] = None,
             source: str = "", priority: EventPriority = EventPriority.NORMAL) -> Event:
        """
        Convenience method to create and publish an event.

        Returns the created event.
        """
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        self.publish(event)
        return event

    def _dispatch(self, event: Event) -> None:
        """Dispatch event to handlers"""
        # Add to history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        handlers_to_remove = []

        # Global handlers first
        for reg in self._global_handlers[:]:
            if not event.propagate:
                break
            if reg.filter is None or reg.filter.matches(event):
                try:
                    reg.handler(event)
                except Exception as e:
                    self._handle_error(event, reg, e)
                if reg.once:
                    handlers_to_remove.append(("*", reg.id))

        # Type-specific handlers
        if event.type in self._handlers:
            for reg in self._handlers[event.type][:]:
                if not event.propagate:
                    break
                if reg.filter is None or reg.filter.matches(event):
                    try:
                        reg.handler(event)
                    except Exception as e:
                        self._handle_error(event, reg, e)
                    if reg.once:
                        handlers_to_remove.append((event.type, reg.id))

        # Remove one-time handlers
        for event_type, handler_id in handlers_to_remove:
            self.unsubscribe(handler_id)

    def _handle_error(self, event: Event, handler: HandlerRegistration, error: Exception) -> None:
        """Handle errors during event dispatch"""
        # Emit error event (but prevent infinite loops)
        if event.type != "error.handler":
            self.emit("error.handler", {
                "event_type": event.type,
                "event_id": event.id,
                "handler_id": handler.id,
                "error": str(error)
            })

    def _process_queue(self) -> None:
        """Worker thread for async processing"""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                self._dispatch(event)
            except queue.Empty:
                continue

    # -------------------------------------------------------------------------
    # History & Replay
    # -------------------------------------------------------------------------

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get recent events from history"""
        with self._lock:
            events = self._history.copy()

        if event_type:
            events = [e for e in events if e.type == event_type]

        return events[-limit:]

    def replay(self, events: List[Event]) -> None:
        """Replay a list of events"""
        for event in events:
            # Create new event with same data but new ID/timestamp
            replayed = Event(
                type=event.type,
                data=event.data.copy(),
                source=f"replay:{event.source}",
                priority=event.priority
            )
            self.publish(replayed)

    def clear_history(self) -> None:
        """Clear event history"""
        with self._lock:
            self._history.clear()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def wait_for(
        self,
        event_type: str,
        timeout: float = 10.0,
        filter: Optional[EventFilter] = None
    ) -> Optional[Event]:
        """
        Block until an event of the specified type is received.

        Returns the event or None if timeout.
        """
        result = [None]
        event_received = threading.Event()

        def handler(event: Event):
            result[0] = event
            event_received.set()

        handler_id = self.subscribe(event_type, handler, filter=filter, once=True)

        if event_received.wait(timeout):
            return result[0]

        self.unsubscribe(handler_id)
        return None

    def shutdown(self) -> None:
        """Shutdown the event bus (for async mode)"""
        self._running = False
        if self._async_mode and self._worker.is_alive():
            self._worker.join(timeout=1.0)


# =============================================================================
# Global Event Bus
# =============================================================================

_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the global event bus instance"""
    global _global_bus
    _global_bus = bus


# =============================================================================
# Decorator for event handlers
# =============================================================================

def on_event(event_type: str, filter: Optional[EventFilter] = None,
             priority: int = 0, bus: Optional[EventBus] = None):
    """
    Decorator for registering event handlers.

    Example:
        @on_event("organism.born")
        def handle_birth(event):
            print(f"New organism: {event.data['name']}")

        @on_event("organism.died", filter=DataFilter(cause="starvation"))
        def handle_starvation(event):
            print(f"Organism starved: {event.data['name']}")
    """
    def decorator(func: EventHandler) -> EventHandler:
        target_bus = bus or get_event_bus()
        target_bus.subscribe(event_type, func, filter=filter, priority=priority)
        return func
    return decorator
