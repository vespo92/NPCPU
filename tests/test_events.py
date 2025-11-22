"""
Tests for Event System

Unit tests for EventBus, Event, filters, and decorators.
"""

import pytest
import threading
import time
from typing import List

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from core.events import (
    Event, EventPriority, EventBus,
    EventFilter, TypeFilter, SourceFilter, PriorityFilter, DataFilter,
    AndFilter, OrFilter, NotFilter,
    HandlerRegistration,
    get_event_bus, set_event_bus, on_event
)


# =============================================================================
# Tests: Event
# =============================================================================

class TestEvent:
    """Tests for Event dataclass"""

    def test_initialization(self):
        """Test event initialization"""
        event = Event(type="test.event", data={"key": "value"})
        assert event.type == "test.event"
        assert event.data["key"] == "value"
        assert event.priority == EventPriority.NORMAL
        assert event.id is not None
        assert event.handled is False
        assert event.propagate is True

    def test_unique_ids(self):
        """Test events get unique IDs"""
        e1 = Event(type="test")
        e2 = Event(type="test")
        assert e1.id != e2.id

    def test_timestamp(self):
        """Test event has timestamp"""
        event = Event(type="test")
        assert event.timestamp is not None

    def test_priority_conversion(self):
        """Test priority int to enum conversion"""
        event = Event(type="test", priority=10)
        assert event.priority == EventPriority.HIGH

    def test_stop_propagation(self):
        """Test stop_propagation method"""
        event = Event(type="test")
        assert event.propagate is True
        event.stop_propagation()
        assert event.propagate is False

    def test_mark_handled(self):
        """Test mark_handled method"""
        event = Event(type="test")
        assert event.handled is False
        event.mark_handled()
        assert event.handled is True

    def test_default_data(self):
        """Test default empty data dict"""
        event = Event(type="test")
        assert event.data == {}

    def test_source(self):
        """Test event source"""
        event = Event(type="test", source="component_a")
        assert event.source == "component_a"


class TestEventPriority:
    """Tests for EventPriority enum"""

    def test_priority_values(self):
        """Test priority ordering"""
        assert EventPriority.LOW.value < EventPriority.NORMAL.value
        assert EventPriority.NORMAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.CRITICAL.value


# =============================================================================
# Tests: Event Filters
# =============================================================================

class TestTypeFilter:
    """Tests for TypeFilter"""

    def test_single_type_match(self):
        """Test matching single type"""
        f = TypeFilter("organism.born")
        event = Event(type="organism.born")
        assert f.matches(event) is True

    def test_single_type_no_match(self):
        """Test non-matching type"""
        f = TypeFilter("organism.born")
        event = Event(type="organism.died")
        assert f.matches(event) is False

    def test_multiple_types(self):
        """Test matching multiple types"""
        f = TypeFilter("organism.born", "organism.died")
        assert f.matches(Event(type="organism.born")) is True
        assert f.matches(Event(type="organism.died")) is True
        assert f.matches(Event(type="organism.moved")) is False


class TestSourceFilter:
    """Tests for SourceFilter"""

    def test_source_match(self):
        """Test matching source"""
        f = SourceFilter("component_a")
        event = Event(type="test", source="component_a")
        assert f.matches(event) is True

    def test_source_no_match(self):
        """Test non-matching source"""
        f = SourceFilter("component_a")
        event = Event(type="test", source="component_b")
        assert f.matches(event) is False

    def test_multiple_sources(self):
        """Test multiple allowed sources"""
        f = SourceFilter("a", "b")
        assert f.matches(Event(type="test", source="a")) is True
        assert f.matches(Event(type="test", source="b")) is True
        assert f.matches(Event(type="test", source="c")) is False


class TestPriorityFilter:
    """Tests for PriorityFilter"""

    def test_priority_match(self):
        """Test minimum priority match"""
        f = PriorityFilter(EventPriority.HIGH)
        assert f.matches(Event(type="test", priority=EventPriority.CRITICAL)) is True
        assert f.matches(Event(type="test", priority=EventPriority.HIGH)) is True
        assert f.matches(Event(type="test", priority=EventPriority.NORMAL)) is False


class TestDataFilter:
    """Tests for DataFilter"""

    def test_exact_value_match(self):
        """Test exact value matching"""
        f = DataFilter(status="active")
        assert f.matches(Event(type="test", data={"status": "active"})) is True
        assert f.matches(Event(type="test", data={"status": "inactive"})) is False

    def test_missing_key(self):
        """Test missing key returns False"""
        f = DataFilter(status="active")
        assert f.matches(Event(type="test", data={})) is False

    def test_callable_predicate(self):
        """Test callable predicate"""
        f = DataFilter(age=lambda a: a > 18)
        assert f.matches(Event(type="test", data={"age": 25})) is True
        assert f.matches(Event(type="test", data={"age": 15})) is False

    def test_multiple_criteria(self):
        """Test multiple criteria must all match"""
        f = DataFilter(status="active", level=lambda l: l > 5)
        assert f.matches(Event(type="test", data={"status": "active", "level": 10})) is True
        assert f.matches(Event(type="test", data={"status": "active", "level": 3})) is False


class TestCompositeFilters:
    """Tests for composite filters (And, Or, Not)"""

    def test_and_filter(self):
        """Test AND combination"""
        f = AndFilter(
            TypeFilter("test"),
            SourceFilter("component_a")
        )
        assert f.matches(Event(type="test", source="component_a")) is True
        assert f.matches(Event(type="test", source="component_b")) is False
        assert f.matches(Event(type="other", source="component_a")) is False

    def test_or_filter(self):
        """Test OR combination"""
        f = OrFilter(
            TypeFilter("type_a"),
            TypeFilter("type_b")
        )
        assert f.matches(Event(type="type_a")) is True
        assert f.matches(Event(type="type_b")) is True
        assert f.matches(Event(type="type_c")) is False

    def test_not_filter(self):
        """Test NOT negation"""
        f = NotFilter(TypeFilter("excluded"))
        assert f.matches(Event(type="allowed")) is True
        assert f.matches(Event(type="excluded")) is False

    def test_operator_overloads(self):
        """Test & | ~ operators"""
        f1 = TypeFilter("test")
        f2 = SourceFilter("component")

        and_filter = f1 & f2
        assert isinstance(and_filter, AndFilter)

        or_filter = f1 | f2
        assert isinstance(or_filter, OrFilter)

        not_filter = ~f1
        assert isinstance(not_filter, NotFilter)


# =============================================================================
# Tests: EventBus
# =============================================================================

class TestEventBusBasic:
    """Basic tests for EventBus"""

    def test_initialization(self):
        """Test bus initialization"""
        bus = EventBus()
        assert bus is not None

    def test_subscribe_returns_id(self):
        """Test subscribe returns handler ID"""
        bus = EventBus()
        handler_id = bus.subscribe("test", lambda e: None)
        assert handler_id is not None
        assert len(handler_id) == 36  # UUID format

    def test_publish_calls_handler(self):
        """Test publish triggers handler"""
        bus = EventBus()
        received = []

        bus.subscribe("test.event", lambda e: received.append(e))
        bus.publish(Event(type="test.event", data={"msg": "hello"}))

        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    def test_emit_convenience(self):
        """Test emit convenience method"""
        bus = EventBus()
        received = []

        bus.subscribe("test", lambda e: received.append(e))
        event = bus.emit("test", {"value": 42})

        assert event.type == "test"
        assert len(received) == 1
        assert received[0].data["value"] == 42

    def test_unsubscribe(self):
        """Test unsubscribing handler"""
        bus = EventBus()
        received = []

        handler_id = bus.subscribe("test", lambda e: received.append(e))
        bus.publish(Event(type="test"))
        assert len(received) == 1

        bus.unsubscribe(handler_id)
        bus.publish(Event(type="test"))
        assert len(received) == 1  # No additional events

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing nonexistent handler"""
        bus = EventBus()
        result = bus.unsubscribe("nonexistent-id")
        assert result is False

    def test_unsubscribe_all(self):
        """Test unsubscribing all handlers"""
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.subscribe("test", lambda e: None)
        bus.subscribe("other", lambda e: None)

        count = bus.unsubscribe_all()
        assert count == 3

    def test_unsubscribe_all_by_type(self):
        """Test unsubscribing all handlers for type"""
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.subscribe("test", lambda e: None)
        bus.subscribe("other", lambda e: None)

        count = bus.unsubscribe_all("test")
        assert count == 2


class TestEventBusHandlers:
    """Tests for EventBus handler features"""

    def test_multiple_handlers(self):
        """Test multiple handlers for same event"""
        bus = EventBus()
        order = []

        bus.subscribe("test", lambda e: order.append("a"))
        bus.subscribe("test", lambda e: order.append("b"))
        bus.publish(Event(type="test"))

        assert len(order) == 2

    def test_handler_priority(self):
        """Test handler priority ordering"""
        bus = EventBus()
        order = []

        bus.subscribe("test", lambda e: order.append("low"), priority=0)
        bus.subscribe("test", lambda e: order.append("high"), priority=10)
        bus.subscribe("test", lambda e: order.append("medium"), priority=5)

        bus.publish(Event(type="test"))
        assert order == ["high", "medium", "low"]

    def test_handler_filter(self):
        """Test handler with filter"""
        bus = EventBus()
        received = []

        bus.subscribe(
            "test",
            lambda e: received.append(e),
            filter=DataFilter(important=True)
        )

        bus.publish(Event(type="test", data={"important": False}))
        assert len(received) == 0

        bus.publish(Event(type="test", data={"important": True}))
        assert len(received) == 1

    def test_once_handler(self):
        """Test one-time handler"""
        bus = EventBus()
        count = {"value": 0}

        bus.subscribe("test", lambda e: count.__setitem__("value", count["value"] + 1), once=True)

        bus.publish(Event(type="test"))
        bus.publish(Event(type="test"))
        bus.publish(Event(type="test"))

        assert count["value"] == 1

    def test_global_handler(self):
        """Test global handler receives all events"""
        bus = EventBus()
        received = []

        bus.subscribe("*", lambda e: received.append(e.type))

        bus.publish(Event(type="type_a"))
        bus.publish(Event(type="type_b"))
        bus.publish(Event(type="type_c"))

        assert received == ["type_a", "type_b", "type_c"]

    def test_stop_propagation(self):
        """Test stop propagation prevents further handlers"""
        bus = EventBus()
        order = []

        def stopping_handler(e):
            order.append("first")
            e.stop_propagation()

        bus.subscribe("test", stopping_handler, priority=10)
        bus.subscribe("test", lambda e: order.append("second"), priority=5)

        bus.publish(Event(type="test"))
        assert order == ["first"]


class TestEventBusHistory:
    """Tests for EventBus history features"""

    def test_history_recording(self):
        """Test events are recorded in history"""
        bus = EventBus(history_size=100)
        bus.emit("test", {"id": 1})
        bus.emit("test", {"id": 2})

        history = bus.get_history()
        assert len(history) == 2

    def test_history_limit(self):
        """Test history respects size limit"""
        bus = EventBus(history_size=5)
        for i in range(10):
            bus.emit("test", {"id": i})

        history = bus.get_history()
        assert len(history) == 5
        assert history[0].data["id"] == 5  # Oldest kept

    def test_history_filter_by_type(self):
        """Test history filtering by type"""
        bus = EventBus()
        bus.emit("type_a", {})
        bus.emit("type_b", {})
        bus.emit("type_a", {})

        history = bus.get_history(event_type="type_a")
        assert len(history) == 2

    def test_clear_history(self):
        """Test clearing history"""
        bus = EventBus()
        bus.emit("test", {})
        bus.emit("test", {})

        bus.clear_history()
        history = bus.get_history()
        assert len(history) == 0

    def test_replay(self):
        """Test event replay"""
        bus = EventBus()
        received = []

        bus.subscribe("test", lambda e: received.append(e.data["id"]))
        bus.emit("test", {"id": 1})
        bus.emit("test", {"id": 2})

        history = bus.get_history()
        bus.replay(history)

        # Original 2 + replayed 2
        assert len(received) == 4


class TestEventBusErrors:
    """Tests for EventBus error handling"""

    def test_handler_exception_isolated(self):
        """Test handler exceptions don't affect other handlers"""
        bus = EventBus()
        results = []

        def failing_handler(e):
            raise ValueError("Test error")

        def working_handler(e):
            results.append("worked")

        bus.subscribe("test", failing_handler, priority=10)
        bus.subscribe("test", working_handler, priority=5)

        bus.publish(Event(type="test"))
        assert results == ["worked"]

    def test_error_event_emitted(self):
        """Test error event is emitted on handler failure"""
        bus = EventBus()
        errors = []

        def failing_handler(e):
            raise ValueError("Test error")

        bus.subscribe("test", failing_handler)
        bus.subscribe("error.handler", lambda e: errors.append(e))

        bus.publish(Event(type="test"))
        assert len(errors) == 1
        assert "Test error" in errors[0].data["error"]


# =============================================================================
# Tests: Global Event Bus
# =============================================================================

class TestGlobalEventBus:
    """Tests for global event bus singleton"""

    def test_get_event_bus(self):
        """Test getting global event bus"""
        bus = get_event_bus()
        assert bus is not None
        assert isinstance(bus, EventBus)

    def test_same_instance(self):
        """Test same instance returned"""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_set_event_bus(self):
        """Test setting custom global bus"""
        original = get_event_bus()
        custom = EventBus()

        set_event_bus(custom)
        assert get_event_bus() is custom

        # Restore original
        set_event_bus(original)


# =============================================================================
# Tests: Decorator
# =============================================================================

class TestOnEventDecorator:
    """Tests for on_event decorator"""

    def test_decorator_registers_handler(self):
        """Test decorator registers handler"""
        bus = EventBus()
        received = []

        @on_event("test.decorated", bus=bus)
        def handler(event):
            received.append(event)

        bus.emit("test.decorated", {"value": 42})
        assert len(received) == 1

    def test_decorator_with_filter(self):
        """Test decorator with filter"""
        bus = EventBus()
        received = []

        @on_event("test", filter=DataFilter(important=True), bus=bus)
        def handler(event):
            received.append(event)

        bus.emit("test", {"important": False})
        bus.emit("test", {"important": True})

        assert len(received) == 1


# =============================================================================
# Tests: Async Mode
# =============================================================================

class TestAsyncEventBus:
    """Tests for async event bus mode"""

    def test_async_initialization(self):
        """Test async bus starts worker thread"""
        bus = EventBus(async_mode=True)
        try:
            assert bus._worker.is_alive()
        finally:
            bus.shutdown()

    def test_async_event_processing(self):
        """Test events processed asynchronously"""
        bus = EventBus(async_mode=True)
        received = []

        try:
            bus.subscribe("test", lambda e: received.append(e))
            bus.emit("test", {"async": True})

            # Wait for processing
            time.sleep(0.2)
            assert len(received) == 1
        finally:
            bus.shutdown()

    def test_shutdown(self):
        """Test graceful shutdown"""
        bus = EventBus(async_mode=True)
        bus.shutdown()

        # Allow time for thread to stop
        time.sleep(0.2)
        assert not bus._running


class TestWaitFor:
    """Tests for wait_for functionality"""

    def test_wait_for_event(self):
        """Test waiting for specific event"""
        bus = EventBus()

        def emit_later():
            time.sleep(0.1)
            bus.emit("awaited", {"success": True})

        thread = threading.Thread(target=emit_later)
        thread.start()

        event = bus.wait_for("awaited", timeout=1.0)
        thread.join()

        assert event is not None
        assert event.data["success"] is True

    def test_wait_for_timeout(self):
        """Test wait_for times out"""
        bus = EventBus()
        event = bus.wait_for("never_happens", timeout=0.1)
        assert event is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
