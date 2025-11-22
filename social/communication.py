"""
Communication System for NPCPU

Implements inter-organism communication with signals, message passing,
and deception detection.

Example:
    from social.communication import CommunicationSystem, SignalType
    from core.abstractions import BaseOrganism

    # Create communication system for an organism
    comm = CommunicationSystem("communication", owner=my_organism)

    # Broadcast a signal
    comm.broadcast_signal(
        signal_type=SignalType.WARNING,
        data={"threat": "predator", "direction": "north"},
        range=10.0
    )

    # Send a direct message
    comm.send_message(
        target_id="other_organism_id",
        content={"message": "food location", "coords": (10, 20)}
    )

    # Check for deception
    credibility = comm.assess_credibility("sender_id", message)
"""

from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid
import math

from core.abstractions import BaseSubsystem, BaseOrganism
from core.events import get_event_bus, Event


class SignalType(Enum):
    """Types of signals organisms can emit"""
    WARNING = "warning"           # Danger alert
    MATING = "mating"            # Mating call
    FOOD_LOCATION = "food"       # Food source discovered
    TERRITORY = "territory"       # Territory marker
    DISTRESS = "distress"        # Help request
    GREETING = "greeting"        # Social acknowledgment
    AGGRESSION = "aggression"    # Threat display
    SUBMISSION = "submission"    # Yield/surrender
    RALLY = "rally"              # Group assembly call
    CUSTOM = "custom"            # User-defined signal


class MessageType(Enum):
    """Types of direct messages"""
    INFORMATION = "information"
    REQUEST = "request"
    RESPONSE = "response"
    COMMAND = "command"
    QUERY = "query"
    PROPOSAL = "proposal"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"


@dataclass
class Signal:
    """
    A broadcast signal that propagates through space.

    Attributes:
        type: Type of signal
        source_id: ID of the emitting organism
        source_position: Position where signal was emitted
        data: Signal payload
        strength: Initial signal strength (0.0 to 1.0)
        range: Maximum propagation distance
        attenuation: How fast signal weakens with distance
        timestamp: When signal was emitted
        truthful: Whether the signal is honest (for deception tracking)
    """
    type: SignalType
    source_id: str
    source_position: Tuple[float, float] = (0.0, 0.0)
    data: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    range: float = 10.0
    attenuation: float = 0.1  # Strength loss per unit distance
    timestamp: datetime = field(default_factory=datetime.now)
    truthful: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_strength_at(self, position: Tuple[float, float]) -> float:
        """Calculate signal strength at a given position"""
        distance = math.sqrt(
            (position[0] - self.source_position[0]) ** 2 +
            (position[1] - self.source_position[1]) ** 2
        )
        if distance > self.range:
            return 0.0
        return max(0.0, self.strength - (distance * self.attenuation))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize signal to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "source_position": list(self.source_position),
            "data": self.data,
            "strength": self.strength,
            "range": self.range,
            "attenuation": self.attenuation,
            "timestamp": self.timestamp.isoformat(),
            "truthful": self.truthful
        }


@dataclass
class Message:
    """
    A direct message between two organisms.

    Attributes:
        sender_id: ID of sending organism
        recipient_id: ID of receiving organism
        type: Type of message
        content: Message content
        requires_response: Whether sender expects a reply
        truthful: Whether message is honest
        timestamp: When message was sent
    """
    sender_id: str
    recipient_id: str
    type: MessageType = MessageType.INFORMATION
    content: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    truthful: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary"""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "type": self.type.value,
            "content": self.content,
            "requires_response": self.requires_response,
            "truthful": self.truthful,
            "timestamp": self.timestamp.isoformat()
        }


class CommunicationSystem(BaseSubsystem):
    """
    Inter-organism communication subsystem.

    Provides:
    - Signal broadcasting (area-of-effect communication)
    - Direct messaging between organisms
    - Signal reception and filtering
    - Deception detection based on history
    - Communication range management

    Example:
        # Create for an organism
        comm = CommunicationSystem("communication", owner=organism)
        organism.add_subsystem(comm)

        # In tick, process received signals
        comm.tick()

        # Broadcast warning
        comm.broadcast_signal(SignalType.WARNING, {"threat": "fire"})

        # Send direct message
        comm.send_message("target_id", {"help": "need food"})

        # Check messages
        for msg in comm.get_received_messages():
            credibility = comm.assess_credibility(msg.sender_id, msg)
            if credibility > 0.5:
                process_message(msg)
    """

    def __init__(
        self,
        name: str = "communication",
        owner: Optional[BaseOrganism] = None,
        base_range: float = 10.0,
        deception_memory: int = 50
    ):
        """
        Initialize the communication system.

        Args:
            name: Subsystem name
            owner: Owning organism
            base_range: Default broadcast range
            deception_memory: How many past communications to remember for deception detection
        """
        super().__init__(name, owner)
        self._base_range = base_range
        self._deception_memory = deception_memory

        # Outbox and inbox
        self._pending_signals: List[Signal] = []
        self._received_signals: List[Signal] = []
        self._pending_messages: List[Message] = []
        self._received_messages: List[Message] = []

        # Communication history for deception detection
        self._communication_history: Dict[str, List[Tuple[datetime, bool]]] = {}

        # Signal handlers
        self._signal_handlers: Dict[SignalType, List[Callable[[Signal], None]]] = {}
        self._message_handlers: List[Callable[[Message], None]] = []

        # Position callback (since subsystem doesn't know position)
        self._position_callback: Optional[Callable[[], Tuple[float, float]]] = None

        # Subscribe to events
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Subscribe to relevant events"""
        bus = get_event_bus()

        # Listen for signals broadcast by other organisms
        bus.subscribe("signal.broadcast", self._on_signal_received)
        # Listen for direct messages
        bus.subscribe("message.sent", self._on_message_received)

    def _on_signal_received(self, event: Event):
        """Handle incoming signal events"""
        signal_data = event.data
        source_id = signal_data.get("source_id")

        # Don't receive our own signals
        if self._owner and source_id == self._owner.id:
            return

        # Create signal from event data
        signal = Signal(
            type=SignalType(signal_data["type"]),
            source_id=source_id,
            source_position=tuple(signal_data.get("position", (0, 0))),
            data=signal_data.get("data", {}),
            strength=signal_data.get("strength", 1.0),
            range=signal_data.get("range", 10.0),
            truthful=signal_data.get("truthful", True)
        )

        # Check if in range
        if self._position_callback:
            my_position = self._position_callback()
            received_strength = signal.get_strength_at(my_position)
            if received_strength > 0:
                self._received_signals.append(signal)

    def _on_message_received(self, event: Event):
        """Handle incoming message events"""
        msg_data = event.data
        recipient_id = msg_data.get("recipient_id")

        # Only process messages meant for us
        if not self._owner or recipient_id != self._owner.id:
            return

        message = Message(
            sender_id=msg_data["sender_id"],
            recipient_id=recipient_id,
            type=MessageType(msg_data.get("type", "information")),
            content=msg_data.get("content", {}),
            requires_response=msg_data.get("requires_response", False),
            truthful=msg_data.get("truthful", True)
        )

        self._received_messages.append(message)

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_position_callback(self, callback: Callable[[], Tuple[float, float]]):
        """
        Set a callback to get the organism's current position.

        This is needed because the subsystem doesn't inherently know
        the organism's position in the world.

        Example:
            comm.set_position_callback(lambda: (organism.x, organism.y))
        """
        self._position_callback = callback

    def set_base_range(self, range_value: float):
        """Set the default broadcast range"""
        self._base_range = max(0.1, range_value)

    # =========================================================================
    # Signal Broadcasting
    # =========================================================================

    def broadcast_signal(
        self,
        signal_type: SignalType,
        data: Optional[Dict[str, Any]] = None,
        strength: float = 1.0,
        range_override: Optional[float] = None,
        truthful: bool = True
    ) -> Signal:
        """
        Broadcast a signal to nearby organisms.

        Args:
            signal_type: Type of signal to broadcast
            data: Signal payload
            strength: Signal strength (0.0 to 1.0)
            range_override: Override default range
            truthful: Whether signal is honest (for deception tracking)

        Returns:
            The broadcast Signal

        Example:
            # Warn about a predator
            comm.broadcast_signal(
                SignalType.WARNING,
                {"threat_type": "predator", "threat_id": predator.id},
                strength=1.0
            )
        """
        if not self._owner:
            raise RuntimeError("CommunicationSystem requires an owner organism")

        position = (0.0, 0.0)
        if self._position_callback:
            position = self._position_callback()

        signal = Signal(
            type=signal_type,
            source_id=self._owner.id,
            source_position=position,
            data=data or {},
            strength=max(0.0, min(1.0, strength)),
            range=range_override or self._base_range,
            truthful=truthful
        )

        # Emit event for other organisms to receive
        bus = get_event_bus()
        bus.emit("signal.broadcast", {
            "type": signal_type.value,
            "source_id": self._owner.id,
            "position": list(position),
            "data": data or {},
            "strength": strength,
            "range": signal.range,
            "truthful": truthful
        })

        # Record for history (deception tracking)
        self._record_communication(self._owner.id, truthful)

        return signal

    # =========================================================================
    # Direct Messaging
    # =========================================================================

    def send_message(
        self,
        target_id: str,
        content: Dict[str, Any],
        msg_type: MessageType = MessageType.INFORMATION,
        requires_response: bool = False,
        truthful: bool = True
    ) -> Message:
        """
        Send a direct message to another organism.

        Args:
            target_id: ID of recipient organism
            content: Message content
            msg_type: Type of message
            requires_response: Whether we expect a reply
            truthful: Whether message is honest

        Returns:
            The sent Message

        Example:
            # Share food location
            comm.send_message(
                ally_id,
                {"type": "food", "location": (10, 20)},
                MessageType.INFORMATION
            )

            # Request help
            response = comm.send_message(
                friend_id,
                {"need": "defense"},
                MessageType.REQUEST,
                requires_response=True
            )
        """
        if not self._owner:
            raise RuntimeError("CommunicationSystem requires an owner organism")

        message = Message(
            sender_id=self._owner.id,
            recipient_id=target_id,
            type=msg_type,
            content=content,
            requires_response=requires_response,
            truthful=truthful
        )

        # Emit event for recipient
        bus = get_event_bus()
        bus.emit("message.sent", {
            "sender_id": self._owner.id,
            "recipient_id": target_id,
            "type": msg_type.value,
            "content": content,
            "requires_response": requires_response,
            "truthful": truthful
        })

        # Record for history
        self._record_communication(self._owner.id, truthful)

        return message

    def reply_to_message(
        self,
        original: Message,
        content: Dict[str, Any],
        accept: bool = True
    ) -> Message:
        """
        Reply to a received message.

        Args:
            original: The message being replied to
            content: Reply content
            accept: Whether this is an acceptance or rejection

        Returns:
            The reply Message
        """
        msg_type = MessageType.ACCEPTANCE if accept else MessageType.REJECTION

        return self.send_message(
            target_id=original.sender_id,
            content={
                "reply_to": original.id,
                **content
            },
            msg_type=msg_type
        )

    # =========================================================================
    # Reception
    # =========================================================================

    def get_received_signals(
        self,
        signal_type: Optional[SignalType] = None,
        clear: bool = True
    ) -> List[Signal]:
        """
        Get signals received since last check.

        Args:
            signal_type: Filter by signal type (None for all)
            clear: Whether to clear the signal buffer

        Returns:
            List of received Signals
        """
        if signal_type:
            signals = [s for s in self._received_signals if s.type == signal_type]
            if clear:
                self._received_signals = [
                    s for s in self._received_signals if s.type != signal_type
                ]
        else:
            signals = self._received_signals.copy()
            if clear:
                self._received_signals.clear()

        return signals

    def get_received_messages(self, clear: bool = True) -> List[Message]:
        """
        Get messages received since last check.

        Args:
            clear: Whether to clear the message buffer

        Returns:
            List of received Messages
        """
        messages = self._received_messages.copy()
        if clear:
            self._received_messages.clear()
        return messages

    def has_pending_signals(self) -> bool:
        """Check if there are unprocessed signals"""
        return len(self._received_signals) > 0

    def has_pending_messages(self) -> bool:
        """Check if there are unprocessed messages"""
        return len(self._received_messages) > 0

    # =========================================================================
    # Signal Handlers
    # =========================================================================

    def register_signal_handler(
        self,
        signal_type: SignalType,
        handler: Callable[[Signal], None]
    ):
        """
        Register a handler for a specific signal type.

        Handlers are called automatically during tick() for matching signals.

        Example:
            def handle_warning(signal):
                if signal.data.get("threat_type") == "predator":
                    organism.flee()

            comm.register_signal_handler(SignalType.WARNING, handle_warning)
        """
        if signal_type not in self._signal_handlers:
            self._signal_handlers[signal_type] = []
        self._signal_handlers[signal_type].append(handler)

    def register_message_handler(self, handler: Callable[[Message], None]):
        """
        Register a handler for incoming messages.

        Example:
            def handle_message(msg):
                if msg.type == MessageType.REQUEST:
                    process_request(msg)

            comm.register_message_handler(handle_message)
        """
        self._message_handlers.append(handler)

    # =========================================================================
    # Deception Detection
    # =========================================================================

    def _record_communication(self, organism_id: str, truthful: bool):
        """Record a communication for deception tracking"""
        if organism_id not in self._communication_history:
            self._communication_history[organism_id] = []

        history = self._communication_history[organism_id]
        history.append((datetime.now(), truthful))

        # Limit history size
        if len(history) > self._deception_memory:
            history.pop(0)

    def record_verification(self, organism_id: str, was_truthful: bool):
        """
        Record whether a communication was verified as truthful.

        Call this when you've verified whether a signal/message was honest.
        This improves future credibility assessments.

        Example:
            # Received warning about predator
            # Later verified there was no predator
            comm.record_verification(signal.source_id, was_truthful=False)
        """
        self._record_communication(organism_id, was_truthful)

    def assess_credibility(
        self,
        organism_id: str,
        communication: Optional[Any] = None
    ) -> float:
        """
        Assess the credibility of an organism based on communication history.

        Returns a score from 0.0 (completely unreliable) to 1.0 (completely reliable).

        Args:
            organism_id: ID of the organism to assess
            communication: Optional specific communication to assess

        Returns:
            Credibility score

        Example:
            credibility = comm.assess_credibility(signal.source_id)
            if credibility > 0.7:
                trust_the_signal()
            elif credibility < 0.3:
                ignore_the_signal()
        """
        history = self._communication_history.get(organism_id, [])

        if not history:
            return 0.5  # Unknown = neutral credibility

        # Weight recent communications more heavily
        weighted_sum = 0.0
        total_weight = 0.0

        for i, (timestamp, truthful) in enumerate(history):
            # More recent = higher weight
            recency_weight = (i + 1) / len(history)
            weighted_sum += recency_weight * (1.0 if truthful else 0.0)
            total_weight += recency_weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def get_known_deceivers(self, threshold: float = 0.3) -> List[str]:
        """
        Get organisms with credibility below threshold.

        Returns list of organism IDs that have been caught lying frequently.
        """
        deceivers = []
        for org_id in self._communication_history:
            if self.assess_credibility(org_id) < threshold:
                deceivers.append(org_id)
        return deceivers

    # =========================================================================
    # BaseSubsystem Implementation
    # =========================================================================

    def tick(self) -> None:
        """
        Process one time step.

        - Dispatches received signals to handlers
        - Dispatches received messages to handlers
        - Cleans up old signals
        """
        if not self._enabled:
            return

        # Process signals
        for signal in self._received_signals[:]:
            if signal.type in self._signal_handlers:
                for handler in self._signal_handlers[signal.type]:
                    try:
                        handler(signal)
                    except Exception:
                        pass  # Handler errors shouldn't crash the system

        # Process messages
        for message in self._received_messages[:]:
            for handler in self._message_handlers:
                try:
                    handler(message)
                except Exception:
                    pass

    def get_state(self) -> Dict[str, Any]:
        """Get current subsystem state"""
        return {
            "base_range": self._base_range,
            "pending_signals": len(self._received_signals),
            "pending_messages": len(self._received_messages),
            "known_organisms": len(self._communication_history),
            "communication_history": {
                org_id: [
                    {"timestamp": ts.isoformat(), "truthful": tf}
                    for ts, tf in history[-10:]  # Last 10 for each
                ]
                for org_id, history in self._communication_history.items()
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore subsystem state"""
        self._base_range = state.get("base_range", self._base_range)

        # Restore communication history
        for org_id, history_data in state.get("communication_history", {}).items():
            self._communication_history[org_id] = [
                (datetime.fromisoformat(h["timestamp"]), h["truthful"])
                for h in history_data
            ]

    def reset(self) -> None:
        """Reset to initial state"""
        super().reset()
        self._received_signals.clear()
        self._received_messages.clear()
        self._communication_history.clear()


class CommunicationChannel:
    """
    A shared communication channel for broadcast communication.

    Useful for coordinating communication in a simulation without
    going through the event bus for every signal.

    Example:
        # Create a global channel
        channel = CommunicationChannel("main", range=100.0)

        # Organisms subscribe
        channel.subscribe(organism.id, (organism.x, organism.y))

        # Broadcast
        channel.broadcast(signal)

        # Get signals at a location
        signals = channel.get_signals_at((10, 20))
    """

    def __init__(self, name: str, default_range: float = 50.0, signal_ttl: float = 5.0):
        """
        Initialize a communication channel.

        Args:
            name: Channel name
            default_range: Default signal range
            signal_ttl: How long signals persist (in ticks)
        """
        self._name = name
        self._default_range = default_range
        self._signal_ttl = signal_ttl

        # Subscribers: organism_id -> position
        self._subscribers: Dict[str, Tuple[float, float]] = {}

        # Active signals
        self._active_signals: List[Tuple[Signal, int]] = []  # (signal, remaining_ttl)

    @property
    def name(self) -> str:
        return self._name

    def subscribe(self, organism_id: str, position: Tuple[float, float]):
        """Subscribe an organism to this channel"""
        self._subscribers[organism_id] = position

    def unsubscribe(self, organism_id: str):
        """Unsubscribe an organism"""
        self._subscribers.pop(organism_id, None)

    def update_position(self, organism_id: str, position: Tuple[float, float]):
        """Update a subscriber's position"""
        if organism_id in self._subscribers:
            self._subscribers[organism_id] = position

    def broadcast(self, signal: Signal) -> int:
        """
        Broadcast a signal on this channel.

        Returns the number of organisms in range.
        """
        self._active_signals.append((signal, int(self._signal_ttl)))

        # Count organisms in range
        in_range = 0
        for org_id, position in self._subscribers.items():
            if org_id != signal.source_id:
                if signal.get_strength_at(position) > 0:
                    in_range += 1

        return in_range

    def get_signals_at(
        self,
        position: Tuple[float, float],
        signal_type: Optional[SignalType] = None
    ) -> List[Tuple[Signal, float]]:
        """
        Get all signals receivable at a position.

        Returns list of (signal, strength) tuples.
        """
        receivable = []
        for signal, _ in self._active_signals:
            if signal_type and signal.type != signal_type:
                continue
            strength = signal.get_strength_at(position)
            if strength > 0:
                receivable.append((signal, strength))
        return receivable

    def tick(self):
        """Advance channel by one time step"""
        # Age signals and remove expired ones
        self._active_signals = [
            (signal, ttl - 1)
            for signal, ttl in self._active_signals
            if ttl > 1
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get channel statistics"""
        return {
            "name": self._name,
            "subscribers": len(self._subscribers),
            "active_signals": len(self._active_signals),
            "default_range": self._default_range
        }
