"""
Inter-Organism Trade Protocols for NPCPU Economic System

Implements direct trade, negotiations, and exchange protocols
between organisms.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.abstractions import BaseSubsystem


class TradeStatus(Enum):
    """Status of a trade proposal."""
    PROPOSED = "proposed"
    COUNTERED = "countered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    COMPLETED = "completed"
    FAILED = "failed"


class TradeType(Enum):
    """Types of trades."""
    DIRECT = "direct"           # Simple resource exchange
    BARTER = "barter"           # Multi-resource swap
    AUCTION = "auction"         # Competitive bidding
    CONTRACT = "contract"       # Long-term agreement
    GIFT = "gift"               # One-way transfer


@dataclass
class TradeItem:
    """
    An item in a trade offer.

    Can be a resource, currency, or service.
    """
    item_type: str              # "resource", "currency", "service"
    type_id: str                # Resource type or currency ID
    quantity: float = 0.0
    quality: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_type": self.item_type,
            "type_id": self.type_id,
            "quantity": self.quantity,
            "quality": self.quality,
            "metadata": self.metadata
        }


@dataclass
class TradeOffer:
    """
    A trade offer between two parties.

    Contains what each party gives and receives.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposer_id: str = ""
    responder_id: str = ""
    trade_type: TradeType = TradeType.DIRECT
    proposer_offers: List[TradeItem] = field(default_factory=list)
    proposer_requests: List[TradeItem] = field(default_factory=list)
    status: TradeStatus = TradeStatus.PROPOSED
    message: str = ""
    counter_offer_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)  # 1 hour default
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if offer is still active."""
        if self.status not in (TradeStatus.PROPOSED, TradeStatus.COUNTERED):
            return False
        return time.time() < self.expires_at

    def accept(self) -> None:
        """Accept the trade offer."""
        self.status = TradeStatus.ACCEPTED

    def reject(self, reason: str = "") -> None:
        """Reject the trade offer."""
        self.status = TradeStatus.REJECTED
        if reason:
            self.metadata["rejection_reason"] = reason

    def complete(self) -> None:
        """Mark trade as completed."""
        self.status = TradeStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self, reason: str = "") -> None:
        """Mark trade as failed."""
        self.status = TradeStatus.FAILED
        if reason:
            self.metadata["failure_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trade offer to dictionary."""
        return {
            "id": self.id,
            "proposer_id": self.proposer_id,
            "responder_id": self.responder_id,
            "trade_type": self.trade_type.value,
            "proposer_offers": [i.to_dict() for i in self.proposer_offers],
            "proposer_requests": [i.to_dict() for i in self.proposer_requests],
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active
        }


@dataclass
class TradeContract:
    """
    A long-term trade agreement.

    Specifies recurring exchanges between parties.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    party_a_id: str = ""
    party_b_id: str = ""
    terms: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    interval: float = 3600.0  # Exchange interval in seconds
    last_execution: Optional[float] = None
    execution_count: int = 0
    max_executions: Optional[int] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_execute(self) -> bool:
        """Check if contract should execute now."""
        if not self.active:
            return False
        if self.end_time and time.time() > self.end_time:
            return False
        if self.max_executions and self.execution_count >= self.max_executions:
            return False
        if self.last_execution is None:
            return True
        return time.time() >= self.last_execution + self.interval

    def execute(self) -> None:
        """Mark contract as executed."""
        self.execution_count += 1
        self.last_execution = time.time()

    def terminate(self, reason: str = "") -> None:
        """Terminate the contract."""
        self.active = False
        if reason:
            self.metadata["termination_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract to dictionary."""
        return {
            "id": self.id,
            "party_a_id": self.party_a_id,
            "party_b_id": self.party_b_id,
            "terms": self.terms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "interval": self.interval,
            "execution_count": self.execution_count,
            "active": self.active
        }


@dataclass
class TradeHistory:
    """
    Trade history for an entity.
    """
    entity_id: str
    completed_trades: List[str] = field(default_factory=list)  # Trade IDs
    total_volume_given: float = 0.0
    total_volume_received: float = 0.0
    trade_partners: Dict[str, int] = field(default_factory=dict)  # Partner ID -> count
    success_rate: float = 1.0
    reputation_score: float = 0.5

    def record_trade(
        self,
        trade_id: str,
        partner_id: str,
        volume_given: float,
        volume_received: float,
        success: bool = True
    ) -> None:
        """Record a completed trade."""
        self.completed_trades.append(trade_id)
        self.total_volume_given += volume_given
        self.total_volume_received += volume_received
        self.trade_partners[partner_id] = self.trade_partners.get(partner_id, 0) + 1

        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate

        # Update reputation based on trading activity
        if success:
            self.reputation_score = min(1.0, self.reputation_score + 0.01)
        else:
            self.reputation_score = max(0.0, self.reputation_score - 0.05)

        # Keep last 1000 trades
        if len(self.completed_trades) > 1000:
            self.completed_trades = self.completed_trades[-1000:]

    def get_partner_trust(self, partner_id: str) -> float:
        """Get trust score for a trading partner."""
        if partner_id not in self.trade_partners:
            return 0.5  # Default trust for new partners
        trade_count = self.trade_partners[partner_id]
        # Trust increases with number of successful trades
        return min(1.0, 0.5 + (trade_count * 0.05))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize history to dictionary."""
        return {
            "entity_id": self.entity_id,
            "trade_count": len(self.completed_trades),
            "total_volume_given": self.total_volume_given,
            "total_volume_received": self.total_volume_received,
            "unique_partners": len(self.trade_partners),
            "success_rate": self.success_rate,
            "reputation_score": self.reputation_score
        }


class TradeNegotiator:
    """
    Handles trade negotiations between parties.

    Provides utilities for evaluating and responding to offers.
    """

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.preferences: Dict[str, float] = {}  # Item type -> preference weight
        self.min_acceptable_ratio: float = 0.8   # Minimum value ratio to accept

    def evaluate_offer(
        self,
        offer: TradeOffer,
        my_valuations: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Evaluate a trade offer.

        Returns (score, recommendation).
        Score > 1.0 means favorable, < 1.0 means unfavorable.
        """
        # Calculate value of what we give
        give_value = 0.0
        for item in offer.proposer_requests:
            base_value = my_valuations.get(item.type_id, 1.0)
            give_value += item.quantity * base_value

        # Calculate value of what we receive
        receive_value = 0.0
        for item in offer.proposer_offers:
            base_value = my_valuations.get(item.type_id, 1.0)
            pref_mult = self.preferences.get(item.type_id, 1.0)
            receive_value += item.quantity * base_value * pref_mult

        if give_value == 0:
            return float('inf'), "accept"

        ratio = receive_value / give_value

        if ratio >= 1.2:
            return ratio, "accept"
        elif ratio >= self.min_acceptable_ratio:
            return ratio, "consider"
        elif ratio >= 0.5:
            return ratio, "counter"
        else:
            return ratio, "reject"

    def generate_counter_offer(
        self,
        original: TradeOffer,
        my_valuations: Dict[str, float],
        target_ratio: float = 1.0
    ) -> TradeOffer:
        """Generate a counter-offer."""
        # Calculate current values
        give_value = sum(
            item.quantity * my_valuations.get(item.type_id, 1.0)
            for item in original.proposer_requests
        )

        # Adjust quantities to achieve target ratio
        counter = TradeOffer(
            proposer_id=self.entity_id,
            responder_id=original.proposer_id,
            trade_type=original.trade_type,
            proposer_offers=original.proposer_requests.copy(),  # Swap
            proposer_requests=[],
            counter_offer_id=original.id,
            message=f"Counter-offer to {original.id}"
        )

        # Adjust requested amounts
        for item in original.proposer_offers:
            base_value = my_valuations.get(item.type_id, 1.0)
            needed_quantity = (give_value * target_ratio) / base_value
            counter.proposer_requests.append(TradeItem(
                item_type=item.item_type,
                type_id=item.type_id,
                quantity=min(item.quantity, needed_quantity),
                quality=item.quality
            ))

        return counter


class TradeSystem(BaseSubsystem):
    """
    Central trade management system.

    Manages trade offers, contracts, and execution.
    """

    def __init__(self, name: str = "trade_system", owner=None):
        super().__init__(name, owner)
        self._offers: Dict[str, TradeOffer] = {}
        self._contracts: Dict[str, TradeContract] = {}
        self._histories: Dict[str, TradeHistory] = {}
        self._offer_callbacks: List[Callable[[TradeOffer], None]] = []
        self._trade_validators: List[Callable[[TradeOffer], Tuple[bool, str]]] = []

    def get_history(self, entity_id: str) -> TradeHistory:
        """Get or create trade history for an entity."""
        if entity_id not in self._histories:
            self._histories[entity_id] = TradeHistory(entity_id=entity_id)
        return self._histories[entity_id]

    def propose_trade(
        self,
        proposer_id: str,
        responder_id: str,
        offers: List[TradeItem],
        requests: List[TradeItem],
        trade_type: TradeType = TradeType.DIRECT,
        message: str = "",
        expires_in: float = 3600.0
    ) -> Tuple[bool, Optional[TradeOffer]]:
        """
        Propose a trade.

        Returns (success, offer).
        """
        offer = TradeOffer(
            proposer_id=proposer_id,
            responder_id=responder_id,
            trade_type=trade_type,
            proposer_offers=offers,
            proposer_requests=requests,
            message=message,
            expires_at=time.time() + expires_in
        )

        # Run validators
        for validator in self._trade_validators:
            valid, reason = validator(offer)
            if not valid:
                offer.reject(reason)
                return False, offer

        self._offers[offer.id] = offer

        # Notify callbacks
        for callback in self._offer_callbacks:
            callback(offer)

        return True, offer

    def get_offer(self, offer_id: str) -> Optional[TradeOffer]:
        """Get a trade offer by ID."""
        return self._offers.get(offer_id)

    def get_pending_offers(self, entity_id: str) -> List[TradeOffer]:
        """Get pending offers for an entity."""
        return [
            offer for offer in self._offers.values()
            if offer.is_active and offer.responder_id == entity_id
        ]

    def get_sent_offers(self, entity_id: str) -> List[TradeOffer]:
        """Get offers sent by an entity."""
        return [
            offer for offer in self._offers.values()
            if offer.proposer_id == entity_id
        ]

    def respond_to_offer(
        self,
        offer_id: str,
        responder_id: str,
        accept: bool,
        reason: str = ""
    ) -> Tuple[bool, str]:
        """
        Respond to a trade offer.

        Returns (success, message).
        """
        offer = self.get_offer(offer_id)
        if not offer:
            return False, "Offer not found"
        if offer.responder_id != responder_id:
            return False, "Not authorized to respond"
        if not offer.is_active:
            return False, "Offer is no longer active"

        if accept:
            offer.accept()
            return True, "Offer accepted"
        else:
            offer.reject(reason)
            return True, "Offer rejected"

    def counter_offer(
        self,
        original_offer_id: str,
        responder_id: str,
        new_offers: List[TradeItem],
        new_requests: List[TradeItem],
        message: str = ""
    ) -> Tuple[bool, Optional[TradeOffer]]:
        """
        Make a counter-offer.

        Returns (success, counter_offer).
        """
        original = self.get_offer(original_offer_id)
        if not original:
            return False, None
        if original.responder_id != responder_id:
            return False, None
        if not original.is_active:
            return False, None

        # Mark original as countered
        original.status = TradeStatus.COUNTERED

        # Create counter-offer
        counter = TradeOffer(
            proposer_id=responder_id,
            responder_id=original.proposer_id,
            trade_type=original.trade_type,
            proposer_offers=new_offers,
            proposer_requests=new_requests,
            message=message,
            counter_offer_id=original_offer_id
        )

        self._offers[counter.id] = counter
        return True, counter

    def execute_trade(
        self,
        offer_id: str,
        executor: Optional[Callable[[TradeOffer], Tuple[bool, str]]] = None
    ) -> Tuple[bool, str]:
        """
        Execute an accepted trade.

        If executor is provided, it handles the actual resource transfer.
        Returns (success, message).
        """
        offer = self.get_offer(offer_id)
        if not offer:
            return False, "Offer not found"
        if offer.status != TradeStatus.ACCEPTED:
            return False, "Offer not accepted"

        # Execute through custom executor or mark as complete
        if executor:
            success, message = executor(offer)
            if success:
                offer.complete()

                # Record in histories
                proposer_volume = sum(i.quantity for i in offer.proposer_offers)
                responder_volume = sum(i.quantity for i in offer.proposer_requests)

                self.get_history(offer.proposer_id).record_trade(
                    offer.id, offer.responder_id, proposer_volume, responder_volume, True
                )
                self.get_history(offer.responder_id).record_trade(
                    offer.id, offer.proposer_id, responder_volume, proposer_volume, True
                )

                return True, message
            else:
                offer.fail(message)
                return False, message
        else:
            offer.complete()
            return True, "Trade marked as complete"

    def create_contract(
        self,
        party_a_id: str,
        party_b_id: str,
        terms: Dict[str, Any],
        interval: float = 3600.0,
        duration: Optional[float] = None,
        max_executions: Optional[int] = None
    ) -> TradeContract:
        """Create a recurring trade contract."""
        contract = TradeContract(
            party_a_id=party_a_id,
            party_b_id=party_b_id,
            terms=terms,
            interval=interval,
            end_time=time.time() + duration if duration else None,
            max_executions=max_executions
        )
        self._contracts[contract.id] = contract
        return contract

    def get_contract(self, contract_id: str) -> Optional[TradeContract]:
        """Get a contract by ID."""
        return self._contracts.get(contract_id)

    def get_entity_contracts(self, entity_id: str) -> List[TradeContract]:
        """Get all contracts for an entity."""
        return [
            c for c in self._contracts.values()
            if c.active and (c.party_a_id == entity_id or c.party_b_id == entity_id)
        ]

    def on_offer(self, callback: Callable[[TradeOffer], None]) -> None:
        """Register callback for new offers."""
        self._offer_callbacks.append(callback)

    def add_validator(
        self,
        validator: Callable[[TradeOffer], Tuple[bool, str]]
    ) -> None:
        """Add a trade validator."""
        self._trade_validators.append(validator)

    def tick(self) -> None:
        """Process one time step."""
        if not self.enabled:
            return

        current_time = time.time()

        # Expire old offers
        for offer in self._offers.values():
            if offer.is_active and current_time > offer.expires_at:
                offer.status = TradeStatus.EXPIRED

        # Process contracts
        for contract in self._contracts.values():
            if contract.should_execute():
                contract.execute()
                # Actual execution would be handled by integration layer

    def get_state(self) -> Dict[str, Any]:
        """Get subsystem state."""
        return {
            "active_offers": len([o for o in self._offers.values() if o.is_active]),
            "active_contracts": len([c for c in self._contracts.values() if c.active]),
            "total_offers": len(self._offers),
            "total_contracts": len(self._contracts)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore subsystem state."""
        pass  # State restoration would be handled by serialization layer

    def to_dict(self) -> Dict[str, Any]:
        """Serialize system to dictionary."""
        return {
            "offers": {oid: o.to_dict() for oid, o in self._offers.items()},
            "contracts": {cid: c.to_dict() for cid, c in self._contracts.items()},
            "histories": {eid: h.to_dict() for eid, h in self._histories.items()}
        }


# Default global trade system
_default_trade_system = TradeSystem()


def get_trade_system() -> TradeSystem:
    """Get the default trade system."""
    return _default_trade_system
