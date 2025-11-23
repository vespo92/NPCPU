"""
Abstract Currency and Value Storage for NPCPU Economic System

Defines currency systems, wallets, and value transfer mechanisms.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class CurrencyType(Enum):
    """Types of currency in the system."""
    ENERGY_CREDITS = "energy_credits"      # Primary currency backed by computation
    TRUST_TOKENS = "trust_tokens"          # Social currency for reputation
    KNOWLEDGE_UNITS = "knowledge_units"    # Information-backed currency
    HARMONY_POINTS = "harmony_points"      # Reward for system contributions


class TransactionStatus(Enum):
    """Status of a currency transaction."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    REVERSED = "reversed"


@dataclass
class Currency:
    """
    A currency definition.

    Currencies are abstract value stores that can be exchanged
    between organisms.
    """
    currency_id: str
    name: str
    currency_type: CurrencyType
    symbol: str
    decimal_places: int = 2
    min_transaction: float = 0.01
    max_supply: Optional[float] = None  # None = unlimited
    current_supply: float = 0.0
    inflation_rate: float = 0.0  # Annual inflation
    backing_ratio: float = 1.0   # Ratio to backing asset
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, amount: float) -> str:
        """Format an amount in this currency."""
        return f"{self.symbol}{amount:.{self.decimal_places}f}"

    def is_valid_amount(self, amount: float) -> bool:
        """Check if amount is valid for this currency."""
        if amount < self.min_transaction:
            return False
        if self.max_supply and self.current_supply + amount > self.max_supply:
            return False
        return True

    def mint(self, amount: float) -> bool:
        """Mint new currency. Returns success."""
        if self.max_supply and self.current_supply + amount > self.max_supply:
            return False
        self.current_supply += amount
        return True

    def burn(self, amount: float) -> bool:
        """Burn currency. Returns success."""
        if amount > self.current_supply:
            return False
        self.current_supply -= amount
        return True


@dataclass
class Transaction:
    """
    A currency transaction between entities.

    All value transfers are recorded as transactions for
    transparency and audit purposes.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    currency_id: str = ""
    sender_id: str = ""
    receiver_id: str = ""
    amount: float = 0.0
    fee: float = 0.0
    status: TransactionStatus = TransactionStatus.PENDING
    memo: str = ""
    timestamp: float = field(default_factory=time.time)
    confirmed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Total amount including fee."""
        return self.amount + self.fee

    @property
    def hash(self) -> str:
        """Generate hash of transaction for verification."""
        data = f"{self.id}{self.sender_id}{self.receiver_id}{self.amount}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def confirm(self) -> None:
        """Confirm the transaction."""
        self.status = TransactionStatus.CONFIRMED
        self.confirmed_at = time.time()

    def reject(self, reason: str = "") -> None:
        """Reject the transaction."""
        self.status = TransactionStatus.REJECTED
        self.metadata["rejection_reason"] = reason

    def reverse(self, reason: str = "") -> None:
        """Reverse a confirmed transaction."""
        self.status = TransactionStatus.REVERSED
        self.metadata["reversal_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transaction to dictionary."""
        return {
            "id": self.id,
            "hash": self.hash,
            "currency_id": self.currency_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "amount": self.amount,
            "fee": self.fee,
            "total": self.total,
            "status": self.status.value,
            "memo": self.memo,
            "timestamp": self.timestamp,
            "confirmed_at": self.confirmed_at,
            "metadata": self.metadata
        }


@dataclass
class Balance:
    """Balance of a single currency."""
    currency_id: str
    available: float = 0.0
    reserved: float = 0.0  # Locked for pending transactions
    last_updated: float = field(default_factory=time.time)

    @property
    def total(self) -> float:
        """Total balance including reserved."""
        return self.available + self.reserved

    def credit(self, amount: float) -> bool:
        """Add to available balance."""
        if amount < 0:
            return False
        self.available += amount
        self.last_updated = time.time()
        return True

    def debit(self, amount: float) -> bool:
        """Remove from available balance."""
        if amount < 0 or amount > self.available:
            return False
        self.available -= amount
        self.last_updated = time.time()
        return True

    def reserve(self, amount: float) -> bool:
        """Move from available to reserved."""
        if amount > self.available:
            return False
        self.available -= amount
        self.reserved += amount
        self.last_updated = time.time()
        return True

    def release(self, amount: float) -> bool:
        """Move from reserved to available."""
        if amount > self.reserved:
            return False
        self.reserved -= amount
        self.available += amount
        self.last_updated = time.time()
        return True

    def commit_reservation(self, amount: float) -> bool:
        """Remove reserved balance (complete pending transaction)."""
        if amount > self.reserved:
            return False
        self.reserved -= amount
        self.last_updated = time.time()
        return True


@dataclass
class Wallet:
    """
    A wallet holding multiple currency balances.

    Each organism has a wallet for managing their currencies.
    """
    owner_id: str
    balances: Dict[str, Balance] = field(default_factory=dict)
    transaction_history: List[str] = field(default_factory=list)  # Transaction IDs
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_balance(self, currency_id: str) -> Balance:
        """Get balance for a currency. Creates if doesn't exist."""
        if currency_id not in self.balances:
            self.balances[currency_id] = Balance(currency_id=currency_id)
        return self.balances[currency_id]

    def get_available(self, currency_id: str) -> float:
        """Get available balance for a currency."""
        return self.get_balance(currency_id).available

    def get_total(self, currency_id: str) -> float:
        """Get total balance for a currency."""
        return self.get_balance(currency_id).total

    def credit(self, currency_id: str, amount: float) -> bool:
        """Credit currency to wallet."""
        return self.get_balance(currency_id).credit(amount)

    def debit(self, currency_id: str, amount: float) -> bool:
        """Debit currency from wallet."""
        return self.get_balance(currency_id).debit(amount)

    def can_afford(self, currency_id: str, amount: float) -> bool:
        """Check if wallet can afford an amount."""
        return self.get_available(currency_id) >= amount

    def record_transaction(self, transaction_id: str) -> None:
        """Record a transaction ID in history."""
        self.transaction_history.append(transaction_id)
        # Keep last 1000 transactions
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]

    def get_total_value(self, exchange_rates: Dict[str, float]) -> float:
        """
        Calculate total wallet value in base currency.

        Args:
            exchange_rates: Currency ID -> base currency rate
        """
        total = 0.0
        for currency_id, balance in self.balances.items():
            rate = exchange_rates.get(currency_id, 1.0)
            total += balance.total * rate
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize wallet to dictionary."""
        return {
            "owner_id": self.owner_id,
            "balances": {
                cid: {
                    "available": b.available,
                    "reserved": b.reserved,
                    "total": b.total
                }
                for cid, b in self.balances.items()
            },
            "transaction_count": len(self.transaction_history),
            "created_at": self.created_at,
            "metadata": self.metadata
        }


class CurrencySystem:
    """
    Central currency management system.

    Manages currency definitions, minting, and transaction processing.
    """

    def __init__(self):
        self._currencies: Dict[str, Currency] = {}
        self._wallets: Dict[str, Wallet] = {}
        self._transactions: Dict[str, Transaction] = {}
        self._pending_transactions: List[str] = []
        self._fee_rate: float = 0.001  # 0.1% transaction fee
        self._register_default_currencies()

    def _register_default_currencies(self):
        """Register default currencies."""
        self.register_currency(Currency(
            currency_id="EC",
            name="Energy Credits",
            currency_type=CurrencyType.ENERGY_CREDITS,
            symbol="⚡",
            decimal_places=4,
            min_transaction=0.0001,
            inflation_rate=0.02
        ))

        self.register_currency(Currency(
            currency_id="TT",
            name="Trust Tokens",
            currency_type=CurrencyType.TRUST_TOKENS,
            symbol="🤝",
            decimal_places=2,
            min_transaction=0.01,
            max_supply=1000000.0
        ))

        self.register_currency(Currency(
            currency_id="KU",
            name="Knowledge Units",
            currency_type=CurrencyType.KNOWLEDGE_UNITS,
            symbol="📚",
            decimal_places=2,
            min_transaction=0.01
        ))

        self.register_currency(Currency(
            currency_id="HP",
            name="Harmony Points",
            currency_type=CurrencyType.HARMONY_POINTS,
            symbol="☯",
            decimal_places=0,
            min_transaction=1.0,
            max_supply=10000000.0
        ))

    def register_currency(self, currency: Currency) -> None:
        """Register a new currency."""
        self._currencies[currency.currency_id] = currency

    def get_currency(self, currency_id: str) -> Optional[Currency]:
        """Get a currency by ID."""
        return self._currencies.get(currency_id)

    def get_wallet(self, owner_id: str) -> Wallet:
        """Get or create wallet for an owner."""
        if owner_id not in self._wallets:
            self._wallets[owner_id] = Wallet(owner_id=owner_id)
        return self._wallets[owner_id]

    def mint_to(self, owner_id: str, currency_id: str, amount: float) -> bool:
        """Mint currency directly to a wallet."""
        currency = self.get_currency(currency_id)
        if not currency:
            return False
        if not currency.mint(amount):
            return False

        wallet = self.get_wallet(owner_id)
        return wallet.credit(currency_id, amount)

    def burn_from(self, owner_id: str, currency_id: str, amount: float) -> bool:
        """Burn currency from a wallet."""
        currency = self.get_currency(currency_id)
        if not currency:
            return False

        wallet = self.get_wallet(owner_id)
        if not wallet.can_afford(currency_id, amount):
            return False

        if not wallet.debit(currency_id, amount):
            return False

        return currency.burn(amount)

    def transfer(
        self,
        sender_id: str,
        receiver_id: str,
        currency_id: str,
        amount: float,
        memo: str = ""
    ) -> Tuple[bool, Optional[Transaction]]:
        """
        Transfer currency between wallets.

        Returns (success, transaction).
        """
        currency = self.get_currency(currency_id)
        if not currency:
            return False, None

        if not currency.is_valid_amount(amount):
            return False, None

        sender_wallet = self.get_wallet(sender_id)
        receiver_wallet = self.get_wallet(receiver_id)

        fee = amount * self._fee_rate
        total = amount + fee

        if not sender_wallet.can_afford(currency_id, total):
            return False, None

        # Create transaction
        transaction = Transaction(
            currency_id=currency_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            fee=fee,
            memo=memo
        )

        # Reserve funds
        sender_balance = sender_wallet.get_balance(currency_id)
        if not sender_balance.reserve(total):
            transaction.reject("Insufficient funds")
            return False, transaction

        # Execute transfer
        sender_balance.commit_reservation(total)
        receiver_wallet.credit(currency_id, amount)

        # Record transaction
        transaction.confirm()
        self._transactions[transaction.id] = transaction
        sender_wallet.record_transaction(transaction.id)
        receiver_wallet.record_transaction(transaction.id)

        return True, transaction

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID."""
        return self._transactions.get(transaction_id)

    def get_exchange_rates(self) -> Dict[str, float]:
        """Get current exchange rates (relative to Energy Credits)."""
        return {
            "EC": 1.0,
            "TT": 2.5,
            "KU": 1.5,
            "HP": 0.5
        }

    def get_supply_info(self) -> Dict[str, Dict[str, Any]]:
        """Get supply information for all currencies."""
        return {
            cid: {
                "name": c.name,
                "current_supply": c.current_supply,
                "max_supply": c.max_supply,
                "inflation_rate": c.inflation_rate
            }
            for cid, c in self._currencies.items()
        }

    def tick(self) -> Dict[str, Any]:
        """Process periodic updates (inflation, etc.)."""
        # Apply minimal inflation per tick (assuming 1 tick = 1 second)
        # Annual inflation applied as 1/31536000 per tick
        for currency in self._currencies.values():
            if currency.inflation_rate > 0:
                tick_inflation = currency.inflation_rate / 31536000
                currency.current_supply *= (1 + tick_inflation)

        return {"currencies": len(self._currencies), "wallets": len(self._wallets)}


# Default global currency system
_default_currency_system = CurrencySystem()


def get_currency_system() -> CurrencySystem:
    """Get the default currency system."""
    return _default_currency_system
