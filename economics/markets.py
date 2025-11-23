"""
Market Dynamics and Price Discovery for NPCPU Economic System

Implements market mechanisms, order books, and price formation.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque


class OrderType(Enum):
    """Types of market orders."""
    MARKET = "market"       # Execute at current price
    LIMIT = "limit"         # Execute at specified price or better
    STOP = "stop"           # Trigger at specified price
    STOP_LIMIT = "stop_limit"  # Stop that becomes limit order


class OrderSide(Enum):
    """Side of the order."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status of an order."""
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class MarketState(Enum):
    """State of a market."""
    OPEN = "open"           # Trading allowed
    CLOSED = "closed"       # No trading
    HALTED = "halted"       # Temporarily stopped
    AUCTION = "auction"     # Price discovery mode


@dataclass
class Order:
    """
    A market order to buy or sell a resource.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    owner_id: str = ""
    order_type: OrderType = OrderType.LIMIT
    side: OrderSide = OrderSide.BUY
    quantity: float = 0.0
    filled_quantity: float = 0.0
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.OPEN
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining(self) -> float:
        """Remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Check if order can still be matched."""
        if self.status not in (OrderStatus.OPEN, OrderStatus.PARTIAL):
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def fill(self, quantity: float, fill_price: float) -> float:
        """
        Fill part of the order.

        Returns actually filled quantity.
        """
        fill_qty = min(quantity, self.remaining)
        self.filled_quantity += fill_qty

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

        return fill_qty

    def cancel(self) -> bool:
        """Cancel the order."""
        if self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False
        self.status = OrderStatus.CANCELLED
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize order to dictionary."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "owner_id": self.owner_id,
            "order_type": self.order_type.value,
            "side": self.side.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining": self.remaining,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at
        }


@dataclass
class Trade:
    """
    A completed trade between two orders.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_id: str = ""
    buy_order_id: str = ""
    sell_order_id: str = ""
    buyer_id: str = ""
    seller_id: str = ""
    quantity: float = 0.0
    price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Total trade value."""
        return self.quantity * self.price

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trade to dictionary."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "buy_order_id": self.buy_order_id,
            "sell_order_id": self.sell_order_id,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "quantity": self.quantity,
            "price": self.price,
            "value": self.value,
            "timestamp": self.timestamp
        }


@dataclass
class OrderBook:
    """
    Order book for a market.

    Maintains bid and ask orders sorted by price.
    """
    market_id: str
    bids: List[Order] = field(default_factory=list)  # Buy orders (highest first)
    asks: List[Order] = field(default_factory=list)  # Sell orders (lowest first)

    def add_order(self, order: Order) -> None:
        """Add an order to the book."""
        order.market_id = self.market_id

        if order.side == OrderSide.BUY:
            self.bids.append(order)
            self.bids.sort(key=lambda o: (-o.price if o.price else float('inf'), o.created_at))
        else:
            self.asks.append(order)
            self.asks.sort(key=lambda o: (o.price if o.price else 0, o.created_at))

    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove an order from the book."""
        for order_list in (self.bids, self.asks):
            for i, order in enumerate(order_list):
                if order.id == order_id:
                    return order_list.pop(i)
        return None

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""
        for order_list in (self.bids, self.asks):
            for order in order_list:
                if order.id == order_id:
                    return order
        return None

    def best_bid(self) -> Optional[float]:
        """Get highest bid price."""
        active_bids = [o for o in self.bids if o.is_active and o.price]
        return active_bids[0].price if active_bids else None

    def best_ask(self) -> Optional[float]:
        """Get lowest ask price."""
        active_asks = [o for o in self.asks if o.is_active and o.price]
        return active_asks[0].price if active_asks else None

    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask - bid
        return None

    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None

    def depth(self, side: OrderSide, levels: int = 5) -> List[Tuple[float, float]]:
        """
        Get order book depth.

        Returns list of (price, quantity) tuples.
        """
        orders = self.bids if side == OrderSide.BUY else self.asks
        active = [o for o in orders if o.is_active and o.price]

        # Aggregate by price
        price_levels: Dict[float, float] = {}
        for order in active:
            price = order.price
            price_levels[price] = price_levels.get(price, 0) + order.remaining

        # Sort and limit
        sorted_levels = sorted(price_levels.items(), reverse=(side == OrderSide.BUY))
        return sorted_levels[:levels]

    def clean_expired(self) -> List[Order]:
        """Remove expired orders. Returns removed orders."""
        expired = []
        current_time = time.time()

        for order_list in (self.bids, self.asks):
            for order in order_list[:]:
                if order.expires_at and current_time > order.expires_at:
                    order.status = OrderStatus.EXPIRED
                    order_list.remove(order)
                    expired.append(order)

        return expired

    def to_dict(self) -> Dict[str, Any]:
        """Serialize order book to dictionary."""
        return {
            "market_id": self.market_id,
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "spread": self.spread(),
            "mid_price": self.mid_price(),
            "bid_count": len([o for o in self.bids if o.is_active]),
            "ask_count": len([o for o in self.asks if o.is_active]),
            "bid_depth": self.depth(OrderSide.BUY),
            "ask_depth": self.depth(OrderSide.SELL)
        }


@dataclass
class PriceHistory:
    """
    Historical price data for a market.
    """
    market_id: str
    prices: deque = field(default_factory=lambda: deque(maxlen=10000))
    volumes: deque = field(default_factory=lambda: deque(maxlen=10000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10000))

    def record(self, price: float, volume: float = 0.0) -> None:
        """Record a price point."""
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(time.time())

    def last_price(self) -> Optional[float]:
        """Get most recent price."""
        return self.prices[-1] if self.prices else None

    def price_change(self, periods: int = 1) -> Optional[float]:
        """Get price change over periods."""
        if len(self.prices) < periods + 1:
            return None
        return self.prices[-1] - self.prices[-(periods + 1)]

    def price_change_pct(self, periods: int = 1) -> Optional[float]:
        """Get percentage price change over periods."""
        if len(self.prices) < periods + 1:
            return None
        old_price = self.prices[-(periods + 1)]
        if old_price == 0:
            return None
        return (self.prices[-1] - old_price) / old_price

    def volatility(self, periods: int = 20) -> Optional[float]:
        """Calculate price volatility over periods."""
        if len(self.prices) < periods:
            return None
        recent = list(self.prices)[-periods:]
        returns = np.diff(recent) / np.array(recent[:-1])
        return float(np.std(returns)) if len(returns) > 0 else None

    def vwap(self, periods: int = 20) -> Optional[float]:
        """Calculate volume-weighted average price."""
        if len(self.prices) < periods or len(self.volumes) < periods:
            return None
        prices = list(self.prices)[-periods:]
        volumes = list(self.volumes)[-periods:]
        total_volume = sum(volumes)
        if total_volume == 0:
            return None
        return sum(p * v for p, v in zip(prices, volumes)) / total_volume

    def moving_average(self, periods: int = 20) -> Optional[float]:
        """Calculate simple moving average."""
        if len(self.prices) < periods:
            return None
        return float(np.mean(list(self.prices)[-periods:]))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize price history summary."""
        return {
            "market_id": self.market_id,
            "last_price": self.last_price(),
            "change_24h": self.price_change(86400),  # Assuming 1 tick per second
            "change_pct_24h": self.price_change_pct(86400),
            "volatility_20": self.volatility(20),
            "vwap_20": self.vwap(20),
            "ma_20": self.moving_average(20),
            "data_points": len(self.prices)
        }


@dataclass
class Market:
    """
    A market for trading resources.

    Manages order matching, price discovery, and trade execution.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    base_resource: str = ""      # Resource being traded
    quote_currency: str = "EC"   # Currency used for pricing
    state: MarketState = MarketState.OPEN
    order_book: OrderBook = None
    price_history: PriceHistory = None
    last_price: float = 1.0
    min_order_size: float = 0.01
    max_order_size: float = 10000.0
    tick_size: float = 0.0001    # Minimum price increment
    maker_fee: float = 0.001     # Fee for providing liquidity
    taker_fee: float = 0.002     # Fee for taking liquidity
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.order_book is None:
            self.order_book = OrderBook(market_id=self.id)
        if self.price_history is None:
            self.price_history = PriceHistory(market_id=self.id)

    def place_order(self, order: Order) -> Tuple[bool, List[Trade]]:
        """
        Place an order in the market.

        Returns (success, list of resulting trades).
        """
        if self.state != MarketState.OPEN:
            return False, []

        if order.quantity < self.min_order_size:
            return False, []
        if order.quantity > self.max_order_size:
            return False, []

        # Round price to tick size
        if order.price:
            order.price = round(order.price / self.tick_size) * self.tick_size

        trades = []

        # Try to match immediately for market and limit orders
        if order.order_type in (OrderType.MARKET, OrderType.LIMIT):
            trades = self._match_order(order)

        # Add remaining quantity to book if limit order
        if order.remaining > 0 and order.order_type == OrderType.LIMIT:
            self.order_book.add_order(order)

        return True, trades

    def _match_order(self, order: Order) -> List[Trade]:
        """Match an order against the order book."""
        trades = []

        if order.side == OrderSide.BUY:
            opposite_orders = [o for o in self.order_book.asks if o.is_active]
        else:
            opposite_orders = [o for o in self.order_book.bids if o.is_active]

        for opposite in opposite_orders:
            if order.remaining <= 0:
                break

            # Check price compatibility
            if order.price and opposite.price:
                if order.side == OrderSide.BUY and order.price < opposite.price:
                    break
                if order.side == OrderSide.SELL and order.price > opposite.price:
                    break

            # Determine fill price (price improvement goes to taker)
            fill_price = opposite.price if opposite.price else order.price or self.last_price

            # Determine fill quantity
            fill_qty = min(order.remaining, opposite.remaining)

            # Execute trade
            order.fill(fill_qty, fill_price)
            opposite.fill(fill_qty, fill_price)

            # Create trade record
            if order.side == OrderSide.BUY:
                trade = Trade(
                    market_id=self.id,
                    buy_order_id=order.id,
                    sell_order_id=opposite.id,
                    buyer_id=order.owner_id,
                    seller_id=opposite.owner_id,
                    quantity=fill_qty,
                    price=fill_price
                )
            else:
                trade = Trade(
                    market_id=self.id,
                    buy_order_id=opposite.id,
                    sell_order_id=order.id,
                    buyer_id=opposite.owner_id,
                    seller_id=order.owner_id,
                    quantity=fill_qty,
                    price=fill_price
                )

            trades.append(trade)

            # Update market price
            self.last_price = fill_price
            self.price_history.record(fill_price, fill_qty)

            # Remove filled orders from book
            if opposite.status == OrderStatus.FILLED:
                self.order_book.remove_order(opposite.id)

        return trades

    def cancel_order(self, order_id: str, owner_id: str) -> bool:
        """Cancel an order. Owner must match."""
        order = self.order_book.get_order(order_id)
        if not order:
            return False
        if order.owner_id != owner_id:
            return False
        if order.cancel():
            self.order_book.remove_order(order_id)
            return True
        return False

    def get_quote(self, side: OrderSide, quantity: float) -> Optional[float]:
        """
        Get estimated price for a quantity.

        Returns average fill price or None if not enough liquidity.
        """
        if side == OrderSide.BUY:
            orders = [o for o in self.order_book.asks if o.is_active]
        else:
            orders = [o for o in self.order_book.bids if o.is_active]

        remaining = quantity
        total_cost = 0.0

        for order in orders:
            if remaining <= 0:
                break
            if not order.price:
                continue

            fill_qty = min(remaining, order.remaining)
            total_cost += fill_qty * order.price
            remaining -= fill_qty

        if remaining > 0:
            return None  # Not enough liquidity

        return total_cost / quantity

    def tick(self) -> Dict[str, Any]:
        """Process one time step."""
        # Clean expired orders
        expired = self.order_book.clean_expired()

        return {
            "expired_orders": len(expired),
            "active_bids": len([o for o in self.order_book.bids if o.is_active]),
            "active_asks": len([o for o in self.order_book.asks if o.is_active]),
            "last_price": self.last_price
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize market to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "base_resource": self.base_resource,
            "quote_currency": self.quote_currency,
            "state": self.state.value,
            "last_price": self.last_price,
            "order_book": self.order_book.to_dict(),
            "price_history": self.price_history.to_dict(),
            "min_order_size": self.min_order_size,
            "max_order_size": self.max_order_size,
            "tick_size": self.tick_size
        }


class MarketManager:
    """
    Central market management system.

    Creates and manages multiple markets.
    """

    def __init__(self):
        self._markets: Dict[str, Market] = {}
        self._trades: List[Trade] = []
        self._trade_callbacks: List[Callable[[Trade], None]] = []

    def create_market(
        self,
        name: str,
        base_resource: str,
        quote_currency: str = "EC",
        initial_price: float = 1.0,
        **kwargs
    ) -> Market:
        """Create a new market."""
        market = Market(
            name=name,
            base_resource=base_resource,
            quote_currency=quote_currency,
            last_price=initial_price,
            **kwargs
        )
        self._markets[market.id] = market
        return market

    def get_market(self, market_id: str) -> Optional[Market]:
        """Get a market by ID."""
        return self._markets.get(market_id)

    def get_market_by_resource(self, resource_type: str) -> Optional[Market]:
        """Get market for a resource type."""
        for market in self._markets.values():
            if market.base_resource == resource_type:
                return market
        return None

    def list_markets(self) -> List[Market]:
        """List all markets."""
        return list(self._markets.values())

    def place_order(
        self,
        market_id: str,
        owner_id: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT
    ) -> Tuple[bool, Optional[Order], List[Trade]]:
        """
        Place an order in a market.

        Returns (success, order, trades).
        """
        market = self.get_market(market_id)
        if not market:
            return False, None, []

        order = Order(
            market_id=market_id,
            owner_id=owner_id,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price
        )

        success, trades = market.place_order(order)

        if success:
            self._trades.extend(trades)
            for trade in trades:
                for callback in self._trade_callbacks:
                    callback(trade)

        return success, order if success else None, trades

    def cancel_order(self, market_id: str, order_id: str, owner_id: str) -> bool:
        """Cancel an order."""
        market = self.get_market(market_id)
        if not market:
            return False
        return market.cancel_order(order_id, owner_id)

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register a trade callback."""
        self._trade_callbacks.append(callback)

    def get_recent_trades(self, market_id: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        trades = self._trades
        if market_id:
            trades = [t for t in trades if t.market_id == market_id]
        return trades[-limit:]

    def tick(self) -> Dict[str, Any]:
        """Process one time step for all markets."""
        results = {}
        for market_id, market in self._markets.items():
            results[market_id] = market.tick()
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get overall market statistics."""
        return {
            "market_count": len(self._markets),
            "total_trades": len(self._trades),
            "markets": {m.id: m.to_dict() for m in self._markets.values()}
        }


# Default global market manager
_default_market_manager = MarketManager()


def get_market_manager() -> MarketManager:
    """Get the default market manager."""
    return _default_market_manager
