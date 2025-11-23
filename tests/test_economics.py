"""
Comprehensive Test Suite for NPCPU Economics System

Tests resource management, currency, markets, trading, scarcity,
wealth distribution, and economic agents.
"""

import pytest
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from economics.resources import (
    Resource, ResourceType, ResourceCategory, ResourceState, ResourceQuality,
    ResourcePool, ResourceRegistry, get_resource_registry
)
from economics.currency import (
    Currency, CurrencyType, Transaction, TransactionStatus,
    Balance, Wallet, CurrencySystem, get_currency_system
)
from economics.markets import (
    Market, MarketState, Order, OrderType, OrderSide, OrderStatus,
    Trade, OrderBook, PriceHistory, MarketManager, get_market_manager
)
from economics.trade import (
    TradeOffer, TradeItem, TradeStatus, TradeType,
    TradeNegotiator, TradeSystem, get_trade_system
)
from economics.scarcity import (
    ScarcityLevel, CompetitionMode, ResourceSupply, ResourceDemand,
    ScarcityState, ResourceAllocator, ScarcityModel, get_scarcity_model
)
from economics.wealth_distribution import (
    WealthClass, WealthRecord, GiniCalculator,
    WealthDistributionTracker, get_wealth_tracker
)
from economics.economic_agents import (
    AgentStrategy, EconomicGoal, EconomicDecision, DecisionType,
    UtilityFunction, EconomicAgent, ProducerAgent, ConsumerAgent,
    create_economic_agent
)


# =============================================================================
# Resource Tests
# =============================================================================

class TestResourceTypes:
    """Tests for resource type definitions."""

    def test_resource_type_creation(self):
        """Test creating a resource type."""
        rt = ResourceType(
            type_id="test_energy",
            name="Test Energy",
            category=ResourceCategory.ENERGY,
            base_value=1.5
        )
        assert rt.type_id == "test_energy"
        assert rt.category == ResourceCategory.ENERGY
        assert rt.base_value == 1.5

    def test_resource_type_equality(self):
        """Test resource type equality."""
        rt1 = ResourceType(type_id="test", name="Test", category=ResourceCategory.ENERGY)
        rt2 = ResourceType(type_id="test", name="Test", category=ResourceCategory.ENERGY)
        rt3 = ResourceType(type_id="other", name="Other", category=ResourceCategory.ENERGY)
        assert rt1 == rt2
        assert rt1 != rt3


class TestResource:
    """Tests for resource instances."""

    @pytest.fixture
    def energy_type(self):
        """Create an energy resource type."""
        return ResourceType(
            type_id="energy",
            name="Energy",
            category=ResourceCategory.ENERGY,
            base_value=1.0,
            decay_rate=0.01,
            renewable=True,
            regen_rate=0.1
        )

    def test_resource_creation(self, energy_type):
        """Test creating a resource."""
        resource = Resource(
            resource_type=energy_type,
            quantity=100.0,
            quality=ResourceQuality.STANDARD
        )
        assert resource.quantity == 100.0
        assert resource.quality == ResourceQuality.STANDARD
        assert resource.state == ResourceState.AVAILABLE

    def test_resource_value(self, energy_type):
        """Test resource value calculation."""
        resource = Resource(resource_type=energy_type, quantity=100.0)
        assert resource.value == 100.0  # 100 * 1.0 * 1.0

        resource.quality = ResourceQuality.REFINED
        assert resource.value == 150.0  # 100 * 1.0 * 1.5

    def test_resource_decay(self, energy_type):
        """Test resource decay."""
        resource = Resource(resource_type=energy_type, quantity=100.0)
        decayed = resource.decay()
        assert decayed == 1.0  # 100 * 0.01
        assert resource.quantity == 99.0

    def test_resource_split(self, energy_type):
        """Test splitting a resource."""
        resource = Resource(resource_type=energy_type, quantity=100.0)
        split = resource.split(30.0)
        assert split is not None
        assert split.quantity == 30.0
        assert resource.quantity == 70.0

    def test_resource_merge(self, energy_type):
        """Test merging resources."""
        resource1 = Resource(resource_type=energy_type, quantity=50.0)
        resource2 = Resource(resource_type=energy_type, quantity=30.0)
        result = resource1.merge(resource2)
        assert result is True
        assert resource1.quantity == 80.0


class TestResourcePool:
    """Tests for resource pools."""

    @pytest.fixture
    def registry(self):
        """Create a resource registry."""
        return ResourceRegistry()

    @pytest.fixture
    def pool(self, registry):
        """Create a resource pool."""
        return ResourcePool(owner_id="test_owner", registry=registry)

    def test_add_resource(self, pool, registry):
        """Test adding a resource to pool."""
        resource = registry.create_resource("energy_compute", 50.0)
        result = pool.add(resource)
        assert result is True
        assert pool.get_total_by_type("energy_compute") == 50.0

    def test_withdraw_resource(self, pool, registry):
        """Test withdrawing from pool."""
        resource = registry.create_resource("energy_compute", 100.0)
        pool.add(resource)
        withdrawn = pool.withdraw("energy_compute", 30.0)
        assert withdrawn is not None
        assert withdrawn.quantity == 30.0
        assert pool.get_total_by_type("energy_compute") == 70.0


# =============================================================================
# Currency Tests
# =============================================================================

class TestCurrency:
    """Tests for currency system."""

    @pytest.fixture
    def currency_system(self):
        """Create a fresh currency system."""
        return CurrencySystem()

    def test_default_currencies(self, currency_system):
        """Test default currencies are registered."""
        assert currency_system.get_currency("EC") is not None
        assert currency_system.get_currency("TT") is not None

    def test_mint_currency(self, currency_system):
        """Test minting currency."""
        result = currency_system.mint_to("user1", "EC", 100.0)
        assert result is True
        wallet = currency_system.get_wallet("user1")
        assert wallet.get_available("EC") == 100.0

    def test_transfer_currency(self, currency_system):
        """Test transferring currency."""
        currency_system.mint_to("sender", "EC", 100.0)
        success, tx = currency_system.transfer("sender", "receiver", "EC", 50.0)
        assert success is True
        assert tx.status == TransactionStatus.CONFIRMED
        sender_wallet = currency_system.get_wallet("sender")
        receiver_wallet = currency_system.get_wallet("receiver")
        assert receiver_wallet.get_available("EC") == 50.0
        assert sender_wallet.get_available("EC") < 50.0  # Minus fee


class TestWallet:
    """Tests for wallet functionality."""

    def test_wallet_balance(self):
        """Test wallet balance operations."""
        wallet = Wallet(owner_id="test")
        wallet.credit("EC", 100.0)
        assert wallet.get_available("EC") == 100.0
        wallet.debit("EC", 30.0)
        assert wallet.get_available("EC") == 70.0

    def test_wallet_can_afford(self):
        """Test can_afford check."""
        wallet = Wallet(owner_id="test")
        wallet.credit("EC", 100.0)
        assert wallet.can_afford("EC", 50.0) is True
        assert wallet.can_afford("EC", 150.0) is False


# =============================================================================
# Market Tests
# =============================================================================

class TestMarket:
    """Tests for market functionality."""

    @pytest.fixture
    def market(self):
        """Create a test market."""
        return Market(
            name="Test Market",
            base_resource="test_resource",
            quote_currency="EC",
            last_price=10.0
        )

    def test_market_creation(self, market):
        """Test market creation."""
        assert market.name == "Test Market"
        assert market.state == MarketState.OPEN
        assert market.last_price == 10.0

    def test_place_order(self, market):
        """Test placing an order."""
        order = Order(
            owner_id="user1",
            side=OrderSide.BUY,
            quantity=5.0,
            price=10.0,
            order_type=OrderType.LIMIT
        )
        success, trades = market.place_order(order)
        assert success is True
        assert len(trades) == 0  # No matching orders

    def test_order_matching(self, market):
        """Test order matching."""
        # Place sell order
        sell_order = Order(
            owner_id="seller",
            side=OrderSide.SELL,
            quantity=10.0,
            price=9.5,
            order_type=OrderType.LIMIT
        )
        market.place_order(sell_order)

        # Place buy order that matches
        buy_order = Order(
            owner_id="buyer",
            side=OrderSide.BUY,
            quantity=5.0,
            price=10.0,
            order_type=OrderType.LIMIT
        )
        success, trades = market.place_order(buy_order)
        assert success is True
        assert len(trades) == 1
        assert trades[0].quantity == 5.0
        assert trades[0].price == 9.5  # Seller's price


class TestOrderBook:
    """Tests for order book functionality."""

    def test_order_book_depth(self):
        """Test order book depth calculation."""
        book = OrderBook(market_id="test")

        # Add some orders
        for i in range(5):
            order = Order(
                owner_id=f"user{i}",
                side=OrderSide.BUY,
                quantity=10.0,
                price=100.0 - i
            )
            book.add_order(order)

        depth = book.depth(OrderSide.BUY)
        assert len(depth) == 5
        assert depth[0][0] == 100.0  # Highest bid first

    def test_order_book_spread(self):
        """Test bid-ask spread."""
        book = OrderBook(market_id="test")

        bid = Order(owner_id="buyer", side=OrderSide.BUY, quantity=10.0, price=99.0)
        ask = Order(owner_id="seller", side=OrderSide.SELL, quantity=10.0, price=101.0)

        book.add_order(bid)
        book.add_order(ask)

        assert book.best_bid() == 99.0
        assert book.best_ask() == 101.0
        assert book.spread() == 2.0


# =============================================================================
# Trade Tests
# =============================================================================

class TestTradeSystem:
    """Tests for trade system."""

    @pytest.fixture
    def trade_system(self):
        """Create a fresh trade system."""
        return TradeSystem()

    def test_propose_trade(self, trade_system):
        """Test proposing a trade."""
        offers = [TradeItem(item_type="resource", type_id="energy", quantity=10.0)]
        requests = [TradeItem(item_type="currency", type_id="EC", quantity=100.0)]

        success, offer = trade_system.propose_trade(
            proposer_id="user1",
            responder_id="user2",
            offers=offers,
            requests=requests
        )

        assert success is True
        assert offer is not None
        assert offer.status == TradeStatus.PROPOSED

    def test_respond_to_trade(self, trade_system):
        """Test responding to a trade."""
        offers = [TradeItem(item_type="resource", type_id="energy", quantity=10.0)]
        requests = [TradeItem(item_type="currency", type_id="EC", quantity=100.0)]

        success, offer = trade_system.propose_trade(
            proposer_id="user1",
            responder_id="user2",
            offers=offers,
            requests=requests
        )

        # Accept the trade
        result, msg = trade_system.respond_to_offer(offer.id, "user2", True)
        assert result is True
        assert offer.status == TradeStatus.ACCEPTED


class TestTradeNegotiator:
    """Tests for trade negotiation."""

    def test_evaluate_offer(self):
        """Test evaluating a trade offer."""
        negotiator = TradeNegotiator(entity_id="buyer")
        offer = TradeOffer(
            proposer_id="seller",
            responder_id="buyer",
            proposer_offers=[TradeItem(item_type="resource", type_id="energy", quantity=10.0)],
            proposer_requests=[TradeItem(item_type="currency", type_id="EC", quantity=5.0)]
        )

        valuations = {"energy": 1.0, "EC": 1.0}
        score, recommendation = negotiator.evaluate_offer(offer, valuations)

        # 10 energy for 5 EC = good deal
        assert score > 1.0
        assert recommendation in ("accept", "consider")


# =============================================================================
# Scarcity Tests
# =============================================================================

class TestScarcity:
    """Tests for scarcity modeling."""

    @pytest.fixture
    def scarcity_model(self):
        """Create a fresh scarcity model."""
        return ScarcityModel()

    def test_register_resource(self, scarcity_model):
        """Test registering a resource for tracking."""
        state = scarcity_model.register_resource(
            resource_type="energy",
            initial_supply=100.0,
            regeneration_rate=1.0
        )
        assert state is not None
        assert state.supply.total_supply == 100.0

    def test_scarcity_levels(self, scarcity_model):
        """Test scarcity level updates."""
        scarcity_model.register_resource("energy", initial_supply=100.0)

        # Add high demand
        scarcity_model.request_resource("energy", "user1", 200.0)

        state = scarcity_model.get_state("energy")
        state.update_level()

        assert state.level in (ScarcityLevel.SCARCE, ScarcityLevel.CRITICAL)


class TestResourceAllocator:
    """Tests for resource allocation."""

    def test_proportional_allocation(self):
        """Test proportional allocation."""
        allocator = ResourceAllocator(mode=CompetitionMode.PROPORTIONAL)

        requests = [
            {"id": "r1", "amount": 50.0},
            {"id": "r2", "amount": 50.0}
        ]

        allocations = allocator.allocate(60.0, requests)
        assert len(allocations) == 2
        assert allocations[0][1] == 30.0  # 50% of 60
        assert allocations[1][1] == 30.0

    def test_priority_allocation(self):
        """Test priority-based allocation."""
        allocator = ResourceAllocator(mode=CompetitionMode.PRIORITY)

        requests = [
            {"id": "r1", "amount": 50.0, "priority": 0.3},
            {"id": "r2", "amount": 50.0, "priority": 0.7}
        ]

        allocations = allocator.allocate(60.0, requests)
        # Higher priority should get full allocation first
        r2_alloc = next(a for a in allocations if a[0] == "r2")
        assert r2_alloc[1] == 50.0


# =============================================================================
# Wealth Distribution Tests
# =============================================================================

class TestGiniCalculator:
    """Tests for Gini coefficient calculation."""

    def test_perfect_equality(self):
        """Test Gini for perfect equality."""
        values = [100.0, 100.0, 100.0, 100.0]
        gini = GiniCalculator.gini_coefficient(values)
        assert gini == 0.0

    def test_high_inequality(self):
        """Test Gini for high inequality."""
        values = [1.0, 1.0, 1.0, 1000.0]
        gini = GiniCalculator.gini_coefficient(values)
        assert gini > 0.7

    def test_lorenz_curve(self):
        """Test Lorenz curve calculation."""
        values = [10.0, 20.0, 30.0, 40.0]
        pop, wealth = GiniCalculator.lorenz_curve(values)
        assert pop[0] == 0.0
        assert pop[-1] == 1.0
        assert wealth[-1] == 1.0


class TestWealthTracker:
    """Tests for wealth distribution tracking."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh wealth tracker."""
        return WealthDistributionTracker()

    def test_register_entity(self, tracker):
        """Test entity registration."""
        record = tracker.register_entity("user1", initial_wealth=100.0)
        assert record.total_wealth == 100.0
        assert record.entity_id == "user1"

    def test_wealth_classes(self, tracker):
        """Test wealth class assignment."""
        # Create a distribution
        for i in range(100):
            tracker.register_entity(f"user{i}", initial_wealth=i * 10.0)

        state = tracker.calculate_distribution()
        class_dist = tracker.get_class_distribution()

        # Should have entities in different classes
        total = sum(class_dist.values())
        assert total == 100


# =============================================================================
# Economic Agent Tests
# =============================================================================

class TestEconomicAgents:
    """Tests for economic agents."""

    def test_agent_creation(self):
        """Test creating an economic agent."""
        agent = EconomicAgent(name="test_agent", strategy=AgentStrategy.BALANCED)
        assert agent.strategy == AgentStrategy.BALANCED

    def test_agent_factory(self):
        """Test agent factory function."""
        producer = create_economic_agent("producer")
        assert isinstance(producer, ProducerAgent)

        consumer = create_economic_agent("consumer")
        assert isinstance(consumer, ConsumerAgent)

    def test_utility_function(self):
        """Test utility function evaluation."""
        utility = UtilityFunction(risk_aversion=0.5)

        # Low risk, good return
        u1 = utility.evaluate(expected_return=0.1, risk=0.1)
        # Same return, higher risk
        u2 = utility.evaluate(expected_return=0.1, risk=0.5)

        assert u1 > u2  # Lower risk should have higher utility

    def test_economic_goal(self):
        """Test economic goal tracking."""
        goal = EconomicGoal(
            goal_type="wealth",
            target_value=1000.0,
            current_value=500.0
        )
        assert goal.progress == 0.5
        assert goal.is_achieved is False

        goal.current_value = 1000.0
        assert goal.is_achieved is True

    def test_agent_tick(self):
        """Test agent tick processing."""
        agent = EconomicAgent(name="test_agent")
        agent.decision_interval = 1  # Make decisions every tick

        # Initialize with empty wallet and pool
        wallet = Wallet(owner_id="test")
        pool = ResourcePool(owner_id="test")
        agent.initialize(wallet, pool)

        # Tick should not raise
        agent.tick()
        assert agent._ticks_since_decision == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestEconomicIntegration:
    """Integration tests for the economics system."""

    def test_full_trade_cycle(self):
        """Test a complete trade cycle."""
        # Set up currency system
        currency_system = CurrencySystem()
        currency_system.mint_to("seller", "EC", 0)
        currency_system.mint_to("buyer", "EC", 1000.0)

        # Set up market
        market_manager = MarketManager()
        market = market_manager.create_market(
            name="Energy Market",
            base_resource="energy",
            initial_price=10.0
        )

        # Seller places sell order
        success, order, trades = market_manager.place_order(
            market_id=market.id,
            owner_id="seller",
            side=OrderSide.SELL,
            quantity=10.0,
            price=10.0
        )
        assert success is True

        # Buyer places buy order
        success, order, trades = market_manager.place_order(
            market_id=market.id,
            owner_id="buyer",
            side=OrderSide.BUY,
            quantity=10.0,
            price=10.0
        )
        assert success is True
        assert len(trades) == 1
        assert trades[0].quantity == 10.0

    def test_scarcity_affects_price(self):
        """Test that scarcity affects price multiplier."""
        model = ScarcityModel()
        model.register_resource("energy", initial_supply=10.0)

        # Low demand - should be abundant
        model.request_resource("energy", "user1", 1.0)
        state = model.get_state("energy")
        state.update_level()
        low_multiplier = state.price_multiplier

        # High demand - should be scarce
        model.request_resource("energy", "user2", 100.0)
        state.update_level()
        high_multiplier = state.price_multiplier

        assert high_multiplier > low_multiplier

    def test_wealth_distribution_updates(self):
        """Test wealth distribution tracking over time."""
        tracker = WealthDistributionTracker()

        # Initial equal distribution
        for i in range(10):
            tracker.register_entity(f"user{i}", initial_wealth=100.0)

        state1 = tracker.calculate_distribution()
        assert state1.gini < 0.1  # Very equal

        # Make some users wealthy
        tracker.update_wealth("user0", total_wealth=1000.0)
        tracker.update_wealth("user1", total_wealth=500.0)

        state2 = tracker.calculate_distribution()
        assert state2.gini > state1.gini  # More unequal


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for economics system."""

    def test_large_order_book(self):
        """Test order book with many orders."""
        book = OrderBook(market_id="test")

        # Add 1000 orders
        for i in range(1000):
            order = Order(
                owner_id=f"user{i}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                quantity=10.0,
                price=100.0 + (i % 10)
            )
            book.add_order(order)

        # Should handle quickly
        assert book.best_bid() is not None
        assert book.best_ask() is not None

    def test_many_wealth_records(self):
        """Test wealth tracking with many entities."""
        tracker = WealthDistributionTracker()

        # Add 1000 entities
        import random
        for i in range(1000):
            tracker.register_entity(f"user{i}", initial_wealth=random.uniform(1, 1000))

        # Calculate distribution should be fast
        state = tracker.calculate_distribution()
        assert state.population == 1000
        assert 0.0 <= state.gini <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
