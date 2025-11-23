"""
NPCPU Economics System

Resource economics, trade systems, market dynamics, and wealth distribution
modeling for digital organisms.

This module provides:
- Resource types and management (resources.py)
- Currency and value storage (currency.py)
- Market dynamics and price discovery (markets.py)
- Inter-organism trade protocols (trade.py)
- Scarcity modeling and competition (scarcity.py)
- Wealth distribution and inequality metrics (wealth_distribution.py)
- Economic decision-making agents (economic_agents.py)
"""

from .resources import (
    Resource,
    ResourceType,
    ResourceCategory,
    ResourceState,
    ResourceQuality,
    ResourcePool,
    ResourceRegistry,
    get_resource_registry
)

from .currency import (
    Currency,
    CurrencyType,
    Transaction,
    TransactionStatus,
    Balance,
    Wallet,
    CurrencySystem,
    get_currency_system
)

from .markets import (
    Market,
    MarketState,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Trade,
    OrderBook,
    PriceHistory,
    MarketManager,
    get_market_manager
)

from .trade import (
    TradeOffer,
    TradeItem,
    TradeStatus,
    TradeType,
    TradeContract,
    TradeHistory,
    TradeNegotiator,
    TradeSystem,
    get_trade_system
)

from .scarcity import (
    ScarcityLevel,
    CompetitionMode,
    AllocationStrategy,
    ResourceSupply,
    ResourceDemand,
    ScarcityState,
    ResourceAllocator,
    ScarcityModel,
    get_scarcity_model
)

from .wealth_distribution import (
    WealthClass,
    DistributionType,
    WealthRecord,
    WealthDistributionState,
    GiniCalculator,
    WealthDistributionTracker,
    get_wealth_tracker
)

from .economic_agents import (
    AgentStrategy,
    DecisionType,
    EconomicGoal,
    EconomicDecision,
    UtilityFunction,
    MarketAnalyzer,
    EconomicAgent,
    ProducerAgent,
    ConsumerAgent,
    TraderAgent,
    create_economic_agent
)

from .tertiary_integration import (
    EconomicDomainMetrics,
    EconomicRefinementAgent,
    EconomicSubsystemBridge,
    get_economic_bridge
)

__all__ = [
    # Resources
    "Resource",
    "ResourceType",
    "ResourceCategory",
    "ResourceState",
    "ResourceQuality",
    "ResourcePool",
    "ResourceRegistry",
    "get_resource_registry",

    # Currency
    "Currency",
    "CurrencyType",
    "Transaction",
    "TransactionStatus",
    "Balance",
    "Wallet",
    "CurrencySystem",
    "get_currency_system",

    # Markets
    "Market",
    "MarketState",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "Trade",
    "OrderBook",
    "PriceHistory",
    "MarketManager",
    "get_market_manager",

    # Trade
    "TradeOffer",
    "TradeItem",
    "TradeStatus",
    "TradeType",
    "TradeContract",
    "TradeHistory",
    "TradeNegotiator",
    "TradeSystem",
    "get_trade_system",

    # Scarcity
    "ScarcityLevel",
    "CompetitionMode",
    "AllocationStrategy",
    "ResourceSupply",
    "ResourceDemand",
    "ScarcityState",
    "ResourceAllocator",
    "ScarcityModel",
    "get_scarcity_model",

    # Wealth Distribution
    "WealthClass",
    "DistributionType",
    "WealthRecord",
    "WealthDistributionState",
    "GiniCalculator",
    "WealthDistributionTracker",
    "get_wealth_tracker",

    # Economic Agents
    "AgentStrategy",
    "DecisionType",
    "EconomicGoal",
    "EconomicDecision",
    "UtilityFunction",
    "MarketAnalyzer",
    "EconomicAgent",
    "ProducerAgent",
    "ConsumerAgent",
    "TraderAgent",
    "create_economic_agent",

    # Tertiary ReBo Integration
    "EconomicDomainMetrics",
    "EconomicRefinementAgent",
    "EconomicSubsystemBridge",
    "get_economic_bridge"
]
