"""
Economic Decision-Making Agents for NPCPU Economic System

Implements autonomous economic agents that make trading, investment,
and resource allocation decisions.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.abstractions import BaseSubsystem
from economics.resources import ResourcePool, ResourceRegistry, Resource, ResourceQuality
from economics.currency import Wallet, CurrencySystem, get_currency_system
from economics.markets import OrderSide, OrderType, MarketManager, get_market_manager


class AgentStrategy(Enum):
    """Economic strategy types."""
    CONSERVATIVE = "conservative"   # Risk-averse, slow growth
    BALANCED = "balanced"           # Moderate risk/reward
    AGGRESSIVE = "aggressive"       # High risk, high reward
    SPECULATOR = "speculator"       # Short-term trading
    ACCUMULATOR = "accumulator"     # Long-term holding
    ARBITRAGEUR = "arbitrageur"     # Exploit price differences


class DecisionType(Enum):
    """Types of economic decisions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    TRADE = "trade"
    INVEST = "invest"
    DIVEST = "divest"
    PRODUCE = "produce"
    CONSUME = "consume"


@dataclass
class EconomicGoal:
    """
    An economic goal for an agent.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_type: str = ""         # "wealth", "resource", "stability"
    target_value: float = 0.0
    current_value: float = 0.0
    priority: float = 0.5
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Progress toward goal (0-1)."""
        if self.target_value == 0:
            return 1.0
        return min(1.0, self.current_value / self.target_value)

    @property
    def is_achieved(self) -> bool:
        """Check if goal is achieved."""
        return self.current_value >= self.target_value


@dataclass
class EconomicDecision:
    """
    A decision made by an economic agent.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType = DecisionType.HOLD
    resource_type: Optional[str] = None
    currency_id: Optional[str] = None
    quantity: float = 0.0
    price: Optional[float] = None
    rationale: str = ""
    confidence: float = 0.5
    expected_utility: float = 0.0
    timestamp: float = field(default_factory=time.time)
    executed: bool = False
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision to dictionary."""
        return {
            "id": self.id,
            "decision_type": self.decision_type.value,
            "resource_type": self.resource_type,
            "currency_id": self.currency_id,
            "quantity": self.quantity,
            "price": self.price,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "expected_utility": self.expected_utility,
            "timestamp": self.timestamp,
            "executed": self.executed
        }


class UtilityFunction:
    """
    Utility function for evaluating economic outcomes.
    """

    def __init__(self, risk_aversion: float = 0.5):
        self.risk_aversion = risk_aversion  # 0 = risk-loving, 1 = risk-averse

    def evaluate(
        self,
        expected_return: float,
        risk: float,
        time_horizon: float = 1.0
    ) -> float:
        """
        Evaluate utility of an economic action.

        Uses a risk-adjusted utility function.
        """
        # Risk penalty increases with risk aversion
        risk_penalty = risk * self.risk_aversion * 2

        # Time discount (future returns worth less)
        time_discount = 1.0 / (1 + 0.05 * time_horizon)

        # Utility = expected return - risk penalty, discounted by time
        utility = (expected_return - risk_penalty) * time_discount

        return utility

    def compare_options(
        self,
        options: List[Dict[str, float]]
    ) -> int:
        """
        Compare multiple options and return index of best.

        Options should have 'return', 'risk', 'time' keys.
        """
        if not options:
            return -1

        utilities = [
            self.evaluate(
                opt.get('return', 0),
                opt.get('risk', 0),
                opt.get('time', 1)
            )
            for opt in options
        ]

        return int(np.argmax(utilities))


class MarketAnalyzer:
    """
    Analyzes market conditions for decision making.
    """

    def __init__(self, market_manager: MarketManager = None):
        self.market_manager = market_manager or get_market_manager()

    def analyze_market(self, market_id: str) -> Dict[str, Any]:
        """Analyze a market's current state."""
        market = self.market_manager.get_market(market_id)
        if not market:
            return {}

        analysis = {
            "market_id": market_id,
            "last_price": market.last_price,
            "spread": market.order_book.spread(),
            "mid_price": market.order_book.mid_price(),
            "bid_depth": len([o for o in market.order_book.bids if o.is_active]),
            "ask_depth": len([o for o in market.order_book.asks if o.is_active])
        }

        # Price trends from history
        history = market.price_history
        analysis["volatility"] = history.volatility() or 0.0
        analysis["ma_20"] = history.moving_average(20)
        analysis["price_change"] = history.price_change_pct(10) or 0.0

        # Trend direction
        ma_20 = analysis["ma_20"]
        if ma_20:
            if market.last_price > ma_20 * 1.02:
                analysis["trend"] = "bullish"
            elif market.last_price < ma_20 * 0.98:
                analysis["trend"] = "bearish"
            else:
                analysis["trend"] = "neutral"
        else:
            analysis["trend"] = "unknown"

        return analysis

    def find_opportunities(self) -> List[Dict[str, Any]]:
        """Find trading opportunities across markets."""
        opportunities = []

        for market in self.market_manager.list_markets():
            analysis = self.analyze_market(market.id)
            if not analysis:
                continue

            # Look for undervalued markets (price below MA)
            ma = analysis.get("ma_20")
            if ma and market.last_price < ma * 0.95:
                opportunities.append({
                    "type": "undervalued",
                    "market_id": market.id,
                    "current_price": market.last_price,
                    "fair_value": ma,
                    "potential_return": (ma - market.last_price) / market.last_price
                })

            # Look for low volatility (stable) opportunities
            vol = analysis.get("volatility", 0)
            if vol < 0.02:
                opportunities.append({
                    "type": "stable",
                    "market_id": market.id,
                    "volatility": vol,
                    "potential_return": 0.01  # Small but reliable
                })

        return opportunities


class EconomicAgent(BaseSubsystem):
    """
    Base class for economic decision-making agents.

    Implements autonomous economic behavior within organisms.
    """

    def __init__(
        self,
        name: str = "economic_agent",
        owner=None,
        strategy: AgentStrategy = AgentStrategy.BALANCED
    ):
        super().__init__(name, owner)
        self.agent_id = str(uuid.uuid4())
        self.strategy = strategy

        # Economic state
        self.wallet: Wallet = None
        self.resource_pool: ResourcePool = None
        self.goals: List[EconomicGoal] = []
        self.decision_history: List[EconomicDecision] = []

        # Configuration
        self.utility_function = UtilityFunction(
            risk_aversion=self._strategy_to_risk_aversion()
        )
        self.market_analyzer = MarketAnalyzer()

        # Decision parameters
        self.decision_interval = 10  # Ticks between decisions
        self._ticks_since_decision = 0
        self._pending_decisions: List[EconomicDecision] = []

    def _strategy_to_risk_aversion(self) -> float:
        """Convert strategy to risk aversion parameter."""
        mapping = {
            AgentStrategy.CONSERVATIVE: 0.8,
            AgentStrategy.BALANCED: 0.5,
            AgentStrategy.AGGRESSIVE: 0.2,
            AgentStrategy.SPECULATOR: 0.1,
            AgentStrategy.ACCUMULATOR: 0.6,
            AgentStrategy.ARBITRAGEUR: 0.3
        }
        return mapping.get(self.strategy, 0.5)

    def initialize(
        self,
        wallet: Wallet,
        resource_pool: ResourcePool,
        initial_goals: List[EconomicGoal] = None
    ) -> None:
        """Initialize the agent with economic resources."""
        self.wallet = wallet
        self.resource_pool = resource_pool
        if initial_goals:
            self.goals = initial_goals

    def add_goal(self, goal: EconomicGoal) -> None:
        """Add an economic goal."""
        self.goals.append(goal)

    def evaluate_state(self) -> Dict[str, Any]:
        """Evaluate current economic state."""
        state = {
            "total_wealth": 0.0,
            "liquid_wealth": 0.0,
            "resource_value": 0.0,
            "goal_progress": {}
        }

        if self.wallet:
            exchange_rates = get_currency_system().get_exchange_rates()
            state["liquid_wealth"] = self.wallet.get_total_value(exchange_rates)

        if self.resource_pool:
            state["resource_value"] = self.resource_pool.get_total_value()

        state["total_wealth"] = state["liquid_wealth"] + state["resource_value"]

        # Update goal progress
        for goal in self.goals:
            if goal.goal_type == "wealth":
                goal.current_value = state["total_wealth"]
            state["goal_progress"][goal.id] = goal.progress

        return state

    def generate_decisions(self) -> List[EconomicDecision]:
        """Generate economic decisions based on current state."""
        decisions = []
        state = self.evaluate_state()

        # Check opportunities
        opportunities = self.market_analyzer.find_opportunities()

        for opp in opportunities:
            decision = self._evaluate_opportunity(opp, state)
            if decision:
                decisions.append(decision)

        # Check goals and generate goal-driven decisions
        for goal in self.goals:
            if not goal.is_achieved:
                goal_decision = self._generate_goal_decision(goal, state)
                if goal_decision:
                    decisions.append(goal_decision)

        # Filter by confidence threshold
        confident_decisions = [d for d in decisions if d.confidence > 0.3]

        # Sort by expected utility
        confident_decisions.sort(key=lambda d: d.expected_utility, reverse=True)

        return confident_decisions[:3]  # Limit to top 3 decisions

    def _evaluate_opportunity(
        self,
        opportunity: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Optional[EconomicDecision]:
        """Evaluate a market opportunity."""
        opp_type = opportunity.get("type")
        potential_return = opportunity.get("potential_return", 0)

        if opp_type == "undervalued" and potential_return > 0.05:
            # Consider buying
            market_id = opportunity["market_id"]
            available_funds = state["liquid_wealth"] * 0.1  # Use 10% of funds

            if available_funds > 0:
                utility = self.utility_function.evaluate(
                    potential_return,
                    0.2,  # Assumed risk
                    5.0   # Time horizon
                )

                return EconomicDecision(
                    decision_type=DecisionType.BUY,
                    resource_type=market_id,
                    quantity=available_funds,
                    price=opportunity["current_price"],
                    rationale=f"Undervalued market, {potential_return*100:.1f}% potential",
                    confidence=min(0.8, potential_return * 5),
                    expected_utility=utility
                )

        return None

    def _generate_goal_decision(
        self,
        goal: EconomicGoal,
        state: Dict[str, Any]
    ) -> Optional[EconomicDecision]:
        """Generate a decision to progress toward a goal."""
        if goal.goal_type == "wealth":
            gap = goal.target_value - goal.current_value
            if gap > 0:
                # Need to accumulate wealth
                return EconomicDecision(
                    decision_type=DecisionType.INVEST,
                    quantity=gap * 0.1,  # Incremental progress
                    rationale=f"Working toward wealth goal: {gap:.2f} remaining",
                    confidence=0.6,
                    expected_utility=gap * goal.priority
                )

        return None

    def execute_decision(self, decision: EconomicDecision) -> bool:
        """Execute an economic decision."""
        if decision.executed:
            return False

        success = False
        result = {}

        if decision.decision_type == DecisionType.BUY:
            success, result = self._execute_buy(decision)
        elif decision.decision_type == DecisionType.SELL:
            success, result = self._execute_sell(decision)
        elif decision.decision_type == DecisionType.HOLD:
            success = True
            result = {"action": "hold"}

        decision.executed = True
        decision.result = result

        self.decision_history.append(decision)

        # Keep last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        return success

    def _execute_buy(self, decision: EconomicDecision) -> Tuple[bool, Dict[str, Any]]:
        """Execute a buy decision."""
        if not self.wallet or not decision.resource_type:
            return False, {"error": "No wallet or resource type"}

        market_manager = get_market_manager()
        market = market_manager.get_market_by_resource(decision.resource_type)

        if not market:
            return False, {"error": "Market not found"}

        # Place buy order
        success, order, trades = market_manager.place_order(
            market_id=market.id,
            owner_id=self.agent_id,
            side=OrderSide.BUY,
            quantity=decision.quantity / (decision.price or market.last_price),
            price=decision.price,
            order_type=OrderType.LIMIT
        )

        return success, {
            "order_id": order.id if order else None,
            "trades": len(trades)
        }

    def _execute_sell(self, decision: EconomicDecision) -> Tuple[bool, Dict[str, Any]]:
        """Execute a sell decision."""
        if not self.resource_pool or not decision.resource_type:
            return False, {"error": "No resource pool or resource type"}

        # Check if we have the resources
        available = self.resource_pool.get_total_by_type(decision.resource_type)
        if available < decision.quantity:
            return False, {"error": "Insufficient resources"}

        market_manager = get_market_manager()
        market = market_manager.get_market_by_resource(decision.resource_type)

        if not market:
            return False, {"error": "Market not found"}

        # Place sell order
        success, order, trades = market_manager.place_order(
            market_id=market.id,
            owner_id=self.agent_id,
            side=OrderSide.SELL,
            quantity=decision.quantity,
            price=decision.price,
            order_type=OrderType.LIMIT
        )

        return success, {
            "order_id": order.id if order else None,
            "trades": len(trades)
        }

    def tick(self) -> None:
        """Process one time step."""
        if not self.enabled:
            return

        self._ticks_since_decision += 1

        # Make decisions periodically
        if self._ticks_since_decision >= self.decision_interval:
            self._ticks_since_decision = 0

            # Generate and execute decisions
            decisions = self.generate_decisions()
            for decision in decisions:
                if decision.confidence > 0.5:
                    self.execute_decision(decision)

    def get_state(self) -> Dict[str, Any]:
        """Get agent state."""
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy.value,
            "economic_state": self.evaluate_state(),
            "goal_count": len(self.goals),
            "decision_count": len(self.decision_history),
            "pending_decisions": len(self._pending_decisions)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state."""
        if "strategy" in state:
            self.strategy = AgentStrategy(state["strategy"])
            self.utility_function.risk_aversion = self._strategy_to_risk_aversion()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        if not self.decision_history:
            return {"decisions": 0, "success_rate": 0.0}

        executed = [d for d in self.decision_history if d.executed]
        successful = [d for d in executed if d.result and not d.result.get("error")]

        return {
            "decisions": len(self.decision_history),
            "executed": len(executed),
            "successful": len(successful),
            "success_rate": len(successful) / max(1, len(executed)),
            "avg_confidence": np.mean([d.confidence for d in executed]) if executed else 0.0,
            "goals_achieved": sum(1 for g in self.goals if g.is_achieved),
            "goals_total": len(self.goals)
        }


class ProducerAgent(EconomicAgent):
    """
    An agent that produces resources.
    """

    def __init__(self, name: str = "producer_agent", owner=None):
        super().__init__(name, owner, AgentStrategy.ACCUMULATOR)
        self.production_rates: Dict[str, float] = {}
        self.production_costs: Dict[str, float] = {}

    def set_production(
        self,
        resource_type: str,
        rate: float,
        cost: float
    ) -> None:
        """Configure production for a resource type."""
        self.production_rates[resource_type] = rate
        self.production_costs[resource_type] = cost

    def produce(self) -> Dict[str, float]:
        """Produce resources based on configuration."""
        produced = {}

        for resource_type, rate in self.production_rates.items():
            cost = self.production_costs.get(resource_type, 0)

            # Check if we can afford production
            if self.wallet and self.wallet.can_afford("EC", cost):
                self.wallet.debit("EC", cost)
                produced[resource_type] = rate

        return produced

    def tick(self) -> None:
        """Process one time step."""
        super().tick()

        # Produce resources
        if self._ticks_since_decision == 0:
            self.produce()


class ConsumerAgent(EconomicAgent):
    """
    An agent that consumes resources.
    """

    def __init__(self, name: str = "consumer_agent", owner=None):
        super().__init__(name, owner, AgentStrategy.BALANCED)
        self.consumption_needs: Dict[str, float] = {}
        self.satisfaction_level: float = 0.5

    def set_need(self, resource_type: str, amount: float) -> None:
        """Set consumption need for a resource type."""
        self.consumption_needs[resource_type] = amount

    def consume(self) -> Dict[str, float]:
        """Consume resources based on needs."""
        consumed = {}
        total_satisfaction = 0.0

        for resource_type, need in self.consumption_needs.items():
            if self.resource_pool:
                available = self.resource_pool.get_total_by_type(resource_type)
                consume_amount = min(need, available)

                if consume_amount > 0:
                    resource = self.resource_pool.withdraw(resource_type, consume_amount)
                    if resource:
                        consumed[resource_type] = consume_amount
                        total_satisfaction += consume_amount / need

        # Update satisfaction level
        if self.consumption_needs:
            self.satisfaction_level = total_satisfaction / len(self.consumption_needs)

        return consumed

    def tick(self) -> None:
        """Process one time step."""
        super().tick()

        # Consume resources periodically
        if self._ticks_since_decision == 0:
            self.consume()


class TraderAgent(EconomicAgent):
    """
    An agent specialized in trading.
    """

    def __init__(self, name: str = "trader_agent", owner=None):
        super().__init__(name, owner, AgentStrategy.SPECULATOR)
        self.trading_pairs: List[Tuple[str, str]] = []
        self.trade_count: int = 0
        self.profit_loss: float = 0.0

    def add_trading_pair(self, resource_a: str, resource_b: str) -> None:
        """Add a trading pair to monitor."""
        self.trading_pairs.append((resource_a, resource_b))

    def find_arbitrage(self) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities."""
        opportunities = []
        market_manager = get_market_manager()

        for pair in self.trading_pairs:
            market_a = market_manager.get_market_by_resource(pair[0])
            market_b = market_manager.get_market_by_resource(pair[1])

            if market_a and market_b:
                # Check for price discrepancies
                price_ratio = market_a.last_price / max(0.001, market_b.last_price)
                if price_ratio > 1.05 or price_ratio < 0.95:
                    opportunities.append({
                        "type": "arbitrage",
                        "pair": pair,
                        "ratio": price_ratio,
                        "potential": abs(1 - price_ratio)
                    })

        return opportunities


# Default agent factory
def create_economic_agent(
    agent_type: str,
    name: str = None,
    owner=None,
    **kwargs
) -> EconomicAgent:
    """Factory function to create economic agents."""
    agent_types = {
        "producer": ProducerAgent,
        "consumer": ConsumerAgent,
        "trader": TraderAgent,
        "default": EconomicAgent
    }

    agent_class = agent_types.get(agent_type, EconomicAgent)
    return agent_class(name=name or f"{agent_type}_agent", owner=owner, **kwargs)
