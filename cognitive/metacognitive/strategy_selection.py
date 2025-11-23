"""
Strategy Selection System for ORACLE-Z Metacognitive Agent

Implements cognitive strategy optimization that dynamically selects
the best cognitive approaches based on task requirements, current
state, and historical performance.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import json
from abc import ABC, abstractmethod


class StrategyType(Enum):
    """Types of cognitive strategies"""
    ANALYTICAL = "analytical"           # Systematic, step-by-step
    HEURISTIC = "heuristic"            # Rule-of-thumb, fast
    INTUITIVE = "intuitive"            # Pattern-based, holistic
    DELIBERATIVE = "deliberative"       # Careful consideration
    REACTIVE = "reactive"              # Fast response
    EXPLORATORY = "exploratory"        # Search and discover
    EXPLOITATIVE = "exploitative"      # Use known approaches
    HYBRID = "hybrid"                  # Combination strategies


class TaskDomain(Enum):
    """Domains of cognitive tasks"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    LEARNING = "learning"
    PLANNING = "planning"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    CREATIVITY = "creativity"
    MEMORY = "memory"


@dataclass
class StrategyContext:
    """Context for strategy selection"""
    task_domain: TaskDomain
    complexity: float  # 0.0 to 1.0
    time_pressure: float  # 0.0 to 1.0
    accuracy_requirement: float  # 0.0 to 1.0
    resource_availability: float  # 0.0 to 1.0
    uncertainty: float  # 0.0 to 1.0
    prior_experience: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Historical performance of a strategy"""
    strategy_type: StrategyType
    success_rate: float = 0.5
    average_time: float = 1.0
    resource_usage: float = 0.5
    accuracy: float = 0.5
    sample_count: int = 0
    contexts: List[StrategyContext] = field(default_factory=list)

    def update(self, success: bool, time_taken: float,
               resources_used: float, accuracy: float,
               context: StrategyContext):
        """Update performance metrics with new observation"""
        self.sample_count += 1
        alpha = 2 / (self.sample_count + 1)  # Exponential moving average

        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(success)
        self.average_time = (1 - alpha) * self.average_time + alpha * time_taken
        self.resource_usage = (1 - alpha) * self.resource_usage + alpha * resources_used
        self.accuracy = (1 - alpha) * self.accuracy + alpha * accuracy

        self.contexts.append(context)
        if len(self.contexts) > 100:
            self.contexts = self.contexts[-100:]


@dataclass
class CognitiveStrategy:
    """A cognitive strategy that can be selected and executed"""
    name: str
    strategy_type: StrategyType
    suitable_domains: List[TaskDomain]
    min_resources: float = 0.1
    max_complexity: float = 1.0
    time_efficiency: float = 0.5  # Higher = faster
    accuracy_potential: float = 0.5  # Higher = more accurate
    adaptability: float = 0.5  # How well it handles unexpected situations

    # Optional customization
    parameters: Dict[str, Any] = field(default_factory=dict)

    def evaluate_suitability(self, context: StrategyContext) -> float:
        """Evaluate how suitable this strategy is for a given context"""
        score = 0.0

        # Domain match
        if context.task_domain in self.suitable_domains:
            score += 0.3
        else:
            score -= 0.2

        # Complexity compatibility
        if context.complexity <= self.max_complexity:
            score += 0.2 * (1 - context.complexity / self.max_complexity)
        else:
            score -= 0.3

        # Time pressure handling
        if context.time_pressure > 0.7:
            score += 0.2 * self.time_efficiency
        else:
            score += 0.1 * (1 - context.time_pressure) * self.accuracy_potential

        # Accuracy requirement
        if context.accuracy_requirement > 0.7:
            score += 0.2 * self.accuracy_potential
        else:
            score += 0.1 * self.time_efficiency

        # Resource availability
        if context.resource_availability < self.min_resources:
            score -= 0.4
        else:
            score += 0.1 * (context.resource_availability - self.min_resources)

        # Uncertainty handling
        score += 0.1 * self.adaptability * context.uncertainty

        return np.clip(score, 0.0, 1.0)


class StrategyOptimizer:
    """
    Core strategy selection and optimization system for ORACLE-Z.

    Dynamically selects optimal cognitive strategies based on:
    - Task context and requirements
    - Historical performance data
    - Current resource availability
    - Uncertainty and risk tolerance
    """

    def __init__(self):
        # Available strategies
        self.strategies: Dict[str, CognitiveStrategy] = {}
        self._initialize_default_strategies()

        # Performance tracking
        self.performance_history: Dict[str, StrategyPerformance] = {}

        # Selection parameters
        self.exploration_rate: float = 0.1  # Probability of trying new strategies
        self.learning_rate: float = 0.1

        # Context-strategy mapping (learned)
        self.context_preferences: Dict[str, Dict[str, float]] = {}

        # Active strategy tracking
        self.current_strategy: Optional[CognitiveStrategy] = None
        self.strategy_stack: List[CognitiveStrategy] = []  # For nested strategies

    def _initialize_default_strategies(self):
        """Initialize the default set of cognitive strategies"""
        default_strategies = [
            CognitiveStrategy(
                name="systematic_analysis",
                strategy_type=StrategyType.ANALYTICAL,
                suitable_domains=[TaskDomain.REASONING, TaskDomain.PROBLEM_SOLVING],
                min_resources=0.3,
                max_complexity=0.9,
                time_efficiency=0.3,
                accuracy_potential=0.9,
                adaptability=0.4
            ),
            CognitiveStrategy(
                name="quick_heuristic",
                strategy_type=StrategyType.HEURISTIC,
                suitable_domains=[TaskDomain.DECISION_MAKING, TaskDomain.PERCEPTION],
                min_resources=0.1,
                max_complexity=0.6,
                time_efficiency=0.9,
                accuracy_potential=0.5,
                adaptability=0.6
            ),
            CognitiveStrategy(
                name="pattern_matching",
                strategy_type=StrategyType.INTUITIVE,
                suitable_domains=[TaskDomain.PERCEPTION, TaskDomain.MEMORY],
                min_resources=0.2,
                max_complexity=0.7,
                time_efficiency=0.7,
                accuracy_potential=0.7,
                adaptability=0.5
            ),
            CognitiveStrategy(
                name="careful_deliberation",
                strategy_type=StrategyType.DELIBERATIVE,
                suitable_domains=[TaskDomain.PLANNING, TaskDomain.DECISION_MAKING],
                min_resources=0.5,
                max_complexity=1.0,
                time_efficiency=0.2,
                accuracy_potential=0.95,
                adaptability=0.3
            ),
            CognitiveStrategy(
                name="fast_reaction",
                strategy_type=StrategyType.REACTIVE,
                suitable_domains=[TaskDomain.PERCEPTION],
                min_resources=0.05,
                max_complexity=0.3,
                time_efficiency=1.0,
                accuracy_potential=0.4,
                adaptability=0.8
            ),
            CognitiveStrategy(
                name="exploration_search",
                strategy_type=StrategyType.EXPLORATORY,
                suitable_domains=[TaskDomain.CREATIVITY, TaskDomain.PROBLEM_SOLVING],
                min_resources=0.4,
                max_complexity=0.8,
                time_efficiency=0.4,
                accuracy_potential=0.6,
                adaptability=0.9
            ),
            CognitiveStrategy(
                name="known_solution",
                strategy_type=StrategyType.EXPLOITATIVE,
                suitable_domains=[TaskDomain.PROBLEM_SOLVING, TaskDomain.REASONING],
                min_resources=0.2,
                max_complexity=0.5,
                time_efficiency=0.8,
                accuracy_potential=0.8,
                adaptability=0.2
            ),
            CognitiveStrategy(
                name="adaptive_hybrid",
                strategy_type=StrategyType.HYBRID,
                suitable_domains=list(TaskDomain),
                min_resources=0.3,
                max_complexity=0.85,
                time_efficiency=0.6,
                accuracy_potential=0.75,
                adaptability=0.85
            ),
            CognitiveStrategy(
                name="deep_learning",
                strategy_type=StrategyType.ANALYTICAL,
                suitable_domains=[TaskDomain.LEARNING, TaskDomain.MEMORY],
                min_resources=0.4,
                max_complexity=0.9,
                time_efficiency=0.3,
                accuracy_potential=0.85,
                adaptability=0.5
            ),
            CognitiveStrategy(
                name="creative_synthesis",
                strategy_type=StrategyType.INTUITIVE,
                suitable_domains=[TaskDomain.CREATIVITY, TaskDomain.PROBLEM_SOLVING],
                min_resources=0.3,
                max_complexity=0.8,
                time_efficiency=0.5,
                accuracy_potential=0.6,
                adaptability=0.7
            )
        ]

        for strategy in default_strategies:
            self.register_strategy(strategy)

    def register_strategy(self, strategy: CognitiveStrategy):
        """Register a new cognitive strategy"""
        self.strategies[strategy.name] = strategy
        if strategy.name not in self.performance_history:
            self.performance_history[strategy.name] = StrategyPerformance(
                strategy_type=strategy.strategy_type
            )

    def select_strategy(self, context: StrategyContext) -> CognitiveStrategy:
        """Select the optimal strategy for a given context"""
        # Calculate suitability scores for all strategies
        scores: Dict[str, float] = {}

        for name, strategy in self.strategies.items():
            # Base suitability
            suitability = strategy.evaluate_suitability(context)

            # Historical performance adjustment
            if name in self.performance_history:
                perf = self.performance_history[name]
                if perf.sample_count > 5:
                    historical_score = (
                        perf.success_rate * 0.4 +
                        perf.accuracy * 0.3 +
                        (1 - perf.average_time) * 0.2 +
                        (1 - perf.resource_usage) * 0.1
                    )
                    suitability = suitability * 0.6 + historical_score * 0.4

            # Context preference adjustment
            context_key = self._get_context_key(context)
            if context_key in self.context_preferences:
                prefs = self.context_preferences[context_key]
                if name in prefs:
                    suitability *= (1 + prefs[name] * 0.2)

            scores[name] = suitability

        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: randomly weight less-tried strategies
            for name in scores:
                if self.performance_history[name].sample_count < 10:
                    scores[name] *= 1.5

        # Select best strategy
        best_strategy_name = max(scores, key=scores.get)
        selected = self.strategies[best_strategy_name]

        self.current_strategy = selected
        return selected

    def _get_context_key(self, context: StrategyContext) -> str:
        """Generate a key for context-based preference learning"""
        return f"{context.task_domain.value}_{int(context.complexity * 3)}_{int(context.time_pressure * 3)}"

    def report_outcome(self,
                      strategy_name: str,
                      context: StrategyContext,
                      success: bool,
                      time_taken: float,
                      resources_used: float,
                      accuracy: float):
        """Report the outcome of using a strategy"""
        if strategy_name not in self.performance_history:
            return

        # Update performance history
        self.performance_history[strategy_name].update(
            success=success,
            time_taken=time_taken,
            resources_used=resources_used,
            accuracy=accuracy,
            context=context
        )

        # Update context preferences
        context_key = self._get_context_key(context)
        if context_key not in self.context_preferences:
            self.context_preferences[context_key] = {}

        prefs = self.context_preferences[context_key]

        # Positive outcome increases preference
        outcome_score = (
            float(success) * 0.5 +
            accuracy * 0.3 +
            (1 - resources_used) * 0.2
        )

        if strategy_name not in prefs:
            prefs[strategy_name] = 0.0

        prefs[strategy_name] += self.learning_rate * (outcome_score - 0.5)
        prefs[strategy_name] = np.clip(prefs[strategy_name], -1.0, 1.0)

    def get_strategy_recommendations(self,
                                     context: StrategyContext,
                                     top_n: int = 3) -> List[Tuple[CognitiveStrategy, float]]:
        """Get top N strategy recommendations with scores"""
        scores = []

        for name, strategy in self.strategies.items():
            suitability = strategy.evaluate_suitability(context)

            if name in self.performance_history:
                perf = self.performance_history[name]
                if perf.sample_count > 0:
                    confidence = min(perf.sample_count / 20, 1.0)
                    historical = perf.success_rate * perf.accuracy
                    suitability = suitability * (1 - confidence * 0.3) + historical * confidence * 0.3

            scores.append((strategy, suitability))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def push_strategy(self, strategy: CognitiveStrategy):
        """Push current strategy onto stack (for nested execution)"""
        if self.current_strategy:
            self.strategy_stack.append(self.current_strategy)
        self.current_strategy = strategy

    def pop_strategy(self) -> Optional[CognitiveStrategy]:
        """Pop strategy from stack"""
        if self.strategy_stack:
            self.current_strategy = self.strategy_stack.pop()
            return self.current_strategy
        return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all strategy performances"""
        return {
            name: {
                "type": perf.strategy_type.value,
                "success_rate": perf.success_rate,
                "accuracy": perf.accuracy,
                "average_time": perf.average_time,
                "resource_usage": perf.resource_usage,
                "sample_count": perf.sample_count
            }
            for name, perf in self.performance_history.items()
        }

    def get_best_strategy_for_domain(self, domain: TaskDomain) -> Optional[CognitiveStrategy]:
        """Get the best performing strategy for a specific domain"""
        domain_strategies = [
            (name, strategy, self.performance_history.get(name))
            for name, strategy in self.strategies.items()
            if domain in strategy.suitable_domains
        ]

        if not domain_strategies:
            return None

        # Score by historical performance
        scored = []
        for name, strategy, perf in domain_strategies:
            if perf and perf.sample_count > 5:
                score = perf.success_rate * perf.accuracy
            else:
                score = strategy.accuracy_potential * 0.5
            scored.append((strategy, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def adapt_exploration_rate(self):
        """Adapt exploration rate based on overall performance"""
        if not self.performance_history:
            return

        avg_success = np.mean([
            p.success_rate for p in self.performance_history.values()
            if p.sample_count > 0
        ])

        # High success = less exploration needed
        if avg_success > 0.8:
            self.exploration_rate = max(0.02, self.exploration_rate * 0.9)
        elif avg_success < 0.5:
            # Poor success = more exploration
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)

    def export_state(self, filepath: str):
        """Export optimizer state to file"""
        state = {
            "strategies": {
                name: {
                    "type": s.strategy_type.value,
                    "domains": [d.value for d in s.suitable_domains],
                    "parameters": s.parameters
                }
                for name, s in self.strategies.items()
            },
            "performance": self.get_performance_summary(),
            "context_preferences": self.context_preferences,
            "exploration_rate": self.exploration_rate
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
