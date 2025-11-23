"""
Wealth Distribution and Inequality Metrics for NPCPU Economic System

Implements Gini coefficient, wealth tracking, and inequality analysis.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class WealthClass(Enum):
    """Wealth classification tiers."""
    DESTITUTE = "destitute"     # Bottom 10%
    POOR = "poor"               # 10-25%
    LOWER_MIDDLE = "lower_middle"  # 25-50%
    MIDDLE = "middle"           # 50-75%
    UPPER_MIDDLE = "upper_middle"  # 75-90%
    WEALTHY = "wealthy"         # 90-99%
    ELITE = "elite"             # Top 1%


class DistributionType(Enum):
    """Types of wealth distribution."""
    EQUAL = "equal"             # Perfect equality
    NORMAL = "normal"           # Bell curve
    PARETO = "pareto"           # Power law (80/20)
    BIMODAL = "bimodal"         # Two peaks
    CONCENTRATED = "concentrated"  # High inequality


@dataclass
class WealthRecord:
    """
    Wealth record for an entity.
    """
    entity_id: str
    total_wealth: float = 0.0
    liquid_wealth: float = 0.0      # Easily accessible
    illiquid_wealth: float = 0.0    # Locked/invested
    debt: float = 0.0
    wealth_class: WealthClass = WealthClass.MIDDLE
    percentile: float = 50.0
    wealth_history: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    @property
    def net_worth(self) -> float:
        """Net worth after debt."""
        return self.total_wealth - self.debt

    def update_wealth(self, new_total: float) -> None:
        """Update total wealth."""
        self.wealth_history.append(self.total_wealth)
        self.total_wealth = new_total
        self.last_updated = time.time()

        # Keep last 1000 records
        if len(self.wealth_history) > 1000:
            self.wealth_history = self.wealth_history[-1000:]

    def wealth_change(self, periods: int = 1) -> float:
        """Get wealth change over periods."""
        if len(self.wealth_history) < periods:
            return 0.0
        return self.total_wealth - self.wealth_history[-periods]

    def wealth_growth_rate(self, periods: int = 100) -> float:
        """Calculate average growth rate over periods."""
        if len(self.wealth_history) < periods:
            return 0.0
        start = self.wealth_history[-periods]
        if start <= 0:
            return 0.0
        return (self.total_wealth - start) / start

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "entity_id": self.entity_id,
            "total_wealth": self.total_wealth,
            "net_worth": self.net_worth,
            "liquid_wealth": self.liquid_wealth,
            "illiquid_wealth": self.illiquid_wealth,
            "debt": self.debt,
            "wealth_class": self.wealth_class.value,
            "percentile": self.percentile,
            "growth_rate": self.wealth_growth_rate()
        }


class GiniCalculator:
    """
    Calculates Gini coefficient and related inequality metrics.
    """

    @staticmethod
    def gini_coefficient(values: List[float]) -> float:
        """
        Calculate Gini coefficient.

        Returns value between 0 (perfect equality) and 1 (perfect inequality).
        """
        if not values or len(values) < 2:
            return 0.0

        # Filter non-negative values
        sorted_values = sorted([v for v in values if v >= 0])
        n = len(sorted_values)
        if n == 0:
            return 0.0

        # Calculate using the formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
        total = sum(sorted_values)
        if total == 0:
            return 0.0

        weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_values))
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

        return max(0.0, min(1.0, gini))

    @staticmethod
    def lorenz_curve(values: List[float]) -> Tuple[List[float], List[float]]:
        """
        Calculate Lorenz curve points.

        Returns (cumulative_population, cumulative_wealth) both 0-1.
        """
        if not values:
            return [0.0, 1.0], [0.0, 1.0]

        sorted_values = sorted(v for v in values if v >= 0)
        n = len(sorted_values)
        total = sum(sorted_values)

        if n == 0 or total == 0:
            return [0.0, 1.0], [0.0, 1.0]

        cum_pop = [0.0]
        cum_wealth = [0.0]
        running_wealth = 0.0

        for i, v in enumerate(sorted_values):
            running_wealth += v
            cum_pop.append((i + 1) / n)
            cum_wealth.append(running_wealth / total)

        return cum_pop, cum_wealth

    @staticmethod
    def palma_ratio(values: List[float]) -> float:
        """
        Calculate Palma ratio (top 10% share / bottom 40% share).

        Higher values indicate more inequality.
        """
        if not values or len(values) < 10:
            return 1.0

        sorted_values = sorted(v for v in values if v >= 0)
        n = len(sorted_values)

        bottom_40_end = int(n * 0.4)
        top_10_start = int(n * 0.9)

        bottom_40_share = sum(sorted_values[:bottom_40_end])
        top_10_share = sum(sorted_values[top_10_start:])

        if bottom_40_share <= 0:
            return float('inf')

        return top_10_share / bottom_40_share

    @staticmethod
    def theil_index(values: List[float]) -> float:
        """
        Calculate Theil index (generalized entropy).

        0 = perfect equality, higher = more inequality.
        """
        if not values:
            return 0.0

        positive_values = [v for v in values if v > 0]
        if not positive_values:
            return 0.0

        n = len(positive_values)
        mean = sum(positive_values) / n

        if mean <= 0:
            return 0.0

        theil = sum((v / mean) * np.log(v / mean) for v in positive_values) / n
        return max(0.0, theil)

    @staticmethod
    def hoover_index(values: List[float]) -> float:
        """
        Calculate Hoover index (Robin Hood index).

        Represents the share of total income that would need to be
        redistributed for perfect equality.
        """
        if not values:
            return 0.0

        total = sum(v for v in values if v >= 0)
        if total <= 0:
            return 0.0

        n = len([v for v in values if v >= 0])
        mean = total / n

        return sum(abs(v - mean) for v in values if v >= 0) / (2 * total)


@dataclass
class WealthDistributionState:
    """
    Current state of wealth distribution.
    """
    gini: float = 0.0
    palma: float = 1.0
    theil: float = 0.0
    hoover: float = 0.0
    distribution_type: DistributionType = DistributionType.NORMAL
    median_wealth: float = 0.0
    mean_wealth: float = 0.0
    total_wealth: float = 0.0
    population: int = 0
    top_1_percent_share: float = 0.0
    top_10_percent_share: float = 0.0
    bottom_50_percent_share: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def classify_distribution(self) -> DistributionType:
        """Classify the type of distribution based on metrics."""
        if self.gini < 0.1:
            self.distribution_type = DistributionType.EQUAL
        elif self.gini > 0.6:
            self.distribution_type = DistributionType.CONCENTRATED
        elif self.top_1_percent_share > 0.3:
            self.distribution_type = DistributionType.PARETO
        else:
            self.distribution_type = DistributionType.NORMAL

        return self.distribution_type

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "gini": self.gini,
            "palma": self.palma,
            "theil": self.theil,
            "hoover": self.hoover,
            "distribution_type": self.distribution_type.value,
            "median_wealth": self.median_wealth,
            "mean_wealth": self.mean_wealth,
            "total_wealth": self.total_wealth,
            "population": self.population,
            "top_1_percent_share": self.top_1_percent_share,
            "top_10_percent_share": self.top_10_percent_share,
            "bottom_50_percent_share": self.bottom_50_percent_share,
            "timestamp": self.timestamp
        }


class WealthDistributionTracker:
    """
    Tracks wealth distribution across all entities.
    """

    def __init__(self):
        self._records: Dict[str, WealthRecord] = {}
        self._history: List[WealthDistributionState] = []
        self._calculator = GiniCalculator()
        self._current_state = WealthDistributionState()

    def register_entity(self, entity_id: str, initial_wealth: float = 0.0) -> WealthRecord:
        """Register an entity for wealth tracking."""
        record = WealthRecord(entity_id=entity_id, total_wealth=initial_wealth)
        self._records[entity_id] = record
        return record

    def get_record(self, entity_id: str) -> Optional[WealthRecord]:
        """Get wealth record for an entity."""
        return self._records.get(entity_id)

    def update_wealth(
        self,
        entity_id: str,
        total_wealth: Optional[float] = None,
        liquid_wealth: Optional[float] = None,
        illiquid_wealth: Optional[float] = None,
        debt: Optional[float] = None
    ) -> Optional[WealthRecord]:
        """Update wealth for an entity."""
        record = self._records.get(entity_id)
        if not record:
            return None

        if total_wealth is not None:
            record.update_wealth(total_wealth)
        if liquid_wealth is not None:
            record.liquid_wealth = liquid_wealth
        if illiquid_wealth is not None:
            record.illiquid_wealth = illiquid_wealth
        if debt is not None:
            record.debt = debt

        return record

    def calculate_distribution(self) -> WealthDistributionState:
        """Calculate current wealth distribution metrics."""
        wealth_values = [r.total_wealth for r in self._records.values()]

        if not wealth_values:
            return self._current_state

        # Sort for percentile calculations
        sorted_wealth = sorted(wealth_values)
        n = len(sorted_wealth)
        total = sum(sorted_wealth)

        # Calculate metrics
        state = WealthDistributionState(
            gini=self._calculator.gini_coefficient(wealth_values),
            palma=self._calculator.palma_ratio(wealth_values),
            theil=self._calculator.theil_index(wealth_values),
            hoover=self._calculator.hoover_index(wealth_values),
            median_wealth=sorted_wealth[n // 2] if n > 0 else 0,
            mean_wealth=total / n if n > 0 else 0,
            total_wealth=total,
            population=n
        )

        # Calculate share metrics
        if total > 0:
            top_1_idx = int(n * 0.99)
            top_10_idx = int(n * 0.90)
            bottom_50_idx = int(n * 0.50)

            state.top_1_percent_share = sum(sorted_wealth[top_1_idx:]) / total
            state.top_10_percent_share = sum(sorted_wealth[top_10_idx:]) / total
            state.bottom_50_percent_share = sum(sorted_wealth[:bottom_50_idx]) / total

        state.classify_distribution()
        self._current_state = state

        # Update individual records with class and percentile
        for record in self._records.values():
            record.percentile = self._calculate_percentile(
                record.total_wealth, sorted_wealth
            )
            record.wealth_class = self._classify_wealth(record.percentile)

        return state

    def _calculate_percentile(self, wealth: float, sorted_wealth: List[float]) -> float:
        """Calculate percentile for a wealth value."""
        if not sorted_wealth:
            return 50.0
        count_below = sum(1 for w in sorted_wealth if w < wealth)
        return (count_below / len(sorted_wealth)) * 100

    def _classify_wealth(self, percentile: float) -> WealthClass:
        """Classify wealth based on percentile."""
        if percentile < 10:
            return WealthClass.DESTITUTE
        elif percentile < 25:
            return WealthClass.POOR
        elif percentile < 50:
            return WealthClass.LOWER_MIDDLE
        elif percentile < 75:
            return WealthClass.MIDDLE
        elif percentile < 90:
            return WealthClass.UPPER_MIDDLE
        elif percentile < 99:
            return WealthClass.WEALTHY
        else:
            return WealthClass.ELITE

    def get_class_distribution(self) -> Dict[WealthClass, int]:
        """Get count of entities in each wealth class."""
        distribution = {wc: 0 for wc in WealthClass}
        for record in self._records.values():
            distribution[record.wealth_class] += 1
        return distribution

    def get_top_entities(self, n: int = 10) -> List[WealthRecord]:
        """Get top n wealthiest entities."""
        sorted_records = sorted(
            self._records.values(),
            key=lambda r: r.total_wealth,
            reverse=True
        )
        return sorted_records[:n]

    def get_bottom_entities(self, n: int = 10) -> List[WealthRecord]:
        """Get bottom n entities by wealth."""
        sorted_records = sorted(
            self._records.values(),
            key=lambda r: r.total_wealth
        )
        return sorted_records[:n]

    def get_lorenz_curve(self) -> Tuple[List[float], List[float]]:
        """Get Lorenz curve for current distribution."""
        wealth_values = [r.total_wealth for r in self._records.values()]
        return self._calculator.lorenz_curve(wealth_values)

    def get_mobility(self, entity_id: str, periods: int = 100) -> Optional[float]:
        """
        Calculate wealth mobility for an entity.

        Returns percentile change over periods.
        """
        record = self._records.get(entity_id)
        if not record or len(record.wealth_history) < periods:
            return None

        # Calculate old percentile
        old_wealth = record.wealth_history[-periods]
        all_wealth = [r.total_wealth for r in self._records.values()]
        old_percentile = self._calculate_percentile(old_wealth, sorted(all_wealth))

        return record.percentile - old_percentile

    def record_snapshot(self) -> None:
        """Record current distribution state in history."""
        state = self.calculate_distribution()
        self._history.append(state)

        # Keep last 1000 snapshots
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    def get_trend(self, periods: int = 100) -> Dict[str, float]:
        """Get trend in inequality metrics over periods."""
        if len(self._history) < periods:
            return {"gini_change": 0.0, "palma_change": 0.0}

        old_state = self._history[-periods]
        current_state = self._current_state

        return {
            "gini_change": current_state.gini - old_state.gini,
            "palma_change": current_state.palma - old_state.palma,
            "theil_change": current_state.theil - old_state.theil
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        state = self.calculate_distribution()
        return {
            "distribution": state.to_dict(),
            "class_distribution": {
                k.value: v for k, v in self.get_class_distribution().items()
            },
            "top_5": [r.to_dict() for r in self.get_top_entities(5)],
            "trend": self.get_trend(),
            "history_length": len(self._history)
        }


# Default global tracker
_default_tracker = WealthDistributionTracker()


def get_wealth_tracker() -> WealthDistributionTracker:
    """Get the default wealth distribution tracker."""
    return _default_tracker
