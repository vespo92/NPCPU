"""
Meta-Learning System for ORACLE-Z Metacognitive Agent

Implements learning-to-learn capabilities that enable the system
to optimize its own learning processes and adapt learning strategies
based on experience.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from abc import ABC, abstractmethod


class LearningStrategy(Enum):
    """Meta-learning strategies"""
    GRADIENT_BASED = "gradient_based"       # Learn to update quickly
    METRIC_BASED = "metric_based"           # Learn similarity measures
    MODEL_BASED = "model_based"             # Learn task structure
    MEMORY_BASED = "memory_based"           # Learn from stored experiences
    CURRICULUM = "curriculum"               # Learn ordering of tasks
    SELF_PACED = "self_paced"              # Learn at own pace
    TRANSFER = "transfer"                   # Transfer learning optimization


class LearningDomain(Enum):
    """Domains for meta-learning"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MOTOR = "motor"
    LANGUAGE = "language"
    SOCIAL = "social"
    ABSTRACT = "abstract"
    PROCEDURAL = "procedural"


@dataclass
class LearningTask:
    """A learning task for meta-learning"""
    task_id: str
    domain: LearningDomain
    complexity: float  # 0.0 to 1.0
    samples_required: int
    similarity_to_prior: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEpisode:
    """Record of a learning episode"""
    episode_id: str
    task: LearningTask
    strategy_used: LearningStrategy
    initial_performance: float
    final_performance: float
    samples_used: int
    time_taken: float
    learning_curve: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def learning_efficiency(self) -> float:
        """Calculate learning efficiency"""
        improvement = self.final_performance - self.initial_performance
        if self.samples_used == 0:
            return 0.0
        return improvement / self.samples_used

    @property
    def success(self) -> bool:
        """Was learning successful?"""
        return self.final_performance > self.initial_performance + 0.1


@dataclass
class LearningRate:
    """Adaptive learning rate with meta-learned parameters"""
    base_rate: float = 0.01
    min_rate: float = 0.0001
    max_rate: float = 0.1
    decay: float = 0.99
    warmup_steps: int = 100
    current_step: int = 0
    momentum: float = 0.9

    def get_rate(self) -> float:
        """Get current learning rate with scheduling"""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            return self.base_rate * (self.current_step / self.warmup_steps)

        # Apply decay
        decayed = self.base_rate * (self.decay ** (self.current_step - self.warmup_steps))
        return np.clip(decayed, self.min_rate, self.max_rate)

    def step(self):
        """Increment step counter"""
        self.current_step += 1


class MetaLearner:
    """
    Core meta-learning system for ORACLE-Z.

    Implements learning-to-learn capabilities:
    - Strategy selection for new tasks
    - Learning rate adaptation
    - Curriculum optimization
    - Knowledge transfer optimization
    - Learning efficiency tracking
    """

    def __init__(self):
        # Learning episode history
        self.episodes: List[LearningEpisode] = []
        self.max_episodes = 1000

        # Strategy performance tracking
        self.strategy_performance: Dict[LearningStrategy, Dict[str, float]] = {
            s: {"success_rate": 0.5, "avg_efficiency": 0.1, "sample_count": 0}
            for s in LearningStrategy
        }

        # Domain-specific meta-knowledge
        self.domain_knowledge: Dict[LearningDomain, Dict[str, Any]] = {
            d: {
                "preferred_strategy": None,
                "avg_complexity": 0.5,
                "learning_curve_shape": "linear",
                "transfer_sources": []
            }
            for d in LearningDomain
        }

        # Adaptive learning rates
        self.learning_rates: Dict[str, LearningRate] = {
            "global": LearningRate()
        }

        # Task embedding space (for similarity)
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dim = 32

        # Curriculum state
        self.curriculum_position = 0
        self.curriculum_tasks: List[str] = []

        # Meta-learning hyperparameters
        self.meta_learning_rate = 0.1
        self.exploration_rate = 0.2
        self.similarity_threshold = 0.7

    def select_strategy(self, task: LearningTask) -> LearningStrategy:
        """Select optimal learning strategy for a task"""
        scores = {}

        for strategy in LearningStrategy:
            score = self._score_strategy(strategy, task)
            scores[strategy] = score

        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore less-used strategies
            counts = [
                self.strategy_performance[s]["sample_count"]
                for s in LearningStrategy
            ]
            min_count = min(counts)
            candidates = [
                s for s in LearningStrategy
                if self.strategy_performance[s]["sample_count"] <= min_count + 5
            ]
            if candidates:
                return np.random.choice(candidates)

        return max(scores, key=scores.get)

    def _score_strategy(self, strategy: LearningStrategy, task: LearningTask) -> float:
        """Score a strategy for a given task"""
        perf = self.strategy_performance[strategy]
        domain_pref = self.domain_knowledge[task.domain]["preferred_strategy"]

        # Base score from historical performance
        base_score = perf["success_rate"] * 0.5 + perf["avg_efficiency"] * 0.5

        # Domain preference bonus
        if domain_pref == strategy:
            base_score *= 1.3

        # Complexity adjustment
        if task.complexity > 0.7:
            if strategy in [LearningStrategy.MODEL_BASED, LearningStrategy.GRADIENT_BASED]:
                base_score *= 1.2
            elif strategy == LearningStrategy.METRIC_BASED:
                base_score *= 0.8

        # Transfer potential
        if task.similarity_to_prior > self.similarity_threshold:
            if strategy in [LearningStrategy.TRANSFER, LearningStrategy.MEMORY_BASED]:
                base_score *= 1.4

        # Low sample scenarios
        if task.samples_required < 10:
            if strategy in [LearningStrategy.METRIC_BASED, LearningStrategy.MEMORY_BASED]:
                base_score *= 1.3

        return base_score

    def record_episode(self, episode: LearningEpisode):
        """Record a learning episode and update meta-knowledge"""
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

        # Update strategy performance
        strategy = episode.strategy_used
        perf = self.strategy_performance[strategy]
        perf["sample_count"] += 1
        alpha = 2 / (perf["sample_count"] + 1)

        perf["success_rate"] = (1 - alpha) * perf["success_rate"] + alpha * float(episode.success)
        perf["avg_efficiency"] = (1 - alpha) * perf["avg_efficiency"] + alpha * episode.learning_efficiency

        # Update domain knowledge
        domain = episode.task.domain
        if episode.success:
            if (self.domain_knowledge[domain]["preferred_strategy"] is None or
                episode.learning_efficiency > self.strategy_performance[
                    self.domain_knowledge[domain]["preferred_strategy"]
                ]["avg_efficiency"]):
                self.domain_knowledge[domain]["preferred_strategy"] = strategy

        # Update task embedding
        self._update_task_embedding(episode.task)

    def _update_task_embedding(self, task: LearningTask):
        """Update embedding for a task"""
        # Create simple embedding from task features
        embedding = np.zeros(self.embedding_dim)

        # Domain encoding
        domain_idx = list(LearningDomain).index(task.domain)
        embedding[:8] = 0
        embedding[domain_idx] = 1.0

        # Complexity encoding
        embedding[8:16] = task.complexity

        # Similarity encoding
        embedding[16:24] = task.similarity_to_prior

        self.task_embeddings[task.task_id] = embedding

    def find_similar_tasks(self, task: LearningTask, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar tasks from history"""
        if not self.task_embeddings:
            return []

        current_embedding = self._get_embedding(task)
        similarities = []

        for task_id, embedding in self.task_embeddings.items():
            similarity = np.dot(current_embedding, embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embedding) + 1e-6
            )
            similarities.append((task_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _get_embedding(self, task: LearningTask) -> np.ndarray:
        """Get or create embedding for a task"""
        if task.task_id in self.task_embeddings:
            return self.task_embeddings[task.task_id]

        # Create temporary embedding
        embedding = np.zeros(self.embedding_dim)
        domain_idx = list(LearningDomain).index(task.domain)
        embedding[domain_idx] = 1.0
        embedding[8:16] = task.complexity
        embedding[16:24] = task.similarity_to_prior

        return embedding

    def get_learning_rate(self, domain: Optional[LearningDomain] = None) -> float:
        """Get adaptive learning rate"""
        if domain and domain.value in self.learning_rates:
            return self.learning_rates[domain.value].get_rate()
        return self.learning_rates["global"].get_rate()

    def adapt_learning_rate(self, performance_delta: float, domain: Optional[LearningDomain] = None):
        """Adapt learning rate based on performance"""
        key = domain.value if domain else "global"

        if key not in self.learning_rates:
            self.learning_rates[key] = LearningRate()

        lr = self.learning_rates[key]

        if performance_delta > 0.1:
            # Good progress, increase rate slightly
            lr.base_rate = min(lr.base_rate * 1.1, lr.max_rate)
        elif performance_delta < -0.05:
            # Negative progress, decrease rate
            lr.base_rate = max(lr.base_rate * 0.9, lr.min_rate)

        lr.step()

    def optimize_curriculum(self, available_tasks: List[LearningTask]) -> List[LearningTask]:
        """Optimize task curriculum for efficient learning"""
        if not available_tasks:
            return []

        # Sort by complexity (start simple)
        sorted_tasks = sorted(available_tasks, key=lambda t: t.complexity)

        # Group by domain for transfer learning
        domain_groups: Dict[LearningDomain, List[LearningTask]] = {}
        for task in sorted_tasks:
            if task.domain not in domain_groups:
                domain_groups[task.domain] = []
            domain_groups[task.domain].append(task)

        # Interleave domains for better generalization
        curriculum = []
        while any(domain_groups.values()):
            for domain in list(domain_groups.keys()):
                if domain_groups[domain]:
                    curriculum.append(domain_groups[domain].pop(0))
                else:
                    del domain_groups[domain]

        return curriculum

    def estimate_samples_needed(self, task: LearningTask) -> int:
        """Estimate samples needed to learn a task"""
        # Base estimate from task properties
        base_samples = int(100 * task.complexity)

        # Adjust for similarity to prior tasks
        similar = self.find_similar_tasks(task, top_k=3)
        if similar:
            avg_similarity = np.mean([s[1] for s in similar])
            if avg_similarity > 0.8:
                base_samples = int(base_samples * 0.3)  # Much less needed
            elif avg_similarity > 0.5:
                base_samples = int(base_samples * 0.6)

        # Domain-specific adjustment
        domain_info = self.domain_knowledge[task.domain]
        if domain_info["preferred_strategy"]:
            strategy_perf = self.strategy_performance[domain_info["preferred_strategy"]]
            if strategy_perf["avg_efficiency"] > 0.1:
                base_samples = int(base_samples / (strategy_perf["avg_efficiency"] * 10 + 1))

        return max(10, base_samples)

    def predict_learning_curve(self,
                              task: LearningTask,
                              strategy: LearningStrategy) -> List[float]:
        """Predict learning curve for a task"""
        samples = self.estimate_samples_needed(task)
        curve_shape = self.domain_knowledge[task.domain]["learning_curve_shape"]

        steps = 10
        curve = []

        for i in range(steps):
            progress = i / steps

            if curve_shape == "linear":
                performance = 0.3 + 0.6 * progress
            elif curve_shape == "exponential":
                performance = 0.3 + 0.6 * (1 - np.exp(-3 * progress))
            elif curve_shape == "logarithmic":
                performance = 0.3 + 0.6 * np.log(1 + progress * 10) / np.log(11)
            elif curve_shape == "sigmoid":
                performance = 0.3 + 0.6 / (1 + np.exp(-10 * (progress - 0.5)))
            else:
                performance = 0.3 + 0.6 * progress

            curve.append(performance)

        return curve

    def get_transfer_opportunities(self, target_task: LearningTask) -> List[Dict[str, Any]]:
        """Identify transfer learning opportunities"""
        opportunities = []

        # Find similar completed tasks
        similar = self.find_similar_tasks(target_task, top_k=10)

        for task_id, similarity in similar:
            # Find episodes for this task
            relevant_episodes = [
                e for e in self.episodes
                if e.task.task_id == task_id and e.success
            ]

            if relevant_episodes:
                best_episode = max(relevant_episodes, key=lambda e: e.final_performance)
                opportunities.append({
                    "source_task_id": task_id,
                    "similarity": similarity,
                    "source_performance": best_episode.final_performance,
                    "strategy_used": best_episode.strategy_used.value,
                    "estimated_benefit": similarity * best_episode.final_performance
                })

        # Sort by estimated benefit
        opportunities.sort(key=lambda x: x["estimated_benefit"], reverse=True)
        return opportunities[:5]

    def analyze_learning_efficiency(self, window: int = 50) -> Dict[str, Any]:
        """Analyze recent learning efficiency"""
        recent = self.episodes[-window:]
        if not recent:
            return {"status": "no_data"}

        successful = [e for e in recent if e.success]
        efficiencies = [e.learning_efficiency for e in recent]

        # Analyze by strategy
        strategy_analysis = {}
        for strategy in LearningStrategy:
            strategy_episodes = [e for e in recent if e.strategy_used == strategy]
            if strategy_episodes:
                strategy_analysis[strategy.value] = {
                    "count": len(strategy_episodes),
                    "success_rate": len([e for e in strategy_episodes if e.success]) / len(strategy_episodes),
                    "avg_efficiency": np.mean([e.learning_efficiency for e in strategy_episodes])
                }

        # Analyze by domain
        domain_analysis = {}
        for domain in LearningDomain:
            domain_episodes = [e for e in recent if e.task.domain == domain]
            if domain_episodes:
                domain_analysis[domain.value] = {
                    "count": len(domain_episodes),
                    "success_rate": len([e for e in domain_episodes if e.success]) / len(domain_episodes),
                    "avg_complexity": np.mean([e.task.complexity for e in domain_episodes])
                }

        return {
            "total_episodes": len(recent),
            "success_rate": len(successful) / len(recent) if recent else 0,
            "mean_efficiency": np.mean(efficiencies) if efficiencies else 0,
            "efficiency_trend": self._calculate_trend(efficiencies),
            "by_strategy": strategy_analysis,
            "by_domain": domain_analysis
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return "insufficient_data"

        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        return "stable"

    def get_meta_learning_state(self) -> Dict[str, Any]:
        """Get current meta-learning state"""
        return {
            "episode_count": len(self.episodes),
            "strategy_performance": {
                s.value: {
                    "success_rate": p["success_rate"],
                    "avg_efficiency": p["avg_efficiency"],
                    "sample_count": int(p["sample_count"])
                }
                for s, p in self.strategy_performance.items()
            },
            "domain_preferences": {
                d.value: {
                    "preferred_strategy": k["preferred_strategy"].value if k["preferred_strategy"] else None,
                    "learning_curve_shape": k["learning_curve_shape"]
                }
                for d, k in self.domain_knowledge.items()
            },
            "global_learning_rate": self.learning_rates["global"].get_rate(),
            "exploration_rate": self.exploration_rate,
            "task_embeddings_count": len(self.task_embeddings)
        }

    def export_state(self, filepath: str):
        """Export meta-learner state"""
        state = {
            "meta_learning_state": self.get_meta_learning_state(),
            "recent_episodes": [
                {
                    "task_id": e.task.task_id,
                    "domain": e.task.domain.value,
                    "strategy": e.strategy_used.value,
                    "success": e.success,
                    "efficiency": e.learning_efficiency
                }
                for e in self.episodes[-50:]
            ],
            "efficiency_analysis": self.analyze_learning_efficiency()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
