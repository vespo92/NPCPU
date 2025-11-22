"""
Metacognitive Architecture for AGI

Enables agents to analyze their own cognition, identify improvements,
and recursively self-improve. This is a foundation for AGI development.

Based on Long-Term Roadmap: Months 7-12 - Advanced Intelligence
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Data Structures
# ============================================================================

class TaskType(Enum):
    """Types of tasks an agent can perform"""
    PERCEPTION = "perception"
    REASONING = "reasoning"
    MEMORY = "memory"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    PLANNING = "planning"
    CREATIVITY = "creativity"
    ADAPTATION = "adaptation"


@dataclass
class Task:
    """A task for the agent to perform"""
    task_id: str
    type: TaskType
    description: str
    difficulty: float = 0.5  # 0-1
    time_limit: float = 1.0  # seconds
    requirements: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    """Result of performing a task"""
    task_id: str
    success: bool
    score: float = 0.0  # 0-1 performance
    response_time: float = 0.0
    perception_errors: float = 0.0
    knowledge_gaps: float = 0.0
    reasoning_errors: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAnalysis:
    """Analysis of task performance"""
    task_type: TaskType
    success: bool
    bottlenecks: List[str]
    underutilized_capabilities: List[str]
    capability_gaps: List[str]
    efficiency_score: float = 0.0
    improvement_potential: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ImprovementProposal:
    """Proposal for improving consciousness"""
    capability: str
    current_score: float
    proposed_score: float
    expected_benefit: float
    cost: float
    priority: int = 1
    rationale: str = ""


# ============================================================================
# Recursive Self-Improvement Engine
# ============================================================================

class RecursiveSelfImprovementEngine:
    """
    Enable agents to improve their own cognition.

    This is the path to AGI: systems that can redesign themselves
    to become more intelligent.

    Features:
    - Performance analysis and bottleneck detection
    - Improvement proposal generation
    - Safe self-modification with verification
    - Meta-learning (learning to learn)
    - Task distribution optimization

    Example:
        engine = RecursiveSelfImprovementEngine(consciousness)

        # Analyze task performance
        analysis = engine.analyze_performance(task, outcome)

        # Get improvement proposals
        proposals = engine.propose_improvements(analysis)

        # Apply improvements
        improved = engine.implement_improvement(proposals[0])
    """

    def __init__(
        self,
        consciousness: GradedConsciousness,
        meta_learning_rate: float = 0.01,
        safety_threshold: float = 0.8
    ):
        self.consciousness = consciousness
        self.improvement_history: List[Dict[str, Any]] = []
        self.meta_learning_rate = meta_learning_rate
        self.safety_threshold = safety_threshold

        # Meta-knowledge about task-capability relationships
        self.meta_knowledge: Dict[TaskType, Dict[str, float]] = {}
        self.task_history: List[Tuple[Task, Outcome]] = []

        # Performance tracking
        self.performance_by_type: Dict[TaskType, List[float]] = defaultdict(list)
        self.improvement_cycles = 0

    def analyze_performance(
        self,
        task: Task,
        outcome: Outcome
    ) -> PerformanceAnalysis:
        """
        Analyze why performance was good/bad.

        Meta-learning: learning to learn better.
        """
        # Record task
        self.task_history.append((task, outcome))
        self.performance_by_type[task.type].append(outcome.score)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(task, outcome)

        # Find underutilized capabilities
        underutilized = self._find_unused_capabilities(task)

        # Find capability gaps
        gaps = self._find_capability_gaps(task, outcome)

        # Calculate efficiency
        efficiency = self._calculate_efficiency(task, outcome)

        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(
            bottlenecks, gaps
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            bottlenecks, underutilized, gaps
        )

        return PerformanceAnalysis(
            task_type=task.type,
            success=outcome.success,
            bottlenecks=bottlenecks,
            underutilized_capabilities=underutilized,
            capability_gaps=gaps,
            efficiency_score=efficiency,
            improvement_potential=improvement_potential,
            recommendations=recommendations
        )

    def _identify_bottlenecks(
        self,
        task: Task,
        outcome: Outcome
    ) -> List[str]:
        """
        Find which capabilities limited performance.

        Example: Failed because perception was too slow.
        """
        bottlenecks = []
        scores = self.consciousness.get_capability_scores()

        # Check perception bottleneck
        if outcome.perception_errors > 0.3:
            if scores["perception_fidelity"] < 0.7:
                bottlenecks.append("perception_fidelity")

        # Check reaction speed bottleneck
        if outcome.response_time > task.time_limit:
            if scores["reaction_speed"] < 0.7:
                bottlenecks.append("reaction_speed")

        # Check memory bottleneck
        if outcome.knowledge_gaps > 0.2:
            if scores["memory_recall_accuracy"] < 0.7:
                bottlenecks.append("memory_recall_accuracy")
            if scores["memory_depth"] < 0.7:
                bottlenecks.append("memory_depth")

        # Check reasoning bottleneck
        if outcome.reasoning_errors > 0.2:
            if scores["meta_cognitive_ability"] < 0.7:
                bottlenecks.append("meta_cognitive_ability")
            if scores["information_integration"] < 0.7:
                bottlenecks.append("information_integration")

        # Task-specific bottlenecks
        for capability, required in task.requirements.items():
            if capability in scores and scores[capability] < required:
                if capability not in bottlenecks:
                    bottlenecks.append(capability)

        return bottlenecks

    def _find_unused_capabilities(self, task: Task) -> List[str]:
        """Find capabilities not needed for this task type"""
        scores = self.consciousness.get_capability_scores()

        # Define task-capability relevance
        task_relevance = {
            TaskType.PERCEPTION: {
                "perception_fidelity": 1.0,
                "reaction_speed": 0.8,
                "qualia_richness": 0.6,
                "introspection_capacity": 0.2
            },
            TaskType.REASONING: {
                "meta_cognitive_ability": 1.0,
                "information_integration": 0.9,
                "introspection_capacity": 0.7,
                "perception_fidelity": 0.3
            },
            TaskType.MEMORY: {
                "memory_depth": 1.0,
                "memory_recall_accuracy": 1.0,
                "information_integration": 0.5,
                "reaction_speed": 0.2
            },
            TaskType.LEARNING: {
                "meta_cognitive_ability": 1.0,
                "memory_depth": 0.9,
                "introspection_capacity": 0.8,
                "information_integration": 0.7
            },
            TaskType.COMMUNICATION: {
                "intentional_coherence": 1.0,
                "information_integration": 0.7,
                "memory_recall_accuracy": 0.6,
                "qualia_richness": 0.4
            },
            TaskType.PLANNING: {
                "meta_cognitive_ability": 1.0,
                "memory_depth": 0.8,
                "intentional_coherence": 0.7,
                "reaction_speed": 0.3
            },
            TaskType.CREATIVITY: {
                "qualia_richness": 1.0,
                "information_integration": 0.9,
                "introspection_capacity": 0.7,
                "reaction_speed": 0.2
            },
            TaskType.ADAPTATION: {
                "reaction_speed": 0.9,
                "perception_fidelity": 0.8,
                "meta_cognitive_ability": 0.7,
                "memory_depth": 0.5
            }
        }

        relevance = task_relevance.get(task.type, {})
        underutilized = []

        for capability, score in scores.items():
            if capability in relevance:
                if relevance[capability] < 0.3 and score > 0.5:
                    underutilized.append(capability)
            elif score > 0.5:
                # Capability not relevant to this task type at all
                underutilized.append(capability)

        return underutilized

    def _find_capability_gaps(
        self,
        task: Task,
        outcome: Outcome
    ) -> List[str]:
        """Find capabilities that need improvement"""
        gaps = []
        scores = self.consciousness.get_capability_scores()

        # Check explicit requirements
        for capability, required in task.requirements.items():
            if capability in scores:
                gap = required - scores[capability]
                if gap > 0.1:
                    gaps.append(capability)

        # Infer gaps from poor performance
        if not outcome.success and outcome.score < 0.5:
            # Look at lowest relevant capabilities
            sorted_caps = sorted(scores.items(), key=lambda x: x[1])
            for cap, score in sorted_caps[:3]:
                if cap not in gaps:
                    gaps.append(cap)

        return gaps

    def _calculate_efficiency(
        self,
        task: Task,
        outcome: Outcome
    ) -> float:
        """Calculate efficiency of task execution"""
        if not outcome.success:
            return 0.0

        # Time efficiency
        time_efficiency = min(1.0, task.time_limit / max(0.01, outcome.response_time))

        # Resource efficiency
        resource_efficiency = 1.0
        if outcome.resource_usage:
            avg_usage = np.mean(list(outcome.resource_usage.values()))
            resource_efficiency = 1.0 - min(1.0, avg_usage)

        # Performance efficiency
        perf_efficiency = outcome.score

        return (time_efficiency + resource_efficiency + perf_efficiency) / 3

    def _calculate_improvement_potential(
        self,
        bottlenecks: List[str],
        gaps: List[str]
    ) -> float:
        """Calculate how much improvement is possible"""
        scores = self.consciousness.get_capability_scores()

        potential = 0.0
        for capability in set(bottlenecks + gaps):
            if capability in scores:
                # Room for improvement
                room = 1.0 - scores[capability]
                potential += room * 0.5  # Assume we can close half the gap

        return min(1.0, potential)

    def _generate_recommendations(
        self,
        bottlenecks: List[str],
        underutilized: List[str],
        gaps: List[str]
    ) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []

        for bottleneck in bottlenecks:
            recommendations.append(
                f"Improve {bottleneck} - currently limiting performance"
            )

        for cap in underutilized:
            recommendations.append(
                f"Consider reducing {cap} to free resources for other capabilities"
            )

        for gap in gaps:
            recommendations.append(
                f"Address capability gap in {gap} to meet task requirements"
            )

        return recommendations

    def propose_improvements(
        self,
        analysis: PerformanceAnalysis
    ) -> List[ImprovementProposal]:
        """
        Propose ways to improve consciousness for this task type.

        This is where self-improvement happens.
        """
        proposals = []
        scores = self.consciousness.get_capability_scores()

        # Address bottlenecks (highest priority)
        for bottleneck in analysis.bottlenecks:
            if bottleneck in scores:
                current = scores[bottleneck]
                proposed = min(1.0, current * 1.2)  # 20% improvement
                cost = self._compute_improvement_cost(bottleneck, proposed - current)

                proposals.append(ImprovementProposal(
                    capability=bottleneck,
                    current_score=current,
                    proposed_score=proposed,
                    expected_benefit=0.15,  # 15% performance improvement
                    cost=cost,
                    priority=1,
                    rationale=f"Bottleneck causing performance limitation"
                ))

        # Address capability gaps (medium priority)
        for gap in analysis.capability_gaps:
            if gap in scores and gap not in analysis.bottlenecks:
                current = scores[gap]
                proposed = min(1.0, current + 0.15)  # Flat improvement
                cost = self._compute_improvement_cost(gap, proposed - current)

                proposals.append(ImprovementProposal(
                    capability=gap,
                    current_score=current,
                    proposed_score=proposed,
                    expected_benefit=0.10,
                    cost=cost,
                    priority=2,
                    rationale=f"Capability gap needs to be addressed"
                ))

        # Reduce underutilized capabilities (lower priority)
        for capability in analysis.underutilized_capabilities:
            if capability in scores:
                current = scores[capability]
                if current > 0.5:
                    proposed = current * 0.8  # 20% reduction
                    proposals.append(ImprovementProposal(
                        capability=capability,
                        current_score=current,
                        proposed_score=proposed,
                        expected_benefit=0.05,  # 5% resource savings
                        cost=-0.1,  # Negative cost = savings
                        priority=3,
                        rationale=f"Underutilized capability, can free resources"
                    ))

        # Sort by priority and expected benefit
        proposals.sort(key=lambda p: (p.priority, -p.expected_benefit))

        return proposals

    def _compute_improvement_cost(
        self,
        capability: str,
        improvement: float
    ) -> float:
        """Compute cost of improving a capability"""
        # Cost increases with improvement amount and current level
        scores = self.consciousness.get_capability_scores()
        current = scores.get(capability, 0.0)

        # Higher levels are more costly to improve
        base_cost = improvement * 0.1
        level_multiplier = 1.0 + current  # Gets harder as you improve

        return base_cost * level_multiplier

    def implement_improvement(
        self,
        proposal: ImprovementProposal
    ) -> GradedConsciousness:
        """
        Actually modify consciousness.

        This is self-modification with safety checks.
        """
        # Safety check
        if not self._is_safe_improvement(proposal):
            raise ValueError(
                f"Improvement proposal failed safety check: {proposal.rationale}"
            )

        improved_scores = self.consciousness.get_capability_scores()
        improved_scores[proposal.capability] = proposal.proposed_score

        # Create improved consciousness
        improved = GradedConsciousness(**improved_scores)

        # Verify improvement
        if not self._verify_improvement(improved, proposal):
            raise ValueError("Improvement verification failed")

        # Record improvement
        self.improvement_history.append({
            "timestamp": time.time(),
            "proposal": {
                "capability": proposal.capability,
                "current": proposal.current_score,
                "proposed": proposal.proposed_score,
                "benefit": proposal.expected_benefit,
                "cost": proposal.cost
            },
            "before": self.consciousness.overall_consciousness_score(),
            "after": improved.overall_consciousness_score()
        })

        self.improvement_cycles += 1
        self.consciousness = improved

        return improved

    def _is_safe_improvement(self, proposal: ImprovementProposal) -> bool:
        """Check if improvement is safe to implement"""
        # Don't allow dramatic changes
        if abs(proposal.proposed_score - proposal.current_score) > 0.5:
            return False

        # Don't allow negative scores
        if proposal.proposed_score < 0.0:
            return False

        # Don't allow scores above 1.0
        if proposal.proposed_score > 1.0:
            return False

        # Check overall consciousness doesn't drop too much
        test_scores = self.consciousness.get_capability_scores()
        test_scores[proposal.capability] = proposal.proposed_score
        test = GradedConsciousness(**test_scores)

        if test.overall_consciousness_score() < self.safety_threshold * self.consciousness.overall_consciousness_score():
            return False

        return True

    def _verify_improvement(
        self,
        improved: GradedConsciousness,
        proposal: ImprovementProposal
    ) -> bool:
        """Verify improvement was applied correctly"""
        scores = improved.get_capability_scores()

        # Verify capability was changed
        if abs(scores[proposal.capability] - proposal.proposed_score) > 0.01:
            return False

        # Verify other capabilities weren't changed
        original_scores = self.consciousness.get_capability_scores()
        for cap, score in scores.items():
            if cap != proposal.capability:
                if abs(score - original_scores[cap]) > 0.01:
                    return False

        return True

    def meta_learn(
        self,
        task_history: Optional[List[Tuple[Task, Outcome]]] = None
    ):
        """
        Learn general patterns about what consciousness works for what tasks.

        Meta-learning: learning about learning.
        """
        if task_history is None:
            task_history = self.task_history

        if not task_history:
            return

        # Build task-performance matrix
        task_types = set(task.type for task, _ in task_history)
        capabilities = list(self.consciousness.get_capability_scores().keys())

        # For each task type, find which capabilities matter most
        capability_importance: Dict[TaskType, Dict[str, float]] = {}

        for task_type in task_types:
            # Filter to this task type
            task_outcomes = [
                (t, o) for t, o in task_history
                if t.type == task_type
            ]

            if len(task_outcomes) < 2:
                continue

            # Compute importance of each capability
            importance = {}
            for capability in capabilities:
                # Correlate capability score with task success
                importance[capability] = self._compute_capability_importance(
                    capability, task_outcomes
                )

            capability_importance[task_type] = importance

        # Update meta-knowledge
        self.meta_knowledge = capability_importance

    def _compute_capability_importance(
        self,
        capability: str,
        task_outcomes: List[Tuple[Task, Outcome]]
    ) -> float:
        """Compute how important a capability is for task success"""
        scores = self.consciousness.get_capability_scores()
        cap_score = scores.get(capability, 0.5)

        # Simple heuristic: correlate with task requirements
        importance_sum = 0.0
        count = 0

        for task, outcome in task_outcomes:
            if capability in task.requirements:
                # Direct requirement
                required = task.requirements[capability]
                success_correlation = outcome.score if cap_score >= required else 0.0
                importance_sum += required * success_correlation
                count += 1
            else:
                # Indirect: weight by task success
                importance_sum += outcome.score * 0.3
                count += 1

        return importance_sum / max(1, count)

    def optimize_for_task_distribution(
        self,
        task_distribution: Dict[TaskType, float]  # task_type -> frequency
    ) -> GradedConsciousness:
        """
        Optimize consciousness for expected task distribution.

        If 80% of tasks are perception, optimize perception capabilities.
        """
        if not self.meta_knowledge:
            self.meta_learn()

        if not self.meta_knowledge:
            return self.consciousness

        # Compute optimal capability allocation
        scores = self.consciousness.get_capability_scores()
        optimal_scores = {}

        for capability in scores.keys():
            # Weighted importance across task distribution
            weighted_importance = sum(
                task_freq * self.meta_knowledge.get(task_type, {}).get(capability, 0.5)
                for task_type, task_freq in task_distribution.items()
            )

            # Allocate capability proportional to importance
            optimal_scores[capability] = weighted_importance

        # Normalize to keep total consciousness roughly constant
        total = sum(optimal_scores.values())
        current_total = sum(scores.values())

        if total > 0:
            optimal_scores = {
                k: v * (current_total / total)
                for k, v in optimal_scores.items()
            }

        # Clamp to valid range
        optimal_scores = {
            k: max(0.0, min(1.0, v))
            for k, v in optimal_scores.items()
        }

        return GradedConsciousness(**optimal_scores)

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of all improvements made"""
        if not self.improvement_history:
            return {
                "total_cycles": 0,
                "total_improvement": 0.0,
                "improved_capabilities": []
            }

        improved_caps = list(set(
            h["proposal"]["capability"]
            for h in self.improvement_history
        ))

        first_score = self.improvement_history[0]["before"]
        last_score = self.improvement_history[-1]["after"]

        return {
            "total_cycles": self.improvement_cycles,
            "total_improvement": last_score - first_score,
            "improvement_percentage": (last_score - first_score) / first_score * 100,
            "improved_capabilities": improved_caps,
            "history_length": len(self.improvement_history)
        }

    def run_improvement_cycle(
        self,
        task: Task,
        outcome: Outcome,
        auto_apply: bool = True
    ) -> Tuple[PerformanceAnalysis, List[ImprovementProposal], Optional[GradedConsciousness]]:
        """Run a complete improvement cycle"""
        # Analyze
        analysis = self.analyze_performance(task, outcome)

        # Propose improvements
        proposals = self.propose_improvements(analysis)

        # Apply best improvement if requested
        improved = None
        if auto_apply and proposals:
            try:
                improved = self.implement_improvement(proposals[0])
            except ValueError:
                pass  # Improvement failed safety check

        # Meta-learn
        if len(self.task_history) % 10 == 0:
            self.meta_learn()

        return analysis, proposals, improved


# ============================================================================
# Meta-Learner
# ============================================================================

class MetaLearner:
    """
    Higher-order learning system that learns how to learn.

    Features:
    - Track learning efficiency across domains
    - Optimize learning strategies
    - Transfer learning between domains
    """

    def __init__(self):
        self.learning_strategies: Dict[str, Dict[str, float]] = {}
        self.domain_performance: Dict[str, List[float]] = defaultdict(list)
        self.strategy_effectiveness: Dict[str, float] = {}

    def register_learning_episode(
        self,
        domain: str,
        strategy: str,
        initial_performance: float,
        final_performance: float,
        steps: int
    ):
        """Record a learning episode for meta-analysis"""
        improvement = final_performance - initial_performance
        efficiency = improvement / max(1, steps)

        self.domain_performance[domain].append(final_performance)

        # Track strategy effectiveness
        if strategy not in self.learning_strategies:
            self.learning_strategies[strategy] = {}

        if domain not in self.learning_strategies[strategy]:
            self.learning_strategies[strategy][domain] = []

        self.learning_strategies[strategy][domain].append(efficiency)

    def recommend_strategy(
        self,
        domain: str,
        similar_domains: List[str] = None
    ) -> str:
        """Recommend best learning strategy for a domain"""
        best_strategy = None
        best_efficiency = -float('inf')

        for strategy, domain_perf in self.learning_strategies.items():
            # Direct domain experience
            if domain in domain_perf and domain_perf[domain]:
                avg_eff = np.mean(domain_perf[domain])
                if avg_eff > best_efficiency:
                    best_efficiency = avg_eff
                    best_strategy = strategy

            # Transfer from similar domains
            if similar_domains:
                for sim_domain in similar_domains:
                    if sim_domain in domain_perf and domain_perf[sim_domain]:
                        avg_eff = np.mean(domain_perf[sim_domain]) * 0.8  # Discount
                        if avg_eff > best_efficiency:
                            best_efficiency = avg_eff
                            best_strategy = strategy

        return best_strategy or "default"

    def get_learning_curve(self, domain: str) -> List[float]:
        """Get learning curve for a domain"""
        return self.domain_performance.get(domain, [])

    def estimate_time_to_mastery(
        self,
        domain: str,
        target_performance: float = 0.9
    ) -> Optional[int]:
        """Estimate steps needed to reach mastery"""
        curve = self.get_learning_curve(domain)
        if len(curve) < 2:
            return None

        # Simple linear extrapolation
        recent_improvement = curve[-1] - curve[-2] if len(curve) > 1 else 0.1
        current = curve[-1] if curve else 0.0
        gap = target_performance - current

        if recent_improvement <= 0:
            return None

        return int(gap / recent_improvement)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Recursive Self-Improvement Engine Demo")
    print("=" * 50)

    # Create consciousness and engine
    consciousness = GradedConsciousness(
        perception_fidelity=0.6,
        reaction_speed=0.5,
        memory_depth=0.7,
        memory_recall_accuracy=0.6,
        introspection_capacity=0.5,
        meta_cognitive_ability=0.4,
        information_integration=0.6,
        intentional_coherence=0.5,
        qualia_richness=0.5
    )

    engine = RecursiveSelfImprovementEngine(consciousness)

    print(f"\n1. Initial consciousness score: {consciousness.overall_consciousness_score():.3f}")

    # Run improvement cycles
    print("\n2. Running improvement cycles...")

    tasks_and_outcomes = [
        (
            Task(
                task_id="task_1",
                type=TaskType.PERCEPTION,
                description="Visual recognition",
                difficulty=0.7,
                time_limit=0.5,
                requirements={"perception_fidelity": 0.8}
            ),
            Outcome(
                task_id="task_1",
                success=False,
                score=0.4,
                response_time=0.8,
                perception_errors=0.4
            )
        ),
        (
            Task(
                task_id="task_2",
                type=TaskType.REASONING,
                description="Logical inference",
                difficulty=0.6,
                time_limit=1.0,
                requirements={"meta_cognitive_ability": 0.7}
            ),
            Outcome(
                task_id="task_2",
                success=True,
                score=0.7,
                response_time=0.6,
                reasoning_errors=0.2
            )
        ),
        (
            Task(
                task_id="task_3",
                type=TaskType.MEMORY,
                description="Recall test",
                difficulty=0.5,
                time_limit=2.0,
                requirements={"memory_recall_accuracy": 0.8}
            ),
            Outcome(
                task_id="task_3",
                success=False,
                score=0.5,
                response_time=1.5,
                knowledge_gaps=0.3
            )
        )
    ]

    for task, outcome in tasks_and_outcomes:
        print(f"\n   Processing {task.type.value} task...")
        analysis, proposals, improved = engine.run_improvement_cycle(task, outcome)

        print(f"     Success: {outcome.success}")
        print(f"     Bottlenecks: {analysis.bottlenecks}")
        print(f"     Improvement potential: {analysis.improvement_potential:.2f}")

        if proposals:
            print(f"     Top proposal: {proposals[0].capability} "
                  f"({proposals[0].current_score:.2f} -> {proposals[0].proposed_score:.2f})")

        if improved:
            print(f"     New consciousness score: {improved.overall_consciousness_score():.3f}")

    # Summary
    print("\n3. Improvement Summary:")
    summary = engine.get_improvement_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # Optimize for task distribution
    print("\n4. Optimizing for task distribution...")
    distribution = {
        TaskType.PERCEPTION: 0.4,
        TaskType.REASONING: 0.3,
        TaskType.MEMORY: 0.2,
        TaskType.LEARNING: 0.1
    }

    engine.meta_learn()
    optimized = engine.optimize_for_task_distribution(distribution)

    print(f"   Optimized consciousness score: {optimized.overall_consciousness_score():.3f}")
    print("   Optimized scores:")
    for cap, score in optimized.get_capability_scores().items():
        print(f"     {cap}: {score:.2f}")
