"""
ORACLE-Z Metacognitive Agent for NPCPU

The main agent that integrates all metacognitive capabilities:
- Self-modeling and diagnosis
- Strategy selection and optimization
- Confidence calibration and uncertainty estimation
- Deep introspection
- Dynamic goal management
- Meta-learning

ORACLE-Z is the 10th agent in the NPCPU workstream system,
responsible for metacognitive bootstrap and self-improvement.
"""

import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import uuid
from enum import Enum, auto

# Import core metacognitive components
from self_model import SelfModel, SelfDiagnosis, CognitiveComponent
from strategy_selection import (
    StrategyOptimizer, StrategyContext, CognitiveStrategy,
    TaskDomain, StrategyType
)
from confidence_calibration import ConfidenceCalibrator, ConfidenceEstimate
from introspection import (
    IntrospectionEngine, IntrospectionDepth,
    CognitiveProcess, ProcessState
)
from goal_management import (
    GoalManager, Goal, GoalType, GoalPriority, GoalStatus
)
from learning_to_learn import (
    MetaLearner, LearningTask, LearningEpisode,
    LearningStrategy, LearningDomain
)
from bootstrap_engine import MetaCognitiveBootstrap, SemanticInsight
from recursive_improvement import RecursiveImprovementEngine


class OracleZState(Enum):
    """Operational states of ORACLE-Z"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    OBSERVING = "observing"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"
    SELF_IMPROVING = "self_improving"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class OracleZConfig:
    """Configuration for ORACLE-Z agent"""
    # Self-improvement settings
    auto_improve: bool = True
    improvement_threshold: float = 0.1
    max_improvement_cycles: int = 10

    # Safety settings
    safety_bounds_enabled: bool = True
    max_confidence_without_verification: float = 0.8
    rollback_on_degradation: bool = True
    degradation_threshold: float = 0.15

    # Introspection settings
    introspection_interval: int = 100  # Steps between introspections
    default_introspection_depth: IntrospectionDepth = IntrospectionDepth.MODERATE

    # Learning settings
    meta_learning_enabled: bool = True
    exploration_rate: float = 0.1

    # Goal settings
    max_active_goals: int = 10
    goal_pruning_enabled: bool = True


@dataclass
class OracleZMetrics:
    """Metrics tracked by ORACLE-Z"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    improvement_cycles: int = 0
    goals_completed: int = 0
    insights_generated: int = 0
    strategies_adapted: int = 0
    anomalies_detected: int = 0
    uptime_seconds: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class OracleZAgent:
    """
    ORACLE-Z: Metacognitive Bootstrap & Self-Improvement Agent

    The central metacognitive agent that coordinates all self-aware
    and self-improving capabilities of the NPCPU system.
    """

    def __init__(self, config: Optional[OracleZConfig] = None):
        self.agent_id = f"oracle_z_{uuid.uuid4().hex[:8]}"
        self.config = config or OracleZConfig()
        self.state = OracleZState.INITIALIZING

        # Initialize logging
        self.logger = logging.getLogger(f"ORACLE-Z.{self.agent_id}")
        self.logger.setLevel(logging.INFO)

        # Core components
        self.self_model = SelfModel(f"{self.agent_id}_self_model")
        self.self_diagnosis = SelfDiagnosis(self.self_model)
        self.strategy_optimizer = StrategyOptimizer()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.introspection_engine = IntrospectionEngine()
        self.goal_manager = GoalManager()
        self.meta_learner = MetaLearner()

        # Bootstrap and improvement engines
        self.bootstrap_engine = MetaCognitiveBootstrap()
        self.improvement_engine = RecursiveImprovementEngine()

        # Metrics and history
        self.metrics = OracleZMetrics()
        self.operation_history: List[Dict[str, Any]] = []
        self.state_snapshots: List[Dict[str, Any]] = []

        # Runtime state
        self.step_count = 0
        self.start_time = datetime.now()
        self.last_introspection = 0
        self.current_operation: Optional[str] = None

        # Initialize
        self._initialize_core_capabilities()
        self._register_core_goals()

        self.state = OracleZState.IDLE
        self.logger.info(f"ORACLE-Z agent {self.agent_id} initialized")

    def _initialize_core_capabilities(self):
        """Initialize core cognitive capabilities in self-model"""
        capabilities = [
            ("self_modeling", 0.8, 0.9),
            ("strategy_selection", 0.75, 0.85),
            ("confidence_calibration", 0.7, 0.8),
            ("deep_introspection", 0.8, 0.9),
            ("goal_management", 0.85, 0.9),
            ("meta_learning", 0.7, 0.85),
            ("self_improvement", 0.75, 0.8),
        ]

        for name, level, reliability in capabilities:
            self.self_model.register_capability(name, level, reliability)

        # Register known limitations
        self.self_model.register_limitation(
            "computational_bounds",
            severity=0.3,
            scope=[CognitiveComponent.REASONING, CognitiveComponent.PLANNING],
            workarounds=["Use heuristics", "Decompose problems"]
        )

        self.self_model.register_limitation(
            "sample_efficiency",
            severity=0.4,
            scope=[CognitiveComponent.LEARNING],
            workarounds=["Transfer learning", "Meta-learning"]
        )

    def _register_core_goals(self):
        """Register fundamental ORACLE-Z goals"""
        # These are added to the core goals already in GoalManager

        self.goal_manager.create_goal(
            name="Maintain Self-Model Accuracy",
            description="Keep the internal self-model accurate and up-to-date",
            goal_type=GoalType.MAINTENANCE,
            priority=GoalPriority.HIGH,
            context={"refresh_interval": 100}
        )

        self.goal_manager.create_goal(
            name="Optimize Learning Efficiency",
            description="Continuously improve meta-learning capabilities",
            goal_type=GoalType.OPTIMIZATION,
            priority=GoalPriority.MEDIUM
        )

        self.goal_manager.create_goal(
            name="Monitor System Health",
            description="Detect and respond to system anomalies",
            goal_type=GoalType.MAINTENANCE,
            priority=GoalPriority.HIGH
        )

    async def step(self) -> Dict[str, Any]:
        """Execute a single step of the ORACLE-Z agent"""
        self.step_count += 1
        step_result = {
            "step": self.step_count,
            "state": self.state.value,
            "actions": [],
            "insights": []
        }

        try:
            # Periodic introspection
            if self.step_count - self.last_introspection >= self.config.introspection_interval:
                introspection_result = await self._perform_introspection()
                step_result["introspection"] = introspection_result
                self.last_introspection = self.step_count

            # Process active goals
            next_goal = self.goal_manager.get_next_goal()
            if next_goal:
                goal_result = await self._pursue_goal(next_goal)
                step_result["goal_pursuit"] = goal_result

            # Check for improvement opportunities
            if self.config.auto_improve:
                improvement = await self._check_improvement_opportunity()
                if improvement:
                    step_result["improvement"] = improvement

            # Update metrics
            self.metrics.total_operations += 1
            self.metrics.last_update = datetime.now()
            self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        except Exception as e:
            self.logger.error(f"Step error: {e}")
            self.state = OracleZState.ERROR_RECOVERY
            await self._handle_error(e)
            step_result["error"] = str(e)
            self.metrics.failed_operations += 1

        return step_result

    async def _perform_introspection(self) -> Dict[str, Any]:
        """Perform deep introspection"""
        self.state = OracleZState.OBSERVING

        # Capture current cognitive process states
        for process in CognitiveProcess:
            self.introspection_engine.capture_snapshot(
                process=process,
                state=ProcessState.ACTIVE if np.random.random() > 0.3 else ProcessState.INACTIVE,
                activation_level=np.random.uniform(0.3, 0.9),
                processing_load=np.random.uniform(0.1, 0.7)
            )

        # Run introspection
        result = self.introspection_engine.introspect(
            target="all",
            depth=self.config.default_introspection_depth
        )

        # Run self-diagnosis
        diagnosis = self.self_diagnosis.diagnose()

        # Generate insights from findings
        insights = await self._generate_insights_from_introspection(result, diagnosis)

        # Record anomalies
        self.metrics.anomalies_detected += len(result.anomalies)

        self.state = OracleZState.IDLE
        return {
            "patterns": result.patterns,
            "anomalies": result.anomalies,
            "insights": insights,
            "diagnosis_issues": diagnosis.get("issues", [])
        }

    async def _generate_insights_from_introspection(self,
                                                    introspection_result,
                                                    diagnosis: Dict) -> List[str]:
        """Generate actionable insights from introspection results"""
        insights = list(introspection_result.insights)

        # Generate semantic insights for the bootstrap engine
        for anomaly in introspection_result.anomalies:
            insight = SemanticInsight(
                pattern=f"anomaly_detected_{anomaly}",
                confidence=0.8,
                context={"source": "introspection"},
                implications=["Investigate anomaly", "Consider system adjustment"],
                transformation_potential=0.4
            )
            await self.bootstrap_engine.process_semantic_insight(insight)

        # Process diagnosis issues
        for issue in diagnosis.get("issues", []):
            if issue.get("severity") == "high":
                insights.append(f"Critical issue: {issue.get('type')} in {issue.get('component', 'system')}")

        self.metrics.insights_generated += len(insights)
        return insights

    async def _pursue_goal(self, goal: Goal) -> Dict[str, Any]:
        """Pursue an active goal"""
        self.state = OracleZState.EXECUTING
        result = {"goal_id": goal.goal_id, "goal_name": goal.name}

        try:
            # Select strategy for this goal
            context = StrategyContext(
                task_domain=self._goal_type_to_domain(goal.goal_type),
                complexity=0.5 + goal.progress * 0.3,
                time_pressure=goal.urgency(),
                accuracy_requirement=0.7,
                resource_availability=0.8,
                uncertainty=1.0 - goal.confidence,
                prior_experience=goal.attempts / 10
            )

            strategy = self.strategy_optimizer.select_strategy(context)
            result["strategy"] = strategy.name

            # Simulate goal progress
            progress_delta = np.random.uniform(0.05, 0.2) * strategy.accuracy_potential
            new_progress = min(1.0, goal.progress + progress_delta)

            # Update goal
            self.goal_manager.update_progress(goal.goal_id, new_progress)
            result["progress_delta"] = progress_delta
            result["new_progress"] = new_progress

            # Record strategy outcome
            self.strategy_optimizer.report_outcome(
                strategy_name=strategy.name,
                context=context,
                success=progress_delta > 0.05,
                time_taken=np.random.uniform(0.1, 1.0),
                resources_used=np.random.uniform(0.2, 0.6),
                accuracy=progress_delta
            )

            if new_progress >= 1.0:
                self.metrics.goals_completed += 1
                result["completed"] = True

            self.metrics.successful_operations += 1

        except Exception as e:
            result["error"] = str(e)
            goal.attempts += 1

        self.state = OracleZState.IDLE
        return result

    def _goal_type_to_domain(self, goal_type: GoalType) -> TaskDomain:
        """Map goal type to task domain"""
        mapping = {
            GoalType.SURVIVAL: TaskDomain.DECISION_MAKING,
            GoalType.MAINTENANCE: TaskDomain.PLANNING,
            GoalType.ACHIEVEMENT: TaskDomain.PROBLEM_SOLVING,
            GoalType.EXPLORATION: TaskDomain.REASONING,
            GoalType.OPTIMIZATION: TaskDomain.REASONING,
            GoalType.ADAPTATION: TaskDomain.LEARNING,
            GoalType.SOCIAL: TaskDomain.DECISION_MAKING,
            GoalType.META: TaskDomain.REASONING
        }
        return mapping.get(goal_type, TaskDomain.REASONING)

    async def _check_improvement_opportunity(self) -> Optional[Dict[str, Any]]:
        """Check if self-improvement should be triggered"""
        # Check system coherence
        current_coherence = self.bootstrap_engine._evaluate_system_coherence()

        # Check if improvement is needed
        if current_coherence < 0.7 or self.step_count % 500 == 0:
            return await self._perform_self_improvement()

        return None

    async def _perform_self_improvement(self) -> Dict[str, Any]:
        """Perform a self-improvement cycle"""
        self.state = OracleZState.SELF_IMPROVING
        self.logger.info("Starting self-improvement cycle")

        result = {
            "cycle": self.metrics.improvement_cycles + 1,
            "improvements": []
        }

        try:
            # Run bootstrap self-improvement
            initial_coherence = self.bootstrap_engine._evaluate_system_coherence()

            await self.bootstrap_engine.recursive_self_improvement(
                cycles=min(self.config.max_improvement_cycles, 3)
            )

            final_coherence = self.bootstrap_engine._evaluate_system_coherence()

            improvement = final_coherence - initial_coherence
            result["coherence_improvement"] = improvement

            # Safety check
            if self.config.safety_bounds_enabled:
                if improvement < -self.config.degradation_threshold:
                    self.logger.warning("Improvement resulted in degradation, rolling back")
                    result["rollback"] = True
                    # In a real system, we would restore previous state here

            self.metrics.improvement_cycles += 1
            self.metrics.strategies_adapted += 1

            # Record learning episode
            episode = LearningEpisode(
                episode_id=f"improve_{self.metrics.improvement_cycles}",
                task=LearningTask(
                    task_id="self_improvement",
                    domain=LearningDomain.ABSTRACT,
                    complexity=0.8,
                    samples_required=100,
                    similarity_to_prior=0.5
                ),
                strategy_used=LearningStrategy.GRADIENT_BASED,
                initial_performance=initial_coherence,
                final_performance=final_coherence,
                samples_used=self.config.max_improvement_cycles,
                time_taken=1.0
            )
            self.meta_learner.record_episode(episode)

        except Exception as e:
            self.logger.error(f"Self-improvement error: {e}")
            result["error"] = str(e)

        self.state = OracleZState.IDLE
        return result

    async def _handle_error(self, error: Exception):
        """Handle errors with recovery"""
        self.logger.warning(f"Handling error: {error}")

        # Run diagnosis
        diagnosis = self.self_diagnosis.diagnose()

        # Create error recovery goal if not exists
        recovery_goal = self.goal_manager.create_goal(
            name=f"Recover from {type(error).__name__}",
            description=str(error),
            goal_type=GoalType.MAINTENANCE,
            priority=GoalPriority.CRITICAL
        )
        self.goal_manager.activate_goal(recovery_goal.goal_id)

        self.state = OracleZState.IDLE

    def estimate_confidence(self, prediction: float, domain: str = "general") -> ConfidenceEstimate:
        """Get calibrated confidence estimate for a prediction"""
        return self.confidence_calibrator.estimate_confidence(
            raw_score=prediction,
            domain=domain,
            use_calibration=True
        )

    def record_prediction_outcome(self, confidence: float, correct: bool, domain: str = "general"):
        """Record outcome for calibration"""
        self.confidence_calibrator.record_outcome(
            predicted_confidence=confidence,
            actual_outcome=correct,
            domain=domain
        )

    async def run(self, max_steps: int = 1000):
        """Run the agent for a specified number of steps"""
        self.logger.info(f"Starting ORACLE-Z run for {max_steps} steps")

        for _ in range(max_steps):
            await self.step()
            await asyncio.sleep(0.01)  # Small delay for async processing

        self.logger.info("ORACLE-Z run completed")
        return self.get_state_summary()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "metrics": {
                "total_operations": self.metrics.total_operations,
                "successful_operations": self.metrics.successful_operations,
                "failed_operations": self.metrics.failed_operations,
                "success_rate": (
                    self.metrics.successful_operations /
                    max(self.metrics.total_operations, 1)
                ),
                "improvement_cycles": self.metrics.improvement_cycles,
                "goals_completed": self.metrics.goals_completed,
                "insights_generated": self.metrics.insights_generated,
                "anomalies_detected": self.metrics.anomalies_detected,
                "uptime_seconds": self.metrics.uptime_seconds
            },
            "self_model": self.self_model.get_state_summary(),
            "goals": self.goal_manager.get_statistics(),
            "strategy_performance": self.strategy_optimizer.get_performance_summary(),
            "calibration": self.confidence_calibrator.get_calibration_summary(),
            "meta_learning": self.meta_learner.get_meta_learning_state(),
            "system_coherence": self.bootstrap_engine._evaluate_system_coherence()
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed diagnostic report"""
        return {
            "summary": self.get_state_summary(),
            "self_model_report": self.self_model.get_detailed_report(),
            "introspection_overview": self.introspection_engine.get_system_overview(),
            "goal_tree": self.goal_manager.get_goal_tree(),
            "learning_efficiency": self.meta_learner.analyze_learning_efficiency(),
            "evolved_system": self.bootstrap_engine.export_evolved_system()
        }

    def export_state(self, base_path: str):
        """Export all state to files"""
        import os
        os.makedirs(base_path, exist_ok=True)

        # Export each component
        self.self_model.export(f"{base_path}/self_model.json")
        self.strategy_optimizer.export_state(f"{base_path}/strategy_optimizer.json")
        self.confidence_calibrator.export_state(f"{base_path}/confidence_calibrator.json")
        self.introspection_engine.export_state(f"{base_path}/introspection.json")
        self.goal_manager.export_state(f"{base_path}/goal_manager.json")
        self.meta_learner.export_state(f"{base_path}/meta_learner.json")

        # Export summary
        with open(f"{base_path}/oracle_z_summary.json", 'w') as f:
            json.dump(self.get_state_summary(), f, indent=2)

        self.logger.info(f"State exported to {base_path}")


async def main():
    """Main entry point for running ORACLE-Z"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run agent
    config = OracleZConfig(
        auto_improve=True,
        improvement_threshold=0.1,
        max_improvement_cycles=5
    )

    agent = OracleZAgent(config)

    print("\n" + "="*60)
    print("ORACLE-Z METACOGNITIVE AGENT")
    print("="*60)

    # Run for demonstration
    await agent.run(max_steps=100)

    # Print summary
    summary = agent.get_state_summary()
    print("\n" + "="*60)
    print("ORACLE-Z SESSION SUMMARY")
    print("="*60)
    print(f"Agent ID: {summary['agent_id']}")
    print(f"Final State: {summary['state']}")
    print(f"\nMetrics:")
    for key, value in summary['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nSystem Coherence: {summary['system_coherence']:.3f}")
    print(f"Self-Model Confidence: {summary['self_model']['model_confidence']:.3f}")
    print(f"Calibration ECE: {summary['calibration']['ece']:.3f}")

    return agent


if __name__ == "__main__":
    asyncio.run(main())
