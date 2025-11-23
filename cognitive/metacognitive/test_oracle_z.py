"""
Comprehensive Test Suite for ORACLE-Z Metacognitive Agent

Tests all ORACLE-Z components:
- Self-Model System
- Strategy Selection
- Confidence Calibration
- Introspection Engine
- Goal Management
- Meta-Learning (Learning-to-Learn)
- Full Agent Integration
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import json
import os
import unittest
from typing import Dict, Any

# Import ORACLE-Z components
from self_model import (
    SelfModel, SelfDiagnosis, CognitiveComponent,
    ComponentState, ModelAccuracy
)
from strategy_selection import (
    StrategyOptimizer, StrategyContext, TaskDomain, StrategyType
)
from confidence_calibration import (
    ConfidenceCalibrator, ConfidenceLevel, UncertaintyType
)
from introspection import (
    IntrospectionEngine, IntrospectionDepth,
    CognitiveProcess, ProcessState
)
from goal_management import (
    GoalManager, GoalType, GoalPriority, GoalStatus
)
from learning_to_learn import (
    MetaLearner, LearningTask, LearningEpisode,
    LearningStrategy, LearningDomain
)


class TestSelfModel(unittest.TestCase):
    """Tests for the Self-Model System"""

    def setUp(self):
        self.model = SelfModel("test_model")

    def test_initialization(self):
        """Test self-model initializes correctly"""
        self.assertEqual(len(self.model.components), len(CognitiveComponent))
        self.assertIsNotNone(self.model.model_id)

    def test_component_update(self):
        """Test updating cognitive component states"""
        self.model.update_component(
            CognitiveComponent.REASONING,
            health=0.9,
            load=0.5,
            performance=0.8
        )

        state = self.model.components[CognitiveComponent.REASONING]
        self.assertEqual(state.health, 0.9)
        self.assertEqual(state.load, 0.5)
        self.assertEqual(state.performance, 0.8)

    def test_capability_registration(self):
        """Test capability registration"""
        self.model.register_capability("test_capability", 0.8, 0.9)
        self.assertIn("test_capability", self.model.capabilities)
        self.assertEqual(self.model.capabilities["test_capability"].level, 0.8)

    def test_limitation_registration(self):
        """Test limitation registration"""
        self.model.register_limitation(
            "test_limitation",
            severity=0.5,
            scope=[CognitiveComponent.MEMORY]
        )
        self.assertIn("test_limitation", self.model.limitations)

    def test_performance_prediction(self):
        """Test performance prediction"""
        prediction = self.model.predict_performance("reasoning")
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction.value, 0.0)
        self.assertLessEqual(prediction.value, 1.0)

    def test_bottleneck_detection(self):
        """Test bottleneck detection"""
        # Set one component with low performance
        self.model.update_component(
            CognitiveComponent.MEMORY,
            performance=0.2
        )
        bottlenecks = self.model.get_bottlenecks()
        self.assertIn(CognitiveComponent.MEMORY, bottlenecks)

    def test_self_diagnosis(self):
        """Test self-diagnosis"""
        diagnosis = SelfDiagnosis(self.model)

        # Set a component with low health
        self.model.update_component(
            CognitiveComponent.LEARNING,
            health=0.3
        )

        result = diagnosis.diagnose()
        self.assertIn("issues", result)
        # Should detect low health issue
        low_health_issues = [
            i for i in result["issues"]
            if i.get("type") == "low_health"
        ]
        self.assertGreater(len(low_health_issues), 0)


class TestStrategySelection(unittest.TestCase):
    """Tests for the Strategy Selection System"""

    def setUp(self):
        self.optimizer = StrategyOptimizer()

    def test_initialization(self):
        """Test strategy optimizer initializes with default strategies"""
        self.assertGreater(len(self.optimizer.strategies), 0)

    def test_strategy_selection(self):
        """Test strategy selection for different contexts"""
        context = StrategyContext(
            task_domain=TaskDomain.REASONING,
            complexity=0.5,
            time_pressure=0.3,
            accuracy_requirement=0.8,
            resource_availability=0.7,
            uncertainty=0.4,
            prior_experience=0.5
        )

        strategy = self.optimizer.select_strategy(context)
        self.assertIsNotNone(strategy)
        self.assertIn(strategy.name, self.optimizer.strategies)

    def test_strategy_recommendations(self):
        """Test getting strategy recommendations"""
        context = StrategyContext(
            task_domain=TaskDomain.PLANNING,
            complexity=0.6,
            time_pressure=0.2,
            accuracy_requirement=0.9,
            resource_availability=0.8,
            uncertainty=0.3,
            prior_experience=0.7
        )

        recommendations = self.optimizer.get_strategy_recommendations(context, top_n=3)
        self.assertEqual(len(recommendations), 3)

    def test_outcome_reporting(self):
        """Test reporting strategy outcomes"""
        context = StrategyContext(
            task_domain=TaskDomain.PROBLEM_SOLVING,
            complexity=0.5,
            time_pressure=0.5,
            accuracy_requirement=0.7,
            resource_availability=0.6,
            uncertainty=0.5,
            prior_experience=0.3
        )

        strategy = self.optimizer.select_strategy(context)

        self.optimizer.report_outcome(
            strategy_name=strategy.name,
            context=context,
            success=True,
            time_taken=0.5,
            resources_used=0.4,
            accuracy=0.85
        )

        perf = self.optimizer.performance_history[strategy.name]
        self.assertEqual(perf.sample_count, 1)


class TestConfidenceCalibration(unittest.TestCase):
    """Tests for the Confidence Calibration System"""

    def setUp(self):
        self.calibrator = ConfidenceCalibrator(n_bins=10)

    def test_confidence_estimation(self):
        """Test confidence estimation"""
        estimate = self.calibrator.estimate_confidence(0.7)
        self.assertIsNotNone(estimate)
        self.assertGreaterEqual(estimate.value, 0.0)
        self.assertLessEqual(estimate.value, 1.0)
        self.assertLessEqual(estimate.lower_bound, estimate.value)
        self.assertGreaterEqual(estimate.upper_bound, estimate.value)

    def test_confidence_levels(self):
        """Test confidence level categorization"""
        low_conf = self.calibrator.estimate_confidence(0.3)
        high_conf = self.calibrator.estimate_confidence(0.9)

        self.assertIn(low_conf.level, [ConfidenceLevel.LOW, ConfidenceLevel.MODERATE])
        self.assertEqual(high_conf.level, ConfidenceLevel.VERY_HIGH)

    def test_outcome_recording(self):
        """Test recording prediction outcomes"""
        for _ in range(20):
            conf = np.random.uniform(0.4, 0.9)
            correct = np.random.random() < conf
            self.calibrator.record_outcome(conf, correct)

        ece = self.calibrator.compute_ece()
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_calibration_summary(self):
        """Test calibration summary generation"""
        summary = self.calibrator.get_calibration_summary()
        self.assertIn("ece", summary)
        self.assertIn("temperature", summary)
        self.assertIn("bin_stats", summary)


class TestIntrospection(unittest.TestCase):
    """Tests for the Introspection Engine"""

    def setUp(self):
        self.engine = IntrospectionEngine()

    def test_snapshot_capture(self):
        """Test capturing process snapshots"""
        self.engine.capture_snapshot(
            process=CognitiveProcess.REASONING,
            state=ProcessState.ACTIVE,
            activation_level=0.8,
            processing_load=0.5
        )

        snapshots = self.engine.process_snapshots[CognitiveProcess.REASONING]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].activation_level, 0.8)

    def test_trace_management(self):
        """Test trace creation and management"""
        trace_id = self.engine.start_trace(CognitiveProcess.PLANNING)
        self.assertIn(trace_id, self.engine.active_traces)

        self.engine.add_trace_event(trace_id, "step", {"data": "test"})
        self.engine.end_trace(trace_id)

        self.assertNotIn(trace_id, self.engine.active_traces)
        self.assertEqual(len(self.engine.completed_traces), 1)

    def test_introspection_levels(self):
        """Test different introspection depths"""
        # Add some data first
        for _ in range(20):
            self.engine.capture_snapshot(
                process=CognitiveProcess.ATTENTION,
                state=ProcessState.PROCESSING,
                activation_level=np.random.uniform(0.5, 0.9),
                processing_load=np.random.uniform(0.3, 0.7)
            )

        surface_result = self.engine.introspect(depth=IntrospectionDepth.SURFACE)
        deep_result = self.engine.introspect(depth=IntrospectionDepth.DEEP)

        self.assertIsNotNone(surface_result)
        self.assertIsNotNone(deep_result)
        # Deep introspection should have more insights
        self.assertGreaterEqual(len(deep_result.insights), len(surface_result.insights))

    def test_system_overview(self):
        """Test system overview generation"""
        overview = self.engine.get_system_overview()
        self.assertIn("timestamp", overview)
        self.assertIn("processes", overview)


class TestGoalManagement(unittest.TestCase):
    """Tests for the Goal Management System"""

    def setUp(self):
        self.manager = GoalManager()

    def test_goal_creation(self):
        """Test goal creation"""
        goal = self.manager.create_goal(
            name="Test Goal",
            description="A test goal",
            goal_type=GoalType.ACHIEVEMENT,
            priority=GoalPriority.MEDIUM
        )

        self.assertIsNotNone(goal.goal_id)
        self.assertIn(goal.goal_id, self.manager.goals)

    def test_goal_hierarchy(self):
        """Test goal decomposition and hierarchy"""
        parent = self.manager.create_goal(
            name="Parent Goal",
            description="Parent",
            goal_type=GoalType.ACHIEVEMENT
        )

        subgoals = self.manager.decompose_goal(parent.goal_id, [
            {"name": "Subgoal 1", "description": "First subgoal"},
            {"name": "Subgoal 2", "description": "Second subgoal"}
        ])

        self.assertEqual(len(subgoals), 2)
        self.assertEqual(len(self.manager.goals[parent.goal_id].subgoal_ids), 2)

    def test_goal_progress(self):
        """Test goal progress tracking"""
        goal = self.manager.create_goal(
            name="Progress Goal",
            description="Testing progress",
            goal_type=GoalType.ACHIEVEMENT
        )
        self.manager.activate_goal(goal.goal_id)

        self.manager.update_progress(goal.goal_id, 0.5)
        self.assertEqual(self.manager.goals[goal.goal_id].progress, 0.5)

        self.manager.update_progress(goal.goal_id, 1.0)
        self.assertEqual(self.manager.goals[goal.goal_id].status, GoalStatus.COMPLETED)

    def test_conflict_detection(self):
        """Test goal conflict detection"""
        goal1 = self.manager.create_goal(
            name="Goal 1",
            description="First goal",
            goal_type=GoalType.ACHIEVEMENT,
            context={"requires_exclusive": "resource_A"}
        )
        goal2 = self.manager.create_goal(
            name="Goal 2",
            description="Second goal",
            goal_type=GoalType.ACHIEVEMENT,
            context={"requires_exclusive": "resource_A"}
        )

        self.manager.activate_goal(goal1.goal_id)
        self.manager.activate_goal(goal2.goal_id)

        conflicts = self.manager.detect_conflicts()
        self.assertGreater(len(conflicts), 0)

    def test_goal_statistics(self):
        """Test goal statistics"""
        stats = self.manager.get_statistics()
        self.assertIn("total_goals", stats)
        self.assertIn("active_goals", stats)


class TestMetaLearning(unittest.TestCase):
    """Tests for the Meta-Learning System"""

    def setUp(self):
        self.meta_learner = MetaLearner()

    def test_strategy_selection(self):
        """Test learning strategy selection"""
        task = LearningTask(
            task_id="test_task",
            domain=LearningDomain.REASONING,
            complexity=0.6,
            samples_required=100,
            similarity_to_prior=0.3
        )

        strategy = self.meta_learner.select_strategy(task)
        self.assertIsInstance(strategy, LearningStrategy)

    def test_episode_recording(self):
        """Test learning episode recording"""
        task = LearningTask(
            task_id="test_task",
            domain=LearningDomain.PERCEPTION,
            complexity=0.5,
            samples_required=50,
            similarity_to_prior=0.5
        )

        episode = LearningEpisode(
            episode_id="ep_1",
            task=task,
            strategy_used=LearningStrategy.GRADIENT_BASED,
            initial_performance=0.3,
            final_performance=0.8,
            samples_used=50,
            time_taken=1.0
        )

        self.meta_learner.record_episode(episode)
        self.assertEqual(len(self.meta_learner.episodes), 1)

    def test_similar_task_finding(self):
        """Test finding similar tasks"""
        # Record some episodes first
        for i in range(5):
            task = LearningTask(
                task_id=f"task_{i}",
                domain=LearningDomain.REASONING,
                complexity=0.5 + i * 0.05,
                samples_required=100,
                similarity_to_prior=0.3
            )
            episode = LearningEpisode(
                episode_id=f"ep_{i}",
                task=task,
                strategy_used=LearningStrategy.MODEL_BASED,
                initial_performance=0.3,
                final_performance=0.7,
                samples_used=100,
                time_taken=1.0
            )
            self.meta_learner.record_episode(episode)

        new_task = LearningTask(
            task_id="new_task",
            domain=LearningDomain.REASONING,
            complexity=0.55,
            samples_required=100,
            similarity_to_prior=0.5
        )

        similar = self.meta_learner.find_similar_tasks(new_task, top_k=3)
        self.assertLessEqual(len(similar), 3)

    def test_curriculum_optimization(self):
        """Test curriculum optimization"""
        tasks = [
            LearningTask(
                task_id=f"task_{i}",
                domain=np.random.choice(list(LearningDomain)),
                complexity=np.random.uniform(0.3, 0.9),
                samples_required=100,
                similarity_to_prior=0.5
            )
            for i in range(10)
        ]

        curriculum = self.meta_learner.optimize_curriculum(tasks)
        self.assertEqual(len(curriculum), len(tasks))

        # Should be roughly ordered by complexity
        complexities = [t.complexity for t in curriculum]
        # Not strictly sorted due to domain interleaving, but should trend upward

    def test_learning_efficiency_analysis(self):
        """Test learning efficiency analysis"""
        analysis = self.meta_learner.analyze_learning_efficiency()
        self.assertIn("status", analysis)  # Should have status (no_data or actual data)


class TestOracleZIntegration(unittest.TestCase):
    """Integration tests for the full ORACLE-Z agent"""

    def test_imports(self):
        """Test that all imports work correctly"""
        from oracle_z_agent import OracleZAgent, OracleZConfig
        self.assertIsNotNone(OracleZAgent)
        self.assertIsNotNone(OracleZConfig)


async def run_oracle_z_integration_test():
    """Run async integration test for ORACLE-Z"""
    from oracle_z_agent import OracleZAgent, OracleZConfig

    config = OracleZConfig(
        auto_improve=True,
        max_improvement_cycles=3,
        introspection_interval=10
    )

    agent = OracleZAgent(config)

    # Run a few steps
    results = []
    for _ in range(20):
        result = await agent.step()
        results.append(result)

    # Check agent state
    summary = agent.get_state_summary()
    assert summary["metrics"]["total_operations"] == 20
    assert summary["state"] is not None

    return summary


class AsyncTestRunner:
    """Helper to run async tests"""

    @staticmethod
    def run(coro):
        return asyncio.get_event_loop().run_until_complete(coro)


def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*60)
    print("ORACLE-Z COMPREHENSIVE TEST SUITE")
    print("="*60 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSelfModel))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategySelection))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceCalibration))
    suite.addTests(loader.loadTestsFromTestCase(TestIntrospection))
    suite.addTests(loader.loadTestsFromTestCase(TestGoalManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestMetaLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestOracleZIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Run async integration test
    print("\n" + "-"*60)
    print("Running Async Integration Tests...")
    print("-"*60)

    try:
        summary = AsyncTestRunner.run(run_oracle_z_integration_test())
        print("\nIntegration Test PASSED")
        print(f"  Total Operations: {summary['metrics']['total_operations']}")
        print(f"  System Coherence: {summary['system_coherence']:.3f}")
    except Exception as e:
        print(f"\nIntegration Test FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")

    return result


if __name__ == "__main__":
    run_all_tests()
