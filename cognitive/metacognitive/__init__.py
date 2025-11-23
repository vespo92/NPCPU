"""
NPCPU Metacognitive Module

This module provides the metacognitive capabilities for the NPCPU system,
enabling self-awareness, self-improvement, and recursive cognitive enhancement.

Main Components:
- ORACLE-Z Agent: The main metacognitive agent integrating all capabilities
- Self-Model: Internal representation of system state and capabilities
- Strategy Selection: Dynamic cognitive strategy optimization
- Confidence Calibration: Uncertainty estimation and calibration
- Introspection: Deep state inspection and pattern detection
- Goal Management: Dynamic goal hierarchy management
- Meta-Learning: Learning-to-learn capabilities
- Bootstrap Engine: Core bootstrapping and self-modification
- Recursive Improvement: Multi-generation self-enhancement
"""

# Core agent
from .oracle_z_agent import (
    OracleZAgent,
    OracleZConfig,
    OracleZState,
    OracleZMetrics
)

# Self-modeling
from .self_model import (
    SelfModel,
    SelfDiagnosis,
    CognitiveComponent,
    ComponentState,
    Capability,
    Limitation,
    ModelAccuracy,
    SelfModelPrediction
)

# Strategy selection
from .strategy_selection import (
    StrategyOptimizer,
    StrategyContext,
    CognitiveStrategy,
    StrategyType,
    TaskDomain,
    StrategyPerformance
)

# Confidence calibration
from .confidence_calibration import (
    ConfidenceCalibrator,
    ConfidenceEstimate,
    ConfidenceLevel,
    UncertaintyType,
    UncertaintyEstimator,
    DomainCalibrator
)

# Introspection
from .introspection import (
    IntrospectionEngine,
    IntrospectionDepth,
    IntrospectionResult,
    CognitiveProcess,
    ProcessState,
    ProcessSnapshot,
    CognitiveTrace
)

# Goal management
from .goal_management import (
    GoalManager,
    Goal,
    GoalType,
    GoalPriority,
    GoalStatus,
    GoalEvent
)

# Meta-learning
from .learning_to_learn import (
    MetaLearner,
    LearningTask,
    LearningEpisode,
    LearningStrategy,
    LearningDomain,
    LearningRate
)

# Bootstrap engine
from .bootstrap_engine import (
    MetaCognitiveBootstrap,
    SemanticInsight,
    DimensionalOperator,
    DimensionalOperatorType,
    TopologicalTransformation,
    ProjectionOperator
)

# Recursive improvement
from .recursive_improvement import (
    RecursiveImprovementEngine,
    CognitiveState,
    ImprovementStrategy
)

__all__ = [
    # Agent
    'OracleZAgent',
    'OracleZConfig',
    'OracleZState',
    'OracleZMetrics',

    # Self-model
    'SelfModel',
    'SelfDiagnosis',
    'CognitiveComponent',
    'ComponentState',
    'Capability',
    'Limitation',
    'ModelAccuracy',
    'SelfModelPrediction',

    # Strategy
    'StrategyOptimizer',
    'StrategyContext',
    'CognitiveStrategy',
    'StrategyType',
    'TaskDomain',
    'StrategyPerformance',

    # Confidence
    'ConfidenceCalibrator',
    'ConfidenceEstimate',
    'ConfidenceLevel',
    'UncertaintyType',
    'UncertaintyEstimator',
    'DomainCalibrator',

    # Introspection
    'IntrospectionEngine',
    'IntrospectionDepth',
    'IntrospectionResult',
    'CognitiveProcess',
    'ProcessState',
    'ProcessSnapshot',
    'CognitiveTrace',

    # Goals
    'GoalManager',
    'Goal',
    'GoalType',
    'GoalPriority',
    'GoalStatus',
    'GoalEvent',

    # Meta-learning
    'MetaLearner',
    'LearningTask',
    'LearningEpisode',
    'LearningStrategy',
    'LearningDomain',
    'LearningRate',

    # Bootstrap
    'MetaCognitiveBootstrap',
    'SemanticInsight',
    'DimensionalOperator',
    'DimensionalOperatorType',
    'TopologicalTransformation',
    'ProjectionOperator',

    # Improvement
    'RecursiveImprovementEngine',
    'CognitiveState',
    'ImprovementStrategy',
]

__version__ = '1.0.0'
__author__ = 'NPCPU Development Team'
