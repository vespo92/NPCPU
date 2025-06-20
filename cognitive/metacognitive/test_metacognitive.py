"""
Test Suite and Demonstration for Meta-Cognitive Bootstrapping

This module provides comprehensive tests and demonstrations of the NPCPU's
meta-cognitive capabilities, including self-modification and recursive improvement.
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from bootstrap_engine import MetaCognitiveBootstrap, SemanticInsight
from dimensional_operators import create_operator_suite
from topological_engine import AdaptiveTopologyEngine, KnowledgeStructure, TopologyType
from recursive_improvement import RecursiveImprovementEngine, CognitiveState


class MetaCognitiveTestSuite:
    """Comprehensive test suite for meta-cognitive capabilities"""
    
    def __init__(self):
        self.results = {}
        self.visualizations = []
        
    async def test_dimensional_operators(self):
        """Test self-modifying dimensional operators"""
        print("\n=== Testing Dimensional Operators ===")
        
        # Create operator suite
        operators = create_operator_suite()
        
        # Test data
        test_manifold = np.random.randn(10, 100)
        
        results = {}
        
        for name, operator in operators.items():
            print(f"\nTesting {name} operator...")
            
            # Test basic operation
            output = operator.operate(test_manifold)
            results[name] = {
                "input_shape": test_manifold.shape,
                "output_shape": output.shape,
                "performance": operator.performance_metrics
            }
            
            # Test self-modification
            insight = SemanticInsight(
                pattern=f"optimize_{name}_performance",
                confidence=0.85,
                context={"operator": name},
                implications=["Improve efficiency"],
                transformation_potential=0.3
            )
            
            candidates = operator.generate_modification_candidates(insight)
            results[name]["modification_candidates"] = len(candidates)
            
            print(f"  Input shape: {test_manifold.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Modification candidates: {len(candidates)}")
            
        self.results["dimensional_operators"] = results
        
    async def test_topological_transformations(self):
        """Test topological transformation engine"""
        print("\n=== Testing Topological Transformations ===")
        
        engine = AdaptiveTopologyEngine()
        
        # Create test knowledge structure
        data = np.random.randn(20, 200)
        structure = KnowledgeStructure(
            data=data,
            topology_type=TopologyType.EUCLIDEAN
        )
        
        print(f"Original topology: {structure.topology_type.value}")
        print(f"Original complexity: {structure.compute_complexity():.3f}")
        print(f"Original persistence: {structure.compute_persistence():.3f}")
        
        results = {}
        
        # Test each transformation
        for name, transformation in engine.transformations.items():
            print(f"\nTesting {name} transformation...")
            
            transformed = transformation.transform(structure)
            preservation = transformation.evaluate_preservation(structure, transformed)
            
            results[name] = {
                "output_topology": transformed.topology_type.value,
                "output_shape": transformed.data.shape,
                "complexity": transformed.compute_complexity(),
                "persistence": transformed.compute_persistence(),
                "preservation": preservation
            }
            
            print(f"  Output topology: {transformed.topology_type.value}")
            print(f"  Output shape: {transformed.data.shape}")
            print(f"  Preservation scores: {preservation}")
            
        self.results["topological_transformations"] = results
        
    async def test_meta_cognitive_bootstrap(self):
        """Test meta-cognitive bootstrapping"""
        print("\n=== Testing Meta-Cognitive Bootstrap ===")
        
        bootstrap = MetaCognitiveBootstrap()
        
        # Run self-improvement cycles
        print("\nRunning recursive self-improvement...")
        await bootstrap.recursive_self_improvement(cycles=5)
        
        # Export evolved system
        evolved = bootstrap.export_evolved_system()
        
        print("\nEvolved System Summary:")
        print(f"  Self-improvement cycles: {evolved['self_improvement_cycles']}")
        print(f"  System coherence: {evolved['system_coherence']:.3f}")
        print("\n  Evolved operators:")
        
        for op_name, op_info in evolved['operators'].items():
            print(f"    {op_name}:")
            print(f"      Version: {op_info['version']}")
            print(f"      Modifications: {op_info['modifications']}")
            print(f"      Performance: {op_info['performance']}")
            
        self.results["meta_cognitive_bootstrap"] = evolved
        
    async def test_recursive_improvement(self):
        """Test full recursive improvement system"""
        print("\n=== Testing Recursive Improvement Engine ===")
        
        engine = RecursiveImprovementEngine()
        
        # Run recursive improvement
        print("\nRunning recursive improvement...")
        results = await engine.run_recursive_improvement(
            target_generations=5,
            target_fitness=0.8
        )
        
        print("\nImprovement Results:")
        print(f"  Initial fitness: {results['initial_fitness']:.3f}")
        print(f"  Final fitness: {results['final_fitness']:.3f}")
        print(f"  Improvement: {results['improvement']:.3f}")
        print(f"  Generations: {results['generations']}")
        
        print("\n  Final cognitive state:")
        for metric, value in results['final_state'].items():
            print(f"    {metric}: {value:.3f}")
            
        print("\n  Knowledge topologies:")
        for name, topology in results['knowledge_topologies'].items():
            print(f"    {name}: {topology}")
            
        self.results["recursive_improvement"] = results
        
        # Export improved system
        export_path = "improved_system_config.json"
        engine.export_improved_system(export_path)
        print(f"\nExported improved system to: {export_path}")
        
    async def test_semantic_insight_processing(self):
        """Test semantic insight generation and processing"""
        print("\n=== Testing Semantic Insight Processing ===")
        
        bootstrap = MetaCognitiveBootstrap()
        
        # Generate various insights
        insights = [
            SemanticInsight(
                pattern="high_dimensional_complexity",
                confidence=0.9,
                context={"dimensions": 100, "samples": 1000},
                implications=["Reduce dimensions", "Apply manifold learning"],
                transformation_potential=0.7
            ),
            SemanticInsight(
                pattern="operator_inefficiency_projection",
                confidence=0.85,
                context={"operator": "projection", "efficiency": 0.6},
                implications=["Optimize projection algorithm", "Use adaptive methods"],
                transformation_potential=0.5
            ),
            SemanticInsight(
                pattern=<7>temporal_pattern_detected",
                confidence=0.8,
                context={"frequency": 10.5, "phase": 0.3},
                implications=["Add temporal operators", "Implement phase coupling"],
                transformation_potential=0.6
            )
        ]
        
        results = {}
        
        for insight in insights:
            print(f"\nProcessing insight: {insight.pattern}")
            print(f"  Confidence: {insight.confidence}")
            print(f"  Transformation potential: {insight.transformation_potential}")
            
            # Process insight
            await bootstrap.process_semantic_insight(insight)
            
            # Check for modifications
            modified_operators = []
            for op_name, operator in bootstrap.operators.items():
                if len(operator.modification_history) > 0:
                    modified_operators.append(op_name)
                    
            results[insight.pattern] = {
                "processed": True,
                "modified_operators": modified_operators
            }
            
            print(f"  Modified operators: {modified_operators}")
            
        self.results["semantic_insights"] = results
        
    def visualize_results(self):
        """Create visualizations of test results"""
        print("\n=== Creating Visualizations ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Meta-Cognitive Bootstrapping Test Results", fontsize=16)
        
        # 1. Operator performance comparison
        ax = axes[0, 0]
        if "dimensional_operators" in self.results:
            operators = list(self.results["dimensional_operators"].keys())
            efficiencies = [
                self.results["dimensional_operators"][op]["performance"]["efficiency"]
                for op in operators
            ]
            
            ax.bar(operators, efficiencies)
            ax.set_title("Operator Efficiency Comparison")
            ax.set_ylabel("Efficiency")
            ax.set_xticklabels(operators, rotation=45)
            
        # 2. Topological transformation preservation
        ax = axes[0, 1]
        if "topological_transformations" in self.results:
            transformations = list(self.results["topological_transformations"].keys())
            semantic_preservation = [
                self.results["topological_transformations"][t]["preservation"].get("semantic", 0)
                for t in transformations
            ]
            
            ax.bar(transformations, semantic_preservation)
            ax.set_title("Transformation Semantic Preservation")
            ax.set_ylabel("Preservation Score")
            ax.set_xticklabels(transformations, rotation=45)
            
        # 3. Recursive improvement progress
        ax = axes[1, 0]
        if "recursive_improvement" in self.results:
            history = self.results["recursive_improvement"].get("improvement_history", [])
            if history:
                generations = [h["generation"] for h in history]
                fitness = [h["final_fitness"] for h in history]
                
                ax.plot(generations, fitness, 'o-')
                ax.set_title("Fitness Evolution")
                ax.set_xlabel("Generation")
                ax.set_ylabel("Fitness")
                ax.grid(True)
                
        # 4. Cognitive state metrics
        ax = axes[1, 1]
        if "recursive_improvement" in self.results:
            final_state = self.results["recursive_improvement"]["final_state"]
            metrics = list(final_state.keys())
            values = list(final_state.values())
            
            ax.bar(metrics, values)
            ax.set_title("Final Cognitive State")
            ax.set_ylabel("Value")
            ax.set_ylim(0, 1)
            
        plt.tight_layout()
        
        # Save visualization
        viz_path = "metacognitive_test_results.png"
        plt.savefig(viz_path)
        print(f"Saved visualization to: {viz_path}")
        
        self.visualizations.append(viz_path)
        
    async def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*50)
        print("NPCPU META-COGNITIVE BOOTSTRAPPING TEST SUITE")
        print("="*50)
        
        start_time = datetime.now()
        
        # Run individual tests
        await self.test_dimensional_operators()
        await self.test_topological_transformations()
        await self.test_semantic_insight_processing()
        await self.test_meta_cognitive_bootstrap()
        await self.test_recursive_improvement()
        
        # Create visualizations
        self.visualize_results()
        
        # Generate test report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "test_suite": "Meta-Cognitive Bootstrapping",
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "tests_completed": len(self.results),
            "results": self.results,
            "visualizations": self.visualizations
        }
        
        # Save report
        report_path = "metacognitive_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print("\n" + "="*50)
        print("TEST SUITE COMPLETED")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Report saved to: {report_path}")
        print("="*50)
        
        return report


async def main():
    """Main entry point for testing"""
    # Create and run test suite
    test_suite = MetaCognitiveTestSuite()
    report = await test_suite.run_all_tests()
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    print(f"Total tests: {report['tests_completed']}")
    print(f"Duration: {report['duration_seconds']:.2f} seconds")
    
    # Check for recursive improvement success
    if "recursive_improvement" in report["results"]:
        improvement = report["results"]["recursive_improvement"]["improvement"]
        if improvement > 0:
            print(f"\n✓ Recursive self-improvement successful! Fitness improved by {improvement:.3f}")
        else:
            print(f"\n✗ Recursive self-improvement needs tuning. Fitness change: {improvement:.3f}")


if __name__ == "__main__":
    asyncio.run(main())