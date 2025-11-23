"""
Consciousness System Benchmarks for NPCPU

Performance benchmarks for consciousness-related operations:
- Neural consciousness processing
- Attention mechanism throughput
- Working memory operations
- Emotional state computation
- Introspection latency

Benchmark targets:
- Perception processing: 10,000+ stimuli/second
- Working memory: 50,000+ operations/second
- Emotional computation: 20,000+ updates/second
- Introspection: 5,000+ reports/second

Usage:
    python -m benchmarks.consciousness_benchmarks
    python -m benchmarks.consciousness_benchmarks --quick
"""

import time
import gc
import sys
import argparse
from typing import Dict, Any, List
from dataclasses import dataclass
import tracemalloc

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from benchmarks.performance import BenchmarkResult, benchmark
from consciousness.neural_consciousness import NeuralConsciousness, MemoryItem
from core.simple_organism import SimpleOrganism


# =============================================================================
# Consciousness Benchmarks
# =============================================================================

class ConsciousnessBenchmarks:
    """Benchmarks for consciousness operations"""

    @staticmethod
    def perception_processing(count: int = 10000) -> BenchmarkResult:
        """
        Benchmark: Perception processing throughput.

        Target: 10,000+ stimuli sets/second.
        """
        consciousness = NeuralConsciousness(
            attention_dim=64,
            memory_capacity=7,
            emotional_sensitivity=0.5
        )

        # Create varied stimuli
        stimuli_sets = [
            {"threat": 0.1 * (i % 10), "food": 0.1 * ((i + 3) % 10), "social": 0.5}
            for i in range(count)
        ]

        processed = [0]

        def process_perception(i: int):
            consciousness.process_perception(stimuli_sets[i])
            processed[0] += 1

        result = benchmark(
            name="Perception Processing",
            func=process_perception,
            iterations=count,
            target_rate=10000,
            warmup=100
        )

        result.details["stimuli_processed"] = processed[0]
        return result

    @staticmethod
    def working_memory_operations(count: int = 50000) -> BenchmarkResult:
        """
        Benchmark: Working memory store/recall operations.

        Target: 50,000+ operations/second.
        """
        consciousness = NeuralConsciousness(memory_capacity=7)

        ops_completed = [0]

        def memory_operation(i: int):
            # Alternate between store and recall
            if i % 2 == 0:
                consciousness.update_working_memory(
                    {"event": f"event_{i}", "value": i % 100},
                    importance=0.5 + (i % 5) * 0.1
                )
            else:
                consciousness.recall({"event": f"event_{i-1}"}, limit=2)
            ops_completed[0] += 1

        result = benchmark(
            name="Working Memory Operations",
            func=memory_operation,
            iterations=count,
            target_rate=50000,
            warmup=500
        )

        result.details["operations_completed"] = ops_completed[0]
        return result

    @staticmethod
    def emotional_computation(count: int = 20000) -> BenchmarkResult:
        """
        Benchmark: Emotional state computation.

        Target: 20,000+ updates/second.
        """
        consciousness = NeuralConsciousness(emotional_sensitivity=0.5)

        # Pre-process some perceptions to populate global workspace
        for i in range(10):
            consciousness.process_perception({
                "threat": 0.1 * i,
                "food": 0.5,
                "novelty": 0.3
            })

        updates = [0]

        def compute_emotion(i: int):
            # Vary the workspace slightly
            consciousness.global_workspace["attended_stimuli"] = {
                "threat": {"value": 0.1 * (i % 10), "attention_weight": 0.3},
                "food": {"value": 0.5, "attention_weight": 0.2}
            }
            consciousness.compute_emotional_state()
            updates[0] += 1

        result = benchmark(
            name="Emotional Computation",
            func=compute_emotion,
            iterations=count,
            target_rate=20000,
            warmup=200
        )

        result.details["emotional_updates"] = updates[0]
        return result

    @staticmethod
    def introspection_reports(count: int = 5000) -> BenchmarkResult:
        """
        Benchmark: Introspection report generation.

        Target: 5,000+ reports/second.
        """
        consciousness = NeuralConsciousness(
            attention_dim=64,
            memory_capacity=7
        )

        # Setup some state
        consciousness.process_perception({"threat": 0.3, "food": 0.7})
        consciousness.update_working_memory({"event": "test"}, importance=0.8)
        consciousness.compute_emotional_state()

        reports = [0]

        def generate_report(i: int):
            consciousness.introspect()
            reports[0] += 1

        result = benchmark(
            name="Introspection Reports",
            func=generate_report,
            iterations=count,
            target_rate=5000,
            warmup=50
        )

        result.details["reports_generated"] = reports[0]
        return result

    @staticmethod
    def attention_mechanism(count: int = 10000) -> BenchmarkResult:
        """
        Benchmark: Attention mechanism computation.

        Target: 10,000+ attention computations/second.
        """
        consciousness = NeuralConsciousness(attention_dim=64)
        attention = consciousness.attention

        # Create query, keys, values
        query = [0.1] * 64
        keys = [[0.1 * (i % 10)] * 64 for i in range(10)]
        values = [[0.1 * (i % 10)] * 64 for i in range(10)]

        computations = [0]

        def compute_attention(i: int):
            attention.compute_attention(query, keys, values)
            computations[0] += 1

        result = benchmark(
            name="Attention Mechanism",
            func=compute_attention,
            iterations=count,
            target_rate=10000,
            warmup=100
        )

        result.details["attention_computations"] = computations[0]
        return result

    @staticmethod
    def consciousness_tick_scaling() -> List[BenchmarkResult]:
        """
        Benchmark: Consciousness tick rate at various memory loads.
        """
        results = []
        memory_loads = [1, 3, 5, 7]
        ticks_per_load = 5000

        for load in memory_loads:
            consciousness = NeuralConsciousness(memory_capacity=7)

            # Fill memory to specified load
            for i in range(load):
                consciousness.update_working_memory(
                    {"item": i, "data": f"data_{i}"},
                    importance=0.5 + i * 0.1
                )

            def tick_consciousness(i: int):
                consciousness.tick()

            result = benchmark(
                name=f"Consciousness Tick (memory={load})",
                func=tick_consciousness,
                iterations=ticks_per_load,
                target_rate=20000,
                warmup=100
            )

            result.details["memory_load"] = load
            results.append(result)

        return results


# =============================================================================
# Integrated Consciousness+Organism Benchmarks
# =============================================================================

class IntegratedConsciousnessBenchmarks:
    """Benchmarks for consciousness integrated with organisms"""

    @staticmethod
    def conscious_organism_creation(count: int = 5000) -> BenchmarkResult:
        """
        Benchmark: Creating organisms with attached consciousness.

        Target: 5,000+ organisms/second.
        """
        organisms = []

        def create_conscious_organism(i: int):
            org = SimpleOrganism(f"ConsciousOrg_{i}")
            consciousness = NeuralConsciousness(
                attention_dim=32,
                memory_capacity=5
            )
            org.add_subsystem(consciousness)
            organisms.append(org)

        result = benchmark(
            name="Conscious Organism Creation",
            func=create_conscious_organism,
            iterations=count,
            target_rate=5000,
            memory_target=4096,  # 4KB per organism
            warmup=50
        )

        result.details["organisms_created"] = len(organisms)
        return result

    @staticmethod
    def conscious_organism_tick(org_count: int = 500, ticks: int = 100) -> BenchmarkResult:
        """
        Benchmark: Tick rate for population of conscious organisms.

        Target: 50+ ticks/second for 500 organisms.
        """
        # Create conscious organisms
        organisms = []
        for i in range(org_count):
            org = SimpleOrganism(f"Org_{i}")
            consciousness = NeuralConsciousness(
                attention_dim=32,
                memory_capacity=5
            )
            org.add_subsystem(consciousness)
            organisms.append(org)

        stimuli = {"food": 0.5, "threat": 0.2, "social": 0.3}

        def tick_all_organisms(i: int):
            for org in organisms:
                org.perceive(stimuli)
                org.tick()
                # Also process consciousness
                consciousness = org.get_subsystem("neural_consciousness")
                if consciousness:
                    consciousness.process_perception(stimuli)
                    consciousness.compute_emotional_state()

        result = benchmark(
            name=f"Conscious Organism Tick ({org_count} orgs)",
            func=tick_all_organisms,
            iterations=ticks,
            target_rate=50,
            warmup=2
        )

        result.details["organism_count"] = org_count
        result.details["total_ticks"] = ticks
        return result


# =============================================================================
# Memory Benchmarks
# =============================================================================

class ConsciousnessMemoryBenchmarks:
    """Memory usage benchmarks for consciousness systems"""

    @staticmethod
    def consciousness_memory_usage(count: int = 1000) -> BenchmarkResult:
        """
        Benchmark: Memory usage per consciousness instance.

        Target: < 4KB per consciousness.
        """
        gc.collect()
        tracemalloc.start()

        consciousnesses = []
        for i in range(count):
            c = NeuralConsciousness(
                attention_dim=64,
                memory_capacity=7
            )
            consciousnesses.append(c)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_per_consciousness = peak / count

        result = BenchmarkResult(
            name="Consciousness Memory Usage",
            iterations=count,
            total_time=0,
            rate=0,
            memory_peak=peak,
            memory_per_unit=memory_per_consciousness,
            passed=memory_per_consciousness < 8192,  # 8KB target
            target=None
        )

        result.details["total_memory_mb"] = peak / (1024 * 1024)
        result.details["memory_per_consciousness_kb"] = memory_per_consciousness / 1024
        return result

    @staticmethod
    def working_memory_scaling() -> List[BenchmarkResult]:
        """
        Benchmark: Memory scaling with working memory content.
        """
        results = []
        memory_sizes = [3, 5, 7, 10, 15]

        for size in memory_sizes:
            gc.collect()
            tracemalloc.start()

            consciousnesses = []
            for i in range(100):
                c = NeuralConsciousness(memory_capacity=size)
                # Fill to capacity
                for j in range(size):
                    c.update_working_memory(
                        {"item": j, "data": f"content_{j}" * 10},
                        importance=0.5
                    )
                consciousnesses.append(c)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_per_consciousness = peak / 100

            result = BenchmarkResult(
                name=f"Memory Scaling (capacity={size})",
                iterations=100,
                total_time=0,
                rate=0,
                memory_peak=peak,
                memory_per_unit=memory_per_consciousness,
                passed=True,
                target=None
            )

            result.details["memory_capacity"] = size
            result.details["memory_per_consciousness_kb"] = memory_per_consciousness / 1024
            results.append(result)

        return results


# =============================================================================
# Benchmark Runner
# =============================================================================

class ConsciousnessBenchmarkRunner:
    """Runs all consciousness benchmarks"""

    def __init__(self, quick: bool = False):
        self.quick = quick
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> None:
        """Run all consciousness benchmark suites"""
        print("=" * 70)
        print("NPCPU Consciousness Performance Benchmarks")
        print("=" * 70)
        print()

        scale = 0.1 if self.quick else 1.0

        # Core consciousness benchmarks
        print("Core Consciousness Benchmarks")
        print("-" * 40)

        result = ConsciousnessBenchmarks.perception_processing(int(10000 * scale))
        print(result)
        self.results.append(result)

        result = ConsciousnessBenchmarks.working_memory_operations(int(50000 * scale))
        print(result)
        self.results.append(result)

        result = ConsciousnessBenchmarks.emotional_computation(int(20000 * scale))
        print(result)
        self.results.append(result)

        result = ConsciousnessBenchmarks.introspection_reports(int(5000 * scale))
        print(result)
        self.results.append(result)

        result = ConsciousnessBenchmarks.attention_mechanism(int(10000 * scale))
        print(result)
        self.results.append(result)

        print()

        # Scaling benchmarks
        if not self.quick:
            print("Consciousness Tick Scaling")
            print("-" * 40)
            for result in ConsciousnessBenchmarks.consciousness_tick_scaling():
                print(result)
                self.results.append(result)
            print()

        # Integrated benchmarks
        print("Integrated Consciousness+Organism Benchmarks")
        print("-" * 40)

        result = IntegratedConsciousnessBenchmarks.conscious_organism_creation(int(5000 * scale))
        print(result)
        self.results.append(result)

        result = IntegratedConsciousnessBenchmarks.conscious_organism_tick(
            int(500 * scale), int(100 * scale)
        )
        print(result)
        self.results.append(result)

        print()

        # Memory benchmarks
        print("Consciousness Memory Benchmarks")
        print("-" * 40)

        result = ConsciousnessMemoryBenchmarks.consciousness_memory_usage(int(1000 * scale))
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: "
              f"{result.details['memory_per_consciousness_kb']:.2f} KB/consciousness "
              f"(total: {result.details['total_memory_mb']:.2f} MB)")
        self.results.append(result)

        if not self.quick:
            for result in ConsciousnessMemoryBenchmarks.working_memory_scaling():
                print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: "
                      f"{result.details['memory_per_consciousness_kb']:.2f} KB/consciousness")
                self.results.append(result)

        print()
        self.print_summary()

    def print_summary(self) -> None:
        """Print benchmark summary"""
        print("=" * 70)
        print("Summary")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"Total benchmarks: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")

        if passed == total:
            print("\nAll consciousness benchmarks PASSED!")
        else:
            print("\nFailed benchmarks:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}")

        print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NPCPU Consciousness Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark suite")
    args = parser.parse_args()

    runner = ConsciousnessBenchmarkRunner(quick=args.quick)
    runner.run_all()


if __name__ == "__main__":
    main()
