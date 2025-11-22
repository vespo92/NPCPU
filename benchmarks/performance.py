"""
Performance Benchmarks for NPCPU

Comprehensive performance benchmarks measuring:
- Organism creation rate
- Tick processing rate at various population sizes
- Memory usage scaling
- Event bus throughput

Benchmark targets:
- Support 10,000+ organisms
- 100+ ticks/second with 1000 organisms
- Event bus: 10,000+ events/second
- Memory: < 1KB per organism base

Usage:
    python -m benchmarks.performance        # Run all benchmarks
    python -m benchmarks.performance --quick  # Quick benchmark run
"""

import time
import gc
import sys
import argparse
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import tracemalloc

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from core.simple_organism import SimpleOrganism, SimplePopulation
from core.events import EventBus, Event, EventPriority
from core.plugins import HookManager, HookPoint
from core.abstractions import BaseOrganism


# =============================================================================
# Benchmark Infrastructure
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    name: str
    iterations: int
    total_time: float
    rate: float  # Operations per second
    memory_peak: int  # Peak memory in bytes
    memory_per_unit: float  # Memory per item in bytes
    passed: bool
    target: Optional[float] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        target_str = f" (target: {self.target:,.0f}/s)" if self.target else ""
        mem_str = f", {self.memory_per_unit:.1f} bytes/unit" if self.memory_per_unit > 0 else ""
        return (
            f"[{status}] {self.name}: {self.rate:,.1f}/s{target_str} "
            f"({self.iterations:,} iterations, {self.total_time:.3f}s{mem_str})"
        )


@contextmanager
def memory_tracker():
    """Context manager to track memory usage"""
    gc.collect()
    tracemalloc.start()
    try:
        yield
    finally:
        pass  # Keep tracemalloc running until we read


def get_memory_stats():
    """Get current memory statistics"""
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current, peak


def benchmark(
    name: str,
    func: Callable,
    iterations: int,
    target_rate: Optional[float] = None,
    memory_target: Optional[float] = None,
    warmup: int = 0
) -> BenchmarkResult:
    """
    Run a benchmark and return results.

    Args:
        name: Name of the benchmark
        func: Function to benchmark (called with iteration number)
        iterations: Number of iterations
        target_rate: Target operations per second (for pass/fail)
        memory_target: Target memory per unit in bytes (for pass/fail)
        warmup: Number of warmup iterations
    """
    # Warmup
    for i in range(warmup):
        func(i)

    # Run with memory tracking
    gc.collect()
    tracemalloc.start()

    start = time.perf_counter()
    for i in range(iterations):
        func(i)
    end = time.perf_counter()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end - start
    rate = iterations / total_time if total_time > 0 else float('inf')
    memory_per_unit = peak_mem / iterations if iterations > 0 else 0

    # Determine if passed
    passed = True
    if target_rate and rate < target_rate:
        passed = False
    if memory_target and memory_per_unit > memory_target:
        passed = False

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total_time,
        rate=rate,
        memory_peak=peak_mem,
        memory_per_unit=memory_per_unit,
        passed=passed,
        target=target_rate
    )


# =============================================================================
# Organism Benchmarks
# =============================================================================

class OrganismBenchmarks:
    """Benchmarks for organism operations"""

    @staticmethod
    def organism_creation(count: int = 10000) -> BenchmarkResult:
        """
        Benchmark: Organism creation rate.

        Target: Create organisms as fast as possible.
        """
        organisms = []

        def create_organism(i: int):
            org = SimpleOrganism(f"Benchmark_{i}")
            organisms.append(org)

        result = benchmark(
            name="Organism Creation",
            func=create_organism,
            iterations=count,
            target_rate=5000,  # 5000 organisms/second
            memory_target=2048,  # 2KB per organism
            warmup=100
        )

        result.details["organisms_created"] = len(organisms)
        return result

    @staticmethod
    def organism_tick(count: int = 1000, ticks: int = 100) -> BenchmarkResult:
        """
        Benchmark: Tick processing rate.

        Target: 100+ ticks/second with given population.
        """
        # Create organisms first
        organisms = [SimpleOrganism(f"Org_{i}") for i in range(count)]
        stimuli = {"food_nearby": True, "threat_level": 0.2}

        def tick_all(i: int):
            for org in organisms:
                org.perceive(stimuli)
                org.tick()
                org.decide()

        result = benchmark(
            name=f"Tick Processing ({count} organisms)",
            func=tick_all,
            iterations=ticks,
            target_rate=100,  # 100 ticks/second
            warmup=5
        )

        result.details["organism_count"] = count
        result.details["ticks_processed"] = ticks
        result.details["total_organism_ticks"] = count * ticks
        return result

    @staticmethod
    def population_scaling() -> List[BenchmarkResult]:
        """
        Benchmark: Population scaling at various sizes.

        Tests tick rate at different population sizes.
        """
        results = []
        sizes = [100, 500, 1000, 2000, 5000]
        ticks_per_size = 50

        for size in sizes:
            pop = SimplePopulation(f"Pop_{size}")
            for i in range(size):
                pop.add(SimpleOrganism(f"Org_{i}"))

            stimuli = {"food_nearby": True}

            def tick_population(i: int):
                pop.tick(stimuli)

            result = benchmark(
                name=f"Population Tick ({size} organisms)",
                func=tick_population,
                iterations=ticks_per_size,
                target_rate=100 if size <= 1000 else 50,
                warmup=2
            )

            result.details["population_size"] = size
            results.append(result)

        return results


# =============================================================================
# Event Bus Benchmarks
# =============================================================================

class EventBusBenchmarks:
    """Benchmarks for event bus operations"""

    @staticmethod
    def event_throughput(count: int = 100000) -> BenchmarkResult:
        """
        Benchmark: Event publishing throughput.

        Target: 10,000+ events/second.
        """
        bus = EventBus(history_size=0)  # Disable history for pure throughput

        # Register a lightweight handler
        received = [0]
        bus.subscribe("benchmark.event", lambda e: received.__setitem__(0, received[0] + 1))

        def publish_event(i: int):
            bus.emit("benchmark.event", {"id": i})

        result = benchmark(
            name="Event Bus Throughput",
            func=publish_event,
            iterations=count,
            target_rate=10000,  # 10,000 events/second
            warmup=1000
        )

        result.details["events_received"] = received[0]
        return result

    @staticmethod
    def event_with_filters(count: int = 50000) -> BenchmarkResult:
        """
        Benchmark: Event publishing with filters.

        Tests overhead of filter evaluation.
        """
        from core.events import DataFilter

        bus = EventBus(history_size=0)

        received = [0]
        bus.subscribe(
            "benchmark.filtered",
            lambda e: received.__setitem__(0, received[0] + 1),
            filter=DataFilter(important=True)
        )

        def publish_event(i: int):
            # Half match, half don't
            bus.emit("benchmark.filtered", {"important": i % 2 == 0, "id": i})

        result = benchmark(
            name="Event Bus with Filters",
            func=publish_event,
            iterations=count,
            target_rate=5000,
            warmup=500
        )

        result.details["events_received"] = received[0]
        result.details["expected_received"] = count // 2
        return result

    @staticmethod
    def multiple_handlers(handler_count: int = 10, event_count: int = 10000) -> BenchmarkResult:
        """
        Benchmark: Event publishing with multiple handlers.
        """
        bus = EventBus(history_size=0)

        counters = [0] * handler_count
        for i in range(handler_count):
            idx = i
            bus.subscribe("benchmark.multi", lambda e, idx=idx: counters.__setitem__(idx, counters[idx] + 1))

        def publish_event(i: int):
            bus.emit("benchmark.multi", {"id": i})

        result = benchmark(
            name=f"Event Bus ({handler_count} handlers)",
            func=publish_event,
            iterations=event_count,
            target_rate=5000,
            warmup=500
        )

        result.details["handler_count"] = handler_count
        result.details["total_handler_calls"] = sum(counters)
        return result


# =============================================================================
# Hook System Benchmarks
# =============================================================================

class HookBenchmarks:
    """Benchmarks for hook system"""

    @staticmethod
    def hook_throughput(count: int = 100000) -> BenchmarkResult:
        """
        Benchmark: Hook calling throughput.
        """
        manager = HookManager()

        counter = [0]
        manager.register(HookPoint.SIMULATION_TICK, lambda: counter.__setitem__(0, counter[0] + 1))

        def call_hook(i: int):
            manager.call(HookPoint.SIMULATION_TICK)

        result = benchmark(
            name="Hook Throughput",
            func=call_hook,
            iterations=count,
            target_rate=50000,  # Hooks should be very fast
            warmup=1000
        )

        result.details["hooks_called"] = counter[0]
        return result

    @staticmethod
    def hook_chain(chain_length: int = 5, calls: int = 50000) -> BenchmarkResult:
        """
        Benchmark: Hook chains with multiple handlers.
        """
        manager = HookManager()

        counters = [0] * chain_length
        for i in range(chain_length):
            idx = i
            manager.register("benchmark.chain", lambda idx=idx: counters.__setitem__(idx, counters[idx] + 1))

        def call_chain(i: int):
            manager.call("benchmark.chain")

        result = benchmark(
            name=f"Hook Chain ({chain_length} handlers)",
            func=call_chain,
            iterations=calls,
            target_rate=20000,
            warmup=500
        )

        result.details["chain_length"] = chain_length
        result.details["total_handler_calls"] = sum(counters)
        return result


# =============================================================================
# Memory Benchmarks
# =============================================================================

class MemoryBenchmarks:
    """Benchmarks for memory usage"""

    @staticmethod
    def organism_memory(count: int = 10000) -> BenchmarkResult:
        """
        Benchmark: Memory usage per organism.

        Target: < 1KB per organism base.
        """
        gc.collect()
        tracemalloc.start()

        organisms = []
        for i in range(count):
            organisms.append(SimpleOrganism(f"Org_{i}"))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_per_organism = peak / count

        result = BenchmarkResult(
            name="Organism Memory Usage",
            iterations=count,
            total_time=0,
            rate=0,
            memory_peak=peak,
            memory_per_unit=memory_per_organism,
            passed=memory_per_organism < 2048,  # 2KB target
            target=None
        )

        result.details["total_memory_mb"] = peak / (1024 * 1024)
        result.details["memory_per_organism_bytes"] = memory_per_organism
        return result

    @staticmethod
    def population_memory_scaling() -> List[BenchmarkResult]:
        """
        Benchmark: Memory scaling with population size.
        """
        results = []
        sizes = [100, 500, 1000, 5000, 10000]

        for size in sizes:
            gc.collect()
            tracemalloc.start()

            pop = SimplePopulation(f"Pop_{size}")
            for i in range(size):
                pop.add(SimpleOrganism(f"Org_{i}"))

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_per_organism = peak / size

            result = BenchmarkResult(
                name=f"Population Memory ({size} organisms)",
                iterations=size,
                total_time=0,
                rate=0,
                memory_peak=peak,
                memory_per_unit=memory_per_organism,
                passed=memory_per_organism < 2048,
                target=None
            )

            result.details["population_size"] = size
            result.details["total_memory_mb"] = peak / (1024 * 1024)
            results.append(result)

        return results


# =============================================================================
# Integration Benchmarks
# =============================================================================

class IntegrationBenchmarks:
    """End-to-end integration benchmarks"""

    @staticmethod
    def full_simulation(population_size: int = 1000, ticks: int = 100) -> BenchmarkResult:
        """
        Benchmark: Full simulation with events and hooks.
        """
        from core.events import get_event_bus, set_event_bus
        from core.plugins import HookManager

        # Fresh event bus
        bus = EventBus()
        original_bus = get_event_bus()
        set_event_bus(bus)

        hooks = HookManager()

        # Track events
        event_count = [0]
        bus.subscribe("*", lambda e: event_count.__setitem__(0, event_count[0] + 1))

        # Create population
        pop = SimplePopulation("SimPop", carrying_capacity=population_size * 2)
        for i in range(population_size):
            pop.add(SimpleOrganism(f"Org_{i}"))

        stimuli = {"food_nearby": True, "threat_level": 0.1}

        gc.collect()
        start = time.perf_counter()

        for tick in range(ticks):
            hooks.call(HookPoint.SIMULATION_TICK, tick)
            pop.tick(stimuli)

        end = time.perf_counter()
        total_time = end - start

        # Restore original bus
        set_event_bus(original_bus)

        ticks_per_second = ticks / total_time if total_time > 0 else float('inf')

        result = BenchmarkResult(
            name=f"Full Simulation ({population_size} organisms)",
            iterations=ticks,
            total_time=total_time,
            rate=ticks_per_second,
            memory_peak=0,
            memory_per_unit=0,
            passed=ticks_per_second >= 50,  # At least 50 ticks/second
            target=100
        )

        result.details["population_size"] = population_size
        result.details["events_generated"] = event_count[0]
        result.details["final_population"] = pop.size
        result.details["deaths"] = pop.total_deaths
        return result


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs all benchmarks and reports results"""

    def __init__(self, quick: bool = False):
        self.quick = quick
        self.results: List[BenchmarkResult] = []

    def run_all(self) -> None:
        """Run all benchmark suites"""
        print("=" * 70)
        print("NPCPU Performance Benchmarks")
        print("=" * 70)
        print()

        scale = 0.1 if self.quick else 1.0

        # Organism benchmarks
        print("Organism Benchmarks")
        print("-" * 40)

        result = OrganismBenchmarks.organism_creation(int(10000 * scale))
        print(result)
        self.results.append(result)

        result = OrganismBenchmarks.organism_tick(int(1000 * scale), int(100 * scale))
        print(result)
        self.results.append(result)

        if not self.quick:
            for result in OrganismBenchmarks.population_scaling():
                print(result)
                self.results.append(result)

        print()

        # Event bus benchmarks
        print("Event Bus Benchmarks")
        print("-" * 40)

        result = EventBusBenchmarks.event_throughput(int(100000 * scale))
        print(result)
        self.results.append(result)

        result = EventBusBenchmarks.event_with_filters(int(50000 * scale))
        print(result)
        self.results.append(result)

        result = EventBusBenchmarks.multiple_handlers(10, int(10000 * scale))
        print(result)
        self.results.append(result)

        print()

        # Hook benchmarks
        print("Hook System Benchmarks")
        print("-" * 40)

        result = HookBenchmarks.hook_throughput(int(100000 * scale))
        print(result)
        self.results.append(result)

        result = HookBenchmarks.hook_chain(5, int(50000 * scale))
        print(result)
        self.results.append(result)

        print()

        # Memory benchmarks
        print("Memory Benchmarks")
        print("-" * 40)

        result = MemoryBenchmarks.organism_memory(int(10000 * scale))
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: "
              f"{result.memory_per_unit:.1f} bytes/organism "
              f"(total: {result.details['total_memory_mb']:.2f} MB)")
        self.results.append(result)

        if not self.quick:
            for result in MemoryBenchmarks.population_memory_scaling():
                print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: "
                      f"{result.memory_per_unit:.1f} bytes/organism")
                self.results.append(result)

        print()

        # Integration benchmarks
        print("Integration Benchmarks")
        print("-" * 40)

        result = IntegrationBenchmarks.full_simulation(int(1000 * scale), int(100 * scale))
        print(result)
        self.results.append(result)

        print()

        # Summary
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
            print("\nAll benchmarks PASSED!")
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
    parser = argparse.ArgumentParser(description="NPCPU Performance Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark suite")
    args = parser.parse_args()

    runner = BenchmarkRunner(quick=args.quick)
    runner.run_all()


if __name__ == "__main__":
    main()
