"""
Parallel Digital Life Simulation Runner

An enhanced simulation runner that leverages parallel processing
for improved performance with large populations.

Features:
- Async organism processing with configurable worker pools
- Batch processing for organism updates
- Concurrent event processing
- Performance metrics and profiling
- Graceful degradation when async not available

Usage:
    from simulation.parallel_runner import ParallelSimulation, ParallelConfig

    config = ParallelConfig(
        name="Parallel Evolution",
        initial_population=1000,
        max_workers=4,
        batch_size=100
    )

    sim = ParallelSimulation(config)
    sim.run()
"""

import time
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from functools import lru_cache
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from organism.digital_body import DigitalBody, OrganismState
from organism.metabolism import ResourceType
from ecosystem.population import Population
from ecosystem.world import World, WorldConfig
from simulation.runner import SimulationConfig, SimulationMetrics, SimulationState


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ParallelConfig(SimulationConfig):
    """Configuration for parallel simulation"""
    max_workers: int = 4              # Number of worker threads/processes
    batch_size: int = 100             # Organisms per batch
    use_processes: bool = False       # Use processes instead of threads
    enable_profiling: bool = False    # Enable performance profiling
    async_events: bool = True         # Process events asynchronously
    cache_consciousness: bool = True  # Enable consciousness caching


@dataclass
class PerformanceMetrics:
    """Performance tracking for parallel simulation"""
    total_ticks: int = 0
    tick_times: List[float] = field(default_factory=list)
    batch_times: List[float] = field(default_factory=list)
    organisms_processed: int = 0
    events_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def avg_tick_time(self) -> float:
        return sum(self.tick_times[-100:]) / len(self.tick_times[-100:]) if self.tick_times else 0

    @property
    def ticks_per_second(self) -> float:
        return 1.0 / self.avg_tick_time if self.avg_tick_time > 0 else 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


# ============================================================================
# Organism Processing Cache
# ============================================================================

class OrganismCache:
    """
    LRU cache for organism state calculations.

    Caches:
    - Consciousness calculations
    - Fitness scores
    - Perception processing results
    """

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, subkey: str) -> Optional[Any]:
        """Get cached value"""
        with self._lock:
            if key in self._cache and subkey in self._cache[key]:
                self.hits += 1
                # Move to end (most recent)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key][subkey]
            self.misses += 1
            return None

    def set(self, key: str, subkey: str, value: Any) -> None:
        """Set cached value"""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = {}
                self._access_order.append(key)

            self._cache[key][subkey] = value

            # Evict if over capacity
            while len(self._access_order) > self.maxsize:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)

    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """
    Processes organisms in batches for improved performance.

    Supports both threaded and process-based parallelism.
    """

    def __init__(
        self,
        max_workers: int = 4,
        batch_size: int = 100,
        use_processes: bool = False
    ):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_processes = use_processes
        self._executor: Optional[ThreadPoolExecutor] = None
        self._cache = OrganismCache(maxsize=5000)

    def start(self) -> None:
        """Start the batch processor"""
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        self._executor = ExecutorClass(max_workers=self.max_workers)

    def stop(self) -> None:
        """Stop the batch processor"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def process_organisms(
        self,
        organisms: List[DigitalBody],
        stimuli: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process organisms in parallel batches.

        Returns list of results for each organism.
        """
        if not organisms:
            return []

        # Create batches
        batches = [
            organisms[i:i + self.batch_size]
            for i in range(0, len(organisms), self.batch_size)
        ]

        results = []

        if self._executor and len(batches) > 1:
            # Parallel processing
            futures = []
            for batch in batches:
                future = self._executor.submit(
                    self._process_batch, batch, stimuli
                )
                futures.append(future)

            for future in as_completed(futures):
                results.extend(future.result())
        else:
            # Sequential fallback
            for batch in batches:
                results.extend(self._process_batch(batch, stimuli))

        return results

    def _process_batch(
        self,
        organisms: List[DigitalBody],
        stimuli: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of organisms"""
        results = []

        for organism in organisms:
            if not organism.is_alive:
                continue

            result = {
                "id": organism.id,
                "alive": True,
                "state": organism.state.value,
            }

            # Check cache for consciousness state
            cached_consciousness = self._cache.get(organism.id, "consciousness_level")
            if cached_consciousness is not None and not self._state_changed(organism):
                result["consciousness_level"] = cached_consciousness
            else:
                # Tick the organism
                organism.tick()

                # Cache the new consciousness level if available
                if hasattr(organism, 'consciousness_level'):
                    self._cache.set(organism.id, "consciousness_level", organism.consciousness_level)
                    result["consciousness_level"] = organism.consciousness_level

            results.append(result)

        return results

    def _state_changed(self, organism: DigitalBody) -> bool:
        """Check if organism state has significantly changed"""
        # Simple heuristic - could be more sophisticated
        cached_state = self._cache.get(organism.id, "last_state")
        current_state = organism.state.value

        if cached_state != current_state:
            self._cache.set(organism.id, "last_state", current_state)
            return True

        return False

    def invalidate_organism(self, organism_id: str) -> None:
        """Invalidate cache for a specific organism"""
        self._cache.invalidate(organism_id)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.get_stats()


# ============================================================================
# Async Event Processor
# ============================================================================

class AsyncEventProcessor:
    """
    Processes simulation events asynchronously.

    Uses an event queue and background worker for non-blocking
    event handling.
    """

    def __init__(self, max_queue_size: int = 10000):
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._handlers: Dict[str, List[Callable]] = {}
        self.events_processed = 0

    def start(self) -> None:
        """Start the event processor"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the event processor"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Emit an event (non-blocking)"""
        try:
            self._queue.put_nowait({
                "type": event_type,
                "data": data,
                "timestamp": time.time()
            })
            return True
        except queue.Full:
            return False

    def _process_events(self) -> None:
        """Background worker that processes events"""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                event_type = event["type"]
                data = event["data"]

                # Call handlers
                if event_type in self._handlers:
                    for handler in self._handlers[event_type]:
                        try:
                            handler(data)
                        except Exception:
                            pass  # Log in production

                # Wildcard handlers
                if "*" in self._handlers:
                    for handler in self._handlers["*"]:
                        try:
                            handler(event)
                        except Exception:
                            pass

                self.events_processed += 1

            except queue.Empty:
                continue

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()


# ============================================================================
# Parallel Simulation
# ============================================================================

class ParallelSimulation:
    """
    Enhanced simulation with parallel processing capabilities.

    Features:
    - Parallel organism processing with configurable workers
    - Async event handling
    - Performance caching
    - Detailed metrics tracking

    Example:
        config = ParallelConfig(
            name="Large Scale Evolution",
            initial_population=5000,
            max_workers=8,
            batch_size=200
        )

        sim = ParallelSimulation(config)
        sim.run()

        print(f"Performance: {sim.perf_metrics.ticks_per_second:.1f} ticks/sec")
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        self.id = str(uuid.uuid4())

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # State
        self.state = SimulationState.INITIALIZED
        self.tick_count = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # World and population
        self.world = World(WorldConfig(
            name=self.config.name,
            season_length=250,
            day_length=24,
            event_frequency=0.03
        ))

        self.population = Population(
            name="primary",
            carrying_capacity=self.config.carrying_capacity
        )
        self.world.add_population(self.population, "Central Core")

        # Metrics
        self.metrics = SimulationMetrics()
        self.perf_metrics = PerformanceMetrics()

        # Parallel components
        self.batch_processor = BatchProcessor(
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            use_processes=self.config.use_processes
        )

        self.event_processor = AsyncEventProcessor() if self.config.async_events else None

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "tick": [],
            "organism_born": [],
            "organism_died": [],
            "completed": []
        }

    def initialize(self) -> None:
        """Initialize the simulation"""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Initializing Parallel Simulation: {self.config.name}")
            print(f"Workers: {self.config.max_workers}, Batch Size: {self.config.batch_size}")
            print(f"{'='*60}")

        # Start parallel components
        self.batch_processor.start()
        if self.event_processor:
            self.event_processor.start()

        # Create initial organisms
        for i in range(self.config.initial_population):
            organism = DigitalBody(name=f"Genesis_{i:03d}")
            self.population.add_organism(organism)

        self.metrics.total_organisms_created = self.config.initial_population

        if self.config.verbose:
            print(f"Created {self.config.initial_population} organisms")

        self.state = SimulationState.RUNNING
        self.start_time = time.time()

    def step(self) -> bool:
        """Execute one simulation step with parallel processing"""
        if self.state != SimulationState.RUNNING:
            return False

        tick_start = time.perf_counter()
        self.tick_count += 1

        # Get living organisms
        organisms = [
            org for org in self.population.organisms.values()
            if org.is_alive
        ]

        # Process organisms in parallel batches
        stimuli = {"food_nearby": True, "threat_level": 0.1}
        results = self.batch_processor.process_organisms(organisms, stimuli)

        # Process results
        for result in results:
            self.perf_metrics.organisms_processed += 1

        # Process population dynamics
        self.population.tick()

        # Process world
        self.world.tick()

        # Feed organisms periodically
        if self.tick_count % 10 == 0:
            self._feed_organisms()

        # Update metrics
        self._update_metrics()

        # Record tick time
        tick_time = time.perf_counter() - tick_start
        self.perf_metrics.tick_times.append(tick_time)
        self.perf_metrics.total_ticks = self.tick_count

        # Callbacks
        for callback in self._callbacks["tick"]:
            if self.event_processor:
                self.event_processor.emit("tick", {"tick": self.tick_count})
            else:
                callback(self.tick_count)

        # Check termination
        if self._should_terminate():
            self.state = SimulationState.COMPLETED
            self.end_time = time.time()
            return False

        # Tick rate delay
        if self.config.tick_rate > 0:
            time.sleep(self.config.tick_rate)

        return True

    def _feed_organisms(self) -> None:
        """Feed resources to organisms"""
        for organism in self.population.organisms.values():
            for pool in self.world.global_resources.values():
                if pool.amount > 10:
                    consumed = pool.consume(5)
                    if consumed > 0:
                        organism.metabolism.energy = min(
                            organism.metabolism.max_energy,
                            organism.metabolism.energy + consumed * 0.3
                        )

    def _update_metrics(self) -> None:
        """Update simulation metrics"""
        self.metrics.total_ticks = self.tick_count
        pop_size = len(self.population.organisms)
        self.metrics.peak_population = max(self.metrics.peak_population, pop_size)

        if self.population.organisms:
            fitnesses = []
            for organism in self.population.organisms.values():
                fitness = self.population.dynamics.calculate_fitness(
                    organism.identity.traits, {}
                )
                fitnesses.append(fitness)
            self.metrics.avg_fitness = np.mean(fitnesses)

    def _should_terminate(self) -> bool:
        """Check termination conditions"""
        if self.tick_count >= self.config.max_ticks:
            return True
        if len(self.population.organisms) == 0:
            self.metrics.extinctions += 1
            return True
        return False

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """Run the complete simulation"""
        if self.state == SimulationState.INITIALIZED:
            self.initialize()

        if self.config.verbose:
            print(f"\nRunning parallel simulation...")
            print("-" * 60)

        while self.step():
            if progress_callback:
                progress_callback(self.tick_count, self.config.max_ticks)

            # Progress output
            if self.config.verbose and self.tick_count % 100 == 0:
                print(f"  [Tick {self.tick_count:5d}] "
                      f"Pop: {len(self.population.organisms):3d} | "
                      f"TPS: {self.perf_metrics.ticks_per_second:.1f}")

        # Cleanup
        self.batch_processor.stop()
        if self.event_processor:
            self.event_processor.stop()

        if self.config.verbose:
            print("-" * 60)
            print("Simulation completed!")
            self._print_summary()

    def _print_summary(self) -> None:
        """Print simulation summary"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        print(f"\n{'='*60}")
        print(f"PARALLEL SIMULATION SUMMARY")
        print(f"{'='*60}")

        print(f"\nPerformance:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total ticks: {self.perf_metrics.total_ticks}")
        print(f"  Avg ticks/second: {self.perf_metrics.ticks_per_second:.1f}")
        print(f"  Organisms processed: {self.perf_metrics.organisms_processed:,}")

        cache_stats = self.batch_processor.get_cache_stats()
        print(f"\nCache Performance:")
        print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
        print(f"  Cache size: {cache_stats['size']}")

        print(f"\nPopulation:")
        print(f"  Initial: {self.config.initial_population}")
        print(f"  Final: {len(self.population.organisms)}")
        print(f"  Peak: {self.metrics.peak_population}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        return {
            "config": {
                "workers": self.config.max_workers,
                "batch_size": self.config.batch_size,
                "use_processes": self.config.use_processes
            },
            "metrics": {
                "total_ticks": self.perf_metrics.total_ticks,
                "ticks_per_second": self.perf_metrics.ticks_per_second,
                "avg_tick_time_ms": self.perf_metrics.avg_tick_time * 1000,
                "organisms_processed": self.perf_metrics.organisms_processed,
                "duration_seconds": duration
            },
            "cache": self.batch_processor.get_cache_stats()
        }

    def on_tick(self, callback: Callable[[int], None]) -> None:
        """Register tick callback"""
        self._callbacks["tick"].append(callback)

    def on_completed(self, callback: Callable[[SimulationMetrics], None]) -> None:
        """Register completion callback"""
        self._callbacks["completed"].append(callback)


# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_performance(
    population_size: int = 1000,
    ticks: int = 500,
    workers: List[int] = [1, 2, 4, 8]
) -> Dict[str, Any]:
    """
    Compare performance across different worker configurations.

    Returns performance metrics for each configuration.
    """
    results = {}

    for num_workers in workers:
        config = ParallelConfig(
            name=f"Benchmark_{num_workers}w",
            initial_population=population_size,
            max_ticks=ticks,
            max_workers=num_workers,
            batch_size=max(50, population_size // (num_workers * 2)),
            verbose=False
        )

        sim = ParallelSimulation(config)
        sim.run()

        results[f"{num_workers}_workers"] = sim.get_performance_report()

    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("NPCPU Parallel Simulation Demo")
    print("=" * 60)

    # Run parallel simulation
    config = ParallelConfig(
        name="Parallel Evolution Demo",
        seed=42,
        initial_population=200,
        max_ticks=500,
        carrying_capacity=500,
        max_workers=4,
        batch_size=50,
        verbose=True
    )

    sim = ParallelSimulation(config)
    sim.run()

    # Print performance report
    report = sim.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Ticks/second: {report['metrics']['ticks_per_second']:.1f}")
    print(f"  Cache hit rate: {report['cache']['hit_rate']*100:.1f}%")
