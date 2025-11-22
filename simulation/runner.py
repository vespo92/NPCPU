"""
Digital Life Simulation Runner

The master orchestrator that brings together all NPCPU systems
to run complete digital life simulations:
- Creates and manages populations of digital organisms
- Runs the world simulation
- Tracks evolution and adaptation
- Provides metrics and visualization
"""

import time
import uuid
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from organism.digital_body import DigitalBody, OrganismState
from organism.metabolism import ResourceType
from ecosystem.population import Population, InteractionType
from ecosystem.world import World, WorldConfig, WorldEventType


# ============================================================================
# Enums
# ============================================================================

class SimulationState(Enum):
    """States of the simulation"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SimulationConfig:
    """Configuration for a simulation run"""
    name: str = "NPCPU Simulation"
    seed: Optional[int] = None       # Random seed for reproducibility
    max_ticks: int = 10000           # Maximum ticks to run
    initial_population: int = 10     # Starting population size
    carrying_capacity: int = 200     # World carrying capacity
    tick_rate: float = 0.0           # Delay between ticks (0 = max speed)
    auto_save_interval: int = 1000   # Auto-save every N ticks
    output_dir: str = "./simulation_output"
    verbose: bool = True


@dataclass
class SimulationMetrics:
    """Metrics tracked during simulation"""
    total_ticks: int = 0
    total_organisms_created: int = 0
    total_deaths: int = 0
    peak_population: int = 0
    total_interactions: int = 0
    total_reproductions: int = 0
    total_mutations: int = 0
    avg_lifespan: float = 0.0
    avg_fitness: float = 0.0
    genetic_diversity: float = 0.0
    world_events: int = 0
    extinctions: int = 0


@dataclass
class GenerationStats:
    """Statistics for a generation"""
    generation: int
    population: int
    avg_fitness: float
    best_traits: Dict[str, float]
    worst_traits: Dict[str, float]
    diversity: float


# ============================================================================
# Simulation
# ============================================================================

class Simulation:
    """
    Master simulation orchestrator for digital life.

    Brings together:
    - Digital organisms (DigitalBody)
    - Populations and social dynamics
    - World environment
    - Evolution tracking

    Example:
        sim = Simulation(SimulationConfig(
            name="Evolution Study",
            initial_population=20,
            max_ticks=5000
        ))

        # Run simulation
        sim.run()

        # Or step manually
        while sim.state == SimulationState.RUNNING:
            sim.step()

        # Get results
        metrics = sim.get_metrics()
        sim.save_results()
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.id = str(uuid.uuid4())

        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # State
        self.state = SimulationState.INITIALIZED
        self.tick_count = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # World
        self.world = World(WorldConfig(
            name=self.config.name,
            season_length=250,
            day_length=24,
            event_frequency=0.03
        ))

        # Population
        self.population = Population(
            name="primary",
            carrying_capacity=self.config.carrying_capacity
        )
        self.world.add_population(self.population, "Central Core")

        # Metrics
        self.metrics = SimulationMetrics()
        self.generation_history: List[GenerationStats] = []
        self.population_history: List[int] = []
        self.fitness_history: List[float] = []

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "tick": [],
            "organism_born": [],
            "organism_died": [],
            "generation": [],
            "milestone": [],
            "completed": []
        }

        # Setup event handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup event handlers"""
        # Track births
        def on_birth(organism):
            self.metrics.total_organisms_created += 1
            self.metrics.total_reproductions += 1
            for callback in self._callbacks["organism_born"]:
                callback(organism)

        self.population.on_organism_added(on_birth)

        # Track deaths
        def on_death(organism, cause):
            self.metrics.total_deaths += 1
            for callback in self._callbacks["organism_died"]:
                callback(organism, cause)

        self.population.on_organism_died(on_death)

        # Track world events
        def on_event(event):
            self.metrics.world_events += 1

        self.world.on_event_started(on_event)

    def initialize(self):
        """Initialize simulation with starting population"""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Initializing: {self.config.name}")
            print(f"{'='*60}")

        # Create initial organisms
        for i in range(self.config.initial_population):
            organism = DigitalBody(name=f"Genesis_{i:03d}")
            self.population.add_organism(organism)

        self.metrics.total_organisms_created = self.config.initial_population

        if self.config.verbose:
            print(f"Created {self.config.initial_population} organisms")
            print(f"World regions: {len(self.world.regions)}")

        self.state = SimulationState.RUNNING
        self.start_time = time.time()

    def step(self) -> bool:
        """
        Execute one simulation step.

        Returns False if simulation should stop.
        """
        if self.state != SimulationState.RUNNING:
            return False

        self.tick_count += 1

        # Process all organisms
        for organism in list(self.population.organisms.values()):
            if organism.is_alive:
                organism.tick()

        # Process population dynamics
        self.population.tick()

        # Process world
        self.world.tick()

        # Feed organisms periodically
        if self.tick_count % 10 == 0:
            self._feed_organisms()

        # Update metrics
        self._update_metrics()

        # Record history
        self.population_history.append(len(self.population.organisms))

        # Check milestones
        self._check_milestones()

        # Callbacks
        for callback in self._callbacks["tick"]:
            callback(self.tick_count)

        # Auto-save
        if self.tick_count % self.config.auto_save_interval == 0:
            self._auto_save()

        # Check termination conditions
        if self._should_terminate():
            self.state = SimulationState.COMPLETED
            self.end_time = time.time()
            for callback in self._callbacks["completed"]:
                callback(self.metrics)
            return False

        # Tick rate delay
        if self.config.tick_rate > 0:
            time.sleep(self.config.tick_rate)

        return True

    def _feed_organisms(self):
        """Feed resources to organisms"""
        for organism in self.population.organisms.values():
            # Get resources from world
            for pool in self.world.global_resources.values():
                if pool.amount > 10:
                    consumed = pool.consume(5)
                    if consumed > 0:
                        organism.metabolism.energy = min(
                            organism.metabolism.max_energy,
                            organism.metabolism.energy + consumed * 0.3
                        )

    def _update_metrics(self):
        """Update simulation metrics"""
        self.metrics.total_ticks = self.tick_count

        pop_size = len(self.population.organisms)
        self.metrics.peak_population = max(self.metrics.peak_population, pop_size)

        # Calculate average fitness
        if self.population.organisms:
            fitnesses = []
            for organism in self.population.organisms.values():
                fitness = self.population.dynamics.calculate_fitness(
                    organism.identity.traits, {}
                )
                fitnesses.append(fitness)
            self.metrics.avg_fitness = np.mean(fitnesses)
            self.fitness_history.append(self.metrics.avg_fitness)

            # Genetic diversity
            stats = self.population.get_stats()
            self.metrics.genetic_diversity = stats.genetic_diversity

        # Interaction count from social network
        self.metrics.total_interactions = sum(
            self.population.social.interaction_count.values()
        )

    def _check_milestones(self):
        """Check for and report milestones"""
        pop_size = len(self.population.organisms)

        # Population milestones
        milestones = [25, 50, 100, 150, 200]
        for milestone in milestones:
            if pop_size >= milestone and self.metrics.peak_population < milestone + 1:
                if self.config.verbose:
                    print(f"  [Tick {self.tick_count}] Milestone: Population reached {milestone}!")
                for callback in self._callbacks["milestone"]:
                    callback("population", milestone)

        # Generation tracking (every 500 ticks)
        if self.tick_count % 500 == 0:
            self._record_generation()

        # Verbose output
        if self.config.verbose and self.tick_count % 100 == 0:
            self._print_status()

    def _record_generation(self):
        """Record generation statistics"""
        if not self.population.organisms:
            return

        # Calculate stats
        fitnesses = []
        traits_collected: Dict[str, List[float]] = {}

        for organism in self.population.organisms.values():
            fitness = self.population.dynamics.calculate_fitness(
                organism.identity.traits, {}
            )
            fitnesses.append(fitness)

            for trait, value in organism.identity.traits.items():
                if trait not in traits_collected:
                    traits_collected[trait] = []
                traits_collected[trait].append(value)

        # Find best/worst traits
        best_traits = {}
        worst_traits = {}
        for trait, values in traits_collected.items():
            best_traits[trait] = max(values)
            worst_traits[trait] = min(values)

        # Diversity
        diversity = np.mean([np.std(v) for v in traits_collected.values()])

        gen_stats = GenerationStats(
            generation=len(self.generation_history) + 1,
            population=len(self.population.organisms),
            avg_fitness=np.mean(fitnesses),
            best_traits=best_traits,
            worst_traits=worst_traits,
            diversity=diversity
        )

        self.generation_history.append(gen_stats)

        for callback in self._callbacks["generation"]:
            callback(gen_stats)

    def _print_status(self):
        """Print current status"""
        pop_size = len(self.population.organisms)
        world_stats = self.world.get_global_stats()

        print(f"  [Tick {self.tick_count:5d}] "
              f"Pop: {pop_size:3d} | "
              f"Season: {world_stats['season']:6s} | "
              f"Year: {world_stats['year']:2d} | "
              f"Fitness: {self.metrics.avg_fitness:.3f}")

    def _should_terminate(self) -> bool:
        """Check if simulation should terminate"""
        # Max ticks reached
        if self.tick_count >= self.config.max_ticks:
            if self.config.verbose:
                print(f"\nMax ticks ({self.config.max_ticks}) reached.")
            return True

        # Extinction
        if len(self.population.organisms) == 0:
            if self.config.verbose:
                print(f"\nExtinction occurred at tick {self.tick_count}.")
            self.metrics.extinctions += 1
            return True

        return False

    def _auto_save(self):
        """Auto-save simulation state"""
        try:
            path = Path(self.config.output_dir)
            path.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "tick": self.tick_count,
                "metrics": self.metrics.__dict__,
                "population_size": len(self.population.organisms)
            }

            checkpoint_file = path / f"checkpoint_{self.tick_count}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)

        except Exception as e:
            if self.config.verbose:
                print(f"  [Warning] Auto-save failed: {e}")

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        """
        Run complete simulation.

        Args:
            progress_callback: Optional callback(current_tick, max_ticks)
        """
        if self.state == SimulationState.INITIALIZED:
            self.initialize()

        if self.config.verbose:
            print(f"\nRunning simulation...")
            print("-" * 60)

        while self.step():
            if progress_callback:
                progress_callback(self.tick_count, self.config.max_ticks)

        if self.config.verbose:
            print("-" * 60)
            print("Simulation completed!")
            self._print_summary()

    def pause(self):
        """Pause simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED

    def resume(self):
        """Resume paused simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING

    def _print_summary(self):
        """Print simulation summary"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())

        print(f"\n{'='*60}")
        print(f"SIMULATION SUMMARY: {self.config.name}")
        print(f"{'='*60}")
        print(f"\nDuration: {duration:.2f} seconds")
        print(f"Total ticks: {self.metrics.total_ticks}")
        print(f"Ticks/second: {self.metrics.total_ticks / max(1, duration):.1f}")

        print(f"\nPopulation:")
        print(f"  Initial: {self.config.initial_population}")
        print(f"  Final: {len(self.population.organisms)}")
        print(f"  Peak: {self.metrics.peak_population}")
        print(f"  Total created: {self.metrics.total_organisms_created}")
        print(f"  Total deaths: {self.metrics.total_deaths}")

        print(f"\nEvolution:")
        print(f"  Reproductions: {self.metrics.total_reproductions}")
        print(f"  Avg fitness: {self.metrics.avg_fitness:.3f}")
        print(f"  Genetic diversity: {self.metrics.genetic_diversity:.4f}")
        print(f"  Generations recorded: {len(self.generation_history)}")

        print(f"\nWorld:")
        print(f"  Events: {self.metrics.world_events}")
        print(f"  Interactions: {self.metrics.total_interactions}")

    def get_metrics(self) -> SimulationMetrics:
        """Get current metrics"""
        return self.metrics

    def get_results(self) -> Dict[str, Any]:
        """Get complete results"""
        return {
            "config": {
                "name": self.config.name,
                "seed": self.config.seed,
                "max_ticks": self.config.max_ticks,
                "initial_population": self.config.initial_population
            },
            "metrics": self.metrics.__dict__,
            "final_population": len(self.population.organisms),
            "population_history": self.population_history,
            "fitness_history": self.fitness_history,
            "generations": [
                {
                    "generation": g.generation,
                    "population": g.population,
                    "avg_fitness": g.avg_fitness,
                    "diversity": g.diversity
                }
                for g in self.generation_history
            ],
            "duration": (self.end_time or time.time()) - (self.start_time or time.time())
        }

    def save_results(self, filepath: Optional[str] = None):
        """Save results to file"""
        if filepath is None:
            path = Path(self.config.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            filepath = str(path / f"results_{self.id[:8]}.json")

        results = self.get_results()

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if self.config.verbose:
            print(f"\nResults saved to: {filepath}")

        return filepath

    def on_tick(self, callback: Callable[[int], None]):
        """Register tick callback"""
        self._callbacks["tick"].append(callback)

    def on_organism_born(self, callback: Callable):
        """Register birth callback"""
        self._callbacks["organism_born"].append(callback)

    def on_organism_died(self, callback: Callable):
        """Register death callback"""
        self._callbacks["organism_died"].append(callback)

    def on_completed(self, callback: Callable[[SimulationMetrics], None]):
        """Register completion callback"""
        self._callbacks["completed"].append(callback)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("NPCPU Digital Life Simulation")
    print("=" * 60)

    # Create simulation
    config = SimulationConfig(
        name="Digital Evolution Study",
        seed=42,
        initial_population=15,
        max_ticks=500,
        carrying_capacity=100,
        tick_rate=0.0,  # Max speed
        verbose=True
    )

    sim = Simulation(config)

    # Run simulation
    sim.run()

    # Get and display results
    results = sim.get_results()

    print(f"\nFinal Results:")
    print(f"  Simulation ID: {sim.id[:8]}")
    print(f"  Final population: {results['final_population']}")
    print(f"  Peak fitness: {max(results['fitness_history']) if results['fitness_history'] else 0:.3f}")

    if sim.generation_history:
        print(f"\nGeneration Evolution:")
        for gen in sim.generation_history[-3:]:
            print(f"  Gen {gen.generation}: pop={gen.population}, "
                  f"fitness={gen.avg_fitness:.3f}, diversity={gen.diversity:.3f}")
