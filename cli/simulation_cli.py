"""
NPCPU Simulation CLI

Command-line interface for managing and running NPCPU simulations.

Features:
- Run simulations with various configurations
- Monitor running simulations
- View statistics and metrics
- Benchmark performance
- Visualize evolutionary data

Usage:
    python -m cli.simulation_cli run --population 100 --ticks 1000
    python -m cli.simulation_cli benchmark --quick
    python -m cli.simulation_cli stats --simulation-id abc123
    python -m cli.simulation_cli visualize --type phylogenetic
"""

import argparse
import sys
import os
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.runner import Simulation, SimulationConfig, SimulationState
from simulation.parallel_runner import ParallelSimulation, ParallelConfig


# ============================================================================
# CLI Configuration
# ============================================================================

@dataclass
class CLIConfig:
    """Configuration for CLI behavior"""
    verbose: bool = True
    output_format: str = "text"  # text, json
    color_output: bool = True


# ============================================================================
# Color Output Helpers
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def disable(cls):
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


def print_header(text: str, char: str = "=") -> None:
    """Print a styled header"""
    width = 60
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}")
    print(text.center(width))
    print(f"{char * width}{Colors.ENDC}")


def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}! {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.BLUE}→ {text}{Colors.ENDC}")


# ============================================================================
# Command Handlers
# ============================================================================

def cmd_run(args: argparse.Namespace) -> int:
    """Run a simulation"""
    print_header("NPCPU Simulation Runner")

    # Build configuration
    if args.parallel:
        config = ParallelConfig(
            name=args.name or "CLI Simulation",
            seed=args.seed,
            initial_population=args.population,
            max_ticks=args.ticks,
            carrying_capacity=args.capacity,
            tick_rate=args.tick_rate,
            max_workers=args.workers,
            batch_size=args.batch_size,
            verbose=not args.quiet,
            output_dir=args.output
        )
        sim = ParallelSimulation(config)
    else:
        config = SimulationConfig(
            name=args.name or "CLI Simulation",
            seed=args.seed,
            initial_population=args.population,
            max_ticks=args.ticks,
            carrying_capacity=args.capacity,
            tick_rate=args.tick_rate,
            verbose=not args.quiet,
            output_dir=args.output
        )
        sim = Simulation(config)

    print_info(f"Starting simulation: {config.name}")
    print_info(f"Population: {config.initial_population}, Max ticks: {config.max_ticks}")

    if args.parallel:
        print_info(f"Parallel mode: {config.max_workers} workers")

    try:
        start_time = time.time()

        # Progress callback for non-verbose mode
        def progress(current: int, total: int) -> None:
            if args.quiet and current % 100 == 0:
                pct = current / total * 100
                bar_len = 30
                filled = int(bar_len * current / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r[{bar}] {pct:.1f}% ({current}/{total})", end="", flush=True)

        sim.run(progress_callback=progress if args.quiet else None)

        if args.quiet:
            print()  # New line after progress bar

        elapsed = time.time() - start_time

        print_success(f"Simulation completed in {elapsed:.2f} seconds")

        # Save results if requested
        if args.save:
            filepath = sim.save_results()
            print_success(f"Results saved to: {filepath}")

        # Print summary
        if hasattr(sim, 'perf_metrics'):
            print_info(f"Performance: {sim.perf_metrics.ticks_per_second:.1f} ticks/sec")

        return 0

    except KeyboardInterrupt:
        print_warning("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        print_error(f"Simulation failed: {e}")
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run performance benchmarks"""
    print_header("NPCPU Performance Benchmarks")

    if args.type == "all" or args.type == "core":
        print_info("Running core benchmarks...")
        from benchmarks.performance import BenchmarkRunner
        runner = BenchmarkRunner(quick=args.quick)
        runner.run_all()

    if args.type == "all" or args.type == "consciousness":
        print_info("Running consciousness benchmarks...")
        from benchmarks.consciousness_benchmarks import ConsciousnessBenchmarkRunner
        runner = ConsciousnessBenchmarkRunner(quick=args.quick)
        runner.run_all()

    if args.type == "parallel":
        print_info("Running parallel performance comparison...")
        from simulation.parallel_runner import compare_performance
        results = compare_performance(
            population_size=args.population or 500,
            ticks=args.ticks or 200,
            workers=[1, 2, 4]
        )
        print("\nParallel Performance Comparison:")
        print("-" * 40)
        for config, data in results.items():
            tps = data['metrics']['ticks_per_second']
            print(f"  {config}: {tps:.1f} ticks/sec")

    print_success("Benchmarks completed")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generate visualizations"""
    print_header("NPCPU Visualization")

    from evolution.genetic_engine import EvolutionEngine
    from evolution.phylogenetic_viz import PhylogeneticVisualizer

    # Create a sample evolution run for visualization
    print_info("Running evolution for visualization...")

    engine = EvolutionEngine(
        population_size=args.population or 50,
        mutation_rate=0.1,
        enable_speciation=True
    )

    gene_specs = {
        "speed": (0.0, 1.0),
        "strength": (0.0, 1.0),
        "intelligence": (0.0, 1.0),
        "endurance": (0.0, 1.0)
    }
    engine.initialize_population(gene_specs)

    def fitness(genome):
        return (
            genome.express("speed") * 0.3 +
            genome.express("strength") * 0.3 +
            genome.express("intelligence") * 0.4
        )

    generations = args.generations or 20
    for gen in range(generations):
        engine.evolve_generation(fitness)

    # Create visualizer
    viz = PhylogeneticVisualizer(engine)

    if args.type == "tree" or args.type == "all":
        print("\n" + viz.generate_ascii_tree(max_depth=args.depth or 5))

    if args.type == "diversity" or args.type == "all":
        print("\n" + viz.generate_diversity_report())

    if args.type == "species" or args.type == "all":
        print("\n" + viz.generate_species_tree())

    print_success("Visualization complete")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show statistics from saved simulation"""
    print_header("Simulation Statistics")

    filepath = args.file
    if not os.path.exists(filepath):
        print_error(f"File not found: {filepath}")
        return 1

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"\n{Colors.BOLD}Simulation Results{Colors.ENDC}")
        print("-" * 40)

        config = data.get('config', {})
        print(f"Name: {config.get('name', 'Unknown')}")
        print(f"Seed: {config.get('seed', 'None')}")

        metrics = data.get('metrics', {})
        print(f"\n{Colors.BOLD}Metrics{Colors.ENDC}")
        print("-" * 40)
        print(f"Total ticks: {metrics.get('total_ticks', 0)}")
        print(f"Peak population: {metrics.get('peak_population', 0)}")
        print(f"Total deaths: {metrics.get('total_deaths', 0)}")
        print(f"Avg fitness: {metrics.get('avg_fitness', 0):.4f}")

        print(f"\nDuration: {data.get('duration', 0):.2f} seconds")
        print(f"Final population: {data.get('final_population', 0)}")

        # Fitness history chart
        fitness_history = data.get('fitness_history', [])
        if fitness_history and args.chart:
            print(f"\n{Colors.BOLD}Fitness History{Colors.ENDC}")
            print("-" * 40)
            print(_ascii_chart(fitness_history, width=50, height=8))

        print_success("Statistics loaded successfully")
        return 0

    except json.JSONDecodeError:
        print_error("Invalid JSON file")
        return 1
    except Exception as e:
        print_error(f"Failed to load statistics: {e}")
        return 1


def _ascii_chart(values: List[float], width: int = 50, height: int = 10) -> str:
    """Create simple ASCII chart"""
    if not values:
        return "No data"

    # Sample if too many values
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    min_val = min(sampled)
    max_val = max(sampled)
    range_val = max_val - min_val if max_val > min_val else 1

    lines = []
    for row in range(height, 0, -1):
        threshold = min_val + (row / height) * range_val
        line = ""
        for val in sampled:
            line += "█" if val >= threshold else " "
        lines.append(f"{threshold:>6.2f} |{line}|")

    lines.append(" " * 7 + "+" + "-" * len(sampled) + "+")
    return "\n".join(lines)


def cmd_info(args: argparse.Namespace) -> int:
    """Show NPCPU information"""
    print_header("NPCPU - Non-Player Cognitive Processing Unit")

    print(f"""
{Colors.BOLD}Version:{Colors.ENDC} 0.1.0 (Alpha)
{Colors.BOLD}License:{Colors.ENDC} MIT

{Colors.BOLD}Description:{Colors.ENDC}
  Consciousness-aware distributed computing framework for
  simulating digital life forms with emergent intelligence.

{Colors.BOLD}Features:{Colors.ENDC}
  • Digital organism simulation
  • Neural consciousness with attention mechanisms
  • Genetic evolution with speciation
  • Parallel processing support
  • Metabolism-consciousness coupling
  • Comprehensive benchmarking

{Colors.BOLD}Modules:{Colors.ENDC}
  • organism/      - Digital body and metabolism
  • consciousness/ - Neural consciousness systems
  • evolution/     - Genetic engine and mutations
  • simulation/    - Sequential and parallel runners
  • ecosystem/     - World and population dynamics

{Colors.BOLD}Commands:{Colors.ENDC}
  run        - Run a simulation
  benchmark  - Run performance benchmarks
  visualize  - Generate visualizations
  stats      - View simulation statistics
  info       - Show this information
""")

    return 0


# ============================================================================
# Main CLI Setup
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        prog="npcpu",
        description="NPCPU Simulation CLI - Manage consciousness simulations"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a simulation")
    run_parser.add_argument("--name", type=str, help="Simulation name")
    run_parser.add_argument("--population", "-p", type=int, default=20,
                           help="Initial population size")
    run_parser.add_argument("--ticks", "-t", type=int, default=1000,
                           help="Maximum simulation ticks")
    run_parser.add_argument("--capacity", "-c", type=int, default=200,
                           help="World carrying capacity")
    run_parser.add_argument("--seed", "-s", type=int, help="Random seed")
    run_parser.add_argument("--tick-rate", type=float, default=0.0,
                           help="Delay between ticks (0 = max speed)")
    run_parser.add_argument("--parallel", action="store_true",
                           help="Use parallel processing")
    run_parser.add_argument("--workers", "-w", type=int, default=4,
                           help="Number of parallel workers")
    run_parser.add_argument("--batch-size", type=int, default=50,
                           help="Batch size for parallel processing")
    run_parser.add_argument("--output", "-o", type=str, default="./simulation_output",
                           help="Output directory")
    run_parser.add_argument("--save", action="store_true",
                           help="Save results to file")
    run_parser.add_argument("--quiet", "-q", action="store_true",
                           help="Minimal output")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--type", choices=["all", "core", "consciousness", "parallel"],
                             default="all", help="Benchmark type")
    bench_parser.add_argument("--quick", action="store_true",
                             help="Quick benchmark mode")
    bench_parser.add_argument("--population", type=int, help="Population for parallel bench")
    bench_parser.add_argument("--ticks", type=int, help="Ticks for parallel bench")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--type", choices=["tree", "diversity", "species", "all"],
                           default="all", help="Visualization type")
    viz_parser.add_argument("--population", type=int, default=50,
                           help="Population size for evolution")
    viz_parser.add_argument("--generations", "-g", type=int, default=20,
                           help="Generations to evolve")
    viz_parser.add_argument("--depth", type=int, default=5,
                           help="Tree depth to display")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="View simulation statistics")
    stats_parser.add_argument("file", type=str, help="Path to results JSON file")
    stats_parser.add_argument("--chart", action="store_true",
                             help="Show fitness chart")

    # Info command
    subparsers.add_parser("info", help="Show NPCPU information")

    return parser


def main() -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    handlers = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "visualize": cmd_visualize,
        "stats": cmd_stats,
        "info": cmd_info,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
