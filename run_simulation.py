#!/usr/bin/env python3
"""
NPCPU Digital Life Simulation - Main Entry Point

Run complete digital life simulations with configurable parameters.

Usage:
    python run_simulation.py                    # Default simulation
    python run_simulation.py --population 50    # Start with 50 organisms
    python run_simulation.py --ticks 10000      # Run for 10000 ticks
    python run_simulation.py --seed 42          # Reproducible with seed
    python run_simulation.py --quiet            # Minimal output
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.runner import Simulation, SimulationConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="NPCPU Digital Life Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Run default simulation
  %(prog)s -p 50 -t 5000            50 organisms, 5000 ticks
  %(prog)s --seed 42 --verbose      Reproducible run with detailed output
  %(prog)s --name "Evolution Test"  Named simulation run
        """
    )

    parser.add_argument(
        "-n", "--name",
        default="NPCPU Simulation",
        help="Name for this simulation run"
    )

    parser.add_argument(
        "-p", "--population",
        type=int,
        default=20,
        help="Initial population size (default: 20)"
    )

    parser.add_argument(
        "-t", "--ticks",
        type=int,
        default=1000,
        help="Maximum ticks to run (default: 1000)"
    )

    parser.add_argument(
        "-c", "--capacity",
        type=int,
        default=200,
        help="Population carrying capacity (default: 200)"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "-o", "--output",
        default="./simulation_output",
        help="Output directory for results"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (minimal output)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create configuration
    config = SimulationConfig(
        name=args.name,
        seed=args.seed,
        max_ticks=args.ticks,
        initial_population=args.population,
        carrying_capacity=args.capacity,
        output_dir=args.output,
        verbose=not args.quiet
    )

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ███╗   ██╗██████╗  ██████╗██████╗ ██╗   ██╗              ║
    ║     ████╗  ██║██╔══██╗██╔════╝██╔══██╗██║   ██║              ║
    ║     ██╔██╗ ██║██████╔╝██║     ██████╔╝██║   ██║              ║
    ║     ██║╚██╗██║██╔═══╝ ██║     ██╔═══╝ ██║   ██║              ║
    ║     ██║ ╚████║██║     ╚██████╗██║     ╚██████╔╝              ║
    ║     ╚═╝  ╚═══╝╚═╝      ╚═════╝╚═╝      ╚═════╝              ║
    ║                                                              ║
    ║              Digital Life Simulation                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print(f"Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Initial Population: {config.initial_population}")
    print(f"  Max Ticks: {config.max_ticks}")
    print(f"  Carrying Capacity: {config.carrying_capacity}")
    print(f"  Seed: {config.seed or 'random'}")

    # Create and run simulation
    sim = Simulation(config)

    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sim.state = "interrupted"

    # Save results if requested
    if args.save:
        filepath = sim.save_results()
        print(f"Results saved to: {filepath}")

    # Return final population count as exit code hint
    return 0 if len(sim.population.organisms) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
