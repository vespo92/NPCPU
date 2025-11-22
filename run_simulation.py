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

    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save complete simulation state (can be resumed)"
    )

    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load and resume from a saved snapshot ID"
    )

    parser.add_argument(
        "--list-saves",
        action="store_true",
        help="List available saved simulations"
    )

    parser.add_argument(
        "--save-dir",
        default="./simulation_saves",
        help="Directory for save files"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Import persistence module
    from simulation.persistence import SimulationPersistence, restore_simulation

    persistence = SimulationPersistence(args.save_dir)

    # Handle list-saves command
    if args.list_saves:
        saves = persistence.list_saves()
        if not saves:
            print("No saved simulations found.")
            return 0

        print(f"\nFound {len(saves)} saved simulation(s):\n")
        for save in saves:
            print(f"  ID: {save.get('snapshot_id', 'unknown')}")
            print(f"  Name: {save.get('simulation_name', 'N/A')}")
            print(f"  Tick: {save.get('tick', 'N/A')}")
            print(f"  Organisms: {save.get('organisms', 'N/A')}")
            print(f"  Created: {save.get('created_at', 'N/A')}")
            if save.get('description'):
                print(f"  Description: {save['description']}")
            print()
        return 0

    # Handle loading a saved simulation
    if args.load:
        print(f"Loading simulation from: {args.load}")
        try:
            snapshot = persistence.load(args.load)
            sim = restore_simulation(snapshot)
            print(f"Restored simulation '{snapshot.simulation_name}' at tick {snapshot.tick}")
            print(f"Organisms: {len(sim.population.organisms)}")
        except FileNotFoundError:
            print(f"Error: No snapshot found with ID '{args.load}'")
            return 1
    else:
        # Create new simulation
        config = SimulationConfig(
            name=args.name,
            seed=args.seed,
            max_ticks=args.ticks,
            initial_population=args.population,
            carrying_capacity=args.capacity,
            output_dir=args.output,
            verbose=not args.quiet
        )
        sim = Simulation(config)

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
    print(f"  Name: {sim.config.name}")
    print(f"  Initial Population: {sim.config.initial_population}")
    print(f"  Max Ticks: {sim.config.max_ticks}")
    print(f"  Carrying Capacity: {sim.config.carrying_capacity}")
    print(f"  Seed: {sim.config.seed or 'random'}")
    if args.load:
        print(f"  Resumed from tick: {sim.tick_count}")

    from simulation.runner import SimulationState

    interrupted = False
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sim.state = SimulationState.PAUSED
        interrupted = True

        # Offer to save on interrupt
        if args.save_state or input("\nSave current state? (y/n): ").lower() == 'y':
            snapshot_id = persistence.save(sim, description="Interrupted simulation")
            print(f"State saved with ID: {snapshot_id}")
            print(f"Resume with: python run_simulation.py --load {snapshot_id}")

    # Save results if requested
    if args.save:
        filepath = sim.save_results()
        print(f"Results saved to: {filepath}")

    # Save complete state if requested (only if not already saved during interrupt)
    if args.save_state and not interrupted:
        snapshot_id = persistence.save(sim, description="Completed simulation")
        print(f"State saved with ID: {snapshot_id}")

    # Return final population count as exit code hint
    return 0 if len(sim.population.organisms) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
