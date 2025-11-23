"""
NPCPU Command Line Interface

Provides CLI tools for managing and running NPCPU simulations.

Usage:
    python -m cli.simulation_cli run --population 100
    python -m cli.simulation_cli benchmark --quick
    python -m cli.simulation_cli visualize --type tree
"""

from cli.simulation_cli import main

__all__ = ["main"]
