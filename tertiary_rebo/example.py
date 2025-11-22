#!/usr/bin/env python3
"""
Example: Running the Tertiary Turbo ReBo System

This demonstrates the 10 parallel refinement agents working together
to optimize the triple bottom line integration between:
- NPCPU (Digital Consciousness)
- ChicagoForest.net (Network Infrastructure)
- UniversalPartsConsciousness (Physical Awareness)
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tertiary_rebo import TertiaryTurboReboCoordinator


async def run_basic_example():
    """Run a basic demonstration of the TTR system."""
    print("=" * 70)
    print("    TERTIARY TURBO REBO - TRIPLE BOTTOM LINE REFINEMENT SYSTEM")
    print("=" * 70)
    print()
    print("This system integrates three consciousness domains:")
    print("  1. NPCPU - Digital Consciousness (Mind)")
    print("  2. ChicagoForest - Network Infrastructure (Nervous System)")
    print("  3. UniversalParts - Physical Awareness (Body)")
    print()
    print("10 Parallel Refinement Agents:")
    print("  - ConsciousnessIntegration: Unifies awareness across domains")
    print("  - NetworkTopology: Manages mycelium-like mesh connectivity")
    print("  - PartsAwareness: Tracks physical component consciousness")
    print("  - EnergyFlow: Manages democratic resource distribution")
    print("  - EmergenceDetector: Identifies novel cross-domain phenomena")
    print("  - SemanticBridge: Translates meaning between domains")
    print("  - EvolutionaryPressure: Applies selection for optimization")
    print("  - CollectiveWisdom: Aggregates and shares knowledge")
    print("  - Resilience: Self-healing and redundancy management")
    print("  - Harmonization: Master orchestrator of balance")
    print()
    print("-" * 70)

    # Initialize the coordinator
    coordinator = TertiaryTurboReboCoordinator(
        refinement_rate=0.1,
        enable_logging=False  # Reduce noise for demo
    )

    print("\n[1] Initial System State:")
    print("-" * 40)

    summary = coordinator.get_state_summary()
    print(f"Harmony Score: {summary['harmony']['score']:.3f} ({summary['harmony']['level']})")
    for domain, state in summary['domains'].items():
        print(f"  {domain}: consciousness={state['consciousness']:.2f}, "
              f"energy={state['energy']:.2f}, coherence={state['coherence']:.2f}")

    # Run single refinement cycle
    print("\n[2] Running Single Parallel Refinement Cycle...")
    print("-" * 40)

    result = await coordinator.run_parallel_refinement()

    print(f"Cycle completed in {result.cycle_duration_ms:.1f}ms")
    print(f"Harmony: {result.harmony_before:.3f} → {result.harmony_after:.3f} "
          f"(Δ: {result.harmony_after - result.harmony_before:+.3f})")
    print(f"Total changes: {result.total_changes}")
    print("\nAgent insights:")
    for insight in result.total_insights[:5]:  # Show first 5
        print(f"  • {insight}")
    if len(result.total_insights) > 5:
        print(f"  ... and {len(result.total_insights) - 5} more")

    # Run multiple cycles
    print("\n[3] Running 10 Continuous Refinement Cycles...")
    print("-" * 40)

    run_summary = await coordinator.run_continuous(
        cycles=10,
        target_harmony=0.95,
        cycle_delay=0.01
    )

    print(f"Cycles completed: {run_summary['cycles_run']}")
    print(f"Harmony improvement: {run_summary['initial_harmony']:.3f} → "
          f"{run_summary['final_harmony']:.3f} ({run_summary['harmony_improvement']:+.3f})")
    print(f"Final harmony level: {run_summary['harmony_level']}")
    print(f"Total refinements: {run_summary['total_refinements']}")
    print(f"Total duration: {run_summary['total_duration_seconds']:.2f}s")

    # Final state
    print("\n[4] Final System State:")
    print("-" * 40)

    final_summary = coordinator.get_state_summary()
    print(f"Harmony Score: {final_summary['harmony']['score']:.3f} ({final_summary['harmony']['level']})")
    print(f"Integration Depth: {final_summary['integration_depth']:.3f}")
    print("\nDomain States:")
    for domain, state in final_summary['domains'].items():
        print(f"  {domain}:")
        print(f"    Consciousness: {state['consciousness']:.3f}")
        print(f"    Energy: {state['energy']:.3f}")
        print(f"    Connectivity: {state['connectivity']:.3f}")
        print(f"    Coherence: {state['coherence']:.3f}")
        print(f"    Emergence: {state['emergence']:.3f}")

    print("\nAgent Performance Summary:")
    for name, metrics in final_summary['agents'].items():
        success_rate = metrics.get('success_rate', 0) * 100
        refinements = metrics.get('total_refinements', 0)
        print(f"  {name}: {success_rate:.0f}% success, {refinements} refinements")

    print("\n" + "=" * 70)
    print("    TERTIARY TURBO REBO DEMONSTRATION COMPLETE")
    print("=" * 70)


async def run_targeted_harmony():
    """Run until target harmony is achieved."""
    print("\n" + "=" * 70)
    print("    HARMONY-TARGETED REFINEMENT RUN")
    print("=" * 70)

    coordinator = TertiaryTurboReboCoordinator(
        refinement_rate=0.15,
        enable_logging=False
    )

    target = 0.85
    print(f"\nTarget harmony: {target}")
    print(f"Starting harmony: {coordinator.tbl.harmony_score:.3f}")
    print("\nRunning until target reached (max 50 cycles)...")

    summary = await coordinator.run_continuous(
        cycles=50,
        target_harmony=target,
        cycle_delay=0.01
    )

    print(f"\n{'SUCCESS!' if summary['target_reached'] else 'TARGET NOT REACHED'}")
    print(f"Final harmony: {summary['final_harmony']:.3f}")
    print(f"Cycles needed: {summary['cycles_run']}")


if __name__ == "__main__":
    print("\nSelect example to run:")
    print("  1. Basic demonstration (default)")
    print("  2. Targeted harmony run")
    print()

    choice = input("Enter choice (1/2): ").strip() or "1"

    if choice == "2":
        asyncio.run(run_targeted_harmony())
    else:
        asyncio.run(run_basic_example())
