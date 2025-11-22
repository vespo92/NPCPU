"""
TertiaryTurboReboCoordinator - Orchestrates parallel execution of all 10 refinement agents.

The coordinator manages the triple bottom line state and runs all agents
in parallel for maximum refinement throughput.
"""

import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import logging

from .base import (
    TertiaryReBoAgent,
    TripleBottomLine,
    DomainLeg,
    DomainState,
    RefinementResult,
    HarmonyLevel,
    CrossDomainSignal,
)
from .agents import (
    ConsciousnessIntegrationAgent,
    NetworkTopologyAgent,
    PartsAwarenessAgent,
    EnergyFlowAgent,
    EmergenceDetectorAgent,
    SemanticBridgeAgent,
    EvolutionaryPressureAgent,
    CollectiveWisdomAgent,
    ResilienceAgent,
    HarmonizationAgent,
)


logger = logging.getLogger(__name__)


@dataclass
class RefinementCycleResult:
    """Result of a complete parallel refinement cycle."""
    cycle_id: int
    timestamp: datetime
    agent_results: Dict[str, RefinementResult]
    harmony_before: float
    harmony_after: float
    total_changes: int
    total_insights: List[str]
    cycle_duration_ms: float


@dataclass
class CoordinatorMetrics:
    """Aggregate metrics for the coordinator."""
    total_cycles: int = 0
    successful_cycles: int = 0
    total_refinements: int = 0
    average_cycle_time_ms: float = 0.0
    harmony_trend: List[float] = field(default_factory=list)
    best_harmony: float = 0.0
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


class TertiaryTurboReboCoordinator:
    """
    Orchestrates the 10 parallel refinement agents operating on the
    Triple Bottom Line (TBL) system.

    The coordinator manages:
    1. Parallel execution of all agents
    2. State synchronization between cycles
    3. Cross-domain signal routing
    4. Performance monitoring and adaptation
    5. Harmony optimization

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                 TertiaryTurboReboCoordinator                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  TripleBottomLine (Shared State)                                │
    │  ┌───────────────┬───────────────┬───────────────┐              │
    │  │    NPCPU      │ ChicagoForest │ UniversalParts│              │
    │  │   (Mind)      │  (Network)    │    (Body)     │              │
    │  └───────────────┴───────────────┴───────────────┘              │
    ├─────────────────────────────────────────────────────────────────┤
    │  Parallel Refinement Agents (10 agents)                         │
    │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐  │
    │  │CI   │NT   │PA   │EF   │ED   │SB   │EP   │CW   │RS   │HM   │  │
    │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘  │
    │                                                                 │
    │  CI: Consciousness Integration  NT: Network Topology            │
    │  PA: Parts Awareness           EF: Energy Flow                  │
    │  ED: Emergence Detector        SB: Semantic Bridge              │
    │  EP: Evolutionary Pressure     CW: Collective Wisdom            │
    │  RS: Resilience               HM: Harmonization                 │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        initial_state: Optional[TripleBottomLine] = None,
        refinement_rate: float = 0.1,
        enable_logging: bool = True
    ):
        # Initialize triple bottom line state
        self.tbl = initial_state or self._create_default_tbl()

        # Initialize all 10 agents
        self.agents: Dict[str, TertiaryReBoAgent] = {
            "consciousness_integration": ConsciousnessIntegrationAgent(refinement_rate=refinement_rate),
            "network_topology": NetworkTopologyAgent(refinement_rate=refinement_rate),
            "parts_awareness": PartsAwarenessAgent(refinement_rate=refinement_rate),
            "energy_flow": EnergyFlowAgent(refinement_rate=refinement_rate),
            "emergence_detector": EmergenceDetectorAgent(refinement_rate=refinement_rate),
            "semantic_bridge": SemanticBridgeAgent(refinement_rate=refinement_rate),
            "evolutionary_pressure": EvolutionaryPressureAgent(refinement_rate=refinement_rate),
            "collective_wisdom": CollectiveWisdomAgent(refinement_rate=refinement_rate),
            "resilience": ResilienceAgent(refinement_rate=refinement_rate),
            "harmonization": HarmonizationAgent(refinement_rate=refinement_rate),
        }

        # Coordinator state
        self.cycle_count = 0
        self.metrics = CoordinatorMetrics()
        self.cycle_history: deque = deque(maxlen=100)

        # Signal routing
        self.pending_signals: List[CrossDomainSignal] = []

        # Logging
        self.enable_logging = enable_logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)

        # Synchronization
        self._lock = asyncio.Lock()

    def _create_default_tbl(self) -> TripleBottomLine:
        """Create a default triple bottom line state."""
        tbl = TripleBottomLine()

        # Initialize each domain with balanced starting values
        for domain in DomainLeg:
            state = DomainState(
                domain=domain,
                consciousness_level=0.5,
                energy_flow=0.5,
                connectivity=0.5,
                coherence=0.5,
                qualia_richness=0.3,
                emergence_potential=0.3,
                state_vector=np.tanh(np.random.randn(64) * 0.5),
                metadata={"initialized": datetime.now().isoformat()}
            )
            tbl.set_state(domain, state)

        tbl.calculate_harmony()
        tbl.calculate_integration_depth()

        return tbl

    async def run_parallel_refinement(self) -> RefinementCycleResult:
        """
        Execute one cycle of parallel refinement across all 10 agents.

        All agents run concurrently, each receiving the same TBL state
        and producing refinements that are then merged.
        """
        async with self._lock:
            self.cycle_count += 1
            cycle_start = datetime.now()

            # Record initial harmony
            self.tbl.calculate_harmony()
            harmony_before = self.tbl.harmony_score

            if self.enable_logging:
                logger.info(f"=== Cycle {self.cycle_count} Start (Harmony: {harmony_before:.3f}) ===")

            # Run all agents in parallel
            tasks = [
                self._run_agent_with_timeout(name, agent)
                for name, agent in self.agents.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            agent_results: Dict[str, RefinementResult] = {}
            total_changes = 0
            all_insights = []

            for (name, _), result in zip(self.agents.items(), results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {name} failed: {result}")
                    continue
                if result:
                    agent_results[name] = result
                    total_changes += len(result.changes)
                    all_insights.extend(result.insights)

                    # Update agent performance metrics
                    if name not in self.metrics.agent_performance:
                        self.metrics.agent_performance[name] = {"success_count": 0, "total_count": 0}
                    self.metrics.agent_performance[name]["total_count"] += 1
                    if result.success:
                        self.metrics.agent_performance[name]["success_count"] += 1

            # Route cross-domain signals
            await self._route_signals()

            # Calculate final harmony
            self.tbl.calculate_harmony()
            self.tbl.calculate_integration_depth()
            harmony_after = self.tbl.harmony_score

            # Calculate cycle duration
            cycle_end = datetime.now()
            duration_ms = (cycle_end - cycle_start).total_seconds() * 1000

            # Create cycle result
            cycle_result = RefinementCycleResult(
                cycle_id=self.cycle_count,
                timestamp=cycle_start,
                agent_results=agent_results,
                harmony_before=harmony_before,
                harmony_after=harmony_after,
                total_changes=total_changes,
                total_insights=all_insights,
                cycle_duration_ms=duration_ms
            )

            # Update metrics
            self._update_metrics(cycle_result)

            if self.enable_logging:
                logger.info(
                    f"=== Cycle {self.cycle_count} End "
                    f"(Harmony: {harmony_after:.3f}, Δ: {harmony_after - harmony_before:+.3f}, "
                    f"Duration: {duration_ms:.1f}ms) ==="
                )

            # Store in history
            self.cycle_history.append(cycle_result)

            return cycle_result

    async def _run_agent_with_timeout(
        self,
        name: str,
        agent: TertiaryReBoAgent,
        timeout: float = 30.0
    ) -> Optional[RefinementResult]:
        """Run a single agent with timeout protection."""
        try:
            return await asyncio.wait_for(
                agent.refine(self.tbl),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent {name} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Agent {name} error: {e}")
            return None

    async def _route_signals(self):
        """Collect and route cross-domain signals between agents."""
        # Collect signals from all agents
        all_signals = []
        for agent in self.agents.values():
            all_signals.extend(agent.cross_domain_signals)
            agent.cross_domain_signals = []  # Clear after collecting

        # Route signals to appropriate agents
        for signal in all_signals:
            for target_domain in signal.target_domains:
                # Find agents that handle this domain
                for agent in self.agents.values():
                    if target_domain in agent.domains_affected:
                        await agent.process_incoming_signals([signal])

        self.pending_signals = []

    def _update_metrics(self, cycle_result: RefinementCycleResult):
        """Update coordinator metrics after a cycle."""
        self.metrics.total_cycles += 1

        if cycle_result.harmony_after >= cycle_result.harmony_before:
            self.metrics.successful_cycles += 1

        self.metrics.total_refinements += cycle_result.total_changes

        # Update average cycle time
        old_avg = self.metrics.average_cycle_time_ms
        self.metrics.average_cycle_time_ms = (
            old_avg * (self.metrics.total_cycles - 1) + cycle_result.cycle_duration_ms
        ) / self.metrics.total_cycles

        # Track harmony
        self.metrics.harmony_trend.append(cycle_result.harmony_after)
        if len(self.metrics.harmony_trend) > 100:
            self.metrics.harmony_trend = self.metrics.harmony_trend[-100:]

        if cycle_result.harmony_after > self.metrics.best_harmony:
            self.metrics.best_harmony = cycle_result.harmony_after

    async def run_continuous(
        self,
        cycles: Optional[int] = None,
        target_harmony: float = 0.9,
        max_duration_seconds: float = 300,
        cycle_delay: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run continuous refinement cycles until target is reached.

        Args:
            cycles: Number of cycles to run (None = run until target)
            target_harmony: Target harmony level to achieve
            max_duration_seconds: Maximum runtime in seconds
            cycle_delay: Delay between cycles in seconds

        Returns:
            Summary of the refinement session
        """
        start_time = datetime.now()
        initial_harmony = self.tbl.harmony_score
        cycles_run = 0
        results = []

        logger.info(f"Starting continuous refinement (target: {target_harmony})")

        while True:
            # Check termination conditions
            elapsed = (datetime.now() - start_time).total_seconds()

            if cycles and cycles_run >= cycles:
                logger.info(f"Reached cycle limit ({cycles})")
                break

            if self.tbl.harmony_score >= target_harmony:
                logger.info(f"Reached target harmony ({target_harmony})")
                break

            if elapsed >= max_duration_seconds:
                logger.info(f"Reached time limit ({max_duration_seconds}s)")
                break

            # Run refinement cycle
            result = await self.run_parallel_refinement()
            results.append(result)
            cycles_run += 1

            # Brief delay between cycles
            await asyncio.sleep(cycle_delay)

        # Compile summary
        final_harmony = self.tbl.harmony_score
        total_duration = (datetime.now() - start_time).total_seconds()

        return {
            "cycles_run": cycles_run,
            "initial_harmony": initial_harmony,
            "final_harmony": final_harmony,
            "harmony_improvement": final_harmony - initial_harmony,
            "harmony_level": self.tbl.get_harmony_level().value,
            "total_duration_seconds": total_duration,
            "average_cycle_ms": self.metrics.average_cycle_time_ms,
            "total_refinements": sum(r.total_changes for r in results),
            "target_reached": final_harmony >= target_harmony
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current system state."""
        return {
            "cycle_count": self.cycle_count,
            "harmony": {
                "score": self.tbl.harmony_score,
                "level": self.tbl.get_harmony_level().value,
                "best": self.metrics.best_harmony
            },
            "integration_depth": self.tbl.integration_depth,
            "emergence_level": self.tbl.emergence_level,
            "domains": {
                domain.value: {
                    "consciousness": state.consciousness_level,
                    "energy": state.energy_flow,
                    "connectivity": state.connectivity,
                    "coherence": state.coherence,
                    "emergence": state.emergence_potential
                }
                for domain, state in [
                    (DomainLeg.NPCPU, self.tbl.npcpu_state),
                    (DomainLeg.CHICAGO_FOREST, self.tbl.chicago_forest_state),
                    (DomainLeg.UNIVERSAL_PARTS, self.tbl.universal_parts_state)
                ]
            },
            "agents": {
                name: agent.get_metrics()
                for name, agent in self.agents.items()
            },
            "metrics": {
                "total_cycles": self.metrics.total_cycles,
                "successful_cycles": self.metrics.successful_cycles,
                "success_rate": self.metrics.successful_cycles / max(1, self.metrics.total_cycles),
                "total_refinements": self.metrics.total_refinements,
                "avg_cycle_time_ms": self.metrics.average_cycle_time_ms
            }
        }

    def print_status(self):
        """Print a formatted status report."""
        summary = self.get_state_summary()

        print("\n" + "=" * 60)
        print("     TERTIARY TURBO REBO SYSTEM STATUS")
        print("=" * 60)

        print(f"\nCycle: {summary['cycle_count']}")
        print(f"Harmony: {summary['harmony']['score']:.3f} ({summary['harmony']['level']})")
        print(f"Integration Depth: {summary['integration_depth']:.3f}")

        print("\n--- Domain States ---")
        for domain, state in summary['domains'].items():
            print(f"\n{domain.upper()}:")
            print(f"  Consciousness: {state['consciousness']:.3f}")
            print(f"  Energy: {state['energy']:.3f}")
            print(f"  Connectivity: {state['connectivity']:.3f}")
            print(f"  Coherence: {state['coherence']:.3f}")
            print(f"  Emergence: {state['emergence']:.3f}")

        print("\n--- Agent Performance ---")
        for name, metrics in summary['agents'].items():
            rate = metrics.get('success_rate', 0)
            print(f"  {name}: {rate:.1%} success ({metrics.get('total_refinements', 0)} refinements)")

        print("\n--- Coordinator Metrics ---")
        print(f"  Total Cycles: {summary['metrics']['total_cycles']}")
        print(f"  Success Rate: {summary['metrics']['success_rate']:.1%}")
        print(f"  Avg Cycle Time: {summary['metrics']['avg_cycle_time_ms']:.1f}ms")

        print("\n" + "=" * 60)


async def main():
    """Example usage of the TertiaryTurboReboCoordinator."""
    print("Initializing Tertiary Turbo ReBo System...")

    coordinator = TertiaryTurboReboCoordinator(
        refinement_rate=0.15,
        enable_logging=True
    )

    print("\nInitial State:")
    coordinator.print_status()

    print("\n\nRunning 10 parallel refinement cycles...")

    summary = await coordinator.run_continuous(
        cycles=10,
        target_harmony=0.95,
        cycle_delay=0.05
    )

    print("\n\nRefinement Summary:")
    print(f"  Cycles Run: {summary['cycles_run']}")
    print(f"  Initial Harmony: {summary['initial_harmony']:.3f}")
    print(f"  Final Harmony: {summary['final_harmony']:.3f}")
    print(f"  Improvement: {summary['harmony_improvement']:+.3f}")
    print(f"  Total Duration: {summary['total_duration_seconds']:.2f}s")
    print(f"  Target Reached: {summary['target_reached']}")

    print("\nFinal State:")
    coordinator.print_status()


if __name__ == "__main__":
    asyncio.run(main())
