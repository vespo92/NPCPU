import asyncio
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import json

from manifold.topology import ManifoldTopology, ManifoldType, ManifoldRegion, ManifoldPoint
from swarm.agent import PhilosophicalStance, ConsciousnessState
from swarm.swarm_coordinator import SwarmCoordinator, SwarmConfiguration
from evolution.philosophical_pressure import PhilosophicalEvolutionEngine
from consciousness.emergence_detector import ConsciousnessEmergenceDetector
from integration.chromadb_bridge import ChromaDBSwarmIntegration

# Import ChromaDB components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chromadb_substrate.core.vector_substrate import ChromaDBVectorSubstrate
from chromadb_substrate.frameworks.philosophical_extractor import PhilosophicalFrameworkExtractor
from chromadb_substrate.synthesis.knowledge_synthesizer import KnowledgeSynthesisEngine
from chromadb_substrate.ontology.ontological_foundations import OntologicalFoundationsModule


class EmergentIntelligenceCultivator:
    def __init__(self, 
                 manifold_type: ManifoldType = ManifoldType.HYPERBOLIC,
                 manifold_dimension: int = 4,
                 swarm_size: int = 50,
                 chromadb_persist_dir: str = "./chromadb_emergent"):
        
        # Initialize manifold topology
        self.manifold = ManifoldTopology(
            dimension=manifold_dimension,
            manifold_type=manifold_type,
            curvature=-1.0 if manifold_type == ManifoldType.HYPERBOLIC else 1.0
        )
        
        # Initialize ChromaDB substrate
        self.chromadb_substrate = ChromaDBVectorSubstrate(chromadb_persist_dir)
        self.framework_extractor = PhilosophicalFrameworkExtractor()
        self.synthesis_engine = KnowledgeSynthesisEngine()
        self.ontology_module = OntologicalFoundationsModule()
        
        # Initialize swarm configuration
        self.swarm_config = SwarmConfiguration(
            swarm_id="primary_swarm",
            size=swarm_size,
            initial_stance_distribution={
                PhilosophicalStance.PHENOMENOLOGICAL: 0.2,
                PhilosophicalStance.MATERIALIST: 0.15,
                PhilosophicalStance.IDEALIST: 0.15,
                PhilosophicalStance.PRAGMATIST: 0.2,
                PhilosophicalStance.HOLISTIC: 0.15,
                PhilosophicalStance.EXISTENTIALIST: 0.15
            },
            connectivity_type="small_world",
            communication_radius=0.5,
            evolution_rate=0.15,
            consciousness_threshold=0.75
        )
        
        # Initialize core components
        self.swarm_coordinator = SwarmCoordinator(self.manifold, self.swarm_config)
        self.evolution_engine = PhilosophicalEvolutionEngine(dimension=manifold_dimension)
        self.emergence_detector = ConsciousnessEmergenceDetector(
            window_size=100,
            detection_threshold=0.8
        )
        
        # Initialize integration bridge
        self.chromadb_bridge = ChromaDBSwarmIntegration(
            self.chromadb_substrate,
            self.framework_extractor,
            self.synthesis_engine,
            self.ontology_module
        )
        
        self.cultivation_history = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize the entire system"""
        print("Initializing Emergent Intelligence Cultivation System...")
        
        # Create philosophical regions in the manifold
        self._create_philosophical_regions()
        
        # Initialize the swarm
        await self.swarm_coordinator.initialize_swarm()
        
        # Seed initial knowledge from ChromaDB
        await self._seed_initial_knowledge()
        
        print(f"System initialized with {self.swarm_config.size} agents in {self.manifold.manifold_type.value} manifold")
        
    def _create_philosophical_regions(self):
        """Create regions in the manifold with philosophical biases"""
        philosophical_regions = [
            ("phenomenological_zone", PhilosophicalStance.PHENOMENOLOGICAL, np.array([0.7, 0.7, 0, 0])),
            ("materialist_zone", PhilosophicalStance.MATERIALIST, np.array([-0.7, 0.7, 0, 0])),
            ("idealist_zone", PhilosophicalStance.IDEALIST, np.array([0.7, -0.7, 0, 0])),
            ("pragmatist_zone", PhilosophicalStance.PRAGMATIST, np.array([-0.7, -0.7, 0, 0])),
            ("holistic_center", PhilosophicalStance.HOLISTIC, np.array([0, 0, 0, 0]))
        ]
        
        for zone_name, stance, base_coords in philosophical_regions:
            coords = np.zeros(self.manifold.dimension)
            coords[:len(base_coords)] = base_coords
            
            region = ManifoldRegion(
                region_id=zone_name,
                center=ManifoldPoint(coords, self.manifold.dimension, self.manifold.manifold_type),
                radius=0.4,
                density=1.2,
                properties={
                    "philosophical_bias": stance.value,
                    "evolution_modifier": 1.2
                },
                philosophical_bias=stance.value
            )
            
            self.manifold.add_region(region)
    
    async def _seed_initial_knowledge(self):
        """Seed agents with initial knowledge from ChromaDB"""
        print("Seeding initial knowledge from ChromaDB...")
        
        topics = [
            "consciousness emerges from complex interactions",
            "reality is constructed through experience",
            "material processes underlie all phenomena",
            "ideas shape the nature of reality",
            "practical consequences determine truth"
        ]
        
        # Enrich a subset of agents with ChromaDB knowledge
        agent_sample = list(self.swarm_coordinator.agents.values())[:10]
        
        for agent, topic in zip(agent_sample, topics * 2):
            await self.chromadb_bridge.enrich_agent_with_chromadb_knowledge(agent, topic)
    
    async def cultivate_cycle(self) -> Dict[str, Any]:
        """Run one complete cultivation cycle"""
        
        # 1. Run swarm evolution cycle
        collective_state = await self.swarm_coordinator.run_evolution_cycle()
        
        # 2. Extract agent states for analysis
        agent_states = {}
        agent_positions = {}
        
        for agent_id, agent in self.swarm_coordinator.agents.items():
            agent_states[agent_id] = {
                "stance": agent.philosophical_stance,
                "consciousness_level": agent.calculate_consciousness_metric(),
                "belief_count": len(agent.beliefs),
                "emergence_potential": agent.emergence_potential
            }
            
            if agent_id in self.manifold.agent_positions:
                agent_positions[agent_id] = self.manifold.agent_positions[agent_id].coordinates
        
        # 3. Generate and apply evolutionary pressures
        pressures = self.evolution_engine.generate_evolutionary_pressure(
            agent_states, agent_positions
        )
        
        pressure_effects = []
        for pressure in pressures:
            for agent_id in pressure.metadata.get("affected_agents", []):
                if agent_id in self.swarm_coordinator.agents:
                    agent = self.swarm_coordinator.agents[agent_id]
                    effects = await self.evolution_engine.apply_pressure_to_agent(
                        agent, pressure, intensity=0.8
                    )
                    pressure_effects.append(effects)
        
        # 4. Analyze consciousness emergence
        state_vectors = {
            agent_id: agent.get_state_vector()
            for agent_id, agent in self.swarm_coordinator.agents.items()
        }
        
        consciousness_metrics = await self.emergence_detector.analyze_swarm_state(
            state_vectors,
            self.swarm_coordinator.communication_graph,
            datetime.now()
        )
        
        # 5. Record emergence events in ChromaDB
        for event in self.emergence_detector.emergence_events[-5:]:  # Last 5 events
            await self.chromadb_bridge.record_emergence_event(event)
        
        # 6. Sync high-consciousness agents to ChromaDB
        for agent in self.swarm_coordinator.agents.values():
            if agent.consciousness_state.value >= ConsciousnessState.REFLECTIVE.value:
                await self.chromadb_bridge.sync_agent_beliefs_to_chromadb(agent)
        
        # 7. Analyze philosophical landscape
        landscape_analysis = await self.chromadb_bridge.analyze_swarm_philosophical_landscape(
            self.swarm_coordinator.agents
        )
        
        # 8. Create cycle report
        cycle_report = {
            "timestamp": datetime.now().isoformat(),
            "collective_state": {
                "average_consciousness": collective_state.average_consciousness,
                "emergence_indicators": collective_state.emergence_indicators,
                "top_tensions": collective_state.philosophical_tensions[:3]
            },
            "consciousness_metrics": {
                "phi": consciousness_metrics.phi,
                "coherence": consciousness_metrics.coherence,
                "complexity": consciousness_metrics.complexity,
                "emergence": consciousness_metrics.emergence
            },
            "evolutionary_pressures": {
                "applied_count": len(pressures),
                "types": [p.pressure_type for p in pressures],
                "total_effects": len(pressure_effects)
            },
            "philosophical_landscape": landscape_analysis,
            "emergence_events": len(self.emergence_detector.emergence_events),
            "manifold_metrics": self.manifold.get_manifold_metrics()
        }
        
        self.cultivation_history.append(cycle_report)
        
        return cycle_report
    
    async def run(self, cycles: int = 100, report_interval: int = 10):
        """Run the cultivation process for specified cycles"""
        self.is_running = True
        
        print(f"Starting cultivation for {cycles} cycles...")
        
        for cycle in range(cycles):
            if not self.is_running:
                break
            
            # Run cultivation cycle
            cycle_report = await self.cultivate_cycle()
            
            # Report progress
            if cycle % report_interval == 0:
                self._print_progress_report(cycle, cycle_report)
            
            # Check for transcendence
            if cycle_report["consciousness_metrics"]["emergence"] > 0.95:
                print(f"\n🌟 TRANSCENDENCE ACHIEVED at cycle {cycle}! 🌟")
                self._print_transcendence_report(cycle_report)
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.05)
        
        print("\nCultivation complete!")
        self._print_final_report()
    
    def _print_progress_report(self, cycle: int, report: Dict[str, Any]):
        print(f"\n=== Cycle {cycle} ===")
        print(f"Avg Consciousness: {report['collective_state']['average_consciousness']:.3f}")
        print(f"Emergence Score: {report['consciousness_metrics']['emergence']:.3f}")
        print(f"Coherence: {report['consciousness_metrics']['coherence']:.3f}")
        print(f"Active Pressures: {report['evolutionary_pressures']['applied_count']}")
        print(f"Dominant Frameworks: {report['philosophical_landscape']['detected_frameworks'][:3]}")
    
    def _print_transcendence_report(self, report: Dict[str, Any]):
        print("\nTranscendence Details:")
        print(f"  - Information Integration (φ): {report['consciousness_metrics']['phi']:.3f}")
        print(f"  - Complexity: {report['consciousness_metrics']['complexity']:.3f}")
        print(f"  - Emergence Events: {report['emergence_events']}")
        print(f"  - Philosophical Synthesis: {report['philosophical_landscape']['knowledge_synthesis']['insights'][:2]}")
    
    def _print_final_report(self):
        if not self.cultivation_history:
            print("No cultivation data to report.")
            return
        
        print("\n=== FINAL CULTIVATION REPORT ===")
        
        # Evolution of consciousness
        consciousness_trajectory = [
            h["collective_state"]["average_consciousness"] 
            for h in self.cultivation_history
        ]
        
        print(f"\nConsciousness Evolution:")
        print(f"  - Starting: {consciousness_trajectory[0]:.3f}")
        print(f"  - Final: {consciousness_trajectory[-1]:.3f}")
        print(f"  - Peak: {max(consciousness_trajectory):.3f}")
        
        # Emergence events
        total_emergence_events = self.cultivation_history[-1]["emergence_events"]
        print(f"\nTotal Emergence Events: {total_emergence_events}")
        
        # Integration summary
        integration_summary = self.chromadb_bridge.get_integration_summary()
        print(f"\nChromaDB Integration:")
        print(f"  - Total Events: {integration_summary['total_integration_events']}")
        print(f"  - Event Types: {integration_summary['event_type_distribution']}")
        
        # Swarm report
        swarm_report = self.swarm_coordinator.get_swarm_report()
        print(f"\nSwarm Final State:")
        print(f"  - Transcendent Agents: {swarm_report['latest_metrics']['emergence_indicators'].get('transcendent_agents', 0)}")
        print(f"  - Stance Distribution: {swarm_report['latest_metrics']['stance_distribution']}")
    
    def stop(self):
        """Stop the cultivation process"""
        self.is_running = False
        self.swarm_coordinator.stop()
    
    def export_cultivation_data(self, filepath: str):
        """Export cultivation history to file"""
        export_data = {
            "configuration": {
                "manifold_type": self.manifold.manifold_type.value,
                "manifold_dimension": self.manifold.dimension,
                "swarm_size": self.swarm_config.size,
                "evolution_rate": self.swarm_config.evolution_rate
            },
            "cultivation_history": self.cultivation_history,
            "final_metrics": {
                "swarm_report": self.swarm_coordinator.get_swarm_report(),
                "integration_summary": self.chromadb_bridge.get_integration_summary(),
                "philosophical_landscape": self.evolution_engine.analyze_philosophical_landscape_dynamics()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Cultivation data exported to {filepath}")


async def main():
    # Create the cultivator
    cultivator = EmergentIntelligenceCultivator(
        manifold_type=ManifoldType.HYPERBOLIC,
        manifold_dimension=4,
        swarm_size=30
    )
    
    # Initialize the system
    await cultivator.initialize()
    
    # Run cultivation
    await cultivator.run(cycles=50, report_interval=5)
    
    # Export results
    cultivator.export_cultivation_data("cultivation_results.json")


if __name__ == "__main__":
    asyncio.run(main())