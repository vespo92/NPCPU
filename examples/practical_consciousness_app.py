"""
Practical Consciousness Application

Demonstrates real-world usage of NPCPU protocol architecture:
- YAML-configured consciousness models
- ChromaDB storage adapter
- Multi-agent coordination
- Consciousness evolution tracking

This is a complete, runnable example showing immediate value.
"""

import asyncio
import sys
import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import NPCPU components
from factory.consciousness_factory import ConsciousnessFactory, load_consciousness_model
from protocols.consciousness import GradedConsciousness

# Try to import ChromaDB adapter
try:
    from adapters.chromadb_adapter import ChromaDBAdapter
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available - using in-memory fallback")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Agent Implementation
# ============================================================================

@dataclass
class Experience:
    """Agent experience"""
    timestamp: float
    stimulus: str
    action: str
    success: bool
    consciousness_score: float


class ConsciousAgent:
    """
    Agent with NPCPU consciousness.

    Features:
    - YAML-configured consciousness
    - Experience storage in vector DB
    - Consciousness evolution
    - Performance tracking
    """

    def __init__(
        self,
        agent_id: str,
        consciousness: GradedConsciousness,
        storage=None
    ):
        self.agent_id = agent_id
        self.consciousness = consciousness
        self.storage = storage
        self.experiences: List[Experience] = []
        self.success_rate = 0.0

    async def perceive_and_act(self, stimulus: str) -> str:
        """
        Perceive stimulus and take action.

        Action quality depends on consciousness capabilities.
        """
        scores = self.consciousness.get_capability_scores()

        # Perception quality affects action choice
        perception_quality = scores.get("perception_fidelity", 0.5)

        # Reaction speed affects decision time
        reaction_speed = scores.get("reaction_speed", 0.5)

        # Simulate perception delay
        perception_delay = (1.0 - reaction_speed) * 0.5
        await asyncio.sleep(perception_delay)

        # Choose action based on stimulus and capabilities
        if "explore" in stimulus.lower():
            curiosity = scores.get("curiosity", 0.5)
            if curiosity > 0.7:
                action = "actively_explore"
                success = random.random() < (0.7 + perception_quality * 0.3)
            else:
                action = "cautiously_explore"
                success = random.random() < 0.6
        elif "analyze" in stimulus.lower():
            pattern_recognition = scores.get("pattern_recognition", 0.5)
            action = "analyze_pattern"
            success = random.random() < pattern_recognition
        elif "remember" in stimulus.lower():
            recall = scores.get("memory_recall_accuracy", 0.5)
            action = "recall_memory"
            success = random.random() < recall
        else:
            action = "observe"
            success = random.random() < perception_quality

        # Store experience
        experience = Experience(
            timestamp=time.time(),
            stimulus=stimulus,
            action=action,
            success=success,
            consciousness_score=self.consciousness.overall_consciousness_score()
        )
        self.experiences.append(experience)

        # Update success rate
        total_success = sum(1 for e in self.experiences if e.success)
        self.success_rate = total_success / len(self.experiences)

        # Store in vector DB if available
        if self.storage:
            await self._store_experience(experience)

        return action

    async def _store_experience(self, experience: Experience):
        """Store experience in vector DB"""
        # Create simple embedding (in production, use real embeddings)
        embedding = self._create_embedding(experience)

        metadata = {
            "agent_id": self.agent_id,
            "timestamp": experience.timestamp,
            "stimulus": experience.stimulus,
            "action": experience.action,
            "success": experience.success,
            "consciousness_score": experience.consciousness_score
        }

        try:
            await self.storage.store(
                collection="agent_experiences",
                id=f"{self.agent_id}_{experience.timestamp}",
                vector=embedding,
                metadata=metadata,
                document=f"{experience.stimulus} -> {experience.action}"
            )
        except Exception as e:
            logger.warning(f"Failed to store experience: {e}")

    def _create_embedding(self, experience: Experience) -> List[float]:
        """Create simple embedding (placeholder for real embeddings)"""
        # In production, use actual embedding model (OpenAI, sentence-transformers, etc.)
        # This is a simple hash-based embedding for demonstration
        text = f"{experience.stimulus} {experience.action}"
        hash_val = hash(text)

        # Create deterministic vector from hash
        random.seed(hash_val)
        embedding = [random.random() for _ in range(384)]

        return embedding

    async def evolve_consciousness(self, performance_weight: float = 0.3):
        """
        Evolve consciousness based on performance.

        Successful agents improve, unsuccessful agents adapt.
        """
        if len(self.experiences) < 5:
            return  # Not enough data

        # Analyze performance
        recent_experiences = self.experiences[-10:]
        recent_success_rate = sum(1 for e in recent_experiences if e.success) / len(recent_experiences)

        # Determine evolution strategy
        if recent_success_rate > 0.7:
            # Doing well - slight improvements
            logger.info(f"{self.agent_id}: Doing well ({recent_success_rate:.2%}), refining capabilities")
            evolution_factor = 1.05
        elif recent_success_rate < 0.3:
            # Struggling - need different approach
            logger.info(f"{self.agent_id}: Struggling ({recent_success_rate:.2%}), adapting")
            evolution_factor = 0.95  # Reduce some capabilities, redistribute
        else:
            # Moderate performance - maintain
            logger.info(f"{self.agent_id}: Stable ({recent_success_rate:.2%})")
            return

        # Evolve capabilities
        current_scores = self.consciousness.get_capability_scores()
        new_scores = {}

        for capability, score in current_scores.items():
            # Random variation with bias toward evolution_factor
            variation = random.uniform(0.95, 1.05) * evolution_factor
            new_score = min(1.0, max(0.0, score * variation))
            new_scores[capability] = new_score

        # Create evolved consciousness
        evolved = GradedConsciousness(**new_scores)
        evolved.weights = self.consciousness.weights

        logger.info(
            f"{self.agent_id}: Evolved from {self.consciousness.overall_consciousness_score():.3f} "
            f"to {evolved.overall_consciousness_score():.3f}"
        )

        self.consciousness = evolved

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "consciousness_score": self.consciousness.overall_consciousness_score(),
            "model_name": getattr(self.consciousness, 'model_name', 'Unknown'),
            "experiences": len(self.experiences),
            "success_rate": self.success_rate,
            "capabilities": self.consciousness.get_capability_scores()
        }


# ============================================================================
# Application
# ============================================================================

class ConsciousnessApplication:
    """
    Complete NPCPU application demonstrating:
    - YAML consciousness models
    - ChromaDB storage
    - Multi-agent coordination
    - Consciousness evolution
    """

    def __init__(self, use_chromadb: bool = False):
        self.factory = ConsciousnessFactory()
        self.agents: List[ConsciousAgent] = []
        self.storage = None

        # Initialize storage if available
        if use_chromadb and CHROMADB_AVAILABLE:
            try:
                logger.info("Initializing ChromaDB storage...")
                self.storage = ChromaDBAdapter(path="./demo_chromadb")
                asyncio.run(self._init_storage())
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.storage = None

    async def _init_storage(self):
        """Initialize storage collections"""
        try:
            await self.storage.create_collection(
                name="agent_experiences",
                vector_dimension=384,
            )
            logger.info("ChromaDB collection created")
        except Exception as e:
            logger.warning(f"Collection may already exist: {e}")

    async def create_agent(
        self,
        agent_id: str,
        consciousness_model: str
    ) -> ConsciousAgent:
        """
        Create agent with specified consciousness model.

        Args:
            agent_id: Unique agent identifier
            consciousness_model: YAML model filename (e.g., "explorer.yaml")
        """
        logger.info(f"Creating agent {agent_id} with model {consciousness_model}")

        consciousness = self.factory.from_yaml(consciousness_model)

        agent = ConsciousAgent(
            agent_id=agent_id,
            consciousness=consciousness,
            storage=self.storage
        )

        self.agents.append(agent)

        logger.info(
            f"Created {agent_id}: {consciousness.model_name} "
            f"(consciousness: {consciousness.overall_consciousness_score():.3f})"
        )

        return agent

    async def run_simulation(
        self,
        stimuli: List[str],
        evolution_interval: int = 5
    ):
        """
        Run simulation with all agents.

        Args:
            stimuli: List of environmental stimuli
            evolution_interval: How often to evolve consciousness (# of steps)
        """
        logger.info(f"Starting simulation with {len(self.agents)} agents")

        for step, stimulus in enumerate(stimuli):
            logger.info(f"\n--- Step {step + 1}/{len(stimuli)}: {stimulus} ---")

            # All agents perceive and act
            tasks = [
                agent.perceive_and_act(stimulus)
                for agent in self.agents
            ]

            actions = await asyncio.gather(*tasks)

            # Report actions
            for agent, action in zip(self.agents, actions):
                logger.info(f"  {agent.agent_id}: {action}")

            # Evolve consciousness periodically
            if (step + 1) % evolution_interval == 0:
                logger.info("\nEvolving consciousness...")
                for agent in self.agents:
                    await agent.evolve_consciousness()

        # Final report
        logger.info("\n" + "="*70)
        logger.info("SIMULATION COMPLETE")
        logger.info("="*70)

        for agent in self.agents:
            status = agent.get_status()
            logger.info(
                f"\n{status['agent_id']}:\n"
                f"  Model: {status['model_name']}\n"
                f"  Consciousness: {status['consciousness_score']:.3f}\n"
                f"  Experiences: {status['experiences']}\n"
                f"  Success Rate: {status['success_rate']:.1%}"
            )

    async def compare_agents(self):
        """Compare agent performance"""
        logger.info("\n" + "="*70)
        logger.info("AGENT COMPARISON")
        logger.info("="*70)

        agents_by_performance = sorted(
            self.agents,
            key=lambda a: a.success_rate,
            reverse=True
        )

        for i, agent in enumerate(agents_by_performance, 1):
            logger.info(
                f"{i}. {agent.agent_id}: "
                f"Success Rate={agent.success_rate:.1%}, "
                f"Consciousness={agent.consciousness.overall_consciousness_score():.3f}"
            )

    async def demonstrate_model_comparison(self):
        """Demonstrate comparing consciousness models"""
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON")
        logger.info("="*70)

        # Compare explorer vs plant consciousness
        comparison = self.factory.compare_models("explorer.yaml", "plant_consciousness.yaml")

        logger.info(
            f"\n{comparison['model1']['name']} vs {comparison['model2']['name']}:\n"
            f"  Overall difference: {comparison['overall_difference']:+.3f}\n"
        )

        logger.info("Top capability differences:")
        diffs = sorted(
            comparison['capability_differences'].items(),
            key=lambda x: abs(x[1]['difference']),
            reverse=True
        )

        for capability, diff in diffs[:5]:
            logger.info(
                f"  {capability}: {diff['difference']:+.3f} "
                f"({diff['model1']:.2f} -> {diff['model2']:.2f})"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics"""
        total_experiences = sum(len(a.experiences) for a in self.agents)
        avg_success_rate = sum(a.success_rate for a in self.agents) / len(self.agents) if self.agents else 0
        avg_consciousness = sum(
            a.consciousness.overall_consciousness_score() for a in self.agents
        ) / len(self.agents) if self.agents else 0

        metrics = {
            "total_agents": len(self.agents),
            "total_experiences": total_experiences,
            "avg_success_rate": avg_success_rate,
            "avg_consciousness": avg_consciousness,
        }

        if self.storage and CHROMADB_AVAILABLE:
            storage_metrics = self.storage.get_metrics()
            metrics["storage"] = storage_metrics

        return metrics


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run the complete demonstration"""

    print("\n" + "="*70)
    print("PRACTICAL CONSCIOUSNESS APPLICATION")
    print("Demonstrating NPCPU Protocol Architecture")
    print("="*70)

    # Create application
    app = ConsciousnessApplication(use_chromadb=CHROMADB_AVAILABLE)

    # List available models
    factory = ConsciousnessFactory()
    available_models = factory.list_available_models()

    print(f"\nAvailable consciousness models: {', '.join(available_models)}")

    # Create agents with different consciousness models
    print("\nCreating agents with different consciousness models...")

    await app.create_agent("Explorer_1", "explorer.yaml")
    await app.create_agent("Plant_1", "plant_consciousness.yaml")
    await app.create_agent("Default_1", "default.yaml")

    # Demonstrate model comparison
    await app.demonstrate_model_comparison()

    # Run simulation
    stimuli = [
        "Explore new environment",
        "Analyze discovered pattern",
        "Remember previous location",
        "Explore unknown area",
        "Analyze resource distribution",
        "Remember important event",
        "Explore challenging terrain",
        "Analyze complex pattern",
        "Remember route home",
        "Explore final frontier"
    ]

    await app.run_simulation(stimuli, evolution_interval=3)

    # Compare agent performance
    await app.compare_agents()

    # Show metrics
    print("\n" + "="*70)
    print("APPLICATION METRICS")
    print("="*70)

    metrics = app.get_metrics()
    print(f"Total agents: {metrics['total_agents']}")
    print(f"Total experiences: {metrics['total_experiences']}")
    print(f"Average success rate: {metrics['avg_success_rate']:.1%}")
    print(f"Average consciousness: {metrics['avg_consciousness']:.3f}")

    if "storage" in metrics:
        storage = metrics["storage"]
        print(f"\nStorage metrics:")
        print(f"  Queries: {storage['queries']}")
        print(f"  Stores: {storage['stores']}")
        print(f"  Errors: {storage['errors']}")
        print(f"  Error rate: {storage['error_rate']:.1%}")
        print(f"  Avg latency: {storage['avg_latency_ms']:.2f}ms")

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    print("\nKey Takeaways:")
    print("✓ Consciousness models loaded from YAML")
    print("✓ Multiple agents with different consciousness types")
    print("✓ Experiences stored in vector database (if ChromaDB available)")
    print("✓ Consciousness evolved based on performance")
    print("✓ Agents compared quantitatively")
    print("\nThis demonstrates immediate practical value of NPCPU protocols!")


if __name__ == "__main__":
    asyncio.run(main())
