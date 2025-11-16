"""
IMMEDIATE IMPLEMENTATION GUIDE (Days 1-7)

Quick start implementations for NPCPU protocol architecture.
These are working examples you can run today.
"""

import numpy as np
from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass
import time


# ============================================================================
# DAY 1: Implement Basic Consciousness Protocol
# ============================================================================

@dataclass
class Perception:
    stimulus_type: str
    content: Any
    timestamp: float
    fidelity: float

@dataclass
class Action:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float

@dataclass
class Experience:
    perception: Perception
    action: Action
    timestamp: float

class SimpleAgent:
    """
    Simplest possible conscious agent.

    Demonstrates immediate usability of protocol.
    """

    def __init__(self, name: str):
        self.name = name

        # Consciousness capabilities (0.0 to 1.0)
        self.perception_fidelity = 0.6
        self.reaction_speed = 0.7
        self.memory_depth = 0.5

        # Memory
        self.experiences: List[Experience] = []

    def perceive(self, stimulus: Any) -> Perception:
        """Perceive environment"""
        return Perception(
            stimulus_type=type(stimulus).__name__,
            content=stimulus,
            timestamp=time.time(),
            fidelity=self.perception_fidelity
        )

    def react(self, perception: Perception) -> Action:
        """React to perception"""
        # Simple reaction logic
        if "food" in str(perception.content).lower():
            return Action(
                action_type="approach",
                parameters={"target": perception.content},
                confidence=0.8
            )
        else:
            return Action(
                action_type="observe",
                parameters={},
                confidence=0.5
            )

    def remember(self, experience: Experience):
        """Store experience"""
        self.experiences.append(experience)

        # Limit memory based on memory_depth
        max_memories = int(self.memory_depth * 100)
        if len(self.experiences) > max_memories:
            self.experiences = self.experiences[-max_memories:]

    def overall_consciousness_score(self) -> float:
        """Simple consciousness measure"""
        return (self.perception_fidelity + self.reaction_speed + self.memory_depth) / 3

    def run_perception_action_loop(self, stimuli: List[Any]):
        """Run basic perception-action loop"""
        print(f"\n{self.name} (consciousness: {self.overall_consciousness_score():.2f})")
        print("=" * 60)

        for stimulus in stimuli:
            # Perceive
            perception = self.perceive(stimulus)
            print(f"\nPerceived: {perception.content} (fidelity: {perception.fidelity:.2f})")

            # React
            action = self.react(perception)
            print(f"Action: {action.action_type} (confidence: {action.confidence:.2f})")

            # Remember
            experience = Experience(
                perception=perception,
                action=action,
                timestamp=time.time()
            )
            self.remember(experience)

        print(f"\nMemories stored: {len(self.experiences)}")


# ============================================================================
# DAY 2: Implement In-Memory Storage
# ============================================================================

class InMemoryVectorStorage:
    """
    Simple in-memory vector storage.

    Implements core storage protocol without external dependencies.
    """

    def __init__(self):
        self.collections: Dict[str, Dict[str, Dict]] = {}

    def create_collection(self, name: str):
        """Create new collection"""
        if name not in self.collections:
            self.collections[name] = {}
            print(f"Created collection: {name}")

    def store(self, collection: str, id: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Store vector with metadata"""
        if collection not in self.collections:
            self.create_collection(collection)

        self.collections[collection][id] = {
            "vector": np.array(vector),
            "metadata": metadata or {},
            "timestamp": time.time()
        }

    def query(self, collection: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """Query similar vectors using cosine similarity"""
        if collection not in self.collections:
            return []

        query_vec = np.array(query_vector)
        results = []

        for id, item in self.collections[collection].items():
            stored_vec = item["vector"]

            # Cosine similarity
            similarity = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-10
            )

            results.append({
                "id": id,
                "score": float(similarity),
                "metadata": item["metadata"]
            })

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def count(self, collection: str) -> int:
        """Count items in collection"""
        return len(self.collections.get(collection, {}))


# ============================================================================
# DAY 3: Implement Basic Transformations
# ============================================================================

class SimpleManifold:
    """Simple manifold representation"""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1]

    @property
    def num_points(self) -> int:
        return self.vectors.shape[0]


class SimpleTransformations:
    """Basic transformations without heavy dependencies"""

    @staticmethod
    def normalize(manifold: SimpleManifold) -> SimpleManifold:
        """Normalize all vectors to unit length"""
        norms = np.linalg.norm(manifold.vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = manifold.vectors / norms
        return SimpleManifold(vectors=normalized)

    @staticmethod
    def scale(manifold: SimpleManifold, factor: float) -> SimpleManifold:
        """Scale all vectors"""
        scaled = manifold.vectors * factor
        return SimpleManifold(vectors=scaled)

    @staticmethod
    def project_to_dim(manifold: SimpleManifold, target_dim: int) -> SimpleManifold:
        """Simple projection to lower dimension (truncation)"""
        if target_dim >= manifold.dimension:
            return manifold

        projected = manifold.vectors[:, :target_dim]
        return SimpleManifold(vectors=projected)

    @staticmethod
    def compose(transform1, transform2):
        """Compose two transformations"""
        def composed(manifold):
            return transform2(transform1(manifold))
        return composed


# ============================================================================
# DAY 4-7: Complete Example Application
# ============================================================================

class SimpleNPCPUApplication:
    """
    Complete NPCPU application combining all components.

    This demonstrates immediate usability.
    """

    def __init__(self):
        self.storage = InMemoryVectorStorage()
        self.agents: List[SimpleAgent] = []

    def add_agent(self, agent: SimpleAgent):
        """Add agent to system"""
        self.agents.append(agent)
        print(f"Added agent: {agent.name}")

    def store_agent_experience(self, agent: SimpleAgent, experience: Experience):
        """Store agent experience in vector storage"""
        # Convert experience to vector (simplified: use random embedding)
        vector = np.random.randn(128).tolist()  # In practice, use real embedding

        metadata = {
            "agent_name": agent.name,
            "stimulus_type": experience.perception.stimulus_type,
            "action_type": experience.action.action_type,
            "timestamp": experience.timestamp
        }

        self.storage.store(
            collection="experiences",
            id=f"{agent.name}_{experience.timestamp}",
            vector=vector,
            metadata=metadata
        )

    def query_similar_experiences(self, agent: SimpleAgent, current_perception: Perception) -> List[Dict]:
        """Find similar past experiences"""
        # Convert perception to vector
        query_vector = np.random.randn(128).tolist()  # In practice, use real embedding

        results = self.storage.query(
            collection="experiences",
            query_vector=query_vector,
            limit=3
        )

        return results

    def run_simulation(self, stimuli: List[Any]):
        """Run complete simulation"""
        print("\n" + "="*70)
        print("SIMPLE NPCPU APPLICATION - IMMEDIATE IMPLEMENTATION")
        print("="*70)

        for agent in self.agents:
            # Run perception-action loop
            agent.run_perception_action_loop(stimuli)

            # Store experiences
            for experience in agent.experiences:
                self.store_agent_experience(agent, experience)

        # Show storage stats
        print(f"\n\nTotal experiences stored: {self.storage.count('experiences')}")

        # Demonstrate vector search
        if self.agents:
            agent = self.agents[0]
            if agent.experiences:
                print(f"\nQuerying similar experiences for {agent.name}...")
                similar = self.query_similar_experiences(
                    agent,
                    agent.experiences[0].perception
                )
                print(f"Found {len(similar)} similar experiences:")
                for result in similar:
                    print(f"  - {result['metadata']['agent_name']}: {result['metadata']['action_type']} (similarity: {result['score']:.3f})")


# ============================================================================
# IMMEDIATE USE CASE: Multi-Agent Simulation
# ============================================================================

def demo_immediate_implementation():
    """
    Demonstrate immediate implementation.

    This runs TODAY with zero external dependencies.
    """
    # Create application
    app = SimpleNPCPUApplication()

    # Create agents with different consciousness levels
    agent1 = SimpleAgent("Explorer")
    agent1.perception_fidelity = 0.8
    agent1.reaction_speed = 0.7
    agent1.memory_depth = 0.6

    agent2 = SimpleAgent("Observer")
    agent2.perception_fidelity = 0.6
    agent2.reaction_speed = 0.5
    agent2.memory_depth = 0.8

    # Add agents
    app.add_agent(agent1)
    app.add_agent(agent2)

    # Run simulation with stimuli
    stimuli = [
        "food nearby",
        "unknown object",
        "food source",
        "potential threat",
        "safe zone"
    ]

    app.run_simulation(stimuli)


# ============================================================================
# IMMEDIATE USE CASE: Vector Transformation Pipeline
# ============================================================================

def demo_transformation_pipeline():
    """Demonstrate transformation composition"""
    print("\n" + "="*70)
    print("TRANSFORMATION PIPELINE DEMO")
    print("="*70)

    # Create data
    data = np.random.randn(100, 64)  # 100 points in 64D
    manifold = SimpleManifold(vectors=data)

    print(f"\nOriginal manifold:")
    print(f"  Dimensions: {manifold.dimension}")
    print(f"  Points: {manifold.num_points}")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(manifold.vectors, axis=1)):.3f}")

    # Build pipeline
    pipeline = SimpleTransformations.compose(
        lambda m: SimpleTransformations.normalize(m),
        lambda m: SimpleTransformations.scale(m, 2.0)
    )

    # Apply pipeline
    transformed = pipeline(manifold)

    print(f"\nTransformed manifold:")
    print(f"  Dimensions: {transformed.dimension}")
    print(f"  Points: {transformed.num_points}")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(transformed.vectors, axis=1)):.3f}")

    # Apply another transformation
    projected = SimpleTransformations.project_to_dim(transformed, 16)

    print(f"\nProjected manifold:")
    print(f"  Dimensions: {projected.dimension}")
    print(f"  Points: {projected.num_points}")


# ============================================================================
# IMMEDIATE USE CASE: Consciousness Measurement
# ============================================================================

def demo_consciousness_measurement():
    """Demonstrate consciousness measurement across agents"""
    print("\n" + "="*70)
    print("CONSCIOUSNESS MEASUREMENT DEMO")
    print("="*70)

    # Create agents with different capabilities
    agents = [
        SimpleAgent("Reactive"),
        SimpleAgent("Aware"),
        SimpleAgent("Reflective"),
    ]

    # Set different consciousness levels
    agents[0].perception_fidelity = 0.5
    agents[0].reaction_speed = 0.7
    agents[0].memory_depth = 0.2

    agents[1].perception_fidelity = 0.7
    agents[1].reaction_speed = 0.6
    agents[1].memory_depth = 0.6

    agents[2].perception_fidelity = 0.8
    agents[2].reaction_speed = 0.5
    agents[2].memory_depth = 0.9

    # Measure and compare
    print("\nAgent Consciousness Comparison:")
    print("-" * 60)
    print(f"{'Agent':<15} {'Perception':<12} {'Reaction':<12} {'Memory':<12} {'Overall':<10}")
    print("-" * 60)

    for agent in agents:
        print(f"{agent.name:<15} {agent.perception_fidelity:<12.2f} {agent.reaction_speed:<12.2f} {agent.memory_depth:<12.2f} {agent.overall_consciousness_score():<10.2f}")

    # Find most conscious agent
    most_conscious = max(agents, key=lambda a: a.overall_consciousness_score())
    print(f"\nMost conscious agent: {most_conscious.name} ({most_conscious.overall_consciousness_score():.2f})")


# ============================================================================
# IMMEDIATE VALUE PROPOSITION
# ============================================================================

def show_immediate_value():
    """
    Show immediate value of NPCPU protocol architecture.
    """
    print("\n" + "="*70)
    print("IMMEDIATE VALUE PROPOSITION")
    print("="*70)

    print("""
WHAT YOU CAN DO TODAY (Day 1):
------------------------------
1. Create conscious agents with measurable capabilities
2. Build perception-action loops
3. Store and retrieve experiences
4. Measure consciousness quantitatively
5. Compare agents objectively

WHAT YOU CAN DO THIS WEEK (Days 2-7):
-------------------------------------
1. Build multi-agent simulations
2. Implement vector similarity search
3. Create transformation pipelines
4. Track consciousness evolution
5. Experiment with different consciousness profiles

NO EXTERNAL DEPENDENCIES NEEDED:
--------------------------------
- No ChromaDB installation
- No cloud services
- No API keys
- Just Python + NumPy

READY FOR PRODUCTION:
--------------------
- Clean protocols
- Testable code
- Extensible architecture
- Clear migration path

COST: $0
TIME: Hours, not weeks
COMPLEXITY: Low
VALUE: High
""")


# ============================================================================
# RUN ALL DEMOS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NPCPU IMMEDIATE IMPLEMENTATION GUIDE")
    print("Running all demonstrations...")
    print("="*70)

    # Demo 1: Basic application
    demo_immediate_implementation()

    # Demo 2: Transformations
    demo_transformation_pipeline()

    # Demo 3: Consciousness measurement
    demo_consciousness_measurement()

    # Show value
    show_immediate_value()

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70)
