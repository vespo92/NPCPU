"""
ZERO DEPENDENCIES DEMO

This demonstrates NPCPU protocol architecture with ZERO external dependencies.
Pure Python only. Runs anywhere Python runs.
"""

import time
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


# ============================================================================
# Core Data Structures
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


# ============================================================================
# Consciousness Protocol - Pure Python Implementation
# ============================================================================

class PureAgent:
    """
    Conscious agent with ZERO dependencies.

    Demonstrates that NPCPU protocols are implementable
    with nothing but pure Python.
    """

    def __init__(self, name: str):
        self.name = name

        # Consciousness capabilities (0.0 to 1.0)
        self.perception_fidelity = 0.6
        self.reaction_speed = 0.7
        self.memory_depth = 0.5
        self.introspection_capacity = 0.4

        # Memory
        self.experiences: List[Experience] = []
        self.beliefs: Dict[str, float] = {}

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
        content_str = str(perception.content).lower()

        # Simple rule-based reactions
        if "food" in content_str:
            return Action(
                action_type="approach",
                parameters={"target": perception.content},
                confidence=0.8
            )
        elif "threat" in content_str:
            return Action(
                action_type="avoid",
                parameters={"danger": perception.content},
                confidence=0.9
            )
        elif "safe" in content_str:
            return Action(
                action_type="rest",
                parameters={},
                confidence=0.7
            )
        else:
            return Action(
                action_type="observe",
                parameters={},
                confidence=0.5
            )

    def remember(self, experience: Experience):
        """Store experience in memory"""
        self.experiences.append(experience)

        # Limit memory based on memory_depth
        max_memories = int(self.memory_depth * 100)
        if len(self.experiences) > max_memories:
            # Forget oldest experiences
            self.experiences = self.experiences[-max_memories:]

    def introspect(self) -> Dict[str, Any]:
        """Examine own internal state"""
        if self.introspection_capacity < 0.3:
            return {"error": "Introspection capacity too low"}

        return {
            "name": self.name,
            "consciousness_score": self.overall_consciousness_score(),
            "memory_count": len(self.experiences),
            "beliefs_count": len(self.beliefs),
            "capabilities": {
                "perception": self.perception_fidelity,
                "reaction": self.reaction_speed,
                "memory": self.memory_depth,
                "introspection": self.introspection_capacity
            }
        }

    def overall_consciousness_score(self) -> float:
        """Calculate overall consciousness"""
        return (
            self.perception_fidelity +
            self.reaction_speed +
            self.memory_depth +
            self.introspection_capacity
        ) / 4

    def can_perform(self, capability: str, minimum_score: float = 0.5) -> bool:
        """Check if agent has sufficient capability"""
        capabilities = {
            "perception": self.perception_fidelity,
            "reaction": self.reaction_speed,
            "memory": self.memory_depth,
            "introspection": self.introspection_capacity
        }
        return capabilities.get(capability, 0.0) >= minimum_score

    def update_belief(self, belief: str, strength: float):
        """Update belief strength"""
        self.beliefs[belief] = max(0.0, min(1.0, strength))  # Clamp to [0, 1]


# ============================================================================
# Storage Protocol - Pure Python Implementation
# ============================================================================

class PureVectorStorage:
    """
    Vector storage with ZERO dependencies.

    Uses pure Python math for similarity calculation.
    """

    def __init__(self):
        self.collections: Dict[str, Dict[str, Dict]] = {}

    def create_collection(self, name: str):
        """Create new collection"""
        if name not in self.collections:
            self.collections[name] = {}
            return True
        return False

    def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any] = None
    ):
        """Store vector with metadata"""
        if collection not in self.collections:
            self.create_collection(collection)

        self.collections[collection][id] = {
            "vector": vector,
            "metadata": metadata or {},
            "timestamp": time.time()
        }

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity - pure Python"""
        if len(vec1) != len(vec2):
            return 0.0

        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        # Cosine similarity
        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def query(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict]:
        """Query similar vectors"""
        if collection not in self.collections:
            return []

        results = []

        for id, item in self.collections[collection].items():
            similarity = self.cosine_similarity(query_vector, item["vector"])

            results.append({
                "id": id,
                "score": similarity,
                "metadata": item["metadata"]
            })

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def count(self, collection: str) -> int:
        """Count items in collection"""
        return len(self.collections.get(collection, {}))


# ============================================================================
# Transformation Protocol - Pure Python Implementation
# ============================================================================

class PureTransformations:
    """
    Vector transformations with ZERO dependencies.

    Pure Python math operations.
    """

    @staticmethod
    def normalize(vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]

    @staticmethod
    def scale(vector: List[float], factor: float) -> List[float]:
        """Scale vector"""
        return [x * factor for x in vector]

    @staticmethod
    def add(vec1: List[float], vec2: List[float]) -> List[float]:
        """Add two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        return [a + b for a, b in zip(vec1, vec2)]

    @staticmethod
    def subtract(vec1: List[float], vec2: List[float]) -> List[float]:
        """Subtract two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        return [a - b for a, b in zip(vec1, vec2)]

    @staticmethod
    def project_to_dim(vector: List[float], target_dim: int) -> List[float]:
        """Project to lower dimension (truncate)"""
        return vector[:target_dim]


# ============================================================================
# Complete Application
# ============================================================================

class ZeroDependencyNPCPU:
    """
    Complete NPCPU application with ZERO dependencies.

    This is the proof that NPCPU protocols are immediately usable.
    """

    def __init__(self):
        self.storage = PureVectorStorage()
        self.agents: List[PureAgent] = []

    def add_agent(self, agent: PureAgent):
        """Add agent to system"""
        self.agents.append(agent)

    def simple_embedding(self, text: str) -> List[float]:
        """
        Simple text embedding (pure Python).

        In production, use proper embeddings (OpenAI, sentence-transformers, etc.)
        This is just for demonstration.
        """
        # Convert text to simple vector based on character frequencies
        vector = [0.0] * 26  # A-Z

        text_lower = text.lower()
        for char in text_lower:
            if 'a' <= char <= 'z':
                idx = ord(char) - ord('a')
                vector[idx] += 1.0

        # Normalize
        return PureTransformations.normalize(vector)

    def store_agent_experience(self, agent: PureAgent, experience: Experience):
        """Store agent experience"""
        # Create embedding from experience
        content_text = f"{experience.perception.stimulus_type} {experience.perception.content} {experience.action.action_type}"
        vector = self.simple_embedding(content_text)

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

    def query_similar_experiences(self, query_text: str, limit: int = 3) -> List[Dict]:
        """Find similar experiences"""
        query_vector = self.simple_embedding(query_text)

        results = self.storage.query(
            collection="experiences",
            query_vector=query_vector,
            limit=limit
        )

        return results

    def run_simulation(self, stimuli: List[str]):
        """Run complete simulation"""
        print("\n" + "="*70)
        print("ZERO DEPENDENCY NPCPU - PURE PYTHON IMPLEMENTATION")
        print("="*70)

        for agent in self.agents:
            print(f"\n{agent.name} (consciousness: {agent.overall_consciousness_score():.2f})")
            print("-" * 60)

            for stimulus in stimuli:
                # Perceive
                perception = agent.perceive(stimulus)

                # React
                action = agent.react(perception)

                # Remember
                experience = Experience(
                    perception=perception,
                    action=action,
                    timestamp=time.time()
                )
                agent.remember(experience)

                # Store in vector DB
                self.store_agent_experience(agent, experience)

                # Print
                print(f"  Stimulus: {stimulus}")
                print(f"  → Action: {action.action_type} (confidence: {action.confidence:.2f})")

            # Introspection
            if agent.can_perform("introspection", 0.3):
                print(f"\n  {agent.name}'s Self-Reflection:")
                introspection = agent.introspect()
                print(f"    Memory: {introspection['memory_count']} experiences")
                print(f"    Consciousness: {introspection['consciousness_score']:.2f}")

        # Show storage stats
        print(f"\n\nStorage Statistics:")
        print(f"  Total experiences: {self.storage.count('experiences')}")

        # Demonstrate semantic search
        print(f"\n\nSemantic Search Demo:")
        query = "food"
        print(f"  Query: '{query}'")
        similar = self.query_similar_experiences(query, limit=3)
        print(f"  Similar experiences:")
        for i, result in enumerate(similar, 1):
            print(f"    {i}. {result['metadata']['agent_name']}: {result['metadata']['action_type']} (similarity: {result['score']:.3f})")


# ============================================================================
# Demonstrations
# ============================================================================

def demo_consciousness_comparison():
    """Compare agents with different consciousness levels"""
    print("\n" + "="*70)
    print("CONSCIOUSNESS COMPARISON")
    print("="*70)

    # Create agents with different capabilities
    reactive = PureAgent("Reactive")
    reactive.perception_fidelity = 0.5
    reactive.reaction_speed = 0.8
    reactive.memory_depth = 0.3
    reactive.introspection_capacity = 0.1

    aware = PureAgent("Aware")
    aware.perception_fidelity = 0.7
    aware.reaction_speed = 0.6
    aware.memory_depth = 0.6
    aware.introspection_capacity = 0.5

    reflective = PureAgent("Reflective")
    reflective.perception_fidelity = 0.8
    reflective.reaction_speed = 0.5
    reflective.memory_depth = 0.8
    reflective.introspection_capacity = 0.8

    agents = [reactive, aware, reflective]

    # Print comparison table
    print("\n{:<12} {:<10} {:<10} {:<10} {:<15} {:<10}".format(
        "Agent", "Perceive", "React", "Memory", "Introspection", "Overall"
    ))
    print("-" * 70)

    for agent in agents:
        print("{:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<15.2f} {:<10.2f}".format(
            agent.name,
            agent.perception_fidelity,
            agent.reaction_speed,
            agent.memory_depth,
            agent.introspection_capacity,
            agent.overall_consciousness_score()
        ))

    # Capability checks
    print("\n\nCapability Analysis:")
    for agent in agents:
        print(f"\n{agent.name}:")
        print(f"  Can perceive well: {agent.can_perform('perception', 0.7)}")
        print(f"  Can introspect: {agent.can_perform('introspection', 0.5)}")
        print(f"  Can remember: {agent.can_perform('memory', 0.6)}")


def demo_transformation_algebra():
    """Demonstrate transformation composition"""
    print("\n" + "="*70)
    print("TRANSFORMATION ALGEBRA")
    print("="*70)

    # Original vector
    vector = [1.0, 2.0, 3.0, 4.0]
    print(f"\nOriginal vector: {vector}")

    # Apply transformations
    normalized = PureTransformations.normalize(vector)
    print(f"Normalized: {[f'{x:.3f}' for x in normalized]}")

    scaled = PureTransformations.scale(normalized, 2.0)
    print(f"Scaled 2x: {[f'{x:.3f}' for x in scaled]}")

    projected = PureTransformations.project_to_dim(scaled, 2)
    print(f"Projected to 2D: {[f'{x:.3f}' for x in projected]}")


def demo_vector_search():
    """Demonstrate vector similarity search"""
    print("\n" + "="*70)
    print("VECTOR SIMILARITY SEARCH")
    print("="*70)

    storage = PureVectorStorage()
    storage.create_collection("test")

    # Store some documents
    documents = [
        "The cat sat on the mat",
        "The dog played in the park",
        "The cat chased the mouse",
        "Birds fly in the sky",
        "The fish swam in the pond"
    ]

    app = ZeroDependencyNPCPU()

    print("\nStoring documents:")
    for i, doc in enumerate(documents):
        vector = app.simple_embedding(doc)
        storage.store(
            collection="test",
            id=f"doc_{i}",
            vector=vector,
            metadata={"text": doc}
        )
        print(f"  {i+1}. {doc}")

    # Query
    query = "cat"
    print(f"\nQuery: '{query}'")
    query_vector = app.simple_embedding(query)
    results = storage.query("test", query_vector, limit=3)

    print("Most similar documents:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['metadata']['text']} (similarity: {result['score']:.3f})")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("NPCPU PROTOCOL ARCHITECTURE - ZERO DEPENDENCIES DEMO")
    print("Pure Python Implementation")
    print("="*70)

    # Demo 1: Consciousness comparison
    demo_consciousness_comparison()

    # Demo 2: Transformation algebra
    demo_transformation_algebra()

    # Demo 3: Vector search
    demo_vector_search()

    # Demo 4: Complete application
    print("\n" + "="*70)
    print("COMPLETE APPLICATION DEMO")
    print("="*70)

    app = ZeroDependencyNPCPU()

    # Create agents
    explorer = PureAgent("Explorer")
    explorer.perception_fidelity = 0.8
    explorer.memory_depth = 0.7
    explorer.introspection_capacity = 0.6

    observer = PureAgent("Observer")
    observer.perception_fidelity = 0.6
    observer.memory_depth = 0.8
    observer.introspection_capacity = 0.7

    app.add_agent(explorer)
    app.add_agent(observer)

    # Run simulation
    stimuli = [
        "food nearby",
        "unknown object",
        "potential threat",
        "safe zone",
        "food source"
    ]

    app.run_simulation(stimuli)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: IMMEDIATE VALUE WITH ZERO DEPENDENCIES")
    print("="*70)
    print("""
✓ Conscious agents with measurable capabilities
✓ Vector storage and semantic search
✓ Transformation algebra
✓ Multi-agent simulation
✓ Pure Python - runs anywhere

Time to implement: Minutes
Dependencies: None
Cost: $0
Value: Immediate
""")


if __name__ == "__main__":
    main()
