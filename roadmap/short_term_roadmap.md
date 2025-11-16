# Short-Term Roadmap (Weeks 1-4)

## Week 1: Foundation Enhancement

### Day 1-2: ChromaDB Adapter

**Goal**: Implement production-grade ChromaDB adapter

```python
# protocols/adapters/chromadb_adapter.py

from protocols.storage import VectorStorageProtocol
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class ChromaDBAdapter:
    """
    Production ChromaDB adapter implementing VectorStorageProtocol.

    Features:
    - Connection pooling
    - Retry logic
    - Error handling
    - Metrics collection
    """

    def __init__(
        self,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        settings: Optional[Settings] = None
    ):
        if host and port:
            # Remote ChromaDB
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local ChromaDB
            self.client = chromadb.PersistentClient(
                path=path or "./chromadb_data",
                settings=settings or Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )

        self.metrics = {
            "queries": 0,
            "stores": 0,
            "errors": 0
        }

    async def create_collection(
        self,
        name: str,
        vector_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        metadata_schema: Optional[Dict[str, type]] = None
    ) -> CollectionInfo:
        """Create collection with retry logic"""
        try:
            # Map distance metric
            chroma_metric = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.EUCLIDEAN: "l2",
                DistanceMetric.DOT_PRODUCT: "ip"
            }[distance_metric]

            # Create collection
            collection = self.client.create_collection(
                name=name,
                metadata={
                    "hnsw:space": chroma_metric,
                    "dimension": vector_dimension
                }
            )

            return CollectionInfo(
                name=name,
                vector_dimension=vector_dimension,
                count=collection.count(),
                metadata_schema=metadata_schema or {},
                created_at=time.time(),
                updated_at=time.time()
            )

        except Exception as e:
            self.metrics["errors"] += 1
            raise

    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> StorageResult:
        """Store with metrics collection"""
        try:
            coll = self.client.get_collection(collection)

            coll.upsert(
                ids=[id],
                embeddings=[vector],
                metadatas=[metadata] if metadata else None,
                documents=[document] if document else None
            )

            self.metrics["stores"] += 1

            return StorageResult(
                success=True,
                id=id,
                collection=collection,
                timestamp=time.time()
            )

        except Exception as e:
            self.metrics["errors"] += 1
            return StorageResult(
                success=False,
                id=id,
                collection=collection,
                timestamp=time.time(),
                error=str(e)
            )

    # ... implement other methods
```

**Deliverables**:
- ✓ Full ChromaDB adapter
- ✓ Unit tests
- ✓ Integration tests
- ✓ Performance benchmarks

### Day 3-4: Pinecone Adapter

**Goal**: Cloud-scale vector storage

```python
# protocols/adapters/pinecone_adapter.py

import pinecone
from protocols.storage import VectorStorageProtocol

class PineconeAdapter:
    """
    Pinecone adapter for cloud-scale vector storage.

    Benefits:
    - Scales to billions of vectors
    - Low latency worldwide
    - Managed infrastructure
    """

    def __init__(self, api_key: str, environment: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.indexes = {}

    async def create_collection(
        self,
        name: str,
        vector_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        replicas: int = 1,
        pods: int = 1
    ) -> CollectionInfo:
        """Create Pinecone index"""
        # Map metric
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "euclidean",
            DistanceMetric.DOT_PRODUCT: "dotproduct"
        }

        pinecone.create_index(
            name=name,
            dimension=vector_dimension,
            metric=metric_map[distance_metric],
            replicas=replicas,
            pods=pods
        )

        self.indexes[name] = pinecone.Index(name)

        return CollectionInfo(
            name=name,
            vector_dimension=vector_dimension,
            count=0,  # Will be updated on first use
            metadata_schema={},
            created_at=time.time(),
            updated_at=time.time()
        )

    # ... implement other methods
```

### Day 5-7: Configuration System

**Goal**: Configuration-driven consciousness models

```yaml
# configs/consciousness_models/plant_consciousness.yaml
model_type: "graded"
name: "Plant Consciousness Model"
description: "Consciousness model optimized for plant agents"

dimensions:
  # Sensory capabilities
  light_perception:
    weight: 2.0
    min: 0.0
    max: 1.0
    description: "Ability to detect and respond to light"

  water_detection:
    weight: 1.8
    min: 0.0
    max: 1.0
    description: "Ability to detect water availability"

  nutrient_sensing:
    weight: 1.5
    min: 0.0
    max: 1.0
    description: "Ability to sense soil nutrients"

  # Cognitive capabilities
  seasonal_memory:
    weight: 1.2
    min: 0.0
    max: 1.0
    description: "Long-term seasonal pattern memory"

  root_network_coherence:
    weight: 1.0
    min: 0.0
    max: 1.0
    description: "Coherence of root network communication"

  growth_intentionality:
    weight: 0.8
    min: 0.0
    max: 1.0
    description: "Goal-directed growth behavior"

  # Meta-cognitive (limited for plants)
  introspection_capacity:
    weight: 0.2
    min: 0.0
    max: 1.0
    description: "Limited self-awareness"

aggregation: "weighted_mean"

thresholds:
  dormant: 0.0
  reactive: 0.2
  responsive: 0.4
  adaptive: 0.6
  conscious: 0.8
```

```python
# protocols/consciousness_factory.py

import yaml
from typing import Dict, Any
from protocols.consciousness import GradedConsciousness

class ConsciousnessFactory:
    """
    Create consciousness models from configuration.

    Enables non-programmers to design consciousness models.
    """

    @staticmethod
    def from_yaml(filepath: str) -> GradedConsciousness:
        """Load consciousness model from YAML"""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        # Extract dimensions and weights
        dimensions = config.get("dimensions", {})
        capability_scores = {}
        weights = {}

        for dim_name, dim_config in dimensions.items():
            # Initialize to midpoint
            default_value = (dim_config["min"] + dim_config["max"]) / 2
            capability_scores[dim_name] = default_value
            weights[dim_name] = dim_config["weight"]

        # Create consciousness instance
        consciousness = GradedConsciousness(**capability_scores)
        consciousness.weights = weights

        return consciousness

    @staticmethod
    def create_custom(
        name: str,
        capabilities: Dict[str, float],
        weights: Dict[str, float]
    ) -> GradedConsciousness:
        """Create custom consciousness programmatically"""
        consciousness = GradedConsciousness(**capabilities)
        consciousness.weights = weights
        consciousness.name = name
        return consciousness
```

## Week 2: Swarm Coordination

### Distributed Consciousness Network

```python
# swarm/distributed_consciousness.py

from protocols.consciousness import GradedConsciousness
from protocols.storage import VectorStorageProtocol
import asyncio
from typing import List, Dict

class DistributedConsciousnessNetwork:
    """
    Network of agents sharing consciousness.

    Features:
    - Shared memory pool
    - Consciousness synchronization
    - Emergent collective intelligence
    - Proof of consciousness validation
    """

    def __init__(
        self,
        storage: VectorStorageProtocol,
        sync_interval_seconds: float = 10.0
    ):
        self.storage = storage
        self.agents: Dict[str, GradedConsciousness] = {}
        self.sync_interval = sync_interval_seconds
        self.running = False

    async def register_agent(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """Register agent in network"""
        self.agents[agent_id] = consciousness

        # Store initial consciousness state
        await self.publish_consciousness_state(agent_id, consciousness)

    async def publish_consciousness_state(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """Publish consciousness to network"""
        # Encode consciousness as vector
        vector = self.encode_consciousness(consciousness)

        metadata = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "overall_score": consciousness.overall_consciousness_score(),
            **consciousness.get_capability_scores()
        }

        await self.storage.store(
            collection="consciousness_network",
            id=agent_id,
            vector=vector,
            metadata=metadata
        )

    def encode_consciousness(
        self,
        consciousness: GradedConsciousness
    ) -> List[float]:
        """Encode consciousness as vector"""
        scores = consciousness.get_capability_scores()
        return list(scores.values())

    async def get_swarm_consciousness(self) -> GradedConsciousness:
        """
        Get collective consciousness of entire swarm.

        This is the emergent consciousness that exceeds
        any individual agent.
        """
        # Retrieve all agent consciousness states
        all_states = []
        async for batch in self.storage.scroll("consciousness_network"):
            for result in batch:
                scores = {
                    k: v for k, v in result.metadata.items()
                    if k not in ["agent_id", "timestamp", "overall_score"]
                }
                all_states.append(GradedConsciousness(**scores))

        if not all_states:
            return GradedConsciousness()

        # Calculate swarm average
        avg_scores = {}
        for capability in all_states[0].get_capability_scores().keys():
            avg_scores[capability] = np.mean([
                state.get_capability_scores()[capability]
                for state in all_states
            ])

        # Swarm consciousness is GREATER than average
        # Add emergence bonus
        emergence_factor = 1.2  # 20% boost from collective intelligence

        swarm = GradedConsciousness(**avg_scores)

        # Boost collective capabilities
        for capability in avg_scores.keys():
            current = getattr(swarm, capability)
            setattr(swarm, capability, min(1.0, current * emergence_factor))

        return swarm

    async def synchronize_agents(self):
        """
        Synchronize all agents with swarm consciousness.

        Partial synchronization maintains individuality.
        """
        swarm_consciousness = await self.get_swarm_consciousness()

        for agent_id, agent_consciousness in self.agents.items():
            # Blend individual and collective (30% swarm, 70% individual)
            synchronized = self.blend_consciousness(
                individual=agent_consciousness,
                collective=swarm_consciousness,
                blend_ratio=0.3
            )

            # Update agent
            self.agents[agent_id] = synchronized

            # Publish update
            await self.publish_consciousness_state(agent_id, synchronized)

    async def run_sync_loop(self):
        """Run continuous synchronization loop"""
        self.running = True

        while self.running:
            await self.synchronize_agents()
            await asyncio.sleep(self.sync_interval)

    def stop(self):
        """Stop synchronization loop"""
        self.running = False
```

## Week 3: Consciousness Evolution

### Genetic Algorithm for Consciousness Optimization

```python
# evolution/consciousness_evolution.py

from protocols.consciousness import GradedConsciousness
from typing import List, Callable
import random

class ConsciousnessEvolutionEngine:
    """
    Evolve optimal consciousness through genetic algorithms.

    Use cases:
    - Task-specific consciousness optimization
    - Multi-objective optimization
    - Adaptive agent design
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def create_initial_population(
        self,
        base_consciousness: GradedConsciousness
    ) -> List[GradedConsciousness]:
        """Create initial population with random variations"""
        population = []

        for _ in range(self.population_size):
            # Clone base
            individual = GradedConsciousness(
                **base_consciousness.get_capability_scores()
            )

            # Add random variation
            scores = individual.get_capability_scores()
            for capability in scores.keys():
                # Mutate: ±20% variation
                variation = random.uniform(0.8, 1.2)
                new_score = scores[capability] * variation
                setattr(individual, capability, np.clip(new_score, 0.0, 1.0))

            population.append(individual)

        return population

    def evaluate_fitness(
        self,
        consciousness: GradedConsciousness,
        fitness_function: Callable[[GradedConsciousness], float]
    ) -> float:
        """Evaluate fitness of consciousness"""
        return fitness_function(consciousness)

    def tournament_selection(
        self,
        population: List[GradedConsciousness],
        fitnesses: List[float],
        tournament_size: int = 3
    ) -> GradedConsciousness:
        """Select parent using tournament selection"""
        # Random tournament
        indices = random.sample(range(len(population)), tournament_size)
        tournament = [(population[i], fitnesses[i]) for i in indices]

        # Return best
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def crossover(
        self,
        parent1: GradedConsciousness,
        parent2: GradedConsciousness
    ) -> GradedConsciousness:
        """Crossover two parents"""
        if random.random() > self.crossover_rate:
            return parent1

        # Get capabilities
        scores1 = parent1.get_capability_scores()
        scores2 = parent2.get_capability_scores()

        # Uniform crossover
        child_scores = {}
        for capability in scores1.keys():
            child_scores[capability] = (
                scores1[capability] if random.random() < 0.5
                else scores2[capability]
            )

        return GradedConsciousness(**child_scores)

    def mutate(
        self,
        consciousness: GradedConsciousness
    ) -> GradedConsciousness:
        """Mutate consciousness"""
        scores = consciousness.get_capability_scores()

        for capability in scores.keys():
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = random.gauss(0, 0.1)
                new_score = scores[capability] + mutation
                scores[capability] = np.clip(new_score, 0.0, 1.0)

        return GradedConsciousness(**scores)

    def evolve(
        self,
        initial_consciousness: GradedConsciousness,
        fitness_function: Callable[[GradedConsciousness], float],
        generations: int = 100,
        verbose: bool = True
    ) -> GradedConsciousness:
        """
        Evolve consciousness to maximize fitness.

        Returns best consciousness found.
        """
        # Initial population
        population = self.create_initial_population(initial_consciousness)

        best_fitness = 0.0
        best_consciousness = None

        for generation in range(generations):
            # Evaluate fitness
            fitnesses = [
                self.evaluate_fitness(ind, fitness_function)
                for ind in population
            ]

            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_consciousness = population[gen_best_idx]

            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

            # Create next generation
            next_population = []

            # Elitism: keep best
            next_population.append(population[gen_best_idx])

            # Create offspring
            while len(next_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutate
                child = self.mutate(child)

                next_population.append(child)

            population = next_population

        return best_consciousness


# Example fitness function
def task_performance_fitness(consciousness: GradedConsciousness) -> float:
    """
    Fitness = how well consciousness performs on a task.

    Example: Optimize for fast perception and action.
    """
    scores = consciousness.get_capability_scores()

    # Weighted combination
    fitness = (
        scores.get("perception_fidelity", 0) * 0.4 +
        scores.get("reaction_speed", 0) * 0.4 +
        scores.get("memory_depth", 0) * 0.2
    )

    return fitness
```

## Week 4: Testing & Documentation

### Comprehensive Test Suite

```python
# tests/test_protocols.py

import pytest
from protocols.consciousness import GradedConsciousness, ConsciousnessAdapter
from protocols.storage import StorageBackendRegistry
from protocols.transformations import TransformationLibrary

class TestConsciousnessProtocol:
    """Test consciousness protocol implementation"""

    def test_graded_consciousness_creation(self):
        """Test creating graded consciousness"""
        consciousness = GradedConsciousness(
            perception_fidelity=0.8,
            reaction_speed=0.7
        )

        assert consciousness.perception_fidelity == 0.8
        assert consciousness.reaction_speed == 0.7
        assert 0.0 <= consciousness.overall_consciousness_score() <= 1.0

    def test_consciousness_evolution(self):
        """Test consciousness can evolve"""
        consciousness = GradedConsciousness(perception_fidelity=0.5)

        evolved = consciousness.evolve("perception_fidelity", 0.2)

        assert evolved.perception_fidelity == 0.7
        assert consciousness.perception_fidelity == 0.5  # Original unchanged

    def test_capability_checking(self):
        """Test capability checking"""
        consciousness = GradedConsciousness(meta_cognitive_ability=0.6)

        assert consciousness.can_perform("meta_cognitive_ability", 0.5)
        assert not consciousness.can_perform("meta_cognitive_ability", 0.7)

    def test_backward_compatibility(self):
        """Test adapter provides backward compatibility"""
        old_style = ConsciousnessAdapter.from_discrete_state("AWARE")

        assert isinstance(old_style, GradedConsciousness)
        assert old_style.describe_state().upper() == "AWARE"


class TestStorageProtocol:
    """Test storage protocol implementation"""

    @pytest.mark.asyncio
    async def test_inmemory_storage(self):
        """Test in-memory storage"""
        storage = StorageBackendRegistry.create("memory")

        # Create collection
        await storage.create_collection("test", vector_dimension=128)

        # Store
        result = await storage.store(
            collection="test",
            id="item1",
            vector=[0.1] * 128,
            metadata={"type": "test"}
        )

        assert result.success

        # Query
        results = await storage.query(
            collection="test",
            query_vector=[0.1] * 128,
            limit=1
        )

        assert len(results) == 1
        assert results[0].id == "item1"


class TestTransformationProtocol:
    """Test transformation protocol implementation"""

    def test_transformation_composition(self):
        """Test composing transformations"""
        proj = TransformationLibrary.projection(32)
        norm = TransformationLibrary.normalization()

        composed = proj @ norm

        assert composed.name == f"({proj.name} ∘ {norm.name})"

    def test_invariant_preservation(self):
        """Test invariants are preserved"""
        rotation = TransformationLibrary.rotation(np.pi/4)

        # Rotation preserves most invariants
        preserved = rotation.preserves()

        assert Invariant.TOPOLOGY in preserved
        assert Invariant.DIMENSION in preserved
```

### Documentation

```markdown
# docs/getting_started.md

# Getting Started with NPCPU

## Installation

```bash
pip install npcpu
```

## Quick Start

### 1. Create a Conscious Agent

```python
from npcpu.protocols.consciousness import GradedConsciousness

agent_consciousness = GradedConsciousness(
    perception_fidelity=0.8,
    memory_depth=0.7,
    introspection_capacity=0.6
)

print(f"Consciousness score: {agent_consciousness.overall_consciousness_score():.2f}")
```

### 2. Store Memories

```python
from npcpu.protocols.storage import StorageBackendRegistry

storage = StorageBackendRegistry.create("chromadb", path="./memories")

await storage.store(
    collection="agent_memories",
    id="memory_1",
    vector=embedding,
    metadata={"type": "experience", "valence": 0.8}
)
```

### 3. Apply Transformations

```python
from npcpu.protocols.transformations import TransformationLibrary

pipeline = (
    TransformationLibrary.projection(64) @
    TransformationLibrary.normalization() @
    TransformationLibrary.crystallization("fractal")
)

result = pipeline.transform(manifold)
```

## Next Steps

- [Consciousness Models Guide](./consciousness_models.md)
- [Storage Backends](./storage_backends.md)
- [Transformation Algebra](./transformations.md)
- [Swarm Coordination](./swarm_coordination.md)
```

## Deliverables by End of Week 4

✓ ChromaDB adapter (production-ready)
✓ Pinecone adapter (cloud-ready)
✓ Configuration system (YAML-based)
✓ Swarm coordination (distributed consciousness)
✓ Evolution engine (genetic algorithms)
✓ Comprehensive tests (>80% coverage)
✓ Documentation (getting started + API reference)

## Success Metrics

- **Performance**: <50ms query latency (local), <200ms (cloud)
- **Scalability**: Handle 1M+ vectors
- **Reliability**: 99.9% uptime
- **Usability**: <10 minutes to first working agent
- **Quality**: >80% test coverage
