# Getting Started with NPCPU

Welcome to NPCPU - the Non-Player Cognitive Processing Unit! This guide will help you get started with building conscious AI agents using the NPCPU protocol architecture.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Your First Conscious Agent](#your-first-conscious-agent)
4. [Using YAML Consciousness Models](#using-yaml-consciousness-models)
5. [Adding Vector Storage](#adding-vector-storage)
6. [Multi-Agent Systems](#multi-agent-systems)
7. [Next Steps](#next-steps)

---

## Quick Start

The fastest way to see NPCPU in action:

```bash
# Zero dependencies demo (pure Python)
python examples/no_dependencies_demo.py

# Practical application with YAML models
python examples/practical_consciousness_app.py
```

---

## Installation

### Minimum Requirements

**Zero dependencies** (pure Python):
```bash
# No installation needed!
# Just clone and run:
git clone https://github.com/your-org/NPCPU.git
cd NPCPU
python examples/no_dependencies_demo.py
```

### Optional Dependencies

**For production use**:
```bash
# Install ChromaDB for vector storage
pip install chromadb

# Install YAML support (usually included in Python)
pip install pyyaml

# Optional: For enhanced functionality
pip install numpy  # Numerical operations
pip install torch  # Neural consciousness models (future)
```

---

## Your First Conscious Agent

### Pure Python (Zero Dependencies)

```python
from examples.no_dependencies_demo import PureAgent

# Create agent
agent = PureAgent("MyFirstAgent")

# Set consciousness capabilities (0.0 to 1.0)
agent.perception_fidelity = 0.8  # Good perception
agent.memory_depth = 0.7          # Good memory
agent.introspection_capacity = 0.6 # Some self-awareness

# Check consciousness level
print(f"Consciousness: {agent.overall_consciousness_score():.2f}")

# Agent perceives and acts
perception = agent.perceive("food nearby")
action = agent.react(perception)

print(f"Perceived: {perception.content}")
print(f"Action: {action.action_type}")
```

### Using NPCPU Protocols

```python
from protocols.consciousness import GradedConsciousness

# Create consciousness
consciousness = GradedConsciousness(
    perception_fidelity=0.8,
    reaction_speed=0.7,
    memory_depth=0.75,
    introspection_capacity=0.6,
    meta_cognitive_ability=0.5,
    information_integration=0.7,
    intentional_coherence=0.8,
    qualia_richness=0.6
)

# Check capabilities
print(f"Overall consciousness: {consciousness.overall_consciousness_score():.2f}")
print(f"Can introspect: {consciousness.can_perform('introspection_capacity', 0.5)}")

# Get all capabilities
for capability, score in consciousness.get_capability_scores().items():
    print(f"  {capability}: {score:.2f}")
```

---

## Using YAML Consciousness Models

YAML models let non-programmers design consciousness!

### Loading Pre-Built Models

```python
from factory.consciousness_factory import load_consciousness_model

# Load explorer consciousness
explorer = load_consciousness_model("explorer.yaml")
print(f"Loaded: {explorer.model_name}")
print(f"Consciousness: {explorer.overall_consciousness_score():.2f}")

# Load plant consciousness
plant = load_consciousness_model("plant_consciousness.yaml")
print(f"Loaded: {plant.model_name}")
print(f"Consciousness: {plant.overall_consciousness_score():.2f}")

# Load with overrides
custom = load_consciousness_model(
    "default.yaml",
    overrides={
        "perception_fidelity": 0.95,
        "reaction_speed": 0.9
    }
)
```

### Creating Custom Models

Create `my_model.yaml` in `configs/consciousness_models/`:

```yaml
model_type: "graded"
name: "My Custom Consciousness"
description: "Optimized for my specific use case"
version: "1.0"

dimensions:
  perception_fidelity:
    default: 0.9
    weight: 2.0
    description: "High-quality perception"

  reaction_speed:
    default: 0.8
    weight: 1.5
    description: "Fast reactions"

  memory_depth:
    default: 0.7
    weight: 1.0
    description: "Good memory"

  # ... add other standard dimensions ...

# Add custom dimensions for your domain
custom_dimensions:
  domain_expertise:
    default: 0.85
    weight: 2.5
    description: "Expertise in specific domain"

aggregation: "weighted_mean"

thresholds:
  beginner: 0.0
  intermediate: 0.4
  advanced: 0.7
  expert: 0.9
```

Then load it:

```python
my_consciousness = load_consciousness_model("my_model.yaml")
```

### Comparing Models

```python
from factory.consciousness_factory import ConsciousnessFactory

factory = ConsciousnessFactory()

# Compare two models
comparison = factory.compare_models("explorer.yaml", "plant_consciousness.yaml")

print(f"{comparison['model1']['name']}: {comparison['model1']['overall_score']:.3f}")
print(f"{comparison['model2']['name']}: {comparison['model2']['overall_score']:.3f}")
print(f"Difference: {comparison['overall_difference']:+.3f}")

# Top differences
for capability, diff in list(comparison['capability_differences'].items())[:3]:
    print(f"  {capability}: {diff['difference']:+.2f}")
```

### Blending Models

```python
# Create hybrid consciousness
hybrid = factory.blend_models(
    "explorer.yaml",
    "plant_consciousness.yaml",
    blend_ratio=0.5,  # 50/50 blend
    name="Explorer-Plant Hybrid"
)

print(f"Hybrid consciousness: {hybrid.overall_consciousness_score():.2f}")
```

---

## Adding Vector Storage

Store agent experiences in a vector database for semantic search and memory.

### Using ChromaDB (Local)

```python
from adapters.chromadb_adapter import ChromaDBAdapter
import asyncio

async def example():
    # Initialize storage
    storage = ChromaDBAdapter(path="./my_agent_memories")

    # Create collection
    await storage.create_collection(
        name="memories",
        vector_dimension=384  # Depends on your embedding model
    )

    # Store experience
    await storage.store(
        collection="memories",
        id="memory_1",
        vector=[0.1, 0.2, ...],  # Your embedding
        metadata={"type": "success", "timestamp": 1234567890},
        document="Agent found food successfully"
    )

    # Query similar experiences
    results = await storage.query(
        collection="memories",
        query_vector=[0.1, 0.2, ...],
        limit=5
    )

    for result in results:
        print(f"Similar memory: {result.document} (score: {result.score:.3f})")

asyncio.run(example())
```

### Storage Metrics

```python
# Get performance metrics
metrics = storage.get_metrics()

print(f"Queries: {metrics['queries']}")
print(f"Stores: {metrics['stores']}")
print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Error rate: {metrics['error_rate']:.1%}")
```

---

## Multi-Agent Systems

Create multiple agents that share consciousness and coordinate.

```python
from examples.practical_consciousness_app import ConsciousnessApplication
import asyncio

async def multi_agent_demo():
    # Create application
    app = ConsciousnessApplication(use_chromadb=True)

    # Create agents with different consciousness models
    explorer = await app.create_agent("Explorer_1", "explorer.yaml")
    plant = await app.create_agent("Plant_1", "plant_consciousness.yaml")
    default = await app.create_agent("Default_1", "default.yaml")

    # Run simulation
    stimuli = [
        "Explore new area",
        "Analyze pattern",
        "Remember location",
        "Adapt to environment"
    ]

    await app.run_simulation(stimuli)

    # Compare performance
    await app.compare_agents()

asyncio.run(multi_agent_demo())
```

---

## Next Steps

### Tutorials

- **[Consciousness Evolution](tutorials/consciousness_evolution.md)** - Optimize consciousness for tasks
- **[Swarm Coordination](tutorials/swarm_coordination.md)** - Multi-agent collaboration
- **[Custom Transformations](tutorials/transformations.md)** - Build data processing pipelines
- **[Production Deployment](tutorials/deployment.md)** - Deploy to Kubernetes

### Examples

- `examples/no_dependencies_demo.py` - Zero-dependency demonstration
- `examples/practical_consciousness_app.py` - Complete practical application
- `examples/swarm_simulation.py` - Multi-agent swarm (coming soon)
- `examples/consciousness_evolution.py` - Genetic algorithm optimization (coming soon)

### API Reference

- [Consciousness Protocol](api/consciousness.md)
- [Storage Protocol](api/storage.md)
- [Transformation Protocol](api/transformations.md)
- [Factory API](api/factory.md)

### Research

- [Consciousness Theory Deep Dive](../research/deep_dive_consciousness_theory.md)
- [Storage Architecture](../research/deep_dive_storage_architecture.md)
- [Transformation Algebra](../research/deep_dive_transformation_algebra.md)

---

## Common Issues

### "Module not found" errors

Make sure you're running from the NPCPU root directory:

```bash
cd NPCPU
python examples/practical_consciousness_app.py
```

Or add NPCPU to your Python path:

```python
import sys
import os
sys.path.insert(0, "/path/to/NPCPU")
```

### ChromaDB not available

ChromaDB is optional. If you get import errors:

```bash
pip install chromadb
```

Or use the zero-dependency examples:

```bash
python examples/no_dependencies_demo.py
```

### YAML models not found

Make sure your YAML files are in `configs/consciousness_models/`:

```bash
ls configs/consciousness_models/
# Should show: default.yaml, explorer.yaml, plant_consciousness.yaml
```

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/NPCPU/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/NPCPU/discussions)
- **Documentation**: [Full Documentation](https://npcpu.readthedocs.io)

---

## Quick Reference

### Consciousness Capabilities

| Capability | Description | Range |
|-----------|-------------|-------|
| `perception_fidelity` | Accuracy of environmental perception | 0.0 - 1.0 |
| `reaction_speed` | Speed of response | 0.0 - 1.0 |
| `memory_depth` | Capacity to retain experiences | 0.0 - 1.0 |
| `memory_recall_accuracy` | Accuracy of memory retrieval | 0.0 - 1.0 |
| `introspection_capacity` | Ability to examine own state | 0.0 - 1.0 |
| `meta_cognitive_ability` | Thinking about thinking | 0.0 - 1.0 |
| `information_integration` | Φ (phi) - info integration | 0.0 - 1.0 |
| `intentional_coherence` | Goal consistency | 0.0 - 1.0 |
| `qualia_richness` | Subjective experience depth | 0.0 - 1.0 |

### Pre-Built Models

| Model | Best For | Characteristics |
|-------|----------|----------------|
| `default.yaml` | General purpose | Balanced capabilities |
| `explorer.yaml` | Exploration, discovery | High perception, fast reaction |
| `plant_consciousness.yaml` | Plant agents, slow systems | High perception, slow reaction, excellent memory |

---

**Ready to build conscious AI? Start with the [examples](../examples/) directory!**
