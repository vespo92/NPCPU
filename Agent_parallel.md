# NPCPU Parallel Agent Implementation Guide

This document defines 10 parallel workstreams for implementing practical features in NPCPU. Each agent should work independently on their assigned task, using the core abstractions in `core/`.

---

## Project Context

NPCPU (Non-Player Cognitive Processing Unit) is a consciousness-aware distributed computing framework. The core abstractions are in place:
- `core/abstractions.py` - Base classes (BaseOrganism, BaseWorld, BasePopulation, BehaviorTree)
- `core/events.py` - Event bus for decoupled communication
- `core/plugins.py` - Plugin system with hooks
- `core/config.py` - Layered configuration management
- `core/simple_organism.py` - Reference implementation

---

## Agent 1: Neural Consciousness Implementation

**Task:** Implement a neural network-based consciousness model using the core abstractions.

**Files to create:**
- `consciousness/neural_consciousness.py`

**Requirements:**
```python
class NeuralConsciousness(BaseSubsystem):
    """
    Neural network-based consciousness using attention mechanisms.

    Features:
    - Attention-based perception weighting
    - Working memory with capacity limits
    - Emotional valence computation
    - Self-model for introspection
    """
```

**Key methods:**
- `process_perception(stimuli)` - Weight and filter incoming stimuli
- `update_working_memory(info)` - Manage limited-capacity memory
- `compute_emotional_state()` - Derive emotional valence from state
- `introspect()` - Generate self-model report

**Dependencies:** numpy, optionally torch

---

## Agent 2: Procedural World Generator

**Task:** Create a procedural world generator that produces varied environments.

**Files to create:**
- `ecosystem/world_generator.py`

**Requirements:**
```python
class WorldGenerator:
    """
    Procedural world generation with biomes, resources, and hazards.

    Features:
    - Noise-based terrain generation
    - Biome classification (forest, desert, ocean, etc.)
    - Resource distribution based on biome
    - Dynamic hazard placement
    """

class ProceduralWorld(BaseWorld):
    """World implementation using procedural generation."""
```

**Key methods:**
- `generate(seed, size)` - Generate world from seed
- `get_biome_at(x, y)` - Get biome type at location
- `spawn_resources()` - Distribute resources by biome rules

**Dependencies:** numpy, noise (optional)

---

## Agent 3: Social Network System

**Task:** Implement rich social dynamics between organisms.

**Files to create:**
- `social/relationships.py`
- `social/communication.py`

**Requirements:**
```python
class SocialNetwork:
    """
    Graph-based social network for organism relationships.

    Features:
    - Relationship types (ally, rival, mate, parent/child)
    - Trust/reputation scores
    - Group/tribe formation
    - Influence propagation
    """

class CommunicationSystem(BaseSubsystem):
    """
    Inter-organism communication.

    Features:
    - Signal types (warning, mating, food location)
    - Signal range and attenuation
    - Deception detection
    """
```

**Key methods:**
- `form_relationship(org1, org2, type)` - Create relationship
- `get_influence(organism)` - Calculate social influence
- `broadcast_signal(sender, signal_type, data)` - Send signal to nearby organisms

---

## Agent 4: Genetic Evolution Engine

**Task:** Build a genetic algorithm system for evolving organism traits.

**Files to create:**
- `evolution/genetic_engine.py`
- `evolution/mutations.py`

**Requirements:**
```python
class Genome:
    """
    Genetic representation of organism traits.

    Features:
    - Gene encoding for traits
    - Dominant/recessive alleles
    - Mutation operators
    - Crossover mechanisms
    """

class EvolutionEngine:
    """
    Manages population-level evolution.

    Features:
    - Fitness function composition
    - Selection strategies (tournament, roulette)
    - Speciation tracking
    - Genetic diversity metrics
    """
```

**Key methods:**
- `crossover(parent1, parent2)` - Combine genomes
- `mutate(genome, rate)` - Apply mutations
- `calculate_fitness(organism, environment)` - Evaluate fitness
- `select_parents(population)` - Choose breeding pairs

---

## Agent 5: Learning & Memory System

**Task:** Implement learning mechanisms for organisms.

**Files to create:**
- `learning/memory_systems.py`
- `learning/reinforcement.py`

**Requirements:**
```python
class MemorySystem(BaseSubsystem):
    """
    Multi-tier memory system.

    Features:
    - Sensory memory (very short term)
    - Working memory (limited capacity)
    - Long-term memory (consolidated experiences)
    - Episodic vs semantic memory
    """

class ReinforcementLearner(BaseSubsystem):
    """
    Q-learning based behavior adaptation.

    Features:
    - State-action value estimation
    - Exploration vs exploitation
    - Experience replay
    - Policy updates
    """
```

**Key methods:**
- `store(memory_type, content, importance)` - Store memory
- `recall(query, memory_type)` - Retrieve relevant memories
- `learn_from_experience(state, action, reward, next_state)` - RL update
- `consolidate()` - Move important memories to long-term

---

## Agent 6: REST API Server

**Task:** Create a REST API for external interaction with simulations.

**Files to create:**
- `api/server.py`
- `api/routes.py`
- `api/models.py`

**Requirements:**
```python
# Using FastAPI
from fastapi import FastAPI

app = FastAPI(title="NPCPU API", version="1.0.0")

# Endpoints needed:
# GET  /simulations - List running simulations
# POST /simulations - Create new simulation
# GET  /simulations/{id} - Get simulation status
# POST /simulations/{id}/tick - Advance simulation
# GET  /simulations/{id}/organisms - List organisms
# GET  /simulations/{id}/organisms/{org_id} - Get organism details
# POST /simulations/{id}/organisms - Spawn organism
# GET  /simulations/{id}/world - Get world state
# POST /simulations/{id}/events - Trigger world event
# WebSocket /simulations/{id}/stream - Real-time updates
```

**Key features:**
- Pydantic models for request/response validation
- WebSocket support for real-time streaming
- Authentication middleware (optional)
- Rate limiting

**Dependencies:** fastapi, uvicorn, pydantic

---

## Agent 7: Visualization Dashboard

**Task:** Create a web-based visualization dashboard.

**Files to create:**
- `visualization/dashboard.py`
- `visualization/static/index.html`
- `visualization/static/app.js`

**Requirements:**
```python
class DashboardServer:
    """
    Web dashboard for simulation visualization.

    Features:
    - Real-time population graphs
    - Organism location map (2D)
    - Fitness/trait distribution charts
    - Event timeline
    - Control panel (pause/resume/speed)
    """
```

**Frontend features:**
- Canvas-based world rendering
- Chart.js for statistics
- WebSocket connection for live updates
- Control buttons for simulation management

**Dependencies:** websockets, aiohttp or flask

---

## Agent 8: Multi-Agent Coordination System

**Task:** Implement swarm intelligence and multi-agent coordination.

**Files to create:**
- `swarm/coordination.py`
- `swarm/collective_behavior.py`

**Requirements:**
```python
class SwarmCoordinator:
    """
    Coordinates multiple organisms for collective behavior.

    Features:
    - Flocking behavior (separation, alignment, cohesion)
    - Task allocation
    - Emergent formation patterns
    - Distributed decision making
    """

class CollectiveMind:
    """
    Shared consciousness for organism groups.

    Features:
    - Knowledge sharing
    - Collective memory
    - Group decision voting
    - Hive mind emergence
    """
```

**Key methods:**
- `calculate_flocking_vectors(organism)` - Boid-like behavior
- `allocate_tasks(tasks, organisms)` - Distribute work
- `vote(proposal, group)` - Collective decision
- `share_knowledge(source, targets, knowledge)` - Propagate info

---

## Agent 9: Persistence & Replay System

**Task:** Enhance state persistence with full replay capability.

**Files to create:**
- `persistence/recorder.py`
- `persistence/replay.py`

**Requirements:**
```python
class SimulationRecorder:
    """
    Records simulation for later replay.

    Features:
    - Delta compression (only store changes)
    - Keyframe intervals
    - Event logging with timestamps
    - Configurable detail levels
    """

class SimulationPlayer:
    """
    Replays recorded simulations.

    Features:
    - Forward/backward playback
    - Variable speed
    - Jump to tick
    - Branch from any point (what-if scenarios)
    """
```

**Key methods:**
- `record_tick(simulation)` - Capture state changes
- `create_keyframe()` - Full state snapshot
- `play(speed)` - Replay at speed
- `branch(tick)` - Create new simulation from point

---

## Agent 10: Test Suite & Benchmarks

**Task:** Create comprehensive tests and performance benchmarks.

**Files to create:**
- `tests/test_core_abstractions.py`
- `tests/test_simple_organism.py`
- `tests/test_events.py`
- `tests/test_plugins.py`
- `benchmarks/performance.py`

**Requirements:**
```python
# Unit tests for all core abstractions
# Integration tests for organism lifecycle
# Performance benchmarks:
# - Organisms per second (creation)
# - Ticks per second (varying population sizes)
# - Memory usage scaling
# - Event throughput
```

**Benchmark targets:**
- Support 10,000+ organisms
- 100+ ticks/second with 1000 organisms
- Event bus: 10,000+ events/second
- Memory: < 1KB per organism base

---

## Coordination Guidelines

### Shared Patterns

All agents should use these patterns:

```python
# Use event bus for communication
from core.events import get_event_bus
bus = get_event_bus()
bus.emit("my_component.event", {"data": value})

# Use hooks for extensibility
from core.plugins import HookPoint
context.hooks.register(HookPoint.ORGANISM_TICK, my_handler)

# Use config for settings
from core.config import get_config
config = get_config()
value = config.get("my_section.setting", default_value)

# Extend base classes
from core.abstractions import BaseSubsystem, BaseOrganism
class MySubsystem(BaseSubsystem):
    def tick(self): ...
    def get_state(self): ...
    def set_state(self, state): ...
```

### File Organization

```
NPCPU/
├── core/                 # Shared abstractions (don't modify)
├── consciousness/        # Agent 1
├── ecosystem/           # Agent 2
├── social/              # Agent 3
├── evolution/           # Agent 4
├── learning/            # Agent 5
├── api/                 # Agent 6
├── visualization/       # Agent 7
├── swarm/               # Agent 8
├── persistence/         # Agent 9
├── tests/               # Agent 10
└── benchmarks/          # Agent 10
```

### Communication Contract

Components communicate via events:

| Event | Publisher | Consumers |
|-------|-----------|-----------|
| `organism.created` | Population | Analytics, Social |
| `organism.died` | Organism | Population, Analytics |
| `organism.moved` | Movement | Visualization, Social |
| `organism.learned` | Learning | Analytics |
| `relationship.formed` | Social | Analytics |
| `world.event` | World | All organisms |
| `simulation.tick` | Simulation | All subsystems |

### Testing Requirements

Each agent must provide:
1. Unit tests for new classes
2. Integration test with SimpleOrganism
3. Example usage in docstrings
4. Type hints on public methods

---

## Execution Instructions

To use this guide with parallel agents:

```bash
# Agent 1 prompt:
"Implement neural consciousness system per Agent_parallel.md Agent 1 spec.
Use core/abstractions.py base classes. Create consciousness/neural_consciousness.py"

# Agent 2 prompt:
"Implement procedural world generator per Agent_parallel.md Agent 2 spec.
Use core/abstractions.py BaseWorld. Create ecosystem/world_generator.py"

# ... etc for agents 3-10
```

Each agent works independently. Integration happens through:
1. Event bus subscriptions
2. Hook registrations
3. Shared base class interfaces

---

## Success Criteria

All implementations complete when:
- [ ] All specified files created
- [ ] Unit tests passing
- [ ] Integration with SimpleOrganism works
- [ ] Example in docstring is runnable
- [ ] No modifications to core/ needed
- [ ] Events properly emitted/consumed
