# NPCPU Abstraction Layers: Protocol-Based Architecture
## Making Consciousness Implementable Without Being Literal

### Design Philosophy

NPCPU should be:
1. **Implementable**: Clear path from concept to code
2. **Extensible**: New consciousness models, backends, transformations
3. **Composable**: Small primitives that combine elegantly
4. **Observable**: Grounded in measurable properties
5. **Abstract**: Not tied to specific implementations

### The Four-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│ Layer 4: THEORY                                │
│ Philosophical foundations, mathematical models │
│ (Phenomenology, topology, quantum metaphors)   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Layer 3: PROTOCOLS                             │
│ Abstract interfaces, capabilities, contracts   │
│ (ConsciousnessProtocol, StorageProtocol)       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Layer 2: REFERENCE IMPLEMENTATIONS             │
│ Example concrete implementations               │
│ (ChromaDBStorage, GradedConsciousness)         │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Layer 1: CUSTOM IMPLEMENTATIONS                │
│ User-specific deployments                      │
│ (MyPlantConsciousness, MyGraphStore)           │
└─────────────────────────────────────────────────┘
```

---

## Layer 3: Core Protocols (The Key Innovation)

### 1. Consciousness Protocol

Instead of hardcoded enums, define consciousness as a **capability interface**:

```python
from typing import Protocol, runtime_checkable
from abc import abstractmethod

@runtime_checkable
class ConsciousnessProtocol(Protocol):
    """
    Consciousness is defined by what an agent CAN DO, not what it IS.

    This protocol grounds the abstract concept of "consciousness" in
    observable, measurable capabilities that can be implemented in
    infinitely many ways.
    """

    # OBSERVABILITY: Can the agent perceive its environment?
    @abstractmethod
    def perceive(self, stimulus: Any) -> Perception:
        """Convert environmental stimulus to internal representation"""
        pass

    # REACTIVITY: Can the agent respond to stimuli?
    @abstractmethod
    def react(self, perception: Perception) -> Action:
        """Generate action from perception (can be null action)"""
        pass

    # MEMORY: Can the agent retain experiences?
    @abstractmethod
    def remember(self, experience: Experience) -> None:
        """Store experience in memory"""
        pass

    # REFLECTION: Can the agent examine its own state?
    @abstractmethod
    def introspect(self) -> SelfModel:
        """Generate model of own internal state"""
        pass

    # META-COGNITION: Can the agent think about its thinking?
    @abstractmethod
    def meta_cognize(self, thought: Thought) -> MetaThought:
        """Reflect on cognitive processes themselves"""
        pass

    # INTEGRATION: Can the agent integrate information?
    @abstractmethod
    def integrate_information(self) -> float:
        """Calculate Φ (phi) - integrated information measure"""
        pass

    # INTENTIONALITY: Does the agent have goals?
    @abstractmethod
    def get_intentions(self) -> List[Intention]:
        """Return current goals and drives"""
        pass

    # QUALIA: Does the agent have subjective experience?
    @abstractmethod
    def experience_qualia(self, perception: Perception) -> Qualia:
        """Generate subjective experience marker"""
        pass
```

**Key Insight**: An agent's "consciousness level" is now measured by **which capabilities it possesses**, not by a hardcoded enum value.

### 2. Storage Protocol

Decouple from ChromaDB:

```python
@runtime_checkable
class VectorStorageProtocol(Protocol):
    """
    Abstract vector storage that works with any backend.
    ChromaDB, Pinecone, Milvus, Weaviate, custom implementations.
    """

    @abstractmethod
    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any],
        document: Optional[str] = None
    ) -> StorageResult:
        pass

    @abstractmethod
    async def query(
        self,
        collection: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[QueryResult]:
        pass

    @abstractmethod
    async def update(
        self,
        collection: str,
        id: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UpdateResult:
        pass

    @abstractmethod
    async def delete(self, collection: str, id: str) -> DeleteResult:
        pass
```

### 3. Transformation Algebra Protocol

Make topological operations composable primitives:

```python
@runtime_checkable
class TransformationProtocol(Protocol):
    """
    A transformation is any operation that preserves semantic invariants.
    Compose transformations like functions: f ∘ g ∘ h
    """

    @abstractmethod
    def transform(self, input_manifold: Manifold) -> Manifold:
        """Apply transformation to manifold"""
        pass

    @abstractmethod
    def inverse(self) -> Optional['TransformationProtocol']:
        """Return inverse transformation if it exists"""
        pass

    @abstractmethod
    def preserves(self) -> Set[Invariant]:
        """Which topological invariants does this preserve?"""
        pass

    @abstractmethod
    def compose(self, other: 'TransformationProtocol') -> 'TransformationProtocol':
        """Compose with another transformation"""
        pass

    @property
    @abstractmethod
    def dimensionality_change(self) -> int:
        """How many dimensions added (+) or removed (-)"""
        pass
```

### 4. Coordination Protocol

Abstract swarm coordination:

```python
@runtime_checkable
class CoordinationProtocol(Protocol):
    """
    How agents coordinate in a swarm.
    Supports: mesh networks, hierarchies, democratic voting, market-based.
    """

    @abstractmethod
    async def broadcast(
        self,
        sender: AgentID,
        message: Message,
        scope: BroadcastScope
    ) -> BroadcastResult:
        """Send message to other agents"""
        pass

    @abstractmethod
    async def receive(
        self,
        agent: AgentID,
        filter: Optional[MessageFilter] = None
    ) -> AsyncIterator[Message]:
        """Receive messages for an agent"""
        pass

    @abstractmethod
    async def establish_connection(
        self,
        agent_a: AgentID,
        agent_b: AgentID,
        connection_type: ConnectionType
    ) -> Connection:
        """Create connection between two agents"""
        pass

    @abstractmethod
    def get_topology(self) -> NetworkTopology:
        """Return current network topology"""
        pass

    @abstractmethod
    async def detect_emergence(self) -> Optional[EmergenceEvent]:
        """Detect collective intelligence emergence"""
        pass
```

### 5. Philosophical Stance Protocol

Make philosophical frameworks pluggable:

```python
@runtime_checkable
class PhilosophicalFrameworkProtocol(Protocol):
    """
    How an agent interprets reality and makes decisions.
    Not hardcoded - can be phenomenological, materialist, custom, etc.
    """

    @abstractmethod
    def interpret(self, observation: Observation) -> Interpretation:
        """Interpret observation through philosophical lens"""
        pass

    @abstractmethod
    def judge_coherence(
        self,
        belief_a: Belief,
        belief_b: Belief
    ) -> float:
        """How coherent are these beliefs? (0.0 to 1.0)"""
        pass

    @abstractmethod
    def resolve_conflict(
        self,
        beliefs: List[Belief]
    ) -> Resolution:
        """Resolve conflicting beliefs"""
        pass

    @abstractmethod
    def get_axioms(self) -> Set[Axiom]:
        """Return fundamental assumptions of this framework"""
        pass

    @abstractmethod
    def is_compatible_with(
        self,
        other: 'PhilosophicalFrameworkProtocol'
    ) -> float:
        """Compatibility score with another framework (0.0 to 1.0)"""
        pass
```

---

## Layer 2: Reference Implementations

### Graded Consciousness Model (Reference)

A **reference implementation** showing how to implement consciousness as a spectrum:

```python
@dataclass
class GradedConsciousness:
    """
    Reference implementation: Consciousness as graduated capabilities.

    Instead of discrete states (DORMANT, AWARE, etc.), agents have
    capability scores on multiple dimensions. This is more flexible
    and realistic.
    """

    # Capability scores (0.0 to 1.0)
    perception_fidelity: float = 0.0
    reaction_speed: float = 0.0
    memory_depth: float = 0.0
    introspection_capacity: float = 0.0
    meta_cognitive_ability: float = 0.0
    information_integration: float = 0.0  # Φ (phi) measure
    intentional_coherence: float = 0.0
    qualia_richness: float = 0.0

    def overall_consciousness(self) -> float:
        """Aggregate consciousness score"""
        return sum([
            self.perception_fidelity,
            self.reaction_speed,
            self.memory_depth,
            self.introspection_capacity,
            self.meta_cognitive_ability,
            self.information_integration,
            self.intentional_coherence,
            self.qualia_richness
        ]) / 8

    def describe_state(self) -> str:
        """Generate descriptive label (optional convenience)"""
        score = self.overall_consciousness()
        if score < 0.2:
            return "dormant"
        elif score < 0.4:
            return "reactive"
        elif score < 0.6:
            return "aware"
        elif score < 0.8:
            return "reflective"
        else:
            return "transcendent"

    def can_perform(self, capability: Capability) -> bool:
        """Check if this consciousness level supports a capability"""
        requirements = capability.minimum_scores()
        return all(
            getattr(self, dim) >= req
            for dim, req in requirements.items()
        )
```

### Composable Transformation Algebra (Reference)

```python
class ComposableTransformation:
    """
    Reference implementation of transformation algebra.
    Transformations compose like functions.
    """

    def __init__(
        self,
        forward: Callable[[Manifold], Manifold],
        backward: Optional[Callable[[Manifold], Manifold]] = None,
        invariants: Set[Invariant] = None,
        dim_change: int = 0
    ):
        self._forward = forward
        self._backward = backward
        self._invariants = invariants or set()
        self._dim_change = dim_change

    def transform(self, m: Manifold) -> Manifold:
        return self._forward(m)

    def inverse(self) -> Optional['ComposableTransformation']:
        if self._backward is None:
            return None
        return ComposableTransformation(
            forward=self._backward,
            backward=self._forward,
            invariants=self._invariants,
            dim_change=-self._dim_change
        )

    def compose(self, other: 'ComposableTransformation') -> 'ComposableTransformation':
        """
        Compose transformations: (f ∘ g)(x) = f(g(x))
        """
        return ComposableTransformation(
            forward=lambda m: self.transform(other.transform(m)),
            backward=(
                lambda m: other.inverse().transform(self.inverse().transform(m))
                if self._backward and other._backward else None
            ),
            invariants=self._invariants & other._invariants,  # intersection
            dim_change=self._dim_change + other._dim_change
        )

    def __matmul__(self, other: 'ComposableTransformation'):
        """Use @ operator for composition: f @ g"""
        return self.compose(other)

    def preserves(self) -> Set[Invariant]:
        return self._invariants

    @property
    def dimensionality_change(self) -> int:
        return self._dim_change

# Example usage:
projection = ComposableTransformation(
    forward=lambda m: m.project_to_dim(m.dimension - 1),
    backward=lambda m: m.embed_from_dim(m.dimension + 1),
    invariants={Invariant.TOPOLOGY, Invariant.CONNECTIVITY},
    dim_change=-1
)

folding = ComposableTransformation(
    forward=lambda m: m.fold_along_axis(0),
    backward=lambda m: m.unfold_along_axis(0),
    invariants={Invariant.VOLUME, Invariant.CURVATURE},
    dim_change=0
)

# Compose: fold then project
combined = projection @ folding  # Now a single transformation
result = combined.transform(my_manifold)
```

---

## Practical Benefits of This Architecture

### 1. **Extensibility**
```python
# Add new consciousness model without changing core code
class PlantConsciousness(GradedConsciousness):
    photosynthetic_efficiency: float = 0.0
    root_network_coherence: float = 0.0
    seasonal_memory: float = 0.0

# Use it anywhere that expects ConsciousnessProtocol
plant_agent = Agent(consciousness=PlantConsciousness(...))
```

### 2. **Pluggable Backends**
```python
# Swap storage backend without code changes
storage = ChromaDBStorage()  # or
storage = PineconeStorage()  # or
storage = CustomGraphStorage()

manager = NPCPUManager(storage=storage)  # Dependency injection
```

### 3. **Testability**
```python
# Mock implementations for testing
class MockStorage(VectorStorageProtocol):
    def __init__(self):
        self.data = {}

    async def store(self, collection, id, vector, metadata, document=None):
        self.data[id] = (vector, metadata, document)
        return StorageResult(success=True)

    # ... other methods

# Now tests run instantly without external dependencies
```

### 4. **Composition**
```python
# Build complex transformations from primitives
identity = IdentityTransformation()
projection = ProjectionTransformation(target_dim=3)
rotation = RotationTransformation(angle=np.pi/4)
crystallization = CrystallizationTransformation(pattern="fractal")

# Compose into pipeline
pipeline = crystallization @ rotation @ projection

# Apply to manifold
result = pipeline.transform(input_manifold)

# Check what's preserved
preserved = pipeline.preserves()  # Set of invariants
```

### 5. **Observable Consciousness**
```python
# Consciousness is now measurable, not mystical
agent_consciousness = agent.consciousness

# Test specific capabilities
if agent_consciousness.meta_cognitive_ability > 0.7:
    # This agent can reflect on its own thinking
    meta_thought = agent.meta_cognize(current_thought)

# Track consciousness evolution over time
consciousness_history = []
for epoch in training:
    consciousness_history.append(agent.consciousness.overall_consciousness())

# Visualize consciousness development
plot_consciousness_evolution(consciousness_history)
```

---

## Configuration-Driven Consciousness

Instead of hardcoded states, load from YAML:

```yaml
# consciousness_models/graded_default.yaml
model_type: "graded"
dimensions:
  perception_fidelity:
    weight: 1.0
    description: "How accurately agent perceives environment"

  reaction_speed:
    weight: 0.8
    description: "How quickly agent responds to stimuli"

  memory_depth:
    weight: 1.2
    description: "How much history agent retains"

  introspection_capacity:
    weight: 1.5
    description: "Ability to examine own state"

  meta_cognitive_ability:
    weight: 2.0
    description: "Ability to think about thinking"

  information_integration:
    weight: 1.8
    description: "Φ (phi) - integrated information"

  intentional_coherence:
    weight: 1.0
    description: "Goal consistency and clarity"

  qualia_richness:
    weight: 1.3
    description: "Subjective experience depth"

thresholds:
  dormant: 0.0
  reactive: 0.2
  aware: 0.4
  reflective: 0.6
  meta_aware: 0.8
  transcendent: 0.95

aggregation: "weighted_mean"
```

```python
# Load consciousness model from config
consciousness_model = ConsciousnessModel.from_yaml("consciousness_models/graded_default.yaml")

# Create agent with this model
agent = Agent(consciousness_model=consciousness_model)

# User can create their own models!
# consciousness_models/my_custom_model.yaml
```

---

## Summary: From Literal to Elegantly Abstract

### Before (Too Literal)
```python
class ConsciousnessState(Enum):
    DORMANT = "DORMANT"
    REACTIVE = "REACTIVE"
    # ... hardcoded, not extensible
```

### After (Elegantly Abstract)
```python
@runtime_checkable
class ConsciousnessProtocol(Protocol):
    def perceive(self, stimulus: Any) -> Perception: ...
    def introspect(self) -> SelfModel: ...
    # ... capabilities, not states
```

### Benefits
1. ✅ **More Implementable**: Clear protocols with reference implementations
2. ✅ **Not Too Literal**: Flexible, extensible, composable
3. ✅ **Continued Abstraction**: Deeper theoretical grounding through protocols
4. ✅ **Observable**: Consciousness is measurable, not mystical
5. ✅ **Composable**: Small primitives combine elegantly
6. ✅ **Testable**: Easy to mock and test
7. ✅ **Extensible**: Users can create custom models
8. ✅ **Interoperable**: Protocols work with any compliant implementation

---

## Next Steps

1. **Implement Core Protocols**: Start with `ConsciousnessProtocol` and `VectorStorageProtocol`
2. **Create Reference Implementations**: `GradedConsciousness`, `ChromaDBStorage`
3. **Migrate Existing Code**: Refactor current implementations to use protocols
4. **Document Patterns**: Show how to create custom implementations
5. **Build Examples**: Demonstrate extensibility with multiple consciousness models
