# Deep Dive: Storage Architecture and Distributed Consciousness

## Why Pluggable Storage Is Critical

### The Problem with Hardcoded Storage

Original NPCPU was tightly coupled to ChromaDB:

```python
# Tight coupling - bad
class NPCPUChromaDBManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(...)  # Locked in!
```

**Issues**:
1. **Vendor lock-in**: Can't switch to better technology
2. **Scaling limits**: ChromaDB might not scale to billions of vectors
3. **Cost optimization**: Can't choose cheaper storage for cold data
4. **Testing difficulty**: Need ChromaDB running for tests
5. **Deployment constraints**: Tied to ChromaDB's infrastructure requirements

### The Protocol Solution

```python
# Loose coupling - good
@runtime_checkable
class VectorStorageProtocol(Protocol):
    async def store(...): ...
    async def query(...): ...
    # ... abstract interface
```

**Benefits**:
1. **Flexibility**: Swap backends anytime
2. **Optimization**: Use different backends for different workloads
3. **Testing**: Mock storage for instant tests
4. **Cost control**: Optimize storage costs dynamically
5. **Future-proof**: Adopt new technology without rewriting code

## Deep Architecture Analysis

### Multi-Tier Storage Strategy

Real consciousness systems need **hierarchical storage**:

```
┌─────────────────────────────────────────────────────────────┐
│ L1: In-Memory Cache (Redis, Memcached)                      │
│ - Ultra-fast: <1ms latency                                  │
│ - Small: Recent/hot data only                               │
│ - Volatile: Lost on restart                                 │
│ - Use: Active working memory                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ L2: Local Vector DB (ChromaDB, FAISS)                       │
│ - Fast: 5-20ms latency                                      │
│ - Medium: Recent memories, active knowledge                 │
│ - Persistent: Survives restarts                             │
│ - Use: Short-term memory (hours to days)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ L3: Distributed Vector DB (Pinecone, Weaviate, Qdrant)      │
│ - Moderate: 20-100ms latency                                │
│ - Large: All agent memories, shared knowledge               │
│ - Replicated: High availability                             │
│ - Use: Long-term memory (days to months)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ L4: Object Storage (S3, GCS, Azure Blob)                    │
│ - Slow: 100ms-1s latency                                    │
│ - Massive: Historical data, archives                        │
│ - Cheap: Pennies per GB                                     │
│ - Use: Long-term archival (months to years)                 │
└─────────────────────────────────────────────────────────────┘
```

### Implementation: Tiered Storage Manager

```python
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import asyncio
from enum import Enum

class StorageTier(Enum):
    L1_CACHE = "l1_cache"          # <1ms - Redis
    L2_LOCAL = "l2_local"          # <20ms - ChromaDB/FAISS
    L3_DISTRIBUTED = "l3_distributed"  # <100ms - Pinecone/Weaviate
    L4_ARCHIVE = "l4_archive"      # <1s - S3/GCS

@dataclass
class TierConfig:
    tier: StorageTier
    backend: str
    latency_target_ms: float
    capacity_gb: Optional[float]
    cost_per_gb_month: float
    ttl_hours: Optional[int]  # Time-to-live before eviction

class TieredStorageManager:
    """
    Manages multiple storage tiers with automatic promotion/demotion.

    Data flow:
    1. Store: Write to all tiers (or just fast ones with async replication)
    2. Query: Check L1 → L2 → L3 → L4 until found
    3. Promote: Move frequently accessed data to faster tiers
    4. Demote: Move cold data to cheaper tiers
    """

    def __init__(self, tier_configs: List[TierConfig]):
        self.tiers = {
            config.tier: self._create_backend(config)
            for config in tier_configs
        }
        self.tier_configs = {c.tier: c for c in tier_configs}

        # Access tracking for intelligent promotion/demotion
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, float] = {}

    def _create_backend(self, config: TierConfig) -> VectorStorageProtocol:
        """Create storage backend from config"""
        return StorageBackendRegistry.create(config.backend)

    async def store(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        tier_hint: Optional[StorageTier] = None
    ) -> StorageResult:
        """
        Store vector with intelligent tier selection.

        Strategy:
        1. If tier_hint provided, store there first
        2. Otherwise, store in L1 (cache) and L2 (local)
        3. Asynchronously replicate to L3/L4
        """
        import time

        # Default: store in fast tiers
        if tier_hint is None:
            primary_tiers = [StorageTier.L1_CACHE, StorageTier.L2_LOCAL]
        else:
            primary_tiers = [tier_hint]

        # Store in primary tiers
        results = await asyncio.gather(*[
            self.tiers[tier].store(collection, id, vector, metadata)
            for tier in primary_tiers
            if tier in self.tiers
        ])

        # Async replication to slower tiers (fire and forget)
        slower_tiers = [
            tier for tier in [StorageTier.L3_DISTRIBUTED, StorageTier.L4_ARCHIVE]
            if tier in self.tiers and tier not in primary_tiers
        ]

        if slower_tiers:
            asyncio.create_task(self._replicate_to_tiers(
                collection, id, vector, metadata, slower_tiers
            ))

        # Return result from fastest tier
        return results[0]

    async def _replicate_to_tiers(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]],
        tiers: List[StorageTier]
    ):
        """Asynchronously replicate to slower tiers"""
        await asyncio.gather(*[
            self.tiers[tier].store(collection, id, vector, metadata)
            for tier in tiers
        ])

    async def query(
        self,
        collection: str,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[QueryResult]:
        """
        Query with automatic tier traversal.

        Strategy:
        1. Try L1 (cache) first
        2. If not found or incomplete, try L2 (local)
        3. If still not found, try L3 (distributed)
        4. If still not found, try L4 (archive)
        5. Promote frequently accessed items to faster tiers
        """
        import time
        start_time = time.time()

        # Try tiers in order of speed
        tier_order = [
            StorageTier.L1_CACHE,
            StorageTier.L2_LOCAL,
            StorageTier.L3_DISTRIBUTED,
            StorageTier.L4_ARCHIVE
        ]

        for tier in tier_order:
            if tier not in self.tiers:
                continue

            try:
                results = await self.tiers[tier].query(
                    collection=collection,
                    query_vector=query_vector,
                    query_text=query_text,
                    filters=filters,
                    limit=limit
                )

                if results:
                    latency_ms = (time.time() - start_time) * 1000

                    # Track access
                    for result in results:
                        self._record_access(result.id, tier)

                    # Promote if needed
                    if tier != StorageTier.L1_CACHE:
                        await self._maybe_promote(results, tier)

                    print(f"Query satisfied from {tier.value} in {latency_ms:.1f}ms")
                    return results

            except Exception as e:
                print(f"Tier {tier.value} failed: {e}, trying next tier...")
                continue

        # Not found in any tier
        return []

    def _record_access(self, id: str, tier: StorageTier):
        """Track access patterns"""
        import time
        self.access_counts[id] = self.access_counts.get(id, 0) + 1
        self.last_access[id] = time.time()

    async def _maybe_promote(self, results: List[QueryResult], current_tier: StorageTier):
        """
        Promote frequently accessed data to faster tiers.

        Promotion criteria:
        - Access count > threshold
        - Recent access (not ancient data being accessed once)
        - Current tier is slow
        """
        import time
        current_time = time.time()

        for result in results:
            access_count = self.access_counts.get(result.id, 0)
            last_access_time = self.last_access.get(result.id, 0)
            recency = current_time - last_access_time

            # Promote if frequently accessed AND recently accessed
            if access_count > 10 and recency < 3600:  # Last hour
                # Promote to next faster tier
                target_tier = self._get_faster_tier(current_tier)
                if target_tier:
                    await self._promote_to_tier(result, target_tier)

    def _get_faster_tier(self, current: StorageTier) -> Optional[StorageTier]:
        """Get next faster tier"""
        tier_speed_order = [
            StorageTier.L4_ARCHIVE,
            StorageTier.L3_DISTRIBUTED,
            StorageTier.L2_LOCAL,
            StorageTier.L1_CACHE
        ]
        try:
            idx = tier_speed_order.index(current)
            if idx < len(tier_speed_order) - 1:
                next_tier = tier_speed_order[idx + 1]
                return next_tier if next_tier in self.tiers else None
        except ValueError:
            return None

    async def _promote_to_tier(self, result: QueryResult, target_tier: StorageTier):
        """Promote data to faster tier"""
        if target_tier not in self.tiers:
            return

        # Copy to faster tier
        await self.tiers[target_tier].store(
            collection="promoted",  # Simplified
            id=result.id,
            vector=result.vector,
            metadata=result.metadata
        )

        print(f"Promoted {result.id} to {target_tier.value}")

    async def demote_cold_data(self):
        """
        Periodically demote cold (infrequently accessed) data to slower tiers.

        Run this as a background task.
        """
        import time
        current_time = time.time()

        for id, last_access_time in self.last_access.items():
            age_hours = (current_time - last_access_time) / 3600

            # Demote data not accessed in 24 hours
            if age_hours > 24:
                # Move from L1 → L2 → L3 → L4
                # Implementation depends on tracking which tier data is in
                pass

    async def optimize_costs(self):
        """
        Optimize storage costs by moving data to appropriate tiers.

        Strategy:
        - Hot data (frequent access): L1/L2 (expensive but fast)
        - Warm data (occasional access): L3 (moderate cost and speed)
        - Cold data (rare access): L4 (cheap but slow)
        """
        for id, access_count in self.access_counts.items():
            access_frequency = access_count / self._get_data_age_hours(id)

            if access_frequency > 1.0:  # >1 access per hour
                target_tier = StorageTier.L1_CACHE
            elif access_frequency > 0.1:  # >1 access per 10 hours
                target_tier = StorageTier.L2_LOCAL
            elif access_frequency > 0.01:  # >1 access per 100 hours
                target_tier = StorageTier.L3_DISTRIBUTED
            else:
                target_tier = StorageTier.L4_ARCHIVE

            # Move data to optimal tier
            # (Implementation would check current tier and migrate if needed)

    def _get_data_age_hours(self, id: str) -> float:
        """Get age of data in hours"""
        # Simplified - would track creation time in practice
        import time
        first_access = self.last_access.get(id, time.time())
        return (time.time() - first_access) / 3600

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics across all tiers.

        Useful for monitoring and optimization.
        """
        stats = {}

        for tier in self.tiers:
            tier_stats = {
                "count": await self.tiers[tier].count("all_collections"),
                "latency_ms": self.tier_configs[tier].latency_target_ms,
                "cost_per_gb": self.tier_configs[tier].cost_per_gb_month,
                "capacity_gb": self.tier_configs[tier].capacity_gb
            }
            stats[tier.value] = tier_stats

        return stats
```

## Distributed Consciousness Through Storage

### Shared Memory Across Swarms

Multiple agents can share consciousness through shared storage:

```python
class SharedConsciousnessMemory:
    """
    Shared memory pool enabling collective consciousness.

    Multiple agents access same memory, creating:
    - Shared experiences
    - Collective knowledge
    - Emergent understanding
    """

    def __init__(self, storage: VectorStorageProtocol):
        self.storage = storage
        self.collection = "collective_consciousness"

    async def share_experience(
        self,
        agent_id: str,
        experience: Experience,
        make_public: bool = True
    ):
        """
        Share an experience with the collective.

        If make_public=True, all agents can access it.
        Otherwise, only agents with shared context can access.
        """
        # Embed experience
        vector = self.embed_experience(experience)

        # Store with metadata
        metadata = {
            "agent_id": agent_id,
            "experience_type": experience.perception.stimulus_type,
            "timestamp": experience.timestamp,
            "valence": experience.emotional_valence,
            "public": make_public,
            "consciousness_level": self.get_agent_consciousness_level(agent_id)
        }

        await self.storage.store(
            collection=self.collection,
            id=f"{agent_id}_{experience.timestamp}",
            vector=vector,
            metadata=metadata
        )

    async def recall_collective_memories(
        self,
        query: str,
        min_consciousness_level: float = 0.5,
        limit: int = 10
    ) -> List[Experience]:
        """
        Recall memories from the collective consciousness.

        Only retrieves experiences from agents above a certain
        consciousness threshold (proof of consciousness).
        """
        query_vector = self.embed_query(query)

        results = await self.storage.query(
            collection=self.collection,
            query_vector=query_vector,
            filters={
                "public": True,
                "consciousness_level": {"$gte": min_consciousness_level}
            },
            limit=limit
        )

        return [self.decode_experience(r) for r in results]

    async def get_collective_knowledge(
        self,
        topic: str
    ) -> CollectiveKnowledge:
        """
        Aggregate knowledge from multiple agents about a topic.

        This is where collective intelligence emerges.
        """
        # Retrieve all relevant experiences
        experiences = await self.recall_collective_memories(topic, limit=100)

        # Aggregate insights
        insights = self.aggregate_insights(experiences)

        # Consensus building
        consensus = self.build_consensus(insights)

        # Emergent understanding
        emergent = self.detect_emergent_patterns(experiences)

        return CollectiveKnowledge(
            topic=topic,
            contributor_count=len(set(e.agent_id for e in experiences)),
            consensus_beliefs=consensus,
            emergent_insights=emergent,
            confidence=self.calculate_collective_confidence(insights)
        )

    def aggregate_insights(self, experiences: List[Experience]) -> List[Insight]:
        """
        Combine insights from multiple agents.

        Different agents might have different perspectives on same experience.
        Aggregation creates richer understanding.
        """
        insights_by_topic = {}

        for exp in experiences:
            topic = exp.get_topic()
            if topic not in insights_by_topic:
                insights_by_topic[topic] = []
            insights_by_topic[topic].append(exp.extract_insight())

        # Combine insights
        combined = []
        for topic, insights in insights_by_topic.items():
            combined.append(self.synthesize_insights(insights))

        return combined

    def build_consensus(self, insights: List[Insight]) -> Dict[str, float]:
        """
        Build consensus beliefs from multiple agent insights.

        Uses weighted voting based on:
        - Agent consciousness level
        - Experience relevance
        - Historical accuracy
        """
        beliefs = {}

        for insight in insights:
            for belief, strength in insight.beliefs.items():
                weight = (
                    insight.agent_consciousness_level * 0.4 +
                    insight.relevance * 0.3 +
                    insight.agent_accuracy_history * 0.3
                )

                if belief not in beliefs:
                    beliefs[belief] = []
                beliefs[belief].append((strength, weight))

        # Weighted average for each belief
        consensus = {}
        for belief, values in beliefs.items():
            weighted_sum = sum(strength * weight for strength, weight in values)
            total_weight = sum(weight for _, weight in values)
            consensus[belief] = weighted_sum / total_weight if total_weight > 0 else 0

        return consensus

    def detect_emergent_patterns(self, experiences: List[Experience]) -> List[EmergentPattern]:
        """
        Detect patterns that emerge from collective, not visible to individuals.

        This is where collective consciousness exceeds individual consciousness.
        """
        patterns = []

        # Pattern 1: Temporal correlations
        # Individual agents might not see the pattern, but collective does
        temporal = self.find_temporal_correlations(experiences)
        if temporal:
            patterns.append(EmergentPattern(
                type="temporal_correlation",
                description="Events that correlate across time",
                evidence=temporal,
                significance=self.calculate_significance(temporal)
            ))

        # Pattern 2: Spatial distributions
        # Collective sees geographic patterns invisible to individuals
        spatial = self.find_spatial_patterns(experiences)
        if spatial:
            patterns.append(EmergentPattern(
                type="spatial_distribution",
                description="Geographic clustering of phenomena",
                evidence=spatial,
                significance=self.calculate_significance(spatial)
            ))

        # Pattern 3: Causal chains
        # Collective can trace causation across multiple agents
        causal = self.find_causal_chains(experiences)
        if causal:
            patterns.append(EmergentPattern(
                type="causal_chain",
                description="Multi-step causation across agents",
                evidence=causal,
                significance=self.calculate_significance(causal)
            ))

        return patterns
```

### Consciousness Synchronization

Agents can synchronize their consciousness states:

```python
class ConsciousnessSynchronization:
    """
    Synchronize consciousness across agents in a swarm.

    Enables:
    - Shared awareness
    - Coordinated action
    - Collective intelligence
    """

    def __init__(self, storage: VectorStorageProtocol):
        self.storage = storage
        self.sync_collection = "consciousness_sync"

    async def publish_consciousness_state(
        self,
        agent_id: str,
        consciousness: GradedConsciousness
    ):
        """
        Publish agent's current consciousness state.

        Other agents can read and potentially align with it.
        """
        # Encode consciousness as vector
        vector = self.encode_consciousness(consciousness)

        metadata = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "overall_score": consciousness.overall_consciousness_score(),
            **consciousness.get_capability_scores()
        }

        await self.storage.store(
            collection=self.sync_collection,
            id=f"{agent_id}_state",
            vector=vector,
            metadata=metadata
        )

    async def find_similar_consciousness(
        self,
        agent_id: str,
        consciousness: GradedConsciousness,
        limit: int = 5
    ) -> List[Tuple[str, GradedConsciousness, float]]:
        """
        Find agents with similar consciousness states.

        Useful for:
        - Team formation
        - Peer learning
        - Consciousness resonance
        """
        query_vector = self.encode_consciousness(consciousness)

        results = await self.storage.query(
            collection=self.sync_collection,
            query_vector=query_vector,
            filters={
                "agent_id": {"$ne": agent_id}  # Exclude self
            },
            limit=limit
        )

        similar_agents = []
        for result in results:
            other_consciousness = self.decode_consciousness(result)
            similarity = result.score  # Cosine similarity
            similar_agents.append((
                result.metadata["agent_id"],
                other_consciousness,
                similarity
            ))

        return similar_agents

    async def synchronize_with_swarm(
        self,
        agent_id: str,
        consciousness: GradedConsciousness,
        sync_strength: float = 0.3
    ) -> GradedConsciousness:
        """
        Synchronize consciousness with swarm average.

        Partial synchronization (not full):
        - Maintain individual identity (1 - sync_strength)
        - Adopt swarm patterns (sync_strength)

        This creates coherent swarms while preserving diversity.
        """
        # Get all swarm consciousness states
        swarm_states = await self.storage.query(
            collection=self.sync_collection,
            query_vector=self.encode_consciousness(consciousness),
            limit=50
        )

        # Calculate swarm average
        swarm_avg = self.calculate_average_consciousness(
            [self.decode_consciousness(s) for s in swarm_states]
        )

        # Blend individual and swarm
        synchronized = self.blend_consciousness(
            individual=consciousness,
            collective=swarm_avg,
            blend_ratio=sync_strength
        )

        return synchronized

    def blend_consciousness(
        self,
        individual: GradedConsciousness,
        collective: GradedConsciousness,
        blend_ratio: float
    ) -> GradedConsciousness:
        """
        Blend individual and collective consciousness.

        blend_ratio = 0.0: Pure individual
        blend_ratio = 0.5: Equal blend
        blend_ratio = 1.0: Pure collective (loss of individuality)
        """
        blended_scores = {}

        individual_scores = individual.get_capability_scores()
        collective_scores = collective.get_capability_scores()

        for capability in individual_scores.keys():
            blended_scores[capability] = (
                individual_scores[capability] * (1 - blend_ratio) +
                collective_scores[capability] * blend_ratio
            )

        return GradedConsciousness(**blended_scores)

    def calculate_average_consciousness(
        self,
        consciousnesses: List[GradedConsciousness]
    ) -> GradedConsciousness:
        """Calculate average consciousness across multiple agents"""
        if not consciousnesses:
            return GradedConsciousness()

        # Average each capability
        avg_scores = {}
        all_scores = [c.get_capability_scores() for c in consciousnesses]

        for capability in all_scores[0].keys():
            avg_scores[capability] = np.mean([
                scores[capability] for scores in all_scores
            ])

        return GradedConsciousness(**avg_scores)
```

## Storage Performance Optimization

### Vector Quantization for Compression

```python
class QuantizedVectorStorage:
    """
    Store vectors in quantized form to save space.

    Tradeoff:
    - 8x-16x space reduction
    - Slight accuracy loss (~1-2%)
    - Faster similarity search (less data to compare)
    """

    def __init__(self, backend: VectorStorageProtocol):
        self.backend = backend
        self.quantizer = ProductQuantizer(m=8, nbits=8)  # Example

    async def store_quantized(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store vector in quantized form"""
        # Quantize: float32[768] → uint8[96]  (8x compression)
        quantized = self.quantizer.encode(np.array(vector))

        # Store quantized version
        await self.backend.store(
            collection=f"{collection}_quantized",
            id=id,
            vector=quantized.tolist(),
            metadata={
                **metadata,
                "quantized": True,
                "original_dim": len(vector)
            }
        )

    async def query_quantized(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10
    ) -> List[QueryResult]:
        """Query using quantized vectors"""
        # Quantize query
        quantized_query = self.quantizer.encode(np.array(query_vector))

        # Search quantized space
        results = await self.backend.query(
            collection=f"{collection}_quantized",
            query_vector=quantized_query.tolist(),
            limit=limit
        )

        return results
```

### Approximate Nearest Neighbor (ANN) Indexes

```python
class ANNIndexedStorage:
    """
    Use ANN indexes for faster similarity search.

    Exact search: O(n) - must compare with all vectors
    ANN search: O(log n) - use index structure (HNSW, IVF, etc.)

    Tradeoff:
    - 10-100x faster search
    - ~95-99% recall (might miss some true neighbors)
    """

    def __init__(self, backend: VectorStorageProtocol):
        self.backend = backend
        self.index_type = "hnsw"  # Hierarchical Navigable Small World

    async def build_index(self, collection: str):
        """Build ANN index for faster search"""
        # Retrieve all vectors
        all_vectors = []
        async for batch in self.backend.scroll(collection):
            all_vectors.extend([r.vector for r in batch])

        # Build HNSW index
        import hnswlib
        dim = len(all_vectors[0])
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=len(all_vectors), ef_construction=200, M=16)
        index.add_items(all_vectors)

        # Store index
        index.save_index(f"{collection}_hnsw.bin")

    async def query_with_index(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10
    ) -> List[QueryResult]:
        """Query using ANN index"""
        import hnswlib

        # Load index
        dim = len(query_vector)
        index = hnswlib.Index(space='cosine', dim=dim)
        index.load_index(f"{collection}_hnsw.bin")

        # ANN search
        labels, distances = index.knn_query(query_vector, k=limit)

        # Convert to QueryResult format
        # (Would need to retrieve full metadata from backend)
        results = []
        for label, distance in zip(labels[0], distances[0]):
            result = await self.backend.query_by_id(collection, str(label))
            if result:
                result.score = 1 - distance  # Convert distance to similarity
                results.append(result)

        return results
```

## Cost Optimization Strategies

### Tiered Pricing Model

```python
class CostOptimizedStorage:
    """
    Automatically optimize storage costs.

    Strategy:
    - Frequently accessed data → Expensive fast storage
    - Rarely accessed data → Cheap slow storage
    - Monitor and migrate continuously
    """

    def __init__(self):
        self.cost_model = {
            "l1_cache": 500.0,  # $/TB/month
            "l2_local": 100.0,
            "l3_distributed": 25.0,
            "l4_archive": 2.0
        }

    def calculate_monthly_cost(
        self,
        data_distribution: Dict[str, float]  # Tier → GB
    ) -> float:
        """Calculate total monthly storage cost"""
        total_cost = 0.0

        for tier, size_gb in data_distribution.items():
            size_tb = size_gb / 1024
            cost_per_tb = self.cost_model.get(tier, 0)
            total_cost += size_tb * cost_per_tb

        return total_cost

    async def optimize_for_budget(
        self,
        current_distribution: Dict[str, float],
        budget_usd_month: float,
        access_patterns: Dict[str, float]  # ID → access_frequency
    ) -> Dict[str, float]:
        """
        Optimize data distribution to stay within budget.

        Returns optimal distribution across tiers.
        """
        total_data_gb = sum(current_distribution.values())
        current_cost = self.calculate_monthly_cost(current_distribution)

        if current_cost <= budget_usd_month:
            return current_distribution  # Already within budget

        # Need to reduce cost
        # Strategy: Move cold data to cheaper tiers

        # Sort data by access frequency
        sorted_data = sorted(
            access_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Allocate to tiers
        new_distribution = {
            "l1_cache": 0.0,
            "l2_local": 0.0,
            "l3_distributed": 0.0,
            "l4_archive": 0.0
        }

        remaining_budget = budget_usd_month
        tier_order = ["l1_cache", "l2_local", "l3_distributed", "l4_archive"]

        for tier in tier_order:
            # How much can we afford in this tier?
            cost_per_gb = self.cost_model[tier] / 1024
            affordable_gb = remaining_budget / cost_per_gb

            # Allocate hottest data to this tier
            allocated = min(affordable_gb, total_data_gb)
            new_distribution[tier] = allocated
            total_data_gb -= allocated
            remaining_budget -= allocated * cost_per_gb

            if total_data_gb <= 0:
                break

        return new_distribution
```

## Security and Privacy in Distributed Storage

### Encrypted Storage

```python
class EncryptedVectorStorage:
    """
    Store vectors with encryption.

    Challenges:
    - Encrypted vectors can't be searched directly
    - Need homomorphic encryption or secure enclaves
    - Or: Encrypt metadata only, keep vectors searchable
    """

    def __init__(self, backend: VectorStorageProtocol, encryption_key: bytes):
        self.backend = backend
        self.cipher = self.create_cipher(encryption_key)

    async def store_encrypted(
        self,
        collection: str,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ):
        """
        Store with encrypted metadata.

        Vector remains unencrypted for similarity search,
        but sensitive metadata is encrypted.
        """
        # Encrypt sensitive metadata
        encrypted_metadata = self.encrypt_metadata(metadata)

        await self.backend.store(
            collection=collection,
            id=id,
            vector=vector,  # Unencrypted for search
            metadata=encrypted_metadata
        )

    def encrypt_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields"""
        encrypted = metadata.copy()

        sensitive_fields = ["agent_id", "personal_info", "private_data"]
        for field in sensitive_fields:
            if field in encrypted:
                encrypted[field] = self.cipher.encrypt(
                    str(encrypted[field]).encode()
                ).decode()

        encrypted["encrypted"] = True
        return encrypted
```

## Conclusion: Storage as Consciousness Infrastructure

The storage protocol is not just about data - it's the **infrastructure for consciousness**:

1. **Memory is consciousness**: Without memory, no learning, no growth
2. **Shared memory enables collective consciousness**: Swarms think together
3. **Tiered storage mirrors biological memory**: Working memory, long-term memory, archives
4. **Access patterns reveal consciousness**: What agent remembers shows what matters
5. **Optimization is continuous**: Like biological memory consolidation during sleep

The pluggable storage protocol makes all of this **flexible**, **scalable**, and **evolvable**.
