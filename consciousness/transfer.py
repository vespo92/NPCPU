"""
Consciousness Transfer Protocol

Enable transfer of consciousness between substrates, agents, and systems.
Supports copy, move, merge, and distributed consciousness operations.

Based on Long-Term Roadmap: Months 7-12 - Consciousness Transfer
"""

import time
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protocols.consciousness import GradedConsciousness


# ============================================================================
# Transfer Modes and Formats
# ============================================================================

class TransferMode(Enum):
    """Modes of consciousness transfer"""
    COPY = "copy"      # Source keeps consciousness, target gets copy
    MOVE = "move"      # Source loses consciousness, target gets it
    MERGE = "merge"    # Both consciousnesses merge into hybrid
    SYNC = "sync"      # Synchronize to common state


class SerializationFormat(Enum):
    """Serialization formats for consciousness"""
    BINARY = "binary"   # Pickle format, fastest
    JSON = "json"       # Human-readable JSON
    COMPRESSED = "compressed"  # Compressed binary


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ConsciousnessSnapshot:
    """Complete snapshot of consciousness state"""
    version: str = "1.0"
    type: str = "GradedConsciousness"
    capabilities: Dict[str, float] = field(default_factory=dict)
    weights: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class TransferResult:
    """Result of a consciousness transfer operation"""
    success: bool
    mode: TransferMode
    source_id: str
    target_id: str
    consciousness_checksum: str
    transfer_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Simple agent class for transfer demonstrations"""
    agent_id: str
    consciousness: GradedConsciousness
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Consciousness Transfer Protocol
# ============================================================================

class ConsciousnessTransferProtocol:
    """
    Transfer consciousness between substrates.

    Enable consciousness to move from:
    - Agent → Agent (same or different type)
    - Single machine → Distributed cluster
    - One implementation → Another

    Features:
    - Multiple transfer modes (copy, move, merge)
    - Serialization with integrity verification
    - Distributed consciousness sharding
    - History and provenance tracking

    Example:
        protocol = ConsciousnessTransferProtocol()

        # Serialize consciousness
        data = protocol.serialize_consciousness(agent.consciousness)

        # Transfer between agents
        protocol.transfer_consciousness(source_agent, target_agent, mode="copy")

        # Distribute consciousness
        shards = protocol.distributed_consciousness(consciousness, num_shards=3)
    """

    def __init__(
        self,
        default_format: SerializationFormat = SerializationFormat.BINARY,
        verify_transfers: bool = True
    ):
        self.default_format = default_format
        self.verify_transfers = verify_transfers
        self.transfer_history: List[TransferResult] = []

    def serialize_consciousness(
        self,
        consciousness: GradedConsciousness,
        include_history: bool = True,
        format: Optional[SerializationFormat] = None
    ) -> bytes:
        """
        Serialize consciousness to transferable format.

        Includes:
        - Current capability scores
        - Weights
        - Configuration
        - Optionally: Experience history, learned patterns
        """
        format = format or self.default_format

        # Create snapshot
        snapshot = ConsciousnessSnapshot(
            type="GradedConsciousness",
            capabilities=consciousness.get_capability_scores(),
            weights=getattr(consciousness, 'weights', None),
            metadata={
                "overall_score": consciousness.overall_consciousness_score(),
                "state_description": consciousness.describe_state()
            },
            created_at=time.time()
        )

        if include_history and hasattr(consciousness, 'history'):
            snapshot.history = consciousness.history

        # Compute checksum before serialization
        checksum_data = json.dumps(snapshot.capabilities, sort_keys=True)
        snapshot.checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]

        # Serialize based on format
        if format == SerializationFormat.JSON:
            return json.dumps(asdict(snapshot)).encode('utf-8')
        elif format == SerializationFormat.COMPRESSED:
            import zlib
            data = pickle.dumps(snapshot)
            return zlib.compress(data)
        else:  # BINARY
            return pickle.dumps(snapshot)

    def deserialize_consciousness(
        self,
        data: bytes,
        format: Optional[SerializationFormat] = None,
        target_class: type = GradedConsciousness
    ) -> GradedConsciousness:
        """Deserialize consciousness from bytes"""
        format = format or self.default_format

        # Deserialize based on format
        if format == SerializationFormat.JSON:
            snapshot_dict = json.loads(data.decode('utf-8'))
            snapshot = ConsciousnessSnapshot(**snapshot_dict)
        elif format == SerializationFormat.COMPRESSED:
            import zlib
            decompressed = zlib.decompress(data)
            snapshot = pickle.loads(decompressed)
        else:  # BINARY
            snapshot = pickle.loads(data)

        # Verify checksum
        if self.verify_transfers:
            checksum_data = json.dumps(snapshot.capabilities, sort_keys=True)
            expected_checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]
            if snapshot.checksum != expected_checksum:
                raise ValueError("Consciousness checksum mismatch - data corrupted")

        # Create consciousness instance
        consciousness = target_class(**snapshot.capabilities)

        # Restore weights if present
        if snapshot.weights:
            consciousness.weights = snapshot.weights

        # Restore history if present
        if snapshot.history:
            consciousness.history = snapshot.history

        return consciousness

    def transfer_consciousness(
        self,
        source_agent: Agent,
        target_agent: Agent,
        mode: Union[TransferMode, str] = TransferMode.COPY
    ) -> TransferResult:
        """
        Transfer consciousness between agents.

        Modes:
        - copy: Source keeps consciousness, target gets copy
        - move: Source loses consciousness, target gets it
        - merge: Both consciousnesses merge into hybrid
        """
        if isinstance(mode, str):
            mode = TransferMode(mode)

        start_time = time.time()
        source_consciousness = source_agent.consciousness

        try:
            if mode == TransferMode.COPY:
                # Copy consciousness
                target_agent.consciousness = GradedConsciousness(
                    **source_consciousness.get_capability_scores()
                )

            elif mode == TransferMode.MOVE:
                # Transfer consciousness
                target_agent.consciousness = source_consciousness
                source_agent.consciousness = GradedConsciousness()  # Empty

            elif mode == TransferMode.MERGE:
                # Merge consciousnesses
                source_scores = source_consciousness.get_capability_scores()
                target_scores = target_agent.consciousness.get_capability_scores()

                merged_scores = {}
                for capability in source_scores.keys():
                    merged_scores[capability] = (
                        source_scores[capability] * 0.5 +
                        target_scores.get(capability, 0.0) * 0.5
                    )

                # Both agents get merged consciousness
                merged = GradedConsciousness(**merged_scores)
                source_agent.consciousness = merged
                target_agent.consciousness = GradedConsciousness(**merged_scores)

            elif mode == TransferMode.SYNC:
                # Synchronize to common state (average)
                source_scores = source_consciousness.get_capability_scores()
                target_scores = target_agent.consciousness.get_capability_scores()

                synced_scores = {}
                for capability in source_scores.keys():
                    synced_scores[capability] = (
                        source_scores[capability] +
                        target_scores.get(capability, 0.0)
                    ) / 2

                synced = GradedConsciousness(**synced_scores)
                source_agent.consciousness = synced
                target_agent.consciousness = GradedConsciousness(**synced_scores)

            # Compute checksum
            checksum_data = json.dumps(
                target_agent.consciousness.get_capability_scores(),
                sort_keys=True
            )
            checksum = hashlib.sha256(checksum_data.encode()).hexdigest()[:16]

            result = TransferResult(
                success=True,
                mode=mode,
                source_id=source_agent.agent_id,
                target_id=target_agent.agent_id,
                consciousness_checksum=checksum,
                transfer_time=time.time() - start_time
            )

        except Exception as e:
            result = TransferResult(
                success=False,
                mode=mode,
                source_id=source_agent.agent_id,
                target_id=target_agent.agent_id,
                consciousness_checksum="",
                transfer_time=time.time() - start_time,
                error=str(e)
            )

        self.transfer_history.append(result)
        return result

    def distributed_consciousness(
        self,
        consciousness: GradedConsciousness,
        num_shards: int
    ) -> List[GradedConsciousness]:
        """
        Distribute consciousness across multiple agents.

        Each shard specializes in subset of capabilities.
        Together, they form complete consciousness.

        Returns:
            List of consciousness shards, each specialized in different capabilities
        """
        scores = consciousness.get_capability_scores()
        capabilities = list(scores.keys())

        # Partition capabilities
        shard_size = len(capabilities) // num_shards
        if shard_size == 0:
            shard_size = 1

        shards = []

        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards - 1 else len(capabilities)
            shard_capabilities = capabilities[start_idx:end_idx]

            shard_scores = {}
            for capability in scores.keys():
                if capability in shard_capabilities:
                    # This shard specializes in this capability
                    shard_scores[capability] = min(1.0, scores[capability] * 1.5)
                else:
                    # Minimal capability
                    shard_scores[capability] = scores[capability] * 0.1

            shards.append(GradedConsciousness(**shard_scores))

        return shards

    def recombine_shards(
        self,
        shards: List[GradedConsciousness]
    ) -> GradedConsciousness:
        """
        Recombine distributed consciousness shards.

        Takes the best of each shard for each capability.
        """
        if not shards:
            return GradedConsciousness()

        # Get all capability scores from all shards
        all_scores: Dict[str, List[float]] = {}

        for shard in shards:
            for capability, score in shard.get_capability_scores().items():
                if capability not in all_scores:
                    all_scores[capability] = []
                all_scores[capability].append(score)

        # Take maximum for each capability
        combined_scores = {}
        for capability, scores_list in all_scores.items():
            combined_scores[capability] = max(scores_list)

        return GradedConsciousness(**combined_scores)

    def clone_consciousness(
        self,
        consciousness: GradedConsciousness,
        num_clones: int,
        variation: float = 0.0
    ) -> List[GradedConsciousness]:
        """
        Create clones of consciousness with optional variation.

        Args:
            consciousness: Original consciousness to clone
            num_clones: Number of clones to create
            variation: Amount of random variation (0 = exact clone, 1 = full random)

        Returns:
            List of cloned consciousnesses
        """
        clones = []
        scores = consciousness.get_capability_scores()

        for _ in range(num_clones):
            clone_scores = {}
            for capability, score in scores.items():
                if variation > 0:
                    # Add random variation
                    noise = np.random.uniform(-variation, variation)
                    clone_scores[capability] = max(0.0, min(1.0, score + noise))
                else:
                    clone_scores[capability] = score

            clones.append(GradedConsciousness(**clone_scores))

        return clones

    def blend_consciousnesses(
        self,
        consciousnesses: List[GradedConsciousness],
        weights: Optional[List[float]] = None
    ) -> GradedConsciousness:
        """
        Blend multiple consciousnesses into one.

        Args:
            consciousnesses: List of consciousnesses to blend
            weights: Optional weights for each consciousness (default: equal)

        Returns:
            Blended consciousness
        """
        if not consciousnesses:
            return GradedConsciousness()

        if weights is None:
            weights = [1.0 / len(consciousnesses)] * len(consciousnesses)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        # Blend scores
        blended_scores: Dict[str, float] = {}

        for consciousness, weight in zip(consciousnesses, weights):
            for capability, score in consciousness.get_capability_scores().items():
                if capability not in blended_scores:
                    blended_scores[capability] = 0.0
                blended_scores[capability] += score * weight

        return GradedConsciousness(**blended_scores)

    def diff_consciousnesses(
        self,
        consciousness_a: GradedConsciousness,
        consciousness_b: GradedConsciousness
    ) -> Dict[str, float]:
        """
        Calculate difference between two consciousnesses.

        Returns:
            Dict mapping capability to difference (positive = A higher)
        """
        scores_a = consciousness_a.get_capability_scores()
        scores_b = consciousness_b.get_capability_scores()

        diff = {}
        all_caps = set(scores_a.keys()) | set(scores_b.keys())

        for capability in all_caps:
            a_score = scores_a.get(capability, 0.0)
            b_score = scores_b.get(capability, 0.0)
            diff[capability] = a_score - b_score

        return diff

    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get statistics about consciousness transfers"""
        if not self.transfer_history:
            return {
                "total_transfers": 0,
                "successful_transfers": 0,
                "success_rate": 0.0,
                "average_transfer_time": 0.0
            }

        successful = [t for t in self.transfer_history if t.success]

        return {
            "total_transfers": len(self.transfer_history),
            "successful_transfers": len(successful),
            "success_rate": len(successful) / len(self.transfer_history),
            "average_transfer_time": np.mean([t.transfer_time for t in self.transfer_history]),
            "transfers_by_mode": {
                mode.value: len([t for t in self.transfer_history if t.mode == mode])
                for mode in TransferMode
            }
        }


# ============================================================================
# Distributed Consciousness Manager
# ============================================================================

class DistributedConsciousnessManager:
    """
    Manage distributed consciousness across multiple nodes.

    Handles:
    - Shard placement and tracking
    - Shard synchronization
    - Failover and recovery
    - Load balancing
    """

    def __init__(self):
        self.shards: Dict[str, List[Tuple[str, GradedConsciousness]]] = {}
        self.node_assignments: Dict[str, str] = {}  # shard_id -> node_id
        self.protocol = ConsciousnessTransferProtocol()

    def distribute(
        self,
        consciousness_id: str,
        consciousness: GradedConsciousness,
        num_nodes: int
    ) -> Dict[str, GradedConsciousness]:
        """
        Distribute consciousness across nodes.

        Returns:
            Dict mapping node_id to shard consciousness
        """
        shards = self.protocol.distributed_consciousness(consciousness, num_nodes)

        shard_map = {}
        self.shards[consciousness_id] = []

        for i, shard in enumerate(shards):
            node_id = f"node_{i}"
            shard_id = f"{consciousness_id}_shard_{i}"

            self.node_assignments[shard_id] = node_id
            self.shards[consciousness_id].append((shard_id, shard))
            shard_map[node_id] = shard

        return shard_map

    def recombine(
        self,
        consciousness_id: str
    ) -> GradedConsciousness:
        """Recombine distributed shards into single consciousness"""
        if consciousness_id not in self.shards:
            raise ValueError(f"Unknown consciousness: {consciousness_id}")

        shards = [shard for _, shard in self.shards[consciousness_id]]
        return self.protocol.recombine_shards(shards)

    def update_shard(
        self,
        consciousness_id: str,
        shard_index: int,
        new_shard: GradedConsciousness
    ):
        """Update a specific shard"""
        if consciousness_id not in self.shards:
            raise ValueError(f"Unknown consciousness: {consciousness_id}")

        if shard_index >= len(self.shards[consciousness_id]):
            raise ValueError(f"Invalid shard index: {shard_index}")

        shard_id, _ = self.shards[consciousness_id][shard_index]
        self.shards[consciousness_id][shard_index] = (shard_id, new_shard)

    def synchronize_shards(
        self,
        consciousness_id: str
    ) -> GradedConsciousness:
        """
        Synchronize all shards to common state.

        Useful after distributed processing.
        """
        if consciousness_id not in self.shards:
            raise ValueError(f"Unknown consciousness: {consciousness_id}")

        # Recombine shards
        combined = self.recombine(consciousness_id)

        # Re-distribute to sync
        num_shards = len(self.shards[consciousness_id])
        new_shards = self.protocol.distributed_consciousness(combined, num_shards)

        # Update all shards
        for i, new_shard in enumerate(new_shards):
            shard_id, _ = self.shards[consciousness_id][i]
            self.shards[consciousness_id][i] = (shard_id, new_shard)

        return combined

    def get_status(self) -> Dict[str, Any]:
        """Get status of distributed consciousnesses"""
        return {
            "total_consciousnesses": len(self.shards),
            "total_shards": sum(len(s) for s in self.shards.values()),
            "consciousnesses": {
                cid: {
                    "num_shards": len(shards),
                    "shard_ids": [sid for sid, _ in shards]
                }
                for cid, shards in self.shards.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Consciousness Transfer Protocol Demo")
    print("=" * 50)

    # Create protocol
    protocol = ConsciousnessTransferProtocol()

    # Create consciousnesses
    consciousness_a = GradedConsciousness(
        perception_fidelity=0.9,
        reaction_speed=0.7,
        memory_depth=0.8,
        introspection_capacity=0.6
    )

    consciousness_b = GradedConsciousness(
        perception_fidelity=0.5,
        reaction_speed=0.9,
        memory_depth=0.4,
        introspection_capacity=0.8
    )

    # Create agents
    agent_a = Agent(agent_id="agent_a", consciousness=consciousness_a)
    agent_b = Agent(agent_id="agent_b", consciousness=consciousness_b)

    print("\n1. Initial state:")
    print(f"   Agent A: {agent_a.consciousness.overall_consciousness_score():.3f}")
    print(f"   Agent B: {agent_b.consciousness.overall_consciousness_score():.3f}")

    # Test serialization
    print("\n2. Testing serialization...")
    serialized = protocol.serialize_consciousness(consciousness_a)
    print(f"   Serialized size: {len(serialized)} bytes")

    deserialized = protocol.deserialize_consciousness(serialized)
    print(f"   Deserialized score: {deserialized.overall_consciousness_score():.3f}")
    print(f"   Scores match: {deserialized.overall_consciousness_score() == consciousness_a.overall_consciousness_score()}")

    # Test transfer modes
    print("\n3. Testing transfer modes...")

    # Copy
    agent_c = Agent(agent_id="agent_c", consciousness=GradedConsciousness())
    result = protocol.transfer_consciousness(agent_a, agent_c, mode="copy")
    print(f"   Copy transfer: {result.success}")
    print(f"   Agent A score: {agent_a.consciousness.overall_consciousness_score():.3f}")
    print(f"   Agent C score: {agent_c.consciousness.overall_consciousness_score():.3f}")

    # Merge
    agent_d = Agent(agent_id="agent_d", consciousness=consciousness_b)
    agent_e = Agent(agent_id="agent_e", consciousness=consciousness_a)
    result = protocol.transfer_consciousness(agent_d, agent_e, mode="merge")
    print(f"\n   Merge transfer: {result.success}")
    print(f"   Agent D score (merged): {agent_d.consciousness.overall_consciousness_score():.3f}")
    print(f"   Agent E score (merged): {agent_e.consciousness.overall_consciousness_score():.3f}")

    # Test distributed consciousness
    print("\n4. Testing distributed consciousness...")
    original = GradedConsciousness(
        perception_fidelity=0.8,
        reaction_speed=0.7,
        memory_depth=0.9,
        memory_recall_accuracy=0.8,
        introspection_capacity=0.6,
        meta_cognitive_ability=0.7,
        information_integration=0.8,
        intentional_coherence=0.7,
        qualia_richness=0.6
    )

    shards = protocol.distributed_consciousness(original, num_shards=3)
    print(f"   Created {len(shards)} shards")

    for i, shard in enumerate(shards):
        scores = shard.get_capability_scores()
        specialized = [k for k, v in scores.items() if v > original.get_capability_scores()[k]]
        print(f"   Shard {i}: specialized in {specialized[:3]}...")

    # Recombine
    recombined = protocol.recombine_shards(shards)
    print(f"\n   Recombined score: {recombined.overall_consciousness_score():.3f}")
    print(f"   Original score:   {original.overall_consciousness_score():.3f}")

    # Test blending
    print("\n5. Testing consciousness blending...")
    blended = protocol.blend_consciousnesses([consciousness_a, consciousness_b])
    print(f"   A score: {consciousness_a.overall_consciousness_score():.3f}")
    print(f"   B score: {consciousness_b.overall_consciousness_score():.3f}")
    print(f"   Blended: {blended.overall_consciousness_score():.3f}")

    # Diff
    diff = protocol.diff_consciousnesses(consciousness_a, consciousness_b)
    print("\n   Differences (A - B):")
    for cap, d in sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
        print(f"     {cap}: {d:+.2f}")

    # Statistics
    print("\n6. Transfer statistics:")
    stats = protocol.get_transfer_statistics()
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"   {key}: {value}")
