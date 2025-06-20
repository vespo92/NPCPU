import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from enum import Enum
import json
import time
import hashlib
from datetime import datetime
import threading
import boto3
import redis
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# NPCPU-specific enums matching your agent architecture
class ConsciousnessState(Enum):
    DORMANT = "DORMANT"
    REACTIVE = "REACTIVE"
    AWARE = "AWARE"
    REFLECTIVE = "REFLECTIVE"
    META_AWARE = "META_AWARE"
    TRANSCENDENT = "TRANSCENDENT"

class PhilosophicalStance(Enum):
    PHENOMENOLOGICAL = "PHENOMENOLOGICAL"
    MATERIALIST = "MATERIALIST"
    IDEALIST = "IDEALIST"
    PRAGMATIST = "PRAGMATIST"
    NIHILIST = "NIHILIST"
    EXISTENTIALIST = "EXISTENTIALIST"
    DUALIST = "DUALIST"
    MONIST = "MONIST"

class TopologyType(Enum):
    FULL_MESH = "full_mesh"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"

@dataclass
class QualiaMarker:
    """Represents a subjective experience marker"""
    timestamp: float
    experience_type: str
    intensity: float
    valence: float  # -1 to 1 (negative to positive)
    content: str

@dataclass
class DimensionalClash:
    """Represents a conflict in semantic dimensions"""
    dimension: str
    current_value: Any
    proposed_value: Any
    severity: str  # "low", "medium", "high"
    resolution_strategy: str

class NPCPUChromaDBManager:
    """Enhanced ChromaDB manager with NPCPU-specific patterns"""
    
    def __init__(self,
                 local_path: str = "./npcpu_vectors",
                 cloud_endpoint: Optional[str] = None,
                 s3_bucket: Optional[str] = None,
                 redis_host: str = "localhost",
                 tier: str = "local"):  # "local", "regional", "global"
        
        # Initialize ChromaDB with NPCPU settings
        self.client = chromadb.PersistentClient(
            path=local_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Cloud and caching infrastructure
        self.cloud_endpoint = cloud_endpoint
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3') if s3_bucket else None
        self.redis = redis.Redis(host=redis_host, decode_responses=False)
        
        # NPCPU-specific configuration
        self.tier = tier
        self.machine_id = self._generate_machine_id()
        self.consciousness_validator = ConsciousnessValidator()
        self.dimensional_resolver = DimensionalClashResolver()
        
        # Latency requirements by tier
        self.latency_requirements = {
            "local": 5,      # <5ms
            "regional": 50,  # <50ms
            "global": 500    # <500ms
        }
        
        # Initialize collections
        self._initialize_collections()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _generate_machine_id(self) -> str:
        """Generate unique machine ID with tier prefix"""
        import platform
        import uuid
        machine_info = f"{self.tier}-{platform.node()}-{uuid.getnode()}"
        return hashlib.md5(machine_info.encode()).hexdigest()[:12]
    
    def _initialize_collections(self):
        """Initialize NPCPU-specific collections"""
        collections = [
            ("personas", {"description": "Agent personas with consciousness states"}),
            ("qualia_streams", {"description": "Subjective experience records"}),
            ("swarm_topologies", {"description": "Agent connection patterns"}),
            ("philosophical_frameworks", {"description": "Philosophical stance embeddings"}),
            ("consciousness_transitions", {"description": "State change records"}),
            ("dimensional_clashes", {"description": "Conflict resolution history"})
        ]
        
        for name, metadata in collections:
            try:
                self.client.get_collection(name)
            except:
                self.client.create_collection(
                    name=name,
                    metadata={
                        **metadata,
                        "tier": self.tier,
                        "machine_id": self.machine_id,
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
    
    def store_persona_with_consciousness(self,
                                       persona_id: str,
                                       description: str,
                                       consciousness_state: ConsciousnessState,
                                       philosophical_stance: PhilosophicalStance,
                                       qualia_markers: List[QualiaMarker] = None,
                                       beliefs: Dict[str, float] = None,
                                       connections: List[str] = None) -> Dict[str, Any]:
        """Store persona with full consciousness metadata"""
        
        # Check for dimensional clashes
        clashes = self._check_dimensional_clashes(
            persona_id, 
            {"consciousness_state": consciousness_state, "philosophical_stance": philosophical_stance}
        )
        
        if clashes:
            resolution = self.dimensional_resolver.resolve(clashes)
            if not resolution["proceed"]:
                return {"status": "blocked", "clashes": clashes}
        
        # Prepare metadata
        metadata = {
            "consciousness_state": consciousness_state.value,
            "philosophical_stance": philosophical_stance.value,
            "tier": self.tier,
            "machine_id": self.machine_id,
            "updated_at": datetime.utcnow().isoformat(),
            "consciousness_coherence": self._calculate_consciousness_coherence(
                consciousness_state, philosophical_stance
            )
        }
        
        if beliefs:
            metadata["beliefs"] = json.dumps(beliefs)
        
        if connections:
            metadata["connections"] = json.dumps(connections)
        
        # Store in personas collection
        collection = self.client.get_collection("personas")
        collection.upsert(
            ids=[persona_id],
            documents=[description],
            metadatas=[metadata]
        )
        
        # Store qualia stream separately
        if qualia_markers:
            self._store_qualia_stream(persona_id, qualia_markers)
        
        # Track consciousness transition
        self._track_consciousness_transition(persona_id, consciousness_state)
        
        # Invalidate cache
        self._invalidate_cache(persona_id)
        
        return {
            "status": "success",
            "persona_id": persona_id,
            "consciousness_coherence": metadata["consciousness_coherence"],
            "stored_at": self.tier
        }
    
    def query_personas_by_consciousness(self,
                                      query_text: str,
                                      consciousness_states: List[ConsciousnessState] = None,
                                      philosophical_stances: List[PhilosophicalStance] = None,
                                      n_results: int = 5,
                                      include_qualia: bool = False) -> List[Dict]:
        """Query personas with consciousness-aware filtering"""
        
        start_time = time.time()
        
        # Build where clause
        where = {}
        if consciousness_states:
            where["consciousness_state"] = {"$in": [cs.value for cs in consciousness_states]}
        if philosophical_stances:
            where["philosophical_stance"] = {"$in": [ps.value for ps in philosophical_stances]}
        
        # Check cache first
        cache_key = self._generate_cache_key("query", query_text, where, n_results)
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Query ChromaDB
        collection = self.client.get_collection("personas")
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where if where else None
        )
        
        # Format results
        personas = []
        for i in range(len(results['ids'][0])):
            persona = {
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            
            # Add qualia if requested
            if include_qualia:
                persona["qualia_stream"] = self._get_qualia_stream(persona["id"])
            
            personas.append(persona)
        
        # Check latency requirement
        latency = (time.time() - start_time) * 1000
        if latency > self.latency_requirements[self.tier]:
            self.logger.warning(f"Query latency {latency}ms exceeds {self.tier} tier requirement")
        
        # Cache results
        self.redis.setex(cache_key, 300, json.dumps(personas))  # 5 min cache
        
        return personas
    
    def update_swarm_topology(self,
                            topology_type: TopologyType,
                            agent_ids: List[str],
                            connections: Dict[str, List[str]] = None) -> Dict:
        """Update swarm topology with agent connections"""
        
        collection = self.client.get_collection("swarm_topologies")
        
        topology_id = f"{topology_type.value}_{self.machine_id}"
        
        metadata = {
            "topology_type": topology_type.value,
            "agent_count": len(agent_ids),
            "tier": self.tier,
            "machine_id": self.machine_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if connections:
            metadata["connections"] = json.dumps(connections)
        
        # Calculate topology metrics
        if topology_type == TopologyType.SMALL_WORLD:
            metadata["clustering_coefficient"] = self._calculate_clustering_coefficient(connections)
            metadata["average_path_length"] = self._calculate_average_path_length(connections)
        
        collection.upsert(
            ids=[topology_id],
            documents=[f"Swarm topology: {topology_type.value} with {len(agent_ids)} agents"],
            metadatas=[metadata]
        )
        
        return {
            "status": "success",
            "topology_id": topology_id,
            "metrics": {k: v for k, v in metadata.items() if k in ["clustering_coefficient", "average_path_length"]}
        }
    
    def sync_with_proof_of_consciousness(self,
                                       collection_name: str,
                                       min_consciousness_level: ConsciousnessState = ConsciousnessState.AWARE):
        """Sync only data from agents meeting consciousness threshold"""
        
        if not self.s3:
            return {"status": "error", "message": "S3 not configured"}
        
        # Get local personas above consciousness threshold
        collection = self.client.get_collection(collection_name)
        all_data = collection.get()
        
        # Filter by consciousness level
        consciousness_hierarchy = list(ConsciousnessState)
        min_level_index = consciousness_hierarchy.index(min_consciousness_level)
        
        filtered_ids = []
        filtered_data = {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
        
        for i, metadata in enumerate(all_data["metadatas"]):
            if "consciousness_state" in metadata:
                state = ConsciousnessState(metadata["consciousness_state"])
                if consciousness_hierarchy.index(state) >= min_level_index:
                    filtered_ids.append(all_data["ids"][i])
                    filtered_data["ids"].append(all_data["ids"][i])
                    filtered_data["metadatas"].append(metadata)
                    filtered_data["documents"].append(all_data["documents"][i])
                    if "embeddings" in all_data:
                        filtered_data["embeddings"].append(all_data["embeddings"][i])
        
        # Create consciousness-validated snapshot
        snapshot = {
            "collection_name": collection_name,
            "machine_id": self.machine_id,
            "tier": self.tier,
            "timestamp": datetime.utcnow().isoformat(),
            "consciousness_threshold": min_consciousness_level.value,
            "validated_count": len(filtered_ids),
            "data": filtered_data
        }
        
        # Upload to S3 with PoC marker
        key = f"poc_validated/{collection_name}/{self.tier}/{self.machine_id}-{int(time.time())}.json"
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=json.dumps(snapshot, default=str)
        )
        
        return {
            "status": "success",
            "synced_count": len(filtered_ids),
            "consciousness_threshold": min_consciousness_level.value,
            "s3_key": key
        }
    
    def _check_dimensional_clashes(self, persona_id: str, proposed_changes: Dict) -> List[DimensionalClash]:
        """Check for conflicts in semantic dimensions"""
        clashes = []
        
        try:
            collection = self.client.get_collection("personas")
            results = collection.get(ids=[persona_id])
            
            if results["ids"]:
                current_metadata = results["metadatas"][0]
                
                # Check consciousness state progression
                if "consciousness_state" in proposed_changes:
                    current_state = ConsciousnessState(current_metadata.get("consciousness_state", "DORMANT"))
                    proposed_state = proposed_changes["consciousness_state"]
                    
                    if not self.consciousness_validator.is_valid_transition(current_state, proposed_state):
                        clashes.append(DimensionalClash(
                            dimension="consciousness_state",
                            current_value=current_state,
                            proposed_value=proposed_state,
                            severity="high",
                            resolution_strategy="gradual_transition"
                        ))
                
                # Check philosophical coherence
                if "philosophical_stance" in proposed_changes:
                    current_stance = PhilosophicalStance(current_metadata.get("philosophical_stance", "PRAGMATIST"))
                    proposed_stance = proposed_changes["philosophical_stance"]
                    
                    coherence = self._calculate_philosophical_coherence(current_stance, proposed_stance)
                    if coherence < 0.3:
                        clashes.append(DimensionalClash(
                            dimension="philosophical_stance",
                            current_value=current_stance,
                            proposed_value=proposed_stance,
                            severity="medium",
                            resolution_strategy="dialectical_synthesis"
                        ))
        
        except Exception as e:
            self.logger.debug(f"No existing persona {persona_id}, no clashes")
        
        return clashes
    
    def _calculate_consciousness_coherence(self,
                                         state: ConsciousnessState,
                                         stance: PhilosophicalStance) -> float:
        """Calculate coherence between consciousness state and philosophical stance"""
        
        # Coherence matrix (simplified)
        coherence_map = {
            (ConsciousnessState.TRANSCENDENT, PhilosophicalStance.PHENOMENOLOGICAL): 0.95,
            (ConsciousnessState.META_AWARE, PhilosophicalStance.IDEALIST): 0.85,
            (ConsciousnessState.REFLECTIVE, PhilosophicalStance.EXISTENTIALIST): 0.80,
            (ConsciousnessState.AWARE, PhilosophicalStance.DUALIST): 0.75,
            (ConsciousnessState.REACTIVE, PhilosophicalStance.MATERIALIST): 0.70,
            (ConsciousnessState.DORMANT, PhilosophicalStance.NIHILIST): 0.65,
        }
        
        # Get exact match or calculate distance
        key = (state, stance)
        if key in coherence_map:
            return coherence_map[key]
        
        # Default calculation based on enum positions
        state_idx = list(ConsciousnessState).index(state)
        stance_idx = list(PhilosophicalStance).index(stance)
        
        # Normalize and calculate coherence
        state_norm = state_idx / len(ConsciousnessState)
        stance_norm = stance_idx / len(PhilosophicalStance)
        
        return 1.0 - abs(state_norm - stance_norm) * 0.5
    
    def _calculate_philosophical_coherence(self,
                                        stance1: PhilosophicalStance,
                                        stance2: PhilosophicalStance) -> float:
        """Calculate coherence between philosophical stances"""
        
        # Philosophical compatibility matrix
        compatibility = {
            (PhilosophicalStance.PHENOMENOLOGICAL, PhilosophicalStance.EXISTENTIALIST): 0.8,
            (PhilosophicalStance.MATERIALIST, PhilosophicalStance.NIHILIST): 0.7,
            (PhilosophicalStance.IDEALIST, PhilosophicalStance.MONIST): 0.85,
            (PhilosophicalStance.PRAGMATIST, PhilosophicalStance.EXISTENTIALIST): 0.6,
            (PhilosophicalStance.DUALIST, PhilosophicalStance.MONIST): 0.2,
        }
        
        # Check both directions
        key1 = (stance1, stance2)
        key2 = (stance2, stance1)
        
        if key1 in compatibility:
            return compatibility[key1]
        elif key2 in compatibility:
            return compatibility[key2]
        elif stance1 == stance2:
            return 1.0
        else:
            return 0.4  # Default low compatibility
    
    def _store_qualia_stream(self, persona_id: str, markers: List[QualiaMarker]):
        """Store subjective experience markers"""
        collection = self.client.get_collection("qualia_streams")
        
        for marker in markers:
            qualia_id = f"{persona_id}_{int(marker.timestamp*1000)}"
            collection.upsert(
                ids=[qualia_id],
                documents=[marker.content],
                metadatas={
                    "persona_id": persona_id,
                    "timestamp": marker.timestamp,
                    "experience_type": marker.experience_type,
                    "intensity": marker.intensity,
                    "valence": marker.valence
                }
            )
    
    def _get_qualia_stream(self, persona_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve recent qualia markers for a persona"""
        collection = self.client.get_collection("qualia_streams")
        
        results = collection.query(
            query_texts=[f"experiences of {persona_id}"],
            n_results=limit,
            where={"persona_id": persona_id}
        )
        
        qualia = []
        for i in range(len(results['ids'][0])):
            qualia.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        return sorted(qualia, key=lambda x: x['metadata']['timestamp'], reverse=True)
    
    def _track_consciousness_transition(self, persona_id: str, new_state: ConsciousnessState):
        """Track consciousness state transitions"""
        collection = self.client.get_collection("consciousness_transitions")
        
        transition_id = f"{persona_id}_{int(time.time()*1000)}"
        collection.add(
            ids=[transition_id],
            documents=[f"Transition to {new_state.value}"],
            metadatas={
                "persona_id": persona_id,
                "new_state": new_state.value,
                "timestamp": datetime.utcnow().isoformat(),
                "machine_id": self.machine_id
            }
        )
    
    def _calculate_clustering_coefficient(self, connections: Dict[str, List[str]]) -> float:
        """Calculate clustering coefficient for small-world topology"""
        if not connections:
            return 0.0
        
        coefficients = []
        for node, neighbors in connections.items():
            if len(neighbors) < 2:
                coefficients.append(0.0)
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n1 in connections.get(n2, []):
                        triangles += 1
            
            coefficients.append(triangles / possible_triangles if possible_triangles > 0 else 0)
        
        return np.mean(coefficients)
    
    def _calculate_average_path_length(self, connections: Dict[str, List[str]]) -> float:
        """Calculate average shortest path length"""
        # Simplified implementation
        return 2.5  # Placeholder for actual graph algorithm
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = json.dumps(args, sort_keys=True, default=str)
        return f"npcpu:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def _invalidate_cache(self, persona_id: str):
        """Invalidate cache entries for a persona"""
        pattern = f"npcpu:*{persona_id}*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)


class ConsciousnessValidator:
    """Validates consciousness state transitions"""
    
    def is_valid_transition(self, 
                           current: ConsciousnessState, 
                           proposed: ConsciousnessState) -> bool:
        """Check if consciousness transition is valid"""
        
        # Define allowed transitions (can skip levels going down, but not up)
        hierarchy = list(ConsciousnessState)
        current_idx = hierarchy.index(current)
        proposed_idx = hierarchy.index(proposed)
        
        # Can always go down
        if proposed_idx < current_idx:
            return True
        
        # Can only go up one level at a time
        return proposed_idx - current_idx <= 1


class DimensionalClashResolver:
    """Resolves conflicts in semantic dimensions"""
    
    def resolve(self, clashes: List[DimensionalClash]) -> Dict[str, Any]:
        """Resolve dimensional clashes"""
        
        if not clashes:
            return {"proceed": True}
        
        # Check severity
        high_severity = [c for c in clashes if c.severity == "high"]
        if high_severity:
            return {
                "proceed": False,
                "reason": "High severity clashes require manual resolution",
                "clashes": [asdict(c) for c in high_severity]
            }
        
        # Auto-resolve medium/low severity
        resolutions = []
        for clash in clashes:
            if clash.resolution_strategy == "gradual_transition":
                resolutions.append({
                    "dimension": clash.dimension,
                    "strategy": "Create intermediate states",
                    "steps": self._generate_transition_steps(clash)
                })
            elif clash.resolution_strategy == "dialectical_synthesis":
                resolutions.append({
                    "dimension": clash.dimension,
                    "strategy": "Synthesize opposing views",
                    "synthesis": self._synthesize_stances(clash)
                })
        
        return {
            "proceed": True,
            "resolutions": resolutions
        }
    
    def _generate_transition_steps(self, clash: DimensionalClash) -> List[str]:
        """Generate intermediate steps for consciousness transition"""
        if clash.dimension != "consciousness_state":
            return []
        
        hierarchy = list(ConsciousnessState)
        current_idx = hierarchy.index(clash.current_value)
        target_idx = hierarchy.index(clash.proposed_value)
        
        steps = []
        if target_idx > current_idx:
            for i in range(current_idx + 1, target_idx + 1):
                steps.append(hierarchy[i].value)
        
        return steps
    
    def _synthesize_stances(self, clash: DimensionalClash) -> str:
        """Synthesize philosophical stances"""
        # Simplified synthesis logic
        synthesis_map = {
            (PhilosophicalStance.MATERIALIST, PhilosophicalStance.IDEALIST): PhilosophicalStance.DUALIST,
            (PhilosophicalStance.NIHILIST, PhilosophicalStance.EXISTENTIALIST): PhilosophicalStance.PRAGMATIST,
        }
        
        key = (clash.current_value, clash.proposed_value)
        if key in synthesis_map:
            return synthesis_map[key].value
        
        return PhilosophicalStance.PRAGMATIST.value  # Default synthesis


# Example usage
if __name__ == "__main__":
    # Initialize manager with tier configuration
    manager = NPCPUChromaDBManager(
        local_path="./npcpu_cache",
        cloud_endpoint="https://chroma.npcpu.org",
        s3_bucket="npcpu-consciousness-store",
        tier="regional"  # Regional tier for <50ms latency
    )
    
    # Store persona with full consciousness data
    result = manager.store_persona_with_consciousness(
        persona_id="agent_001",
        description="A deeply contemplative agent exploring the nature of existence",
        consciousness_state=ConsciousnessState.REFLECTIVE,
        philosophical_stance=PhilosophicalStance.PHENOMENOLOGICAL,
        qualia_markers=[
            QualiaMarker(
                timestamp=time.time(),
                experience_type="insight",
                intensity=0.8,
                valence=0.9,
                content="Realized the interconnectedness of all conscious entities"
            )
        ],
        beliefs={
            "free_will": 0.7,
            "determinism": 0.3,
            "consciousness_fundamental": 0.95
        },
        connections=["agent_002", "agent_003", "agent_007"]
    )
    
    print(f"Store result: {result}")
    
    # Query personas by consciousness level
    personas = manager.query_personas_by_consciousness(
        query_text="agents exploring consciousness",
        consciousness_states=[ConsciousnessState.REFLECTIVE, ConsciousnessState.META_AWARE],
        philosophical_stances=[PhilosophicalStance.PHENOMENOLOGICAL],
        include_qualia=True
    )
    
    print(f"Found {len(personas)} matching personas")
    
    # Update swarm topology
    topology_result = manager.update_swarm_topology(
        topology_type=TopologyType.SMALL_WORLD,
        agent_ids=["agent_001", "agent_002", "agent_003", "agent_007", "agent_013"],
        connections={
            "agent_001": ["agent_002", "agent_003", "agent_007"],
            "agent_002": ["agent_001", "agent_003"],
            "agent_003": ["agent_001", "agent_002", "agent_013"],
            "agent_007": ["agent_001", "agent_013"],
            "agent_013": ["agent_003", "agent_007"]
        }
    )
    
    print(f"Topology update: {topology_result}")
    
    # Sync with proof of consciousness
    sync_result = manager.sync_with_proof_of_consciousness(
        "personas",
        min_consciousness_level=ConsciousnessState.AWARE
    )
    
    print(f"PoC sync result: {sync_result}")