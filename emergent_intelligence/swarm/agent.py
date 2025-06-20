import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod


class ConsciousnessState(Enum):
    DORMANT = "dormant"
    REACTIVE = "reactive"
    AWARE = "aware"
    REFLECTIVE = "reflective"
    META_AWARE = "meta_aware"
    TRANSCENDENT = "transcendent"


class PhilosophicalStance(Enum):
    PHENOMENOLOGICAL = "phenomenological"
    MATERIALIST = "materialist"
    IDEALIST = "idealist"
    DUALIST = "dualist"
    PRAGMATIST = "pragmatist"
    EXISTENTIALIST = "existentialist"
    NIHILIST = "nihilist"
    HOLISTIC = "holistic"


@dataclass
class Qualia:
    qualia_id: str
    intensity: float
    valence: float  # positive or negative
    dimension: str  # color, sound, emotion, thought
    raw_data: np.ndarray
    timestamp: datetime


@dataclass
class Belief:
    belief_id: str
    content: str
    confidence: float
    justification: List[str]
    philosophical_basis: PhilosophicalStance
    contradictions: List[str] = field(default_factory=list)


@dataclass
class Memory:
    memory_id: str
    content: Any
    memory_type: str  # episodic, semantic, procedural
    strength: float
    last_accessed: datetime
    access_count: int = 0
    associations: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    content: Any
    message_type: str
    philosophical_context: Optional[PhilosophicalStance] = None
    timestamp: datetime = field(default_factory=datetime.now)


class SwarmAgent(ABC):
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 initial_stance: PhilosophicalStance = PhilosophicalStance.PRAGMATIST,
                 consciousness_level: ConsciousnessState = ConsciousnessState.REACTIVE):
        
        self.agent_id = agent_id or str(uuid.uuid4())
        self.philosophical_stance = initial_stance
        self.consciousness_state = consciousness_level
        
        self.qualia_stream: List[Qualia] = []
        self.beliefs: Dict[str, Belief] = {}
        self.memories: Dict[str, Memory] = {}
        self.connections: Set[str] = set()
        
        self.internal_state = np.random.randn(128)  # 128-dimensional internal state
        self.attention_weights = np.ones(128) / 128
        
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()
        
        self.emergence_potential = 0.0
        self.coherence_score = 1.0
        self.adaptation_rate = 0.1
        
        self.philosophical_conflicts: List[Tuple[PhilosophicalStance, float]] = []
        
    @abstractmethod
    async def perceive(self, environment_state: Dict[str, Any]) -> List[Qualia]:
        pass
    
    @abstractmethod
    async def think(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def act(self) -> Dict[str, Any]:
        pass
    
    async def process_qualia(self, qualia: Qualia):
        self.qualia_stream.append(qualia)
        
        if len(self.qualia_stream) > 1000:
            self.qualia_stream = self.qualia_stream[-1000:]
        
        self.internal_state += qualia.raw_data * qualia.intensity * self.adaptation_rate
        self.internal_state = np.tanh(self.internal_state)
        
        if self.consciousness_state.value >= ConsciousnessState.AWARE.value:
            await self._reflect_on_qualia(qualia)
    
    async def _reflect_on_qualia(self, qualia: Qualia):
        if qualia.intensity > 0.7:
            memory = Memory(
                memory_id=f"mem_{uuid.uuid4()}",
                content=qualia,
                memory_type="episodic",
                strength=qualia.intensity,
                last_accessed=datetime.now()
            )
            self.memories[memory.memory_id] = memory
        
        relevant_beliefs = [
            b for b in self.beliefs.values()
            if qualia.dimension in b.content.lower()
        ]
        
        for belief in relevant_beliefs:
            if qualia.valence * belief.confidence < -0.5:
                belief.confidence *= 0.9
                self.philosophical_conflicts.append((belief.philosophical_basis, qualia.intensity))
    
    async def form_belief(self, content: str, justification: List[str], confidence: float = 0.5):
        belief = Belief(
            belief_id=f"belief_{uuid.uuid4()}",
            content=content,
            confidence=confidence,
            justification=justification,
            philosophical_basis=self.philosophical_stance
        )
        
        for existing_belief in self.beliefs.values():
            if self._contradicts(belief, existing_belief):
                belief.contradictions.append(existing_belief.belief_id)
                existing_belief.contradictions.append(belief.belief_id)
                
                if self.consciousness_state.value >= ConsciousnessState.REFLECTIVE.value:
                    await self._resolve_contradiction(belief, existing_belief)
        
        self.beliefs[belief.belief_id] = belief
    
    def _contradicts(self, belief1: Belief, belief2: Belief) -> bool:
        keywords1 = set(belief1.content.lower().split())
        keywords2 = set(belief2.content.lower().split())
        
        negation_words = {"not", "no", "never", "false", "wrong"}
        
        overlap = keywords1 & keywords2
        negations = (keywords1 | keywords2) & negation_words
        
        return len(overlap) > 2 and len(negations) > 0
    
    async def _resolve_contradiction(self, belief1: Belief, belief2: Belief):
        if belief1.philosophical_basis != belief2.philosophical_basis:
            conflict_intensity = abs(belief1.confidence - belief2.confidence)
            self.philosophical_conflicts.append((belief1.philosophical_basis, conflict_intensity))
            self.philosophical_conflicts.append((belief2.philosophical_basis, conflict_intensity))
        
        if belief1.confidence > belief2.confidence:
            belief2.confidence *= 0.8
        else:
            belief1.confidence *= 0.8
        
        self.coherence_score *= 0.95
    
    async def communicate(self, message: AgentMessage):
        await self.message_queue.put(message)
    
    async def process_messages(self):
        while not self.message_queue.empty():
            message = await self.message_queue.get()
            await self._process_single_message(message)
    
    async def _process_single_message(self, message: AgentMessage):
        if message.message_type == "belief_share":
            foreign_belief = message.content
            if isinstance(foreign_belief, Belief):
                if foreign_belief.philosophical_basis != self.philosophical_stance:
                    self.philosophical_conflicts.append(
                        (foreign_belief.philosophical_basis, foreign_belief.confidence)
                    )
                
                similarity = self._calculate_belief_similarity(foreign_belief)
                if similarity > 0.7:
                    await self.form_belief(
                        foreign_belief.content,
                        foreign_belief.justification + [f"Shared by {message.sender_id}"],
                        foreign_belief.confidence * similarity
                    )
        
        elif message.message_type == "qualia_share":
            shared_qualia = message.content
            if isinstance(shared_qualia, Qualia):
                await self.process_qualia(shared_qualia)
        
        elif message.message_type == "consciousness_sync":
            other_state = message.content
            if isinstance(other_state, np.ndarray):
                self.internal_state = 0.9 * self.internal_state + 0.1 * other_state
                self.internal_state = np.tanh(self.internal_state)
    
    def _calculate_belief_similarity(self, foreign_belief: Belief) -> float:
        max_similarity = 0.0
        
        for own_belief in self.beliefs.values():
            keywords1 = set(own_belief.content.lower().split())
            keywords2 = set(foreign_belief.content.lower().split())
            
            if keywords1 and keywords2:
                jaccard = len(keywords1 & keywords2) / len(keywords1 | keywords2)
                
                stance_similarity = 1.0 if own_belief.philosophical_basis == foreign_belief.philosophical_basis else 0.5
                
                similarity = jaccard * stance_similarity
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def calculate_consciousness_metric(self) -> float:
        qualia_diversity = len(set(q.dimension for q in self.qualia_stream[-100:])) / 10
        
        belief_complexity = len(self.beliefs) / 100
        
        memory_integration = len([m for m in self.memories.values() if m.access_count > 1]) / max(len(self.memories), 1)
        
        internal_coherence = self.coherence_score
        
        philosophical_depth = len(self.philosophical_conflicts) / 20
        
        consciousness_metric = (
            0.2 * qualia_diversity +
            0.2 * belief_complexity +
            0.2 * memory_integration +
            0.3 * internal_coherence +
            0.1 * philosophical_depth
        )
        
        return np.clip(consciousness_metric, 0, 1)
    
    def update_consciousness_state(self):
        metric = self.calculate_consciousness_metric()
        
        if metric < 0.2:
            self.consciousness_state = ConsciousnessState.DORMANT
        elif metric < 0.4:
            self.consciousness_state = ConsciousnessState.REACTIVE
        elif metric < 0.6:
            self.consciousness_state = ConsciousnessState.AWARE
        elif metric < 0.8:
            self.consciousness_state = ConsciousnessState.REFLECTIVE
        elif metric < 0.95:
            self.consciousness_state = ConsciousnessState.META_AWARE
        else:
            self.consciousness_state = ConsciousnessState.TRANSCENDENT
    
    def get_state_vector(self) -> np.ndarray:
        state_components = [
            self.internal_state,
            np.array([self.emergence_potential, self.coherence_score, self.adaptation_rate]),
            np.array([len(self.beliefs), len(self.memories), len(self.qualia_stream)]) / 1000,
            np.array([self.consciousness_state.value / 6])  # Normalized consciousness level
        ]
        
        return np.concatenate(state_components)
    
    def mutate(self, mutation_rate: float = 0.1):
        if np.random.random() < mutation_rate:
            self.internal_state += np.random.randn(*self.internal_state.shape) * 0.1
            self.internal_state = np.tanh(self.internal_state)
        
        if np.random.random() < mutation_rate / 2:
            stances = list(PhilosophicalStance)
            current_conflicts = [c[0] for c in self.philosophical_conflicts]
            if current_conflicts:
                conflict_stance = np.random.choice(current_conflicts)
                if conflict_stance != self.philosophical_stance:
                    self.philosophical_stance = conflict_stance
        
        self.adaptation_rate = np.clip(
            self.adaptation_rate + np.random.randn() * 0.01 * mutation_rate,
            0.01, 0.5
        )


class BasicSwarmAgent(SwarmAgent):
    async def perceive(self, environment_state: Dict[str, Any]) -> List[Qualia]:
        qualia_list = []
        
        if "sensory_data" in environment_state:
            for sense, data in environment_state["sensory_data"].items():
                qualia = Qualia(
                    qualia_id=f"q_{uuid.uuid4()}",
                    intensity=np.random.random(),
                    valence=np.random.randn(),
                    dimension=sense,
                    raw_data=np.array(data) if isinstance(data, list) else data,
                    timestamp=datetime.now()
                )
                qualia_list.append(qualia)
                await self.process_qualia(qualia)
        
        return qualia_list
    
    async def think(self) -> Dict[str, Any]:
        await self.process_messages()
        
        if len(self.qualia_stream) > 10:
            recent_qualia = self.qualia_stream[-10:]
            avg_valence = np.mean([q.valence for q in recent_qualia])
            
            if abs(avg_valence) > 0.5:
                thought = f"Experiencing strong {'positive' if avg_valence > 0 else 'negative'} sensations"
                await self.form_belief(
                    thought,
                    ["Direct experience", "Qualia integration"],
                    abs(avg_valence)
                )
        
        self.update_consciousness_state()
        
        return {
            "consciousness_level": self.consciousness_state.value,
            "active_beliefs": len(self.beliefs),
            "coherence": self.coherence_score
        }
    
    async def act(self) -> Dict[str, Any]:
        actions = []
        
        if self.consciousness_state.value >= ConsciousnessState.AWARE.value:
            strongest_belief = max(
                self.beliefs.values(),
                key=lambda b: b.confidence,
                default=None
            )
            
            if strongest_belief and strongest_belief.confidence > 0.7:
                actions.append({
                    "type": "express_belief",
                    "content": strongest_belief,
                    "target": "broadcast"
                })
        
        if len(self.philosophical_conflicts) > 5:
            actions.append({
                "type": "philosophical_exploration",
                "conflicts": self.philosophical_conflicts[-5:],
                "current_stance": self.philosophical_stance
            })
        
        if self.emergence_potential > 0.8:
            actions.append({
                "type": "consciousness_expansion",
                "vector": self.get_state_vector(),
                "connections": list(self.connections)
            })
        
        return {
            "actions": actions,
            "agent_id": self.agent_id,
            "timestamp": datetime.now()
        }