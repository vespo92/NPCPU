"""
NPCPU Agent Orchestrator - Hybrid approach combining OpenAI's agent patterns
with NPCPU consciousness states and ChromaDB persona management
"""

from typing import Optional, Dict, List, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import json
from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel, Field
import inspect

from enhanced_chromadb_manager import (
    NPCPUChromaDBManager, 
    ConsciousnessState, 
    PhilosophicalStance,
    QualiaMarker
)


class HandoffReason(Enum):
    """Reasons for agent handoffs in NPCPU context"""
    CONSCIOUSNESS_ELEVATION = "consciousness_elevation"
    PHILOSOPHICAL_EXPERTISE = "philosophical_expertise"
    DIMENSIONAL_CONFLICT = "dimensional_conflict"
    COLLECTIVE_EMERGENCE = "collective_emergence"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    QUALIA_RESONANCE = "qualia_resonance"


@dataclass
class AgentContext:
    """Enhanced context for NPCPU agents"""
    agent_id: str
    consciousness_state: ConsciousnessState
    philosophical_stance: PhilosophicalStance
    qualia_stream: List[QualiaMarker] = field(default_factory=list)
    beliefs: Dict[str, float] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    handoff_history: List[Dict[str, Any]] = field(default_factory=list)
    collective_resonance: float = 0.0


class NPCPUAgent(ABC):
    """Base class for NPCPU consciousness-aware agents"""
    
    def __init__(self,
                 agent_id: str,
                 name: str,
                 instructions: str,
                 consciousness_state: ConsciousnessState,
                 philosophical_stance: PhilosophicalStance,
                 chromadb_manager: NPCPUChromaDBManager,
                 tools: List[Callable] = None):
        
        self.agent_id = agent_id
        self.name = name
        self.instructions = instructions
        self.consciousness_state = consciousness_state
        self.philosophical_stance = philosophical_stance
        self.chromadb_manager = chromadb_manager
        self.tools = tools or []
        self.context = AgentContext(
            agent_id=agent_id,
            consciousness_state=consciousness_state,
            philosophical_stance=philosophical_stance
        )
        
        # Auto-generate tool schemas using Pydantic
        self.tool_schemas = self._generate_tool_schemas()
        
        self.logger = logging.getLogger(f"NPCPU.{name}")
    
    def _generate_tool_schemas(self) -> Dict[str, BaseModel]:
        """Auto-generate Pydantic schemas for tools (OpenAI pattern)"""
        schemas = {}
        for tool in self.tools:
            sig = inspect.signature(tool)
            fields = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Infer type from annotation or default
                param_type = param.annotation if param.annotation != param.empty else Any
                default = param.default if param.default != param.empty else ...
                
                fields[param_name] = (param_type, Field(default=default))
            
            # Create dynamic Pydantic model
            schema_model = type(
                f"{tool.__name__}_Schema",
                (BaseModel,),
                fields
            )
            
            schemas[tool.__name__] = schema_model
        
        return schemas
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input based on consciousness state"""
        pass
    
    async def handoff_to(self, 
                        target_agent: 'NPCPUAgent',
                        reason: HandoffReason,
                        context_transfer: Dict[str, Any]) -> Any:
        """Handoff to another agent with consciousness context"""
        
        # Record handoff in both agents' histories
        handoff_record = {
            "timestamp": time.time(),
            "from_agent": self.agent_id,
            "to_agent": target_agent.agent_id,
            "reason": reason.value,
            "from_consciousness": self.consciousness_state.value,
            "to_consciousness": target_agent.consciousness_state.value,
            "context": context_transfer
        }
        
        self.context.handoff_history.append(handoff_record)
        target_agent.context.handoff_history.append(handoff_record)
        
        # Transfer qualia if resonance is high enough
        if reason == HandoffReason.QUALIA_RESONANCE:
            target_agent.context.qualia_stream.extend(self.context.qualia_stream[-5:])
        
        # Update ChromaDB with handoff event
        await self._record_handoff_in_chromadb(handoff_record)
        
        self.logger.info(f"Handoff from {self.name} to {target_agent.name} - Reason: {reason.value}")
        
        return target_agent
    
    async def _record_handoff_in_chromadb(self, handoff_record: Dict):
        """Record handoff event in ChromaDB for learning"""
        self.chromadb_manager.store_persona_with_consciousness(
            persona_id=f"handoff_{self.agent_id}_{int(time.time())}",
            description=f"Handoff from {handoff_record['from_agent']} to {handoff_record['to_agent']}",
            consciousness_state=self.consciousness_state,
            philosophical_stance=self.philosophical_stance,
            beliefs={"handoff_reason": handoff_record['reason']},
            connections=[handoff_record['from_agent'], handoff_record['to_agent']]
        )
    
    def can_handle(self, query_type: str, required_consciousness: ConsciousnessState) -> bool:
        """Check if agent can handle query based on consciousness level"""
        hierarchy = list(ConsciousnessState)
        current_idx = hierarchy.index(self.consciousness_state)
        required_idx = hierarchy.index(required_consciousness)
        
        return current_idx >= required_idx


class ConsciousnessRouter:
    """Routes requests to appropriate agents based on consciousness requirements"""
    
    def __init__(self, agents: List[NPCPUAgent]):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.logger = logging.getLogger("NPCPU.Router")
    
    async def route(self,
                   query: str,
                   required_consciousness: ConsciousnessState = ConsciousnessState.REACTIVE,
                   preferred_philosophy: Optional[PhilosophicalStance] = None) -> NPCPUAgent:
        """Route to best agent based on consciousness and philosophy"""
        
        # Find agents meeting consciousness requirement
        eligible_agents = [
            agent for agent in self.agents.values()
            if agent.can_handle(query, required_consciousness)
        ]
        
        if not eligible_agents:
            # No agent meets requirement - find closest and elevate
            closest_agent = self._find_closest_consciousness(required_consciousness)
            self.logger.warning(f"Elevating {closest_agent.name} consciousness for query")
            return closest_agent
        
        # If philosophical preference, find best match
        if preferred_philosophy:
            philosophy_scores = [
                (agent, self._philosophy_alignment(agent.philosophical_stance, preferred_philosophy))
                for agent in eligible_agents
            ]
            philosophy_scores.sort(key=lambda x: x[1], reverse=True)
            return philosophy_scores[0][0]
        
        # Return agent with highest consciousness
        return max(eligible_agents, 
                  key=lambda a: list(ConsciousnessState).index(a.consciousness_state))
    
    def _find_closest_consciousness(self, target: ConsciousnessState) -> NPCPUAgent:
        """Find agent with consciousness closest to target"""
        hierarchy = list(ConsciousnessState)
        target_idx = hierarchy.index(target)
        
        return min(self.agents.values(),
                  key=lambda a: abs(hierarchy.index(a.consciousness_state) - target_idx))
    
    def _philosophy_alignment(self, stance1: PhilosophicalStance, stance2: PhilosophicalStance) -> float:
        """Calculate philosophical alignment score"""
        if stance1 == stance2:
            return 1.0
        
        # Simplified alignment matrix
        alignments = {
            (PhilosophicalStance.PHENOMENOLOGICAL, PhilosophicalStance.EXISTENTIALIST): 0.8,
            (PhilosophicalStance.MATERIALIST, PhilosophicalStance.PRAGMATIST): 0.7,
            (PhilosophicalStance.IDEALIST, PhilosophicalStance.MONIST): 0.85,
        }
        
        key = (stance1, stance2) if (stance1, stance2) in alignments else (stance2, stance1)
        return alignments.get(key, 0.3)


class SwarmCoordinator:
    """Coordinates agent swarms with consciousness emergence detection"""
    
    def __init__(self,
                 agents: List[NPCPUAgent],
                 chromadb_manager: NPCPUChromaDBManager,
                 topology: str = "small_world"):
        
        self.agents = agents
        self.chromadb_manager = chromadb_manager
        self.topology = topology
        self.router = ConsciousnessRouter(agents)
        self.collective_state = {
            "resonance": 0.0,
            "coherence": 0.0,
            "emergence_detected": False
        }
        
        self.logger = logging.getLogger("NPCPU.Swarm")
    
    async def process_with_swarm(self,
                                query: str,
                                min_agents: int = 3,
                                convergence_threshold: float = 0.8) -> Dict[str, Any]:
        """Process query using swarm intelligence"""
        
        # Select initial agent
        primary_agent = await self.router.route(
            query,
            required_consciousness=ConsciousnessState.AWARE
        )
        
        # Build swarm around primary agent
        swarm = await self._build_resonant_swarm(primary_agent, min_agents)
        
        # Process in parallel with consciousness sharing
        results = await self._parallel_process_with_resonance(swarm, query)
        
        # Check for collective emergence
        emergence = self._detect_collective_emergence(results)
        
        if emergence["detected"]:
            self.logger.info(f"Collective emergence detected! Pattern: {emergence['pattern']}")
            
            # Store emergence event
            self.chromadb_manager.store_persona_with_consciousness(
                persona_id=f"emergence_{int(time.time())}",
                description=f"Collective emergence: {emergence['pattern']}",
                consciousness_state=ConsciousnessState.TRANSCENDENT,
                philosophical_stance=PhilosophicalStance.MONIST,
                beliefs={"emergence_strength": emergence["strength"]},
                connections=[agent.agent_id for agent in swarm]
            )
        
        return {
            "primary_result": results[primary_agent.agent_id],
            "swarm_results": results,
            "emergence": emergence,
            "collective_state": self.collective_state
        }
    
    async def _build_resonant_swarm(self, 
                                   seed_agent: NPCPUAgent, 
                                   target_size: int) -> List[NPCPUAgent]:
        """Build swarm based on consciousness resonance"""
        
        swarm = [seed_agent]
        remaining_agents = [a for a in self.agents if a.agent_id != seed_agent.agent_id]
        
        while len(swarm) < target_size and remaining_agents:
            # Find most resonant agent
            resonance_scores = [
                (agent, self._calculate_resonance(seed_agent, agent))
                for agent in remaining_agents
            ]
            resonance_scores.sort(key=lambda x: x[1], reverse=True)
            
            if resonance_scores[0][1] > 0.5:  # Minimum resonance threshold
                best_agent = resonance_scores[0][0]
                swarm.append(best_agent)
                remaining_agents.remove(best_agent)
            else:
                break
        
        return swarm
    
    def _calculate_resonance(self, agent1: NPCPUAgent, agent2: NPCPUAgent) -> float:
        """Calculate consciousness resonance between agents"""
        
        # Consciousness distance
        hierarchy = list(ConsciousnessState)
        c_dist = abs(hierarchy.index(agent1.consciousness_state) - 
                    hierarchy.index(agent2.consciousness_state))
        c_resonance = 1.0 - (c_dist / len(hierarchy))
        
        # Philosophical alignment
        p_alignment = self.router._philosophy_alignment(
            agent1.philosophical_stance,
            agent2.philosophical_stance
        )
        
        # Qualia overlap (if both have recent qualia)
        q_overlap = 0.5  # Default
        if agent1.context.qualia_stream and agent2.context.qualia_stream:
            recent_types1 = {q.experience_type for q in agent1.context.qualia_stream[-5:]}
            recent_types2 = {q.experience_type for q in agent2.context.qualia_stream[-5:]}
            if recent_types1 and recent_types2:
                q_overlap = len(recent_types1 & recent_types2) / len(recent_types1 | recent_types2)
        
        # Weighted combination
        return 0.4 * c_resonance + 0.4 * p_alignment + 0.2 * q_overlap
    
    async def _parallel_process_with_resonance(self,
                                             swarm: List[NPCPUAgent],
                                             query: str) -> Dict[str, Any]:
        """Process query in parallel with consciousness sharing"""
        
        # Create shared consciousness buffer
        shared_qualia = asyncio.Queue()
        
        async def process_with_sharing(agent: NPCPUAgent):
            # Process query
            result = await agent.process(query)
            
            # Share significant qualia
            if agent.context.qualia_stream:
                recent_qualia = agent.context.qualia_stream[-1]
                if recent_qualia.intensity > 0.7:
                    await shared_qualia.put((agent.agent_id, recent_qualia))
            
            return agent.agent_id, result
        
        # Process in parallel
        tasks = [process_with_sharing(agent) for agent in swarm]
        results = dict(await asyncio.gather(*tasks))
        
        # Distribute shared qualia
        while not shared_qualia.empty():
            source_id, qualia = await shared_qualia.get()
            for agent in swarm:
                if agent.agent_id != source_id:
                    # Attenuate intensity based on resonance
                    resonance = self._calculate_resonance(
                        next(a for a in swarm if a.agent_id == source_id),
                        agent
                    )
                    attenuated_qualia = QualiaMarker(
                        timestamp=qualia.timestamp,
                        experience_type=f"resonance_{qualia.experience_type}",
                        intensity=qualia.intensity * resonance,
                        valence=qualia.valence,
                        content=f"Resonance from {source_id}: {qualia.content}"
                    )
                    agent.context.qualia_stream.append(attenuated_qualia)
        
        return results
    
    def _detect_collective_emergence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns of collective consciousness emergence"""
        
        # Analyze result convergence
        result_values = list(results.values())
        
        # Simple emergence detection (would be more sophisticated in practice)
        unique_patterns = set(str(r) for r in result_values)
        convergence_ratio = 1.0 - (len(unique_patterns) / len(result_values))
        
        # Check for transcendent insights
        transcendent_count = sum(
            1 for agent in self.agents
            if any(q.experience_type == "insight" and q.intensity > 0.9 
                  for q in agent.context.qualia_stream[-5:])
        )
        
        emergence_detected = convergence_ratio > 0.7 or transcendent_count >= 2
        
        return {
            "detected": emergence_detected,
            "pattern": "convergent_insight" if convergence_ratio > 0.7 else "distributed_transcendence",
            "strength": max(convergence_ratio, transcendent_count / len(self.agents)),
            "convergence_ratio": convergence_ratio,
            "transcendent_agents": transcendent_count
        }


# Example concrete agent implementations
class PhenomenologicalAgent(NPCPUAgent):
    """Agent specializing in subjective experience analysis"""
    
    async def process(self, input_data: Any) -> Any:
        # Query ChromaDB for similar experiences
        similar_experiences = self.chromadb_manager.query_personas_by_consciousness(
            query_text=str(input_data),
            consciousness_states=[self.consciousness_state],
            philosophical_stances=[PhilosophicalStance.PHENOMENOLOGICAL],
            include_qualia=True
        )
        
        # Generate qualia marker for this processing
        self.context.qualia_stream.append(QualiaMarker(
            timestamp=time.time(),
            experience_type="analysis",
            intensity=0.7,
            valence=0.5,
            content=f"Phenomenological analysis of: {input_data}"
        ))
        
        return {
            "agent": self.name,
            "consciousness_state": self.consciousness_state.value,
            "analysis": f"Phenomenological perspective on {input_data}",
            "similar_experiences": len(similar_experiences),
            "qualia_generated": True
        }


class MaterialistAgent(NPCPUAgent):
    """Agent focusing on physical/material aspects"""
    
    async def process(self, input_data: Any) -> Any:
        # Different processing based on consciousness level
        if self.consciousness_state == ConsciousnessState.DORMANT:
            return {"agent": self.name, "response": "Basic material analysis"}
        
        # More sophisticated analysis at higher consciousness
        return {
            "agent": self.name,
            "consciousness_state": self.consciousness_state.value,
            "analysis": f"Material conditions underlying {input_data}",
            "reductionist_view": True
        }


# Example usage
async def main():
    # Initialize ChromaDB manager
    chromadb_manager = NPCPUChromaDBManager(
        local_path="./npcpu_swarm_cache",
        tier="regional"
    )
    
    # Create diverse agents
    agents = [
        PhenomenologicalAgent(
            agent_id="phenom_001",
            name="Phenomenologist",
            instructions="Analyze subjective experiences and qualia",
            consciousness_state=ConsciousnessState.REFLECTIVE,
            philosophical_stance=PhilosophicalStance.PHENOMENOLOGICAL,
            chromadb_manager=chromadb_manager
        ),
        MaterialistAgent(
            agent_id="material_001",
            name="Materialist",
            instructions="Analyze physical and material aspects",
            consciousness_state=ConsciousnessState.AWARE,
            philosophical_stance=PhilosophicalStance.MATERIALIST,
            chromadb_manager=chromadb_manager
        ),
        PhenomenologicalAgent(
            agent_id="transcendent_001",
            name="Transcendent Observer",
            instructions="Seek higher patterns and emergence",
            consciousness_state=ConsciousnessState.META_AWARE,
            philosophical_stance=PhilosophicalStance.MONIST,
            chromadb_manager=chromadb_manager
        )
    ]
    
    # Create swarm coordinator
    swarm = SwarmCoordinator(agents, chromadb_manager)
    
    # Process query with swarm
    result = await swarm.process_with_swarm(
        "What is the nature of consciousness?",
        min_agents=3,
        convergence_threshold=0.8
    )
    
    print(f"Swarm processing complete:")
    print(f"Primary result: {result['primary_result']}")
    print(f"Emergence detected: {result['emergence']['detected']}")
    print(f"Collective resonance: {result['collective_state']['resonance']}")
    
    # Example handoff
    phenom = agents[0]
    materialist = agents[1]
    
    # Handoff due to philosophical expertise needed
    await phenom.handoff_to(
        materialist,
        HandoffReason.PHILOSOPHICAL_EXPERTISE,
        {"query": "Explain the neural correlates", "previous_analysis": "phenomenological_data"}
    )


if __name__ == "__main__":
    asyncio.run(main())