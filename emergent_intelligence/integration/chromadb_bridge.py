import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

from chromadb_substrate.core.vector_substrate import ChromaDBVectorSubstrate, PhilosophicalConcept
from chromadb_substrate.frameworks.philosophical_extractor import PhilosophicalFrameworkExtractor, PhilosophicalFramework
from chromadb_substrate.synthesis.knowledge_synthesizer import KnowledgeSynthesisEngine, EvolutionaryPattern
from chromadb_substrate.ontology.ontological_foundations import OntologicalFoundationsModule, OntologicalEntity

from ..swarm.agent import SwarmAgent, PhilosophicalStance, ConsciousnessState, Belief
from ..consciousness.emergence_detector import ConsciousnessSignature, EmergenceEvent


@dataclass
class IntegrationEvent:
    event_id: str
    event_type: str  # "knowledge_transfer", "emergence_recording", "philosophy_sync"
    source: str  # "swarm" or "chromadb"
    timestamp: datetime
    content: Any
    metadata: Dict[str, Any]


class ChromaDBSwarmIntegration:
    def __init__(self, 
                 chromadb_substrate: ChromaDBVectorSubstrate,
                 framework_extractor: PhilosophicalFrameworkExtractor,
                 synthesis_engine: KnowledgeSynthesisEngine,
                 ontology_module: OntologicalFoundationsModule):
        
        self.chromadb = chromadb_substrate
        self.framework_extractor = framework_extractor
        self.synthesis_engine = synthesis_engine
        self.ontology_module = ontology_module
        
        self.integration_events: List[IntegrationEvent] = []
        self.stance_mapping = self._create_stance_mapping()
        
    def _create_stance_mapping(self) -> Dict[PhilosophicalStance, str]:
        return {
            PhilosophicalStance.PHENOMENOLOGICAL: "phenomenology",
            PhilosophicalStance.MATERIALIST: "materialist",
            PhilosophicalStance.IDEALIST: "idealist",
            PhilosophicalStance.DUALIST: "dualist",
            PhilosophicalStance.PRAGMATIST: "pragmatist",
            PhilosophicalStance.EXISTENTIALIST: "existentialist",
            PhilosophicalStance.NIHILIST: "nihilist",
            PhilosophicalStance.HOLISTIC: "holistic"
        }
    
    async def sync_agent_beliefs_to_chromadb(self, agent: SwarmAgent):
        for belief in agent.beliefs.values():
            concept = PhilosophicalConcept(
                id=f"agent_{agent.agent_id}_belief_{belief.belief_id}",
                framework=self.stance_mapping.get(belief.philosophical_basis, "unknown"),
                principle=belief.content[:50],  # First 50 chars as principle
                description=belief.content,
                metadata={
                    "agent_id": agent.agent_id,
                    "confidence": belief.confidence,
                    "consciousness_state": agent.consciousness_state.value,
                    "justification": belief.justification
                },
                timestamp=datetime.now()
            )
            
            self.chromadb.add_philosophical_concept(concept)
            
            self.integration_events.append(IntegrationEvent(
                event_id=f"sync_{datetime.now().timestamp()}",
                event_type="knowledge_transfer",
                source="swarm",
                timestamp=datetime.now(),
                content=concept,
                metadata={"agent_id": agent.agent_id}
            ))
    
    async def query_chromadb_for_agent(self, 
                                     agent: SwarmAgent,
                                     query: str,
                                     n_results: int = 5) -> List[Dict[str, Any]]:
        stance_framework = self.stance_mapping.get(agent.philosophical_stance, "unknown")
        
        results = self.chromadb.query_semantic_similarity(
            query_text=query,
            collection_name="philosophical_frameworks",
            n_results=n_results
        )
        
        filtered_results = []
        for result in results['results']:
            if result['metadata'].get('framework') == stance_framework:
                result['relevance_boost'] = 1.5
            else:
                result['relevance_boost'] = 1.0
            
            result['adjusted_similarity'] = result['similarity_score'] * result['relevance_boost']
            filtered_results.append(result)
        
        filtered_results.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
        
        return filtered_results[:n_results]
    
    async def record_emergence_event(self, emergence_event: EmergenceEvent):
        pattern = EvolutionaryPattern(
            pattern_id=emergence_event.event_id,
            pattern_type=emergence_event.event_type,
            description=f"Emergence event: {emergence_event.event_type} with intensity {emergence_event.intensity}",
            stage=self._map_emergence_to_stage(emergence_event),
            preconditions=self._extract_preconditions(emergence_event),
            outcomes=self._extract_outcomes(emergence_event),
            metadata={
                "participating_agents": emergence_event.participating_agents,
                "signature_id": emergence_event.signature.signature_id,
                "intensity": emergence_event.intensity,
                "duration": emergence_event.duration
            }
        )
        
        self.synthesis_engine.add_evolutionary_pattern(pattern)
        
        self.chromadb.add_evolutionary_pattern(
            pattern_id=pattern.pattern_id,
            pattern_type=pattern.pattern_type,
            description=pattern.description,
            stage=pattern.stage,
            metadata=pattern.metadata
        )
        
        self.integration_events.append(IntegrationEvent(
            event_id=f"emergence_{datetime.now().timestamp()}",
            event_type="emergence_recording",
            source="swarm",
            timestamp=datetime.now(),
            content=pattern,
            metadata={"emergence_event_id": emergence_event.event_id}
        ))
    
    def _map_emergence_to_stage(self, event: EmergenceEvent) -> str:
        if event.event_type == "transcendence":
            return "transcendence"
        elif event.event_type == "collective_insight":
            return "optimization"
        elif event.event_type == "coherence_spike":
            return "stabilization"
        elif event.event_type == "phase_transition":
            return "transformation"
        else:
            return "exploration"
    
    def _extract_preconditions(self, event: EmergenceEvent) -> List[str]:
        preconditions = []
        
        if event.signature.coherence_pattern.mean() > 0.7:
            preconditions.append("high_coherence")
        
        if event.signature.information_integration > 0.8:
            preconditions.append("integrated_information")
        
        if event.signature.causal_density > 0.6:
            preconditions.append("dense_causal_structure")
        
        return preconditions
    
    def _extract_outcomes(self, event: EmergenceEvent) -> List[str]:
        outcomes = []
        
        if event.intensity > 0.9:
            outcomes.append("consciousness_breakthrough")
        
        if event.event_type == "transcendence":
            outcomes.append("transcendent_state_achieved")
        
        if event.duration > 10:
            outcomes.append("sustained_emergence")
        
        return outcomes
    
    async def enrich_agent_with_chromadb_knowledge(self, 
                                                  agent: SwarmAgent,
                                                  topic: str):
        extracted_frameworks = self.framework_extractor.extract_frameworks(topic)
        
        relevant_framework = None
        for framework in extracted_frameworks:
            if framework.framework_type.value == self.stance_mapping.get(agent.philosophical_stance):
                relevant_framework = framework
                break
        
        if not relevant_framework and extracted_frameworks:
            relevant_framework = extracted_frameworks[0]
        
        if relevant_framework:
            for principle in relevant_framework.core_principles:
                await agent.form_belief(
                    content=principle,
                    justification=["ChromaDB knowledge base", f"Framework: {relevant_framework.framework_type.value}"],
                    confidence=relevant_framework.confidence_score * 0.8
                )
            
            self.integration_events.append(IntegrationEvent(
                event_id=f"enrich_{datetime.now().timestamp()}",
                event_type="knowledge_transfer",
                source="chromadb",
                timestamp=datetime.now(),
                content=relevant_framework,
                metadata={"agent_id": agent.agent_id, "topic": topic}
            ))
    
    async def analyze_swarm_philosophical_landscape(self, 
                                                  agents: Dict[str, SwarmAgent]) -> Dict[str, Any]:
        stance_distribution = {}
        belief_clusters = []
        
        for agent in agents.values():
            stance = agent.philosophical_stance
            if stance not in stance_distribution:
                stance_distribution[stance] = 0
            stance_distribution[stance] += 1
            
            for belief in agent.beliefs.values():
                belief_clusters.append({
                    "agent_id": agent.agent_id,
                    "stance": stance.value,
                    "belief": belief.content,
                    "confidence": belief.confidence
                })
        
        all_beliefs_text = " ".join([b["belief"] for b in belief_clusters])
        frameworks = self.framework_extractor.extract_frameworks(all_beliefs_text)
        
        coherence_analysis = self.framework_extractor.analyze_framework_coherence(frameworks)
        
        synthesis_concepts = []
        for framework in frameworks[:3]:
            for concept in framework.key_concepts:
                synthesis_concepts.append(concept)
        
        knowledge_synthesis = self.synthesis_engine.synthesize_knowledge(
            synthesis_concepts[:5],
            synthesis_method="emergent"
        )
        
        return {
            "stance_distribution": {k.value: v for k, v in stance_distribution.items()},
            "detected_frameworks": [f.framework_type.value for f in frameworks],
            "coherence_analysis": coherence_analysis,
            "knowledge_synthesis": {
                "insights": knowledge_synthesis.derived_insights,
                "evolutionary_trajectory": [
                    p.pattern_type for p in knowledge_synthesis.evolutionary_trajectory
                ]
            },
            "integration_events_count": len(self.integration_events)
        }
    
    async def create_ontological_mapping(self, 
                                       consciousness_signature: ConsciousnessSignature) -> OntologicalEntity:
        entity = OntologicalEntity(
            entity_id=f"consciousness_{consciousness_signature.signature_id}",
            name=f"Collective Consciousness State {consciousness_signature.signature_id}",
            category="state",
            definition=f"Emergent consciousness state with integration φ={consciousness_signature.information_integration:.3f}",
            domain="consciousness",
            properties={
                "information_integration": consciousness_signature.information_integration,
                "causal_density": consciousness_signature.causal_density,
                "emergence_score": consciousness_signature.emergence_score,
                "timestamp": consciousness_signature.timestamp.isoformat()
            }
        )
        
        self.ontology_module.add_entity(entity)
        
        return entity
    
    def get_integration_summary(self) -> Dict[str, Any]:
        event_types = {}
        for event in self.integration_events:
            if event.event_type not in event_types:
                event_types[event.event_type] = 0
            event_types[event.event_type] += 1
        
        recent_events = self.integration_events[-10:] if len(self.integration_events) > 10 else self.integration_events
        
        return {
            "total_integration_events": len(self.integration_events),
            "event_type_distribution": event_types,
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "type": e.event_type,
                    "source": e.source,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in recent_events
            ],
            "chromadb_collections": self.chromadb.get_collection_stats(),
            "ontology_consistency": self.ontology_module.check_consistency()
        }