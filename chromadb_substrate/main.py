from core.vector_substrate import ChromaDBVectorSubstrate, PhilosophicalConcept
from frameworks.philosophical_extractor import PhilosophicalFrameworkExtractor
from embeddings.semantic_embedder import ArchitecturalPrincipleEmbedder
from synthesis.knowledge_synthesizer import KnowledgeSynthesisEngine, EvolutionaryPattern, KnowledgeNode
from ontology.ontological_foundations import (
    OntologicalFoundationsModule, 
    OntologicalEntity, 
    OntologicalCategory,
    RelationType,
    OntologicalAxiom
)
from datetime import datetime
import json


class ChromaDBSecondaryAgent:
    def __init__(self, persist_directory: str = "./chromadb_data"):
        self.vector_substrate = ChromaDBVectorSubstrate(persist_directory)
        self.framework_extractor = PhilosophicalFrameworkExtractor()
        self.principle_embedder = ArchitecturalPrincipleEmbedder()
        self.synthesis_engine = KnowledgeSynthesisEngine()
        self.ontology_module = OntologicalFoundationsModule()
        
        self._initialize_foundational_knowledge()
    
    def _initialize_foundational_knowledge(self):
        foundational_principles = [
            {
                "id": "arch_001",
                "name": "Separation of Concerns",
                "description": "Different aspects of a system should be separated into distinct sections",
                "category": "modularity"
            },
            {
                "id": "arch_002",
                "name": "Single Responsibility",
                "description": "Each component should have one and only one reason to change",
                "category": "modularity"
            },
            {
                "id": "arch_003",
                "name": "Emergent Behavior",
                "description": "Complex behaviors arise from simple interactions between components",
                "category": "emergence"
            },
            {
                "id": "arch_004",
                "name": "Feedback Loops",
                "description": "Systems self-regulate through positive and negative feedback mechanisms",
                "category": "cybernetics"
            }
        ]
        
        for principle in foundational_principles:
            self.vector_substrate.add_architectural_principle(
                principle_id=principle["id"],
                name=principle["name"],
                description=principle["description"],
                category=principle["category"]
            )
            
            self.principle_embedder.embed_principle(
                principle["description"],
                metadata=principle
            )
        
        foundational_concepts = [
            PhilosophicalConcept(
                id="phil_001",
                framework="systems_theory",
                principle="Holism",
                description="The whole system exhibits properties not present in individual parts",
                timestamp=datetime.now()
            ),
            PhilosophicalConcept(
                id="phil_002",
                framework="phenomenology",
                principle="Intentionality",
                description="Consciousness is always directed towards objects in experience",
                timestamp=datetime.now()
            ),
            PhilosophicalConcept(
                id="phil_003",
                framework="constructivism",
                principle="Active Construction",
                description="Knowledge is actively built by the cognizing subject",
                timestamp=datetime.now()
            )
        ]
        
        for concept in foundational_concepts:
            self.vector_substrate.add_philosophical_concept(concept)
        
        ontological_entities = [
            OntologicalEntity(
                entity_id="onto_001",
                name="System",
                category=OntologicalCategory.ENTITY,
                definition="A collection of interacting components forming a complex whole",
                domain="systems",
                properties={"complexity": "high", "emergence": True}
            ),
            OntologicalEntity(
                entity_id="onto_002",
                name="Component",
                category=OntologicalCategory.ENTITY,
                definition="A modular part of a system with defined interfaces",
                domain="systems",
                relations=[(RelationType.PART_OF, "onto_001")]
            ),
            OntologicalEntity(
                entity_id="onto_003",
                name="Emergence",
                category=OntologicalCategory.PROCESS,
                definition="The process by which complex patterns arise from simple interactions",
                domain="complexity",
                relations=[(RelationType.EMERGES_FROM, "onto_002")]
            )
        ]
        
        for entity in ontological_entities:
            self.ontology_module.add_entity(entity)
        
        axiom = OntologicalAxiom(
            axiom_id="axiom_001",
            statement="Every system exhibits emergent properties beyond its components",
            domain="systems",
            entities_involved=["onto_001", "onto_003"]
        )
        self.ontology_module.add_axiom(axiom)
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        frameworks = self.framework_extractor.extract_frameworks(text)
        
        framework_analysis = self.framework_extractor.analyze_framework_coherence(frameworks)
        
        semantic_query_results = []
        for framework in frameworks[:3]:  # Top 3 frameworks
            results = self.vector_substrate.query_semantic_similarity(
                framework.context,
                n_results=3
            )
            semantic_query_results.append({
                "framework": framework.framework_type.value,
                "similar_concepts": results['results']
            })
        
        primary_concepts = []
        for framework in frameworks:
            for concept in framework.key_concepts:
                node = KnowledgeNode(
                    node_id=f"node_{concept}_{datetime.now().timestamp()}",
                    content=framework.key_concepts[concept],
                    node_type="concept",
                    source_collection="extracted",
                    relationships=[],
                    weight=framework.confidence_score,
                    timestamp=datetime.now()
                )
                self.synthesis_engine.add_knowledge_node(node)
                primary_concepts.append(node.node_id)
        
        synthesis = self.synthesis_engine.synthesize_knowledge(
            primary_concepts[:5],
            synthesis_method="emergent"
        )
        
        return {
            "extracted_frameworks": [
                {
                    "type": f.framework_type.value,
                    "confidence": f.confidence_score,
                    "principles": f.core_principles,
                    "key_concepts": list(f.key_concepts.keys())
                }
                for f in frameworks
            ],
            "framework_coherence": framework_analysis,
            "semantic_matches": semantic_query_results,
            "knowledge_synthesis": {
                "method": synthesis.synthesis_method,
                "insights": synthesis.derived_insights,
                "confidence": synthesis.confidence_score
            }
        }
    
    def query_knowledge_base(self, query: str, query_type: str = "semantic") -> Dict[str, Any]:
        if query_type == "semantic":
            results = {}
            for collection in ["philosophical_frameworks", "architectural_principles"]:
                results[collection] = self.vector_substrate.query_semantic_similarity(
                    query, 
                    collection_name=collection,
                    n_results=5
                )
            return results
        
        elif query_type == "ontological":
            entities = self.ontology_module.query_ontology(
                "by_domain",
                {"domain": query}
            )
            return {
                "entities": [
                    {
                        "id": e.entity_id,
                        "name": e.name,
                        "definition": e.definition,
                        "category": e.category.value
                    }
                    for e in entities
                ]
            }
        
        elif query_type == "principle":
            similar_principles = self.principle_embedder.find_similar_principles(
                query,
                top_k=5,
                threshold=0.3
            )
            return {
                "similar_principles": [
                    {
                        "text": p[0],
                        "similarity": p[1],
                        "metadata": p[2]
                    }
                    for p in similar_principles
                ]
            }
        
        return {"error": "Unknown query type"}
    
    def add_evolutionary_pattern(self, 
                               pattern_type: str,
                               description: str,
                               stage: str,
                               metadata: Optional[Dict[str, Any]] = None):
        pattern = EvolutionaryPattern(
            pattern_id=f"evo_{datetime.now().timestamp()}",
            pattern_type=pattern_type,
            description=description,
            stage=stage,
            preconditions=metadata.get("preconditions", []) if metadata else [],
            outcomes=metadata.get("outcomes", []) if metadata else [],
            metadata=metadata or {}
        )
        
        self.synthesis_engine.add_evolutionary_pattern(pattern)
        
        self.vector_substrate.add_evolutionary_pattern(
            pattern_id=pattern.pattern_id,
            pattern_type=pattern_type,
            description=description,
            stage=stage,
            metadata=metadata
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        collection_stats = self.vector_substrate.get_collection_stats()
        
        ontology_consistency = self.ontology_module.check_consistency()
        
        principle_clusters = self.principle_embedder.compute_principle_clusters()
        
        return {
            "vector_collections": collection_stats,
            "ontology_status": {
                "total_entities": len(self.ontology_module.entities),
                "total_axioms": len(self.ontology_module.axioms),
                "consistency": ontology_consistency
            },
            "embeddings": {
                "total_principles": len(self.principle_embedder.principle_embeddings),
                "clusters": len(principle_clusters)
            },
            "synthesis_engine": {
                "graph_nodes": self.synthesis_engine.knowledge_graph.number_of_nodes(),
                "graph_edges": self.synthesis_engine.knowledge_graph.number_of_edges(),
                "patterns": len(self.synthesis_engine.patterns_repository)
            }
        }


def example_usage():
    agent = ChromaDBSecondaryAgent()
    
    sample_text = """
    The system exhibits emergent properties through the interaction of its distributed components.
    Each component maintains autonomy while participating in the collective behavior.
    This creates a feedback loop where individual actions influence the global state,
    which in turn affects individual behavior. The architecture follows principles of
    self-organization and adaptation, allowing the system to evolve over time.
    """
    
    analysis = agent.process_text_input(sample_text)
    print("Text Analysis Results:")
    print(json.dumps(analysis, indent=2))
    
    query_results = agent.query_knowledge_base(
        "self-organization and feedback",
        query_type="semantic"
    )
    print("\nSemantic Query Results:")
    print(json.dumps(query_results, indent=2))
    
    agent.add_evolutionary_pattern(
        pattern_type="adaptation",
        description="System adapts to environmental changes through feedback mechanisms",
        stage="optimization",
        metadata={
            "preconditions": ["feedback_loops_established", "sensing_capability"],
            "outcomes": ["improved_performance", "resilience"]
        }
    )
    
    status = agent.get_system_status()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    example_usage()