from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import networkx as nx
from collections import defaultdict


class OntologicalCategory(Enum):
    ENTITY = "entity"
    PROCESS = "process"
    RELATION = "relation"
    PROPERTY = "property"
    EVENT = "event"
    STATE = "state"
    CONCEPT = "concept"
    PRINCIPLE = "principle"


class RelationType(Enum):
    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    PRECEDES = "precedes"
    CONTRADICTS = "contradicts"
    COMPLEMENTS = "complements"
    INSTANTIATES = "instantiates"
    EMERGES_FROM = "emerges_from"


@dataclass
class OntologicalEntity:
    entity_id: str
    name: str
    category: OntologicalCategory
    definition: str
    domain: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[RelationType, str]] = field(default_factory=list)
    axioms: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologicalAxiom:
    axiom_id: str
    statement: str
    formal_representation: Optional[str] = None
    domain: str = "general"
    entities_involved: List[str] = field(default_factory=list)
    proof_sketch: Optional[str] = None


@dataclass
class DomainOntology:
    domain_name: str
    description: str
    root_concepts: List[str]
    axioms: List[OntologicalAxiom]
    entities: Dict[str, OntologicalEntity]
    timestamp: datetime


class OntologicalFoundationsModule:
    def __init__(self):
        self.ontology_graph = nx.DiGraph()
        self.entities: Dict[str, OntologicalEntity] = {}
        self.axioms: Dict[str, OntologicalAxiom] = {}
        self.domains: Dict[str, DomainOntology] = {}
        self.foundational_categories = self._initialize_foundational_categories()
        self.universal_relations = self._initialize_universal_relations()
        
    def _initialize_foundational_categories(self) -> Dict[str, Any]:
        return {
            "being": {
                "description": "That which exists or has existence",
                "subcategories": ["substance", "accident", "essence", "existence"],
                "axioms": [
                    "Everything that exists has a mode of being",
                    "Being is convertible with unity, truth, and goodness"
                ]
            },
            "becoming": {
                "description": "Process of change and transformation",
                "subcategories": ["motion", "change", "development", "evolution"],
                "axioms": [
                    "All becoming presupposes being",
                    "Change requires potentiality and actuality"
                ]
            },
            "relation": {
                "description": "Connection or association between entities",
                "subcategories": ["causal", "logical", "spatial", "temporal"],
                "axioms": [
                    "Relations require at least two relata",
                    "Some relations are symmetric, others asymmetric"
                ]
            },
            "unity": {
                "description": "State of being one or undivided",
                "subcategories": ["simple", "composite", "collective", "systematic"],
                "axioms": [
                    "Unity can be substantial or accidental",
                    "The whole is greater than the sum of its parts"
                ]
            }
        }
    
    def _initialize_universal_relations(self) -> Dict[RelationType, Dict[str, Any]]:
        return {
            RelationType.IS_A: {
                "properties": ["transitive", "asymmetric"],
                "inverse": None,
                "description": "Subsumption or class membership"
            },
            RelationType.HAS_A: {
                "properties": ["non-transitive"],
                "inverse": RelationType.PART_OF,
                "description": "Possession or composition"
            },
            RelationType.PART_OF: {
                "properties": ["transitive", "asymmetric"],
                "inverse": RelationType.HAS_A,
                "description": "Mereological relation"
            },
            RelationType.CAUSES: {
                "properties": ["transitive", "asymmetric"],
                "inverse": None,
                "description": "Causal relationship"
            },
            RelationType.DEPENDS_ON: {
                "properties": ["non-symmetric"],
                "inverse": None,
                "description": "Ontological dependence"
            }
        }
    
    def add_entity(self, entity: OntologicalEntity):
        self.entities[entity.entity_id] = entity
        
        self.ontology_graph.add_node(
            entity.entity_id,
            name=entity.name,
            category=entity.category.value,
            domain=entity.domain,
            definition=entity.definition
        )
        
        for relation_type, target_id in entity.relations:
            self.ontology_graph.add_edge(
                entity.entity_id,
                target_id,
                relation_type=relation_type.value
            )
    
    def add_axiom(self, axiom: OntologicalAxiom):
        self.axioms[axiom.axiom_id] = axiom
        
        for entity_id in axiom.entities_involved:
            if entity_id in self.entities:
                self.entities[entity_id].axioms.append(axiom.axiom_id)
    
    def create_domain_ontology(self, 
                             domain_name: str,
                             description: str,
                             root_concepts: List[str]) -> DomainOntology:
        domain_entities = {
            eid: entity for eid, entity in self.entities.items()
            if entity.domain == domain_name
        }
        
        domain_axioms = [
            axiom for axiom in self.axioms.values()
            if axiom.domain == domain_name or axiom.domain == "general"
        ]
        
        domain_ontology = DomainOntology(
            domain_name=domain_name,
            description=description,
            root_concepts=root_concepts,
            axioms=domain_axioms,
            entities=domain_entities,
            timestamp=datetime.now()
        )
        
        self.domains[domain_name] = domain_ontology
        return domain_ontology
    
    def infer_relations(self, entity_id: str) -> List[Tuple[RelationType, str, str]]:
        inferred_relations = []
        
        if entity_id not in self.ontology_graph:
            return inferred_relations
        
        for relation_type, properties in self.universal_relations.items():
            if "transitive" in properties["properties"]:
                transitive_closure = self._compute_transitive_closure(entity_id, relation_type)
                for target in transitive_closure:
                    inferred_relations.append((relation_type, entity_id, target))
        
        inferred_relations.extend(self._infer_inverse_relations(entity_id))
        
        return inferred_relations
    
    def _compute_transitive_closure(self, start_node: str, relation_type: RelationType) -> Set[str]:
        closure = set()
        to_visit = [start_node]
        visited = set()
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for _, target, data in self.ontology_graph.edges(current, data=True):
                if data.get('relation_type') == relation_type.value:
                    closure.add(target)
                    to_visit.append(target)
        
        return closure
    
    def _infer_inverse_relations(self, entity_id: str) -> List[Tuple[RelationType, str, str]]:
        inverse_relations = []
        
        for source, _, data in self.ontology_graph.in_edges(entity_id, data=True):
            relation_str = data.get('relation_type')
            if relation_str:
                try:
                    relation_type = RelationType(relation_str)
                    inverse = self.universal_relations[relation_type].get('inverse')
                    if inverse:
                        inverse_relations.append((inverse, entity_id, source))
                except ValueError:
                    pass
        
        return inverse_relations
    
    def check_consistency(self) -> Dict[str, Any]:
        inconsistencies = []
        
        contradiction_pairs = self._find_contradictions()
        inconsistencies.extend([
            f"Contradiction between {e1} and {e2}"
            for e1, e2 in contradiction_pairs
        ])
        
        circular_dependencies = self._detect_circular_dependencies()
        inconsistencies.extend([
            f"Circular dependency: {' -> '.join(cycle)}"
            for cycle in circular_dependencies
        ])
        
        violated_constraints = self._check_constraint_violations()
        inconsistencies.extend(violated_constraints)
        
        return {
            "is_consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "timestamp": datetime.now().isoformat()
        }
    
    def _find_contradictions(self) -> List[Tuple[str, str]]:
        contradictions = []
        
        for source, target, data in self.ontology_graph.edges(data=True):
            if data.get('relation_type') == RelationType.CONTRADICTS.value:
                contradictions.append((source, target))
        
        return contradictions
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        dependency_edges = [
            (s, t) for s, t, d in self.ontology_graph.edges(data=True)
            if d.get('relation_type') == RelationType.DEPENDS_ON.value
        ]
        
        dependency_graph = nx.DiGraph(dependency_edges)
        
        try:
            cycles = list(nx.simple_cycles(dependency_graph))
            return cycles
        except:
            return []
    
    def _check_constraint_violations(self) -> List[str]:
        violations = []
        
        for entity_id, entity in self.entities.items():
            for constraint in entity.constraints:
                if not self._evaluate_constraint(entity, constraint):
                    violations.append(f"Constraint violation for {entity_id}: {constraint}")
        
        return violations
    
    def _evaluate_constraint(self, entity: OntologicalEntity, constraint: str) -> bool:
        return True
    
    def query_ontology(self, 
                      query_type: str,
                      parameters: Dict[str, Any]) -> List[OntologicalEntity]:
        results = []
        
        if query_type == "by_category":
            category = OntologicalCategory(parameters.get("category"))
            results = [e for e in self.entities.values() if e.category == category]
            
        elif query_type == "by_domain":
            domain = parameters.get("domain")
            results = [e for e in self.entities.values() if e.domain == domain]
            
        elif query_type == "by_relation":
            relation_type = RelationType(parameters.get("relation_type"))
            target_id = parameters.get("target_id")
            results = [
                self.entities[eid] for eid in self.entities
                if any(r[0] == relation_type and r[1] == target_id 
                      for r in self.entities[eid].relations)
            ]
            
        elif query_type == "by_property":
            property_name = parameters.get("property_name")
            property_value = parameters.get("property_value")
            results = [
                e for e in self.entities.values()
                if e.properties.get(property_name) == property_value
            ]
        
        return results
    
    def compute_ontological_distance(self, entity1_id: str, entity2_id: str) -> float:
        if entity1_id not in self.ontology_graph or entity2_id not in self.ontology_graph:
            return float('inf')
        
        try:
            path_length = nx.shortest_path_length(
                self.ontology_graph.to_undirected(),
                entity1_id,
                entity2_id
            )
            return float(path_length)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def export_ontology(self, output_path: str, domain: Optional[str] = None):
        if domain and domain in self.domains:
            export_data = {
                "domain": domain,
                "ontology": {
                    "description": self.domains[domain].description,
                    "root_concepts": self.domains[domain].root_concepts,
                    "entities": [
                        {
                            "id": e.entity_id,
                            "name": e.name,
                            "category": e.category.value,
                            "definition": e.definition,
                            "properties": e.properties,
                            "relations": [(r[0].value, r[1]) for r in e.relations],
                            "axioms": e.axioms,
                            "constraints": e.constraints
                        }
                        for e in self.domains[domain].entities.values()
                    ],
                    "axioms": [
                        {
                            "id": a.axiom_id,
                            "statement": a.statement,
                            "formal_representation": a.formal_representation,
                            "entities_involved": a.entities_involved
                        }
                        for a in self.domains[domain].axioms
                    ]
                }
            }
        else:
            export_data = {
                "domain": "all",
                "ontology": {
                    "entities": [
                        {
                            "id": e.entity_id,
                            "name": e.name,
                            "category": e.category.value,
                            "definition": e.definition,
                            "domain": e.domain,
                            "properties": e.properties,
                            "relations": [(r[0].value, r[1]) for r in e.relations],
                            "axioms": e.axioms,
                            "constraints": e.constraints
                        }
                        for e in self.entities.values()
                    ],
                    "axioms": [
                        {
                            "id": a.axiom_id,
                            "statement": a.statement,
                            "formal_representation": a.formal_representation,
                            "domain": a.domain,
                            "entities_involved": a.entities_involved
                        }
                        for a in self.axioms.values()
                    ],
                    "graph_metrics": {
                        "total_entities": len(self.entities),
                        "total_relations": self.ontology_graph.number_of_edges(),
                        "total_axioms": len(self.axioms),
                        "domains": list(self.domains.keys())
                    }
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)