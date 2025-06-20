import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PhilosophicalConcept:
    id: str
    framework: str
    principle: str
    description: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class ChromaDBVectorSubstrate:
    def __init__(self, persist_directory: str = "./chromadb_data"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self._initialize_collections()
    
    def _initialize_collections(self):
        self.philosophical_frameworks = self.client.get_or_create_collection(
            name="philosophical_frameworks",
            embedding_function=self.embedding_function,
            metadata={"description": "Core philosophical frameworks and principles"}
        )
        
        self.architectural_principles = self.client.get_or_create_collection(
            name="architectural_principles",
            embedding_function=self.embedding_function,
            metadata={"description": "System architecture design principles"}
        )
        
        self.evolutionary_patterns = self.client.get_or_create_collection(
            name="evolutionary_patterns",
            embedding_function=self.embedding_function,
            metadata={"description": "Patterns of system evolution and adaptation"}
        )
        
        self.ontological_foundations = self.client.get_or_create_collection(
            name="ontological_foundations",
            embedding_function=self.embedding_function,
            metadata={"description": "Fundamental ontological structures"}
        )
    
    def add_philosophical_concept(self, concept: PhilosophicalConcept):
        self.philosophical_frameworks.add(
            documents=[concept.description],
            metadatas=[{
                "framework": concept.framework,
                "principle": concept.principle,
                "timestamp": concept.timestamp.isoformat() if concept.timestamp else datetime.now().isoformat(),
                **(concept.metadata or {})
            }],
            ids=[concept.id]
        )
    
    def query_semantic_similarity(self, 
                                query_text: str, 
                                collection_name: str = "philosophical_frameworks",
                                n_results: int = 5) -> Dict[str, Any]:
        collection = getattr(self, collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        return self._format_query_results(results)
    
    def _format_query_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity_score': 1 - results['distances'][0][i]
            })
        
        return {
            'results': formatted_results,
            'query_metadata': {
                'timestamp': datetime.now().isoformat(),
                'result_count': len(formatted_results)
            }
        }
    
    def extract_framework_patterns(self, framework_name: str) -> List[Dict[str, Any]]:
        results = self.philosophical_frameworks.get(
            where={"framework": framework_name},
            include=["metadatas", "documents"]
        )
        
        patterns = []
        for i in range(len(results['ids'])):
            patterns.append({
                'id': results['ids'][i],
                'principle': results['metadatas'][i].get('principle', ''),
                'description': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return patterns
    
    def synthesize_knowledge(self, concepts: List[str]) -> Dict[str, Any]:
        all_results = []
        
        for concept in concepts:
            for collection_name in ['philosophical_frameworks', 'architectural_principles', 
                                   'evolutionary_patterns', 'ontological_foundations']:
                results = self.query_semantic_similarity(concept, collection_name, n_results=3)
                all_results.extend(results['results'])
        
        unique_results = {}
        for result in all_results:
            if result['id'] not in unique_results or result['similarity_score'] > unique_results[result['id']]['similarity_score']:
                unique_results[result['id']] = result
        
        synthesis = {
            'synthesized_concepts': list(unique_results.values()),
            'concept_count': len(unique_results),
            'source_queries': concepts,
            'timestamp': datetime.now().isoformat()
        }
        
        return synthesis
    
    def add_architectural_principle(self, 
                                  principle_id: str,
                                  name: str,
                                  description: str,
                                  category: str,
                                  metadata: Optional[Dict[str, Any]] = None):
        self.architectural_principles.add(
            documents=[description],
            metadatas=[{
                "name": name,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }],
            ids=[principle_id]
        )
    
    def add_evolutionary_pattern(self,
                               pattern_id: str,
                               pattern_type: str,
                               description: str,
                               stage: str,
                               metadata: Optional[Dict[str, Any]] = None):
        self.evolutionary_patterns.add(
            documents=[description],
            metadatas=[{
                "pattern_type": pattern_type,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }],
            ids=[pattern_id]
        )
    
    def add_ontological_foundation(self,
                                 foundation_id: str,
                                 concept: str,
                                 definition: str,
                                 domain: str,
                                 metadata: Optional[Dict[str, Any]] = None):
        self.ontological_foundations.add(
            documents=[definition],
            metadatas=[{
                "concept": concept,
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }],
            ids=[foundation_id]
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        stats = {}
        for collection_name in ['philosophical_frameworks', 'architectural_principles',
                               'evolutionary_patterns', 'ontological_foundations']:
            collection = getattr(self, collection_name)
            stats[collection_name] = {
                'count': collection.count(),
                'metadata': collection.metadata
            }
        return stats