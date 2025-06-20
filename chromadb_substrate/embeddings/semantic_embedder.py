import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torch
from dataclasses import dataclass
import json


@dataclass
class SemanticEmbedding:
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    embedding_model: str
    dimension: int


class ArchitecturalPrincipleEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.principle_embeddings: Dict[str, SemanticEmbedding] = {}
        self.architectural_categories = self._initialize_categories()
        
    def _initialize_categories(self) -> Dict[str, List[str]]:
        return {
            "modularity": [
                "separation of concerns",
                "loose coupling",
                "high cohesion",
                "interface segregation",
                "dependency inversion"
            ],
            "scalability": [
                "horizontal scaling",
                "vertical scaling",
                "load balancing",
                "distributed systems",
                "microservices"
            ],
            "resilience": [
                "fault tolerance",
                "graceful degradation",
                "circuit breakers",
                "retry mechanisms",
                "redundancy"
            ],
            "maintainability": [
                "clean code",
                "documentation",
                "testability",
                "refactoring",
                "code reviews"
            ],
            "performance": [
                "optimization",
                "caching",
                "lazy loading",
                "async processing",
                "resource efficiency"
            ],
            "security": [
                "authentication",
                "authorization",
                "encryption",
                "input validation",
                "least privilege"
            ]
        }
    
    def embed_principle(self, principle_text: str, metadata: Optional[Dict[str, Any]] = None) -> SemanticEmbedding:
        embedding = self.model.encode(principle_text, convert_to_numpy=True)
        
        semantic_embedding = SemanticEmbedding(
            text=principle_text,
            embedding=embedding,
            metadata=metadata or {},
            embedding_model=self.model_name,
            dimension=self.embedding_dimension
        )
        
        principle_id = metadata.get('id', principle_text[:50])
        self.principle_embeddings[principle_id] = semantic_embedding
        
        return semantic_embedding
    
    def embed_batch(self, principles: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[SemanticEmbedding]:
        embeddings = self.model.encode(principles, convert_to_numpy=True, show_progress_bar=True)
        
        if metadata_list is None:
            metadata_list = [{}] * len(principles)
        
        semantic_embeddings = []
        for i, (text, embedding, metadata) in enumerate(zip(principles, embeddings, metadata_list)):
            semantic_embedding = SemanticEmbedding(
                text=text,
                embedding=embedding,
                metadata=metadata,
                embedding_model=self.model_name,
                dimension=self.embedding_dimension
            )
            semantic_embeddings.append(semantic_embedding)
            
            principle_id = metadata.get('id', f"{text[:50]}_{i}")
            self.principle_embeddings[principle_id] = semantic_embedding
        
        return semantic_embeddings
    
    def find_similar_principles(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[str, float, Dict[str, Any]]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        similarities = []
        for principle_id, semantic_embedding in self.principle_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                semantic_embedding.embedding.reshape(1, -1)
            )[0, 0]
            
            if similarity >= threshold:
                similarities.append((
                    semantic_embedding.text,
                    float(similarity),
                    semantic_embedding.metadata
                ))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def categorize_principle(self, principle_text: str) -> Dict[str, float]:
        principle_embedding = self.model.encode(principle_text, convert_to_numpy=True)
        
        category_scores = {}
        for category, examples in self.architectural_categories.items():
            category_embeddings = self.model.encode(examples, convert_to_numpy=True)
            
            similarities = cosine_similarity(
                principle_embedding.reshape(1, -1),
                category_embeddings
            )[0]
            
            category_scores[category] = float(np.mean(similarities))
        
        total_score = sum(category_scores.values())
        if total_score > 0:
            category_scores = {k: v/total_score for k, v in category_scores.items()}
        
        return category_scores
    
    def compute_principle_clusters(self, min_samples: int = 2, eps: float = 0.3) -> Dict[int, List[str]]:
        if len(self.principle_embeddings) < min_samples:
            return {0: list(self.principle_embeddings.keys())}
        
        embeddings_matrix = np.vstack([se.embedding for se in self.principle_embeddings.values()])
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings_matrix)
        
        clusters = {}
        for idx, (principle_id, label) in enumerate(zip(self.principle_embeddings.keys(), cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(principle_id)
        
        return clusters
    
    def reduce_dimensions(self, n_components: int = 2) -> Dict[str, np.ndarray]:
        if len(self.principle_embeddings) < n_components:
            return {}
        
        embeddings_matrix = np.vstack([se.embedding for se in self.principle_embeddings.values()])
        
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)
        
        reduced_dict = {}
        for idx, principle_id in enumerate(self.principle_embeddings.keys()):
            reduced_dict[principle_id] = reduced_embeddings[idx]
        
        return reduced_dict
    
    def compute_conceptual_distance(self, principle1_id: str, principle2_id: str) -> float:
        if principle1_id not in self.principle_embeddings or principle2_id not in self.principle_embeddings:
            return float('inf')
        
        embedding1 = self.principle_embeddings[principle1_id].embedding
        embedding2 = self.principle_embeddings[principle2_id].embedding
        
        cosine_sim = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0, 0]
        
        return 1 - cosine_sim
    
    def generate_principle_summary(self, principle_ids: List[str]) -> str:
        if not principle_ids:
            return "No principles to summarize"
        
        valid_principles = [
            self.principle_embeddings[pid].text 
            for pid in principle_ids 
            if pid in self.principle_embeddings
        ]
        
        if not valid_principles:
            return "No valid principles found"
        
        embeddings = np.vstack([
            self.principle_embeddings[pid].embedding 
            for pid in principle_ids 
            if pid in self.principle_embeddings
        ])
        
        centroid = np.mean(embeddings, axis=0)
        
        similarities = []
        for pid in principle_ids:
            if pid in self.principle_embeddings:
                similarity = cosine_similarity(
                    centroid.reshape(1, -1),
                    self.principle_embeddings[pid].embedding.reshape(1, -1)
                )[0, 0]
                similarities.append((pid, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        most_representative_id = similarities[0][0] if similarities else principle_ids[0]
        most_representative_text = self.principle_embeddings[most_representative_id].text
        
        return f"Core principle: {most_representative_text} (representing {len(valid_principles)} principles)"
    
    def export_embeddings(self, output_path: str):
        export_data = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "principles": []
        }
        
        for principle_id, semantic_embedding in self.principle_embeddings.items():
            export_data["principles"].append({
                "id": principle_id,
                "text": semantic_embedding.text,
                "embedding": semantic_embedding.embedding.tolist(),
                "metadata": semantic_embedding.metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_embeddings(self, input_path: str):
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        if import_data["model_name"] != self.model_name:
            raise ValueError(f"Model mismatch: expected {self.model_name}, got {import_data['model_name']}")
        
        self.principle_embeddings.clear()
        
        for principle_data in import_data["principles"]:
            semantic_embedding = SemanticEmbedding(
                text=principle_data["text"],
                embedding=np.array(principle_data["embedding"]),
                metadata=principle_data["metadata"],
                embedding_model=self.model_name,
                dimension=self.embedding_dimension
            )
            self.principle_embeddings[principle_data["id"]] = semantic_embedding