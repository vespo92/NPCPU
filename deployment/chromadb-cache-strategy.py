import chromadb
from typing import Optional, Dict, Any, List
import redis
import pickle
import time
from functools import lru_cache
import hashlib

class ChromaDBCache:
    """Intelligent caching layer for ChromaDB with Redis backend"""
    
    def __init__(self, 
                 chromadb_client: chromadb.Client,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 ttl: int = 3600):  # 1 hour default TTL
        
        self.chroma = chromadb_client
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.ttl = ttl
        
    def _cache_key(self, collection_name: str, query: str, k: int) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(f"{query}-{k}".encode()).hexdigest()
        return f"chroma:{collection_name}:{query_hash}"
    
    def query_with_cache(self, 
                        collection_name: str,
                        query_texts: List[str],
                        n_results: int = 10,
                        where: Optional[Dict] = None) -> Dict[str, Any]:
        """Query with caching layer"""
        
        # Generate cache key
        cache_key = self._cache_key(
            collection_name, 
            str(query_texts) + str(where), 
            n_results
        )
        
        # Check cache first
        cached = self.redis.get(cache_key)
        if cached:
            return pickle.loads(cached)
        
        # Query ChromaDB
        collection = self.chroma.get_collection(collection_name)
        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )
        
        # Cache results
        self.redis.setex(
            cache_key,
            self.ttl,
            pickle.dumps(results)
        )
        
        return results
    
    def invalidate_collection_cache(self, collection_name: str):
        """Invalidate all cache entries for a collection"""
        pattern = f"chroma:{collection_name}:*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)


class HybridPersonaStore:
    """Hybrid local/cloud persona management with intelligent caching"""
    
    def __init__(self, 
                 local_path: str = "./persona_cache",
                 cloud_endpoint: Optional[str] = None):
        
        # Local ChromaDB instance
        self.local_client = chromadb.PersistentClient(path=local_path)
        
        # Optional cloud ChromaDB instance
        self.cloud_client = None
        if cloud_endpoint:
            self.cloud_client = chromadb.HttpClient(host=cloud_endpoint)
        
        # Cache layer
        self.cache = ChromaDBCache(self.local_client)
        
        # Track local modifications
        self.dirty_collections = set()
        
    def get_persona(self, user_id: str, use_cloud_fallback: bool = True) -> Optional[Dict]:
        """Get persona with local-first, cloud-fallback strategy"""
        
        # Try local first (with cache)
        try:
            results = self.cache.query_with_cache(
                "personas",
                [f"user:{user_id}"],
                n_results=1,
                where={"user_id": user_id}
            )
            
            if results['ids'][0]:
                return {
                    "id": results['ids'][0][0],
                    "metadata": results['metadatas'][0][0],
                    "document": results['documents'][0][0]
                }
        except Exception as e:
            print(f"Local query failed: {e}")
        
        # Fallback to cloud if enabled
        if use_cloud_fallback and self.cloud_client:
            try:
                cloud_collection = self.cloud_client.get_collection("personas")
                results = cloud_collection.query(
                    query_texts=[f"user:{user_id}"],
                    n_results=1,
                    where={"user_id": user_id}
                )
                
                if results['ids'][0]:
                    # Cache locally for next time
                    self._cache_cloud_result(results)
                    return {
                        "id": results['ids'][0][0],
                        "metadata": results['metadatas'][0][0],
                        "document": results['documents'][0][0]
                    }
            except Exception as e:
                print(f"Cloud query failed: {e}")
        
        return None
    
    def update_persona(self, user_id: str, persona_data: Dict):
        """Update persona locally with deferred cloud sync"""
        
        collection = self.local_client.get_or_create_collection("personas")
        
        # Update locally
        collection.upsert(
            ids=[f"persona_{user_id}"],
            documents=[persona_data.get("description", "")],
            metadatas={
                "user_id": user_id,
                "updated_at": time.time(),
                **persona_data.get("metadata", {})
            }
        )
        
        # Mark for sync
        self.dirty_collections.add("personas")
        
        # Invalidate cache
        self.cache.invalidate_collection_cache("personas")
    
    def sync_to_cloud(self):
        """Sync dirty collections to cloud"""
        if not self.cloud_client:
            return
        
        for collection_name in self.dirty_collections:
            try:
                # Get local data
                local_collection = self.local_client.get_collection(collection_name)
                data = local_collection.get()
                
                # Push to cloud
                cloud_collection = self.cloud_client.get_or_create_collection(collection_name)
                if data['ids']:
                    cloud_collection.upsert(
                        ids=data['ids'],
                        embeddings=data['embeddings'],
                        metadatas=data['metadatas'],
                        documents=data['documents']
                    )
                
                print(f"Synced {collection_name} to cloud")
            except Exception as e:
                print(f"Failed to sync {collection_name}: {e}")
        
        self.dirty_collections.clear()
    
    def _cache_cloud_result(self, results: Dict):
        """Cache cloud query results locally"""
        try:
            collection = self.local_client.get_or_create_collection("personas")
            if results['ids'][0]:
                collection.upsert(
                    ids=results['ids'][0],
                    embeddings=results['embeddings'][0] if results.get('embeddings') else None,
                    metadatas=results['metadatas'][0],
                    documents=results['documents'][0]
                )
        except Exception as e:
            print(f"Failed to cache cloud result: {e}")


# Example usage patterns
if __name__ == "__main__":
    # Initialize hybrid store
    store = HybridPersonaStore(
        local_path="./persona_cache",
        cloud_endpoint="https://chroma.myorg.com"  # Optional cloud instance
    )
    
    # Get persona (local-first, cloud-fallback)
    persona = store.get_persona("user123")
    
    # Update persona (local-first, deferred sync)
    store.update_persona("user123", {
        "description": "Prefers concise technical answers with examples",
        "metadata": {
            "learning_style": "visual",
            "expertise_level": "intermediate"
        }
    })
    
    # Periodic sync to cloud (could be triggered by cron, on shutdown, etc.)
    store.sync_to_cloud()