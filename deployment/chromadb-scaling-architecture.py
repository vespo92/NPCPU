import chromadb
from chromadb.config import Settings
import boto3
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List
import threading
import time

class PersonaVectorDBManager:
    """Manages ChromaDB instances with cloud synchronization for persona agents"""
    
    def __init__(self, 
                 local_path: str = "./persona_vectors",
                 s3_bucket: str = "persona-vectors-global",
                 sync_interval: int = 300):  # 5 minutes
        
        # Local ChromaDB instance
        self.client = chromadb.PersistentClient(
            path=local_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Cloud storage client
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        
        # Sync configuration
        self.sync_interval = sync_interval
        self.machine_id = self._get_machine_id()
        self.sync_thread = None
        self.running = False
        
    def _get_machine_id(self) -> str:
        """Generate unique machine identifier"""
        import platform
        import uuid
        machine_info = f"{platform.node()}-{uuid.getnode()}"
        return hashlib.md5(machine_info.encode()).hexdigest()[:8]
    
    def get_or_create_collection(self, name: str, **kwargs) -> chromadb.Collection:
        """Get or create a collection with cloud sync metadata"""
        try:
            collection = self.client.get_collection(name)
        except:
            collection = self.client.create_collection(
                name=name,
                metadata={
                    "created_at": datetime.utcnow().isoformat(),
                    "machine_id": self.machine_id,
                    **kwargs
                }
            )
        return collection
    
    def sync_to_cloud(self, collection_name: str):
        """Push local changes to cloud storage"""
        collection = self.client.get_collection(collection_name)
        
        # Get all data from collection
        data = collection.get(include=["embeddings", "metadatas", "documents"])
        
        # Create snapshot with metadata
        snapshot = {
            "collection_name": collection_name,
            "machine_id": self.machine_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "count": collection.count()
        }
        
        # Upload to S3
        key = f"collections/{collection_name}/snapshots/{self.machine_id}-{int(time.time())}.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(snapshot, default=str)
        )
        
        # Update manifest
        self._update_manifest(collection_name)
    
    def sync_from_cloud(self, collection_name: str):
        """Pull and merge changes from cloud storage"""
        # Get latest snapshots from all machines
        prefix = f"collections/{collection_name}/snapshots/"
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return
        
        # Find most recent snapshot per machine
        latest_snapshots = {}
        for obj in response['Contents']:
            parts = obj['Key'].split('/')[-1].split('-')
            machine_id = parts[0]
            timestamp = int(parts[1].split('.')[0])
            
            if machine_id not in latest_snapshots or timestamp > latest_snapshots[machine_id][1]:
                latest_snapshots[machine_id] = (obj['Key'], timestamp)
        
        # Merge snapshots
        collection = self.get_or_create_collection(collection_name)
        
        for machine_id, (key, _) in latest_snapshots.items():
            if machine_id == self.machine_id:
                continue  # Skip own data
                
            # Download snapshot
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            snapshot = json.loads(obj['Body'].read())
            
            # Merge data - using upsert to handle conflicts
            if snapshot['data']['ids']:
                collection.upsert(
                    ids=snapshot['data']['ids'],
                    embeddings=snapshot['data']['embeddings'],
                    metadatas=snapshot['data']['metadatas'],
                    documents=snapshot['data']['documents']
                )
    
    def _update_manifest(self, collection_name: str):
        """Update collection manifest with sync metadata"""
        manifest_key = f"collections/{collection_name}/manifest.json"
        
        try:
            # Get existing manifest
            obj = self.s3.get_object(Bucket=self.bucket, Key=manifest_key)
            manifest = json.loads(obj['Body'].read())
        except:
            manifest = {"machines": {}, "created_at": datetime.utcnow().isoformat()}
        
        # Update machine entry
        manifest["machines"][self.machine_id] = {
            "last_sync": datetime.utcnow().isoformat(),
            "collection_count": self.client.get_collection(collection_name).count()
        }
        
        # Save manifest
        self.s3.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=json.dumps(manifest)
        )
    
    def start_auto_sync(self, collections: List[str]):
        """Start background sync thread"""
        self.running = True
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            args=(collections,),
            daemon=True
        )
        self.sync_thread.start()
    
    def _sync_loop(self, collections: List[str]):
        """Background sync loop"""
        while self.running:
            for collection_name in collections:
                try:
                    self.sync_from_cloud(collection_name)
                    self.sync_to_cloud(collection_name)
                except Exception as e:
                    print(f"Sync error for {collection_name}: {e}")
            
            time.sleep(self.sync_interval)
    
    def stop_auto_sync(self):
        """Stop background sync"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join()


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = PersonaVectorDBManager(
        local_path="./persona_cache",
        s3_bucket="my-org-persona-vectors",
        sync_interval=300  # 5 minutes
    )
    
    # Create/get persona collection
    personas = manager.get_or_create_collection(
        "user_personas",
        embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    )
    
    # Add persona data
    personas.add(
        ids=["persona_001"],
        documents=["User prefers technical explanations with code examples"],
        metadatas={"user_id": "user123", "created_by": manager.machine_id}
    )
    
    # Start auto-sync
    manager.start_auto_sync(["user_personas"])
    
    # Manual sync when switching machines
    manager.sync_from_cloud("user_personas")