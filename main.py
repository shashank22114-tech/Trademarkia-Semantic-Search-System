import os
import numpy as np
import skfuzzy as fuzz
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class SemanticSearchSystem:
    def __init__(self, data_path, num_clusters=20, sample_size=2000):
        print("Starting FastAPI and Semantic System...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.num_clusters = num_clusters
        
        # Load and clean data
        texts = []
        for category in os.listdir(data_path):
            cat_path = os.path.join(data_path, category)
            if os.path.isdir(cat_path):
                for file_name in os.listdir(cat_path)[:100]:
                    with open(os.path.join(cat_path, file_name), 'r', errors='ignore') as f:
                        parts = f.read().split('\n\n', 1)
                        if len(parts) > 1 and parts[1].strip():
                            texts.append(parts[1].strip())
        
        # Initial Embedding & Clustering (Part 1 & 2) [cite: 8]
        print(f"Embedding {len(texts)} documents...")
        self.embeddings = self.model.encode(texts)
        
        print("Clustering corpus...")
        # m=1.2 helps prevent uniform distributions in high-dimensional space
        self.cluster_centers, _, _, _, _, _, _ = fuzz.cluster.cmeans(
            self.embeddings.T, c=self.num_clusters, m=1.2, error=0.005, maxiter=1000
        )
        
        # State Management & Cache built from first principles [cite: 9, 10, 11]
        self.stats = {"total_entries": 0, "hit_count": 0, "miss_count": 0}
        
        # Cache partitioned by cluster to optimize lookup time complexity 
        self.cache = {i: [] for i in range(num_clusters)} 

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def reset_cache(self):
        self.cache = {i: [] for i in range(self.num_clusters)}
        self.stats = {"total_entries": 0, "hit_count": 0, "miss_count": 0}

# Initialize system globally so the API can access it
system = SemanticSearchSystem('20_newsgroups')

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    query_text = request.query
    query_emb = system.model.encode([query_text])[0]
    
    # Predict fuzzy distribution for the query
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        query_emb.reshape(-1, 1), system.cluster_centers, m=1.2, error=0.005, maxiter=1000
    )
    dominant_cluster = int(np.argmax(u))
    
    # 1. Check Semantic Cache (ONLY within the dominant cluster for O(K) efficiency)
    # Threshold is tunable. 0.85 balances strictness with flexibility [cite: 35]
    threshold = 0.85 
    for cached_item in system.cache[dominant_cluster]:
        sim = system.cosine_similarity(query_emb, cached_item["emb"])
        if sim >= threshold:
            system.stats["hit_count"] += 1
            return {
                "query": query_text,
                "cache_hit": True,
                "matched_query": cached_item["query"],
                "similarity_score": round(float(sim), 3),
                "dominant_cluster": dominant_cluster,
                "result": cached_item["result"]
            }

    # 2. Cache Miss: Compute, Store, and Return [cite: 52]
    system.stats["miss_count"] += 1
    system.stats["total_entries"] += 1
    
    # Simulated search result for the assignment
    simulated_result = f"Retrieved documents from cluster {dominant_cluster} matching: '{query_text}'"
    
    # Store in the partitioned cache [cite: 38]
    system.cache[dominant_cluster].append({
        "query": query_text,
        "emb": query_emb,
        "result": simulated_result
    })
    
    return {
        "query": query_text,
        "cache_hit": False,
        "dominant_cluster": dominant_cluster,
        "result": simulated_result
    }

@app.get("/cache/stats")
async def get_stats():
    total_calls = system.stats["hit_count"] + system.stats["miss_count"]
    hit_rate = system.stats["hit_count"] / total_calls if total_calls > 0 else 0.0
    return {
        "total_entries": system.stats["total_entries"],
        "hit_count": system.stats["hit_count"],
        "miss_count": system.stats["miss_count"],
        "hit_rate": round(hit_rate, 3)
    }

@app.delete("/cache")
async def clear_cache():
    system.reset_cache()
    return {"message": "Cache entirely flushed"}