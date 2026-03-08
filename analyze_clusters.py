import os
import numpy as np
import skfuzzy as fuzz
from sentence_transformers import SentenceTransformer

print("--- GENERATING CLUSTER ANALYSIS EVIDENCE ---")
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
data_path = '20_newsgroups'
# Grabbing a small, fast sample just for this boundary analysis
for category in os.listdir(data_path):
    cat_path = os.path.join(data_path, category)
    if os.path.isdir(cat_path):
        for file_name in os.listdir(cat_path)[:15]: 
            with open(os.path.join(cat_path, file_name), 'r', errors='ignore') as f:
                parts = f.read().split('\n\n', 1)
                if len(parts) > 1 and parts[1].strip():
                    texts.append(parts[1].strip())

print("Embedding data for analysis...")
embeddings = model.encode(texts)

print("Running fuzzy math...")
# We use m=1.2 to sharpen the boundaries in high dimensional space
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    embeddings.T, c=20, m=1.2, error=0.005, maxiter=1000
)

# 1. Find a Core Document
max_probs = np.max(u, axis=0)
core_doc_idx = np.argmax(max_probs)
core_cluster = np.argmax(u[:, core_doc_idx])

print(f"\n1. CORE DOCUMENT (Strongly belongs to Cluster {core_cluster}):")
print(f"Confidence: {max_probs[core_doc_idx]*100:.1f}%")
print(f"Text Snippet: {texts[core_doc_idx][:250]}...\n")

# 2. Find a Boundary Document
sorted_u = np.sort(u, axis=0)
uncertainty_gaps = sorted_u[-1, :] - sorted_u[-2, :]
boundary_doc_idx = np.argmin(uncertainty_gaps)
top_2_clusters = np.argsort(u[:, boundary_doc_idx])[-2:]

print(f"2. BOUNDARY DOCUMENT (Highly Uncertain):")
print(f"Torn between Cluster {top_2_clusters[1]} ({sorted_u[-1, boundary_doc_idx]*100:.1f}%) and Cluster {top_2_clusters[0]} ({sorted_u[-2, boundary_doc_idx]*100:.1f}%)")
print(f"Text Snippet: {texts[boundary_doc_idx][:250]}...")