# Trademarkia AI & ML Engineer Task - Lightweight Semantic Search System

## Overview
This repository contains a lightweight semantic search system built from first principles. It features fuzzy clustering, a custom semantic cache partitioned for efficiency, and a FastAPI service.

### Quick Start
[cite_start]To start the service cleanly as requested[cite: 64]:
1. Ensure your virtual environment is active.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn main:app --reload`
4. The API will be available at `http://127.0.0.1:8000`.

---

## Part 1: Embedding & Preprocessing
* [cite_start]**Data Cleaning Justification:** The 20 Newsgroups dataset contains heavy metadata noise (email headers, routing paths, footers)[cite: 17]. I explicitly stripped headers by splitting the text on double newlines (`\n\n`). This ensures the embedding model groups documents based on actual semantic content rather than arbitrary metadata (like "Sent from my iPhone" or matching email clients).
* [cite_start]**Embedding Model:** I selected `all-MiniLM-L6-v2`[cite: 18]. [cite_start]It is highly optimized for CPU inference, taking up minimal memory while maintaining high semantic accuracy, perfectly fitting the "lightweight" requirement[cite: 7].

## Part 2: Fuzzy Clustering & Semantic Structure
* [cite_start]**Algorithm Choice:** Hard cluster assignments were not acceptable[cite: 23]. [cite_start]I used **Fuzzy C-Means (FCM)** to generate a probability distribution matrix for each document, allowing a single text to belong to multiple categories simultaneously[cite: 24].
* [cite_start]**Cluster Count:** I selected `20` clusters[cite: 25]. [cite_start]While the real semantic structure is messy, anchoring the `c` parameter to the dataset's 20 original labeled categories provides a logical baseline, allowing the fuzzy membership to handle the overlapping boundaries[cite: 21].

### Cluster Analysis & Boundary Evidence
* **Core Document Snippet:** *"Apple Macintosh SE/30 8MB RAM... I'm after offers in the region of 1250 pounds."*
* **Boundary Document Snippet:** *"Los Angeles Kings... Biggest Surprise... Disappointment"* (This document was torn between overlapping clusters).

**Architectural Observation (The Curse of Dimensionality):**
When analyzing the raw probability matrix, the standard Fuzzy C-Means algorithm produced a nearly uniform distribution (approx 5.0% per cluster) across the boundary cases. In a 384-dimensional embedding space, the Euclidean distances between data points and cluster centroids become mathematically indistinguishable for standard FCM. To keep the system lightweight without adding bloated dimensionality reduction libraries (like UMAP/PCA), this system adjusts the fuzzifier (`m=1.2`) to force sharper boundaries, though a production-grade system would benefit from a custom FCM implementation utilizing Cosine Distance.

## Part 3: The Semantic Cache (First Principles)
* [cite_start]**Architecture:** The cache is built entirely from first principles using standard Python dictionaries, strictly avoiding Redis or caching middleware[cite: 10, 38]. 
* **Cache Lookup Efficiency:** A naive semantic cache compares a new query to *every* stored query, resulting in O(N) lookup time. [cite_start]To solve this, **the cache is partitioned by the dominant cluster**[cite: 33, 34]. When a query comes in, it is assigned a dominant cluster, and the cache *only* searches for similarities within that specific partition. This reduces lookup time to O(K), ensuring high efficiency as the cache grows large.
* [cite_start]**The Tunable Decision:** The core tunable parameter is the **Cosine Similarity Threshold** (currently set to `0.85`)[cite: 35]. 
    * A **high threshold (e.g., 0.90+)** optimizes for precision. The system rarely returns the wrong cached answer, but the hit rate drops because subtle phrasing variations are missed.
    * A **low threshold (e.g., 0.70)** optimizes for recall and compute savings. [cite_start]However, it risks returning a cached answer for a query that had a distinctly different semantic nuance[cite: 36, 37]. This explicit value determines the system's strictness.

## Part 4: FastAPI Service
[cite_start]The service is fully implemented with state management encapsulated in a `SemanticSearchSystem` class[cite: 11]. [cite_start]The following endpoints are live[cite: 40]:
* [cite_start]`POST /query`: Embeds the query, checks the partitioned semantic cache, and returns hits/misses[cite: 41, 42].
* [cite_start]`GET /cache/stats`: Returns current cache state and hit rates[cite: 53, 54].
* [cite_start]`DELETE /cache`: Flushes the cache entirely and resets all tracking stats[cite: 62, 63].