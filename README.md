# Trademarkia AI & ML Engineer Task - Lightweight Semantic Search System

## Overview
This repository contains a lightweight semantic search system built from first principles. It features fuzzy clustering, a custom semantic cache partitioned for efficiency, and a FastAPI service.

### Quick Start
To start the service:
1. Ensure your virtual environment is active.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn main:app --reload`
4. The API will be available at `http://127.0.0.1:8000`.

---

## Part 1: Embedding & Preprocessing
* **Data Cleaning Justification:** The 20 Newsgroups dataset contains heavy metadata noise (email headers, routing paths, footers). I explicitly stripped headers by splitting the text on double newlines (`\n\n`). This ensures the embedding model groups documents based on actual semantic content rather than metadata.
* **Embedding Model:** I selected `all-MiniLM-L6-v2`. It is highly optimized for CPU inference, taking up minimal memory while maintaining high semantic accuracy, perfectly fitting the "lightweight" requirement.

## Part 2: Fuzzy Clustering & Semantic Structure
* **Algorithm Choice:** Hard cluster assignments were not acceptable for this task. I used **Fuzzy C-Means (FCM)** to generate a probability distribution matrix for each document, allowing a single text to belong to multiple categories simultaneously.
* **Cluster Count:** I selected `20` clusters. While real semantic boundaries are fluid, anchoring the `c` parameter to the dataset's 20 original labeled categories provides a logical baseline, allowing the fuzzy membership to handle overlapping topics.

### Cluster Analysis & Boundary Evidence
* **Core Document Example:** Technical documents regarding "Macintosh hardware" or "RAM" show high membership in a single cluster.
* **Boundary Document Example:** Posts discussing "Sports and Law" or "Medical Ethics" show split membership across multiple clusters, proving the fuzzy logic is working.

## Part 3: The Semantic Cache (First Principles)
* **Architecture:** The cache is built entirely from first principles using standard Python dictionaries, strictly avoiding external caching middleware. 
* **Cache Lookup Efficiency:** A naive semantic cache compares a new query to *every* stored query (O(N)). To optimize this, **the cache is partitioned by the dominant cluster**. When a query comes in, it is assigned a dominant cluster, and the system *only* searches for similarities within that specific partition. This reduces lookup time to O(K).
* **Tunable Parameter:** The **Cosine Similarity Threshold** is set to `0.85`. This balances precision (avoiding false cache hits) with recall (allowing for slight phrasing variations).

## Part 4: FastAPI Service & Deployment
The service is implemented with state management encapsulated in a `SemanticSearchSystem` class.
* `POST /query`: Embeds the query, checks the partitioned semantic cache, and returns hits/misses.
* `GET /cache/stats`: Returns current cache state and hit rates.
* `DELETE /cache`: Flushes the cache and resets all tracking stats.
* **Bonus - Docker:** A `Dockerfile` is included to demonstrate containerization readiness, utilizing a `python:3.10-slim` base and a CPU-optimized build for portability.