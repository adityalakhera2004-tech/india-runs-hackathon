# India Runs Hackathon - Track 1: AI Semantic Talent Ranking Engine

An enterprise-grade, high-throughput hybrid retrieval pipeline designed to rank candidate profiles against technical job descriptions using state-of-the-art Natural Language Processing (NLP) models, lexical indices, and strict corporate filtering heuristics.

## 🚀 Architectural Overview

This system utilizes a **Weighted Hybrid Search (30% Lexical + 30% Structural + 40% Dense Semantic)** architecture to bypass the limitations of keyword-only parsers and standalone embedding similarity models.

1. **Deterministic Heuristic Filtering:** Pre-screens stream data to filter out mismatched job sectors (e.g., keyword-stuffed sales/marketing roles) and checks platform activity metrics to ensure recruiter outreach viability.
2. **Lexical Indexing (BM25):** Tokenizes text and indexes the candidate pool using the `BM25Okapi` algorithm to establish explicit matching scores for critical production frameworks and tools.
3. **Dense Semantic Embedding Space:** Vectorizes candidate profile text into a high-dimensional dense vector space using the `BAAI/bge-small-en-v1.5` model to capture contextual synonyms and semantic meaning.
4. **Weighted Score Fusion:** Synthesizes normalized metrics into a singular unified `hybrid_score` to output an optimized, high-fidelity top 100 selection.

## 📊 Performance and Scaling Optimizations

- **Batch Inference Vectorization:** Avoids iterative CPU/GPU context-switching overheads by collecting candidate corpora into an isolated batch operation (`model.encode(corpus)`). This slashes computational latency across high-volume datasets.
- **Bi-Encoder Architecture:** Utilizes the highly optimized `bge-small` configuration, achieving a high placement rank on the MTEB (Massive Text Embedding Benchmark) leaderboard while preserving extremely low inference latency and memory footprints.

## 💻 Tech Stack
- **Language:** Python 3
- **Deep Learning / NLP Frameworks:** Sentence-Transformers, PyTorch
- **Lexical Math Engine:** Rank-BM25
- **Pre-trained Model Stack:** BAAI / BGE-Small-En-v1.5

---
*Developed by Aditya Lakhera (Incoming Final-Year B.Tech Computer Science Student, 2027 Graduation Batch).*
