# NeuralVec Studio

A custom-built, full-stack vector search platform engineered for high-dimensional semantic retrieval, real-time performance analytics, and dynamic index experimentation.

---

## Technical Overview

NeuralVec Studio demonstrates the core mechanics of vector similarity engines without relying on external vector databases (such as FAISS or Pinecone). It implements both exact ground-truth retrieval and an approximate nearest-neighbor (ANN) index built directly on NumPy.

* **Vector Dimension**: 384D dense embeddings generated using `all-MiniLM-L6-v2`.
* **Dataset Scale**: 50,000 vectors pre-indexed in memory.
* **Indexing Mechanism**: Inverted File (IVF-Flat) with custom K-Means partitioning (100 centroids).
* **Execution Engines**: Dual-mode querying (Exact Brute-Force vs. Approximate IVF-Flat).
* **Control Parameter**: Configurable `nprobe` for real-time accuracy vs. latency trade-off evaluation.

---

## Performance Metrics

Evaluation conducted across 500 test queries against a 50,000-vector dataset (Top-K = 10):

### Search Performance Benchmark

| Strategy | Index Type | Average Latency | Recall@10 | Speed Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Brute-Force** | Linear Scan | 91.80 ms | 100.00% | 1.0x (Baseline) |
| **IVF-Flat** | Centroid Partition (`nprobe=5`) | **5.90 ms** | **99.54%** | **~15.5x Faster** |

### Latency vs. Recall Trade-off (`nprobe` Sweep)

| Clusters Searched (`nprobe`) | Query Latency | Recall@10 Accuracy |
| :---: | :---: | :---: |
| **1** | 3.46 ms | 92.78% |
| **2** | 6.13 ms | 97.58% |
| **5** | 14.56 ms | 99.44% |
| **10** | 23.06 ms | 100.00% |
| **20** | 30.90 ms | 100.00% |

---

## Architecture & Data Flow

```text
[ User Interface (React / Vite) ]
               │
               ▼
[ FastAPI Backend Engine ]
               │
       ┌───────┴───────┐
       ▼               ▼
 [ Embedding ]  [ Vector Store ]
 (384D Space)          │
               ┌───────┴───────┐
               ▼               ▼
        [ Brute-Force ]   [ IVF-Flat ]
          (Exact Scan)    (K-Means 100)

Repository Setup
Backend Service

cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
uvicorn app.main:app --reload

Frontend Application

cd frontend
npm install
npm run dev

API References
GET /health - Service health status check

GET /stats - Current vector count and index metadata

POST /search - Execute similarity query (method: brute | ivf)

POST /vectors - Ingest new vector records

DELETE /vectors/{id} - Remove vector record by ID

Developer
Roshni Maheshwari
B.Tech Computer Science & Engineering
Medi-Caps University, Indore