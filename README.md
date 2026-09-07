# MiniVecDB

A lightweight vector search engine built from scratch using Python and NumPy.

MiniVecDB implements both exact Brute-Force vector search and an approximate IVF-Flat (Inverted File - Flat) index without using external vector database or nearest-neighbor indexing libraries.

The project is designed to demonstrate how vector similarity search works internally and how approximate indexing can significantly reduce search latency while maintaining high recall.

---

## Features

- **50,000 vectors**
- **384-dimensional vector space**
- Exact Brute-Force similarity search
- IVF-Flat approximate nearest-neighbor search
- K-Means clustering implemented from scratch using NumPy
- Configurable `nprobe`
- Top-K similarity search
- Insert vectors
- Delete vectors
- FastAPI REST API
- React + Vite frontend
- Sentence Transformers embeddings
- Benchmarking and Recall@10 evaluation
- Nprobe latency vs recall experiment

---

## Architecture

```text
                         MiniVecDB
                             |
              +--------------+--------------+
              |                             |
          FastAPI Backend              React Frontend
              |
       +------+-------+
       |              |
   Embedding       Vector Store
    Service            |
       |        +------+------+
       |        |             |
       |    Brute Force    IVF-Flat
       |        |             |
       |     Exact        Approximate
       |     Search         Search
       |                       |
       |                   K-Means
       |                  100 Clusters
       |                       |
       |                  Inverted Lists
       |
   384D Embeddings
```

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python |
| **API** | FastAPI |
| **Numerical Computing** | NumPy |
| **Embeddings** | Sentence Transformers |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Frontend** | React + Vite |
| **Styling** | Tailwind CSS |
| **Version Control** | Git + GitHub |

> *No FAISS, Chroma, Pinecone, sklearn.neighbors, or other vector indexing libraries are used for the core search implementation.*

---

## How Vector Search Works

Each document is converted into a 384-dimensional embedding vector. For a query:

```text
User Query
    |
    v
Embedding Model
    |
    v
384D Query Vector
    |
    +----------------------+
    |                      |
    v                      v
Brute Force            IVF-Flat
    |                      |
Compare with           Find nearest
all vectors             clusters
    |                      |
    |                 Search selected
    |                    clusters
    |                      |
    +----------+-----------+
               |
               v
          Top-K Results
```

### Brute-Force Search
The Brute-Force index compares the query vector against every active vector. For 50,000 vectors:

```text
Query
  |
  +--> Vector 1
  +--> Vector 2
  +--> Vector 3
  ...
  +--> Vector 50,000
```

This provides the exact nearest-neighbor result and is therefore used as the ground truth for evaluating IVF-Flat. Similarity is calculated using normalized vectors and cosine similarity.

**Advantages:**
- Exact results
- Simple implementation
- No training required

**Disadvantages:**
- Search cost grows with the number of vectors
- Every query examines the complete dataset

### IVF-Flat
IVF-Flat reduces the number of vectors examined for every query. The implementation uses K-Means clustering implemented directly with NumPy. 

The 50,000 vectors are divided into **100 clusters**. Each vector is assigned to one cluster.

During search:
```text
Query Vector
     |
     v
Compare with 100 centroids
     |
     v
Select nearest Nprobe clusters
     |
     v
Search vectors inside selected clusters
     |
     v
Return Top-K results
```
The original vectors are stored without compression, hence the name **IVF-Flat**.

### Nprobe
`nprobe` controls how many IVF clusters are searched for each query.
- A smaller value searches fewer vectors and provides lower latency.
- A larger value searches more clusters and generally improves recall.

```text
Nprobe increases
       |
       v
More clusters searched
       |
       v
More candidates examined
       |
       v
Higher recall
       |
       v
Higher latency
```

---

## Benchmark

The system was evaluated using:
- **Vectors:** 50,000
- **Dimensions:** 384
- **Queries:** 500
- **Top-K:** 10
- **IVF Clusters:** 100
- **Nprobe:** 5

### Exact vs Approximate Search

| Method | Avg Latency |
| :--- | :--- |
| Brute Force | 91.8066 ms/query |
| IVF-Flat | 5.9007 ms/query |

**Results:**
- **Speedup:** 15.56x
- **Recall@10:** 99.54%

*IVF-Flat achieved approximately 15.56× lower search latency than the exact brute-force baseline while maintaining 99.54% Recall@10 under the evaluated configuration.*

### Nprobe Trade-off
The system was also evaluated with different nprobe values.

| Nprobe | Avg Latency | Recall@10 |
| :--- | :--- | :--- |
| 1 | 3.46 ms | 92.78% |
| 2 | 6.13 ms | 97.58% |
| 5 | 14.56 ms | 99.44% |
| 10 | 23.06 ms | 100.00% |
| 20 | 30.90 ms | 100.00% |

*The experiment demonstrates the fundamental IVF trade-off between search latency and recall.*

---

## Project Structure

```text
MiniVecDB/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── brute_force.py
│   │   ├── embeddings.py
│   │   ├── evaluation.py
│   │   ├── ivf_flat.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── vector_store.py
│   │
│   ├── data/
│   │   ├── vectors.npy
│   │   ├── queries.npy
│   │   ├── documents.json
│   │   └── benchmark_results.json
│   │
│   ├── generate_data.py
│   ├── run_evaluation.py
│   ├── test_embeddings.py
│   ├── test_brute_force.py
│   ├── test_ivf.py
│   ├── test_nprobe.py
│   └── requirements.txt
│
└── frontend/
    ├── src/
    ├── public/
    ├── package.json
    ├── vite.config.js
    └── index.html
```

---

## Installation

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate it on Windows:
   ```bash
   . env\Scripts ctivate
   ```
   *(On macOS/Linux: `source venv/bin/activate`)*
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Generate Dataset

The project uses 50,000 vectors, 384 dimensions, and 500 evaluation queries. To generate the dataset:
```bash
python generate_data.py
```
The generated files are stored inside `backend/data/`.

### Run Backend

From the backend directory:
```bash
uvicorn app.main:app --reload
```
- The API will be available at: `http://127.0.0.1:8000`
- Swagger API documentation: `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | API health check |
| **GET** | `/stats` | Vector database statistics |
| **POST** | `/vectors` | Insert a vector |
| **POST** | `/search` | Search vectors |
| **DELETE** | `/vectors/{id}` | Delete a vector |
| **POST** | `/benchmark` | Run benchmark |
| **GET** | `/benchmark/nprobe` | Nprobe trade-off benchmark |

### Search Request

**Example IVF-Flat search:**
```json
{
  "text": "Explain object oriented programming in Java",
  "k": 10,
  "method": "ivf",
  "nprobe": 5
}
```

**Example brute-force search:**
```json
{
  "text": "Explain object oriented programming in Java",
  "k": 10,
  "method": "brute",
  "nprobe": 5
}
```

---

## Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at: `http://localhost:5173`. 
The interface allows users to:
- Enter a natural-language query
- Select Brute-Force or IVF-Flat
- Configure Top-K and Nprobe
- View similarity scores and vector IDs
- Compare search results

---

## Evaluation

- The main benchmark can be executed using:
  ```bash
  python run_evaluation.py
  ```
  *The results are saved to `data/benchmark_results.json`.*
- The Nprobe experiment can be executed using:
  ```bash
  python test_nprobe.py
  ```

---

## Design Decisions

- **Why NumPy?** NumPy provides efficient vectorized numerical operations while allowing the core indexing algorithms to be implemented manually.
- **Why Brute Force?** Brute Force provides exact nearest-neighbor results and acts as the ground-truth baseline for Recall@10.
- **Why IVF-Flat?** IVF-Flat demonstrates how approximate nearest-neighbor search can reduce the amount of computation required for each query.
- **Why Nprobe?** Nprobe provides a tunable accuracy-latency trade-off.
- **Why 384 dimensions?** The `all-MiniLM-L6-v2` embedding model produces 384-dimensional embeddings, making it suitable for the required vector-search workload.

---

## Limitations

This project is an educational implementation of a vector search engine. Current limitations include:
- In-memory index storage
- No persistent index serialization
- K-Means training is performed when building the IVF index
- Single-process architecture
- No distributed indexing
- No concurrent write management
- No advanced quantization techniques

*These limitations are intentional to keep the implementation focused on understanding vector search and IVF-Flat indexing from first principles.*

---

## Future Improvements

- Persistent index serialization
- Incremental IVF retraining
- Parallel query processing
- Better cluster initialization
- Product Quantization
- Hierarchical clustering
- Batch search optimization
- Distributed vector search
- Metadata filtering
- Improved result diversification

---

## Author

**Sahil Malaiya**  
B.Tech Computer Science & Engineering  
Medi-Caps University, Indore