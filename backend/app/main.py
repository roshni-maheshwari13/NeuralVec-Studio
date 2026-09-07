import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    VectorInsertRequest,
    SearchRequest
)

from app.vector_store import VectorStore
from app.benchmark import run_benchmark


app = FastAPI(
    title="MiniVecDB",
    description="Custom Vector Search Engine with Brute-Force and IVF-Flat",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


store = VectorStore()

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/stats")
def stats():
    return store.stats()


@app.post("/vectors")
def insert_vector(request: VectorInsertRequest):

    if request.vector is not None:

        vector = request.vector

    elif request.text is not None:

        vector = store.encode(request.text)

    else:

        raise HTTPException(
            status_code=400,
            detail="Provide either vector or text"
        )

    if len(vector) != store.stats()["dimension"]:
        raise HTTPException(
            status_code=400,
            detail=f"Vector must have {store.stats()['dimension']} dimensions"
        )

    vector_id = store.insert(vector)

    return {
        "message": "Vector inserted successfully",
        "id": vector_id
    }


@app.post("/search")
def search(request: SearchRequest):

    if request.vector is not None:

        query_vector = request.vector

    elif request.text is not None:

        query_vector = store.encode(request.text)

    else:

        raise HTTPException(
            status_code=400,
            detail="Provide either vector or text"
        )

    if len(query_vector) != store.stats()["dimension"]:
        raise HTTPException(
            status_code=400,
            detail=f"Query vector must have {store.stats()['dimension']} dimensions"
        )

    if request.method not in ["ivf", "brute"]:
        raise HTTPException(
            status_code=400,
            detail="method must be 'ivf' or 'brute'"
        )

    results = store.search(
        query_vector=query_vector,
        k=request.k,
        method=request.method,
        nprobe=request.nprobe
    )

    return {
        "results": results,
        "method": request.method
    }


@app.delete("/vectors/{vector_id}")
def delete_vector(vector_id: int):

    success = store.delete(vector_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Vector not found"
        )

    return {
        "message": "Vector deleted successfully",
        "id": vector_id
    }
@app.post("/benchmark")
def benchmark(
    k: int = 10,
    nprobe: int = 5
):

    queries = np.load(
        "data/queries.npy"
    ).astype(np.float32)

    results = run_benchmark(
        vectors=store.vectors,
        brute_force=store.brute_force,
        ivf=store.ivf,
        queries=queries,
        k=k,
        nprobe=nprobe
    )

    return results
@app.get("/benchmark/nprobe")
def nprobe_benchmark():

    queries = np.load(
        "data/queries.npy"
    ).astype(np.float32)

    nprobe_values = [1, 2, 5, 10, 20]

    results = []

    for nprobe in nprobe_values:

        benchmark_result = run_benchmark(
            vectors=store.vectors,
            brute_force=store.brute_force,
            ivf=store.ivf,
            queries=queries,
            k=10,
            nprobe=nprobe
        )

        results.append(benchmark_result)

    return {
        "results": results
    }