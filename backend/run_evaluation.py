import json
import time

import numpy as np

from app.brute_force import BruteForceIndex
from app.ivf_flat import IVFFlatIndex


VECTOR_FILE = "data/vectors.npy"
QUERY_FILE = "data/queries.npy"

N_CLUSTERS = 100
NPROBE = 5
TOP_K = 10


def normalize_results(results):
    """
    Convert either tuple-style or dict-style search results
    into a list of (id, score).
    """

    normalized = []

    for result in results:

        if isinstance(result, dict):
            vector_id = int(result["id"])
            score = float(result["score"])

        else:
            vector_id = int(result[0])
            score = float(result[1])

        normalized.append((vector_id, score))

    return normalized


def calculate_recall(exact_results, approximate_results, k=10):
    """
    Recall@K =
    number of exact top-K results found by approximate search
    divided by K.
    """

    exact_ids = {
        vector_id
        for vector_id, _ in exact_results[:k]
    }

    approximate_ids = {
        vector_id
        for vector_id, _ in approximate_results[:k]
    }

    if not exact_ids:
        return 0.0

    return len(
        exact_ids.intersection(approximate_ids)
    ) / len(exact_ids)


def main():

    print("=" * 60)
    print("MiniVecDB Evaluation")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load vectors
    # ---------------------------------------------------------

    vectors = np.load(VECTOR_FILE).astype(np.float32)

    print(
        f"Loaded vectors: {vectors.shape}"
    )

    if vectors.shape != (50_000, 384):
        raise RuntimeError(
            f"Expected vectors shape (50000, 384), "
            f"got {vectors.shape}"
        )

    # ---------------------------------------------------------
    # Load evaluation queries
    # ---------------------------------------------------------

    queries = np.load(QUERY_FILE).astype(np.float32)

    print(
        f"Loaded queries: {queries.shape}"
    )

    if queries.shape != (500, 384):
        raise RuntimeError(
            f"Expected queries shape (500, 384), "
            f"got {queries.shape}"
        )

    print(
        f"\nEvaluating {len(queries)} queries"
    )

    print(
        f"Top-K    : {TOP_K}"
    )

    print(
        f"Clusters : {N_CLUSTERS}"
    )

    print(
        f"Nprobe   : {NPROBE}"
    )

    # ---------------------------------------------------------
    # Build Brute Force index
    # ---------------------------------------------------------

    print("\nBuilding Brute-Force index...")

    brute = BruteForceIndex()

    build_start = time.perf_counter()

    brute.build(vectors)

    brute_build_time = (
        time.perf_counter() - build_start
    )

    print(
        f"Brute-Force build time: "
        f"{brute_build_time:.4f} seconds"
    )

    # ---------------------------------------------------------
    # Build IVF-Flat index
    # ---------------------------------------------------------

    print("\nBuilding IVF-Flat index...")

    ivf = IVFFlatIndex(
        n_clusters=N_CLUSTERS
    )

    build_start = time.perf_counter()

    ivf.build(vectors)

    ivf_build_time = (
        time.perf_counter() - build_start
    )

    print(
        f"IVF-Flat build time: "
        f"{ivf_build_time:.4f} seconds"
    )

    # ---------------------------------------------------------
    # Evaluate Brute Force
    # ---------------------------------------------------------

    print("\nRunning Brute-Force search...")

    brute_times = []
    exact_results_all = []

    for query in queries:

        start = time.perf_counter()

        results = brute.search(
            query,
            TOP_K
        )

        elapsed = (
            time.perf_counter() - start
        )

        brute_times.append(elapsed)

        exact_results_all.append(
            normalize_results(results)
        )

    brute_total = sum(brute_times)

    brute_avg = (
        brute_total / len(queries)
    )

    print(
        f"Brute-Force total time : "
        f"{brute_total * 1000:.2f} ms"
    )

    print(
        f"Brute-Force avg/query  : "
        f"{brute_avg * 1000:.4f} ms"
    )

    # ---------------------------------------------------------
    # Evaluate IVF
    # ---------------------------------------------------------

    print("\nRunning IVF-Flat search...")

    ivf_times = []
    ivf_results_all = []

    for query in queries:

        start = time.perf_counter()

        results = ivf.search(
            query,
            TOP_K,
            NPROBE
        )

        elapsed = (
            time.perf_counter() - start
        )

        ivf_times.append(elapsed)

        ivf_results_all.append(
            normalize_results(results)
        )

    ivf_total = sum(ivf_times)

    ivf_avg = (
        ivf_total / len(queries)
    )

    print(
        f"IVF-Flat total time     : "
        f"{ivf_total * 1000:.2f} ms"
    )

    print(
        f"IVF-Flat avg/query      : "
        f"{ivf_avg * 1000:.4f} ms"
    )

    # ---------------------------------------------------------
    # Calculate Recall@10
    # ---------------------------------------------------------

    recalls = []

    for exact, approximate in zip(
        exact_results_all,
        ivf_results_all
    ):

        recall = calculate_recall(
            exact,
            approximate,
            TOP_K
        )

        recalls.append(recall)

    recall_at_10 = float(
        np.mean(recalls)
    )

    # ---------------------------------------------------------
    # Speedup
    # ---------------------------------------------------------

    if ivf_avg > 0:
        speedup = (
            brute_avg / ivf_avg
        )
    else:
        speedup = 0.0

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Vectors          : {len(vectors):,}"
    )

    print(
        f"Dimensions       : {vectors.shape[1]}"
    )

    print(
        f"Queries          : {len(queries):,}"
    )

    print(
        f"Top-K            : {TOP_K}"
    )

    print(
        f"IVF clusters     : {N_CLUSTERS}"
    )

    print(
        f"Nprobe           : {NPROBE}"
    )

    print("-" * 60)

    print(
        f"Brute Force      : "
        f"{brute_avg * 1000:.4f} ms/query"
    )

    print(
        f"IVF-Flat         : "
        f"{ivf_avg * 1000:.4f} ms/query"
    )

    print(
        f"Speedup          : "
        f"{speedup:.2f}x"
    )

    print(
        f"Recall@10        : "
        f"{recall_at_10 * 100:.2f}%"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    benchmark_results = {
        "vectors": int(len(vectors)),
        "dimensions": int(vectors.shape[1]),
        "queries": int(len(queries)),
        "top_k": TOP_K,
        "ivf_clusters": N_CLUSTERS,
        "nprobe": NPROBE,

        "brute_force": {
            "total_ms": brute_total * 1000,
            "avg_ms_per_query": brute_avg * 1000
        },

        "ivf_flat": {
            "total_ms": ivf_total * 1000,
            "avg_ms_per_query": ivf_avg * 1000
        },

        "speedup": speedup,

        "recall_at_10": recall_at_10
    }

    with open(
        "data/benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            benchmark_results,
            file,
            indent=2
        )

    print(
        "\nSaved:"
    )

    print(
        "  data/benchmark_results.json"
    )


if __name__ == "__main__":
    main()