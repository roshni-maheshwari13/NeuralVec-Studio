import time

import numpy as np

from app.brute_force import BruteForceIndex
from app.ivf_flat import IVFFlatIndex


NUM_QUERIES = 500
TOP_K = 10


def calculate_recall(exact_results, approximate_results):
    """
    Calculate Recall@K for one query.
    """

    exact_ids = {
        result["id"]
        for result in exact_results
    }

    approximate_ids = {
        result["id"]
        for result in approximate_results
    }

    if not exact_ids:
        return 0.0

    return len(
        exact_ids.intersection(approximate_ids)
    ) / len(exact_ids)


def select_queries(vectors, seed_embeddings):
    """
    Select 500 query vectors.

    We use the first 500 seed embeddings as semantic
    query vectors. These are independent of the
    50,000 indexed vectors.
    """

    if len(seed_embeddings) < NUM_QUERIES:
        raise ValueError(
            "At least 500 seed embeddings are required."
        )

    return seed_embeddings[:NUM_QUERIES].copy()


def evaluate(
    vectors,
    queries,
    n_clusters=100,
    nprobe=5
):
    """
    Evaluate Brute Force against IVF-Flat.
    """

    print("\n" + "=" * 60)
    print("BUILDING BRUTE-FORCE INDEX")
    print("=" * 60)

    brute_force = BruteForceIndex(vectors)

    print(
        f"Brute-force vectors: "
        f"{brute_force.count()}"
    )

    print("\n" + "=" * 60)
    print("BUILDING IVF-FLAT INDEX")
    print("=" * 60)

    ivf = IVFFlatIndex(
        n_clusters=n_clusters,
        nprobe=nprobe,
        max_iterations=10,
        random_seed=42
    )

    ivf.build(vectors)

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print("\nWarming up...")

    for query in queries[:5]:
        brute_force.search(query, k=TOP_K)
        ivf.search(
            query,
            k=TOP_K,
            nprobe=nprobe
        )

    # -----------------------------------------------------
    # Brute Force benchmark
    # -----------------------------------------------------

    print("\nRunning Brute-Force benchmark...")

    exact_results_all = []

    brute_start = time.perf_counter()

    for query in queries:
        results = brute_force.search(
            query,
            k=TOP_K
        )

        exact_results_all.append(results)

    brute_end = time.perf_counter()

    brute_total_ms = (
        brute_end - brute_start
    ) * 1000

    brute_average_ms = (
        brute_total_ms / len(queries)
    )

    # -----------------------------------------------------
    # IVF benchmark
    # -----------------------------------------------------

    print("Running IVF-Flat benchmark...")

    ivf_results_all = []

    ivf_start = time.perf_counter()

    for query in queries:
        results = ivf.search(
            query,
            k=TOP_K,
            nprobe=nprobe
        )

        ivf_results_all.append(results)

    ivf_end = time.perf_counter()

    ivf_total_ms = (
        ivf_end - ivf_start
    ) * 1000

    ivf_average_ms = (
        ivf_total_ms / len(queries)
    )

    # -----------------------------------------------------
    # Recall
    # -----------------------------------------------------

    recalls = []

    for exact_results, approximate_results in zip(
        exact_results_all,
        ivf_results_all
    ):
        recall = calculate_recall(
            exact_results,
            approximate_results
        )

        recalls.append(recall)

    average_recall = float(
        np.mean(recalls)
    )

    # -----------------------------------------------------
    # Speedup
    # -----------------------------------------------------

    if ivf_average_ms > 0:
        speedup = (
            brute_average_ms /
            ivf_average_ms
        )
    else:
        speedup = 0.0

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    results = {
        "vectors": len(vectors),
        "dimension": vectors.shape[1],
        "queries": len(queries),
        "top_k": TOP_K,
        "n_clusters": n_clusters,
        "nprobe": nprobe,
        "brute_force_total_ms": brute_total_ms,
        "brute_force_average_ms": brute_average_ms,
        "ivf_total_ms": ivf_total_ms,
        "ivf_average_ms": ivf_average_ms,
        "speedup": speedup,
        "recall_at_10": average_recall,
    }

    return results


def print_results(results):
    print("\n")
    print("=" * 60)
    print("VECTOR SEARCH BENCHMARK")
    print("=" * 60)

    print(
        f"Vectors              : "
        f"{results['vectors']}"
    )

    print(
        f"Dimensions           : "
        f"{results['dimension']}"
    )

    print(
        f"Queries              : "
        f"{results['queries']}"
    )

    print(
        f"Top-K                : "
        f"{results['top_k']}"
    )

    print(
        f"IVF clusters         : "
        f"{results['n_clusters']}"
    )

    print(
        f"nprobe               : "
        f"{results['nprobe']}"
    )

    print("\n" + "-" * 60)

    print(
        f"Brute Force total    : "
        f"{results['brute_force_total_ms']:.2f} ms"
    )

    print(
        f"Brute Force average  : "
        f"{results['brute_force_average_ms']:.4f} ms/query"
    )

    print("\n" + "-" * 60)

    print(
        f"IVF-Flat total       : "
        f"{results['ivf_total_ms']:.2f} ms"
    )

    print(
        f"IVF-Flat average     : "
        f"{results['ivf_average_ms']:.4f} ms/query"
    )

    print("\n" + "-" * 60)

    print(
        f"Speedup              : "
        f"{results['speedup']:.2f}x"
    )

    print(
        f"Recall@10            : "
        f"{results['recall_at_10'] * 100:.2f}%"
    )

    print("=" * 60)