import time
import numpy as np


def run_benchmark(
    vectors,
    brute_force,
    ivf,
    queries,
    k=10,
    nprobe=5
):
    # -----------------------------
    # Ground truth using brute force
    # -----------------------------
    ground_truth = []

    start = time.perf_counter()

    for query in queries:
        results = brute_force.search(query, k)

        ids = []

        for result in results:
            if isinstance(result, dict):
                ids.append(int(result["id"]))
            else:
                ids.append(int(result[0]))

        ground_truth.append(ids)

    brute_time = time.perf_counter() - start

    # -----------------------------
    # IVF search
    # -----------------------------
    ivf_results = []

    start = time.perf_counter()

    for query in queries:
        results = ivf.search(
            query,
            k,
            nprobe
        )

        ids = []

        for result in results:
            if isinstance(result, dict):
                ids.append(int(result["id"]))
            else:
                ids.append(int(result[0]))

        ivf_results.append(ids)

    ivf_time = time.perf_counter() - start

    # -----------------------------
    # Recall@K
    # -----------------------------
    recalls = []

    for exact_ids, approx_ids in zip(
        ground_truth,
        ivf_results
    ):

        exact_set = set(exact_ids)
        approx_set = set(approx_ids)

        overlap = len(
            exact_set.intersection(approx_set)
        )

        recalls.append(
            overlap / k
        )

    recall = float(np.mean(recalls))

    # -----------------------------
    # Metrics
    # -----------------------------
    num_queries = len(queries)

    brute_avg_ms = (
        brute_time / num_queries
    ) * 1000

    ivf_avg_ms = (
        ivf_time / num_queries
    ) * 1000

    speedup = (
        brute_time / ivf_time
        if ivf_time > 0
        else 0
    )

    return {
        "vectors": len(vectors),
        "dimensions": vectors.shape[1],
        "queries": num_queries,
        "top_k": k,
        "nprobe": nprobe,
        "brute_force_total_ms": round(
            brute_time * 1000,
            2
        ),
        "brute_force_avg_ms": round(
            brute_avg_ms,
            4
        ),
        "ivf_total_ms": round(
            ivf_time * 1000,
            2
        ),
        "ivf_avg_ms": round(
            ivf_avg_ms,
            4
        ),
        "speedup": round(
            speedup,
            2
        ),
        "recall_at_k": round(
            recall,
            4
        ),
        "recall_percentage": round(
            recall * 100,
            2
        )
    }