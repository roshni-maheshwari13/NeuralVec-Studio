import time
import numpy as np

from app.brute_force import BruteForceIndex
from app.ivf_flat import IVFFlatIndex


NUM_QUERIES = 500
TOP_K = 10

NPROBE_VALUES = [1, 2, 5, 10, 20]


def recall_at_k(exact_results, approximate_results):
    exact_ids = {
        result["id"]
        for result in exact_results
    }

    approximate_ids = {
        result["id"]
        for result in approximate_results
    }

    return len(
        exact_ids.intersection(approximate_ids)
    ) / len(exact_ids)


def main():

    print("=" * 70)
    print("MiniVecDB - nprobe Trade-off Experiment")
    print("=" * 70)

    vectors = np.load(
        "data/vectors.npy"
    )

    queries = np.load(
        "data/queries.npy"
    )

    queries = queries[:NUM_QUERIES]

    print(f"Vectors : {vectors.shape}")
    print(f"Queries : {queries.shape}")

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    print("\nBuilding brute-force ground truth...")

    brute_force = BruteForceIndex(vectors)

    ground_truth = []

    start = time.perf_counter()

    for query in queries:
        ground_truth.append(
            brute_force.search(
                query,
                k=TOP_K
            )
        )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Ground truth generated in "
        f"{elapsed:.2f} ms"
    )

    # -----------------------------------------------------
    # Build IVF once
    # -----------------------------------------------------

    print("\nBuilding IVF-Flat index...")

    ivf = IVFFlatIndex(
        n_clusters=100,
        nprobe=1,
        max_iterations=10,
        random_seed=42
    )

    ivf.build(vectors)

    # -----------------------------------------------------
    # nprobe experiment
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        f"{'nprobe':<10}"
        f"{'Avg Latency (ms)':<22}"
        f"{'Total Time (ms)':<22}"
        f"{'Recall@10':<15}"
    )
    print("-" * 70)

    for nprobe in NPROBE_VALUES:

        approximate_results = []

        start = time.perf_counter()

        for query in queries:

            results = ivf.search(
                query,
                k=TOP_K,
                nprobe=nprobe
            )

            approximate_results.append(
                results
            )

        total_ms = (
            time.perf_counter() - start
        ) * 1000

        average_ms = (
            total_ms / len(queries)
        )

        recalls = []

        for exact, approximate in zip(
            ground_truth,
            approximate_results
        ):
            recalls.append(
                recall_at_k(
                    exact,
                    approximate
                )
            )

        average_recall = np.mean(
            recalls
        )

        print(
            f"{nprobe:<10}"
            f"{average_ms:<22.4f}"
            f"{total_ms:<22.2f}"
            f"{average_recall * 100:<15.2f}%"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()