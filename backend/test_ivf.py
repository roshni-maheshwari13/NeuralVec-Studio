import numpy as np

from app.ivf_flat import IVFFlatIndex


def main():
    print("=" * 60)
    print("Testing IVF-Flat Vector Index")
    print("=" * 60)

    # Load our 50,000 vectors.
    vectors = np.load("data/vectors.npy")

    print(f"Loaded vectors: {vectors.shape}")

    # Create IVF index.
    index = IVFFlatIndex(
        n_clusters=100,
        nprobe=5,
        max_iterations=10,
        random_seed=42
    )

    # Build index.
    index.build(vectors)

    print("\nIndex statistics:")
    print("Total vectors:", index.total_count())
    print("Active vectors:", index.count())
    print("Dimensions:", index.dimension())
    print("Clusters:", index.n_clusters)

    # Query using vector 0.
    query = vectors[0]

    results = index.search(
        query,
        k=10
    )

    print("\nTop 10 IVF-Flat results:")

    for result in results:
        print(
            f"ID: {result['id']:6d} | "
            f"Score: {result['score']:.6f}"
        )

    # Because IVF is approximate, vector 0 should
    # normally be in a nearby cluster and should be
    # retrieved with reasonable nprobe.
    found = any(
        result["id"] == 0
        for result in results
    )

    print("\nVector 0 found:", found)

    # Test delete.
    print("\nDeleting vector 0...")

    deleted = index.delete(0)

    print("Delete successful:", deleted)
    print("Active vectors:", index.count())

    results_after_delete = index.search(
        query,
        k=10
    )

    found_after_delete = any(
        result["id"] == 0
        for result in results_after_delete
    )

    print(
        "Vector 0 found after delete:",
        found_after_delete
    )

    assert not found_after_delete

    print("\n" + "=" * 60)
    print("IVF-FLAT TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()