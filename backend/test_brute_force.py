import numpy as np

from app.brute_force import BruteForceIndex


def main():
    print("=" * 60)
    print("Testing Brute-Force Vector Index")
    print("=" * 60)

    # Load our actual 50,000-vector dataset.
    vectors = np.load("data/vectors.npy")

    print(f"Loaded vectors: {vectors.shape}")

    # Build the index.
    index = BruteForceIndex(vectors)

    print(f"Index count: {index.count()}")
    print(f"Dimension: {index.dimension()}")

    # Use one existing vector as a query.
    query = vectors[0]

    # Search for top 10.
    results = index.search(query, k=10)

    print("\nTop 10 results:")

    for result in results:
        print(
            f"ID: {result['id']:6d} | "
            f"Score: {result['score']:.6f}"
        )

    # The query vector itself should be the best match.
    print("\nBest match ID:", results[0]["id"])

    # Test delete.
    print("\nDeleting vector ID 0...")

    deleted = index.delete(0)

    print("Delete successful:", deleted)
    print("Active vectors:", index.count())

    # Search again.
    results_after_delete = index.search(
        query,
        k=10
    )

    print("\nTop result after deleting ID 0:")

    print(
        f"ID: {results_after_delete[0]['id']} | "
        f"Score: {results_after_delete[0]['score']:.6f}"
    )

    # ID 0 must not appear.
    assert results_after_delete[0]["id"] != 0

    print("\n" + "=" * 60)
    print("BRUTE-FORCE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()