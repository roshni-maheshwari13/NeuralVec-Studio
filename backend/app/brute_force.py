import numpy as np


class BruteForceIndex:
    """
    Exact brute-force vector index.

    Every search compares the query against all active vectors.
    NumPy is used for the vector arithmetic.
    """

    def __init__(self, vectors=None):
        self.vectors = np.empty((0, 384), dtype=np.float32)
        self.active = np.empty(0, dtype=bool)

        if vectors is not None:
            self.build(vectors)

    def build(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array.")

        if len(vectors) == 0:
            raise ValueError("At least one vector is required.")

        self.vectors = vectors.copy()

        # All vectors are active initially.
        self.active = np.ones(len(vectors), dtype=bool)

    def _validate_query(self, query):
        query = np.asarray(query, dtype=np.float32)

        if query.ndim != 1:
            raise ValueError("Query must be a 1D vector.")

        if len(self.vectors) > 0 and query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Query dimension must be {self.vectors.shape[1]}."
            )

        return query

    def _cosine_similarity(self, query, vectors):
        """
        Calculate cosine similarity between one query
        and multiple vectors.

        The vectors are normalized during this calculation,
        so this works even if the input vectors are not normalized.
        """

        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError("Query vector cannot be zero.")

        vector_norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        vector_norms = np.maximum(vector_norms, 1e-12)

        normalized_vectors = vectors / vector_norms
        normalized_query = query / query_norm

        return normalized_vectors @ normalized_query

    def search(self, query, k=10):
        """
        Exact top-k nearest-neighbour search.

        Returns:
            list of dictionaries containing vector ID and score.
        """

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not np.any(self.active):
            return []

        query = self._validate_query(query)

        active_indices = np.flatnonzero(self.active)

        active_vectors = self.vectors[active_indices]

        scores = self._cosine_similarity(
            query,
            active_vectors
        )

        k = min(k, len(scores))

        # Get the indices of the k largest scores.
        top_indices = np.argpartition(
            -scores,
            k - 1
        )[:k]

        # Sort the selected results by similarity.
        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        results = []

        for index in top_indices:
            vector_id = int(active_indices[index])
            score = float(scores[index])

            results.append(
                {
                    "id": vector_id,
                    "score": score
                }
            )

        return results

    def add(self, vector):
        """
        Add a new vector and return its ID.
        """

        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("Vector must be 1-dimensional.")

        if len(self.vectors) > 0:
            if vector.shape[0] != self.vectors.shape[1]:
                raise ValueError(
                    f"Vector dimension must be {self.vectors.shape[1]}."
                )

        vector = vector.reshape(1, -1)

        self.vectors = np.vstack(
            [self.vectors, vector]
        )

        self.active = np.append(
            self.active,
            True
        )

        return len(self.vectors) - 1

    def delete(self, vector_id):
        """
        Mark a vector as deleted.

        We don't physically remove it from the NumPy array.
        This keeps vector IDs stable.
        """

        if vector_id < 0 or vector_id >= len(self.vectors):
            raise IndexError("Vector ID does not exist.")

        if not self.active[vector_id]:
            return False

        self.active[vector_id] = False

        return True

    def count(self):
        """
        Return the number of active vectors.
        """

        return int(np.sum(self.active))

    def total_count(self):
        """
        Return total stored vectors including deleted ones.
        """

        return len(self.vectors)

    def dimension(self):
        """
        Return vector dimension.
        """

        if len(self.vectors) == 0:
            return 0

        return self.vectors.shape[1]