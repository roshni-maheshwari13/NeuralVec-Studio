import numpy as np


class IVFFlatIndex:
    """
    IVF-Flat vector index implemented from scratch using NumPy.

    IVF:
        Vectors are divided into clusters.

    Flat:
        Vectors inside selected clusters are searched
        using exact cosine similarity.
    """

    def __init__(
        self,
        n_clusters=100,
        nprobe=5,
        max_iterations=20,
        random_seed=42
    ):
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.max_iterations = max_iterations
        self.random_seed = random_seed

        self.vectors = None
        self.centroids = None
        self.assignments = None
        self.inverted_lists = None
        self.active = None

    # ---------------------------------------------------------
    # K-Means
    # ---------------------------------------------------------

    def _initialize_centroids(self, vectors):
        rng = np.random.default_rng(self.random_seed)

        indices = rng.choice(
            len(vectors),
            size=self.n_clusters,
            replace=False
        )

        return vectors[indices].copy()

    def _assign_clusters(self, vectors, centroids):
        """
        Assign every vector to its nearest centroid
        using cosine similarity.
        """

        vector_norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        centroid_norms = np.linalg.norm(
            centroids,
            axis=1,
            keepdims=True
        )

        normalized_vectors = vectors / np.maximum(
            vector_norms,
            1e-12
        )

        normalized_centroids = centroids / np.maximum(
            centroid_norms,
            1e-12
        )

        # Shape:
        # vectors:   N x D
        # centroids: C x D
        #
        # Result:
        # N x C
        similarities = normalized_vectors @ normalized_centroids.T

        return np.argmax(similarities, axis=1)

    def _update_centroids(self, vectors, assignments):
        dimension = vectors.shape[1]

        centroids = np.zeros(
            (self.n_clusters, dimension),
            dtype=np.float32
        )

        rng = np.random.default_rng(self.random_seed)

        for cluster_id in range(self.n_clusters):

            members = vectors[
                assignments == cluster_id
            ]

            if len(members) == 0:
                # Reinitialize empty cluster.
                random_index = rng.integers(
                    0,
                    len(vectors)
                )

                centroids[cluster_id] = vectors[
                    random_index
                ]

            else:
                centroid = np.mean(
                    members,
                    axis=0
                )

                norm = np.linalg.norm(centroid)

                if norm > 0:
                    centroid = centroid / norm

                centroids[cluster_id] = centroid

        return centroids

    def _train_kmeans(self, vectors):
        """
        Train centroids using K-Means implemented from scratch.
        """

        centroids = self._initialize_centroids(vectors)

        for iteration in range(self.max_iterations):

            assignments = self._assign_clusters(
                vectors,
                centroids
            )

            new_centroids = self._update_centroids(
                vectors,
                assignments
            )

            movement = np.linalg.norm(
                new_centroids - centroids
            )

            centroids = new_centroids

            print(
                f"K-Means iteration "
                f"{iteration + 1}/{self.max_iterations} "
                f"| centroid movement: {movement:.6f}"
            )

            if movement < 1e-4:
                print("K-Means converged.")
                break

        return centroids

    # ---------------------------------------------------------
    # Build IVF index
    # ---------------------------------------------------------

    def build(self, vectors):
        vectors = np.asarray(
            vectors,
            dtype=np.float32
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D array."
            )

        if len(vectors) < self.n_clusters:
            raise ValueError(
                "Number of vectors must be >= number of clusters."
            )

        self.vectors = vectors.copy()

        self.active = np.ones(
            len(vectors),
            dtype=bool
        )

        print(
            f"Training IVF-Flat with "
            f"{len(vectors)} vectors..."
        )

        self.centroids = self._train_kmeans(
            self.vectors
        )

        self.assignments = self._assign_clusters(
            self.vectors,
            self.centroids
        )

        self._build_inverted_lists()

        print(
            f"IVF-Flat index built with "
            f"{self.n_clusters} clusters."
        )

    def _build_inverted_lists(self):
        self.inverted_lists = {
            cluster_id: []
            for cluster_id in range(self.n_clusters)
        }

        for vector_id, cluster_id in enumerate(
            self.assignments
        ):
            self.inverted_lists[
                int(cluster_id)
            ].append(vector_id)

    # ---------------------------------------------------------
    # Similarity
    # ---------------------------------------------------------

    def _cosine_similarity(self, query, vectors):
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            raise ValueError(
                "Query vector cannot be zero."
            )

        vector_norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        normalized_vectors = vectors / np.maximum(
            vector_norms,
            1e-12
        )

        normalized_query = query / query_norm

        return normalized_vectors @ normalized_query

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(self, query, k=10, nprobe=None):
        if self.vectors is None:
            raise RuntimeError(
                "Index has not been built."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        query = np.asarray(
            query,
            dtype=np.float32
        )

        if query.ndim != 1:
            raise ValueError(
                "Query must be a 1D vector."
            )

        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Query dimension must be "
                f"{self.vectors.shape[1]}."
            )

        if nprobe is None:
            nprobe = self.nprobe

        nprobe = max(
            1,
            min(nprobe, self.n_clusters)
        )

        # ---------------------------------------------
        # Find nearest centroids
        # ---------------------------------------------

        centroid_scores = self._cosine_similarity(
            query,
            self.centroids
        )

        nearest_clusters = np.argpartition(
            -centroid_scores,
            nprobe - 1
        )[:nprobe]

        # ---------------------------------------------
        # Collect candidate vectors
        # ---------------------------------------------

        candidate_ids = []

        for cluster_id in nearest_clusters:

            ids = self.inverted_lists[
                int(cluster_id)
            ]

            candidate_ids.extend(ids)

        if not candidate_ids:
            return []

        candidate_ids = np.asarray(
            candidate_ids,
            dtype=np.int64
        )

        # Remove deleted vectors.
        candidate_ids = candidate_ids[
            self.active[candidate_ids]
        ]

        if len(candidate_ids) == 0:
            return []

        # ---------------------------------------------
        # Exact search inside selected clusters
        # ---------------------------------------------

        candidate_vectors = self.vectors[
            candidate_ids
        ]

        scores = self._cosine_similarity(
            query,
            candidate_vectors
        )

        k = min(
            k,
            len(scores)
        )

        top_indices = np.argpartition(
            -scores,
            k - 1
        )[:k]

        top_indices = top_indices[
            np.argsort(
                -scores[top_indices]
            )
        ]

        results = []

        for index in top_indices:

            vector_id = int(
                candidate_ids[index]
            )

            results.append(
                {
                    "id": vector_id,
                    "score": float(
                        scores[index]
                    )
                }
            )

        return results

    # ---------------------------------------------------------
    # Insert
    # ---------------------------------------------------------

    def add(self, vector):
        if self.vectors is None:
            raise RuntimeError(
                "Index has not been built."
            )

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        if vector.ndim != 1:
            raise ValueError(
                "Vector must be 1-dimensional."
            )

        if vector.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Vector dimension must be "
                f"{self.vectors.shape[1]}."
            )

        # Normalize new vector.
        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError(
                "Vector cannot be zero."
            )

        vector = vector / norm

        vector_id = len(self.vectors)

        self.vectors = np.vstack(
            [
                self.vectors,
                vector.reshape(1, -1)
            ]
        )

        self.active = np.append(
            self.active,
            True
        )

        # Find nearest centroid.
        similarities = self._cosine_similarity(
            vector,
            self.centroids
        )

        cluster_id = int(
            np.argmax(similarities)
        )

        self.inverted_lists[
            cluster_id
        ].append(vector_id)

        self.assignments = np.append(
            self.assignments,
            cluster_id
        )

        return vector_id

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(self, vector_id):
        if vector_id < 0 or vector_id >= len(
            self.vectors
        ):
            raise IndexError(
                "Vector ID does not exist."
            )

        if not self.active[vector_id]:
            return False

        self.active[vector_id] = False

        return True

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def count(self):
        return int(
            np.sum(self.active)
        )

    def total_count(self):
        return len(self.vectors)

    def dimension(self):
        if self.vectors is None:
            return 0

        return self.vectors.shape[1]

    def cluster_sizes(self):
        return {
            cluster_id: len(ids)
            for cluster_id, ids
            in self.inverted_lists.items()
        }