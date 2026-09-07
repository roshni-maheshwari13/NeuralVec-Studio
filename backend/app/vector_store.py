import json
import numpy as np

from app.brute_force import BruteForceIndex
from app.ivf_flat import IVFFlatIndex
from app.embeddings import EmbeddingService


class VectorStore:

    def __init__(self):
        print("Loading embedding model...")

        self.embedding_service = EmbeddingService()

        # ----------------------------------------------------
        # Load vectors
        # ----------------------------------------------------

        print("Loading vectors...")

        self.vectors = np.load(
            "data/vectors.npy"
        ).astype(np.float32)

        # ----------------------------------------------------
        # Load documents
        # ----------------------------------------------------

        print("Loading documents...")

        with open(
            "data/documents.json",
            "r",
            encoding="utf-8"
        ) as f:
            self.documents = json.load(f)

        # ----------------------------------------------------
        # Validate dataset
        # ----------------------------------------------------

        if len(self.vectors) != len(self.documents):
            raise RuntimeError(
                f"Vector/document count mismatch: "
                f"{len(self.vectors)} vectors but "
                f"{len(self.documents)} documents."
            )

        if self.vectors.shape[1] != 384:
            raise RuntimeError(
                f"Expected 384-dimensional vectors, "
                f"got {self.vectors.shape[1]}."
            )

        print(
            f"Loaded {len(self.vectors):,} vectors "
            f"with {self.vectors.shape[1]} dimensions."
        )

        # ----------------------------------------------------
        # Create indexes
        # ----------------------------------------------------

        self.brute_force = BruteForceIndex()

        self.ivf = IVFFlatIndex(
            n_clusters=100
        )

        # ----------------------------------------------------
        # Build Brute-Force index
        # ----------------------------------------------------

        print("Building Brute-Force index...")

        self.brute_force.build(
            self.vectors
        )

        # ----------------------------------------------------
        # Build IVF-Flat index
        # ----------------------------------------------------

        print("Building IVF-Flat index...")

        self.ivf.build(
            self.vectors
        )

        print("Vector store initialized.")

    # ========================================================
    # EMBEDDING
    # ========================================================

    def encode(self, text):
        """
        Convert text into a normalized
        384-dimensional vector.
        """

        embedding = self.embedding_service.encode(
            [text]
        )[0]

        return embedding.astype(
            np.float32
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_vector,
        k=10,
        method="ivf",
        nprobe=5
    ):
        """
        Search using either:

        - brute: Exact Brute-Force search
        - ivf: Approximate IVF-Flat search

        Extra candidates are retrieved first and then
        diversified for better user-facing results.
        """

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate query dimension
        # ----------------------------------------------------

        if query_vector.shape[0] != 384:
            raise ValueError(
                f"Query vector must have 384 dimensions, "
                f"got {query_vector.shape[0]}."
            )

        # ----------------------------------------------------
        # Retrieve extra candidates
        # ----------------------------------------------------

        candidate_k = min(
            max(k * 5, 50),
            len(self.vectors)
        )

        # ----------------------------------------------------
        # Select search method
        # ----------------------------------------------------

        if method == "brute":

            results = self.brute_force.search(
                query_vector,
                candidate_k
            )

        elif method == "ivf":

            results = self.ivf.search(
                query_vector,
                candidate_k,
                nprobe
            )

        else:

            raise ValueError(
                "method must be 'brute' or 'ivf'"
            )

        # ----------------------------------------------------
        # Prepare candidates
        # ----------------------------------------------------

        candidates = []

        for result in results:

            if isinstance(result, dict):

                vector_id = int(
                    result["id"]
                )

                score = float(
                    result["score"]
                )

            else:

                vector_id = int(
                    result[0]
                )

                score = float(
                    result[1]
                )

            if (
                0 <= vector_id
                < len(self.documents)
            ):

                document = self.documents[
                    vector_id
                ]

                candidates.append({
                    "id": vector_id,
                    "score": score,
                    "document": document
                })

        # ----------------------------------------------------
        # Diversify results
        # ----------------------------------------------------

        selected = []

        used_concept_a = set()
        used_pairs = set()

        for candidate in candidates:

            document = candidate["document"]

            concept_a = document.get(
                "concept_a"
            )

            concept_b = document.get(
                "concept_b"
            )

            # Create a unique concept pair
            pair = (
                concept_a,
                concept_b
            )

            # Skip exact duplicate pairs
            if pair in used_pairs:
                continue

            # Avoid showing the same primary concept
            # repeatedly in the top results.
            if concept_a in used_concept_a:
                continue

            selected.append(candidate)

            used_concept_a.add(concept_a)
            used_pairs.add(pair)

            if len(selected) == k:
                break


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------
        # If there are not enough unique primary concepts,
        # fill remaining slots using highest-scoring candidates.

        if len(selected) < k:

            selected_ids = {
                candidate["id"]
                for candidate in selected
            }

            for candidate in candidates:

                if candidate["id"] in selected_ids:
                    continue

                selected.append(candidate)

                if len(selected) == k:
                    break

        # ----------------------------------------------------
        # Sort selected results by similarity score
        # ----------------------------------------------------
        selected.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # ----------------------------------------------------
        # Format final results
        # ----------------------------------------------------

        formatted_results = []

        for candidate in selected:

            vector_id = candidate["id"]

            score = candidate["score"]

            document = candidate["document"]

            text = document.get(
                "text",
                "No document text available"
            )

            topic = document.get(
                "topic",
                None
            )

            concept_a = document.get(
                "concept_a",
                None
            )

            concept_b = document.get(
                "concept_b",
                None
            )

            formatted_results.append({

                "id": vector_id,

                "score": round(
                    score,
                    4
                ),

                "text": text,

                "topic": topic,

                "concept_a": concept_a,

                "concept_b": concept_b
            })

        return formatted_results

    # ========================================================
    # INSERT
    # ========================================================

    def insert(
        self,
        vector
    ):
        """
        Insert the same vector into
        both indexes.
        """

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        if vector.shape[0] != 384:
            raise ValueError(
                f"Vector must have 384 dimensions, "
                f"got {vector.shape[0]}."
            )

        brute_id = self.brute_force.add(
            vector
        )

        ivf_id = self.ivf.add(
            vector
        )

        if brute_id != ivf_id:
            raise RuntimeError(
                "Index ID mismatch"
            )

        return brute_id

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        vector_id
    ):
        """
        Delete a vector from both indexes.
        """

        brute_result = self.brute_force.delete(
            vector_id
        )

        ivf_result = self.ivf.delete(
            vector_id
        )

        return (
            brute_result
            and ivf_result
        )

    # ========================================================
    # STATS
    # ========================================================

    def stats(self):
        """
        Return vector database statistics.
        """

        return {

            "total_vectors":
                self.brute_force.total_count(),

            "active_vectors":
                self.brute_force.count(),

            "dimension":
                self.brute_force.dimension(),

            "ivf_clusters":
                self.ivf.n_clusters
        }