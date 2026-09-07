from app.embeddings import EmbeddingService


def main():
    service = EmbeddingService()

    texts = [
        "Java is an object oriented programming language.",
        "Python is commonly used for machine learning.",
        "Spring Boot is used to build Java backend applications."
    ]

    embeddings = service.encode(texts)

    print("Number of texts:", len(texts))
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", service.dimension())
    print("First vector:", embeddings[0])


if __name__ == "__main__":
    main()