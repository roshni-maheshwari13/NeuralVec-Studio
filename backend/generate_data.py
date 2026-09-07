import json
from pathlib import Path

import numpy as np

from app.embeddings import EmbeddingService


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_VECTORS = 50_000
NUM_QUERIES = 500
BATCH_SIZE = 128

RANDOM_SEED = 42
DATA_DIR = Path("data")


# ============================================================
# TECHNICAL KNOWLEDGE
# ============================================================

TOPICS = {
    "java": [
        "Java is an object oriented programming language.",
        "Java classes encapsulate data and behavior into objects.",
        "Java objects are instances created from classes.",
        "Java inheritance allows a class to reuse behavior from another class.",
        "Java polymorphism allows different objects to respond to the same interface.",
        "Java abstraction hides unnecessary implementation details.",
        "Java interfaces define contracts that implementing classes must follow.",
        "Java abstract classes provide shared behavior for related classes.",
        "Java constructors initialize newly created objects.",
        "Java method overloading allows multiple methods with the same name.",
        "Java method overriding allows subclasses to provide specialized behavior.",
        "Java encapsulation protects object state using access modifiers.",
        "Java public members can be accessed from other classes.",
        "Java private members can only be accessed inside their declaring class.",
        "Java protected members support controlled access through inheritance.",
        "Java static members belong to the class rather than individual objects.",
        "Java final variables cannot be reassigned after initialization.",
        "Java exceptions are used to handle abnormal program conditions.",
        "Checked exceptions must generally be handled or declared.",
        "Unchecked exceptions usually represent programming errors.",
        "Java try catch blocks handle exceptions during program execution.",
        "Java finally blocks can execute cleanup logic.",
        "Java collections provide reusable data structures.",
        "ArrayList stores elements in a dynamically sized array.",
        "LinkedList stores elements using linked nodes.",
        "HashMap stores key value pairs using hashing.",
        "HashSet stores unique elements.",
        "Java generics provide compile time type safety.",
        "Java streams provide a functional approach to processing collections.",
        "Lambda expressions allow concise implementations of functional interfaces.",
        "The Java Virtual Machine executes Java bytecode.",
        "Java bytecode provides platform independence.",
        "The Java compiler converts source code into bytecode.",
        "JIT compilation can improve Java application execution performance.",
        "Garbage collection automatically manages unused Java objects.",
        "Java threads allow multiple tasks to execute concurrently.",
        "Synchronization helps protect shared data in concurrent programs.",
        "Spring Boot simplifies development of Java backend applications.",
        "Spring dependency injection manages relationships between application components.",
        "Spring Boot REST controllers expose HTTP endpoints.",
        "JPA provides an abstraction for mapping Java objects to database tables.",
    ],

    "python": [
        "Python is a high level programming language.",
        "Python uses dynamic typing for variables.",
        "Python functions encapsulate reusable pieces of logic.",
        "Python lists store ordered collections of values.",
        "Python tuples store immutable ordered collections.",
        "Python dictionaries store data using key value pairs.",
        "Python sets store unique values.",
        "Python list comprehensions provide concise collection creation.",
        "Python generators produce values lazily.",
        "Python decorators modify or extend function behavior.",
        "Python exceptions handle runtime errors.",
        "Python modules organize reusable Python code.",
        "Python packages group related modules.",
        "Python virtual environments isolate project dependencies.",
        "NumPy provides efficient numerical operations on arrays.",
        "Pandas provides data structures for data analysis.",
        "Python is widely used for machine learning applications.",
        "Python functions can accept positional arguments.",
        "Python functions can accept keyword arguments.",
        "Python supports object oriented programming through classes.",
        "Python inheritance allows classes to reuse behavior.",
        "Python FastAPI can be used to build REST APIs.",
        "Python type hints improve code readability and tooling.",
        "Python asynchronous programming can handle concurrent I/O operations.",
        "Python file handling allows programs to read and write files.",
        "Python JSON libraries support structured data exchange.",
        "Python logging helps developers monitor application behavior.",
        "Python unit testing verifies individual pieces of application logic.",
        "Python multiprocessing can execute work across processes.",
        "Python threading can help with I/O-bound tasks.",
        "Python memory management uses automatic garbage collection.",
        "Python lambda expressions create small anonymous functions.",
        "Python map applies a function across iterable values.",
        "Python filter selects values from an iterable.",
        "Python sorting supports custom comparison keys.",
        "Python command line applications can read arguments from users.",
        "Python environment variables can store configuration values.",
        "Python applications can communicate with databases.",
        "Python APIs commonly exchange information using JSON.",
        "Python applications can integrate machine learning models.",
        "Python scripts can automate repetitive development tasks.",
        "Python exception handling improves application robustness.",
    ],

    "databases": [
        "A primary key uniquely identifies a row in a relational table.",
        "A foreign key creates a relationship between database tables.",
        "Database normalization reduces unnecessary data duplication.",
        "An SQL JOIN combines related rows from multiple tables.",
        "Indexes can improve the performance of database queries.",
        "Transactions help maintain consistency in relational databases.",
        "ACID properties describe important guarantees of database transactions.",
        "SQL GROUP BY aggregates rows based on selected columns.",
        "SQL WHERE filters rows according to conditions.",
        "SQL ORDER BY sorts query results.",
        "SQL HAVING filters grouped query results.",
        "SQL INSERT adds new rows to a table.",
        "SQL UPDATE modifies existing rows.",
        "SQL DELETE removes rows from a table.",
        "Database constraints help maintain data integrity.",
        "Unique constraints prevent duplicate values.",
        "NOT NULL constraints require a value to be present.",
        "Composite keys use multiple columns to identify a row.",
        "One-to-many relationships connect one record to many records.",
        "Many-to-many relationships require an association structure.",
        "Database views represent reusable query results.",
        "Stored procedures contain reusable database logic.",
        "Database triggers execute automatically when certain events occur.",
        "Query optimization improves database execution efficiency.",
        "Database indexes use additional storage to accelerate lookups.",
        "B-tree indexes are commonly used for ordered database searches.",
        "Database transactions can be committed or rolled back.",
        "Isolation controls how transactions interact with each other.",
        "Database locking helps prevent conflicting concurrent updates.",
        "Deadlocks can occur when transactions wait for each other.",
        "MySQL is a relational database management system.",
        "MongoDB is a document oriented database.",
        "NoSQL databases can store flexible document structures.",
        "Database schemas describe the structure of stored data.",
        "Connection pooling reuses database connections.",
        "Database backups help recover from data loss.",
        "Replication maintains copies of database data.",
        "Sharding distributes database data across multiple servers.",
        "Pagination limits the amount of data returned by a query.",
        "Parameterized queries help protect applications from SQL injection.",
        "Database migrations manage schema changes over time.",
        "Read replicas can distribute database read traffic.",
    ],

    "networking": [
        "TCP provides reliable communication between applications.",
        "UDP provides connectionless communication with lower overhead.",
        "DNS translates domain names into IP addresses.",
        "HTTP is an application layer protocol used by the web.",
        "HTTPS secures HTTP communication using TLS.",
        "ARP maps an IP address to a MAC address on a local network.",
        "A router forwards packets between different networks.",
        "A switch forwards Ethernet frames within a local network.",
        "The TCP three way handshake establishes a connection.",
        "IP addresses identify devices or interfaces on networks.",
        "IPv4 addresses use a 32 bit addressing scheme.",
        "IPv6 addresses use a 128 bit addressing scheme.",
        "Subnetting divides a network into smaller logical networks.",
        "CIDR notation represents network prefixes.",
        "DHCP automatically assigns network configuration to clients.",
        "The DHCP DORA process includes discover offer request and acknowledgment.",
        "NAT translates private addresses into public addresses.",
        "A default gateway provides access to other networks.",
        "MAC addresses identify network interfaces at the data link layer.",
        "Ethernet is commonly used for local area networks.",
        "VLANs logically separate devices within switched networks.",
        "Trunk links can carry traffic from multiple VLANs.",
        "STP helps prevent switching loops.",
        "Routing tables determine where packets should be forwarded.",
        "Static routes are manually configured by administrators.",
        "Dynamic routing protocols exchange network reachability information.",
        "ICMP is used for network diagnostics and control messages.",
        "Ping commonly uses ICMP echo messages.",
        "Traceroute can help identify network paths.",
        "Firewalls filter network traffic based on configured rules.",
        "ACLs control which network traffic is permitted or denied.",
        "Load balancers distribute traffic across multiple servers.",
        "HTTP status codes communicate the result of web requests.",
        "WebSocket connections support bidirectional communication.",
        "DNS caching reduces repeated name resolution requests.",
        "Network latency represents the time required for communication.",
        "Packet loss occurs when network packets fail to reach their destination.",
        "Bandwidth represents the capacity of a network connection.",
        "TCP congestion control manages traffic during network congestion.",
        "OSI model divides networking functionality into conceptual layers.",
        "The TCP IP model describes common internet communication layers.",
        "Encapsulation adds protocol information as data moves through layers.",
    ],

    "ai": [
        "Machine learning models learn patterns from training data.",
        "An embedding represents information as a numerical vector.",
        "Semantic search finds information based on meaning rather than exact words.",
        "Large language models can generate and understand natural language.",
        "Retrieval augmented generation combines retrieval with language generation.",
        "Cosine similarity measures the angle between two vectors.",
        "Neural networks learn representations using layers of parameters.",
        "Vector search is commonly used in modern AI applications.",
        "Supervised learning uses labeled training examples.",
        "Unsupervised learning discovers patterns without labeled outputs.",
        "Classification predicts categories from input data.",
        "Regression predicts continuous numerical values.",
        "Clustering groups similar data points together.",
        "Training adjusts model parameters to reduce prediction error.",
        "Inference uses a trained model to produce predictions.",
        "Overfitting occurs when a model memorizes training data too closely.",
        "Regularization can reduce overfitting in machine learning models.",
        "Feature engineering transforms raw data into useful model inputs.",
        "Neural network weights are learned during model training.",
        "Activation functions introduce nonlinear behavior into neural networks.",
        "Transformers use attention mechanisms to process sequences.",
        "Self attention allows tokens to consider other tokens in a sequence.",
        "Tokenization converts text into smaller units for language models.",
        "Large language models are commonly trained on large text datasets.",
        "Prompt engineering structures instructions for language models.",
        "Fine tuning adapts a pretrained model to a specific task.",
        "RAG retrieves external information before generating an answer.",
        "Vector databases store numerical representations for similarity search.",
        "Nearest neighbor search finds vectors close to a query vector.",
        "Approximate nearest neighbor search trades some accuracy for speed.",
        "Exact vector search compares a query against every stored vector.",
        "Recall measures how many relevant results are retrieved.",
        "Top K search returns the K most similar results.",
        "Embeddings can represent text based on semantic meaning.",
        "Sentence transformers generate embeddings for text.",
        "AI applications can combine retrieval systems with LLMs.",
        "Model evaluation measures the quality of predictions or generated output.",
        "Inference latency measures the time required to produce a model result.",
        "Batch processing can improve model inference throughput.",
        "Normalization can improve similarity calculations between embeddings.",
        "Machine learning pipelines commonly include preprocessing training and evaluation.",
        "AI systems can use APIs to access hosted language models.",
    ],

    "algorithms": [
        "Binary search repeatedly divides a sorted search space in half.",
        "Merge sort uses a divide and conquer strategy.",
        "Hash tables provide efficient average case key lookup.",
        "Breadth first search explores a graph level by level.",
        "Depth first search explores one path before backtracking.",
        "A priority queue can be implemented using a heap.",
        "Dynamic programming solves problems using overlapping subproblems.",
        "Time complexity describes how an algorithm scales with input size.",
        "Space complexity describes additional memory used by an algorithm.",
        "Quick sort partitions data around a selected pivot.",
        "Insertion sort builds a sorted portion one element at a time.",
        "Selection sort repeatedly selects the smallest remaining element.",
        "Heap sort uses a heap data structure to order elements.",
        "A binary tree contains nodes with at most two children.",
        "A binary search tree maintains an ordering between child nodes.",
        "Tree traversal can be performed using preorder inorder or postorder.",
        "A balanced tree maintains controlled height for efficient operations.",
        "Graphs contain vertices connected by edges.",
        "Dijkstra algorithm finds shortest paths with nonnegative edge weights.",
        "Topological sorting orders vertices of a directed acyclic graph.",
        "Union find efficiently manages connected components.",
        "Greedy algorithms make locally optimal choices.",
        "Backtracking explores candidate solutions and reverses unsuccessful choices.",
        "Recursion allows a function to call itself.",
        "Divide and conquer breaks a problem into smaller subproblems.",
        "Two pointer techniques process sequences using moving indices.",
        "Sliding window techniques maintain a range over sequential data.",
        "Prefix sums allow efficient range sum calculations.",
        "Hash maps can count frequencies efficiently.",
        "Stacks support last in first out operations.",
        "Queues support first in first out operations.",
        "Deques allow insertion and removal from both ends.",
        "Heaps provide efficient access to minimum or maximum elements.",
        "Big O notation describes asymptotic algorithm growth.",
        "Amortized analysis studies average cost over a sequence of operations.",
        "Stable sorting preserves the relative order of equal elements.",
        "Graph traversal can detect connectivity between vertices.",
        "Dynamic programming often stores results of repeated subproblems.",
        "Memoization caches results of recursive computations.",
        "Tabulation builds dynamic programming results iteratively.",
        "Algorithm correctness requires that an algorithm produce valid results.",
        "Efficient algorithms reduce computation for large inputs.",
    ],

    "web": [
        "React is a JavaScript library for building user interfaces.",
        "REST APIs expose application functionality through HTTP endpoints.",
        "JSON is commonly used for exchanging structured data between services.",
        "Authentication verifies the identity of a user.",
        "Authorization determines which actions an authenticated user can perform.",
        "HTTP status code 404 indicates that a requested resource was not found.",
        "Frontend applications communicate with backend services through APIs.",
        "Responsive design allows interfaces to adapt to different screen sizes.",
        "HTTP GET requests commonly retrieve resources.",
        "HTTP POST requests commonly create resources.",
        "HTTP PUT requests commonly update resources.",
        "HTTP DELETE requests commonly remove resources.",
        "HTTP headers carry metadata between clients and servers.",
        "Cookies can store small pieces of client related information.",
        "JWT tokens can represent authenticated user information.",
        "CORS controls cross origin browser requests.",
        "Middleware can process HTTP requests before they reach handlers.",
        "REST APIs commonly use resource oriented URL structures.",
        "API pagination limits the amount of data returned at once.",
        "API validation checks whether incoming data is acceptable.",
        "Rate limiting controls how frequently clients can call an API.",
        "Caching can reduce repeated API computation.",
        "Web applications commonly use frontend and backend components.",
        "React components encapsulate reusable interface behavior.",
        "React state stores data that can change during component execution.",
        "React props pass information between components.",
        "Virtual DOM techniques help React update interfaces efficiently.",
        "JavaScript promises represent asynchronous operations.",
        "Async await provides syntax for asynchronous JavaScript code.",
        "Node.js allows JavaScript to execute on the server.",
        "Express.js provides tools for building Node.js APIs.",
        "Tailwind CSS provides utility classes for styling interfaces.",
        "Vite provides a fast development environment for frontend applications.",
        "Web accessibility improves usability for people with disabilities.",
        "Client side routing allows navigation without full page reloads.",
        "Form validation helps prevent invalid user input.",
        "HTTPS protects web communication from interception.",
        "Content security policies can reduce certain web attacks.",
        "SQL injection is a security risk caused by unsafe database queries.",
        "Cross site scripting can execute malicious scripts in web pages.",
        "API documentation helps developers understand available endpoints.",
        "Frontend applications often consume JSON responses from backend APIs.",
    ],

    "cloud": [
        "Cloud computing provides computing resources over a network.",
        "Containers package applications together with their dependencies.",
        "Horizontal scaling adds more instances to handle increased traffic.",
        "Load balancers distribute incoming requests across backend servers.",
        "Caching can reduce repeated computation and database access.",
        "Monitoring helps engineers detect application failures.",
        "Object storage is commonly used for files and large unstructured data.",
        "Distributed systems coordinate multiple independent computing nodes.",
        "Virtual machines provide isolated computing environments.",
        "Container orchestration manages groups of application containers.",
        "Kubernetes is commonly used to orchestrate containers.",
        "Cloud storage can provide durable access to application files.",
        "Auto scaling adjusts resources according to workload.",
        "Horizontal scaling increases capacity by adding instances.",
        "Vertical scaling increases the resources of an existing instance.",
        "Load balancing improves application availability and distribution.",
        "Health checks help identify unavailable application instances.",
        "Cloud monitoring collects metrics and application information.",
        "Centralized logging helps diagnose distributed application problems.",
        "Message queues allow asynchronous communication between services.",
        "Event driven systems react to events produced by other components.",
        "Microservices divide applications into independently deployable services.",
        "Service discovery helps applications locate other services.",
        "Cloud networking connects virtual machines containers and services.",
        "Identity and access management controls access to cloud resources.",
        "Secrets management protects sensitive application configuration.",
        "Infrastructure as code defines infrastructure using configuration files.",
        "Continuous deployment automates application releases.",
        "Cloud databases provide managed data storage services.",
        "Serverless computing runs application code without managing servers directly.",
        "Cloud regions provide geographically distributed infrastructure.",
        "Availability zones provide isolated infrastructure within regions.",
        "Redundancy improves resilience against infrastructure failures.",
        "Disaster recovery prepares systems for major failures.",
        "Backups protect applications against accidental data loss.",
        "CDNs distribute content closer to end users.",
        "Cloud cost optimization reduces unnecessary resource usage.",
        "Autoscaling can respond to changes in application traffic.",
        "Distributed systems require careful handling of failures.",
        "Eventual consistency allows replicas to converge over time.",
        "Cloud applications often use multiple independent services.",
        "Observability combines metrics logs and traces to understand systems.",
    ],
}


# ============================================================
# VARIATION CONTEXTS
# ============================================================

CONTEXTS = [
    "in a production application",
    "in a backend service",
    "during system design",
    "when debugging an application",
    "when preparing for a technical interview",
    "when optimizing performance",
    "when designing a scalable system",
    "when testing software",
    "when maintaining an existing codebase",
    "in a real world software project",
    "when building an API",
    "when analyzing application behavior",
    "when learning the concept",
    "when reviewing system architecture",
    "when troubleshooting a technical problem",
    "when implementing a new feature",
    "when improving reliability",
    "when working with large datasets",
    "when designing distributed systems",
    "when building an AI application",
]


PERSPECTIVES = [
    "Concept overview",
    "Implementation note",
    "Practical example",
    "Engineering perspective",
    "Performance consideration",
    "Debugging perspective",
    "Design consideration",
    "Developer takeaway",
    "Interview preparation",
    "Production consideration",
]


# ============================================================
# CREATE 50,000 DIVERSE DOCUMENTS
# ============================================================

def build_documents():
    """
    Generate exactly 50,000 technically meaningful documents.

    Each document combines two different concepts from the same domain.
    This avoids creating hundreds of near-duplicate documents for one concept.
    """

    documents = []

    templates = [
        "{a} {b} Both are important concepts when working with {topic} systems.",

        "A developer working with {topic} should understand {a} Another useful concept to study is {b}.",

        "In {topic} development, {a} It is also useful to understand {b} when building practical systems.",

        "From a {topic} engineering perspective, {a} A related area of knowledge is {b}.",

        "For {topic} interviews and practical development, {a} Developers should also be familiar with {b}.",

        "{b} is another important {topic} concept. It can be studied alongside {a}.",

        "When learning {topic}, {a} Another area worth understanding is {b}.",

        "Practical {topic} work involves many concepts, including {a} It also requires knowledge of {b}."
    ]

    target_per_topic = TOTAL_VECTORS // len(TOPICS)

    for topic, concepts in TOPICS.items():

        if len(concepts) < 2:
            raise RuntimeError(
                f"Topic '{topic}' needs at least 2 concepts."
            )

        topic_documents = []

        for i in range(len(concepts)):

            for j in range(i + 1, len(concepts)):

                a = concepts[i]
                b = concepts[j]

                for template in templates:

                    text = template.format(
                        topic=topic,
                        a=a,
                        b=b
                    )

                    topic_documents.append({
                        "vector_id": 0,
                        "topic": topic,
                        "concept_a": a,
                        "concept_b": b,
                        "text": text
                    })

                    if len(topic_documents) >= target_per_topic:
                        break

                if len(topic_documents) >= target_per_topic:
                    break

            if len(topic_documents) >= target_per_topic:
                break

        if len(topic_documents) != target_per_topic:
            raise RuntimeError(
                f"Could not generate {target_per_topic} documents "
                f"for topic '{topic}'. Generated {len(topic_documents)}."
            )

        documents.extend(topic_documents)

    # Assign final vector IDs
    for idx, document in enumerate(documents):
        document["vector_id"] = idx

    # Final validation
    if len(documents) != TOTAL_VECTORS:
        raise RuntimeError(
            f"Expected {TOTAL_VECTORS} documents, "
            f"generated {len(documents)}."
        )

    unique_texts = {doc["text"] for doc in documents}

    if len(unique_texts) != len(documents):
        raise RuntimeError(
            f"Duplicate documents detected: "
            f"{len(documents) - len(unique_texts)} duplicates."
        )

    print(f"Generated {len(documents):,} unique documents.")

    for topic in TOPICS:
        count = sum(1 for doc in documents if doc["topic"] == topic)
        print(f"  {topic}: {count:,}")

    return documents


# ============================================================
# BATCH EMBEDDING
# ============================================================

def encode_in_batches(embedding_service, texts):

    embeddings = []

    total = len(texts)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = texts[start:end]

        batch_embeddings = embedding_service.encode(
            batch
        )

        embeddings.append(
            batch_embeddings
        )

        print(
            f"Embedded {end:,}/{total:,} texts",
            end="\r"
        )

    print()

    return np.vstack(
        embeddings
    ).astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("MiniVecDB Dataset Generator")
    print("=" * 70)

    print("\nConfiguration")

    print(
        f"Total vectors       : {TOTAL_VECTORS:,}"
    )

    print(
        f"Evaluation queries  : {NUM_QUERIES}"
    )

    print(
        f"Embedding dimension : 384"
    )

    # --------------------------------------------------------
    # Remove old data
    # --------------------------------------------------------

    old_files = [
        "vectors.npy",
        "seed_embeddings.npy",
        "queries.npy",
        "documents.json",
        "metadata.json",
        "seed_documents.json",
        "query_texts.json",
        "benchmark_results.json",
    ]

    print("\nRemoving old generated data...")

    for filename in old_files:

        path = DATA_DIR / filename

        if path.exists():

            path.unlink()

            print(
                f"Removed: {filename}"
            )

    # --------------------------------------------------------
    # Create documents
    # --------------------------------------------------------

    print("\nCreating technical documents...")

    documents = build_documents()

    print(
        f"Documents created: {len(documents):,}"
    )

    if len(documents) != TOTAL_VECTORS:

        raise RuntimeError(
            f"Expected {TOTAL_VECTORS:,} documents "
            f"but generated {len(documents):,}."
        )

    # --------------------------------------------------------
    # Check uniqueness
    # --------------------------------------------------------

    texts = [
        document["text"]
        for document in documents
    ]

    unique_texts = len(
        set(texts)
    )

    print(
        f"Unique documents : {unique_texts:,}"
    )

    if unique_texts != TOTAL_VECTORS:

        raise RuntimeError(
            "Duplicate documents detected."
        )

    # --------------------------------------------------------
    # Topic distribution
    # --------------------------------------------------------

    topic_counts = {}

    for document in documents:

        topic = document["topic"]

        topic_counts[topic] = (
            topic_counts.get(topic, 0) + 1
        )

    print("\nTopic distribution:")

    for topic, count in topic_counts.items():

        print(
            f"  {topic:<15} {count:,}"
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_service = EmbeddingService()

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating vector embeddings...")
    print(
        "This may take some time depending on CPU/GPU."
    )

    vectors = encode_in_batches(
        embedding_service,
        texts
    )

    print(
        f"Vector matrix shape: {vectors.shape}"
    )

    dimension = vectors.shape[1]

    if dimension != 384:

        raise RuntimeError(
            f"Expected 384 dimensions, "
            f"got {dimension}."
        )

    # --------------------------------------------------------
    # Generate evaluation queries
    # --------------------------------------------------------

    print("\nGenerating evaluation queries...")

    query_texts = []

    query_indices = np.linspace(
        0,
        len(documents) - 1,
        NUM_QUERIES,
        dtype=int
    )

    for idx in query_indices:
        document = documents[idx]

        concept_a = document["concept_a"]

        query_text = (
            f"Explain the following {document['topic']} concept: "
            f"{concept_a}"
        )

        query_texts.append(query_text)

    print(
        f"Generated {len(query_texts)} evaluation queries."
    )

# Generate query embeddings
    queries = encode_in_batches(
        embedding_service,
        query_texts
    )

    print(
        f"Query matrix shape: {queries.shape}"
    )

    if queries.shape != (NUM_QUERIES, 384):
        raise RuntimeError(
            f"Expected query matrix shape "
            f"({NUM_QUERIES}, 384), "
            f"got {queries.shape}."
        )

# --------------------------------------------------------
# Save NumPy arrays
# --------------------------------------------------------

    print("\nSaving NumPy arrays...")

    np.save(
        DATA_DIR / "vectors.npy",
        vectors
    )

    np.save(
        DATA_DIR / "queries.npy",
        queries
    )

    # --------------------------------------------------------
    # Save documents
    # --------------------------------------------------------

    print("Saving documents...")

    with open(
        DATA_DIR / "documents.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with open(
        DATA_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Save query texts
    # --------------------------------------------------------

    with open(
        DATA_DIR / "query_texts.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            query_texts,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Documents          : {len(documents):,}"
    )

    print(
        f"Vectors            : {len(vectors):,}"
    )

    print(
        f"Dimension          : {dimension}"
    )

    print(
        f"Queries            : {len(queries):,}"
    )

    print(
        f"Unique texts       : {unique_texts:,}"
    )

    print("\nFiles created:")

    print("  data/vectors.npy")
    print("  data/queries.npy")
    print("  data/documents.json")
    print("  data/metadata.json")
    print("  data/query_texts.json")

    print("\nDataset is ready for MiniVecDB.")


if __name__ == "__main__":
    main()