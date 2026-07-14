

Rag documentation en · MD
# RAG (Retrieval-Augmented Generation) Documentation
 
## What is a RAG?
 
Retrieval-Augmented Generation (RAG) is an architecture (or technique) that enhances an LLM's (Large Language Model) capabilities by connecting it to an external knowledge source (database, documents, APIs, etc). Instead of relying solely on the information the LLM was trained on, the model retrieves relevant, up-to-date elements from these sources and integrates them into the generation process. This allows an LLM to produce a more accurate, up-to-date, and domain-specific response without the need to retrain it.
 
## What problems does RAG solve in the context of our project?
 
The goal of RAG in this project is to speed up the search for regulations concerning a specific product and its partner import or export countries. Without this technology, CAPD members would have to manually search through thousands of pages spread across multiple documents before feeding the database. Thanks to RAG and structured Pydantic models, we can instead ask the AI to extract the necessary information in a structured form; this allows it to be processed directly and to feed the database through a fully automated flow.
 
## The 2 phases of RAG
 
RAG is divided into two important and complementary phases. The data ingestion phase is the process by which information sources are transformed and stored in the vector database. The retrieval phase is the process by which the best and most relevant excerpts of information are retrieved to generate a response.
 
### Ingestion Phase (also called the "indexing pipeline")
 
The ingestion phase transforms raw data sources into queryable vectors, stored in the vector database. It takes place in several sequential steps:
 
1. **Sources** collects raw data from the various information sources. These can be in several formats such as: PDF, HTML, plain text, etc. In the context of this project, the sources are legislative documents on the import and export of goods, in PDF format.
2. **Document transformation** this step transforms the documents into a format more compatible with RAG (in this project's case, the Markdown format). The transformation of documents into Markdown format is done using an LLM. However, another very good free option is the Docling library, which is capable of transforming any type of information source (even a website) into Markdown.
3. **Chunking** the documents are split into segments of reasonable size (chunks), in order to respect the context limit of the embedding models and to keep each excerpt sufficiently precise and self-contained to be relevant during retrieval. The algorithm used for the chunking process is based on the document's structure ([see the chunking algorithm](#chunking-algorithm---structure-based-chunking))
4. **Embeddings** each chunk is transformed into **two complementary vector representations**, generated in parallel:
    - **Dense vector**: the embedding model used is OpenAI's `text-embedding-3-large`, which creates vectors with 3072 dimensions. A dense vector captures the semantic meaning of the chunk in a continuous space of a few hundred to a few thousand dimensions.
    - **Sparse vector**: produced by a method such as BM25 or SPLADE, which captures the presence and importance of specific keywords in the chunk within a very high-dimensional space, tens of thousands of dimensions (mostly zeros).
5. **Storage** the two vectors (dense and sparse) are stored as two named vectors on the **same point** in Qdrant. This avoids duplicating chunks and later allows hybrid search to be performed (see the [Hybrid search](#hybrid-search) section).
This dual vectorization at ingestion time is what makes it possible, on the retrieval side, to combine semantic search (dense) with keyword search (sparse) — each compensating for the other's blind spots (see _Dense vector_, _Sparse search_, _Hybrid search_ below).
 
#### Chunking algorithm - structure-based chunking
 
Structure-based chunking splits a document according to its sections instead of a fixed number of characters. If a section exceeds the size limit of a chunk, the algorithm splits the section into several sections while keeping the title for each one. This algorithm is used because legislative documents follow a hierarchy of sections that includes strict structures such as tables.
 
![Image description](docs/Images/RAG%20-%20Ingestion%20phase.jpg)
 
### Retrieval Phase - three steps
 
#### Retrieval - techniques/strategies
 
These are techniques or strategies that improve the quality of the "retrieval" of information excerpts from the vector database. "Retrieval" is a very important step in generating the right response for the user, because the text excerpts are then combined with the user's initial prompt to create a final prompt sent to an LLM. If these text excerpts are not accurate, the LLM's response won't be either. Embedding models (the transformation of text into a numerical representation) fall into three categories: Dense vector (Embeddings), Sparse vector (BM25 or Keyword), and Hybrid (a combination of the two)
 
##### Dense vector
 
Used for semantic understanding and search.
 
Each chunk is transformed by an embedding model (OpenAI's text-embedding-3-large) into a 3072-dimension vector. The user's query is transformed the same way, then cosine similarity is computed between the query vector and the chunk vectors stored in Qdrant to find the semantically closest ones. This approach captures rephrasings and synonyms well, but can miss exact matches on rare terms, error codes, or precise names.
 
##### Sparse Vector
 
Used for keyword understanding and search
 
Each chunk is transformed into a very high-dimensional vector (one dimension per word in the vocabulary), mostly made up of zeros. Common methods are BM25 (classic term-frequency weighting) and SPLADE (a learned, enriched version). Matching between the query and a chunk is done by overlapping active dimensions (shared words), rather than by geometric proximity as with dense vectors. This approach is precise on exact terms, but does not capture synonyms or rephrasings.
 
##### Hybrid search
 
Combines the two previous approaches: the query is sent in parallel to both dense and sparse search, each returning its own ranked list of results. These two lists are then merged into a single ranking, generally using the **Reciprocal Rank Fusion (RRF)** method, which combines results based on their position in each list rather than their raw scores (cosine scores and BM25 scores are not on the same scale and cannot be compared directly). Qdrant natively supports RRF via its `query_points` API, by combining `Prefetch` on the dense and sparse vectors of the same collection.
 
#### Reranking
 
Once the best candidates have been retrieved through hybrid search, they go through a reranking process. This process uses a cross-encoder (Cohere's `rerank-v4.0-fast`) which takes the <prompt, candidate> pairs and encodes them **jointly**. This allows a much more accurate relevance score to be computed, since the model directly compares the content of the query and the candidate rather than measuring a simple geometric distance between two pre-computed vectors. The goal is to reorder the list so that when the top 5 candidates are taken, they are guaranteed to include the relevant chunks.
 
![Image description](docs/Images/RAG%20-%20Retrieval%20phase.jpg)
 
## Diagrams
 
### Deployment diagram
 
The purpose of this diagram is to present the various services involved in this project and the interactions between them.
 
![Image description](docs/Images/deployment.png)
 
## Associated costs
 
The system's costs fall into two distinct categories:
 
1. **Hosting (infrastructure) costs**, which are fixed and recurring every month, regardless of usage volume.
2. **AI model inference costs**, which are variable and proportional to actual usage (number of documents ingested, number of questions asked).
### Hosting costs
 
All of the system's components are hosted on Render, with the exception of the vector database, which uses Qdrant Cloud's free tier.
 
| Component | Hosting | Monthly cost |
| --- | --- | --- |
| RAG service (FastAPI) | Render | $7 |
| Relational database (PostgreSQL) | Render Basic 1 GB | $19 |
| Vector database (Qdrant) | Qdrant Cloud (free tier) | Free |
| **Total** | | **$26 / month** |
 
### External services vs. open source cost evaluation
 
An important architectural decision was determining whether the embedding and reranking models should be run in-house (open source models served by Ollama) or consumed via external services (OpenAI and Cohere). Both approaches were evaluated based on total cost of ownership.
 
#### Usage costs of external services
 
External services are billed based on usage, with no fixed cost. No compute resources need to be provisioned or maintained.
 
| Service | Role in the pipeline | Rate |
| --- | --- | --- |
| OpenAI `text-embedding-3-large` | Dense vectorization (ingestion and queries) | $0.13 / 1M tokens |
| Cohere `rerank-v4.0-fast` | Reranking of candidates | $2.00 / 1,000 searches |
| Claude Sonnet 4.6 | Final response generation | $3 / 1M input tokens, $15 / 1M output tokens |
| BM25 (fastembed, run locally) | Sparse vectorization | Free |
 
#### Deployment and infrastructure costs for open source (Ollama)
 
Self-hosting the models eliminates usage-based billing, but requires provisioning a machine capable of running them. The table below compares hosting options for the `qwen3-embedding` embedding model, in its two sizes.
 
| Model | Render (CPU only, no GPU option) | Azure, CPU VM | Azure, GPU VM |
| --- | --- | --- | --- |
| `qwen3-embedding:0.6b` (~639 MB) | Standard: $25/month (2 GB RAM, 1 CPU). The Starter tier ($7/month, 512 MB) is insufficient once the overhead of the Ollama runtime is added. | D2s_v5: roughly $70 to $90/month (2 vCPU, 8 GB RAM), significantly oversized for a 0.6b model. | Not relevant: the 0.6b model runs adequately on CPU. |
| `qwen3-embedding:4b` (~2.5 GB) | Pro Plus: $175/month (8 GB RAM, 4 CPU) recommended for headroom. CPU inference remains noticeably slower at query time. | D2s_v5: roughly $70 to $90/month (8 GB RAM), functional but limited by the CPU, with latency comparable to Render's Pro Plus tier. | NC4as_T4_v3: roughly $384/month (T4 GPU), necessary to achieve acceptable real-time embedding latency. |
 
Running the reranker in-house presents an additional constraint. The `BAAI/bge-reranker-v2-m3` model weighs 2.3 GB and runs into two limitations on the current infrastructure:
 
1. **Lack of GPU (CUDA)**, essential for speeding up the cross-encoder's pass over the candidates and keeping latency acceptable.
2. **RAM**, as the model requires an amount of memory that the affordable hosting tiers do not provide.
Note that moving dense vector generation to OpenAI significantly lightened the RAG service's footprint: hybrid search now runs without difficulty on Render's base CPU, whereas the fully local approach previously required 7 to 8 GB of RAM. The reranker remains the only component that would still justify a dedicated machine.
 
#### Verdict
 
The external services approach was chosen. The system's actual inference cost sits around a few dollars per month for a typical usage volume, whereas the most economical viable open source configuration would add at least $25 per month in fixed costs, and more still once a GPU becomes necessary for the reranker. The financial break-even point would only be reached at a query volume far higher than this project's, and the infrastructure's operating cost would still need to be absorbed on top of that. External services also offer superior model quality, stable latency, and no maintenance.
 
### Complete ingestion phase
 
The complete ingestion pipeline takes about 3 minutes and 13 seconds and is run only twice a month, since legislation changes infrequently.
 
| Metric | Value |
| --- | --- |
| Total chunks in Qdrant | 4,311 |
| Total embedding tokens | 347,596 |
| Average tokens per chunk | 80.6 |
 
Cost of one complete ingestion cycle: 347,596 tokens × $0.13 / 1M ≈ **$0.045**, or about **$0.09 per month** for the two runs. Sparse vectorization (BM25) is computed locally and incurs no cost.
 
### Prompt + response
 
Cost of a question followed by its response, measured on the first endpoint.
 
| Component | Quantity | Cost |
| --- | --- | --- |
| Prompt embedding (OpenAI) | 21 tokens | $0.000003 |
| Reranking (Cohere) | 1 search unit | $0.0020 |
| Generation, input (Claude) | 2,368 tokens × $3 / 1M | $0.000308 |
| Generation, output (Claude) | 196 tokens × $15 / 1M | $0.00294 |
| **Total** | | **$0.005251** |
 
A complete request therefore costs about half a cent. Reranking ($0.0020) and output generation ($0.00294) alone account for over 94% of the cost, while the prompt embedding is negligible.
 
It should be noted that unit costs decrease in an inverse exponential fashion with volume: the fixed portion of ingestion is amortized across all requests, so the average cost per question tends toward the marginal inference cost as usage increases. For reference, at 1,000 questions per month, the system's total cost comes to about $33 in hosting plus $5.34 in inference, for a total of about **$38 per month**.
 
