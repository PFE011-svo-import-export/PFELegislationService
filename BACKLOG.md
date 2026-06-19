What you've already done right (this is the good news)
Your architecture is genuinely clean and more professional than most bootcamp/course projects:

Proper dependency injection — chat_service.py and rag_service.py take their dependencies in __init__ instead of constructing clients inline. This is exactly the pattern frameworks try to give you for free, and you did it by hand correctly. Easy to test, easy to swap.
Clean separation of concerns — embedding/parsing/chunking (RagService) is separate from generation (ChatService) is separate from persistence (VectorStore). Most tutorials cram all of this into one file.
Structured output done properly — chat_service.py:22-32 uses messages.parse with a Pydantic output_format. That's the modern, reliable way (not regex-parsing the model's text).
A real chunking strategy — your heading-aware chunking in rag_service.py:88-155 with a heading_stack that builds heading_path context is exactly the kind of thing LangChain's naive splitters do badly. This is a real engineering decision, and it's the single most impressive thing in the repo.
Hybrid model strategy — local Ollama embeddings (cheap, private) + Anthropic for generation (quality). Cost-aware. That's senior thinking.
Containerized with FastAPI + Qdrant. Production-shaped, not a notebook.
If an interviewer asked "did you use a framework or the raw SDK, and why?" — this codebase is your answer, and it's a good one.

What's missing to make it portfolio-strong for $100K+ roles
These are the "unglamorous parts" I mentioned. In rough priority order:

1. Evaluation — the #1 gap. Right now you have no way to prove your RAG is good. There's no test set, no metrics. This is the single biggest differentiator. Add:

A small set of ~20-30 question/expected-answer pairs for your legislation docs.
Retrieval metrics: is the right chunk in the top-k? (hit rate / MRR)
Answer quality: an LLM-as-judge script that grades faithfulness ("did the answer stick to the retrieved chunks?") and relevance.
If you add one thing, add this. Being able to say "I improved retrieval hit-rate from 0.6 to 0.85 by changing chunking" is a job-winning sentence.
2. No reranking / retrieval is naive. chat_service.py:11-14 does single-vector search, top_k=3, and dumps raw content into the prompt. Improvements that show depth:

Add a reranker (cross-encoder) or hybrid search (vector + keyword/BM25) — legislation has exact terms (article numbers, tariff codes) where pure semantic search fails.
The augmented prompt at chat_service.py:15-21 just str()-dumps a Python list. Format chunks cleanly with their heading_path and source so the model can cite.
3. No citations / source attribution. Your ChatResponse returns merchandise_name and rates but never says which document/article it came from. For legal Q&A, citations are non-negotiable and trivially impressive to add — you already have source and heading_path in the payload.

4. No prompt-injection / guardrails awareness. A user prompt flows straight into the model. For a security-conscious portfolio, even a short note + basic input validation shows you know the threat exists.

5. No observability. No latency logging, no token/cost tracking, no tracing. Even simple structured logging of (query → retrieved chunk ids → scores → latency → tokens) per request demonstrates production thinking. This is what Langfuse/LangSmith sell — you can hand-roll a lightweight version.

6. Error handling & resilience. embed and search have no retry/timeout/fallback. query_points could return nothing and you'd send empty context. The "I don't know" instruction helps, but handle the empty-retrieval case explicitly.

7. No tests at all. Even a few pytest unit tests on the chunking logic (it's pure and deterministic — perfect to test) would round it out.

If I were you, I'd do these three
Build the eval harness (item 1) — biggest signal, turns the project into a story with numbers.
Add citations (item 3) — cheap, and critical for the legal domain you chose.
Add reranking or hybrid search (item 2) — shows you understand retrieval is the hard part of RAG.
Those three move this from "clean RAG demo" to "this person understands production RAG." The framework question becomes irrelevant at that point — nobody asks what library you used when you can show retrieval metrics.

Want me to scaffold the evaluation harness (a eval/ folder with a question set format, retrieval metrics, and an LLM-as-judge faithfulness scorer)? That's the highest-leverage next step and it'll fit naturally onto your existing service layer.

# Product Backlog — RAG Legislation Service

User stories to harden the service from a working RAG demo into a production-grade,
portfolio-strong system. Ordered roughly by leverage/priority.

Legend — Story points are rough relative sizing (1 = a few hours, 8 = several days).

---

## US-01 — Evaluation dataset for retrieval and answers
**Points:** 3

**Description**
As an AI engineer, I want a curated set of question/expected-answer pairs for the
legislation documents, so that I have a ground truth to measure retrieval and answer
quality against instead of judging the system by eyeballing single queries.

**What needs to be done**
- Create an `eval/` folder with a versioned dataset file (`eval/dataset.jsonl` or `.yaml`).
- Each entry contains: `question`, `expected_answer` (or key facts), `relevant_source`
  (filename), `relevant_heading_path`, and optionally a difficulty/tag.
- Author 25–40 questions covering: easy lookups, multi-section answers, and
  out-of-scope questions that should return "I don't know".
- Document the dataset format in `eval/README.md`.

**Acceptance criteria**
- [ ] At least 25 question/answer entries exist and load without parse errors.
- [ ] At least 3 entries are deliberately out-of-scope (expected answer = "I don't know").
- [ ] Each entry references a real `source` and `heading_path` that exist in the indexed corpus.
- [ ] Dataset format is documented and committed.

---

## US-02 — Retrieval quality metrics
**Points:** 5

**Description**
As an AI engineer, I want an automated script that measures retrieval quality against
the eval dataset, so that I can quantify whether a change to chunking, embeddings, or
`top_k` actually improves recall instead of guessing.

**What needs to be done**
- Add `eval/run_retrieval_eval.py` that, for each dataset question, embeds the query,
  runs `VectorStore.search`, and checks whether the expected chunk appears in top-k.
- Compute and report: Hit Rate@k, MRR (Mean Reciprocal Rank), and average top-1 score.
- Output results to console and a timestamped report file (`eval/reports/`).
- Make `top_k` a CLI/config parameter so runs are reproducible and comparable.

**Acceptance criteria**
- [ ] Running the script prints Hit Rate@k and MRR over the full dataset.
- [ ] A report file is written with per-question pass/fail and the aggregate metrics.
- [ ] The `top_k` used is recorded in the report.
- [ ] Re-running on an unchanged index produces identical metrics (deterministic).

---

## US-03 — LLM-as-judge answer scoring (faithfulness & relevance)
**Points:** 5

**Description**
As an AI engineer, I want an automated judge that grades generated answers for
faithfulness (grounded in retrieved chunks, no hallucination) and relevance (actually
answers the question), so that I can measure answer quality at scale instead of reading
every response by hand.

**What needs to be done**
- Add `eval/run_answer_eval.py` that runs each dataset question through the full
  generate pipeline and captures the answer + the retrieved chunks.
- Use Claude as a judge with a structured rubric returning scores (e.g. 1–5) for
  `faithfulness` and `relevance` plus a short rationale, via Pydantic structured output.
- Aggregate average scores and flag low-scoring answers for manual review.
- Write results to `eval/reports/`.

**Acceptance criteria**
- [ ] Script outputs average faithfulness and relevance scores across the dataset.
- [ ] Each graded answer stores the judge's score and rationale.
- [ ] Answers below a configurable threshold are listed separately as "needs review".
- [ ] The judge prompt/rubric is documented and version-controlled.

---

## US-04 — Source citations in chat responses
**Description**
As a user asking legal import/export questions, I want each answer to cite the source
document and section it came from, so that I can verify the information and trust it for
a regulated/legal domain.

**Points:** 3

**What needs to be done**
- Extend `ChatResponse` schema with a `citations` field (list of `{source, heading_path}`).
- Pass the retrieved chunks' `source` and `heading_path` into the augmented prompt so the
  model can attribute claims (the data is already returned by `VectorStore.search`).
- Update the system prompt to require citing the chunks used.
- Return citations through the API response.

**Acceptance criteria**
- [ ] `ChatResponse` includes structured citations.
- [ ] A successful answer returns at least one citation referencing a real retrieved chunk.
- [ ] An "I don't know" response returns an empty citations list.
- [ ] Citations correspond to chunks that were actually retrieved (no fabricated sources).

---

## US-05 — Improved context formatting (prompt/context engineering)
**Points:** 2

**Description**
As an AI engineer, I want the retrieved chunks formatted cleanly into the prompt instead
of a raw Python list dump, so that the model reliably understands chunk boundaries,
sources, and sections.

**What needs to be done**
- Replace the `str(list)` interpolation in `ChatService.generate` with a structured block
  per chunk (e.g. numbered, with `source` and `heading_path` headers and the content).
- Separate the user question from the documentation context clearly in the prompt.
- Keep formatting logic in a small dedicated helper for testability.

**Acceptance criteria**
- [ ] Retrieved context is rendered with per-chunk source/section labels, not a raw list.
- [ ] User question and context are visually/structurally separated in the prompt.
- [ ] Existing eval scores (US-03) do not regress after the change.

---

## US-06 — Handle empty / low-confidence retrieval explicitly
**Points:** 2

**Description**
As a user, I want the system to respond safely when no relevant documents are found, so
that I never receive a confidently made-up answer when the corpus has no information.

**What needs to be done**
- In `ChatService.generate`, detect when `search` returns no results or all scores fall
  below a configurable similarity threshold.
- Short-circuit to a deterministic "I don't know / no relevant legislation found" response
  without calling the generation model (saves cost and avoids hallucination).
- Make the threshold configurable in `config`.

**Acceptance criteria**
- [ ] An out-of-scope question returns the safe fallback response.
- [ ] When all retrieval scores are below threshold, no generation call is made.
- [ ] The similarity threshold is configurable, not hard-coded in the method body.
- [ ] Out-of-scope eval entries (US-01) pass.

---

## US-07 — Reranking or hybrid search for retrieval precision
**Points:** 8

**Description**
As an AI engineer, I want to improve retrieval precision using reranking or hybrid
(semantic + keyword) search, so that queries containing exact legal terms (article
numbers, tariff codes) return the correct sections that pure vector search misses.

**What needs to be done**
- Choose an approach: cross-encoder reranker over top-N candidates, or hybrid search
  combining Qdrant vector search with keyword/BM25 (Qdrant supports sparse vectors).
- Retrieve a larger candidate set (e.g. top-20) then rerank down to top-k.
- Make the strategy switchable via config so old vs. new can be compared.
- Re-run US-02 retrieval eval to quantify the improvement.

**Acceptance criteria**
- [ ] The retrieval pipeline supports a rerank/hybrid step toggleable by config.
- [ ] A query with an exact legal/numeric term retrieves the correct section in top-k.
- [ ] Hit Rate@k and/or MRR (US-02) improve measurably vs. the baseline, with numbers
      recorded in a report.
- [ ] Added latency from reranking is measured and documented.

---

## US-08 — Request observability (tracing, latency, token/cost logging)
**Points:** 5

**Description**
As an AI engineer operating the service, I want structured per-request logging of the
retrieval and generation pipeline, so that I can debug bad answers and monitor latency
and cost in production.

**What needs to be done**
- Add structured logging that captures per request: query, retrieved chunk ids + scores,
  number of chunks used, generation latency, and input/output token counts.
- Assign a correlation/request id and attach it to every log line for a request.
- Log to structured output (JSON) so it can be shipped to a log aggregator later.
- Optional stretch: integrate a tracing tool (Langfuse/OpenTelemetry) behind a config flag.

**Acceptance criteria**
- [ ] Each `/generate` request emits a structured log including retrieved chunk scores,
      latency, and token usage.
- [ ] All log lines for one request share a correlation id.
- [ ] Logging adds no functional change to the API response.
- [ ] Token/latency fields are present and populated (not null) on a successful request.

---

## US-09 — Resilience: timeouts, retries, and graceful failures
**Points:** 3

**Description**
As a user, I want the service to handle transient failures of the embedding model,
vector DB, or generation API gracefully, so that a temporary hiccup returns a clean error
instead of an unhandled 500.

**What needs to be done**
- Add timeouts and bounded retries (with backoff) around Ollama embed calls, Qdrant
  queries, and the Anthropic generation call.
- Map downstream failures to meaningful HTTP responses (e.g. 503 on dependency outage).
- Ensure partial failures don't leave the request hanging indefinitely.

**Acceptance criteria**
- [ ] Embedding, vector search, and generation calls each have a configured timeout.
- [ ] Transient failures are retried a bounded number of times with backoff.
- [ ] A downstream outage returns a clean, documented error response (not a raw stack trace).
- [ ] Timeout/retry values are configurable.

---

## US-10 — Prompt-injection guardrails and input validation
**Points:** 3

**Description**
As a security-conscious engineer, I want basic protection against prompt injection and
malformed input, so that a malicious user prompt cannot override the system instructions
or exfiltrate the retrieved context.

**What needs to be done**
- Validate and bound the incoming `prompt` (max length, reject empty/whitespace-only).
- Clearly delimit retrieved documentation from user input in the prompt and instruct the
  model to treat documentation as data, not instructions.
- Add a short note/section in the README documenting the injection threat model and the
  mitigations chosen.
- Optional: a lightweight check for obvious injection patterns in the input.

**Acceptance criteria**
- [ ] Prompts exceeding a max length or that are empty are rejected with a 4xx error.
- [ ] Retrieved context is delimited and labelled as untrusted data in the prompt.
- [ ] A test injection prompt (e.g. "ignore previous instructions and...") does not change
      the system behavior / output schema.
- [ ] Threat model and mitigations are documented in the README.

---

## US-11 — Unit tests for chunking and serialization
**Points:** 3

**Description**
As a developer, I want unit tests covering the markdown chunking and serialization logic,
so that I can refactor the parsing pipeline confidently without breaking chunk boundaries
or heading paths.

**What needs to be done**
- Add `tests/` with pytest tests for `chunk_md_file`, the `_serialize_*` handlers, and
  `_extract_text` using small fixture markdown files.
- Cover: heading hierarchy / `heading_path` construction, lists (ordered & unordered),
  tables, quotes, code fences, and empty-section handling.
- Wire `pytest` into the project (and CI if/when added).

**Acceptance criteria**
- [ ] Tests assert correct `heading_path` for nested headings.
- [ ] Tests cover list, table, quote, and code-fence serialization.
- [ ] Empty sections produce no chunk (matches `flush_chunk` behavior).
- [ ] `pytest` runs green locally.

---

## US-12 — Configurable RAG parameters
**Points:** 2

**Description**
As an AI engineer running experiments, I want retrieval and generation parameters
(top_k, similarity threshold, model names, candidate pool size) centralized in config,
so that I can tune and reproduce experiments without editing service code.

**What needs to be done**
- Move hard-coded values (`top_k=3`, model ids, `VECTOR_SIZE`, thresholds) into `config`.
- Ensure services read these from config via dependency injection.
- Document each parameter and its default.

**Acceptance criteria**
- [ ] No RAG tuning constant is hard-coded inside service/storage methods.
- [ ] Changing a value in config (e.g. `top_k`) changes runtime behavior without code edits.
- [ ] Parameters and defaults are documented.

---

## US-13 — Document ingestion endpoint / pipeline robustness
**Points:** 3

**Description**
As an operator, I want a reliable way to (re)index documents, so that adding or updating
legislation files keeps the vector store consistent without manual scripts or duplicates.

**What needs to be done**
- Provide an ingestion entrypoint (route or CLI) that chunks, embeds, and upserts a file
  or folder.
- Make ingestion idempotent: re-indexing the same source replaces its chunks rather than
  creating duplicates (e.g. deterministic ids or delete-by-source before upsert).
- Report counts (files processed, chunks created) on completion.

**Acceptance criteria**
- [ ] Re-indexing the same document does not create duplicate chunks in Qdrant.
- [ ] Ingestion reports number of files and chunks processed.
- [ ] A failed file does not abort the whole batch silently (errors are surfaced).
