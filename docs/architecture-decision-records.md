# CodeWise Architecture Decision Records (ADRs)

**Author**: Sam Ballington
**Project**: CodeWise - AI-powered Code Intelligence Platform
**Timeline**: July 2025 - September 2025
**Last Updated**: February 2026

---

## Table of Contents

1. [ADR-001: SQLite + VSS over Neo4j](#adr-001-sqlite-over-neo4j-for-knowledge-graph)
2. [ADR-002: FAISS over Managed Vector Databases](#adr-002-faiss-over-managed-vector-databases)
3. [ADR-003: WebSocket over REST Polling](#adr-003-websocket-over-rest-polling)
4. [ADR-004: Tool Consolidation (8 → 2 Tools)](#adr-004-tool-consolidation-8--2-tools)
5. [ADR-005: 4-Layer Caching Strategy](#adr-005-4-layer-caching-strategy)
6. [ADR-007: BGE-large-en-v1.5 over MiniLM](#adr-007-bge-large-en-v15-embedding-model)
7. [ADR-008: Cerebras over OpenAI for LLM Provider](#adr-008-cerebras-over-openai-for-llm-provider)
8. [ADR-009: Docker 4-Service Architecture](#adr-009-docker-multi-service-architecture)
9. [ADR-010: Tree-sitter for AST Parsing](#adr-010-tree-sitter-for-ast-extraction)
10. [ADR-012: FastAPI over Flask/Django](#adr-012-fastapi-over-flaskdjango)

---

# ADR-001: SQLite over Neo4j for Knowledge Graph

**Status**: Accepted
**Date**: 2025-08-15
**Deciders**: Sam Ballington

## Context

CodeWise needed a persistent storage layer for its Knowledge Graph - a structure tracking code symbols (functions, classes, modules) and their relationships (calls, inherits, imports). The KG needed to support recursive traversal queries (e.g., "find all callers of function X up to depth 3") while fitting into a containerized deployment without adding operational overhead.

## Decision

Use SQLite with custom schema (nodes, edges, chunks, files tables) and the sqlite-vss extension for vector similarity search, instead of a dedicated graph database like Neo4j.

## Options Considered

1. **Neo4j** — Pros: Native graph traversal (Cypher), built for relationship queries. Cons: Requires separate server process, 500MB+ JVM memory overhead, Docker Compose complexity, operational burden for a single-developer project.

2. **PostgreSQL + pgvector** — Pros: Mature, combined relational + vector search. Cons: Separate server process, overkill for 440-node graph.

3. **SQLite + VSS** (Chosen) — Pros: Zero administration, embedded in process, ACID compliant, file-based (portable), recursive CTEs for graph traversal, VSS extension for vector search. Cons: Manual relationship traversal, single-writer limitation.

## Rationale

At ~440 nodes and ~17,800 estimated relationships, SQLite's recursive CTEs provide sufficient graph traversal capability without the operational overhead of a separate database server. The decision prioritized **operational simplicity** and **zero-dependency deployment** over native graph query language support.

Key factors: scale fit (well within SQLite's comfortable range), deployment simplicity (no additional container, no JVM), ACID compliance, and portability (single `.db` file).

Performance tuning applied:
```sql
PRAGMA journal_mode = WAL;
PRAGMA cache_size = -64000;      -- 64MB cache
PRAGMA mmap_size = 268435456;    -- 256MB memory-mapped I/O
PRAGMA temp_store = MEMORY;
```

## Consequences

**Positive**: Zero operational overhead; ACID compliance with WAL mode; single file (`codewise.db`) simplifies backup and debugging; recursive CTEs handle caller chain queries; VSS extension provides vector similarity search in the same database.

**Negative**: Relationship traversal requires manual recursive CTEs instead of Cypher; single-writer limitation serializes write-heavy indexing; no built-in graph visualization.

**Lesson**: Don't over-engineer infrastructure for a problem that doesn't exist yet. At 440 nodes, operational simplicity wins decisively over a graph DB.

---

# ADR-002: FAISS over Managed Vector Databases

**Status**: Accepted
**Date**: 2025-07-28
**Deciders**: Sam Ballington

## Context

CodeWise needed a vector similarity search engine to find semantically related code chunks. The system processes code repositories into embedding vectors (1024 dimensions via BGE-large-en-v1.5) and needs to retrieve top-k most similar chunks for a given query, working reliably in a Docker container with no external service dependencies.

## Decision

Use FAISS (Facebook AI Similarity Search) as a local, in-process vector index instead of managed vector database services.

## Options Considered

1. **Pinecone** — Pros: Fully managed, scalable. Cons: Per-query pricing, vendor lock-in, requires internet, data leaves local environment.

2. **Weaviate** — Pros: Rich query language, multi-modal support. Cons: Separate server process, 1-2GB memory overhead, additional Docker container.

3. **ChromaDB** — Pros: Python-native, easy integration. Cons: Less mature, performance concerns at scale.

4. **FAISS** (Chosen) — Pros: In-process (no server), battle-tested at Facebook scale, excellent performance, full control over index lifecycle, works offline, no per-query costs. Cons: No built-in metadata filtering, manual index persistence.

## Rationale

Three principles drove this decision: (1) **Local-first architecture** — proprietary code shouldn't leave the local environment; (2) **Operational simplicity** — no additional servers or external API dependencies; (3) **Cost predictability** — no per-query pricing. FAISS provides sub-millisecond similarity search for current index sizes.

## Consequences

**Positive**: Zero external dependencies; full control over embedding generation; predictable costs; private by default; sub-millisecond query performance.

**Negative**: Must implement own metadata filtering (project scoping); manual index persistence (`index.faiss` + `meta.json`); no built-in distribution.

**Lesson**: For a single-user tool, managed vector DBs add complexity and cost without proportional benefit. The in-process model eliminated an entire category of operational issues.

---

# ADR-003: WebSocket over REST Polling

**Status**: Accepted
**Date**: 2025-07-10
**Deciders**: Sam Ballington

## Context

CodeWise needs to stream LLM responses to the frontend in real-time. Architecture analysis queries can take 5-25 seconds, and users need progressive feedback (acknowledgment, context gathering status, streaming response text) rather than waiting for a complete response.

## Decision

Use WebSocket connections (`ws://localhost:8000/ws`) for all query interactions instead of REST endpoints with polling.

## Options Considered

1. **REST with Polling** — Pros: Simpler server, stateless. Cons: Increased latency, higher request overhead, no real-time streaming.

2. **Server-Sent Events (SSE)** — Pros: Simpler than WebSocket, auto-reconnection. Cons: Unidirectional only (client can't send without new HTTP request).

3. **WebSocket** (Chosen) — Pros: Full-duplex, low latency, persistent connection, supports structured message types. Cons: Connection management complexity, stateful (harder to scale horizontally).

## Rationale

The bidirectional nature of WebSocket was essential: the client sends user messages, project context (@mentions), model selection, and keep-alive pings; the server streams acknowledgments, context gathering progress, search results, streaming LLM tokens, completion signals, and errors. The message type system (`acknowledgment`, `context_gathering_start`, `context_search`, `final_result`, `completion`, `error`) provides fine-grained progress updates that would be awkward to model with REST or SSE.

## Consequences

**Positive**: Real-time streaming with sub-100ms latency; clean message type system; persistent connection eliminates reconnection overhead; bidirectional communication supports project context persistence across messages.

**Negative**: Connection management complexity; stateful architecture makes horizontal scaling harder; keep-alive pings required to prevent timeout disconnections.

**Lesson**: Real-time streaming is table stakes for LLM-powered applications. The streaming approach eliminated the "loading spinner for 15 seconds" UX anti-pattern.

---

# ADR-004: Tool Consolidation (8 → 2 Tools)

**Status**: Accepted
**Date**: 2025-08-06 (first consolidation), 2025-08-17 (final consolidation)
**Deciders**: Sam Ballington

## Context

The original CodeWise agent exposed 8 separate tools to the LLM: `code_search`, `file_glimpse`, `list_entities`, `read_file`, `search_by_extension`, `search_file_content`, `get_project_structure`, `find_related_files`. The LLM struggled to choose the right tool from 8 overlapping options, leading to "tool selection paralysis" — wrong tool selection, redundant tool usage, and failed tool chaining.

## Decision

Progressively consolidate from 8 tools to 3, then from 3 to 2:

- **Phase 1** (Aug 6): 8 → 3 tools (`smart_search`, `examine_files`, `analyze_relationships`)
- **Phase 2** (Aug 17): 3 → 2 tools — `query_codebase` (all information discovery) + `navigate_filesystem` (structural file exploration)

## Options Considered

1. **Keep 8 specialized tools** with better prompting — Cons: LLM paralysis proven by experience; overlapping functionality confuses the model.

2. **Single universal tool** — Cons: Loses ability to convey intent; parameter space becomes unwieldy.

3. **2-tool architecture** (Chosen) — Pros: Clear separation of concerns (conceptual vs structural), eliminates paralysis, enables mandatory workflows. Cons: Internal routing complexity moves to application code.

## Rationale

The key insight: **move routing logic from probabilistic LLM decisions to deterministic Python code** (QueryClassifier + QueryRouter). The planning document identified the problem as "decision paralysis" and the solution as "deterministic routing." Mandatory workflows in the system prompt (Workflow A: Standard Query, Workflow B: Codebase Onboarding, Workflow C: Diagram Generation) replaced open-ended tool selection.

## Consequences

**Positive**: 87% complexity reduction in agent code (3,357 lines → 435 lines); 58% accuracy improvement (60% → 95% discovery accuracy); 50-97% response time improvement (15-30s → 0.5-15s); eliminated entire categories of tool selection bugs.

**Negative**: Internal routing complexity now lives in `query_codebase` implementation; harder to add narrow, specialized capabilities without modifying the unified tools.

**Lesson**: Fewer, smarter tools beat many simple tools when an LLM is the orchestrator. This was CodeWise's single biggest architectural win.

---

# ADR-005: 4-Layer Caching Strategy

**Status**: Accepted
**Date**: 2025-08-18
**Deciders**: Sam Ballington

## Context

CodeWise's query pipeline has several expensive computational stages: discovery (~1-5s), embedding generation (~200-500ms per batch), AST parsing (~500ms-2s per file), and LLM API calls (~2-15s, uncacheable). Repeated queries about the same project were re-running all stages unnecessarily.

## Decision

Implement a 4-layer caching architecture, each layer targeting a specific bottleneck:

1. **Discovery Cache** — Cross-session file discovery results (JSON, 24h TTL)
2. **Embedding Cache** — BGE embedding vectors (HDF5/NumPy, persistent)
3. **Chunk Cache** — Hierarchical AST chunk structures (JSON per file, 7 days TTL)
4. **Metrics Cache** — Real-time performance monitoring (in-memory)

## Options Considered

1. **Simple key-value cache (Redis/Memcached)** — Cons: Requires separate server; doesn't handle vector data efficiently; generic approach misses domain-specific optimizations.

2. **Single unified cache layer** — Cons: Different data types (file lists, vectors, AST trees) have different storage requirements and invalidation patterns.

3. **Domain-specific 4-layer cache** (Chosen) — Pros: Each layer optimized for its data type, independent TTLs, targeted invalidation.

## Rationale

Each layer caches fundamentally different data:

| Layer | Data Type | Optimal Storage |
|-------|-----------|-----------------|
| Discovery | File lists + symbols | JSON file |
| Embedding | Float32 vectors (1024D) | HDF5/NumPy |
| Chunk | AST hierarchies | JSON files |
| Metrics | Statistics | In-memory |

A single cache strategy couldn't efficiently handle all four. Vector data needs NumPy-optimized storage; discovery results need file-signature-based invalidation; chunk hierarchies need parser-version-aware keys.

## Consequences

**Positive**: 85% cache hit rate for repeated queries; discovery cache eliminates 1-5s pipeline; embedding cache eliminates redundant BGE inference; chunk cache eliminates redundant AST parsing; performance API endpoints (`/api/cache/performance`) provide visibility.

**Negative**: 4 separate cache modules to maintain; cache invalidation complexity; disk space usage for persistent caches (max 2GB for embeddings).

**Lesson**: Different data types demand different caching strategies. The metrics layer (Layer 4) was surprisingly valuable for debugging — it immediately shows which layer is underperforming and why.

---

# ADR-007: BGE-large-en-v1.5 Embedding Model

**Status**: Accepted (superseded MiniLM-L6-v2)
**Date**: 2025-08-15
**Deciders**: Sam Ballington

## Context

CodeWise originally used `all-MiniLM-L6-v2` (384 dimensions) for embedding generation. While functional, retrieval quality was inconsistent — conceptual code queries often returned only partially relevant results.

## Decision

Upgrade to `BAAI/bge-large-en-v1.5` (1024 dimensions) as the primary embedding model.

## Options Considered

1. **all-MiniLM-L6-v2** (384D) — Pros: Fast, small, low memory. Cons: Lower semantic richness for code; 384D limits representation capacity.

2. **OpenAI text-embedding-3-small** (1536D) — Pros: High quality. Cons: API dependency, per-token cost, data leaves local environment.

3. **BAAI/bge-large-en-v1.5** (1024D) (Chosen) — Pros: State-of-the-art for retrieval tasks, runs locally via sentence-transformers, supports query instruction prefixes. Cons: ~1.3GB model size, slower inference than MiniLM.

4. **BAAI/bge-small-en-v1.5** (384D) — Gives up representation capacity that helps with code semantics.

## Rationale

1024D vs 384D provides a richer embedding space for code retrieval where subtle semantic differences ("authentication" vs "authorization") must be captured. BGE-large was chosen over OpenAI embeddings to maintain the local-first, no-API-dependency architecture. The query instruction prefix feature provides better retrieval:

```python
query_embedding = model.encode(f"Represent this code query: {query}")
```

## Consequences

**Positive**: 15-25% improvement across P@3, P@5, R@10, and MRR metrics (benchmarked via golden set evaluation); richer 1024D space captures more nuance; local inference — no API dependency.

**Negative**: ~1.3GB model size increases Docker image and cold start time; slower inference than MiniLM (mitigated by embedding cache — ADR-005).

**Lesson**: Embedding quality is the foundation of RAG quality. No amount of clever re-ranking compensates for poor base embeddings. Build the benchmark framework before choosing the model.

---

# ADR-008: Cerebras over OpenAI for LLM Provider

**Status**: Accepted (superseded multi-provider architecture)
**Date**: 2025-08-06
**Deciders**: Sam Ballington

## Context

CodeWise originally used OpenAI GPT-4 Turbo with a multi-provider architecture supporting runtime switching between OpenAI, Kimi K2, and Moonshot. The frontend had a provider toggle button. Maintaining multiple providers added complexity to tool calling, prompt engineering, and testing.

## Decision

Standardize on Cerebras as the sole LLM provider using their native SDK, and remove the multi-provider architecture.

## Options Considered

1. **OpenAI GPT-4 Turbo** — Pros: Market leader, excellent tool calling. Cons: Expensive ($30/1M input tokens), rate limits.

2. **Multi-provider** (OpenAI + Kimi + Moonshot) — Pros: Flexibility, redundancy. Cons: Different tool calling formats per provider, prompt engineering per provider, testing complexity multiplied.

3. **Cerebras with gpt-oss-120b** (Chosen) — Pros: Fast inference (custom hardware), competitive pricing, native SDK with built-in tool use loop, `reasoning_effort` parameter for quality/speed tradeoff, 65K context window. Cons: Single-vendor dependency, smaller ecosystem.

## Rationale

The switch was triggered by practical experience: tool calling behavior varied between providers; system prompts needed provider-specific tuning; testing required running the same suite against each provider; the frontend toggle added UI complexity without proportional user value.

Cerebras's SDK handled the tool-use loop natively, eliminating our custom reasoning framework entirely.

## Consequences

**Positive**: Eliminated provider-specific bugs; SDK-native tool use loop replaced 3,357 lines of custom agent code with 435 lines; `reasoning_effort` parameter (low/medium/high) provides quality/speed tradeoff per query; supported models: gpt-oss-120b, qwen-3-coder-480b, qwen-3-235b-a22b-thinking-2507.

**Negative**: Single-vendor dependency risk; smaller community; can't A/B test providers without re-adding the provider abstraction.

**Lesson**: Multi-provider flexibility sounds good in theory but multiplies engineering complexity in practice. Optimize for one provider deeply rather than supporting many shallowly.

---

# ADR-009: Docker Multi-Service Architecture

**Status**: Accepted
**Date**: 2025-07-10
**Deciders**: Sam Ballington

## Context

CodeWise consists of a React frontend, a FastAPI backend, an indexing service, and an MCP (Model Context Protocol) server for file operations. These components have different runtime requirements, scaling characteristics, and update frequencies.

## Decision

Deploy as 4 separate Docker containers orchestrated with Docker Compose.

## Options Considered

1. **Monolith** (single container) — Pros: Simple deployment. Cons: Can't scale individually; mixed runtimes (Node + Python); larger image.

2. **2 containers** (frontend + backend-with-everything) — Pros: Simpler. Cons: Backend would mix query serving with CPU-intensive indexing, affecting query latency.

3. **4 containers** (Chosen) — Pros: Separation of concerns, independent scaling, mixed runtimes (Node.js + Python), index rebuilds don't affect query latency. Cons: Docker Compose complexity, inter-service networking.

## Rationale

The key driver was separating the **indexer** from the **backend**: indexing is CPU/memory intensive (AST parsing, embedding generation), while query serving needs low latency. Running both in the same container means indexing degrades query performance. The MCP server was separated for the security boundary around file system access.

## Consequences

**Positive**: Index rebuilds don't impact query response times; each service has its own requirements.txt (smaller images); independent restart/update of any service; clear security boundary for file system access.

**Negative**: Docker Compose configuration to maintain; inter-service communication overhead; volume sharing complexity; higher total memory usage.

| Container | Port | Tech | Responsibility |
|-----------|------|------|----------------|
| frontend | 3000 | Next.js 14 | UI, WebSocket client |
| backend | 8000 | FastAPI | Query orchestration, LLM integration |
| indexer | 8002 | Python + FAISS | Embedding generation, index builds |
| mcp_server | 8001 | FastAPI | Secure file operations |

**Lesson**: The indexer separation proved its value every time a project was re-indexed — queries continued at full speed with no degradation.

---

# ADR-010: Tree-sitter for AST Extraction

**Status**: Accepted
**Date**: 2025-08-15
**Deciders**: Sam Ballington

## Context

CodeWise needs to parse source code in 10+ programming languages to extract symbols (functions, classes, variables) and relationships (calls, imports, inheritance). Building custom parsers for each language would be prohibitively expensive and error-prone.

## Decision

Use tree-sitter as the primary AST parsing engine via the `tree-sitter-language-pack` Python package, with language-specific fallbacks (e.g., Esprima for JavaScript) when tree-sitter grammars are unavailable.

## Options Considered

1. **Custom regex-based parsers** per language — Cons: Fragile, misses edge cases, doesn't understand scope/nesting, unmaintainable at 10+ languages.

2. **Language-specific AST libraries** (Python `ast`, Esprima for JS, etc.) — Pros: High accuracy per language. Cons: Different API per language, significant implementation effort per language.

3. **Tree-sitter** (Chosen) — Pros: Unified API across 100+ languages, incremental parsing, battle-tested (used by GitHub, Neovim, Helix), concrete syntax tree. Cons: Grammar files can be large; some languages have incomplete grammars.

## Rationale

Tree-sitter provided the only path to supporting 10+ languages with a unified parsing interface. The `tree-sitter-language-pack` package bundles pre-compiled grammars for all common languages, eliminating the need to compile grammars from source. The fallback strategy (tree-sitter → language-specific library → regex patterns) ensures graceful degradation when grammars are incomplete.

## Consequences

**Positive**: Single API for parsing Python, Java, JavaScript, TypeScript, Go, Rust, Swift, C#, PHP, Ruby, Kotlin; incremental parsing enables efficient re-parsing of modified files; active community maintains grammars for new language versions.

**Negative**: `tree-sitter-language-pack` adds significant package size; some languages have incomplete grammars (handled by fallback strategy).

The two-pass AST-to-graph pipeline: **Pass 1** (SymbolCollector) discovers all symbols using tree-sitter; **Pass 2** (RelationshipExtractor) resolves relationships across files.

Supported: `*.py, *.js, *.ts, *.jsx, *.tsx, *.java, *.cpp, *.c, *.h, *.cs, *.php, *.rb, *.go, *.rs, *.swift, *.m, *.kt`

**Lesson**: Tree-sitter was essential for the multi-language promise. No other approach scales to 10+ languages with a unified interface.

---

# ADR-012: FastAPI over Flask/Django

**Status**: Accepted
**Date**: 2025-07-10
**Deciders**: Sam Ballington

## Context

The backend needs to handle WebSocket connections for real-time streaming, REST endpoints for project management and monitoring, and async I/O for concurrent operations (LLM API calls, file system operations, database queries).

## Decision

Use FastAPI as the backend framework.

## Options Considered

1. **Flask** — Pros: Simple, widely known. Cons: No native async support; WebSocket requires extensions; no built-in request validation.

2. **Django** — Pros: Batteries included, ORM. Cons: Heavy for this use case; synchronous by default; ORM unnecessary with SQLite + custom schema.

3. **FastAPI** (Chosen) — Pros: Native async/await, built-in WebSocket support, Pydantic validation, automatic OpenAPI docs, excellent performance with uvicorn ASGI. Cons: Smaller ecosystem than Flask/Django.

## Rationale

FastAPI's native async support was the deciding factor. The query pipeline is inherently asynchronous: receive WebSocket message → run discovery pipeline → query vector store → call LLM API (streaming) → stream response back. Flask would require threading or greenlet patches; Django would require Django Channels. FastAPI handles it natively.

## Consequences

**Positive**: Native async WebSocket handling without extensions; Pydantic models provide automatic request validation; auto-generated OpenAPI docs at `/docs` and `/redoc`; excellent performance with uvicorn.

**Negative**: Less "batteries included" than Django (no admin panel, no ORM); smaller ecosystem of third-party extensions.

FastAPI handles all CodeWise's needs: WebSocket endpoint at `/ws`, 20+ REST endpoints for projects/cache/validation/indexer management, Pydantic models for all request/response schemas, router-based organization.

**Lesson**: Straightforward choice for an async-first Python backend — FastAPI is the de facto standard for this use case.
