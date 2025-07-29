# CodeWise Developer Onboarding Assistant - Project Overview

## Project Purpose

CodeWise is a developer tool that uses AI to help programmers understand unfamiliar codebases through natural language queries. The primary use case is reducing the time it takes new team members to become productive in existing projects.

### Problem Statement
When developers join new teams or work on unfamiliar codebases, they typically spend 2-3 weeks learning how the system works, where functionality is implemented, and how components interact. This involves reading documentation (often outdated), exploring code manually, and asking senior developers questions.

### Solution Approach
CodeWise indexes a codebase and allows developers to ask questions like "How does user authentication work?" or "Show me all database entities" and receive contextual answers with relevant code snippets and explanations.

### Target Users
- New developers joining existing teams
- Contractors working on unfamiliar projects
- Code reviewers needing context on changes
- Open source contributors exploring projects

## Technical Implementation

### Current Architecture
The system consists of four main components:

1. **Indexer Service** (Python)
   - Parses source code using AST (Abstract Syntax Tree) analysis
   - Creates semantic chunks preserving code context
   - Generates vector embeddings using sentence-transformers
   - Builds BM25 keyword index for exact term matching

2. **Backend API** (FastAPI)
   - Handles WebSocket connections for real-time streaming
   - Manages multiple AI providers (Cerebras, OpenAI, Kimi)
   - Implements hybrid search combining vector similarity and keyword matching
   - Contains agent system with tool calling for code exploration

3. **Frontend** (Next.js/React)
   - Real-time chat interface with streaming responses
   - Project selection and context management
   - Code syntax highlighting and formatting

4. **Search Engine** (Hybrid)
   - Vector search using FAISS for semantic similarity
   - BM25 index for exact keyword matching
   - Result fusion and ranking algorithms
   - Context assembly and token optimization

### Key Technical Components

**AST-Aware Chunking:**
- Parses Python, JavaScript, TypeScript source code
- Preserves function/class boundaries and context
- Maintains import statements and decorators
- Creates searchable chunks while preserving semantic meaning

**Agent System:**
- Uses LLM tool calling to decide between different search strategies
- Tools include: code_search, list_entities, file_glimpse, read_file
- Implements investigation patterns (find entities → examine files → explain relationships)
- Project context isolation to prevent cross-contamination

**Hybrid Search:**
- Combines vector embeddings (semantic) with BM25 (keyword) search
- Query analysis to determine optimal search strategy
- Result fusion using weighted scoring
- Context re-ranking based on relevance and token limits

### Core Demo Scenarios
1. **Authentication Flow Analysis** - Trace complete login process across multiple files
2. **Database Schema Exploration** - Map all entities and relationships
3. **Impact Analysis** - Identify what breaks when modifying specific components
4. **Architecture Overview** - Explain high-level system structure and patterns

## Proposed Metrics

### Technical Performance Metrics
- **Response Time:** 95th percentile under 3 seconds for standard queries
- **Search Accuracy:** Relevant results in top 5 for 90% of queries
- **System Reliability:** Zero crashes during demo scenarios
- **Index Coverage:** 98%+ of source files successfully processed and searchable

### Demo Scenario Success Metrics
- **Consistency:** Same query produces similar quality responses across multiple runs
- **Completeness:** Each demo scenario provides comprehensive answers with code examples
- **Accuracy:** Technical explanations are factually correct for the codebase
- **Relevance:** Returned code snippets directly relate to the query

### Portfolio Effectiveness Metrics
- **Demo Quality:** 2-minute video effectively showcases key capabilities
- **Code Quality:** Clean, well-documented code with comprehensive test coverage
- **Documentation:** README and technical docs clearly explain architecture and usage
- **Deployment:** Live system accessible and functional for recruiter evaluation

### User Experience Metrics
- **Interface Responsiveness:** All UI interactions provide immediate feedback
- **Error Handling:** Graceful degradation with helpful error messages
- **Mobile Compatibility:** Functional on different screen sizes and devices
- **Loading States:** Clear progress indicators during processing

## Technical Challenges and Solutions

### Challenge 1: Agent Tool Selection Reliability
**Problem:** Current agent inconsistently chooses appropriate tools, sometimes failing to use entity discovery for database queries.
**Solution:** Simplify tool selection logic with explicit query classification and fallback strategies.

### Challenge 2: Response Consistency
**Problem:** Same query can produce different quality responses depending on LLM behavior.
**Solution:** Implement response templates, quality scoring, and retry logic for subpar responses.

### Challenge 3: Performance at Scale
**Problem:** Large codebases may have slow indexing and query response times.
**Solution:** Optimize indexing pipeline, implement caching, and use streaming responses for better perceived performance.

### Challenge 4: Framework-Specific Understanding
**Problem:** Generic search may miss framework-specific patterns (Spring Boot annotations, Django ORM).
**Solution:** Enhanced AST parsing with framework-aware chunking and specialized entity detection.

## Success Criteria

### Minimum Viable Demo
- All 4 core demo scenarios work reliably
- Professional UI with proper error handling
- Sub-3 second response times for standard queries
- Clean, deployable codebase with documentation

### Portfolio Excellence
- Compelling demo video showcasing technical depth
- Comprehensive README with architecture diagrams
- Live deployment accessible to recruiters
- Evidence of AI/ML integration, full-stack development, and system design skills

### Technical Validation
- Comprehensive test suite with >80% coverage
- Performance benchmarks meeting target metrics
- Error handling covering edge cases and failures
- Code quality suitable for professional review

This project demonstrates advanced technical skills in AI/ML, full-stack development, and system architecture while solving a real problem that developers face daily.