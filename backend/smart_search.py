#!/usr/bin/env python3
"""
Smart Search Tool for CodeWise - Simplified 3-Tool Architecture
Combines vector search, BM25 keyword search, and entity discovery into a single intelligent interface.

This replaces the complex 8-tool system with one powerful tool that automatically:
- Detects query intent (entity, file, general)
- Routes to appropriate search strategies
- Combines results intelligently
- Returns structured, relevant results
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

try:
    from hybrid_search import HybridSearchEngine, SearchResult
    from bm25_index import BM25Index, BM25Result
    from vector_store import get_vector_store
    from discovery_pipeline import DiscoveryPipeline
    from path_resolver import PathResolver
    from directory_filters import get_project_from_path
    from project_context import get_context_manager
    from file_content_cache import get_global_content_cache
except ImportError:
    from backend.hybrid_search import HybridSearchEngine, SearchResult
    from backend.bm25_index import BM25Index, BM25Result
    from backend.vector_store import get_vector_store
    from backend.discovery_pipeline import DiscoveryPipeline
    from backend.path_resolver import PathResolver
    from directory_filters import get_project_from_path
    from project_context import get_context_manager
    from file_content_cache import get_global_content_cache
try:
    from query_context import QueryExecutionContext
except ImportError:
    from backend.query_context import QueryExecutionContext
try:
    # Phase 3 components now replaced by SDK native reasoning
    # QueryClassifier eliminated - SDK handles intent classification
    Phase3QueryIntent = None
    QueryAnalysis = None
except ImportError:
    from query_context import QueryExecutionContext
    # Phase 3 components now replaced by SDK native reasoning
    Phase3QueryIntent = None
    QueryAnalysis = None
try:
    from backend.context.code_annotator import CodeLensAnnotator
    from storage.database_manager import DatabaseManager
except ImportError:
    try:
        from context.code_annotator import CodeLensAnnotator
        from storage.database_manager import DatabaseManager  
    except ImportError:
        # Graceful fallback if Phase 3 components not available
        CodeLensAnnotator = None
        DatabaseManager = None

logger = logging.getLogger(__name__)


# Use Phase 3 QueryIntent for consistency
QueryIntent = Phase3QueryIntent


@dataclass
class SmartSearchResult:
    """Enhanced search result with intent and strategy metadata"""
    chunk_id: int
    file_path: str
    snippet: str
    relevance_score: float
    query_intent: QueryIntent
    search_strategy: List[str]  # Which search methods were used
    matched_terms: List[str]
    metadata: Dict[str, Any]
    confidence: float  # Confidence in result relevance (0-1)


class QueryIntentAnalyzer:
    """Analyzes queries to determine intent and optimal search strategy"""
    
    def __init__(self):
        # Entity-related keywords
        self.entity_keywords = {
            'entity', 'entities', 'model', 'models', 'database', 'db', 'table', 'tables',
            'schema', 'schemas', 'orm', '@entity', 'create table', 'user model',
            'data model', 'entity class', 'jpa', 'hibernate', 'sqlalchemy',
            'django model', 'mongoose', 'sequelize'
        }
        
        # File-related keywords
        self.file_keywords = {
            'file', 'files', 'directory', 'folder', 'structure', 'organization',
            'config', 'configuration', 'package.json', 'requirements.txt',
            'dockerfile', 'readme', 'main.py', 'index.js', 'app.py'
        }
        
        # Architecture-related keywords
        self.architecture_keywords = {
            'architecture', 'overview', 'structure', 'design', 'system',
            'components', 'modules', 'organization', 'flow', 'pattern',
            'framework', 'stack', 'tech stack', 'dependencies'
        }
        
        # Technical patterns that suggest specific intent (mapped to Phase 3 QueryIntent)
        self.technical_patterns = {
            QueryIntent.SPECIFIC_SYMBOL: [
                r'@Entity\b', r'CREATE\s+TABLE', r'class\s+\w+\s*\(.*Model\)',
                r'models?\.\w+', r'database\s+schema', r'entity\s+relationship',
                r'\.\w{2,4}\b', r'/(src|app|lib|config)/', r'package\.json',
                r'requirements\.txt', r'Dockerfile', r'\.env'
            ],
            QueryIntent.STRUCTURAL: [
                r'main\s+entry\s+point', r'application\s+structure',
                r'system\s+design', r'tech\s+stack'
            ]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine intent and search strategy
        
        Returns:
            Dict containing:
            - intent: QueryIntent enum
            - confidence: float (0-1)
            - search_strategies: List[str] - recommended search methods
            - query_terms: List[str] - extracted terms
            - entity_hints: List[str] - potential entity names
            - file_hints: List[str] - potential file names/types
        """
        query_lower = query.lower()
        query_terms = re.findall(r'\b\w+\b', query_lower)
        
        # Calculate intent scores (mapped to Phase 3 QueryIntent)
        intent_scores = {
            QueryIntent.SPECIFIC_SYMBOL: self._calculate_entity_score(query_lower, query_terms) + self._calculate_file_score(query_lower, query_terms),
            QueryIntent.STRUCTURAL: self._calculate_architecture_score(query_lower, query_terms),
            QueryIntent.CONCEPTUAL: 0.4,  # For semantic queries
            QueryIntent.EXPLORATORY: 0.3,  # Default baseline
            QueryIntent.DEBUGGING: 0.2  # Lower baseline
        }
        
        # Add pattern-based scoring
        for intent, patterns in self.technical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    intent_scores[intent] += 0.3
        
        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[primary_intent]
        
        # Extract hints
        entity_hints = self._extract_entity_hints(query)
        file_hints = self._extract_file_hints(query)
        
        # Determine search strategies based on intent
        search_strategies = self._determine_search_strategies(
            primary_intent, confidence, query_lower
        )
        
        return {
            'intent': primary_intent,
            'confidence': min(confidence, 1.0),
            'search_strategies': search_strategies,
            'query_terms': query_terms,
            'entity_hints': entity_hints,
            'file_hints': file_hints,
            'intent_scores': intent_scores
        }
    
    def _calculate_entity_score(self, query_lower: str, terms: List[str]) -> float:
        """Calculate score for entity intent"""
        score = 0.0
        
        # Direct keyword matches
        for term in terms:
            if term in self.entity_keywords:
                score += 0.4
        
        # Partial matches
        for keyword in self.entity_keywords:
            if keyword in query_lower:
                score += 0.2
        
        return score
    
    def _calculate_file_score(self, query_lower: str, terms: List[str]) -> float:
        """Calculate score for file intent"""
        score = 0.0
        
        # Direct keyword matches
        for term in terms:
            if term in self.file_keywords:
                score += 0.4
        
        # File extension mentions
        if re.search(r'\.\w{2,4}\b', query_lower):
            score += 0.3
        
        return score
    
    def _calculate_architecture_score(self, query_lower: str, terms: List[str]) -> float:
        """Calculate score for architecture intent"""
        score = 0.0
        
        # Direct keyword matches
        for term in terms:
            if term in self.architecture_keywords:
                score += 0.4
        
        # Architecture question patterns
        if any(phrase in query_lower for phrase in [
            'how does', 'how is', 'explain the', 'what is the',
            'show me the', 'overview of', 'structure of'
        ]):
            score += 0.2
        
        return score
    
    def _extract_entity_hints(self, query: str) -> List[str]:
        """Extract potential entity names from query"""
        hints = []
        
        # Look for capitalized words that might be entity names
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:Entity|Model)?\b',  # UserEntity, User
            r'\b(?:User|Order|Product|Customer|Item|Account|Profile)\b'  # Common entities
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, query)
            hints.extend(matches)
        
        return list(set(hints))  # Remove duplicates
    
    def _extract_file_hints(self, query: str) -> List[str]:
        """Extract potential file names/types from query"""
        hints = []
        
        # File extensions
        ext_matches = re.findall(r'\b\w+\.\w{2,4}\b', query)
        hints.extend(ext_matches)
        
        # Common file names
        file_patterns = [
            r'\b(?:main|index|app|config|settings|models|views|controllers?)\.\w+',
            r'\b(?:package\.json|requirements\.txt|Dockerfile|README\.md)\b'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            hints.extend(matches)
        
        return list(set(hints))
    
    def _determine_search_strategies(self, intent: QueryIntent, confidence: float, query: str) -> List[str]:
        """Determine optimal search strategies based on intent"""
        strategies: List[str] = []

        if intent == QueryIntent.ENTITY:
            strategies = ['entity_discovery', 'vector_search', 'bm25_search']
        elif intent == QueryIntent.FILE:
            strategies = ['file_search', 'vector_search', 'bm25_search']
        elif intent == QueryIntent.ARCHITECTURE:
            strategies = ['vector_search', 'structure_analysis', 'bm25_search']
        else:  # GENERAL
            strategies = ['vector_search', 'bm25_search']

        # Heuristic: if the query likely targets an exact filename/extension or uses quotes, prefer hybrid
        q = query.lower()
        has_exact_filename = bool(re.search(r"\b\w+\.(xml|properties|gradle|md|yml|yaml|json|toml|ini|cfg)\b", q))
        has_quotes = ('"' in query) or ("'" in query)

        if has_exact_filename or has_quotes or (intent == QueryIntent.FILE and confidence >= 0.6):
            if 'hybrid_search' not in strategies:
                strategies.insert(0, 'hybrid_search')
        elif confidence > 0.7:
            strategies.insert(0, 'hybrid_search')

        return strategies


class EntityDiscovery:
    """Enhanced entity discovery with pattern matching and metadata extraction"""
    
    def __init__(self):
        self.entity_patterns = {
            'jpa': re.compile(r'@Entity\b.*?class\s+(\w+)', re.IGNORECASE | re.DOTALL),
            'django': re.compile(r'class\s+(\w+)\s*\([^)]*Model[^)]*\)', re.IGNORECASE),
            'sqlalchemy': re.compile(r'class\s+(\w+)\s*\([^)]*Base[^)]*\)', re.IGNORECASE),
            'sequelize': re.compile(r'const\s+(\w+)\s*=.*?sequelize\.define', re.IGNORECASE),
            'mongoose': re.compile(r'const\s+(\w+)Schema\s*=.*?mongoose\.Schema', re.IGNORECASE),
            'sql': re.compile(r'CREATE\s+TABLE\s+(\w+)', re.IGNORECASE)
        }
    
    async def discover_entities(self, workspace_path: str = "/workspace", hints: List[str] = None) -> List[Dict[str, Any]]:
        """
        Discover database entities in the workspace
        
        Args:
            workspace_path: Path to search for entities
            hints: Optional hints about entity names to look for
            
        Returns:
            List of entity information dictionaries
        """
        entities = []
        workspace = Path(workspace_path)
        
        # File extensions that commonly contain entities
        entity_extensions = {'.py', '.java', '.kt', '.js', '.ts', '.sql'}
        
        try:
            for file_path in workspace.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix in entity_extensions and
                    not any(ignore in str(file_path) for ignore in ['.git', 'node_modules', '__pycache__', '.pytest_cache'])):
                    
                    entities.extend(await self._analyze_file_for_entities(file_path, hints))
        
        except Exception as e:
            logger.error(f"Error during entity discovery: {e}")
        
        # Sort by confidence and remove duplicates
        unique_entities = {}
        for entity in entities:
            key = f"{entity['name']}:{entity['file_path']}"
            if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
                unique_entities[key] = entity
        
        return sorted(unique_entities.values(), key=lambda x: x['confidence'], reverse=True)
    
    async def _analyze_file_for_entities(self, file_path: Path, hints: List[str] = None) -> List[Dict[str, Any]]:
        """Analyze a single file for entity patterns"""
        entities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(4096)  # Read first 4KB for performance
            
            for framework, pattern in self.entity_patterns.items():
                matches = pattern.findall(content)
                for match in matches:
                    entity_name = match if isinstance(match, str) else match[0]
                    
                    # Calculate confidence based on hints and context
                    confidence = self._calculate_entity_confidence(
                        entity_name, content, framework, hints
                    )
                    
                    entities.append({
                        'name': entity_name,
                        'framework': framework,
                        'file_path': str(file_path),
                        'confidence': confidence,
                        'snippet': self._extract_entity_snippet(content, entity_name)
                    })
        
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
        
        return entities
    
    def _calculate_entity_confidence(self, name: str, content: str, framework: str, hints: List[str] = None) -> float:
        """Calculate confidence score for detected entity"""
        confidence = 0.6  # Base confidence
        
        # Boost for hint matches
        if hints:
            for hint in hints:
                if hint.lower() in name.lower():
                    confidence += 0.3
        
        # Framework-specific boosts
        framework_boosts = {
            'jpa': 0.2,
            'django': 0.2,
            'sqlalchemy': 0.2,
            'sql': 0.3  # SQL DDL is very confident
        }
        confidence += framework_boosts.get(framework, 0.1)
        
        # Content analysis boosts
        if '@Table' in content or 'CREATE TABLE' in content.upper():
            confidence += 0.1
        if 'id' in content.lower() and ('primary' in content.lower() or '@Id' in content):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_entity_snippet(self, content: str, entity_name: str) -> str:
        """Extract a relevant snippet around the entity definition"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if entity_name in line:
                # Extract 3 lines before and after
                start = max(0, i - 2)
                end = min(len(lines), i + 4)
                snippet_lines = lines[start:end]
                return '\n'.join(snippet_lines)
        
        return f"Entity: {entity_name}"


class SmartSearchEngine:
    """
    Phase 3 Complete: Intelligence-powered search with adaptive classification,
    KG expansion, and Code Lens integration
    """
    
    def __init__(self):
        # Legacy components (Phase 1 & 2)
        self.hybrid_search = HybridSearchEngine()
        self.entity_discovery = EntityDiscovery()
        self.path_resolver = PathResolver()
        self.discovery_pipeline = DiscoveryPipeline(self.path_resolver)
        self.content_cache = get_global_content_cache()
        
        # Phase 3: Intelligence layer
        self.vector_store = get_vector_store()
        try:
            self.db_manager = DatabaseManager()
            # QueryClassifier eliminated - SDK now handles all intent classification
            self.query_classifier = None  # SDK native reasoning replaces this
            self.code_annotator = CodeLensAnnotator(self.db_manager)
            self.phase3_available = True
            logger.info("✅ Phase 3 components initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Phase 3 components not available, falling back to legacy: {e}")
            self.phase3_available = False
            # Fallback to legacy analyzer
            self.intent_analyzer = QueryIntentAnalyzer()
        
        # Search configuration
        self.max_results = 15
        self.min_confidence = 0.3
    
    async def search(self, query: str, k: int = 10, strategy_override: List[str] = None, mentioned_projects: List[str] = None) -> Dict[str, Any]:
        """
        Perform intelligent search combining multiple strategies
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            strategy_override: Optional list of strategies to force use
            mentioned_projects: Optional list of projects to filter results by
            
        Returns:
            Dict containing:
            - results: List[SmartSearchResult]
            - query_analysis: Dict with intent analysis
            - search_strategies_used: List[str]
            - total_results: int
            - execution_time: float
        """
        import time
        start_time = time.time()
        
        logger.debug(f"🧠 SMART SEARCH: '{query}' (k={k})")
        
        # Analyze query intent
        query_analysis = self.intent_analyzer.analyze_query(query)
        logger.debug(f"Query intent: {query_analysis['intent'].value} (confidence: {query_analysis['confidence']:.2f})")
        
        # Determine search strategies
        strategies = strategy_override or query_analysis['search_strategies']
        logger.debug(f"Search strategies: {strategies}")
        
        # Execute search strategies
        all_results = []
        strategies_used = []
        
        for strategy in strategies:
            try:
                results = await self._execute_strategy(strategy, query, query_analysis)
                if results:
                    all_results.extend(results)
                    strategies_used.append(strategy)
                    logger.debug(f"Strategy '{strategy}' returned {len(results)} results")
            except Exception as e:
                logger.error(f"Strategy '{strategy}' failed: {e}")
        
        # Merge and rank results
        final_results = self._merge_and_rank_results(all_results, query_analysis)
        
        # Apply project filtering if mentioned_projects is provided
        if mentioned_projects:
            project_filtered_results = []
            context_manager = get_context_manager()
            
            # CRITICAL FIX: Set project context before filtering
            if mentioned_projects and len(mentioned_projects) > 0:
                primary_project = mentioned_projects[0]
                logger.info(f"🔧 CONTEXT FIX: Setting project context to '{primary_project}' before filtering")
                context_manager.set_project_context(primary_project, mentioned_projects)
            
            # DEBUG: Check current context state AFTER setting
            current_context = context_manager.get_current_context()
            logger.info(f"🔍 FILTER DEBUG: mentioned_projects={mentioned_projects}, current_context={current_context.name if current_context else None}")
            
            # Optimized filtering with minimal logging
            logger.debug(f"🔍 FILTERING DEBUG: Processing {len(final_results)} results for project filtering")
            
            # Cache project determinations to avoid redundant calls
            file_projects = {}
            filtered_out_count = 0
            
            for result in final_results:
                # Normalize file path to full workspace path for filtering
                file_path = result.file_path
                if not file_path.startswith('/workspace/'):
                    file_path = f"/workspace/{file_path.lstrip('/')}"
                
                # Cache project determination to avoid redundant calls
                if file_path not in file_projects:
                    file_projects[file_path] = context_manager.get_context_for_file(file_path)
                
                file_project = file_projects[file_path]
                
                # Use cached project info instead of calling is_file_in_current_context
                in_context = (file_project == current_context.name if current_context else True)
                
                if not in_context:
                    filtered_out_count += 1
                    logger.debug(f"🚫 FILTERED OUT: {file_path} (project: {file_project})")
                else:
                    logger.debug(f"✅ INCLUDED: {file_path} (project: {file_project})")
                    project_filtered_results.append(result)
            
            logger.info(f"🎯 PROJECT FILTERING COMPLETE: {len(final_results)} -> {len(project_filtered_results)} results for projects {mentioned_projects}")
            final_results = project_filtered_results
        
        # Apply confidence filters and limits
        filtered_results = [
            result for result in final_results 
            if result.confidence >= self.min_confidence
        ][:k]
        
        execution_time = time.time() - start_time
        
        logger.info(f"🧠 SMART SEARCH COMPLETE: {len(filtered_results)} results in {execution_time:.2f}s")
        
        # NEW: Discovery Pipeline Enhancement for Task 5 Extension (optimized)
        # Skip discovery pipeline for architecture queries to improve performance
        if query_analysis['intent'] == QueryIntent.ARCHITECTURE:
            logger.debug("🚫 DISCOVERY PIPELINE: Skipped for architecture query (performance optimization)")
            auto_examine_files = []
        else:
            auto_examine_files = await self._enhance_with_discovery_pipeline(query, filtered_results, query_analysis)
        
        return {
            'results': filtered_results,
            'query_analysis': query_analysis,
            'search_strategies_used': strategies_used,
            'total_results': len(all_results),
            'execution_time': execution_time,
            'auto_examine_files': auto_examine_files or []  # Files recommended for auto-examination
        }
    
    async def search_with_context(
        self,
        query: str,
        context: QueryExecutionContext,
        k: int = 10,
        strategy_override: List[str] = None,
        mentioned_projects: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform intelligent search with shared query context (PERFORMANCE OPTIMIZED).
        
        This is the KEY ARCHITECTURAL FIX - uses context to prevent multiple
        discovery pipeline runs within the same query.
        
        Args:
            query: Search query string
            context: Shared query execution context
            k: Maximum number of results to return
            strategy_override: Optional list of strategies to force use
            mentioned_projects: Optional list of projects to filter results by
            
        Returns:
            Dict containing search results and metadata
        """
        import time
        start_time = time.perf_counter()
        
        logger.info(f"🧠 CONTEXT-AWARE SEARCH: '{query}' with context {context.query_id[:8]}")
        
        # Mark search tool as executing
        await context.mark_tool_executed("smart_search_start")
        
        # Analyze query intent (same as legacy method)
        query_analysis = self.intent_analyzer.analyze_query(query)
        logger.debug(f"Query intent: {query_analysis['intent'].value} (confidence: {query_analysis['confidence']:.2f})")
        
        # Determine search strategies (same as legacy method)
        strategies = strategy_override or query_analysis['search_strategies']
        logger.debug(f"Search strategies: {strategies}")
        
        # Execute search strategies (same as legacy method)
        all_results = []
        strategies_used = []
        
        for strategy in strategies:
            try:
                results = await self._execute_strategy(strategy, query, query_analysis)
                if results:
                    all_results.extend(results)
                    strategies_used.append(strategy)
                    logger.debug(f"Strategy '{strategy}' returned {len(results)} results")
            except Exception as e:
                logger.error(f"Strategy '{strategy}' failed: {e}")
        
        # Merge and rank results (same as legacy method)
        final_results = self._merge_and_rank_results(all_results, query_analysis)
        
        # Apply project filtering if mentioned_projects is provided
        if mentioned_projects:
            # Ensure project context is set prior to filtering for consistent behavior
            try:
                context_manager = get_context_manager()
                primary_project = mentioned_projects[0]
                context_manager.set_project_context(primary_project, mentioned_projects)
            except Exception as e:
                logger.debug(f"Project context setup skipped due to error: {e}")

            project_filtered_results: List[SmartSearchResult] = []
            filtered_out_count = 0

            for result in final_results:
                # Normalize file path to absolute workspace path for reliable context checks
                file_path = result.file_path
                if not file_path.startswith('/workspace/'):
                    file_path = f"/workspace/{file_path.lstrip('/')}"

                # Use centralized context manager for consistent scoping
                in_context = True
                try:
                    in_context = get_context_manager().is_file_in_current_context(file_path)
                except Exception:
                    # Fallback: best-effort project extraction if context manager not available
                    result_project = self._extract_project_from_path(file_path)
                    in_context = result_project in set(mentioned_projects)

                if in_context:
                    project_filtered_results.append(result)
                    logger.debug(f"✅ INCLUDED: {file_path}")
                else:
                    filtered_out_count += 1
                    logger.debug(f"🚫 FILTERED OUT: {file_path}")

            if project_filtered_results:
                logger.info(
                    f"🎯 PROJECT FILTERING COMPLETE: {len(final_results)} -> {len(project_filtered_results)} "
                    f"results for projects {mentioned_projects} (excluded: {filtered_out_count})"
                )
                filtered_results = project_filtered_results
            else:
                # Keep results to avoid empty responses, but downgrade message severity
                logger.debug(
                    f"PROJECT FILTERING yielded 0 matches for projects {mentioned_projects}; "
                    f"returning unfiltered results ({len(final_results)})"
                )
                filtered_results = final_results
        else:
            filtered_results = final_results
        
        # Limit results
        if len(filtered_results) > k:
            filtered_results = filtered_results[:k]
        
        execution_time = time.perf_counter() - start_time
        
        logger.info(f"🧠 SMART SEARCH COMPLETE: {len(filtered_results)} results in {execution_time:.2f}s")
        
        # Context-aware discovery pipeline enhancement (THE KEY FIX)
        auto_examine_files = await self._enhance_with_discovery_pipeline_context_aware(
            query, filtered_results, query_analysis, context
        )
        
        # Build result dict
        tool_result = {
            'results': filtered_results,
            'query_analysis': query_analysis,
            'search_strategies_used': strategies_used,
            'total_results': len(all_results),
            'execution_time': execution_time,
            'auto_examine_files': auto_examine_files or []
        }
        
        # Mark tool execution complete with timing
        await context.mark_tool_executed("smart_search", tool_result, execution_time)
        
        return tool_result
    
    async def _enhance_with_discovery_pipeline_context_aware(
        self,
        query: str,
        search_results: List[SmartSearchResult],
        query_analysis: Dict,
        context: QueryExecutionContext
    ) -> Optional[List[str]]:
        """
        Context-aware discovery pipeline enhancement.
        
        This method uses the query context to ensure discovery pipeline runs
        ONCE per query, not once per tool execution.
        
        This is the CORE ARCHITECTURAL FIX.
        """
        try:
            logger.info(f"🚀 DISCOVERY PIPELINE: Starting enhancement (context: {context.query_id[:8]})")
            
            # Check if we should run discovery enhancement
            if not self._should_enhance_discovery(query, search_results, query_analysis):
                logger.info("🚫 DISCOVERY PIPELINE: Skipped - conditions not met")
                return None
            
            # Context-aware discovery execution (THE KEY FIX)
            enhanced_results = await context.get_or_run_discovery(
                self._run_discovery_pipeline_once,
                search_results,
                query,
                query_analysis
            )
            
            # Extract auto-examine files from results
            if enhanced_results and 'recommended_examinations' in enhanced_results:
                auto_examine_files = enhanced_results['recommended_examinations']
                logger.info(f"🔍 DISCOVERY PIPELINE: Found {len(auto_examine_files)} files to auto-examine")
                logger.info(f"🔍 DISCOVERY PIPELINE: Auto-examine files: {auto_examine_files}")
                return auto_examine_files
            else:
                logger.info("🔍 DISCOVERY PIPELINE: No files recommended for auto-examination")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ DISCOVERY PIPELINE: Context-aware enhancement failed: {e}")
            return None
    
    async def _run_discovery_pipeline_once(
        self,
        search_results: List[SmartSearchResult],
        query: str,
        query_analysis: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Run the discovery pipeline exactly once per query context.
        
        This method is called by the context's get_or_run_discovery(),
        ensuring it runs only once even if multiple tools call it.
        """
        logger.info(f"🔍 DISCOVERY PIPELINE: Executing (THIS SHOULD ONLY HAPPEN ONCE PER QUERY)")
        
        # Deduplicate search results by file_path before processing
        unique_results = {}
        for result in search_results:
            if result.file_path not in unique_results:
                unique_results[result.file_path] = result
            elif result.confidence > unique_results[result.file_path].confidence:
                unique_results[result.file_path] = result
        
        deduplicated_results = list(unique_results.values())
        logger.info(f"📄 DISCOVERY DOC FILES: {len(search_results)} -> {len(deduplicated_results)} unique files")
        
        # Convert search results to the format expected by discovery pipeline
        formatted_results = []
        for result in deduplicated_results:
            # For documentation files, try to get full content instead of just snippets
            content = result.snippet
            if self._is_documentation_file(result.file_path):
                full_content = await self._get_full_file_content(result.file_path)
                if full_content:
                    content = full_content
                    logger.info(f"📄 DISCOVERY: Got full content for {result.file_path} ({len(content)} chars)")
            
            formatted_results.append({
                'file_path': result.file_path,
                'content': content,
                'confidence': result.confidence
            })
        
        # Determine query type for discovery
        query_type = self._classify_query_type_for_discovery(query, query_analysis)
        
        # Run discovery pipeline
        enhanced_results = await self.discovery_pipeline.enhance_search_results(
            formatted_results,
            query_type,
            []  # No specific projects filter in smart_search
        )
        
        # Return structured results
        if enhanced_results and enhanced_results.recommended_examinations:
            metadata = enhanced_results.discovery_metadata
            logger.info(
                f"📊 DISCOVERY PIPELINE: {metadata.total_references_found} refs found, "
                f"{metadata.discovery_time_ms:.1f}ms"
            )
            
            return {
                'recommended_examinations': enhanced_results.recommended_examinations,
                'metadata': {
                    'total_references': metadata.total_references_found,
                    'discovery_time_ms': metadata.discovery_time_ms,
                    'files_processed': len(formatted_results)
                }
            }
        else:
            return {
                'recommended_examinations': [],
                'metadata': {
                    'total_references': 0,
                    'discovery_time_ms': 0.0,
                    'files_processed': len(formatted_results)
                }
            }
    
    async def _execute_strategy(self, strategy: str, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute a specific search strategy"""
        if strategy == 'hybrid_search':
            return await self._hybrid_search_strategy(query, analysis)
        elif strategy == 'entity_discovery':
            return await self._entity_discovery_strategy(query, analysis)
        elif strategy == 'vector_search':
            return await self._vector_search_strategy(query, analysis)
        elif strategy == 'bm25_search':
            return await self._bm25_search_strategy(query, analysis)
        elif strategy == 'file_search':
            return await self._file_search_strategy(query, analysis)
        elif strategy == 'structure_analysis':
            return await self._structure_analysis_strategy(query, analysis)
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return []
    
    async def _hybrid_search_strategy(self, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute hybrid search strategy"""
        try:
            search_results = await self.hybrid_search.search(query, k=self.max_results)
            return [
                SmartSearchResult(
                    chunk_id=result.chunk_id,
                    file_path=result.file_path,
                    snippet=result.snippet,
                    relevance_score=result.relevance_score,
                    query_intent=analysis['intent'],
                    search_strategy=['hybrid'],
                    matched_terms=result.matched_terms,
                    metadata={'search_type': result.search_type},
                    confidence=result.relevance_score
                )
                for result in search_results
            ]
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _entity_discovery_strategy(self, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute entity discovery strategy"""
        try:
            entities = await self.entity_discovery.discover_entities(
                hints=analysis.get('entity_hints', [])
            )
            
            results = []
            for i, entity in enumerate(entities[:8]):  # Limit entity results
                results.append(SmartSearchResult(
                    chunk_id=i,
                    file_path=entity['file_path'],
                    snippet=entity['snippet'],
                    relevance_score=entity['confidence'],
                    query_intent=QueryIntent.ENTITY,
                    search_strategy=['entity_discovery'],
                    matched_terms=[entity['name']],
                    metadata={
                        'entity_name': entity['name'],
                        'framework': entity['framework']
                    },
                    confidence=entity['confidence']
                ))
            
            return results
        except Exception as e:
            logger.error(f"Entity discovery failed: {e}")
            return []
    
    async def _vector_search_strategy(self, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute vector search strategy"""
        try:
            vector_store = get_vector_store()
            vector_results = vector_store.query(query, k=self.max_results)
            
            results = []
            for i, result in enumerate(vector_results):
                # Handle both 2-tuple (path, snippet) and 3-tuple (path, snippet, score) formats
                if len(result) == 3:
                    file_path, snippet, vector_score = result
                    score = float(vector_score)  # Use actual score from vector store
                else:
                    file_path, snippet = result
                    score = max(0.1, 1.0 - (i * 0.1))  # Fallback decreasing scores
                results.append(SmartSearchResult(
                    chunk_id=i,
                    file_path=file_path,
                    snippet=snippet,
                    relevance_score=score,
                    query_intent=analysis['intent'],
                    search_strategy=['vector'],
                    matched_terms=[],
                    metadata={'search_type': 'vector'},
                    confidence=score
                ))
            
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _bm25_search_strategy(self, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute BM25 search strategy using HybridSearchEngine's BM25 index."""
        try:
            # Access BM25 index from hybrid engine
            bm25 = getattr(self.hybrid_search, "bm25_index", None)
            if bm25 is None:
                return []

            bm25_results = bm25.search(query, k=self.max_results)

            # Normalize BM25 scores to 0-1 range for confidence
            scores = [r.score for r in bm25_results] if bm25_results else []
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            denom = (max_score - min_score) or 1.0

            results: List[SmartSearchResult] = []
            for r in bm25_results:
                norm = (r.score - min_score) / denom
                results.append(
                    SmartSearchResult(
                        chunk_id=r.chunk_id,
                        file_path=r.file_path,
                        snippet=r.snippet,
                        relevance_score=norm,
                        query_intent=analysis['intent'],
                        search_strategy=['bm25'],
                        matched_terms=r.matched_terms,
                        metadata={'search_type': 'bm25', 'raw_score': r.score},
                        confidence=norm,
                    )
                )

            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def _file_search_strategy(self, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute file search strategy for file-specific queries"""
        try:
            import subprocess
            workspace = Path("/workspace")
            results = []
            
            # Search for files mentioned in query
            file_hints = analysis.get('file_hints', [])
            for hint in file_hints:
                try:
                    cmd = ["find", "/workspace", "-name", f"*{hint}*", "-type", "f"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                        for i, file_path in enumerate(files[:5]):  # Limit to 5 per hint
                            results.append(SmartSearchResult(
                                chunk_id=i,
                                file_path=file_path,
                                snippet=f"File: {Path(file_path).name}",
                                relevance_score=0.8,
                                query_intent=QueryIntent.FILE,
                                search_strategy=['file_search'],
                                matched_terms=[hint],
                                metadata={'file_hint': hint},
                                confidence=0.8
                            ))
                except Exception as e:
                    logger.debug(f"File search for '{hint}' failed: {e}")
            
            return results
        except Exception as e:
            logger.error(f"File search strategy failed: {e}")
            return []
    
    async def _structure_analysis_strategy(self, query: str, analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Execute structure analysis for architecture queries"""
        try:
            # This would analyze project structure
            # For now, return key architectural files
            key_files = [
                'package.json', 'requirements.txt', 'setup.py', 'Dockerfile',
                'main.py', 'app.py', 'index.js', 'server.js', 'manage.py'
            ]
            
            workspace = Path("/workspace")
            results = []
            
            for file_name in key_files:
                matches = list(workspace.rglob(file_name))
                for i, file_path in enumerate(matches[:3]):  # Limit per file type
                    results.append(SmartSearchResult(
                        chunk_id=i,
                        file_path=str(file_path),
                        snippet=f"Architecture file: {file_name}",
                        relevance_score=0.7,
                        query_intent=QueryIntent.ARCHITECTURE,
                        search_strategy=['structure_analysis'],
                        matched_terms=[file_name],
                        metadata={'file_type': 'architectural'},
                        confidence=0.7
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            return []
    
    def _merge_and_rank_results(self, results: List[SmartSearchResult], analysis: Dict[str, Any]) -> List[SmartSearchResult]:
        """Merge results from different strategies and apply intelligent ranking"""
        if not results:
            return []
        
        # Remove duplicates based on file_path + snippet similarity
        unique_results = {}
        for result in results:
            # Create a key based on file path and snippet start
            key = f"{result.file_path}:{result.snippet[:100]}"
            
            if key not in unique_results or result.confidence > unique_results[key].confidence:
                unique_results[key] = result
        
        merged_results = list(unique_results.values())
        
        # Apply intent-based boosting
        intent = analysis['intent']
        for result in merged_results:
            if result.query_intent == intent:
                result.confidence *= 1.2  # Boost matching intent
            
            # Boost entity results for entity queries
            if intent == QueryIntent.ENTITY and 'entity_discovery' in result.search_strategy:
                result.confidence *= 1.3
        
        # Sort by confidence score
        merged_results.sort(key=lambda x: x.confidence, reverse=True)
        
        return merged_results
    
    async def _enhance_with_discovery_pipeline(self, query: str, search_results: List[SmartSearchResult], query_analysis: Dict) -> Optional[List[str]]:
        """
        Enhance search results using the discovery pipeline for README-driven discovery
        Returns list of files recommended for auto-examination
        """
        try:
            logger.info("🚀 DISCOVERY PIPELINE: Starting enhancement")
            
            # Check if we should run discovery enhancement
            if not self._should_enhance_discovery(query, search_results, query_analysis):
                logger.info("🚫 DISCOVERY PIPELINE: Skipped - conditions not met")
                return None
            
            # PERFORMANCE FIX: Deduplicate search results by file_path before processing
            unique_results = {}
            for result in search_results:
                if result.file_path not in unique_results:
                    unique_results[result.file_path] = result
                elif result.confidence > unique_results[result.file_path].confidence:
                    unique_results[result.file_path] = result
            
            deduplicated_results = list(unique_results.values())
            logger.info(f"📄 DISCOVERY DOC FILES: {len(search_results)} -> {len(deduplicated_results)} unique files")
            
            # Convert search results to the format expected by discovery pipeline
            formatted_results = []
            for result in deduplicated_results:
                # For documentation files, try to get full content instead of just snippets
                content = result.snippet
                if self._is_documentation_file(result.file_path):
                    full_content = await self._get_full_file_content(result.file_path)
                    if full_content:
                        content = full_content
                        logger.info(f"📄 DISCOVERY: Got full content for {result.file_path} ({len(content)} chars)")
                
                formatted_results.append({
                    'file_path': result.file_path,
                    'content': content,
                    'confidence': result.confidence
                })
            
            # Determine query type for discovery
            query_type = self._classify_query_type_for_discovery(query, query_analysis)
            
            # Run discovery pipeline
            enhanced_results = await self.discovery_pipeline.enhance_search_results(
                formatted_results, 
                query_type, 
                []  # No specific projects filter in smart_search
            )
            
            # Return files recommended for auto-examination
            if enhanced_results.recommended_examinations:
                logger.info(f"🔍 DISCOVERY PIPELINE: Found {len(enhanced_results.recommended_examinations)} files to auto-examine")
                logger.info(f"🔍 DISCOVERY PIPELINE: Auto-examine files: {enhanced_results.recommended_examinations}")
                
                # Log discovery metadata
                metadata = enhanced_results.discovery_metadata
                logger.info(f"📊 DISCOVERY PIPELINE: {metadata.total_references_found} refs found, "
                           f"{metadata.discovery_time_ms:.1f}ms")
                
                return enhanced_results.recommended_examinations
            else:
                logger.info("🔍 DISCOVERY PIPELINE: No files recommended for auto-examination")
                return None
            
        except Exception as e:
            logger.warning(f"⚠️ DISCOVERY PIPELINE: Enhancement failed: {e}")
            # Graceful fallback - continue without discovery enhancement
            return None
    
    def _should_enhance_discovery(self, query: str, search_results: List[SmartSearchResult], query_analysis: Dict) -> bool:
        """Determine if discovery enhancement should be applied"""
        # Check both intent and query content for architecture-related terms
        intent = query_analysis.get('intent', QueryIntent.GENERAL)
        
        discovery_worthy_intents = [QueryIntent.ARCHITECTURE, QueryIntent.ENTITY, QueryIntent.GENERAL]
        architecture_keywords = ['architecture', 'system', 'components', 'interact', 'structure', 'infinite-kanvas', 'readme']
        
        intent_match = intent in discovery_worthy_intents
        keyword_match = any(keyword in query.lower() for keyword in architecture_keywords)
        
        logger.info(f"🔍 DISCOVERY CONDITIONS: Intent={intent}, IntentMatch={intent_match}, KeywordMatch={keyword_match}")
        
        if not (intent_match or keyword_match):
            logger.info(f"🚫 DISCOVERY SKIPPED: No intent/keyword match")
            return False
        
        # Only enhance if we found documentation files
        has_documentation = any(
            'readme' in result.file_path.lower() or 
            'doc' in result.file_path.lower() or
            result.file_path.endswith('.md')
            for result in search_results
        )
        
        logger.info(f"📄 DISCOVERY DOCS: HasDocs={has_documentation}")
        if has_documentation:
            doc_files = [r.file_path for r in search_results if 'readme' in r.file_path.lower() or r.file_path.endswith('.md')]
            logger.info(f"📄 DISCOVERY DOC FILES: {doc_files}")
        
        return has_documentation
    
    def _classify_query_type_for_discovery(self, query: str, query_analysis: Dict):
        """Convert query analysis to QueryType enum for discovery pipeline"""
        from backend.response_formatter import QueryType
        
        intent = query_analysis.get('intent', QueryIntent.GENERAL)
        
        # If intent is general, analyze query content for specific patterns
        if intent == QueryIntent.GENERAL:
            query_lower = query.lower()
            if any(word in query_lower for word in ['architecture', 'system', 'components', 'interact', 'structure']):
                return QueryType.ARCHITECTURE
            elif any(word in query_lower for word in ['database', 'entities', 'models', 'schema']):
                return QueryType.DATABASE_ENTITIES
            elif any(word in query_lower for word in ['auth', 'login', 'security', 'user']):
                return QueryType.AUTHENTICATION
        
        # Map QueryIntent to QueryType
        intent_mapping = {
            QueryIntent.ARCHITECTURE: QueryType.ARCHITECTURE,
            QueryIntent.ENTITY: QueryType.DATABASE_ENTITIES,
            QueryIntent.FILE: QueryType.GENERAL,
            QueryIntent.GENERAL: QueryType.GENERAL
        }
        
        return intent_mapping.get(intent, QueryType.GENERAL)
    
    def _is_documentation_file(self, file_path: str) -> bool:
        """Check if a file is a documentation file that should be read in full"""
        file_path_lower = file_path.lower()
        return (
            'readme' in file_path_lower or
            file_path_lower.endswith('.md') or
            'doc' in file_path_lower
        )
    
    async def _get_full_file_content(self, file_path: str) -> Optional[str]:
        """Get full content of a file for discovery analysis with caching"""
        try:
            # Construct full workspace path
            if not file_path.startswith('/workspace/'):
                full_path = f"/workspace/{file_path}"
            else:
                full_path = file_path
            
            # PERFORMANCE FIX: Check cache first
            cached_content = self.content_cache.get(full_path)
            if cached_content is not None:
                logger.debug(f"📄 DISCOVERY: Using cached content for {file_path} ({len(cached_content)} chars)")
                return cached_content
            
            # Read the file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cache the content for future use
            self.content_cache.put(full_path, content)
            
            # Limit content size to prevent memory issues
            max_content_size = 50000  # 50KB limit
            if len(content) > max_content_size:
                content = content[:max_content_size] + "\n... [truncated]"
            
            return content
            
        except Exception as e:
            logger.debug(f"Failed to read full content for {file_path}: {e}")
            return None
    
    def get_search_capabilities(self) -> Dict[str, Any]:
        """Return information about search capabilities"""
        return {
            'supported_intents': [intent.value for intent in QueryIntent],
            'search_strategies': [
                'hybrid_search', 'entity_discovery', 'vector_search', 
                'bm25_search', 'file_search', 'structure_analysis'
            ],
            'max_results': self.max_results,
            'min_confidence': self.min_confidence
        }
    
    def _extract_project_from_path(self, file_path: str) -> str:
        """Extract project name from file path using existing utility function"""
        return get_project_from_path(file_path)
    
    # ==================== PHASE 3 ENHANCED SEARCH METHODS ====================
    
    async def search_phase3(self, query: str, k: int = 10, strategy_override: List[str] = None, mentioned_projects: List[str] = None) -> Dict[str, Any]:
        """
        Phase 3: Complete intelligent search with adaptive classification,
        KG expansion, and enhanced code presentation.
        """
        import time
        start_time = time.time()
        
        logger.info(f"🧠 PHASE 3 SMART SEARCH: '{query}' (k={k})")
        
        if not self.phase3_available:
            logger.warning("⚠️ Phase 3 not available, falling back to legacy search")
            return await self.search(query, k, strategy_override, mentioned_projects)
        
        # Phase 3: SDK native reasoning eliminates custom classification
        # All reasoning is now handled by Cerebras SDK - use adaptive search strategy
        logger.info("Using SDK-native intelligent search - no custom classification needed")
        
        # Use adaptive search strategy (eliminates custom query analysis)
        search_results = []
        try:
            # SDK will handle intent analysis through tool selection
            # Use hybrid search as baseline with KG enhancement if available
            if self.db_manager:
                # Try KG-enhanced search first
                search_results = await self._structural_search_phase3(query, k)
                if not search_results:
                    # Fallback to semantic search
                    search_results = await self._semantic_search_enhanced(query, k)
            else:
                # Pure semantic search when KG unavailable
                search_results = await self._semantic_search_enhanced(query, k)
                
        except Exception as e:
            logger.error(f"Phase 3 enhanced search failed: {e}, falling back to legacy")
            return await self.search(query, k, strategy_override, mentioned_projects)
        
        # Phase 3: Enhance results with Code Lens
        enhanced_results = []
        for result in search_results:
            try:
                enhanced_snippet = self._enhance_snippet_with_code_lens(result.snippet, result.file_path)
                enhanced_result = SmartSearchResult(
                    chunk_id=result.chunk_id,
                    file_path=result.file_path,
                    snippet=enhanced_snippet,
                    relevance_score=result.relevance_score,
                    query_intent=query_analysis.intent,
                    search_strategy=[query_analysis.reasoning],
                    matched_terms=getattr(result, 'matched_terms', []),
                    metadata={
                        'query_analysis': query_analysis,
                        'enhancement_applied': 'code_lens',
                        'hierarchical_context': self._extract_hierarchical_context(result)
                    },
                    confidence=result.relevance_score
                )
                enhanced_results.append(enhanced_result)
            except Exception as e:
                logger.warning(f"Failed to enhance result: {e}")
                enhanced_results.append(result)  # Fallback to original
        
        execution_time = time.time() - start_time
        logger.info(f"✅ Phase 3 search completed in {execution_time:.2f}s, {len(enhanced_results)} results")
        
        return {
            'results': enhanced_results,
            'query_analysis': {
                'intent': query_analysis.intent.value,
                'confidence': query_analysis.confidence,
                'reasoning': query_analysis.reasoning,
                'vector_weight': query_analysis.vector_weight,
                'bm25_weight': query_analysis.bm25_weight
            },
            'search_strategies_used': ['phase3_intelligent'],
            'total_results': len(enhanced_results),
            'execution_time': execution_time
        }
    
    def _enhance_snippet_with_code_lens(self, snippet: str, file_path: str) -> str:
        """Add Code Lens annotations to search result snippets"""
        try:
            # Detect if snippet contains code
            if self._is_code_snippet(snippet):
                annotated_snippet = self.code_annotator.annotate_code(snippet, file_path)
                return annotated_snippet
        except Exception as e:
            logger.debug(f"Code lens enhancement failed: {e}")
        
        return snippet  # Return as-is if not code or enhancement fails
    
    def _is_code_snippet(self, snippet: str) -> bool:
        """Determine if snippet contains code"""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'function ', '()', '{', '}',
            '=>', 'const ', 'let ', 'var ', 'public ', 'private ', 'static'
        ]
        return any(indicator in snippet for indicator in code_indicators)
    
    async def _precision_search_phase3(self, query: str, analysis: QueryAnalysis, limit: int) -> List[SearchResult]:
        """High-precision search for specific symbol queries"""
        try:
            # Use the enhanced hybrid search with intelligent weighting
            results, _ = await self.hybrid_search.search_async(query, limit)
            
            # Filter and boost exact symbol matches
            boosted_results = []
            for result in results:
                # Check if result contains exact symbol matches
                if self._contains_exact_symbols(result, query):
                    result.relevance_score *= 1.5  # Boost exact matches
                
                boosted_results.append(result)
            
            return sorted(boosted_results, key=lambda x: x.relevance_score, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Precision search failed: {e}")
            return []
    
    async def _structural_search_phase3(self, query: str, limit: int) -> List[SearchResult]:
        """KG-powered search for relationship queries"""
        try:
            # Extract relationship intent from query
            relationship_keywords = {
                'calls': ['calls', 'invokes', 'uses'],
                'inherits': ['inherits', 'extends', 'derives'],
                'imports': ['imports', 'requires', 'includes']
            }
            
            detected_relationship = None
            for rel_type, keywords in relationship_keywords.items():
                if any(kw in query.lower() for kw in keywords):
                    detected_relationship = rel_type
                    break
            
            if detected_relationship:
                # Use KG for factual relationship queries - placeholder for now
                logger.info(f"Using KG search for {detected_relationship} relationship")
                # TODO: Implement KG relationship search when KG integration is ready
                
            # Fallback to semantic search
            return await self._semantic_search_phase3(query, None, limit)
        except Exception as e:
            logger.error(f"Structural search failed: {e}")
            return []
    
    async def _semantic_search_phase3(self, query: str, analysis: QueryAnalysis = None, limit: int = 10) -> List[SearchResult]:
        """Vector-heavy semantic search for conceptual queries"""
        try:
            if analysis and analysis.vector_weight > 0.7:
                # Pure vector search for highly semantic queries
                try:
                    query_embedding = self.vector_store.embed_query(query)
                    # Use vector store directly for pure semantic search
                    vector_results = self.vector_store.similarity_search(query, k=limit)
                    return [self._convert_to_search_result(result) for result in vector_results]
                except Exception as e:
                    logger.warning(f"Pure vector search failed: {e}, falling back to hybrid")
            
            # Balanced hybrid search
            results, _ = await self.hybrid_search.search_async(query, limit)
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _convert_to_search_result(self, vector_result) -> SearchResult:
        """Convert vector store result to SearchResult format"""
        # Placeholder conversion - adapt based on actual vector store format
        return SearchResult(
            chunk_id=getattr(vector_result, 'chunk_id', 0),
            file_path=getattr(vector_result, 'file_path', ''),
            snippet=getattr(vector_result, 'page_content', str(vector_result)),
            relevance_score=getattr(vector_result, 'similarity_score', 0.5)
        )
    
    def _contains_exact_symbols(self, result: SearchResult, query: str) -> bool:
        """Check if result contains exact symbol matches"""
        # Simple exact match check - can be enhanced
        query_terms = query.lower().split()
        snippet_lower = result.snippet.lower()
        return any(term in snippet_lower for term in query_terms if len(term) > 2)
    
    def _extract_hierarchical_context(self, result: SearchResult) -> Dict[str, Any]:
        """Extract hierarchical context for enhanced metadata"""
        return {
            'file_type': Path(result.file_path).suffix if hasattr(result, 'file_path') else '',
            'project': self._extract_project_from_path(result.file_path) if hasattr(result, 'file_path') else ''
        }


# Singleton instance for easy import
_smart_search_engine = None

def get_smart_search_engine() -> SmartSearchEngine:
    """Get singleton smart search engine instance"""
    global _smart_search_engine
    if _smart_search_engine is None:
        _smart_search_engine = SmartSearchEngine()
    return _smart_search_engine


async def smart_search(query: str, k: int = 10, strategy_override: List[str] = None, mentioned_projects: List[str] = None) -> Dict[str, Any]:
    """
    Main smart search function - simplified interface for the agent
    
    Args:
        query: Search query string
        k: Maximum number of results to return
        strategy_override: Optional list of strategies to force use
        mentioned_projects: Optional list of projects to filter results by
        
    Returns:
        Dict with search results and metadata
    """
    engine = get_smart_search_engine()
    
    # Use Phase 3 enhanced search if available
    if hasattr(engine, 'phase3_available') and engine.phase3_available:
        return await engine.search_phase3(query, k, strategy_override, mentioned_projects)
    else:
        return await engine.search(query, k, strategy_override, mentioned_projects)