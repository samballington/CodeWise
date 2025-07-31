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

from backend.hybrid_search import HybridSearchEngine, SearchResult
from backend.bm25_index import BM25Index, BM25Result
from backend.vector_store import get_vector_store
from backend.discovery_pipeline import DiscoveryPipeline
from backend.path_resolver import PathResolver

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification"""
    ENTITY = "entity"           # Database entities, models, schemas
    FILE = "file"              # Specific files, file types, structure
    GENERAL = "general"        # General code search, functionality
    ARCHITECTURE = "architecture"  # System overview, high-level structure


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
        
        # Technical patterns that suggest specific intent
        self.technical_patterns = {
            QueryIntent.ENTITY: [
                r'@Entity\b', r'CREATE\s+TABLE', r'class\s+\w+\s*\(.*Model\)',
                r'models?\.\w+', r'database\s+schema', r'entity\s+relationship'
            ],
            QueryIntent.FILE: [
                r'\.\w{2,4}\b', r'/(src|app|lib|config)/', r'package\.json',
                r'requirements\.txt', r'Dockerfile', r'\.env'
            ],
            QueryIntent.ARCHITECTURE: [
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
        
        # Calculate intent scores
        intent_scores = {
            QueryIntent.ENTITY: self._calculate_entity_score(query_lower, query_terms),
            QueryIntent.FILE: self._calculate_file_score(query_lower, query_terms),
            QueryIntent.ARCHITECTURE: self._calculate_architecture_score(query_lower, query_terms),
            QueryIntent.GENERAL: 0.3  # Default baseline
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
        strategies = []
        
        if intent == QueryIntent.ENTITY:
            strategies = ['entity_discovery', 'vector_search', 'bm25_search']
        elif intent == QueryIntent.FILE:
            strategies = ['file_search', 'vector_search', 'bm25_search']
        elif intent == QueryIntent.ARCHITECTURE:
            strategies = ['vector_search', 'structure_analysis', 'bm25_search']
        else:  # GENERAL
            strategies = ['vector_search', 'bm25_search']
        
        # Add hybrid search for high-confidence queries
        if confidence > 0.7:
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
    Unified smart search engine that replaces multiple tools with intelligent routing
    
    This is the main implementation of the smart_search tool for the simplified 3-tool architecture.
    """
    
    def __init__(self):
        self.hybrid_search = HybridSearchEngine()
        self.intent_analyzer = QueryIntentAnalyzer()
        self.entity_discovery = EntityDiscovery()
        
        # Discovery pipeline for Task 5 extension
        self.path_resolver = PathResolver()
        self.discovery_pipeline = DiscoveryPipeline(self.path_resolver)
        
        # Search configuration
        self.max_results = 15
        self.min_confidence = 0.3
    
    async def search(self, query: str, k: int = 10, strategy_override: List[str] = None) -> Dict[str, Any]:
        """
        Perform intelligent search combining multiple strategies
        
        Args:
            query: Search query string
            k: Maximum number of results to return
            strategy_override: Optional list of strategies to force use
            
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
        
        logger.info(f"🧠 SMART SEARCH: '{query}' (k={k})")
        
        # Analyze query intent
        query_analysis = self.intent_analyzer.analyze_query(query)
        logger.info(f"Query intent: {query_analysis['intent'].value} (confidence: {query_analysis['confidence']:.2f})")
        
        # Determine search strategies
        strategies = strategy_override or query_analysis['search_strategies']
        logger.info(f"Search strategies: {strategies}")
        
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
        
        # Apply filters and limits
        filtered_results = [
            result for result in final_results 
            if result.confidence >= self.min_confidence
        ][:k]
        
        execution_time = time.time() - start_time
        
        logger.info(f"🧠 SMART SEARCH COMPLETE: {len(filtered_results)} results in {execution_time:.2f}s")
        
        # NEW: Discovery Pipeline Enhancement for Task 5 Extension
        auto_examine_files = await self._enhance_with_discovery_pipeline(query, filtered_results, query_analysis)
        
        return {
            'results': filtered_results,
            'query_analysis': query_analysis,
            'search_strategies_used': strategies_used,
            'total_results': len(all_results),
            'execution_time': execution_time,
            'auto_examine_files': auto_examine_files or []  # Files recommended for auto-examination
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
            search_results = self.hybrid_search.search(query, k=self.max_results)
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
            for i, (file_path, snippet) in enumerate(vector_results):
                score = max(0.1, 1.0 - (i * 0.1))  # Decreasing scores
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
        """Execute BM25 search strategy"""
        # This would use the BM25 index directly if needed
        # For now, rely on hybrid search which includes BM25
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
            
            # Convert search results to the format expected by discovery pipeline
            formatted_results = []
            for result in search_results:
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
        """Get full content of a file for discovery analysis"""
        try:
            # Construct full workspace path
            if not file_path.startswith('/workspace/'):
                full_path = f"/workspace/{file_path}"
            else:
                full_path = file_path
            
            # Read the file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
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


# Singleton instance for easy import
_smart_search_engine = None

def get_smart_search_engine() -> SmartSearchEngine:
    """Get singleton smart search engine instance"""
    global _smart_search_engine
    if _smart_search_engine is None:
        _smart_search_engine = SmartSearchEngine()
    return _smart_search_engine


async def smart_search(query: str, k: int = 10, strategy_override: List[str] = None) -> Dict[str, Any]:
    """
    Main smart search function - simplified interface for the agent
    
    Args:
        query: Search query string
        k: Maximum number of results to return
        strategy_override: Optional list of strategies to force use
        
    Returns:
        Dict with search results and metadata
    """
    engine = get_smart_search_engine()
    return await engine.search(query, k, strategy_override)