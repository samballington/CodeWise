#!/usr/bin/env python3
"""
Discovery Pipeline - README-driven architectural discovery system

This module implements intelligent discovery that enhances search results by
extracting and following file references found in documentation files.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path

from response_formatter import QueryType
from path_resolver import PathResolver

logger = logging.getLogger(__name__)

@dataclass
class FileReference:
    """Represents a discovered file reference from documentation"""
    file_path: str
    reference_type: str  # 'api_endpoint', 'component', 'config', 'directory', etc.
    confidence: float    # 0.0 to 1.0
    context: str        # Surrounding text that mentioned this file
    priority: int       # Lower number = higher priority for examination
    source_file: str    # File where this reference was found

@dataclass
class DiscoveryMetadata:
    """Metadata about the discovery process"""
    total_references_found: int
    high_confidence_references: int
    auto_examinations_triggered: int
    discovery_time_ms: float
    source_files_analyzed: List[str]

@dataclass
class EnhancedSearchResults:
    """Extended search results with discovered files"""
    original_results: List[Dict]  # Original search results
    discovered_files: List[FileReference]
    discovery_metadata: DiscoveryMetadata
    recommended_examinations: List[str]  # File paths to auto-examine

class DiscoveryExtractor(ABC):
    """Base class for file reference extractors"""
    
    @abstractmethod
    def can_extract(self, file_path: str, content: str) -> bool:
        """Determine if this extractor can process the file"""
        pass
    
    @abstractmethod
    def extract_references(self, content: str, source_file: str) -> List[FileReference]:
        """Extract file references from content"""
        pass

class ReadmeFileExtractor(DiscoveryExtractor):
    """Extract architectural references from README files"""
    
    # Regex patterns for different types of file references
    PATTERNS = {
        'api_endpoints': {
            'regex': r'/api/[a-zA-Z0-9/_-]+',
            'confidence': 0.9,
            'priority': 1
        },
        'file_paths': {
            'regex': r'`([^`]*\.(ts|tsx|js|jsx|py|md|json|config\.(ts|js|mjs)))`',
            'confidence': 0.95,
            'priority': 1
        },
        'components': {
            'regex': r'\b([A-Z][a-zA-Z0-9]*(?:Component|Hook|Provider|Context|Dialog|Modal))\b',
            'confidence': 0.7,
            'priority': 2
        },
        'directories': {
            'regex': r'\b(src/[a-zA-Z0-9/_-]+)',
            'confidence': 0.8,
            'priority': 2
        },
        'config_files': {
            'regex': r'\b([a-zA-Z0-9._-]+\.config\.(ts|js|mjs|json))\b',
            'confidence': 0.9,
            'priority': 1
        },
        'package_files': {
            'regex': r'\b(package\.json|tsconfig\.json|next\.config\.(js|mjs)|tailwind\.config\.(ts|js))\b',
            'confidence': 0.95,
            'priority': 1
        },
        'trpc_routers': {
            'regex': r'\b(router|tRPC|trpc).*?([a-zA-Z][a-zA-Z0-9]*(?:Router|Api))\b',
            'confidence': 0.8,
            'priority': 1
        }
    }
    
    def can_extract(self, file_path: str, content: str) -> bool:
        """Check if this is a README file"""
        return 'readme' in file_path.lower() or 'README' in file_path
    
    def extract_references(self, content: str, source_file: str) -> List[FileReference]:
        """Extract file references using regex patterns"""
        references = []
        
        for ref_type, pattern_info in self.PATTERNS.items():
            pattern = pattern_info['regex']
            confidence = pattern_info['confidence']
            priority = pattern_info['priority']
            
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Extract the file path (use group 1 if it exists, otherwise group 0)
                file_path = match.group(1) if match.groups() else match.group(0)
                
                # Get surrounding context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end].strip()
                
                # Clean up the file path
                file_path = self._clean_file_path(file_path, ref_type)
                
                if file_path:
                    references.append(FileReference(
                        file_path=file_path,
                        reference_type=ref_type,
                        confidence=confidence,
                        context=context,
                        priority=priority,
                        source_file=source_file
                    ))
        
        # Remove duplicates while preserving highest confidence
        return self._deduplicate_references(references)
    
    def _clean_file_path(self, file_path: str, ref_type: str) -> str:
        """Clean and normalize file paths"""
        if not file_path:
            return ""
        
        # Remove backticks and quotes
        file_path = file_path.strip('`"\'')
        
        # Handle API endpoints - convert to actual file paths
        if ref_type == 'api_endpoints':
            # Convert /api/fal to src/app/api/fal/route.ts (Next.js pattern)
            if file_path.startswith('/api/'):
                api_path = file_path[5:]  # Remove '/api/'
                return f"src/app/api/{api_path}/route.ts"
        
        # Handle components - try to find actual component files
        if ref_type == 'components':
            # Look for common component patterns
            return f"src/components/**/{file_path}.tsx"
        
        # Handle directories - add common file patterns
        if ref_type == 'directories':
            if not file_path.endswith('/'):
                file_path += '/'
        
        return file_path
    
    def _deduplicate_references(self, references: List[FileReference]) -> List[FileReference]:
        """Remove duplicate references, keeping highest confidence"""
        seen = {}
        
        for ref in references:
            key = (ref.file_path, ref.reference_type)
            if key not in seen or ref.confidence > seen[key].confidence:
                seen[key] = ref
        
        return list(seen.values())

class DiscoveryPipeline:
    """
    Main discovery pipeline that enhances search results by finding and
    following file references in documentation.
    """
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self.extractors = {
            'readme': ReadmeFileExtractor(),
            # Future extractors can be added here
        }
        
        # Configuration
        self.MAX_AUTO_EXAMINATIONS = 3  # Reduced to avoid overwhelming
        self.MIN_CONFIDENCE_THRESHOLD = 0.6  # Lowered to be more permissive
        self.DISCOVERY_TIMEOUT = 10.0  # seconds
    
    async def enhance_search_results(
        self, 
        search_results: List[Dict], 
        query_type: QueryType,
        projects: List[str]
    ) -> EnhancedSearchResults:
        """
        Main entry point - analyzes search results and discovers additional files
        """
        import time
        start_time = time.time()
        
        logger.info(f"🔍 Starting discovery pipeline for query type: {query_type}")
        
        try:
            # Extract file references from search results
            all_references = []
            source_files_analyzed = []
            
            for result in search_results:
                file_path = result.get('file_path', '')
                content = result.get('content', '')
                
                if self._should_analyze_file(file_path, content):
                    references = self.extract_file_references(content, file_path)
                    all_references.extend(references)
                    source_files_analyzed.append(file_path)
                    
                    logger.info(f"📄 Analyzed {file_path}: found {len(references)} references")
            
            # Prioritize discoveries based on query type
            prioritized_references = self.prioritize_discoveries(all_references, query_type)
            
            # Select files for auto-examination
            recommended_examinations = self._select_auto_examinations(
                prioritized_references, projects
            )
            
            # Create metadata
            discovery_time = (time.time() - start_time) * 1000
            high_confidence_refs = len([r for r in all_references if r.confidence >= 0.8])
            
            metadata = DiscoveryMetadata(
                total_references_found=len(all_references),
                high_confidence_references=high_confidence_refs,
                auto_examinations_triggered=len(recommended_examinations),
                discovery_time_ms=discovery_time,
                source_files_analyzed=source_files_analyzed
            )
            
            logger.info(f"✅ Discovery complete: {len(all_references)} refs, {len(recommended_examinations)} auto-exams")
            
            return EnhancedSearchResults(
                original_results=search_results,
                discovered_files=prioritized_references,
                discovery_metadata=metadata,
                recommended_examinations=recommended_examinations
            )
            
        except Exception as e:
            logger.error(f"❌ Discovery pipeline error: {e}")
            # Graceful fallback - return original results
            return EnhancedSearchResults(
                original_results=search_results,
                discovered_files=[],
                discovery_metadata=DiscoveryMetadata(0, 0, 0, 0.0, []),
                recommended_examinations=[]
            )
    
    def extract_file_references(self, content: str, source_file: str) -> List[FileReference]:
        """Extract file references from content using appropriate extractors"""
        references = []
        
        for extractor_name, extractor in self.extractors.items():
            if extractor.can_extract(source_file, content):
                try:
                    refs = extractor.extract_references(content, source_file)
                    references.extend(refs)
                    logger.debug(f"🔧 {extractor_name} extractor found {len(refs)} references")
                except Exception as e:
                    logger.warning(f"⚠️ {extractor_name} extractor failed: {e}")
        
        return references
    
    def prioritize_discoveries(
        self, 
        references: List[FileReference], 
        query_type: QueryType
    ) -> List[FileReference]:
        """Prioritize discovered files based on query type and relevance"""
        
        # Query-type specific priority adjustments
        priority_adjustments = self._get_priority_adjustments(query_type)
        
        # Apply adjustments
        for ref in references:
            for ref_type, adjustment in priority_adjustments.items():
                if ref.reference_type == ref_type:
                    ref.priority += adjustment
        
        # Sort by priority (lower number = higher priority), then by confidence
        return sorted(references, key=lambda r: (r.priority, -r.confidence))
    
    def _should_analyze_file(self, file_path: str, content: str) -> bool:
        """Determine if a file should be analyzed for references"""
        if not content or len(content) < 100:
            return False
        
        # Check if any extractor can handle this file
        return any(
            extractor.can_extract(file_path, content) 
            for extractor in self.extractors.values()
        )
    
    def _get_priority_adjustments(self, query_type: QueryType) -> Dict[str, int]:
        """Get priority adjustments based on query type"""
        adjustments = {
            QueryType.ARCHITECTURE: {
                'api_endpoints': -2,  # Higher priority (lower number)
                'config_files': -2,
                'directories': -1,
                'components': -1,
                'file_paths': -1
            },
            QueryType.DATABASE_ENTITIES: {
                'file_paths': -2,
                'directories': -1,
                'api_endpoints': 0
            },
            QueryType.AUTHENTICATION: {
                'api_endpoints': -2,
                'file_paths': -1,
                'components': -1
            },
            QueryType.GENERAL: {
                'file_paths': -1,
                'api_endpoints': 0
            }
        }
        
        return adjustments.get(query_type, {})
    
    def _select_auto_examinations(
        self, 
        references: List[FileReference], 
        projects: List[str]
    ) -> List[str]:
        """Select files for automatic examination"""
        selected = []
        
        logger.info(f"🔍 AUTO-EXAM SELECTION: {len(references)} references to evaluate")
        
        for i, ref in enumerate(references):
            logger.info(f"🔍 AUTO-EXAM REF {i+1}: {ref.file_path} (confidence: {ref.confidence}, type: {ref.reference_type})")
            
            if len(selected) >= self.MAX_AUTO_EXAMINATIONS:
                logger.info(f"🔍 AUTO-EXAM: Reached max limit ({self.MAX_AUTO_EXAMINATIONS})")
                break
            
            if ref.confidence < self.MIN_CONFIDENCE_THRESHOLD:
                logger.info(f"🔍 AUTO-EXAM: Skipped {ref.file_path} - low confidence ({ref.confidence} < {self.MIN_CONFIDENCE_THRESHOLD})")
                continue
            
            # Resolve the file path
            resolved_path = self._resolve_file_path(ref.file_path, projects)
            logger.info(f"🔍 AUTO-EXAM: Resolved {ref.file_path} -> {resolved_path}")
            
            if resolved_path and resolved_path not in selected:
                selected.append(resolved_path)
                logger.info(f"✅ AUTO-EXAM: Added {resolved_path}")
            elif not resolved_path:
                logger.info(f"❌ AUTO-EXAM: Failed to resolve {ref.file_path}")
            else:
                logger.info(f"⚠️ AUTO-EXAM: Duplicate {resolved_path}")
        
        logger.info(f"🎯 AUTO-EXAM FINAL: Selected {len(selected)} files: {selected}")
        return selected
    
    def _resolve_file_path(self, file_path: str, projects: List[str]) -> Optional[str]:
        """Resolve a discovered file path to an actual file"""
        try:
            # Handle wildcard patterns (like src/components/**/*.tsx)
            if '**' in file_path or '*' in file_path:
                logger.debug(f"Skipping wildcard pattern: {file_path}")
                return None
            
            # Try different resolution strategies
            candidates = []
            
            # Strategy 1: Try with projects if provided
            if projects:
                for project in projects:
                    candidates.append(f"{project}/{file_path}")
            
            # Strategy 2: Try direct path resolution
            candidates.append(file_path)
            
            # Strategy 3: Try with infinite-kanvas prefix (since that's our main project)
            if not file_path.startswith('infinite-kanvas/'):
                candidates.append(f"infinite-kanvas/{file_path}")
            
            # Test each candidate
            for candidate in candidates:
                try:
                    logger.info(f"🔍 AUTO-EXAM: Trying to resolve candidate: {candidate}")
                    resolved_path, exists = self.path_resolver.resolve_file_path(candidate)
                    logger.info(f"🔍 AUTO-EXAM: Candidate {candidate} -> {resolved_path} (exists: {exists})")
                    if exists:
                        logger.info(f"✅ AUTO-EXAM: Successfully resolved {file_path} -> {resolved_path}")
                        return resolved_path
                except Exception as e:
                    logger.warning(f"❌ AUTO-EXAM: Failed to resolve candidate {candidate}: {e}")
                    continue
            
            logger.debug(f"Failed to resolve any candidate for {file_path}: {candidates}")
            return None
            
        except Exception as e:
            logger.debug(f"Failed to resolve path {file_path}: {e}")
            return None