"""
Enhanced Context Delivery System for LLM Optimization

This module provides intelligent context preparation, token counting, and optimization
for delivering relevant code context to language models.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
# import tiktoken  # Optional dependency

from backend.hybrid_search import HybridSearchEngine, SearchResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """Enhanced context chunk with metadata"""
    text: str
    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    token_count: int
    chunk_type: str  # 'code', 'config', 'documentation'
    metadata: Dict = field(default_factory=dict)


@dataclass
class ContextPackage:
    """Complete context package for LLM consumption"""
    formatted_context: str
    total_tokens: int
    chunks_included: int
    files_covered: int
    relevance_threshold: float
    context_summary: str
    attribution: List[str]  # File paths with line numbers


class TokenCounter:
    """Accurate token counting for different LLM models"""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize token counter for specific model
        
        Args:
            model_name: Name of the LLM model for accurate token counting
        """
        self.model_name = model_name
        self.encoding = None
        
        # Try to initialize tiktoken if available
        try:
            import tiktoken
            if "gpt-4" in model_name.lower():
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_name.lower():
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to GPT-4 encoding
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            logger.info(f"Initialized tiktoken for model: {model_name}")
        except ImportError:
            logger.info(f"tiktoken not available, using estimation for model: {model_name}")
        except Exception as e:
            logger.warning(f"Error initializing tiktoken: {e}, using estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model-specific tokenizer or estimation"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.error(f"Error counting tokens with tiktoken: {e}")
        
        # Fallback to estimation
        return self.estimate_tokens(text)
    
    def estimate_tokens(self, text: str) -> int:
        """Fast token estimation without full tokenization"""
        # Rough estimation: ~4 characters per token for code
        return max(1, len(text) // 4)


class ContextOptimizer:
    """Optimize context selection within token limits"""
    
    def __init__(self, token_counter: TokenCounter):
        self.token_counter = token_counter
        
        # Context formatting templates
        self.templates = {
            'file_header': "=== {file_path} (lines {start_line}-{end_line}) ===\n",
            'code_block': "```{language}\n{content}\n```\n",
            'context_summary': "RELEVANT CONTEXT ({chunks} chunks from {files} files):\n\n",
            'attribution': "Sources: {sources}\n\n"
        }
    
    def optimize_context(self, search_results: List[SearchResult], 
                        max_tokens: int = 4000,
                        min_relevance: float = 0.3) -> ContextPackage:
        """
        Optimize context selection within token limits
        
        Args:
            search_results: List of search results to consider
            max_tokens: Maximum token budget for context
            min_relevance: Minimum relevance score threshold
            
        Returns:
            ContextPackage with optimized context
        """
        logger.info(f"Optimizing context: {len(search_results)} candidates, "
                   f"max_tokens={max_tokens}, min_relevance={min_relevance}")
        
        # Filter by relevance threshold
        filtered_results = [
            result for result in search_results 
            if result.relevance_score >= min_relevance
        ]
        
        if not filtered_results:
            logger.warning("No results meet relevance threshold")
            return self._create_empty_context_package()
        
        # Convert to context chunks with token counts
        context_chunks = []
        for result in filtered_results:
            chunk = self._create_context_chunk(result)
            if chunk:
                context_chunks.append(chunk)
        
        # Sort by relevance score (descending)
        context_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Select chunks within token budget
        selected_chunks = self._select_chunks_within_budget(context_chunks, max_tokens)
        
        # Merge adjacent chunks from same file
        merged_chunks = self._merge_adjacent_chunks(selected_chunks)
        
        # Format final context
        context_package = self._format_context_package(merged_chunks, min_relevance)
        
        logger.info(f"Context optimization complete: {context_package.chunks_included} chunks, "
                   f"{context_package.total_tokens} tokens, {context_package.files_covered} files")
        
        return context_package
    
    def _create_context_chunk(self, search_result: SearchResult) -> Optional[ContextChunk]:
        """Convert search result to context chunk with token count"""
        try:
            # Determine chunk type based on file extension and content
            chunk_type = self._determine_chunk_type(search_result.file_path, search_result.snippet)
            
            # Count tokens in the snippet
            token_count = self.token_counter.count_tokens(search_result.snippet)
            
            # Extract line numbers if available in metadata
            start_line = 1
            end_line = 1
            
            return ContextChunk(
                text=search_result.snippet,
                file_path=search_result.file_path,
                start_line=start_line,
                end_line=end_line,
                relevance_score=search_result.relevance_score,
                token_count=token_count,
                chunk_type=chunk_type,
                metadata={
                    'search_type': search_result.search_type,
                    'matched_terms': search_result.matched_terms,
                    'vector_score': search_result.vector_score,
                    'bm25_score': search_result.bm25_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating context chunk: {e}")
            return None
    
    def _determine_chunk_type(self, file_path: str, content: str) -> str:
        """Determine the type of content chunk"""
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext in ['py', 'js', 'ts', 'tsx', 'jsx', 'java', 'cpp', 'c', 'go', 'rs']:
            return 'code'
        elif file_ext in ['json', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'env']:
            return 'config'
        elif file_ext in ['md', 'txt', 'rst', 'adoc']:
            return 'documentation'
        else:
            # Analyze content to determine type
            if any(keyword in content.lower() for keyword in ['def ', 'function ', 'class ', 'import ', 'const ']):
                return 'code'
            elif any(keyword in content.lower() for keyword in ['config', 'setting', '=']):
                return 'config'
            else:
                return 'documentation'
    
    def _select_chunks_within_budget(self, chunks: List[ContextChunk], max_tokens: int) -> List[ContextChunk]:
        """Select chunks that fit within token budget using greedy algorithm"""
        selected = []
        total_tokens = 0
        
        # Reserve tokens for formatting overhead
        formatting_overhead = 200  # Estimated tokens for headers, formatting, etc.
        available_tokens = max_tokens - formatting_overhead
        
        for chunk in chunks:
            # Estimate tokens needed including formatting
            chunk_with_formatting = self._estimate_formatted_chunk_tokens(chunk)
            
            if total_tokens + chunk_with_formatting <= available_tokens:
                selected.append(chunk)
                total_tokens += chunk_with_formatting
            else:
                # Check if we can fit a truncated version
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 100:  # Minimum useful chunk size
                    truncated_chunk = self._truncate_chunk(chunk, remaining_tokens)
                    if truncated_chunk:
                        selected.append(truncated_chunk)
                        break
                else:
                    break
        
        logger.debug(f"Selected {len(selected)} chunks using {total_tokens} tokens")
        return selected
    
    def _estimate_formatted_chunk_tokens(self, chunk: ContextChunk) -> int:
        """Estimate tokens needed for chunk including formatting"""
        # Base chunk tokens
        base_tokens = chunk.token_count
        
        # Add formatting overhead (header, code blocks, etc.)
        header_tokens = self.token_counter.estimate_tokens(
            self.templates['file_header'].format(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line
            )
        )
        
        # Code block formatting if applicable
        code_block_overhead = 10 if chunk.chunk_type == 'code' else 0
        
        return base_tokens + header_tokens + code_block_overhead
    
    def _truncate_chunk(self, chunk: ContextChunk, max_tokens: int) -> Optional[ContextChunk]:
        """Truncate chunk to fit within token limit"""
        if max_tokens < 50:  # Too small to be useful
            return None
        
        # Estimate characters per token for this chunk
        chars_per_token = len(chunk.text) / max(1, chunk.token_count)
        
        # Calculate target character count
        target_chars = int(max_tokens * chars_per_token * 0.8)  # 80% safety margin
        
        if target_chars >= len(chunk.text):
            return chunk
        
        # Truncate at word boundary
        truncated_text = chunk.text[:target_chars]
        last_space = truncated_text.rfind(' ')
        if last_space > target_chars * 0.7:  # Don't truncate too much
            truncated_text = truncated_text[:last_space] + "..."
        else:
            truncated_text = truncated_text + "..."
        
        # Create truncated chunk
        truncated_chunk = ContextChunk(
            text=truncated_text,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            relevance_score=chunk.relevance_score * 0.8,  # Reduce score for truncation
            token_count=self.token_counter.count_tokens(truncated_text),
            chunk_type=chunk.chunk_type,
            metadata=chunk.metadata
        )
        
        return truncated_chunk
    
    def _merge_adjacent_chunks(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Merge adjacent chunks from the same file when beneficial"""
        if len(chunks) <= 1:
            return chunks
        
        # Group chunks by file
        file_groups = defaultdict(list)
        for chunk in chunks:
            file_groups[chunk.file_path].append(chunk)
        
        merged_chunks = []
        
        for file_path, file_chunks in file_groups.items():
            if len(file_chunks) == 1:
                merged_chunks.extend(file_chunks)
                continue
            
            # Sort chunks by line number
            file_chunks.sort(key=lambda x: x.start_line)
            
            # Merge adjacent or overlapping chunks
            current_chunk = file_chunks[0]
            
            for next_chunk in file_chunks[1:]:
                # Check if chunks are adjacent or overlapping
                if (next_chunk.start_line <= current_chunk.end_line + 5 and  # Allow small gaps
                    current_chunk.token_count + next_chunk.token_count < 1000):  # Don't create huge chunks
                    
                    # Merge chunks
                    merged_text = current_chunk.text + "\n\n" + next_chunk.text
                    merged_chunk = ContextChunk(
                        text=merged_text,
                        file_path=current_chunk.file_path,
                        start_line=current_chunk.start_line,
                        end_line=next_chunk.end_line,
                        relevance_score=max(current_chunk.relevance_score, next_chunk.relevance_score),
                        token_count=self.token_counter.count_tokens(merged_text),
                        chunk_type=current_chunk.chunk_type,
                        metadata={**current_chunk.metadata, **next_chunk.metadata}
                    )
                    current_chunk = merged_chunk
                else:
                    # Can't merge, add current chunk and start new one
                    merged_chunks.append(current_chunk)
                    current_chunk = next_chunk
            
            # Add the last chunk
            merged_chunks.append(current_chunk)
        
        logger.debug(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks
    
    def _format_context_package(self, chunks: List[ContextChunk], min_relevance: float) -> ContextPackage:
        """Format chunks into final context package"""
        if not chunks:
            return self._create_empty_context_package()
        
        # Build formatted context
        context_parts = []
        
        # Add context summary
        files_covered = len(set(chunk.file_path for chunk in chunks))
        summary = self.templates['context_summary'].format(
            chunks=len(chunks),
            files=files_covered
        )
        context_parts.append(summary)
        
        # Add each chunk with proper formatting
        attribution = []
        for chunk in chunks:
            # File header
            header = self.templates['file_header'].format(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line
            )
            context_parts.append(header)
            
            # Content with appropriate formatting
            if chunk.chunk_type == 'code':
                language = self._detect_language(chunk.file_path)
                formatted_content = self.templates['code_block'].format(
                    language=language,
                    content=chunk.text
                )
            else:
                formatted_content = chunk.text + "\n"
            
            context_parts.append(formatted_content)
            
            # Add to attribution
            attribution.append(f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
        
        # Combine all parts
        formatted_context = "".join(context_parts)
        
        # Count total tokens
        total_tokens = self.token_counter.count_tokens(formatted_context)
        
        # Create context summary
        context_summary = f"Retrieved {len(chunks)} relevant code chunks from {files_covered} files"
        
        return ContextPackage(
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            chunks_included=len(chunks),
            files_covered=files_covered,
            relevance_threshold=min_relevance,
            context_summary=context_summary,
            attribution=attribution
        )
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = file_path.split('.')[-1].lower()
        
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'tsx': 'typescript',
            'jsx': 'javascript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust',
            'php': 'php',
            'rb': 'ruby',
            'sh': 'bash',
            'sql': 'sql',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'toml': 'toml',
            'xml': 'xml'
        }
        
        return language_map.get(ext, 'text')
    
    def _create_empty_context_package(self) -> ContextPackage:
        """Create empty context package when no relevant context is found"""
        return ContextPackage(
            formatted_context="No relevant context found in the codebase.",
            total_tokens=10,
            chunks_included=0,
            files_covered=0,
            relevance_threshold=0.0,
            context_summary="No relevant context available",
            attribution=[]
        )


class ContextDeliverySystem:
    """Main context delivery system coordinating search and optimization"""
    
    def __init__(self, hybrid_search: HybridSearchEngine, model_name: str = "gpt-4"):
        """
        Initialize context delivery system
        
        Args:
            hybrid_search: Hybrid search engine instance
            model_name: LLM model name for token counting
        """
        self.hybrid_search = hybrid_search
        self.token_counter = TokenCounter(model_name)
        self.context_optimizer = ContextOptimizer(self.token_counter)
        
        # Default parameters
        self.default_max_tokens = 4000
        self.default_min_relevance = 0.3
        self.default_max_results = 10
        
    def prepare_context(self, query: str, max_tokens: int = None, 
                       min_relevance: float = None, max_results: int = None) -> ContextPackage:
        """
        Prepare optimized context for LLM consumption
        
        Args:
            query: Search query for context retrieval
            max_tokens: Maximum token budget (uses default if None)
            min_relevance: Minimum relevance threshold (uses default if None)
            max_results: Maximum search results to consider (uses default if None)
            
        Returns:
            ContextPackage with optimized context
        """
        # Use defaults if not specified
        max_tokens = max_tokens or self.default_max_tokens
        min_relevance = min_relevance or self.default_min_relevance
        max_results = max_results or self.default_max_results
        
        logger.info(f"Preparing context for query: '{query}' "
                   f"(max_tokens={max_tokens}, min_relevance={min_relevance})")
        
        try:
            # Perform hybrid search
            search_results = self.hybrid_search.search(
                query, k=max_results, min_relevance=min_relevance * 0.8  # Lower threshold for search
            )
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return self.context_optimizer._create_empty_context_package()
            
            # Optimize context within token budget
            context_package = self.context_optimizer.optimize_context(
                search_results, max_tokens, min_relevance
            )
            
            return context_package
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return self.context_optimizer._create_empty_context_package()
    
    def get_context_statistics(self) -> Dict:
        """Get context delivery system statistics"""
        encoding_name = "estimation"
        if self.token_counter.encoding:
            encoding_name = self.token_counter.encoding.name
        
        return {
            'token_counter': {
                'model_name': self.token_counter.model_name,
                'encoding_name': encoding_name
            },
            'default_parameters': {
                'max_tokens': self.default_max_tokens,
                'min_relevance': self.default_min_relevance,
                'max_results': self.default_max_results
            },
            'hybrid_search_stats': self.hybrid_search.get_search_statistics()
        }
    
    def update_parameters(self, max_tokens: int = None, min_relevance: float = None, 
                         max_results: int = None):
        """Update default parameters"""
        if max_tokens is not None:
            self.default_max_tokens = max_tokens
        if min_relevance is not None:
            self.default_min_relevance = min_relevance
        if max_results is not None:
            self.default_max_results = max_results
        
        logger.info(f"Updated context delivery parameters: "
                   f"max_tokens={self.default_max_tokens}, "
                   f"min_relevance={self.default_min_relevance}, "
                   f"max_results={self.default_max_results}")