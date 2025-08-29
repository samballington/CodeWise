"""
Enhanced File Examination with Hierarchical Context

Integrates Phase 1 hierarchical chunking to provide structured analysis
of files with semantic relationships and context awareness.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Phase 1 imports
from ..indexer.chunkers.hierarchical_chunker import HierarchicalChunker
from ..indexer.storage.enhanced_metadata_store import EnhancedMetadataStore
from ..indexer.storage.context_reconstructor import ContextReconstructor

logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Structured file analysis result."""
    file_path: str
    basic_info: Dict[str, Any]
    hierarchical_structure: Dict[str, Any]
    key_symbols: List[Dict[str, Any]]
    context_relationships: List[Dict[str, Any]]
    semantic_summary: str
    success: bool
    error_message: Optional[str] = None


class EnhancedFileExaminer:
    """
    Enhanced file examination leveraging Phase 1 hierarchical chunking.
    
    Provides structured context from hierarchical chunk relationships
    in addition to traditional file content analysis.
    """
    
    def __init__(self, storage_dir: str = ".vector_cache"):
        """
        Initialize enhanced file examiner.
        
        Args:
            storage_dir: Directory for metadata storage
        """
        self.chunker = HierarchicalChunker()
        self.metadata_store = EnhancedMetadataStore(storage_dir)
        self.context_reconstructor = ContextReconstructor(self.metadata_store)
        
        logger.info("Enhanced file examiner initialized")
    
    async def examine_files(self, file_paths: List[str], detail_level: str = "summary",
                           use_cached_analysis: bool = True) -> str:
        """
        Enhanced file examination with hierarchical context analysis.
        
        Args:
            file_paths: List of file paths to examine
            detail_level: Level of detail (summary, structure, full, hierarchical)
            use_cached_analysis: Whether to use cached hierarchical analysis
            
        Returns:
            Formatted examination results
        """
        if not file_paths:
            return "❌ No files provided for examination"
        
        logger.info(f"Enhanced file examination: {len(file_paths)} files, detail={detail_level}")
        
        results = []
        results.append(f"📄 ENHANCED FILE EXAMINATION ({detail_level} level)")
        results.append("🔗 Includes hierarchical context analysis from Phase 1 chunking")
        results.append("=" * 70)
        
        analyses = []
        for file_path in file_paths[:10]:  # Limit to 10 files
            analysis = await self._analyze_single_file(
                file_path, detail_level, use_cached_analysis
            )
            analyses.append(analysis)
            
            # Format analysis for display
            formatted = self._format_file_analysis(analysis, detail_level)
            results.extend(formatted)
        
        # Add summary if multiple files
        if len(analyses) > 1:
            summary = self._generate_multi_file_summary(analyses)
            results.extend(["", "📋 MULTI-FILE SUMMARY", "=" * 40] + summary)
        
        return "\n".join(results)
    
    async def _analyze_single_file(self, file_path: str, detail_level: str,
                                  use_cached: bool) -> FileAnalysis:
        """Analyze a single file with hierarchical context."""
        try:
            # Resolve file path
            workspace_path = Path('/workspace')
            if not file_path.startswith('/workspace'):
                full_path = workspace_path / file_path.lstrip('./')
            else:
                full_path = Path(file_path)
            
            if not full_path.exists():
                return FileAnalysis(
                    file_path=file_path,
                    basic_info={},
                    hierarchical_structure={},
                    key_symbols=[],
                    context_relationships=[],
                    semantic_summary="",
                    success=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # Read file content
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic file info
            lines = content.split('\n')
            basic_info = {
                'size_chars': len(content),
                'line_count': len(lines),
                'file_extension': full_path.suffix,
                'relative_path': str(full_path.relative_to(workspace_path))
            }
            
            # Get or create hierarchical analysis
            hierarchical_chunks = None
            if use_cached:
                # Try to get from metadata store
                file_chunks = self.metadata_store.get_chunks_by_file(str(full_path))
                if file_chunks:
                    logger.debug(f"Using cached hierarchical analysis for {file_path}")
                else:
                    # Create new hierarchical analysis
                    hierarchical_chunks = self.chunker.chunk_file(content, full_path)
                    logger.debug(f"Created new hierarchical analysis for {file_path}")
            else:
                # Always create fresh analysis
                hierarchical_chunks = self.chunker.chunk_file(content, full_path)
            
            # Build hierarchical structure analysis
            if hierarchical_chunks:
                hierarchical_structure = self._build_hierarchical_analysis(hierarchical_chunks)
                key_symbols = self._extract_symbols(hierarchical_chunks)
                context_relationships = self._extract_relationships(hierarchical_chunks)
            else:
                # Use cached data
                file_context = self.context_reconstructor.get_file_context(str(full_path))
                hierarchical_structure = file_context
                key_symbols = file_context.get('symbols', [])
                context_relationships = self._build_relationships_from_cached(file_chunks)
            
            # Generate semantic summary
            semantic_summary = self._generate_semantic_summary(
                basic_info, hierarchical_structure, key_symbols
            )
            
            return FileAnalysis(
                file_path=file_path,
                basic_info=basic_info,
                hierarchical_structure=hierarchical_structure,
                key_symbols=key_symbols,
                context_relationships=context_relationships,
                semantic_summary=semantic_summary,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return FileAnalysis(
                file_path=file_path,
                basic_info={},
                hierarchical_structure={},
                key_symbols=[],
                context_relationships=[],
                semantic_summary="",
                success=False,
                error_message=str(e)
            )
    
    def _build_hierarchical_analysis(self, chunks) -> Dict[str, Any]:
        """Build structured file analysis from hierarchical chunks."""
        analysis = {
            'file_summary': None,
            'major_components': [],
            'symbols': [],
            'complexity_analysis': {},
            'chunk_distribution': {}
        }
        
        # Categorize chunks
        summaries = []
        blocks = []
        symbols = []
        
        for chunk in chunks:
            if chunk.type.value == 'summary':
                summaries.append(chunk)
            elif chunk.type.value == 'block':
                blocks.append(chunk)
            elif chunk.type.value == 'symbol':
                symbols.append(chunk)
        
        # File summary
        if summaries:
            summary_chunk = summaries[0]  # Use first summary
            analysis['file_summary'] = {
                'overview': summary_chunk.content[:300],
                'key_exports': summary_chunk.key_exports,
                'key_imports': summary_chunk.key_imports,
                'architecture_notes': summary_chunk.architecture_notes,
                'total_child_chunks': len(summary_chunk.child_chunk_ids)
            }
        
        # Major components (blocks)
        for block in blocks:
            analysis['major_components'].append({
                'id': block.id,
                'type': block.block_type,
                'line_range': [block.line_start, block.line_end],
                'children_count': len(block.child_chunk_ids),
                'complexity': block.complexity_score,
                'imports': block.imports,
                'exports': block.exports
            })
        
        # Symbols
        for symbol in symbols:
            analysis['symbols'].append({
                'id': symbol.id,
                'name': symbol.symbol_name,
                'type': symbol.symbol_type,
                'line_range': [symbol.line_start, symbol.line_end],
                'parameters': symbol.parameters,
                'return_type': symbol.return_type,
                'docstring': symbol.docstring[:100] if symbol.docstring else None,
                'complexity': symbol.complexity_score,
                'decorators': symbol.decorators
            })
        
        # Complexity analysis
        if chunks:
            complexities = [chunk.complexity_score for chunk in chunks if hasattr(chunk, 'complexity_score')]
            analysis['complexity_analysis'] = {
                'avg_complexity': sum(complexities) / len(complexities) if complexities else 0,
                'max_complexity': max(complexities) if complexities else 0,
                'complex_chunks': len([c for c in complexities if c > 0.7])
            }
        
        # Chunk distribution
        analysis['chunk_distribution'] = {
            'summary_chunks': len(summaries),
            'block_chunks': len(blocks),
            'symbol_chunks': len(symbols),
            'total_chunks': len(chunks)
        }
        
        return analysis
    
    def _extract_symbols(self, chunks) -> List[Dict[str, Any]]:
        """Extract symbol information from chunks."""
        symbols = []
        
        for chunk in chunks:
            if chunk.type.value == 'symbol':
                symbol_info = {
                    'name': chunk.symbol_name,
                    'type': chunk.symbol_type,
                    'signature': self._build_signature(chunk),
                    'docstring': chunk.docstring[:150] if chunk.docstring else None,
                    'complexity': chunk.complexity_score,
                    'line_range': [chunk.line_start, chunk.line_end],
                    'parent_context': chunk.parent_chunk_id
                }
                symbols.append(symbol_info)
        
        # Sort by line number
        symbols.sort(key=lambda x: x['line_range'][0])
        return symbols
    
    def _extract_relationships(self, chunks) -> List[Dict[str, Any]]:
        """Extract hierarchical relationships from chunks."""
        relationships = []
        
        for chunk in chunks:
            if hasattr(chunk, 'parent_chunk_id') and chunk.parent_chunk_id:
                relationships.append({
                    'type': 'parent_child',
                    'child': chunk.id,
                    'parent': chunk.parent_chunk_id,
                    'child_type': chunk.type.value
                })
            
            if hasattr(chunk, 'child_chunk_ids'):
                for child_id in chunk.child_chunk_ids:
                    relationships.append({
                        'type': 'contains',
                        'container': chunk.id,
                        'contained': child_id,
                        'container_type': chunk.type.value
                    })
        
        return relationships
    
    def _build_relationships_from_cached(self, file_chunks) -> List[Dict[str, Any]]:
        """Build relationships from cached chunk data."""
        relationships = []
        
        for chunk in file_chunks:
            chunk_data = chunk.get('chunk_data', {})
            
            if chunk_data.get('parent_chunk_id'):
                relationships.append({
                    'type': 'parent_child',
                    'child': chunk['chunk_id'],
                    'parent': chunk_data['parent_chunk_id'],
                    'child_type': chunk.get('chunk_type', 'unknown')
                })
            
            if chunk_data.get('child_chunk_ids'):
                for child_id in chunk_data['child_chunk_ids']:
                    relationships.append({
                        'type': 'contains',
                        'container': chunk['chunk_id'],
                        'contained': child_id,
                        'container_type': chunk.get('chunk_type', 'unknown')
                    })
        
        return relationships
    
    def _build_signature(self, symbol_chunk) -> str:
        """Build function/method signature from symbol chunk."""
        if symbol_chunk.symbol_type in ['function', 'method', 'async_function']:
            params = ', '.join(symbol_chunk.parameters) if symbol_chunk.parameters else ''
            return_type = f" -> {symbol_chunk.return_type}" if symbol_chunk.return_type else ""
            return f"{symbol_chunk.symbol_name}({params}){return_type}"
        elif symbol_chunk.symbol_type == 'class':
            return f"class {symbol_chunk.symbol_name}"
        else:
            return symbol_chunk.symbol_name
    
    def _generate_semantic_summary(self, basic_info: Dict, structure: Dict, 
                                  symbols: List[Dict]) -> str:
        """Generate semantic summary of the file."""
        summary_parts = []
        
        # File type and size
        extension = basic_info.get('file_extension', '')
        size = basic_info.get('size_chars', 0)
        lines = basic_info.get('line_count', 0)
        
        if extension == '.py':
            summary_parts.append("Python module")
        elif extension in ['.js', '.jsx']:
            summary_parts.append("JavaScript module")
        elif extension in ['.ts', '.tsx']:
            summary_parts.append("TypeScript module")
        else:
            summary_parts.append(f"{extension} file")
        
        summary_parts.append(f"({lines} lines, {size:,} chars)")
        
        # Symbol analysis
        if symbols:
            symbol_types = {}
            for symbol in symbols:
                sym_type = symbol['type']
                symbol_types[sym_type] = symbol_types.get(sym_type, 0) + 1
            
            type_descriptions = []
            for sym_type, count in symbol_types.items():
                if count > 1:
                    type_descriptions.append(f"{count} {sym_type}s")
                else:
                    type_descriptions.append(f"1 {sym_type}")
            
            if type_descriptions:
                summary_parts.append(f"containing {', '.join(type_descriptions)}")
        
        # Complexity analysis
        if structure.get('complexity_analysis'):
            complexity = structure['complexity_analysis']
            avg_complexity = complexity.get('avg_complexity', 0)
            if avg_complexity > 0.7:
                summary_parts.append("(high complexity)")
            elif avg_complexity > 0.4:
                summary_parts.append("(moderate complexity)")
            else:
                summary_parts.append("(low complexity)")
        
        return " ".join(summary_parts)
    
    def _format_file_analysis(self, analysis: FileAnalysis, detail_level: str) -> List[str]:
        """Format file analysis for display."""
        results = []
        
        if not analysis.success:
            results.append(f"\n❌ FILE: {analysis.file_path}")
            results.append(f"   Error: {analysis.error_message}")
            return results
        
        results.append(f"\n📁 FILE: {analysis.file_path}")
        results.append(f"   📊 {analysis.semantic_summary}")
        
        # Basic info
        basic = analysis.basic_info
        results.append(f"   Size: {basic['size_chars']:,} chars | Lines: {basic['line_count']:,}")
        
        if detail_level in ['structure', 'hierarchical', 'full']:
            # Hierarchical structure
            structure = analysis.hierarchical_structure
            
            if structure.get('file_summary'):
                summary = structure['file_summary']
                results.append(f"   🏗️  Architecture: {summary.get('architecture_notes', 'N/A')}")
                
                if summary.get('key_exports'):
                    results.append(f"   📤 Exports: {', '.join(summary['key_exports'][:5])}")
                
                if summary.get('key_imports'):
                    results.append(f"   📥 Imports: {', '.join(summary['key_imports'][:5])}")
            
            # Chunk distribution
            if structure.get('chunk_distribution'):
                dist = structure['chunk_distribution']
                results.append(f"   🧩 Chunks: {dist['total_chunks']} total "
                             f"({dist['symbol_chunks']} symbols, {dist['block_chunks']} blocks)")
            
            # Complexity analysis
            if structure.get('complexity_analysis'):
                complexity = structure['complexity_analysis']
                results.append(f"   🎯 Complexity: {complexity['avg_complexity']:.2f} avg, "
                             f"{complexity['complex_chunks']} high-complexity chunks")
        
        if detail_level in ['hierarchical', 'full']:
            # Key symbols
            if analysis.key_symbols:
                results.append("   📋 Key Symbols:")
                for symbol in analysis.key_symbols[:8]:  # Show top 8
                    signature = symbol['signature']
                    complexity_indicator = "🔴" if symbol['complexity'] > 0.7 else "🟡" if symbol['complexity'] > 0.4 else "🟢"
                    results.append(f"      {complexity_indicator} {signature}")
                
                if len(analysis.key_symbols) > 8:
                    results.append(f"      ... and {len(analysis.key_symbols) - 8} more symbols")
            
            # Relationships
            if analysis.context_relationships:
                parent_child_count = len([r for r in analysis.context_relationships if r['type'] == 'parent_child'])
                contains_count = len([r for r in analysis.context_relationships if r['type'] == 'contains'])
                results.append(f"   🔗 Relationships: {parent_child_count} hierarchical, {contains_count} containment")
        
        results.append("   " + "-" * 50)
        return results
    
    def _generate_multi_file_summary(self, analyses: List[FileAnalysis]) -> List[str]:
        """Generate summary across multiple files."""
        summary = []
        
        successful = [a for a in analyses if a.success]
        failed = [a for a in analyses if not a.success]
        
        summary.append(f"✅ Successfully analyzed: {len(successful)} files")
        if failed:
            summary.append(f"❌ Failed to analyze: {len(failed)} files")
        
        if successful:
            # Total metrics
            total_lines = sum(a.basic_info.get('line_count', 0) for a in successful)
            total_chars = sum(a.basic_info.get('size_chars', 0) for a in successful)
            
            summary.append(f"📊 Total: {total_lines:,} lines, {total_chars:,} characters")
            
            # File types
            extensions = {}
            for analysis in successful:
                ext = analysis.basic_info.get('file_extension', 'unknown')
                extensions[ext] = extensions.get(ext, 0) + 1
            
            if extensions:
                ext_summary = ', '.join(f"{count} {ext}" for ext, count in extensions.items())
                summary.append(f"📁 File types: {ext_summary}")
            
            # Symbol summary
            total_symbols = sum(len(a.key_symbols) for a in successful)
            if total_symbols > 0:
                symbol_types = {}
                for analysis in successful:
                    for symbol in analysis.key_symbols:
                        sym_type = symbol['type']
                        symbol_types[sym_type] = symbol_types.get(sym_type, 0) + 1
                
                symbol_summary = ', '.join(f"{count} {sym_type}s" for sym_type, count in symbol_types.items())
                summary.append(f"🎯 Symbols: {symbol_summary}")
        
        return summary