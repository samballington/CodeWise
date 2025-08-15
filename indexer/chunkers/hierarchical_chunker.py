"""
Hierarchical Code Chunker with Bidirectional Relationships

Replaces the existing flat AST chunking system with a sophisticated
hierarchical structure that maintains explicit parent-child relationships
and supports multiple chunk granularities.
"""

import logging
import ast
from pathlib import Path
from typing import List, Dict, Optional, Union, Set, Any
from collections import defaultdict

# Import our schemas and parsers
from ..schemas.chunk_schemas import (
    SymbolChunk, BlockChunk, SummaryChunk, ChunkType, AnyChunk,
    create_chunk_id, validate_chunk_hierarchy
)
from ..parsers.tree_sitter_parser import TreeSitterFactory, FallbackNode

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Advanced code chunker that builds hierarchical structures with bidirectional relationships.
    
    Key features:
    - Unified tree-sitter parsing for multiple languages
    - In-memory relationship tracking during AST traversal
    - Persistent bidirectional links for cross-session context reconstruction
    - Three chunk granularities: Symbol, Block, Summary
    """
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 50):
        """
        Initialize the hierarchical chunker.
        
        Args:
            max_chunk_size: Maximum characters in a chunk
            min_chunk_size: Minimum characters for a valid chunk
        """
        self.parser_factory = TreeSitterFactory()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # In-memory relationship tracking during traversal
        self._chunk_counter = 0
        self._parent_stack: List[str] = []
        self._relationship_map: Dict[str, Dict] = {}
        self._current_file_path = ""
        self._source_lines: List[str] = []
    
    def chunk_file(self, content: str, file_path: Path) -> List[AnyChunk]:
        """
        Chunk a file into hierarchical structure with bidirectional relationships.
        
        Args:
            content: Source code content
            file_path: Path to the file being chunked
            
        Returns:
            List of hierarchical chunks with bidirectional relationships
        """
        self._reset_state()
        self._current_file_path = str(file_path)
        self._source_lines = content.splitlines()
        
        logger.info(f"Chunking file: {file_path}")
        
        # Parse the content
        tree = self.parser_factory.parse_content(content, file_path)
        if not tree:
            logger.warning(f"Failed to parse {file_path}, using fallback chunking")
            return self._fallback_chunking(content, file_path)
        
        chunks = []
        
        # Create file-level summary chunk first
        file_summary = self._create_file_summary(content, file_path)
        chunks.append(file_summary)
        self._parent_stack.append(file_summary.id)
        
        # Initialize relationship tracking for summary chunk
        self._relationship_map[file_summary.id] = {
            'parent_id': None,
            'child_ids': [],
            'type': ChunkType.SUMMARY
        }
        
        # Traverse tree and extract hierarchical chunks
        self._traverse_node(tree.root_node, content, chunks)
        
        # Apply bidirectional relationships to chunk objects
        self._apply_bidirectional_relationships(chunks)
        
        # Validate the chunk hierarchy
        validation_results = validate_chunk_hierarchy(chunks)
        if not validation_results['valid']:
            logger.warning(f"Chunk validation failed for {file_path}: {validation_results['errors']}")
        
        logger.info(f"Created {len(chunks)} chunks for {file_path}: {validation_results['type_distribution']}")
        return chunks
    
    def _reset_state(self):
        """Reset internal state for new file processing."""
        self._chunk_counter = 0
        self._parent_stack.clear()
        self._relationship_map.clear()
        self._current_file_path = ""
        self._source_lines.clear()
    
    def _create_file_summary(self, content: str, file_path: Path) -> SummaryChunk:
        """Create file-level summary chunk."""
        chunk_id = create_chunk_id(str(file_path), chunk_type=ChunkType.SUMMARY)
        
        # Extract key information for summary
        key_exports, key_imports = self._extract_file_interface(content, file_path.suffix)
        architecture_notes = self._generate_architecture_notes(content, file_path)
        
        # Create content snippet (first few lines or docstring)
        summary_content = self._create_summary_content(content)
        
        return SummaryChunk(
            id=chunk_id,
            content=summary_content,
            summary_type="file",
            child_chunk_ids=[],  # Will be populated during traversal
            file_path=str(file_path),
            line_start=1,
            line_end=len(self._source_lines),
            key_exports=key_exports,
            key_imports=key_imports,
            architecture_notes=architecture_notes
        )
    
    def _traverse_node(self, node: Any, content: str, chunks: List[AnyChunk]):
        """
        Traverse AST node and build chunks with in-memory relationship tracking.
        
        Args:
            node: AST node (tree-sitter or fallback)
            content: Source content
            chunks: List to append chunks to
        """
        # Check if this node should create a chunk
        chunk = self._create_chunk_from_node(node, content)
        
        if chunk:
            chunks.append(chunk)
            
            # Update bidirectional relationships in memory
            current_parent_id = self._parent_stack[-1] if self._parent_stack else None
            
            # Initialize relationship for this chunk
            self._relationship_map[chunk.id] = {
                'parent_id': current_parent_id,
                'child_ids': [],
                'type': chunk.type
            }
            
            # Add this chunk as child to its parent
            if current_parent_id and current_parent_id in self._relationship_map:
                self._relationship_map[current_parent_id]['child_ids'].append(chunk.id)
            
            # Push to stack if this chunk can contain others
            if chunk.type in [ChunkType.SUMMARY, ChunkType.BLOCK]:
                self._parent_stack.append(chunk.id)
                
                # Process children
                for child in node.children:
                    self._traverse_node(child, content, chunks)
                
                # Pop from stack
                self._parent_stack.pop()
        else:
            # Process children even if no chunk created
            for child in node.children:
                self._traverse_node(child, content, chunks)
    
    def _create_chunk_from_node(self, node: Any, content: str) -> Optional[AnyChunk]:
        """
        Create appropriate chunk from AST node.
        
        Args:
            node: AST node
            content: Source content
            
        Returns:
            Chunk instance or None if node doesn't warrant a chunk
        """
        node_type = node.type
        start_line, start_col = node.start_point
        end_line, end_col = node.end_point
        
        # Adjust for 1-indexed lines
        start_line += 1
        end_line += 1
        
        # Extract content for this node
        node_content = self._extract_node_content(start_line, end_line, start_col, end_col)
        
        # Skip very small chunks
        if len(node_content.strip()) < self.min_chunk_size:
            return None
        
        # Determine chunk type based on node type
        if self._is_symbol_node(node_type):
            return self._create_symbol_chunk(node, node_content, start_line, end_line)
        elif self._is_block_node(node_type):
            return self._create_block_chunk(node, node_content, start_line, end_line)
        
        return None
    
    def _is_symbol_node(self, node_type: str) -> bool:
        """Check if node represents a symbol (function, class, etc.)."""
        symbol_types = {
            # Python
            'FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'Import', 'ImportFrom',
            'function_definition', 'class_definition', 'import_statement',
            # JavaScript/TypeScript
            'FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression',
            'ClassDeclaration', 'MethodDefinition', 'VariableDeclaration',
            'function_declaration', 'class_declaration', 'method_definition',
            # Other languages
            'function', 'method', 'class', 'interface', 'struct', 'enum'
        }
        return node_type in symbol_types
    
    def _is_block_node(self, node_type: str) -> bool:
        """Check if node represents a block (class body, module, etc.)."""
        block_types = {
            # Python
            'Module', 'ClassDef',
            # JavaScript/TypeScript  
            'Program', 'ClassBody', 'BlockStatement',
            # General
            'module', 'class_body', 'block', 'namespace'
        }
        return node_type in block_types
    
    def _create_symbol_chunk(self, node: Any, content: str, start_line: int, end_line: int) -> SymbolChunk:
        """Create a symbol chunk from AST node."""
        node_type = node.type
        
        # Extract symbol information
        symbol_name = self._extract_symbol_name(node)
        symbol_type = self._map_symbol_type(node_type)
        
        # Generate unique chunk ID
        chunk_id = create_chunk_id(
            self._current_file_path, 
            symbol_name, 
            ChunkType.SYMBOL
        )
        
        # Extract additional metadata
        parameters = self._extract_parameters(node)
        return_type = self._extract_return_type(node)
        docstring = self._extract_docstring(node, content)
        decorators = self._extract_decorators(node)
        complexity_score = self._calculate_complexity(node, content)
        
        # Get parent chunk ID
        parent_chunk_id = self._parent_stack[-1] if self._parent_stack else None
        
        return SymbolChunk(
            id=chunk_id,
            content=content,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            parent_chunk_id=parent_chunk_id,
            file_path=self._current_file_path,
            line_start=start_line,
            line_end=end_line,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            decorators=decorators,
            complexity_score=complexity_score
        )
    
    def _create_block_chunk(self, node: Any, content: str, start_line: int, end_line: int) -> BlockChunk:
        """Create a block chunk from AST node."""
        node_type = node.type
        
        # Map node type to block type
        block_type = self._map_block_type(node_type)
        
        # Generate unique chunk ID
        chunk_id = create_chunk_id(
            self._current_file_path,
            chunk_type=ChunkType.BLOCK
        )
        
        # Extract imports and exports
        imports = self._extract_imports_from_node(node)
        exports = self._extract_exports_from_node(node)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(node, content)
        
        # Get parent chunk ID
        parent_chunk_id = self._parent_stack[-1] if self._parent_stack else None
        
        return BlockChunk(
            id=chunk_id,
            content=content,
            block_type=block_type,
            parent_chunk_id=parent_chunk_id,
            child_chunk_ids=[],  # Will be populated during relationship application
            file_path=self._current_file_path,
            line_start=start_line,
            line_end=end_line,
            imports=imports,
            exports=exports,
            complexity_score=complexity_score
        )
    
    def _apply_bidirectional_relationships(self, chunks: List[AnyChunk]):
        """Apply the in-memory relationship map to chunk objects."""
        logger.debug(f"Applying bidirectional relationships to {len(chunks)} chunks")
        
        for chunk in chunks:
            if chunk.id in self._relationship_map:
                relationships = self._relationship_map[chunk.id]
                
                # Apply based on chunk type
                if isinstance(chunk, SummaryChunk):
                    chunk.child_chunk_ids = relationships['child_ids']
                elif isinstance(chunk, BlockChunk):
                    chunk.parent_chunk_id = relationships['parent_id']
                    chunk.child_chunk_ids = relationships['child_ids']
                elif isinstance(chunk, SymbolChunk):
                    chunk.parent_chunk_id = relationships['parent_id']
        
        # Clear in-memory map after applying
        self._relationship_map.clear()
    
    def _extract_node_content(self, start_line: int, end_line: int, 
                             start_col: int, end_col: int) -> str:
        """Extract content for a node from source lines."""
        if start_line > len(self._source_lines) or end_line > len(self._source_lines):
            return ""
        
        if start_line == end_line:
            # Single line
            line = self._source_lines[start_line - 1]
            return line[start_col:end_col] if end_col > start_col else line[start_col:]
        else:
            # Multi-line
            lines = []
            for i in range(start_line - 1, min(end_line, len(self._source_lines))):
                if i == start_line - 1:
                    # First line
                    lines.append(self._source_lines[i][start_col:])
                elif i == end_line - 1:
                    # Last line
                    lines.append(self._source_lines[i][:end_col])
                else:
                    # Middle lines
                    lines.append(self._source_lines[i])
            return '\n'.join(lines)
    
    def _extract_symbol_name(self, node: Any) -> str:
        """Extract symbol name from AST node."""
        if hasattr(node, '_ast_node'):
            # Fallback node
            ast_node = node._ast_node
            if hasattr(ast_node, 'name'):
                return ast_node.name
        
        # For tree-sitter nodes, we would extract from node structure
        # This is a simplified implementation
        return f"symbol_{self._chunk_counter}"
    
    def _map_symbol_type(self, node_type: str) -> str:
        """Map AST node type to symbol type."""
        mapping = {
            'FunctionDef': 'function',
            'AsyncFunctionDef': 'async_function',
            'ClassDef': 'class',
            'Import': 'import',
            'ImportFrom': 'import',
            'function_definition': 'function',
            'class_definition': 'class',
            'FunctionDeclaration': 'function',
            'ClassDeclaration': 'class',
            'MethodDefinition': 'method',
            'VariableDeclaration': 'variable'
        }
        return mapping.get(node_type, 'unknown')
    
    def _map_block_type(self, node_type: str) -> str:
        """Map AST node type to block type."""
        mapping = {
            'Module': 'module',
            'ClassDef': 'class_body',
            'Program': 'module',
            'ClassBody': 'class_body',
            'BlockStatement': 'block'
        }
        return mapping.get(node_type, 'module')
    
    def _extract_parameters(self, node: Any) -> List[str]:
        """Extract function parameters from AST node."""
        if hasattr(node, '_ast_node'):
            ast_node = node._ast_node
            if hasattr(ast_node, 'args') and hasattr(ast_node.args, 'args'):
                return [arg.arg for arg in ast_node.args.args]
        return []
    
    def _extract_return_type(self, node: Any) -> Optional[str]:
        """Extract return type annotation."""
        if hasattr(node, '_ast_node'):
            ast_node = node._ast_node
            if hasattr(ast_node, 'returns') and ast_node.returns:
                return ast.unparse(ast_node.returns)
        return None
    
    def _extract_docstring(self, node: Any, content: str) -> Optional[str]:
        """Extract docstring from function or class."""
        if hasattr(node, '_ast_node'):
            ast_node = node._ast_node
            if hasattr(ast_node, 'body') and ast_node.body:
                first = ast_node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    if isinstance(first.value.value, str):
                        return first.value.value
        return None
    
    def _extract_decorators(self, node: Any) -> List[str]:
        """Extract decorator names."""
        if hasattr(node, '_ast_node'):
            ast_node = node._ast_node
            if hasattr(ast_node, 'decorator_list'):
                decorators = []
                for decorator in ast_node.decorator_list:
                    if hasattr(decorator, 'id'):
                        decorators.append(decorator.id)
                    elif hasattr(decorator, 'attr'):
                        decorators.append(decorator.attr)
                return decorators
        return []
    
    def _calculate_complexity(self, node: Any, content: str) -> float:
        """Calculate complexity score for a chunk."""
        # Simplified complexity calculation
        lines = content.count('\n') + 1
        # More lines = higher complexity, but cap at 1.0
        return min(lines / 100.0, 1.0)
    
    def _extract_imports_from_node(self, node: Any) -> List[str]:
        """Extract import statements from block."""
        # Implementation would depend on node structure
        return []
    
    def _extract_exports_from_node(self, node: Any) -> List[str]:
        """Extract export statements from block."""
        # Implementation would depend on node structure
        return []
    
    def _extract_file_interface(self, content: str, file_extension: str) -> tuple[List[str], List[str]]:
        """Extract key exports and imports from file content."""
        exports = []
        imports = []
        
        if file_extension == '.py':
            # Python-specific extraction
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not node.name.startswith('_'):  # Public symbols
                            exports.append(node.name)
            except SyntaxError:
                pass
        
        return exports, imports
    
    def _generate_architecture_notes(self, content: str, file_path: Path) -> str:
        """Generate architectural observations for the file."""
        notes = []
        
        # File type analysis
        extension = file_path.suffix
        if extension == '.py':
            if 'class ' in content:
                notes.append("Contains class definitions")
            if 'def ' in content:
                notes.append("Contains function definitions")
            if 'import ' in content:
                notes.append("Has external dependencies")
        
        # Size analysis
        lines = content.count('\n') + 1
        if lines > 1000:
            notes.append("Large file - consider refactoring")
        elif lines < 50:
            notes.append("Small utility file")
        
        return "; ".join(notes) if notes else "Standard code file"
    
    def _create_summary_content(self, content: str) -> str:
        """Create summary content from file content."""
        lines = content.splitlines()
        
        # Try to find module docstring first
        if len(lines) > 0:
            # Look for module-level docstring
            for i, line in enumerate(lines[:10]):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Found docstring, extract it
                    quote_type = '"""' if stripped.startswith('"""') else "'''"
                    docstring_lines = []
                    
                    if stripped.count(quote_type) >= 2:
                        # Single line docstring
                        return stripped.strip(quote_type).strip()
                    else:
                        # Multi-line docstring
                        docstring_lines.append(stripped.strip(quote_type))
                        for j in range(i + 1, len(lines)):
                            line = lines[j]
                            if quote_type in line:
                                docstring_lines.append(line[:line.index(quote_type)])
                                break
                            docstring_lines.append(line)
                        return '\n'.join(docstring_lines).strip()
        
        # Fallback: use first few non-empty lines
        non_empty_lines = [line for line in lines[:5] if line.strip()]
        return '\n'.join(non_empty_lines[:3]) if non_empty_lines else "Code file"
    
    def _fallback_chunking(self, content: str, file_path: Path) -> List[AnyChunk]:
        """Fallback chunking when AST parsing fails."""
        logger.warning(f"Using fallback chunking for {file_path}")
        
        # Create a simple summary chunk
        summary_chunk = SummaryChunk(
            id=create_chunk_id(str(file_path), chunk_type=ChunkType.SUMMARY),
            content=content[:min(500, len(content))],
            summary_type="file",
            child_chunk_ids=[],
            file_path=str(file_path),
            line_start=1,
            line_end=len(content.splitlines()),
            key_exports=[],
            key_imports=[],
            architecture_notes="Fallback parsing - AST analysis failed"
        )
        
        return [summary_chunk]