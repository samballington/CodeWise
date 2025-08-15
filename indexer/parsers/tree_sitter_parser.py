"""
Tree-sitter Parser Factory for Unified Code Analysis

Provides unified parsing capabilities across multiple programming languages
using tree-sitter. Falls back to existing AST parsers when tree-sitter 
grammars are not available.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import ast
import re

# Tree-sitter imports (optional)
try:
    from tree_sitter import Language, Parser, Tree, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Define placeholder classes for type hints
    class Tree:
        @property
        def root_node(self): pass
    class Node:
        @property 
        def children(self): pass
        @property
        def type(self): pass
        @property
        def start_point(self): pass
        @property
        def end_point(self): pass

# JavaScript/TypeScript parsing fallback
try:
    import esprima
    ESPRIMA_AVAILABLE = True
except ImportError:
    ESPRIMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class TreeSitterFactory:
    """
    Factory for creating language-specific parsers using tree-sitter.
    
    Provides a unified interface for parsing multiple programming languages
    with fallback support for existing AST parsers when tree-sitter grammars
    are not available.
    """
    
    def __init__(self, grammars_dir: str = "grammars/"):
        """
        Initialize the parser factory.
        
        Args:
            grammars_dir: Directory containing compiled tree-sitter grammars
        """
        self.grammars_dir = Path(grammars_dir)
        self._parsers: Dict[str, Any] = {}
        self._languages: Dict[str, Any] = {}
        self._initialize_languages()
    
    def _initialize_languages(self) -> None:
        """Initialize available language parsers."""
        if not TREE_SITTER_AVAILABLE:
            logger.info("Tree-sitter not available, using fallback parsers")
            return
        
        # Language configurations for tree-sitter
        language_configs = {
            'python': 'tree-sitter-python',
            'javascript': 'tree-sitter-javascript', 
            'typescript': 'tree-sitter-typescript/typescript',
            'go': 'tree-sitter-go',
            'rust': 'tree-sitter-rust',
            'java': 'tree-sitter-java',
            'cpp': 'tree-sitter-cpp',
            'c': 'tree-sitter-c'
        }
        
        for lang_name, grammar_path in language_configs.items():
            try:
                so_file = self.grammars_dir / f"{grammar_path}.so"
                if so_file.exists():
                    language = Language(str(so_file))
                    self._languages[lang_name] = language
                    parser = Parser()
                    parser.set_language(language)
                    self._parsers[lang_name] = parser
                    logger.debug(f"Loaded tree-sitter parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to load {lang_name} grammar: {e}")
    
    def get_parser(self, file_extension: str) -> Optional[Any]:
        """
        Get parser for a given file extension.
        
        Args:
            file_extension: File extension (e.g., '.py', '.js')
            
        Returns:
            Parser instance or None if not available
        """
        # Map file extensions to language parsers
        extension_map = {
            '.py': 'python',
            '.pyx': 'python',
            '.pyw': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript', 
            '.mjs': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.hpp': 'cpp',
            '.hxx': 'cpp',
            '.c': 'c',
            '.h': 'c'
        }
        
        language = extension_map.get(file_extension.lower())
        if language and language in self._parsers:
            return self._parsers[language]
        
        return None
    
    def parse_content(self, content: str, file_path: Path) -> Optional[Tree]:
        """
        Parse content using appropriate parser.
        
        Args:
            content: Source code content
            file_path: Path to the file being parsed
            
        Returns:
            Parsed tree or None if parsing fails
        """
        if TREE_SITTER_AVAILABLE:
            parser = self.get_parser(file_path.suffix)
            if parser:
                try:
                    return parser.parse(bytes(content, 'utf8'))
                except Exception as e:
                    logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
        
        # Fallback to language-specific parsers
        return self._fallback_parse(content, file_path)
    
    def _fallback_parse(self, content: str, file_path: Path) -> Optional['FallbackTree']:
        """
        Fallback parsing using existing AST parsers.
        
        Args:
            content: Source code content
            file_path: Path to the file being parsed
            
        Returns:
            Fallback tree wrapper or None
        """
        extension = file_path.suffix.lower()
        
        if extension == '.py':
            return self._parse_python_fallback(content)
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            return self._parse_javascript_fallback(content)
        
        return None
    
    def _parse_python_fallback(self, content: str) -> Optional['FallbackTree']:
        """Parse Python using built-in AST."""
        try:
            tree = ast.parse(content)
            return FallbackTree(tree, 'python')
        except SyntaxError as e:
            logger.warning(f"Python AST parsing failed: {e}")
            return None
    
    def _parse_javascript_fallback(self, content: str) -> Optional['FallbackTree']:
        """Parse JavaScript/TypeScript using esprima."""
        if not ESPRIMA_AVAILABLE:
            return None
        
        try:
            tree = esprima.parseScript(content, options={'loc': True, 'range': True})
            return FallbackTree(tree, 'javascript')
        except Exception as e:
            logger.warning(f"JavaScript parsing failed: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        extensions = ['.py', '.pyx', '.pyw']  # Python always supported
        
        if ESPRIMA_AVAILABLE:
            extensions.extend(['.js', '.jsx', '.ts', '.tsx', '.mjs'])
        
        if TREE_SITTER_AVAILABLE:
            # Add tree-sitter supported extensions
            tree_sitter_extensions = [
                '.go', '.rs', '.java', '.cpp', '.cxx', '.cc', 
                '.hpp', '.hxx', '.c', '.h'
            ]
            extensions.extend(tree_sitter_extensions)
        
        return sorted(set(extensions))
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about available parsers."""
        return {
            'tree_sitter_available': TREE_SITTER_AVAILABLE,
            'esprima_available': ESPRIMA_AVAILABLE,
            'supported_languages': list(self._parsers.keys()) if TREE_SITTER_AVAILABLE else [],
            'supported_extensions': self.get_supported_extensions(),
            'fallback_parsers': {
                'python': True,
                'javascript': ESPRIMA_AVAILABLE
            }
        }


class FallbackTree:
    """
    Wrapper for non-tree-sitter AST trees to provide unified interface.
    
    This allows the hierarchical chunker to work with both tree-sitter
    and traditional AST parsers through the same interface.
    """
    
    def __init__(self, ast_tree: Any, language: str):
        """
        Initialize fallback tree wrapper.
        
        Args:
            ast_tree: AST tree from language-specific parser
            language: Programming language name
        """
        self._ast_tree = ast_tree
        self._language = language
        self.root_node = FallbackNode(ast_tree, language)
    
    def walk(self):
        """Walk through the tree nodes."""
        yield from self.root_node.walk()


class FallbackNode:
    """
    Wrapper for AST nodes to provide tree-sitter-like interface.
    """
    
    def __init__(self, ast_node: Any, language: str, parent: Optional['FallbackNode'] = None):
        """
        Initialize fallback node wrapper.
        
        Args:
            ast_node: AST node from language-specific parser
            language: Programming language name
            parent: Parent node wrapper
        """
        self._ast_node = ast_node
        self._language = language
        self._parent = parent
        self._children = []
        self._initialize_children()
    
    def _initialize_children(self):
        """Initialize child nodes."""
        if self._language == 'python':
            self._initialize_python_children()
        elif self._language == 'javascript':
            self._initialize_javascript_children()
    
    def _initialize_python_children(self):
        """Initialize children for Python AST nodes."""
        if hasattr(self._ast_node, 'body'):
            for child in self._ast_node.body:
                self._children.append(FallbackNode(child, self._language, self))
        
        # Add other child attributes for different node types
        if hasattr(self._ast_node, 'orelse'):
            for child in self._ast_node.orelse:
                self._children.append(FallbackNode(child, self._language, self))
    
    def _initialize_javascript_children(self):
        """Initialize children for JavaScript AST nodes."""
        if hasattr(self._ast_node, 'body'):
            body = self._ast_node.body
            if isinstance(body, list):
                for child in body:
                    self._children.append(FallbackNode(child, self._language, self))
            else:
                self._children.append(FallbackNode(body, self._language, self))
    
    @property
    def type(self) -> str:
        """Get node type."""
        if self._language == 'python':
            return self._ast_node.__class__.__name__
        elif self._language == 'javascript':
            return getattr(self._ast_node, 'type', 'Unknown')
        return 'Unknown'
    
    @property
    def children(self) -> List['FallbackNode']:
        """Get child nodes."""
        return self._children
    
    @property
    def start_point(self) -> tuple:
        """Get start position (row, column)."""
        if self._language == 'python':
            return (getattr(self._ast_node, 'lineno', 1) - 1, 
                   getattr(self._ast_node, 'col_offset', 0))
        elif self._language == 'javascript':
            loc = getattr(self._ast_node, 'loc', {})
            start = loc.get('start', {'line': 1, 'column': 0})
            return (start['line'] - 1, start['column'])
        return (0, 0)
    
    @property
    def end_point(self) -> tuple:
        """Get end position (row, column)."""
        if self._language == 'python':
            return (getattr(self._ast_node, 'end_lineno', self.start_point[0]) - 1,
                   getattr(self._ast_node, 'end_col_offset', self.start_point[1]))
        elif self._language == 'javascript':
            loc = getattr(self._ast_node, 'loc', {})
            end = loc.get('end', {'line': 1, 'column': 0})
            return (end['line'] - 1, end['column'])
        return self.start_point
    
    def walk(self):
        """Walk through node and its children."""
        yield self
        for child in self._children:
            yield from child.walk()


# Factory instance for global use
tree_sitter_factory = TreeSitterFactory()