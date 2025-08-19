"""
Symbol Collector - Pass 1 of AST-to-Graph Pipeline

Discovers all symbols (functions, classes, variables) across all files.
Builds in-memory symbol table for Pass 2 relationship extraction.

Architectural Pattern: Visitor pattern for AST traversal with language-agnostic 
symbol extraction using tree-sitter node types.
"""

from typing import Dict, Set, List, Optional, Any
from pathlib import Path
import logging
import re
try:
    from tree_sitter import Tree, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Mock classes for graceful degradation
    class Tree:
        pass
    class Node:
        pass

# Import Phase 1 components
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from indexer.parsers.tree_sitter_parser import TreeSitterFactory
    TREE_SITTER_FACTORY_AVAILABLE = True
except ImportError:
    TREE_SITTER_FACTORY_AVAILABLE = False
    class TreeSitterFactory:
        def parse_content(self, content, file_path):
            return None
from storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SymbolCollector:
    """
    Pass 1: Discovers all symbols (functions, classes, variables) across all files.
    
    Core Responsibility: Build comprehensive symbol table that enables accurate
    symbol resolution in Pass 2 relationship extraction.
    
    Design Decision: Language-agnostic approach using tree-sitter node types
    provides consistent symbol extraction across 87+ supported file types.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.parser_factory = TreeSitterFactory() if TREE_SITTER_FACTORY_AVAILABLE else None
        self.symbol_table: Dict[str, Dict] = {}  # Global symbol table: symbol_id -> symbol_info
        self.file_symbols: Dict[str, Set[str]] = {}  # Per-file symbol tracking: file_path -> symbol_ids
        self.collection_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'symbols_discovered': 0,
            'symbols_by_type': {},
            'processing_errors': []
        }
    
    def collect_all_symbols(self, file_paths: List[Path]) -> Dict[str, Dict]:
        """
        Discover all symbols across all files in the codebase.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Global symbol table for use in Pass 2
        """
        logger.info(f"Starting symbol collection across {len(file_paths)} files")
        
        # Reset collection stats
        self.collection_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'symbols_discovered': 0,
            'symbols_by_type': {},
            'processing_errors': []
        }
        
        # Check if tree-sitter is available
        if not TREE_SITTER_AVAILABLE or not TREE_SITTER_FACTORY_AVAILABLE:
            logger.warning("Tree-sitter not available. Symbol collection will use fallback methods.")
            return self._fallback_symbol_collection(file_paths)
        
        for file_path in file_paths:
            try:
                self._collect_file_symbols(file_path)
                self.collection_stats['files_processed'] += 1
                
                # Update file processing status
                file_symbols = self.file_symbols.get(str(file_path), set())
                self.db_manager.update_file_status(
                    str(file_path), 
                    'completed',
                    language=self._detect_file_language(file_path),
                    nodes_count=len(file_symbols)
                )
                
            except Exception as e:
                error_msg = f"Symbol collection failed for {file_path}: {e}"
                logger.error(error_msg)
                self.collection_stats['files_failed'] += 1
                self.collection_stats['processing_errors'].append(error_msg)
                
                # Update file error status
                self.db_manager.update_file_status(
                    str(file_path), 
                    'error',
                    error_message=str(e)
                )
                continue
        
        self.collection_stats['symbols_discovered'] = len(self.symbol_table)
        
        logger.info(f"Symbol collection completed:")
        logger.info(f"  Files processed: {self.collection_stats['files_processed']}")
        logger.info(f"  Files failed: {self.collection_stats['files_failed']}")
        logger.info(f"  Symbols discovered: {self.collection_stats['symbols_discovered']}")
        
        return self.symbol_table
    
    def _collect_file_symbols(self, file_path: Path):
        """Process a single file and extract all symbol definitions."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding='latin-1')
            except Exception as e:
                logger.warning(f"Could not read {file_path} with any encoding: {e}")
                return
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return
        
        # Mark file as processing
        self.db_manager.update_file_status(str(file_path), 'processing')
        
        # Parse with detailed error reporting
        tree = None
        if self.parser_factory:
            try:
                tree = self.parser_factory.parse_content(content, file_path)
            except Exception as e:
                logger.error(f"Parse exception for {file_path}: {e}")
                return
        
        if not tree:
            logger.warning(f"Could not parse {file_path} - parser returned None")
            return
        
        file_symbols = set()
        self._traverse_for_symbols(tree.root_node, content, file_path, file_symbols)
        self.file_symbols[str(file_path)] = file_symbols
        
        logger.debug(f"Collected {len(file_symbols)} symbols from {file_path}")
    
    def _traverse_for_symbols(self, node: Node, content: str, file_path: Path, 
                             file_symbols: Set[str]):
        """Recursively traverse AST to find symbol definitions."""
        
        # Language-agnostic symbol extraction using tree-sitter node types
        # These node types are consistent across tree-sitter grammars
        symbol_node_types = {
            # Python
            'function_definition',           # def func():
            'async_function_definition',     # async def func():
            'class_definition',              # class MyClass:
            'assignment',                    # variable = value
            'import_statement',              # import module
            'import_from_statement',         # from module import symbol
            
            # JavaScript/TypeScript
            'function_declaration',          # function func() {}
            'function_expression',           # const func = function() {}
            'arrow_function',                # const func = () => {}
            'class_declaration',             # class MyClass {}
            'method_definition',             # method() {} in class
            'variable_declaration',          # const/let/var declarations
            'export_statement',              # export declarations
            'import_statement',              # import statements
            
            # Java
            'method_declaration',            # public void method()
            'constructor_declaration',       # public MyClass()
            'field_declaration',             # private int field
            'interface_declaration',         # interface MyInterface
            'enum_declaration',              # enum MyEnum
            
            # C/C++
            'function_declarator',           # function declarations
            'struct_specifier',              # struct definitions
            'enum_specifier',                # enum definitions
            'type_definition',               # typedef
            
            # Rust
            'function_item',                 # fn function_name()
            'struct_item',                   # struct StructName
            'enum_item',                     # enum EnumName
            'impl_item',                     # impl blocks
            'trait_item',                    # trait definitions
            
            # Go
            'function_declaration',          # func name()
            'method_declaration',            # func (receiver) name()
            'type_declaration',              # type declarations
            'var_declaration',               # var declarations
            'const_declaration',             # const declarations
        }
        
        if node.type in symbol_node_types:
            symbol_info = self._extract_symbol_info(node, content, file_path)
            if symbol_info:
                symbol_id = symbol_info['id']
                self.symbol_table[symbol_id] = symbol_info
                file_symbols.add(symbol_id)
                
                # Insert into database
                success = self.db_manager.insert_node(
                    node_id=symbol_id,
                    node_type=symbol_info['type'],
                    name=symbol_info['name'],
                    file_path=str(file_path),
                    line_start=symbol_info['line_start'],
                    line_end=symbol_info['line_end'],
                    signature=symbol_info.get('signature'),
                    docstring=symbol_info.get('docstring'),
                    properties=symbol_info.get('properties', {})
                )
                
                if success:
                    # Update symbol type statistics
                    symbol_type = symbol_info['type']
                    self.collection_stats['symbols_by_type'][symbol_type] = \
                        self.collection_stats['symbols_by_type'].get(symbol_type, 0) + 1
                else:
                    logger.warning(f"Failed to insert symbol {symbol_id} into database")
        
        # Recurse into children
        for child in node.children:
            self._traverse_for_symbols(child, content, file_path, file_symbols)
    
    def _extract_symbol_info(self, node: Node, content: str, file_path: Path) -> Optional[Dict]:
        """
        Extract symbol information from an AST node.
        
        Returns comprehensive symbol metadata including location,
        signature, and language-specific properties.
        """
        try:
            lines = content.splitlines()
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # Extract symbol name based on node type
            symbol_name = self._get_symbol_name(node, content)
            if not symbol_name:
                return None
            
            # Generate unique ID: file::symbol::line for disambiguation
            sanitized_file = re.sub(r'[^a-zA-Z0-9_.]', '_', file_path.stem)
            sanitized_symbol = re.sub(r'[^a-zA-Z0-9_]', '_', symbol_name)
            symbol_id = f"{sanitized_file}::{sanitized_symbol}::{start_line}"
            
            # Extract signature/definition (first line of symbol)
            symbol_text = content[node.start_byte:node.end_byte]
            first_line = symbol_text.split('\n')[0].strip()
            
            # Extract docstring if available
            docstring = self._extract_docstring(node, content)
            
            # Map node type to our symbol taxonomy
            symbol_type = self._map_node_type_to_symbol_type(node.type)
            
            # Extract language-specific properties
            properties = self._extract_language_properties(node, content, file_path)
            
            symbol_info = {
                'id': symbol_id,
                'name': symbol_name,
                'type': symbol_type,
                'file_path': str(file_path),
                'line_start': start_line,
                'line_end': end_line,
                'signature': first_line,
                'docstring': docstring,
                'properties': {
                    'node_type': node.type,
                    'full_text': symbol_text[:500],  # Truncate for storage efficiency
                    'language': self._detect_file_language(file_path),
                    **properties
                }
            }
            
            return symbol_info
            
        except Exception as e:
            logger.error(f"Failed to extract symbol info from {node.type}: {e}")
            return None
    
    def _get_symbol_name(self, node: Node, content: str) -> Optional[str]:
        """Extract the actual name of the symbol from AST node."""
        
        # Common patterns for finding names in different node types
        # Different languages use different field names for symbol identifiers
        name_fields = ['name', 'id', 'left', 'declarator', 'pattern']
        
        for field in name_fields:
            name_node = node.child_by_field_name(field)
            if name_node:
                name_text = content[name_node.start_byte:name_node.end_byte]
                # Extract just the identifier, handle complex patterns
                return self._extract_identifier_from_text(name_text)
        
        # Fallback: look for identifier nodes in immediate children
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
        
        # For variable declarations, look deeper
        if 'declaration' in node.type or 'assignment' in node.type:
            for child in node.children:
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        return content[grandchild.start_byte:grandchild.end_byte]
        
        return None
    
    def _extract_identifier_from_text(self, text: str) -> str:
        """Extract a clean identifier from potentially complex text."""
        # Handle function pointers, generics, etc.
        # Extract the primary identifier
        
        # Remove type annotations and generics
        text = re.sub(r'<[^>]+>', '', text)  # Remove generics
        text = re.sub(r'\\([^)]*\\)', '', text)  # Remove parameter lists
        
        # Extract the main identifier
        match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
        if match:
            return match.group(0)
        
        return text.strip()
    
    def _map_node_type_to_symbol_type(self, node_type: str) -> str:
        """Map tree-sitter node types to our symbol taxonomy."""
        type_mapping = {
            # Functions
            'function_definition': 'function',
            'async_function_definition': 'function',
            'function_declaration': 'function',
            'function_expression': 'function',
            'arrow_function': 'function',
            'method_declaration': 'method',
            'method_definition': 'method',
            'constructor_declaration': 'method',
            'function_item': 'function',
            
            # Classes and Types
            'class_definition': 'class',
            'class_declaration': 'class',
            'interface_declaration': 'interface',
            'struct_specifier': 'struct',
            'struct_item': 'struct',
            'enum_declaration': 'enum',
            'enum_specifier': 'enum',
            'enum_item': 'enum',
            'trait_item': 'trait',
            'impl_item': 'impl',
            'type_declaration': 'type',
            'type_definition': 'type',
            
            # Variables
            'variable_declaration': 'variable',
            'field_declaration': 'variable',
            'assignment': 'variable',
            'var_declaration': 'variable',
            'const_declaration': 'constant',
            
            # Imports
            'import_statement': 'import',
            'import_from_statement': 'import',
            'export_statement': 'export'
        }
        
        return type_mapping.get(node_type, 'unknown')
    
    def _extract_docstring(self, node: Node, content: str) -> Optional[str]:
        """Extract docstring from function/class node if present."""
        try:
            # Look for string literals as first statement in function/class body
            for child in node.children:
                if child.type in ('block', 'suite', 'compound_statement'):  # Function/class body
                    for statement in child.children:
                        if statement.type in ('expression_statement', 'string_literal', 'string'):
                            text = content[statement.start_byte:statement.end_byte]
                            # Basic docstring detection - starts with quotes
                            if any(text.strip().startswith(quote) for quote in ['"""', "'''", '"', "'"]):
                                # Clean up the docstring
                                cleaned = text.strip().strip('"\'').strip()
                                return cleaned[:200] + '...' if len(cleaned) > 200 else cleaned
            
            # Alternative: look for comment blocks before the symbol
            # This catches languages like Java, C++ that use /** */ or // comments
            start_line = node.start_point[0]
            lines = content.splitlines()
            
            # Look backwards from symbol for comment blocks
            comment_lines = []
            for i in range(max(0, start_line - 5), start_line):
                if i < len(lines):
                    line = lines[i].strip()
                    if line.startswith(('///', '//', '/*', '*', '#')):
                        comment_lines.append(line.lstrip('/*#* '))
            
            if comment_lines:
                docstring = ' '.join(comment_lines)
                return docstring[:200] + '...' if len(docstring) > 200 else docstring
            
        except Exception as e:
            logger.debug(f"Failed to extract docstring: {e}")
        
        return None
    
    def _extract_language_properties(self, node: Node, content: str, file_path: Path) -> Dict[str, Any]:
        """Extract language-specific properties from the symbol."""
        properties = {}
        
        try:
            language = self._detect_file_language(file_path)
            
            if language == 'python':
                properties.update(self._extract_python_properties(node, content))
            elif language in ['javascript', 'typescript']:
                properties.update(self._extract_js_properties(node, content))
            elif language == 'java':
                properties.update(self._extract_java_properties(node, content))
            elif language in ['c', 'cpp']:
                properties.update(self._extract_c_properties(node, content))
            elif language == 'rust':
                properties.update(self._extract_rust_properties(node, content))
            elif language == 'go':
                properties.update(self._extract_go_properties(node, content))
            
        except Exception as e:
            logger.debug(f"Failed to extract language properties: {e}")
        
        return properties
    
    def _detect_file_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c', '.h': 'c',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.scala': 'scala',
            '.clj': 'clojure'
        }
        
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    def _extract_python_properties(self, node: Node, content: str) -> Dict[str, Any]:
        """Extract Python-specific properties."""
        properties = {}
        
        # Check for decorators
        decorators = []
        if node.type in ['function_definition', 'async_function_definition', 'class_definition']:
            # Look for decorator nodes before this node
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    if child.type == 'decorator':
                        decorator_text = content[child.start_byte:child.end_byte]
                        decorators.append(decorator_text.strip())
        
        if decorators:
            properties['decorators'] = decorators
        
        # Check for async functions
        if node.type == 'async_function_definition':
            properties['is_async'] = True
        
        # Extract function parameters
        if node.type in ['function_definition', 'async_function_definition']:
            parameters = self._extract_python_parameters(node, content)
            if parameters:
                properties['parameters'] = parameters
        
        return properties
    
    def _extract_python_parameters(self, node: Node, content: str) -> List[str]:
        """Extract function parameters from Python function."""
        parameters = []
        
        try:
            for child in node.children:
                if child.type == 'parameters':
                    for param_child in child.children:
                        if param_child.type == 'identifier':
                            param_name = content[param_child.start_byte:param_child.end_byte]
                            parameters.append(param_name)
                        elif param_child.type == 'typed_parameter':
                            # Handle typed parameters: name: type
                            param_text = content[param_child.start_byte:param_child.end_byte]
                            parameters.append(param_text)
        except Exception:
            pass
        
        return parameters
    
    def _extract_js_properties(self, node: Node, content: str) -> Dict[str, Any]:
        """Extract JavaScript/TypeScript-specific properties."""
        properties = {}
        
        # Check for arrow functions
        if node.type == 'arrow_function':
            properties['is_arrow_function'] = True
        
        # Check for async functions
        node_text = content[node.start_byte:node.end_byte]
        if 'async' in node_text:
            properties['is_async'] = True
        
        return properties
    
    def _extract_java_properties(self, node: Node, content: str) -> Dict[str, Any]:
        """Extract Java-specific properties."""
        properties = {}
        
        # Extract modifiers (public, private, static, etc.)
        modifiers = []
        node_text = content[node.start_byte:node.end_byte]
        
        java_modifiers = ['public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized']
        for modifier in java_modifiers:
            if modifier in node_text:
                modifiers.append(modifier)
        
        if modifiers:
            properties['modifiers'] = modifiers
        
        return properties
    
    def _extract_c_properties(self, node: Node, content: str) -> Dict[str, Any]:
        """Extract C/C++-specific properties."""
        properties = {}
        
        # Check for static/extern keywords
        node_text = content[node.start_byte:node.end_byte]
        if 'static' in node_text:
            properties['is_static'] = True
        if 'extern' in node_text:
            properties['is_extern'] = True
        
        return properties
    
    def _extract_rust_properties(self, node: Node, content: str) -> Dict[str, Any]:
        """Extract Rust-specific properties."""
        properties = {}
        
        # Check for pub keyword
        node_text = content[node.start_byte:node.end_byte]
        if 'pub' in node_text:
            properties['is_public'] = True
        
        return properties
    
    def _extract_go_properties(self, node: Node, content: str) -> Dict[str, Any]:
        """Extract Go-specific properties."""
        properties = {}
        
        # Check for capitalized names (public in Go)
        symbol_name = self._get_symbol_name(node, content)
        if symbol_name and symbol_name[0].isupper():
            properties['is_exported'] = True
        
        return properties
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the symbol collection process."""
        return {
            **self.collection_stats,
            'files_in_symbol_table': len(self.file_symbols),
            'average_symbols_per_file': (
                self.collection_stats['symbols_discovered'] / 
                max(self.collection_stats['files_processed'], 1)
            ),
            'symbol_table_size': len(self.symbol_table)
        }
    
    def get_symbols_by_file(self, file_path: str) -> Set[str]:
        """Get all symbol IDs discovered in a specific file."""
        return self.file_symbols.get(file_path, set())
    
    def get_symbols_by_type(self, symbol_type: str) -> List[Dict]:
        """Get all symbols of a specific type."""
        return [
            symbol for symbol in self.symbol_table.values()
            if symbol['type'] == symbol_type
        ]
    
    def _fallback_symbol_collection(self, file_paths: List[Path]) -> Dict[str, Dict]:
        """
        Fallback symbol collection when tree-sitter is not available.
        Uses regex-based parsing for basic symbol extraction.
        """
        logger.info("Using fallback symbol collection (regex-based)")
        
        # Basic regex patterns for common programming constructs
        patterns = {
            'python_function': re.compile(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE),
            'python_class': re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE),
            'js_function': re.compile(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE),
            'js_const': re.compile(r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=', re.MULTILINE),
            'java_class': re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE),
            'java_method': re.compile(r'(public|private|protected)?\s*(static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE)
        }
        
        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_ext = file_path.suffix.lower()
                
                # Select appropriate patterns based on file extension
                active_patterns = []
                if file_ext == '.py':
                    active_patterns = ['python_function', 'python_class']
                elif file_ext in ['.js', '.ts']:
                    active_patterns = ['js_function', 'js_const']
                elif file_ext == '.java':
                    active_patterns = ['java_class', 'java_method']
                else:
                    # Try all patterns for unknown file types
                    active_patterns = list(patterns.keys())
                
                file_symbols = set()
                
                for pattern_name in active_patterns:
                    pattern = patterns[pattern_name]
                    for match in pattern.finditer(content):
                        symbol_name = match.group(1) if len(match.groups()) >= 1 else match.group(3) if len(match.groups()) >= 3 else None
                        if symbol_name:
                            # Create basic symbol info
                            symbol_id = f"{file_path.stem}::{symbol_name}::{match.start()}"
                            symbol_type = self._map_pattern_to_symbol_type(pattern_name)
                            
                            symbol_info = {
                                'id': symbol_id,
                                'name': symbol_name,
                                'type': symbol_type,
                                'file_path': str(file_path),
                                'line_start': content[:match.start()].count('\n') + 1,
                                'line_end': content[:match.end()].count('\n') + 1,
                                'signature': match.group(0).strip(),
                                'docstring': None,
                                'properties': {'extraction_method': 'regex_fallback'}
                            }
                            
                            self.symbol_table[symbol_id] = symbol_info
                            file_symbols.add(symbol_id)
                            
                            # Insert into database
                            self.db_manager.insert_node(
                                node_id=symbol_id,
                                node_type=symbol_type,
                                name=symbol_name,
                                file_path=str(file_path),
                                line_start=symbol_info['line_start'],
                                line_end=symbol_info['line_end'],
                                signature=symbol_info['signature'],
                                docstring=symbol_info['docstring'],
                                properties=symbol_info['properties']
                            )
                
                self.file_symbols[str(file_path)] = file_symbols
                self.collection_stats['files_processed'] += 1
                self.collection_stats['symbols_discovered'] += len(file_symbols)
                
                logger.debug(f"Fallback extraction found {len(file_symbols)} symbols in {file_path}")
                
            except Exception as e:
                logger.error(f"Fallback symbol collection failed for {file_path}: {e}")
                self.collection_stats['files_failed'] += 1
                self.collection_stats['processing_errors'].append(str(e))
        
        logger.info(f"Fallback symbol collection completed: {self.collection_stats['symbols_discovered']} symbols from {self.collection_stats['files_processed']} files")
        return self.symbol_table
    
    def _map_pattern_to_symbol_type(self, pattern_name: str) -> str:
        """Map regex pattern names to symbol types."""
        mapping = {
            'python_function': 'function',
            'python_class': 'class',
            'js_function': 'function',
            'js_const': 'variable',
            'java_class': 'class',
            'java_method': 'method'
        }
        return mapping.get(pattern_name, 'unknown')


if __name__ == "__main__":
    # CLI interface for testing symbol collection
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Symbol Collector")
    parser.add_argument("--path", required=True, help="Path to analyze")
    parser.add_argument("--db-path", default="test_symbols.db", help="Database file path")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from storage.database_setup import setup_codewise_database
        
        # Set up database
        setup = setup_codewise_database(args.db_path)
        db_manager = DatabaseManager(args.db_path)
        
        # Create symbol collector
        collector = SymbolCollector(db_manager)
        
        # Collect symbols
        target_path = Path(args.path)
        if target_path.is_file():
            file_paths = [target_path]
        else:
            # Find all code files
            extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.rs', '.go'}
            file_paths = [f for f in target_path.rglob("*") if f.suffix in extensions]
        
        symbol_table = collector.collect_all_symbols(file_paths)
        
        print(f"\\nSymbol collection completed:")
        print(f"Files processed: {len(file_paths)}")
        print(f"Symbols discovered: {len(symbol_table)}")
        
        if args.stats:
            stats = collector.get_collection_statistics()
            print(f"\\nDetailed Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # Show some example symbols
        print(f"\\nExample symbols:")
        for i, (symbol_id, symbol_info) in enumerate(symbol_table.items()):
            if i >= 5:  # Show first 5
                break
            print(f"  {symbol_info['type']}: {symbol_info['name']} in {symbol_info['file_path']}")
        
        db_manager.close()
        
    except Exception as e:
        print(f"Symbol collection failed: {e}")
        import traceback
        traceback.print_exc()