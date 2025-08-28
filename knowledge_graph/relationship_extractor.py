"""
Relationship Extractor - Pass 2 of AST-to-Graph Pipeline

Extracts relationships between symbols using the symbol table from Pass 1.
Builds the edges in the Knowledge Graph by analyzing function calls, 
imports, inheritance, and other code relationships.

Architectural Pattern: Two-pass compiler design ensures accurate symbol
resolution by having complete symbol table before relationship extraction.
"""

from typing import Dict, Set, List, Optional, Any, Tuple
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
from storage.path_manager import get_path_manager

logger = logging.getLogger(__name__)


class ImportResolver:
    """
    Resolves import statements to actual symbol references.
    
    Handles complex import patterns like aliasing, wildcard imports,
    and cross-module references across different programming languages.
    """
    
    def __init__(self, symbol_table: Dict[str, Dict]):
        self.symbol_table = symbol_table
        self.import_cache: Dict[str, Dict[str, str]] = {}  # file_path -> {alias: full_symbol_id}
    
    def resolve_import_aliases(self, file_path: str, content: str, tree: Tree) -> Dict[str, str]:
        """
        Build import alias mapping for a file.
        
        Args:
            file_path: Path to the file being analyzed
            content: File content
            tree: Parsed AST tree
            
        Returns:
            Dictionary mapping local names to full symbol IDs
        """
        if file_path in self.import_cache:
            return self.import_cache[file_path]
        
        aliases = {}
        self._extract_imports_from_tree(tree.root_node, content, aliases)
        
        self.import_cache[file_path] = aliases
        return aliases
    
    def _extract_imports_from_tree(self, node: Node, content: str, aliases: Dict[str, str]):
        """Recursively extract import statements and build alias mappings."""
        
        import_node_types = {
            'import_statement',          # Python: import module
            'import_from_statement',     # Python: from module import symbol
            'import_declaration',        # JavaScript: import ... from
            'import_static_statement',   # Java: import static
            'use_declaration',           # Rust: use module::symbol
            'using_declaration',         # C++: using namespace
        }
        
        if node.type in import_node_types:
            self._extract_import_aliases(node, content, aliases)
        
        # Recurse into children
        for child in node.children:
            self._extract_imports_from_tree(child, content, aliases)
    
    def _extract_import_aliases(self, node: Node, content: str, aliases: Dict[str, str]):
        """Extract alias mappings from a specific import node."""
        try:
            import_text = content[node.start_byte:node.end_byte]
            
            if node.type == 'import_statement':
                # Python: import pandas as pd
                if ' as ' in import_text:
                    match = re.search(r'import\\s+([\\w\\.]+)\\s+as\\s+(\\w+)', import_text)
                    if match:
                        module_name, alias = match.groups()
                        aliases[alias] = self._find_symbol_by_module(module_name)
                else:
                    # Python: import pandas
                    match = re.search(r'import\\s+([\\w\\.]+)', import_text)
                    if match:
                        module_name = match.group(1)
                        # Direct module name mapping
                        aliases[module_name.split('.')[-1]] = self._find_symbol_by_module(module_name)
            
            elif node.type == 'import_from_statement':
                # Python: from pandas import DataFrame as DF
                # from module import symbol1, symbol2
                module_match = re.search(r'from\\s+([\\w\\.]+)\\s+import', import_text)
                if module_match:
                    module_name = module_match.group(1)
                    
                    # Extract imported symbols
                    import_part = import_text.split('import')[1].strip()
                    symbols = [s.strip() for s in import_part.split(',')]
                    
                    for symbol in symbols:
                        if ' as ' in symbol:
                            original, alias = symbol.split(' as ')
                            aliases[alias.strip()] = self._find_symbol_in_module(module_name, original.strip())
                        else:
                            symbol_name = symbol.strip()
                            aliases[symbol_name] = self._find_symbol_in_module(module_name, symbol_name)
            
            elif node.type == 'import_declaration':
                # JavaScript/TypeScript: import { func as f } from 'module'
                if 'from' in import_text:
                    # Extract module name
                    module_match = re.search(r"from\\s+['\"]([^'\"]+)['\"]", import_text)
                    if module_match:
                        module_name = module_match.group(1)
                        
                        # Extract imported symbols
                        if '{' in import_text and '}' in import_text:
                            import_section = import_text[import_text.find('{')+1:import_text.find('}')]
                            symbols = [s.strip() for s in import_section.split(',')]
                            
                            for symbol in symbols:
                                if ' as ' in symbol:
                                    original, alias = symbol.split(' as ')
                                    aliases[alias.strip()] = self._find_symbol_in_module(module_name, original.strip())
                                else:
                                    symbol_name = symbol.strip()
                                    aliases[symbol_name] = self._find_symbol_in_module(module_name, symbol_name)
                        
        except Exception as e:
            logger.debug(f"Failed to extract import aliases: {e}")
    
    def _find_symbol_by_module(self, module_name: str) -> Optional[str]:
        """Find a symbol ID for a given module name."""
        # Look for modules or files matching this name
        for symbol_id, symbol_info in self.symbol_table.items():
            if (symbol_info['type'] == 'module' or 
                symbol_info['file_path'].endswith(f"{module_name}.py") or
                symbol_info['file_path'].endswith(f"{module_name}.js")):
                return symbol_id
        return None
    
    def _find_symbol_in_module(self, module_name: str, symbol_name: str) -> Optional[str]:
        """Find a specific symbol within a module."""
        # Look for symbols with matching name in files matching module
        for symbol_id, symbol_info in self.symbol_table.items():
            if (symbol_info['name'] == symbol_name and
                (module_name in symbol_info['file_path'] or
                 symbol_info['file_path'].endswith(f"{module_name}.py") or
                 symbol_info['file_path'].endswith(f"{module_name}.js"))):
                return symbol_id
        return None


class RelationshipExtractor:
    """
    Pass 2: Extracts relationships between symbols using the symbol table.
    
    Core Responsibility: Build the edges in the Knowledge Graph by analyzing
    function calls, method calls, inheritance, imports, and containment relationships.
    
    Design Decision: Relationship extraction requires complete symbol table to
    accurately resolve cross-file references and aliased symbols.
    """
    
    def __init__(self, db_manager: DatabaseManager, symbol_table: Dict[str, Dict], 
                 project_name: Optional[str] = None):
        self.db_manager = db_manager
        self.symbol_table = symbol_table
        self.project_name = project_name
        self.path_manager = get_path_manager()
        self.parser_factory = TreeSitterFactory() if TREE_SITTER_FACTORY_AVAILABLE else None
        self.import_resolver = ImportResolver(symbol_table)
        
        # Log PathManager integration
        logger.info(f"RelationshipExtractor initialized with PathManager (project: {project_name})")
        
        # Extraction statistics
        self.extraction_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'relationships_found': 0,
            'relationships_by_type': {},
            'processing_errors': []
        }
    
    def extract_relationships(self, file_paths: List[Path]):
        """
        Extract all relationships across the codebase.
        
        Args:
            file_paths: List of file paths to process for relationships
        """
        logger.info(f"Extracting relationships from {len(file_paths)} files")
        
        # Reset extraction stats
        self.extraction_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'relationships_found': 0,
            'relationships_by_type': {},
            'processing_errors': []
        }
        
        # Check if tree-sitter is available
        if not TREE_SITTER_AVAILABLE or not TREE_SITTER_FACTORY_AVAILABLE:
            logger.warning("Tree-sitter not available. Relationship extraction will use fallback methods.")
            return self._fallback_relationship_extraction(file_paths)
        
        for file_path in file_paths:
            try:
                self._extract_file_relationships(file_path)
                self.extraction_stats['files_processed'] += 1
                
            except Exception as e:
                error_msg = f"Relationship extraction failed for {file_path}: {e}"
                logger.error(error_msg)
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['processing_errors'].append(error_msg)
        
        logger.info(f"Relationship extraction completed:")
        logger.info(f"  Files processed: {self.extraction_stats['files_processed']}")
        logger.info(f"  Files failed: {self.extraction_stats['files_failed']}")
        logger.info(f"  Relationships found: {self.extraction_stats['relationships_found']}")
        
        # Log relationship type distribution
        for rel_type, count in self.extraction_stats['relationships_by_type'].items():
            logger.info(f"    {rel_type}: {count}")
    
    def _extract_file_relationships(self, file_path: Path):
        """Extract relationships from a single file."""
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
        
        tree = self.parser_factory.parse_content(content, file_path)
        if not tree:
            logger.warning(f"Could not parse {file_path}")
            return
        
        # Build import context for this file
        import_aliases = self.import_resolver.resolve_import_aliases(str(file_path), content, tree)
        
        # Track local symbols for containment relationships
        local_symbols = self._get_local_symbols(file_path)
        
        # Find relationships in AST
        self._traverse_for_relationships(
            tree.root_node, content, file_path, import_aliases, local_symbols
        )
    
    def _get_local_symbols(self, file_path: Path) -> Dict[str, str]:
        """Get all symbols defined in this file."""
        local_symbols = {}
        file_path_str = str(file_path)
        
        for symbol_id, symbol_info in self.symbol_table.items():
            if symbol_info['file_path'] == file_path_str:
                local_symbols[symbol_info['name']] = symbol_id
        
        return local_symbols
    
    def _traverse_for_relationships(self, node: Node, content: str, file_path: Path,
                                  import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Recursively traverse AST to find relationship patterns."""
        
        # Relationship extraction patterns mapped to handler functions
        relationship_patterns = {
            # Function and method calls
            'call_expression': self._handle_function_call,
            'call': self._handle_function_call,
            
            # Attribute access (method calls, property access)
            'attribute': self._handle_attribute_access,
            'member_expression': self._handle_attribute_access,
            'field_expression': self._handle_attribute_access,
            
            # Class inheritance
            'class_definition': self._handle_class_inheritance,
            'class_declaration': self._handle_class_inheritance,
            
            # Import relationships
            'import_statement': self._handle_import_relationship,
            'import_from_statement': self._handle_import_relationship,
            'import_declaration': self._handle_import_relationship,
            
            # Variable assignments and references
            'assignment': self._handle_variable_assignment,
            'assignment_expression': self._handle_variable_assignment,
            
            # Function definitions (for containment)
            'function_definition': self._handle_containment,
            'method_definition': self._handle_containment,
            'class_definition': self._handle_containment,
        }
        
        handler = relationship_patterns.get(node.type)
        if handler:
            try:
                handler(node, content, file_path, import_aliases, local_symbols)
            except Exception as e:
                logger.debug(f"Relationship handler failed for {node.type}: {e}")
        
        # Recurse into children
        for child in node.children:
            self._traverse_for_relationships(child, content, file_path, import_aliases, local_symbols)
    
    def _handle_function_call(self, node: Node, content: str, file_path: Path,
                            import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Handle function call relationships."""
        try:
            # Extract the function being called
            called_function = self._extract_called_function_name(node, content)
            if not called_function:
                return
            
            # Find the calling context (which function/method contains this call)
            calling_context = self._find_calling_context(node, content, local_symbols)
            if not calling_context:
                return
            
            # Resolve the called function to a symbol ID
            target_symbol_id = self._resolve_symbol_reference(
                called_function, file_path, import_aliases, local_symbols
            )
            
            if target_symbol_id:
                # Create 'calls' relationship
                success = self.db_manager.insert_edge(
                    source_id=calling_context,
                    target_id=target_symbol_id,
                    edge_type='calls',
                    file_path=str(file_path),
                    line_number=node.start_point[0] + 1
                )
                
                if success:
                    self._update_relationship_stats('calls')
                    logger.debug(f"Added call relationship: {calling_context} -> {target_symbol_id}")
        
        except Exception as e:
            logger.debug(f"Failed to handle function call: {e}")
    
    def _extract_called_function_name(self, node: Node, content: str) -> Optional[str]:
        """Extract the name of the function being called."""
        try:
            # For call_expression: func(), obj.method(), module.func()
            
            # Look for the function identifier
            for child in node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
                elif child.type in ('attribute', 'member_expression', 'field_expression'):
                    # Handle method calls: obj.method()
                    return self._extract_attribute_name(child, content)
            
            # Fallback: extract from the first part of the call
            call_text = content[node.start_byte:node.end_byte]
            
            # Remove arguments to get just the function name
            paren_index = call_text.find('(')
            if paren_index != -1:
                func_part = call_text[:paren_index].strip()
                
                # Handle dotted names: obj.method -> method
                if '.' in func_part:
                    return func_part.split('.')[-1]
                else:
                    return func_part
            
        except Exception as e:
            logger.debug(f"Failed to extract called function name: {e}")
        
        return None
    
    def _extract_attribute_name(self, node: Node, content: str) -> Optional[str]:
        """Extract attribute/method name from attribute access."""
        try:
            # Look for the attribute field
            attr_node = node.child_by_field_name('attribute')
            if attr_node:
                return content[attr_node.start_byte:attr_node.end_byte]
            
            # Fallback: look for identifier nodes
            for child in node.children:
                if child.type == 'identifier':
                    return content[child.start_byte:child.end_byte]
        
        except Exception as e:
            logger.debug(f"Failed to extract attribute name: {e}")
        
        return None
    
    def _find_calling_context(self, node: Node, content: str, 
                            local_symbols: Dict[str, str]) -> Optional[str]:
        """Find which function/method contains this call."""
        try:
            # Walk up the AST to find the containing function/method
            current = node.parent
            
            while current:
                if current.type in ('function_definition', 'async_function_definition', 
                                  'method_definition', 'function_declaration'):
                    # Extract the function name
                    func_name = self._get_symbol_name_from_node(current, content)
                    if func_name and func_name in local_symbols:
                        return local_symbols[func_name]
                
                current = current.parent
            
        except Exception as e:
            logger.debug(f"Failed to find calling context: {e}")
        
        return None
    
    def _get_symbol_name_from_node(self, node: Node, content: str) -> Optional[str]:
        """Extract symbol name from a definition node."""
        # Look for name field
        name_node = node.child_by_field_name('name')
        if name_node:
            return content[name_node.start_byte:name_node.end_byte]
        
        # Fallback: look for identifier
        for child in node.children:
            if child.type == 'identifier':
                return content[child.start_byte:child.end_byte]
        
        return None
    
    def _handle_attribute_access(self, node: Node, content: str, file_path: Path,
                               import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Handle attribute access patterns (may indicate method calls or property access)."""
        try:
            # Extract object and attribute
            obj_name = None
            attr_name = None
            
            # Get object being accessed
            obj_node = node.child_by_field_name('object')
            if obj_node and obj_node.type == 'identifier':
                obj_name = content[obj_node.start_byte:obj_node.end_byte]
            
            # Get attribute being accessed
            attr_name = self._extract_attribute_name(node, content)
            
            if obj_name and attr_name:
                # Try to resolve the object type and create relationship
                obj_symbol_id = self._resolve_symbol_reference(
                    obj_name, file_path, import_aliases, local_symbols
                )
                
                if obj_symbol_id:
                    # This could be a method call or property access
                    # Create a 'uses' relationship
                    calling_context = self._find_calling_context(node, content, local_symbols)
                    if calling_context:
                        success = self.db_manager.insert_edge(
                            source_id=calling_context,
                            target_id=obj_symbol_id,
                            edge_type='uses',
                            file_path=str(file_path),
                            line_number=node.start_point[0] + 1,
                            properties={'attribute': attr_name}
                        )
                        
                        if success:
                            self._update_relationship_stats('uses')
        
        except Exception as e:
            logger.debug(f"Failed to handle attribute access: {e}")
    
    def _handle_class_inheritance(self, node: Node, content: str, file_path: Path,
                                import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Handle class inheritance relationships."""
        try:
            # Extract class name
            class_name = self._get_symbol_name_from_node(node, content)
            if not class_name or class_name not in local_symbols:
                return
            
            class_symbol_id = local_symbols[class_name]
            
            # Look for inheritance (superclasses, base classes)
            superclasses = self._extract_superclasses(node, content)
            
            for superclass in superclasses:
                # Resolve superclass to symbol ID
                parent_symbol_id = self._resolve_symbol_reference(
                    superclass, file_path, import_aliases, local_symbols
                )
                
                if parent_symbol_id:
                    # Create 'inherits' relationship
                    success = self.db_manager.insert_edge(
                        source_id=class_symbol_id,
                        target_id=parent_symbol_id,
                        edge_type='inherits',
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1
                    )
                    
                    if success:
                        self._update_relationship_stats('inherits')
                        logger.debug(f"Added inheritance: {class_symbol_id} -> {parent_symbol_id}")
        
        except Exception as e:
            logger.debug(f"Failed to handle class inheritance: {e}")
    
    def _extract_superclasses(self, node: Node, content: str) -> List[str]:
        """Extract superclass names from class definition."""
        superclasses = []
        
        try:
            # Look for argument_list or superclasses field
            for child in node.children:
                if child.type in ('argument_list', 'superclasses'):
                    for grandchild in child.children:
                        if grandchild.type == 'identifier':
                            superclass = content[grandchild.start_byte:grandchild.end_byte]
                            superclasses.append(superclass)
                        elif grandchild.type in ('attribute', 'member_expression'):
                            # Handle module.Class inheritance
                            attr_name = self._extract_attribute_name(grandchild, content)
                            if attr_name:
                                superclasses.append(attr_name)
        
        except Exception as e:
            logger.debug(f"Failed to extract superclasses: {e}")
        
        return superclasses
    
    def _handle_import_relationship(self, node: Node, content: str, file_path: Path,
                                  import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Handle import relationships between modules."""
        try:
            import_text = content[node.start_byte:node.end_byte]
            
            # Extract imported modules/symbols
            imported_items = self._extract_imported_items(import_text, node.type)
            
            for imported_item in imported_items:
                # Find the importing symbol (usually file-level)
                file_symbol_id = self._get_file_module_symbol(file_path)
                
                # Find the imported symbol
                imported_symbol_id = self._resolve_import_target(imported_item)
                
                if file_symbol_id and imported_symbol_id:
                    # Create 'imports' relationship
                    success = self.db_manager.insert_edge(
                        source_id=file_symbol_id,
                        target_id=imported_symbol_id,
                        edge_type='imports',
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        properties={'import_text': import_text}
                    )
                    
                    if success:
                        self._update_relationship_stats('imports')
        
        except Exception as e:
            logger.debug(f"Failed to handle import relationship: {e}")
    
    def _extract_imported_items(self, import_text: str, import_type: str) -> List[str]:
        """Extract the names of items being imported."""
        imported_items = []
        
        try:
            if import_type == 'import_statement':
                # Python: import module1, module2
                match = re.search(r'import\\s+(.+)', import_text)
                if match:
                    modules = [m.strip().split(' as ')[0] for m in match.group(1).split(',')]
                    imported_items.extend(modules)
            
            elif import_type == 'import_from_statement':
                # Python: from module import item1, item2
                match = re.search(r'from\\s+([\\w\\.]+)\\s+import\\s+(.+)', import_text)
                if match:
                    module, items = match.groups()
                    item_names = [item.strip().split(' as ')[0] for item in items.split(',')]
                    imported_items.extend(item_names)
            
            elif import_type == 'import_declaration':
                # JavaScript: import { item1, item2 } from 'module'
                if 'from' in import_text:
                    module_match = re.search(r"from\\s+['\"]([^'\"]+)['\"]", import_text)
                    if module_match:
                        imported_items.append(module_match.group(1))
        
        except Exception as e:
            logger.debug(f"Failed to extract imported items: {e}")
        
        return imported_items
    
    def _get_file_module_symbol(self, file_path: Path) -> Optional[str]:
        """Get the symbol ID for the file/module itself."""
        file_path_str = str(file_path)
        
        # Look for a module symbol for this file
        for symbol_id, symbol_info in self.symbol_table.items():
            if (symbol_info['file_path'] == file_path_str and 
                symbol_info['type'] in ('module', 'file')):
                return symbol_id
        
        # Create a synthetic module symbol if none exists
        module_name = file_path.stem
        module_symbol_id = f"{module_name}::module::0"
        
        success = self.db_manager.insert_node(
            node_id=module_symbol_id,
            node_type='module',
            name=module_name,
            file_path=self._normalize_file_path_for_storage(file_path_str),
            line_start=1,
            line_end=1,
            signature=f"module {module_name}"
        )
        
        if success:
            # Add to symbol table
            self.symbol_table[module_symbol_id] = {
                'id': module_symbol_id,
                'name': module_name,
                'type': 'module',
                'file_path': file_path_str,
                'line_start': 1,
                'line_end': 1
            }
            return module_symbol_id
        
        return None
    
    def _resolve_import_target(self, imported_item: str) -> Optional[str]:
        """Resolve an imported item to its symbol ID."""
        # Look for symbols matching the imported name
        candidates = []
        
        for symbol_id, symbol_info in self.symbol_table.items():
            symbol_name = symbol_info['name']
            
            # Exact name match
            if symbol_name == imported_item:
                candidates.append(symbol_id)
            
            # Module path match (for module imports)
            elif (symbol_info['type'] == 'module' and 
                  imported_item in symbol_info['file_path']):
                candidates.append(symbol_id)
        
        # Return first candidate (could be improved with better resolution logic)
        return candidates[0] if candidates else None
    
    def _handle_variable_assignment(self, node: Node, content: str, file_path: Path,
                                  import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Handle variable assignment relationships."""
        try:
            # This could create 'defines' relationships
            # For now, we'll focus on the other relationship types
            pass
        
        except Exception as e:
            logger.debug(f"Failed to handle variable assignment: {e}")
    
    def _handle_containment(self, node: Node, content: str, file_path: Path,
                          import_aliases: Dict[str, str], local_symbols: Dict[str, str]):
        """Handle containment relationships (functions in classes, etc.)."""
        try:
            # Extract the containing symbol
            container_name = self._get_symbol_name_from_node(node, content)
            if not container_name or container_name not in local_symbols:
                return
            
            container_symbol_id = local_symbols[container_name]
            
            # Find symbols contained within this one
            contained_symbols = self._find_contained_symbols(node, content, local_symbols)
            
            for contained_symbol_id in contained_symbols:
                # Create 'contains' relationship
                success = self.db_manager.insert_edge(
                    source_id=container_symbol_id,
                    target_id=contained_symbol_id,
                    edge_type='contains',
                    file_path=str(file_path),
                    line_number=node.start_point[0] + 1
                )
                
                if success:
                    self._update_relationship_stats('contains')
        
        except Exception as e:
            logger.debug(f"Failed to handle containment: {e}")
    
    def _find_contained_symbols(self, container_node: Node, content: str,
                              local_symbols: Dict[str, str]) -> List[str]:
        """Find symbols contained within a container node."""
        contained = []
        
        # Look for function/method definitions within this container
        def find_nested_definitions(node):
            if node.type in ('function_definition', 'method_definition'):
                symbol_name = self._get_symbol_name_from_node(node, content)
                if symbol_name and symbol_name in local_symbols:
                    contained.append(local_symbols[symbol_name])
            
            for child in node.children:
                find_nested_definitions(child)
        
        find_nested_definitions(container_node)
        return contained
    
    def _resolve_symbol_reference(self, symbol_name: str, file_path: Path,
                                import_aliases: Dict[str, str], 
                                local_symbols: Dict[str, str]) -> Optional[str]:
        """
        Resolve a symbol name to its full symbol ID.
        
        Resolution order:
        1. Import aliases (highest priority)
        2. Local symbols in same file
        3. Global symbol search
        """
        
        # 1. Check import aliases first
        if symbol_name in import_aliases:
            return import_aliases[symbol_name]
        
        # 2. Check local symbols
        if symbol_name in local_symbols:
            return local_symbols[symbol_name]
        
        # 3. Global search in symbol table
        candidates = []
        for symbol_id, symbol_info in self.symbol_table.items():
            if symbol_info['name'] == symbol_name:
                candidates.append(symbol_id)
        
        # Return first candidate (could be improved with more sophisticated resolution)
        return candidates[0] if candidates else None
    
    def _update_relationship_stats(self, relationship_type: str):
        """Update relationship extraction statistics."""
        self.extraction_stats['relationships_found'] += 1
        self.extraction_stats['relationships_by_type'][relationship_type] = \
            self.extraction_stats['relationships_by_type'].get(relationship_type, 0) + 1
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about relationship extraction."""
        return {
            **self.extraction_stats,
            'total_symbols_in_table': len(self.symbol_table),
            'average_relationships_per_file': (
                self.extraction_stats['relationships_found'] /
                max(self.extraction_stats['files_processed'], 1)
            )
        }
    
    def _fallback_relationship_extraction(self, file_paths: List[Path]):
        """
        Fallback relationship extraction when tree-sitter is not available.
        Uses regex-based parsing for basic relationship detection.
        """
        logger.info("Using fallback relationship extraction (regex-based)")
        
        # Basic regex patterns for relationships
        patterns = {
            'function_call': re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE),
            'import_from': re.compile(r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import', re.MULTILINE),
            'import_simple': re.compile(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', re.MULTILINE),
            'class_inheritance': re.compile(r'class\s+\w+\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE),
            'method_call': re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', re.MULTILINE)
        }
        
        for file_path in file_paths:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Find potential relationships using regex
                relationships_found = 0
                
                # Process each pattern type
                for pattern_name, pattern in patterns.items():
                    for match in pattern.finditer(content):
                        target_name = match.group(1)
                        
                        # Look for matching symbols in our symbol table
                        matching_symbols = [
                            symbol_id for symbol_id, symbol_info in self.symbol_table.items()
                            if symbol_info['name'] == target_name
                        ]
                        
                        if matching_symbols:
                            # Find source symbols in the same file
                            file_symbols = [
                                symbol_id for symbol_id, symbol_info in self.symbol_table.items()
                                if symbol_info['file_path'] == str(file_path)
                            ]
                            
                            # Create relationships from file symbols to target
                            for source_id in file_symbols:
                                for target_id in matching_symbols:
                                    if source_id != target_id:  # Avoid self-references
                                        relationship_type = self._map_pattern_to_relationship_type(pattern_name)
                                        
                                        # Insert relationship into database
                                        success = self.db_manager.insert_edge(
                                            source_id=source_id,
                                            target_id=target_id,
                                            edge_type=relationship_type,
                                            file_path=str(file_path),
                                            line_number=content[:match.start()].count('\n') + 1,
                                            properties={'extraction_method': 'regex_fallback'}
                                        )
                                        
                                        if success:
                                            relationships_found += 1
                                            self.extraction_stats['relationships_found'] += 1
                                            
                                            # Update stats by type
                                            if relationship_type not in self.extraction_stats['relationships_by_type']:
                                                self.extraction_stats['relationships_by_type'][relationship_type] = 0
                                            self.extraction_stats['relationships_by_type'][relationship_type] += 1
                
                self.extraction_stats['files_processed'] += 1
                logger.debug(f"Fallback extraction found {relationships_found} relationships in {file_path}")
                
            except Exception as e:
                logger.error(f"Fallback relationship extraction failed for {file_path}: {e}")
                self.extraction_stats['files_failed'] += 1
                self.extraction_stats['processing_errors'].append(str(e))
        
        logger.info(f"Fallback relationship extraction completed: {self.extraction_stats['relationships_found']} relationships from {self.extraction_stats['files_processed']} files")
    
    def _map_pattern_to_relationship_type(self, pattern_name: str) -> str:
        """Map regex pattern names to relationship types."""
        mapping = {
            'function_call': 'calls',
            'import_from': 'imports',
            'import_simple': 'imports', 
            'class_inheritance': 'inherits',
            'method_call': 'calls'
        }
        return mapping.get(pattern_name, 'uses')
    
    def _normalize_file_path_for_storage(self, file_path_str: str) -> str:
        """
        Convert file path using centralized path manager for consistent storage.
        
        Args:
            file_path_str: File path string (absolute or relative)
            
        Returns:
            Normalized workspace-relative path with project prefix
        """
        try:
            # Use PathManager with project context for consistent normalization
            normalized = self.path_manager.normalize_for_storage(file_path_str, self.project_name)
            logger.debug(f"Normalized path: {file_path_str} -> {normalized}")
            return normalized
        except Exception as e:
            # Fallback to original logic if normalization fails
            logger.warning(f"Path normalization failed for {file_path_str}: {e}, using fallback")
            if '/workspace/' in file_path_str:
                workspace_relative = file_path_str.split('/workspace/', 1)[-1]
                return workspace_relative
            return file_path_str


if __name__ == "__main__":
    # CLI interface for testing relationship extraction
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Relationship Extractor")
    parser.add_argument("--path", required=True, help="Path to analyze")
    parser.add_argument("--db-path", default="test_relationships.db", help="Database file path")
    parser.add_argument("--stats", action="store_true", help="Show extraction statistics")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from storage.database_setup import setup_codewise_database
        from knowledge_graph.symbol_collector import SymbolCollector
        
        # Set up database
        setup = setup_codewise_database(args.db_path)
        db_manager = DatabaseManager(args.db_path)
        
        # First run symbol collection
        collector = SymbolCollector(db_manager)
        
        target_path = Path(args.path)
        if target_path.is_file():
            file_paths = [target_path]
        else:
            extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.rs', '.go'}
            file_paths = [f for f in target_path.rglob("*") if f.suffix in extensions]
        
        print("Running symbol collection...")
        symbol_table = collector.collect_all_symbols(file_paths)
        
        # Now run relationship extraction
        print("Running relationship extraction...")
        extractor = RelationshipExtractor(db_manager, symbol_table)
        extractor.extract_relationships(file_paths)
        
        if args.stats:
            print("\\nSymbol Collection Statistics:")
            symbol_stats = collector.get_collection_statistics()
            for key, value in symbol_stats.items():
                print(f"  {key}: {value}")
            
            print("\\nRelationship Extraction Statistics:")
            rel_stats = extractor.get_extraction_statistics()
            for key, value in rel_stats.items():
                print(f"  {key}: {value}")
        
        # Show database statistics
        db_stats = db_manager.get_statistics()
        print(f"\\nDatabase Statistics:")
        print(f"  Nodes: {db_stats['nodes_total']}")
        print(f"  Edges: {db_stats['edges_total']}")
        print(f"  Edge types: {db_stats.get('edge_types', {})}")
        
        db_manager.close()
        
    except Exception as e:
        print(f"Relationship extraction failed: {e}")
        import traceback
        traceback.print_exc()