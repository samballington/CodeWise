"""
AST-Based Chunking System for CodeWise Indexer

This module provides intelligent code structure preservation through AST-based chunking
for various programming languages and file types.
"""

import ast
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# JavaScript/TypeScript AST parsing
try:
    import esprima
    ESPRIMA_AVAILABLE = True
except ImportError:
    ESPRIMA_AVAILABLE = False
    logging.warning("esprima not available - JavaScript/TypeScript chunking will use regex fallback")

# Tree-sitter universal parsing
try:
    from indexer.parsers.tree_sitter_parser import TreeSitterFactory
    TREE_SITTER_FACTORY_AVAILABLE = True
except ImportError:
    TREE_SITTER_FACTORY_AVAILABLE = False
    logging.warning("TreeSitterFactory not available - falling back to regex parsing")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Enhanced metadata for code chunks"""
    file_type: str
    chunk_type: str  # 'function', 'class', 'config_section', 'markdown_section', 'module'
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    parent_context: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    text: str
    metadata: ChunkMetadata
    file_path: str
    start_line: int
    end_line: int
    
    def __post_init__(self):
        """Ensure metadata line numbers match chunk line numbers"""
        if self.metadata.line_start == 0:
            self.metadata.line_start = self.start_line
        if self.metadata.line_end == 0:
            self.metadata.line_end = self.end_line


class BaseChunker(ABC):
    """Abstract base class for all chunkers"""
    
    @abstractmethod
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk content based on file type and structure"""
        pass

class PythonASTChunker(BaseChunker):
    """AST-based chunker for Python files"""
    
    def __init__(self):
        self.source_lines: List[str] = []
        self.file_path: str = ""
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk Python content using AST analysis"""
        self.source_lines = content.splitlines()
        self.file_path = str(file_path)
        
        try:
            # Parse the Python code
            tree = ast.parse(content)
            chunks = []
            
            # Extract module-level imports first
            module_imports = self._extract_imports(tree)
            
            # Process each top-level node
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._create_function_chunk(node, module_imports)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_class_chunk(node, module_imports)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Skip imports as they're included in other chunks
                    continue
                else:
                    # Handle other module-level code
                    chunk = self._create_module_level_chunk(node, module_imports)
                    if chunk:
                        chunks.append(chunk)
            
            # If no chunks were created, create a single chunk for the entire file
            if not chunks:
                chunks.append(self._create_fallback_chunk(content, module_imports))
            
            logger.debug(f"Created {len(chunks)} chunks for Python file: {file_path}")
            return chunks
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python file {file_path}: {e}")
            return self._fallback_chunk(content, file_path)
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            return self._fallback_chunk(content, file_path) 
   
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements from the module"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        
        return imports
    
    def _create_function_chunk(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                              module_imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a function definition"""
        try:
            # Get function source
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Extract function text
            function_lines = self.source_lines[start_line-1:end_line]
            function_text = '\n'.join(function_lines)
            
            # Extract decorators
            decorators = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(ast.unparse(decorator))
            
            # Extract docstring
            docstring = None
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                docstring = node.body[0].value.value
            
            # Create metadata
            metadata = ChunkMetadata(
                file_type="python",
                chunk_type="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                function_name=node.name,
                imports=module_imports,
                line_start=start_line,
                line_end=end_line,
                docstring=docstring,
                decorators=decorators
            )
            
            return CodeChunk(
                text=function_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating function chunk: {e}")
            return None   
 
    def _create_class_chunk(self, node: ast.ClassDef, module_imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a class definition"""
        try:
            # Include decorators (if any) so we capture lines like `@dataclass`.
            if node.decorator_list:
                start_line = min(deco.lineno for deco in node.decorator_list)
            else:
                start_line = node.lineno
            end_line = node.end_lineno or start_line
            
            # Extract class text
            class_lines = self.source_lines[start_line-1:end_line]
            class_text = '\n'.join(class_lines)
            
            # Extract base classes
            base_classes = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_classes.append(ast.unparse(base))
            
            # Extract decorators
            decorators = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Attribute):
                    decorators.append(ast.unparse(decorator))
            
            # Extract docstring
            docstring = None
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
                docstring = node.body[0].value.value
            
            # Create metadata
            metadata = ChunkMetadata(
                file_type="python",
                chunk_type="class",
                class_name=node.name,
                imports=module_imports,
                line_start=start_line,
                line_end=end_line,
                dependencies=base_classes,
                docstring=docstring,
                decorators=decorators
            )
            
            return CodeChunk(
                text=class_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating class chunk: {e}")
            return None  
  
    def _create_module_level_chunk(self, node: ast.AST, module_imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for module-level code (constants, etc.)"""
        try:
            start_line = getattr(node, 'lineno', 1)
            end_line = getattr(node, 'end_lineno', start_line)
            
            # Skip single-line statements that are too small
            if end_line - start_line < 2:
                return None
            
            # Extract code text
            code_lines = self.source_lines[start_line-1:end_line]
            code_text = '\n'.join(code_lines)
            
            # Create metadata
            metadata = ChunkMetadata(
                file_type="python",
                chunk_type="module_level",
                imports=module_imports,
                line_start=start_line,
                line_end=end_line
            )
            
            return CodeChunk(
                text=code_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating module-level chunk: {e}")
            return None
    
    def _create_fallback_chunk(self, content: str, module_imports: List[str]) -> CodeChunk:
        """Create a single chunk for the entire file when no specific chunks are found"""
        lines = content.splitlines()
        
        metadata = ChunkMetadata(
            file_type="python",
            chunk_type="module",
            imports=module_imports,
            line_start=1,
            line_end=len(lines)
        )
        
        return CodeChunk(
            text=content,
            metadata=metadata,
            file_path=self.file_path,
            start_line=1,
            end_line=len(lines)
        )
    
    def _fallback_chunk(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback to character-based chunking when AST parsing fails"""
        logger.info(f"Using fallback chunking for {file_path}")
        
        # Simple character-based chunking
        chunk_size = 400
        chunks = []
        
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i+chunk_size]
            
            # Find approximate line numbers
            start_char = i
            end_char = min(i + chunk_size, len(content))
            start_line = content[:start_char].count('\n') + 1
            end_line = content[:end_char].count('\n') + 1
            
            metadata = ChunkMetadata(
                file_type="python",
                chunk_type="fallback",
                line_start=start_line,
                line_end=end_line
            )
            
            chunks.append(CodeChunk(
                text=chunk_text,
                metadata=metadata,
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line
            ))
        
        return chunks

class JavaScriptASTChunker(BaseChunker):
    """AST-based chunker for JavaScript and TypeScript files using esprima"""
    
    def __init__(self):
        self.source_lines: List[str] = []
        self.file_path: str = ""
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript content using AST analysis"""
        self.source_lines = content.splitlines()
        self.file_path = str(file_path)
        
        if not ESPRIMA_AVAILABLE:
            logger.warning(f"esprima not available, falling back to regex parsing for {file_path}")
            return self._regex_fallback(content, file_path)
        
        try:
            # Parse JavaScript/TypeScript using esprima
            # Try different parsing modes for better compatibility
            ast_tree = None
            
            if str(file_path).endswith(('.ts', '.tsx')):
                # For TypeScript, try to strip basic type annotations first
                cleaned_content = self._strip_typescript_syntax(content)
                try:
                    ast_tree = esprima.parseModule(cleaned_content, {'loc': True, 'range': True})
                except:
                    # Fallback to script parsing
                    ast_tree = esprima.parseScript(cleaned_content, {'loc': True, 'range': True})
            else:
                # For JavaScript, try module parsing first (supports import/export)
                try:
                    ast_tree = esprima.parseModule(content, {'loc': True, 'range': True})
                except:
                    # Fallback to script parsing
                    ast_tree = esprima.parseScript(content, {'loc': True, 'range': True})
            
            chunks = []
            
            # Extract imports first
            imports = self._extract_imports_from_ast(ast_tree)
            
            # Process each top-level node
            for node in ast_tree.body:
                try:
                    chunk = self._process_node(node, imports)
                    if chunk:
                        chunks.append(chunk)
                except Exception as node_error:
                    logger.error(f"Error processing node {node.type}: {node_error}")
                    import traceback
                    logger.debug(f"Node processing traceback: {traceback.format_exc()}")
                    continue
            
            # Sort chunks by line number
            chunks.sort(key=lambda x: x.start_line)
            
            # If no chunks found, create fallback
            if not chunks:
                chunks.append(self._create_fallback_chunk(content, imports))
            
            logger.debug(f"Created {len(chunks)} chunks for JS/TS file: {file_path}")
            return chunks
            
        except Exception as e:
            logger.warning(f"AST parsing failed for {file_path}: {e}, falling back to regex")
            logger.debug(f"AST parsing error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return self._regex_fallback(content, file_path)
    
    def _strip_typescript_syntax(self, content: str) -> str:
        """Strip basic TypeScript syntax to make it parseable by esprima"""
        # Remove type annotations from function parameters and return types
        # This is a basic implementation - more complex TS features may still cause issues
        
        # Remove type annotations from function parameters: (param: Type) -> (param)
        content = re.sub(r'(\w+)\s*:\s*[A-Za-z_][A-Za-z0-9_<>\[\]|&\s]*(?=\s*[,)])', r'\1', content)
        
        # Remove return type annotations: ): Type => { -> ) => {
        content = re.sub(r'\)\s*:\s*[A-Za-z_][A-Za-z0-9_<>\[\]|&\s]*(?=\s*[=>{])', ')', content)
        
        # Remove interface declarations (basic)
        content = re.sub(r'interface\s+\w+\s*\{[^}]*\}', '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove type aliases
        content = re.sub(r'type\s+\w+\s*=\s*[^;]+;', '', content)
        
        # Remove generic type parameters: <T> -> ''
        content = re.sub(r'<[A-Za-z_][A-Za-z0-9_,\s]*>', '', content)
        
        # Remove 'as' type assertions: value as Type -> value
        content = re.sub(r'\s+as\s+[A-Za-z_][A-Za-z0-9_<>\[\]|&\s]*', '', content)
        
        # Remove optional property markers: prop?: -> prop:
        content = re.sub(r'(\w+)\?:', r'\1:', content)
        
        return content
    
    def _extract_imports_from_ast(self, ast_tree) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        
        try:
            for node in ast_tree.body:
                if hasattr(node, 'type'):
                    if node.type == 'ImportDeclaration':
                        # Handle ES6 imports: import { x } from 'module'
                        source = getattr(node.source, 'value', '')
                        specifiers = []
                        for spec in node.specifiers:
                            if hasattr(spec, 'imported') and spec.imported:
                                specifiers.append(spec.imported.name)
                            elif hasattr(spec, 'local') and spec.local:
                                specifiers.append(spec.local.name)
                        
                        if specifiers:
                            imports.append(f"import {{ {', '.join(specifiers)} }} from '{source}'")
                        else:
                            imports.append(f"import '{source}'")
                    
                    elif node.type == 'VariableDeclaration':
                        # Handle CommonJS requires: const x = require('module')
                        for declarator in node.declarations:
                            if (hasattr(declarator, 'init') and declarator.init and
                                hasattr(declarator.init, 'type') and 
                                declarator.init.type == 'CallExpression' and
                                hasattr(declarator.init, 'callee') and declarator.init.callee and
                                hasattr(declarator.init.callee, 'name') and
                                declarator.init.callee.name == 'require'):
                                
                                if hasattr(declarator, 'id') and declarator.id and hasattr(declarator.id, 'name'):
                                    var_name = declarator.id.name
                                    module_name = declarator.init.arguments[0].value if declarator.init.arguments else ''
                                    imports.append(f"const {var_name} = require('{module_name}')")
        except Exception as e:
            logger.error(f"Error extracting imports: {e}")
            import traceback
            logger.debug(f"Import extraction traceback: {traceback.format_exc()}")
        
        return imports
    
    def _process_node(self, node, imports: List[str]) -> Optional[CodeChunk]:
        """Process an AST node and create appropriate chunk"""
        if not hasattr(node, 'type'):
            return None
        
        if node.type == 'FunctionDeclaration':
            return self._create_function_chunk_from_ast(node, imports, "function")
        elif node.type == 'ClassDeclaration':
            return self._create_class_chunk_from_ast(node, imports)
        elif node.type == 'VariableDeclaration':
            # Check if this is an arrow function assignment
            for declarator in node.declarations:
                if (hasattr(declarator, 'init') and 
                    hasattr(declarator.init, 'type') and 
                    declarator.init.type == 'ArrowFunctionExpression'):
                    return self._create_arrow_function_chunk_from_ast(declarator, imports)
        elif node.type == 'ExportNamedDeclaration' or node.type == 'ExportDefaultDeclaration':
            # Handle exported functions and classes
            if hasattr(node, 'declaration') and node.declaration:
                return self._process_node(node.declaration, imports)
        
        return None
    
    def _create_function_chunk_from_ast(self, node, imports: List[str], chunk_type: str) -> Optional[CodeChunk]:
        """Create a chunk for a function from AST node"""
        try:
            function_name = node.id.name if hasattr(node, 'id') and node.id and hasattr(node.id, 'name') else 'anonymous'
            
            # Get line numbers from AST
            start_line = node.loc.start.line if hasattr(node, 'loc') and node.loc else 1
            end_line = node.loc.end.line if hasattr(node, 'loc') and node.loc else start_line
            
            # Extract function text
            function_lines = self.source_lines[start_line-1:end_line]
            function_text = '\n'.join(function_lines)
            
            # Determine file type
            file_type = "typescript" if self.file_path.endswith(('.ts', '.tsx')) else "javascript"
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type=chunk_type,
                function_name=function_name,
                imports=imports,
                line_start=start_line,
                line_end=end_line
            )
            
            return CodeChunk(
                text=function_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating function chunk from AST: {e}")
            return None
    
    def _create_class_chunk_from_ast(self, node, imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a class from AST node"""
        try:
            class_name = node.id.name if hasattr(node, 'id') and node.id and hasattr(node.id, 'name') else 'anonymous'
            
            # Get line numbers from AST
            start_line = node.loc.start.line if hasattr(node, 'loc') and node.loc else 1
            end_line = node.loc.end.line if hasattr(node, 'loc') and node.loc else start_line
            
            # Extract class text
            class_lines = self.source_lines[start_line-1:end_line]
            class_text = '\n'.join(class_lines)
            
            # Extract base classes
            base_classes = []
            if hasattr(node, 'superClass') and node.superClass:
                if hasattr(node.superClass, 'name'):
                    base_classes.append(node.superClass.name)
            
            # Determine file type
            file_type = "typescript" if self.file_path.endswith(('.ts', '.tsx')) else "javascript"
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type="class",
                class_name=class_name,
                imports=imports,
                line_start=start_line,
                line_end=end_line,
                dependencies=base_classes
            )
            
            return CodeChunk(
                text=class_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating class chunk from AST: {e}")
            return None
    
    def _create_arrow_function_chunk_from_ast(self, declarator, imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for an arrow function from AST node"""
        try:
            function_name = declarator.id.name if hasattr(declarator, 'id') and declarator.id and hasattr(declarator.id, 'name') else 'anonymous'
            
            # Get line numbers from AST
            start_line = declarator.loc.start.line if hasattr(declarator, 'loc') and declarator.loc else 1
            end_line = declarator.loc.end.line if hasattr(declarator, 'loc') and declarator.loc else start_line
            
            # Extract function text
            function_lines = self.source_lines[start_line-1:end_line]
            function_text = '\n'.join(function_lines)
            
            # Determine file type
            file_type = "typescript" if self.file_path.endswith(('.ts', '.tsx')) else "javascript"
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type="arrow_function",
                function_name=function_name,
                imports=imports,
                line_start=start_line,
                line_end=end_line
            )
            
            return CodeChunk(
                text=function_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating arrow function chunk from AST: {e}")
            return None
    
    def _extract_imports(self) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for line in self.source_lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('const ') and 'require(' in stripped or
                stripped.startswith('export ')):
                imports.append(stripped)
        
        return imports
    
    def _find_functions(self, imports: List[str]) -> List[CodeChunk]:
        """Find function declarations"""
        chunks = []
        
        for i, line in enumerate(self.source_lines):
            stripped = line.strip()
            
            # Regular function declarations
            if re.match(r'^\s*function\s+\w+', line):
                match = re.search(r'function\s+(\w+)', line)
                if match:
                    function_name = match.group(1)
                    chunk = self._create_function_chunk(i + 1, function_name, "function", imports)
                    if chunk:
                        chunks.append(chunk)
            
            # Export function declarations
            elif re.match(r'^\s*export\s+function\s+\w+', line):
                match = re.search(r'export\s+function\s+(\w+)', line)
                if match:
                    function_name = match.group(1)
                    chunk = self._create_function_chunk(i + 1, function_name, "function", imports)
                    if chunk:
                        chunks.append(chunk)
            
            # Async function declarations
            elif re.match(r'^\s*async\s+function\s+\w+', line):
                match = re.search(r'async\s+function\s+(\w+)', line)
                if match:
                    function_name = match.group(1)
                    chunk = self._create_function_chunk(i + 1, function_name, "async_function", imports)
                    if chunk:
                        chunks.append(chunk)
            
            # Arrow functions assigned to const
            elif re.match(r'^\s*const\s+\w+\s*=\s*.*=>', line):
                match = re.search(r'const\s+(\w+)\s*=', line)
                if match:
                    function_name = match.group(1)
                    chunk = self._create_function_chunk(i + 1, function_name, "arrow_function", imports)
                    if chunk:
                        chunks.append(chunk)
            
            # Arrow functions inside other constructs (like event handlers)
            elif re.search(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', line):
                match = re.search(r'const\s+(\w+)\s*=', line)
                if match:
                    function_name = match.group(1)
                    chunk = self._create_function_chunk(i + 1, function_name, "arrow_function", imports)
                    if chunk:
                        chunks.append(chunk)
        
        return chunks
    
    def _find_classes(self, imports: List[str]) -> List[CodeChunk]:
        """Find class declarations"""
        chunks = []
        
        for i, line in enumerate(self.source_lines):
            if re.match(r'^\s*class\s+\w+', line) or re.match(r'^\s*export\s+class\s+\w+', line):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    class_name = match.group(1)
                    chunk = self._create_class_chunk(i + 1, class_name, imports)
                    if chunk:
                        chunks.append(chunk)
        
        return chunks
    
    def _create_function_chunk(self, start_line: int, function_name: str, 
                              chunk_type: str, imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a function"""
        try:
            # Find the end of the function by counting braces
            end_line = self._find_block_end(start_line - 1)
            
            if end_line == -1:
                return None
            
            # Extract function text
            function_lines = self.source_lines[start_line-1:end_line]
            function_text = '\n'.join(function_lines)
            
            # Determine file type
            file_type = "typescript" if self.file_path.endswith(('.ts', '.tsx')) else "javascript"
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type=chunk_type,
                function_name=function_name,
                imports=imports,
                line_start=start_line,
                line_end=end_line
            )
            
            return CodeChunk(
                text=function_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating function chunk: {e}")
            return None
    
    def _create_class_chunk(self, start_line: int, class_name: str, imports: List[str]) -> Optional[CodeChunk]:
        """Create a chunk for a class"""
        try:
            end_line = self._find_block_end(start_line - 1)
            
            if end_line == -1:
                return None
            
            # Extract class text
            class_lines = self.source_lines[start_line-1:end_line]
            class_text = '\n'.join(class_lines)
            
            # Determine file type
            file_type = "typescript" if self.file_path.endswith(('.ts', '.tsx')) else "javascript"
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type="class",
                class_name=class_name,
                imports=imports,
                line_start=start_line,
                line_end=end_line
            )
            
            return CodeChunk(
                text=class_text,
                metadata=metadata,
                file_path=self.file_path,
                start_line=start_line,
                end_line=end_line
            )
            
        except Exception as e:
            logger.error(f"Error creating class chunk: {e}")
            return None
    
    def _find_block_end(self, start_line_idx: int) -> int:
        """Heuristic block terminator: balanced braces OR first empty line after they close."""
        brace_count = 0
        found_opening = False
        
        for i in range(start_line_idx, len(self.source_lines)):
            line = self.source_lines[i]
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    
                    if found_opening and brace_count == 0:
                        # End either at the closing brace line OR the following blank line
                        if i + 1 < len(self.source_lines) and self.source_lines[i + 1].strip() == "":
                            return i + 2
                        return i + 1
        
        return -1  # fallback if unbalanced
   
    def _regex_fallback(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback to regex-based parsing when AST parsing fails"""
        logger.info(f"Using regex fallback for {file_path}")
        
        chunks = []
        imports = self._extract_imports()
        
        # Find functions and classes using regex
        chunks.extend(self._find_functions(imports))
        chunks.extend(self._find_classes(imports))
        
        # Sort chunks by line number
        chunks.sort(key=lambda x: x.start_line)
        
        # If no chunks found, create fallback
        if not chunks:
            chunks.append(self._create_fallback_chunk(content, imports))
        
        return chunks
   
    def _create_fallback_chunk(self, content: str, imports: List[str]) -> CodeChunk:
        """Create a single chunk for the entire file"""
        lines = content.splitlines()
        file_type = "typescript" if self.file_path.endswith(('.ts', '.tsx')) else "javascript"
        
        metadata = ChunkMetadata(
            file_type=file_type,
            chunk_type="module",
            imports=imports,
            line_start=1,
            line_end=len(lines)
        )
        
        return CodeChunk(
            text=content,
            metadata=metadata,
            file_path=self.file_path,
            start_line=1,
            end_line=len(lines)
        )
    
    def _fallback_chunk_duplicate2(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback to character-based chunking"""
        logger.info(f"Using fallback chunking for {file_path}")
        
        chunk_size = 400
        chunks = []
        file_type = "typescript" if str(file_path).endswith(('.ts', '.tsx')) else "javascript"
        
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i+chunk_size]
            
            # Calculate line numbers
            start_char = i
            end_char = min(i + chunk_size, len(content))
            start_line = content[:start_char].count('\n') + 1
            end_line = content[:end_char].count('\n') + 1
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type="fallback",
                line_start=start_line,
                line_end=end_line
            )
            
            chunks.append(CodeChunk(
                text=chunk_text,
                metadata=metadata,
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line
            ))
        
        return chunks


class ConfigChunker(BaseChunker):
    """Chunker for configuration files"""
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk configuration files as single chunks"""
        lines = content.splitlines()
        file_type = file_path.suffix[1:] if file_path.suffix else "config"
        
        metadata = ChunkMetadata(
            file_type=file_type,
            chunk_type="config_file",
            line_start=1,
            line_end=len(lines)
        )
        
        return [CodeChunk(
            text=content,
            metadata=metadata,
            file_path=str(file_path),
            start_line=1,
            end_line=len(lines)
        )]


class MarkdownChunker(BaseChunker):
    """Chunker for Markdown files with headline-based sections"""
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk Markdown by headline sections with overlap"""
        lines = content.splitlines()
        chunks = []
        current_section = []
        current_start = 1
        section_title = "Introduction"
        
        for i, line in enumerate(lines):
            # Check if this is a heading
            if line.startswith('#'):
                # Save previous section if it exists
                if current_section:
                    section_text = '\n'.join(current_section)
                    
                    # Add 50-character overlap from next section
                    overlap = ""
                    if i < len(lines):
                        next_lines = lines[i:i+3]  # Get next few lines
                        overlap = '\n'.join(next_lines)[:50]
                        if overlap:
                            section_text += f"\n\n{overlap}..."
                    
                    metadata = ChunkMetadata(
                        file_type="markdown",
                        chunk_type="markdown_section",
                        parent_context=section_title,
                        line_start=current_start,
                        line_end=i
                    )
                    
                    chunks.append(CodeChunk(
                        text=section_text,
                        metadata=metadata,
                        file_path=str(file_path),
                        start_line=current_start,
                        end_line=i
                    ))
                
                # Start new section
                current_section = [line]
                current_start = i + 1
                section_title = line.strip('#').strip()
            else:
                current_section.append(line)
        
        found_headings = any(l.startswith('#') for l in lines)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section)
            metadata = ChunkMetadata(
                file_type="markdown",
                chunk_type="markdown_section" if found_headings else "markdown_file",
                parent_context=section_title if found_headings else None,
                line_start=current_start,
                line_end=len(lines)
            )
            
            chunks.append(CodeChunk(
                text=section_text,
                metadata=metadata,
                file_path=str(file_path),
                start_line=current_start,
                end_line=len(lines)
            ))
        
        # If no sections found, create single chunk
        if not chunks:
            metadata = ChunkMetadata(
                file_type="markdown",
                chunk_type="markdown_file",
                line_start=1,
                line_end=len(lines)
            )
            
            chunks.append(CodeChunk(
                text=content,
                metadata=metadata,
                file_path=str(file_path),
                start_line=1,
                end_line=len(lines)
            ))
        
        return chunks

class SmallFileChunker(BaseChunker):
    """Chunker for small files that should be kept as single chunks"""
    
    def __init__(self, max_size: int = 300):
        self.max_size = max_size
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk small files as single chunks"""
        if len(content) > self.max_size:
            return []  # Not a small file, should use other chunkers
        
        lines = content.splitlines()
        file_type = file_path.suffix[1:] if file_path.suffix else "text"
        
        metadata = ChunkMetadata(
            file_type=file_type,
            chunk_type="small_file",
            line_start=1,
            line_end=len(lines)
        )
        
        return [CodeChunk(
            text=content,
            metadata=metadata,
            file_path=str(file_path),
            start_line=1,
            end_line=len(lines)
        )]


class TreeSitterChunker(BaseChunker):
    """Universal chunker using tree-sitter for multiple languages"""
    
    def __init__(self):
        if TREE_SITTER_FACTORY_AVAILABLE:
            self.factory = TreeSitterFactory()
        else:
            self.factory = None
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk content using tree-sitter AST parsing"""
        if not self.factory:
            return self._fallback_chunk(content, file_path)
        
        try:
            # Parse with tree-sitter
            tree = self.factory.parse_content(content, file_path)
            if not tree or not hasattr(tree, 'root_node'):
                return self._fallback_chunk(content, file_path)
            
            chunks = []
            lines = content.split('\\n')
            
            # Extract symbols from tree
            symbols = self._extract_symbols(tree.root_node, lines)
            
            if not symbols:
                return self._fallback_chunk(content, file_path)
            
            # Create chunks for each symbol
            for symbol in symbols:
                start_line = max(0, symbol['start_line'])
                end_line = min(len(lines), symbol['end_line'])
                
                chunk_text = '\\n'.join(lines[start_line:end_line + 1])
                
                metadata = ChunkMetadata(
                    file_type=file_path.suffix.lstrip('.'),
                    chunk_type=symbol['type'],
                    function_name=symbol.get('name') if symbol['type'] == 'function' else None,
                    class_name=symbol.get('name') if symbol['type'] == 'class' else None,
                    imports=symbol.get('imports', []),
                    line_start=start_line,
                    line_end=end_line,
                    docstring=symbol.get('docstring')
                )
                
                chunks.append(CodeChunk(
                    text=chunk_text,
                    metadata=metadata,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Tree-sitter chunking failed for {file_path}: {e}")
            return self._fallback_chunk(content, file_path)
    
    def _extract_symbols(self, node, lines) -> List[Dict]:
        """Extract symbols (functions, classes, etc.) from tree-sitter AST"""
        symbols = []
        
        def traverse(node):
            # Common tree-sitter node types across languages
            if node.type in ['function_definition', 'function_declaration', 'method_definition', 'function', 'method_declaration']:
                name = self._get_symbol_name(node)
                if name:
                    symbols.append({
                        'type': 'function',
                        'name': name,
                        'start_line': node.start_point[0],
                        'end_line': node.end_point[0],
                        'docstring': self._get_docstring(node, lines)
                    })
            
            elif node.type in ['class_definition', 'class_declaration', 'class', 'struct_declaration', 'interface_declaration', 'struct']:
                name = self._get_symbol_name(node)
                if name:
                    symbols.append({
                        'type': 'class',
                        'name': name,
                        'start_line': node.start_point[0],
                        'end_line': node.end_point[0],
                        'docstring': self._get_docstring(node, lines)
                    })
            
            elif node.type in ['import_declaration', 'import_statement', 'from_import']:
                import_name = self._get_import_name(node, lines)
                if import_name:
                    symbols.append({
                        'type': 'import',
                        'name': import_name,
                        'start_line': node.start_point[0],
                        'end_line': node.end_point[0],
                        'imports': [import_name]
                    })
            
            # Recursively traverse children
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _get_symbol_name(self, node) -> Optional[str]:
        """Extract symbol name from tree-sitter node"""
        # Look for identifier child nodes
        for child in node.children:
            if child.type in ['identifier', 'name', 'type_identifier']:
                return child.text.decode('utf-8') if isinstance(child.text, bytes) else str(child.text)
        return None
    
    def _get_docstring(self, node, lines) -> Optional[str]:
        """Extract docstring/comment before the symbol"""
        start_line = node.start_point[0]
        if start_line > 0 and start_line - 1 < len(lines):
            prev_line = lines[start_line - 1].strip()
            if prev_line.startswith('//') or prev_line.startswith('#') or '/**' in prev_line:
                return prev_line
        return None
    
    def _get_import_name(self, node, lines) -> Optional[str]:
        """Extract import name from import statement"""
        # Get the text of the import line
        start_line = node.start_point[0]
        if start_line < len(lines):
            import_line = lines[start_line].strip()
            # Simple regex extraction for common import patterns
            import re
            patterns = [
                r'import\s+([\w.]+)',           # import java.util.List
                r'from\s+([\w.]+)\s+import',   # from typing import List  
                r'#include\s+[<"]([\w./]+)[>"]', # #include <stdio.h>
                r'use\s+([\w:]+);'             # use std::collections::HashMap;
            ]
            for pattern in patterns:
                match = re.search(pattern, import_line)
                if match:
                    return match.group(1)
        return None
    
    def _fallback_chunk(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback to simple chunking"""
        lines = content.split('\n')
        metadata = ChunkMetadata(
            file_type=file_path.suffix.lstrip('.'),
            chunk_type="fallback",
            line_start=1,
            line_end=len(lines)
        )
        return [CodeChunk(
            text=content, 
            metadata=metadata, 
            file_path=str(file_path),
            start_line=1,
            end_line=len(lines)
        )]


class ASTChunker:
    """Main chunking coordinator that selects appropriate chunker based on file type"""
    
    def __init__(self):
        # Initialize tree-sitter chunker for universal language support
        self.tree_sitter_chunker = TreeSitterChunker() if TREE_SITTER_FACTORY_AVAILABLE else None
        
        self.chunkers = {
            'python': PythonASTChunker(),
            'javascript': JavaScriptASTChunker(),
            'typescript': JavaScriptASTChunker(),  # Use same chunker for TS
            'config': ConfigChunker(),
            'markdown': MarkdownChunker(),
            'small_file': SmallFileChunker()
        }
        
        # Add tree-sitter support for additional languages
        if self.tree_sitter_chunker:
            tree_sitter_languages = [
                'java', 'swift', 'rust', 'kotlin', 'dart', 'go', 
                'c', 'cpp', 'csharp', 'php', 'ruby', 'scala'
            ]
            for lang in tree_sitter_languages:
                self.chunkers[lang] = self.tree_sitter_chunker
    
    def chunk_content(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk content based on file type and structure"""
        try:
            # Determine file type
            file_type = self.get_file_type(file_path)
            
            # Prefer language-aware chunker first
            if file_type in self.chunkers:
                chunks = self.chunkers[file_type].chunk_content(content, file_path)
                if chunks:
                    # If the result is just a fallback and file is small, prefer small_file chunker
                    if (
                        len(content) <= 300 and
                        len(chunks) == 1 and
                        chunks[0].metadata.chunk_type == "fallback"
                    ):
                        return self.chunkers['small_file'].chunk_content(content, file_path)
                    return chunks

            # Fallback to small file heuristic
            if len(content) <= 300:
                return self.chunkers['small_file'].chunk_content(content, file_path)
            
            # Fallback to character-based chunking
            return self._fallback_chunk(content, file_path)
            
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            return self._fallback_chunk(content, file_path)
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension"""
        extension = file_path.suffix.lower()
        
        if extension == '.py':
            return 'python'
        elif extension in ['.js', '.jsx']:
            return 'javascript'
        elif extension in ['.ts', '.tsx']:
            return 'typescript'
        elif extension == '.md':
            return 'markdown'
        elif extension in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env']:
            return 'config'
        else:
            return 'text'
    
    def _fallback_chunk(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback to character-based chunking"""
        chunk_size = 400
        chunks = []
        file_type = file_path.suffix[1:] if file_path.suffix else "text"
        
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i+chunk_size]
        
            # Calculate line numbers
            start_char = i
            end_char = min(i + chunk_size, len(content))
            start_line = content[:start_char].count('\n') + 1
            end_line = content[:end_char].count('\n') + 1
            
            metadata = ChunkMetadata(
                file_type=file_type,
                chunk_type="fallback",
                line_start=start_line,
                line_end=end_line
            )
            
            chunks.append(CodeChunk(
                text=chunk_text,
                metadata=metadata,
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line
            ))
        
        return chunks

# ---------------------------------------------------------------------------
# Legacy compatibility shims (for backward-compatibility with older tests)
# These lightweight aliases expose the previously used class names so that
# external code importing them continues to work without modification.
# ---------------------------------------------------------------------------
class JavaScriptChunker(ASTChunker):
    """Alias maintained for backward-compatibility (maps to ASTChunker)."""
    def __init__(self):
        super().__init__()


class TypeScriptChunker(ASTChunker):
    """Alias maintained for backward-compatibility (maps to ASTChunker)."""
    def __init__(self):
        super().__init__()


class PythonChunker(ASTChunker):
    """Alias maintained for backward-compatibility (maps to ASTChunker)."""
    def __init__(self):
        super().__init__()


# Explicitly export legacy names so `from ast_chunker import XYZ` succeeds
__all__ = [
    "ASTChunker",
    "JavaScriptChunker",
    "TypeScriptChunker",
    "PythonChunker",
]