"""
Code Lens Annotator for Phase 3.2.2

Enriches code snippets with contextual information from the Knowledge Graph,
creating an IDE-like experience with inline documentation and relationships.

This transforms code snippets from static text into rich, interactive content
with contextual information drawn from the Knowledge Graph.
"""

import re
import logging
import time
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

# Import database manager with fallback for Docker
try:
    from storage.database_manager import DatabaseManager
except ImportError:
    # Fallback for Docker environment
    try:
        from ..storage.database_manager import DatabaseManager
    except ImportError:
        # Fallback for relative imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class CodeLensAnnotator:
    """
    Enriches code snippets with contextual information from the Knowledge Graph,
    creating an IDE-like experience with inline documentation and relationships.
    
    Features:
    - Function signature and docstring tooltips
    - Class inheritance information
    - Import relationship tracking
    - File-level context summaries
    - Caching for performance
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.annotation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self._initialize_patterns()
        
        logger.info("CodeLensAnnotator initialized with Knowledge Graph integration")
    
    def _initialize_patterns(self):
        """Initialize regex patterns for code element detection"""
        self.patterns = {
            # Function calls: function_name(
            'function_call': re.compile(r'(\w+)\s*\('),
            
            # Class usage: ClassName (must start with uppercase)
            'class_usage': re.compile(r'\b([A-Z][a-zA-Z0-9_]*)\b'),
            
            # Method calls: object.method(
            'method_call': re.compile(r'(\w+)\.(\w+)\s*\('),
            
            # Variable assignments: variable_name =
            'variable_assignment': re.compile(r'(\w+)\s*='),
            
            # Import statements: import module or from module import item
            'import_statement': re.compile(r'(?:from\s+(\w+)\s+)?import\s+(.+)'),
            
            # Function definitions: def function_name(
            'function_def': re.compile(r'def\s+(\w+)\s*\('),
            
            # Class definitions: class ClassName
            'class_def': re.compile(r'class\s+(\w+)'),
            
            # Decorators: @decorator_name
            'decorator': re.compile(r'@(\w+)')
        }
    
    def annotate_code(self, code_snippet: str, file_context: Optional[str] = None) -> str:
        """
        Add rich contextual annotations to a code snippet.
        
        Args:
            code_snippet: Raw code text
            file_context: Optional file path for better context resolution
            
        Returns:
            Annotated code with HTML/Markdown annotations
        """
        try:
            lines = code_snippet.split('\n')
            annotated_lines = []
            
            # Add file-level context summary if available
            file_summary = None
            if file_context:
                file_summary = self._get_file_summary(file_context)
            
            for line_num, line in enumerate(lines, 1):
                annotated_line = self._annotate_line(line, file_context, line_num)
                annotated_lines.append(annotated_line)
            
            result = '\n'.join(annotated_lines)
            
            # Add file context header if available
            if file_summary:
                header = f"<!-- File: {file_context} -->\n<!-- {file_summary} -->\n"
                result = header + result
            
            return result
            
        except Exception as e:
            logger.error(f"Code annotation failed: {e}")
            return code_snippet  # Return original if annotation fails
    
    def _annotate_line(self, line: str, file_context: Optional[str], line_num: int) -> str:
        """Annotate a single line of code with contextual information"""
        annotated_line = line
        
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            return line
        
        try:
            # Function call annotations
            annotated_line = self._annotate_function_calls(annotated_line, file_context)
            
            # Class usage annotations
            annotated_line = self._annotate_class_usage(annotated_line, file_context)
            
            # Method call annotations
            annotated_line = self._annotate_method_calls(annotated_line, file_context)
            
            # Function/class definition annotations
            annotated_line = self._annotate_definitions(annotated_line, file_context)
            
            # Import statement annotations
            annotated_line = self._annotate_imports(annotated_line)
            
        except Exception as e:
            logger.debug(f"Line annotation failed for line {line_num}: {e}")
            # Return original line if annotation fails
            return line
        
        return annotated_line
    
    def _annotate_function_calls(self, line: str, file_context: Optional[str]) -> str:
        """Annotate function calls with context information"""
        for match in self.patterns['function_call'].finditer(line):
            func_name = match.group(1)
            
            # Skip obvious non-functions (like if, for, etc.)
            if func_name.lower() in ['if', 'for', 'while', 'with', 'except', 'class', 'def']:
                continue
            
            context = self._get_function_context(func_name, file_context)
            if context:
                # Create HTML annotation with tooltip
                tooltip = context["tooltip"].replace('"', '&quot;')
                replacement = f'<span title="{tooltip}" class="code-lens-function">{func_name}</span>'
                
                # Replace only the function name, not the parenthesis
                line = line.replace(func_name + '(', replacement + '(', 1)
        
        return line
    
    def _annotate_class_usage(self, line: str, file_context: Optional[str]) -> str:
        """Annotate class usage with inheritance and context information"""
        for match in self.patterns['class_usage'].finditer(line):
            class_name = match.group(1)
            
            # Skip common words that might be capitalized but aren't classes
            if class_name.lower() in ['true', 'false', 'none', 'null', 'self', 'this']:
                continue
            
            # Only annotate if we find actual class context
            context = self._get_class_context(class_name, file_context)
            if context and len(class_name) > 2:  # Avoid false positives on short names
                tooltip = context["tooltip"].replace('"', '&quot;')
                replacement = f'<span title="{tooltip}" class="code-lens-class">{class_name}</span>'
                
                # Be careful to only replace the specific occurrence
                line = line.replace(class_name, replacement, 1)
        
        return line
    
    def _annotate_method_calls(self, line: str, file_context: Optional[str]) -> str:
        """Annotate method calls with class and method context"""
        for match in self.patterns['method_call'].finditer(line):
            obj_name, method_name = match.groups()
            
            context = self._get_method_context(obj_name, method_name, file_context)
            if context:
                tooltip = context["tooltip"].replace('"', '&quot;')
                replacement = f'{obj_name}.<span title="{tooltip}" class="code-lens-method">{method_name}</span>'
                
                # Replace the entire obj.method pattern
                original = f'{obj_name}.{method_name}'
                line = line.replace(original, replacement, 1)
        
        return line
    
    def _annotate_definitions(self, line: str, file_context: Optional[str]) -> str:
        """Annotate function and class definitions"""
        # Function definitions
        func_match = self.patterns['function_def'].search(line)
        if func_match:
            func_name = func_match.group(1)
            context = self._get_function_context(func_name, file_context)
            if context:
                tooltip = context["tooltip"].replace('"', '&quot;')
                replacement = f'def <span title="{tooltip}" class="code-lens-def">{func_name}</span>'
                line = line.replace(f'def {func_name}', replacement, 1)
        
        # Class definitions
        class_match = self.patterns['class_def'].search(line)
        if class_match:
            class_name = class_match.group(1)
            context = self._get_class_context(class_name, file_context)
            if context:
                tooltip = context["tooltip"].replace('"', '&quot;')
                replacement = f'class <span title="{tooltip}" class="code-lens-def">{class_name}</span>'
                line = line.replace(f'class {class_name}', replacement, 1)
        
        return line
    
    def _annotate_imports(self, line: str) -> str:
        """Annotate import statements with module information"""
        match = self.patterns['import_statement'].search(line)
        if match:
            from_module, imported_items = match.groups()
            
            # Get context for imported items
            items = [item.strip() for item in imported_items.split(',')]
            for item in items:
                if item and not item.startswith('*'):
                    context = self._get_import_context(item, from_module)
                    if context:
                        tooltip = context["tooltip"].replace('"', '&quot;')
                        replacement = f'<span title="{tooltip}" class="code-lens-import">{item}</span>'
                        line = line.replace(item, replacement, 1)
        
        return line
    
    def _get_function_context(self, func_name: str, file_context: Optional[str]) -> Optional[Dict]:
        """Get contextual information for a function"""
        cache_key = f"func_{func_name}_{file_context or 'global'}"
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Search for function in current file first, then globally
            query = """
                SELECT name, docstring, signature, file_path, line_start
                FROM nodes 
                WHERE type = 'function' AND name = ?
            """
            
            params = [func_name]
            if file_context:
                # First try exact file match, then broader search
                query += " AND file_path = ?"
                params.append(file_context)
            
            query += " ORDER BY file_path LIMIT 1"
            
            result = cursor.execute(query, params).fetchone()
            
            # If no result with file context, try global search
            if not result and file_context:
                result = cursor.execute("""
                    SELECT name, docstring, signature, file_path, line_start
                    FROM nodes 
                    WHERE type = 'function' AND name = ?
                    ORDER BY file_path LIMIT 1
                """, (func_name,)).fetchone()
            
            if not result:
                return None
            
            # Get call count for popularity indicator
            call_count = cursor.execute("""
                SELECT COUNT(*) as calls
                FROM edges e
                JOIN nodes n ON e.target_id = n.id
                WHERE n.name = ? AND e.type = 'calls'
            """, (func_name,)).fetchone()
            
            context = {
                'tooltip': self._build_function_tooltip(result, call_count),
                'signature': result['signature'],
                'docstring': result['docstring'],
                'location': f"{result['file_path']}:{result['line_start']}"
            }
            
            # Cache the result
            self._add_to_cache(cache_key, context)
            return context
            
        except Exception as e:
            logger.debug(f"Function context lookup failed for {func_name}: {e}")
            return None
    
    def _get_class_context(self, class_name: str, file_context: Optional[str]) -> Optional[Dict]:
        """Get contextual information for a class"""
        cache_key = f"class_{class_name}_{file_context or 'global'}"
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Find class definition
            query = """
                SELECT name, docstring, file_path, line_start
                FROM nodes 
                WHERE type = 'class' AND name = ?
            """
            
            params = [class_name]
            if file_context:
                query += " AND file_path = ?"
                params.append(file_context)
            
            query += " ORDER BY file_path LIMIT 1"
            
            result = cursor.execute(query, params).fetchone()
            
            if not result:
                return None
            
            # Get inheritance info
            inheritance = cursor.execute("""
                SELECT n2.name as parent_class
                FROM edges e
                JOIN nodes n1 ON e.source_id = n1.id
                JOIN nodes n2 ON e.target_id = n2.id
                WHERE n1.name = ? AND e.type = 'inherits'
            """, (class_name,)).fetchall()
            
            # Get method count
            method_count = cursor.execute("""
                SELECT COUNT(*) as methods
                FROM nodes 
                WHERE type = 'method' AND file_path = ? AND name LIKE ?
            """, (result['file_path'], f"%{class_name}%")).fetchone()
            
            context = {
                'tooltip': self._build_class_tooltip(result, inheritance, method_count),
                'docstring': result['docstring'],
                'location': f"{result['file_path']}:{result['line_start']}",
                'inheritance': [row['parent_class'] for row in inheritance]
            }
            
            # Cache the result
            self._add_to_cache(cache_key, context)
            return context
            
        except Exception as e:
            logger.debug(f"Class context lookup failed for {class_name}: {e}")
            return None
    
    def _get_method_context(self, obj_name: str, method_name: str, file_context: Optional[str]) -> Optional[Dict]:
        """Get contextual information for a method call"""
        cache_key = f"method_{obj_name}_{method_name}_{file_context or 'global'}"
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Try to find method definition
            result = cursor.execute("""
                SELECT name, docstring, signature, file_path, line_start
                FROM nodes 
                WHERE type = 'method' AND name = ?
                ORDER BY file_path
                LIMIT 1
            """, (method_name,)).fetchone()
            
            if result:
                context = {
                    'tooltip': f"📋 {method_name}() | 📁 {result['file_path'].split('/')[-1]}:{result['line_start']}",
                    'signature': result['signature'],
                    'location': f"{result['file_path']}:{result['line_start']}"
                }
                
                # Add docstring if available
                if result['docstring']:
                    first_line = result['docstring'].split('\n')[0].strip()
                    if len(first_line) > 40:
                        first_line = first_line[:37] + "..."
                    context['tooltip'] = f"📋 {method_name}() | 📝 {first_line} | 📁 {result['file_path'].split('/')[-1]}"
                
                # Cache the result
                self._add_to_cache(cache_key, context)
                return context
            
        except Exception as e:
            logger.debug(f"Method context lookup failed for {obj_name}.{method_name}: {e}")
        
        return None
    
    def _get_import_context(self, item_name: str, from_module: Optional[str]) -> Optional[Dict]:
        """Get contextual information for imported items"""
        cache_key = f"import_{item_name}_{from_module or 'direct'}"
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Look for the imported item in the knowledge graph
            result = cursor.execute("""
                SELECT name, type, file_path, docstring
                FROM nodes 
                WHERE name = ? AND type IN ('function', 'class', 'module')
                ORDER BY type
                LIMIT 1
            """, (item_name,)).fetchone()
            
            if result:
                item_type = result['type']
                emoji = {'function': '🔧', 'class': '🏛️', 'module': '📦'}.get(item_type, '📋')
                
                tooltip = f"{emoji} {item_type} {item_name}"
                if from_module:
                    tooltip += f" from {from_module}"
                
                if result['docstring']:
                    first_line = result['docstring'].split('\n')[0].strip()
                    if len(first_line) > 30:
                        first_line = first_line[:27] + "..."
                    tooltip += f" | 📝 {first_line}"
                
                context = {
                    'tooltip': tooltip,
                    'type': item_type,
                    'module': from_module
                }
                
                # Cache the result
                self._add_to_cache(cache_key, context)
                return context
            
        except Exception as e:
            logger.debug(f"Import context lookup failed for {item_name}: {e}")
        
        return None
    
    def _build_function_tooltip(self, func_data: Dict, call_data: Dict) -> str:
        """Build tooltip text for function annotations"""
        tooltip_parts = []
        
        # Function signature
        if func_data['signature']:
            tooltip_parts.append(f"📋 {func_data['signature']}")
        else:
            tooltip_parts.append(f"📋 {func_data['name']}()")
        
        # Docstring (first line only)
        if func_data['docstring']:
            first_line = func_data['docstring'].split('\n')[0].strip()
            if len(first_line) > 60:
                first_line = first_line[:57] + "..."
            tooltip_parts.append(f"📝 {first_line}")
        
        # Location
        file_name = func_data['file_path'].split('/')[-1]
        tooltip_parts.append(f"📁 {file_name}:{func_data['line_start']}")
        
        # Popularity indicator
        if call_data and call_data['calls'] > 0:
            tooltip_parts.append(f"🔗 Called {call_data['calls']} times")
        
        return " | ".join(tooltip_parts)
    
    def _build_class_tooltip(self, class_data: Dict, inheritance: List[Dict], method_count: Dict) -> str:
        """Build tooltip text for class annotations"""
        tooltip_parts = []
        
        # Class name
        tooltip_parts.append(f"🏛️ class {class_data['name']}")
        
        # Inheritance
        if inheritance:
            parents = [inh['parent_class'] for inh in inheritance[:2]]  # Max 2 parents
            parent_text = ", ".join(parents)
            if len(inheritance) > 2:
                parent_text += f" + {len(inheritance)-2} more"
            tooltip_parts.append(f"⬆️ extends {parent_text}")
        
        # Method count
        if method_count and method_count['methods'] > 0:
            tooltip_parts.append(f"🔧 {method_count['methods']} methods")
        
        # Docstring (first line only)
        if class_data['docstring']:
            first_line = class_data['docstring'].split('\n')[0].strip()
            if len(first_line) > 50:
                first_line = first_line[:47] + "..."
            tooltip_parts.append(f"📝 {first_line}")
        
        # Location
        file_name = class_data['file_path'].split('/')[-1]
        tooltip_parts.append(f"📁 {file_name}:{class_data['line_start']}")
        
        return " | ".join(tooltip_parts)
    
    def _get_file_summary(self, file_path: str) -> Optional[str]:
        """Get high-level summary of a file's purpose"""
        cache_key = f"file_summary_{file_path}"
        
        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached.get('summary')
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Count different types of symbols in the file
            counts = cursor.execute("""
                SELECT type, COUNT(*) as count
                FROM nodes
                WHERE file_path = ?
                GROUP BY type
            """, (file_path,)).fetchall()
            
            if not counts:
                return None
            
            # Build summary
            type_counts = {row['type']: row['count'] for row in counts}
            summary_parts = []
            
            if 'class' in type_counts:
                summary_parts.append(f"{type_counts['class']} classes")
            if 'function' in type_counts:
                summary_parts.append(f"{type_counts['function']} functions")
            if 'method' in type_counts:
                summary_parts.append(f"{type_counts['method']} methods")
            if 'import' in type_counts:
                summary_parts.append(f"{type_counts['import']} imports")
            
            summary = "Contains: " + ", ".join(summary_parts) if summary_parts else None
            
            # Cache the result
            if summary:
                self._add_to_cache(cache_key, {'summary': summary})
            
            return summary
            
        except Exception as e:
            logger.debug(f"File summary failed for {file_path}: {e}")
            return None
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get item from cache if not expired"""
        if key in self.annotation_cache:
            entry = self.annotation_cache[key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return entry['data']
            else:
                # Remove expired entry
                del self.annotation_cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: Dict):
        """Add item to cache with timestamp"""
        self.annotation_cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Simple cache cleanup - remove old entries if cache gets too large
        if len(self.annotation_cache) > 1000:
            # Remove oldest 100 entries
            sorted_items = sorted(self.annotation_cache.items(), 
                                key=lambda x: x[1]['timestamp'])
            for old_key, _ in sorted_items[:100]:
                del self.annotation_cache[old_key]
    
    def clear_cache(self):
        """Clear the annotation cache"""
        self.annotation_cache.clear()
        logger.info("Code annotation cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_items': len(self.annotation_cache),
            'cache_ttl_seconds': self.cache_ttl,
            'cache_hit_rate': 'Not tracked'  # Could add hit/miss tracking later
        }