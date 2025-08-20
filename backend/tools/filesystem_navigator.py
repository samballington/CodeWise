"""
Filesystem Navigator Implementation

Provides deterministic filesystem operations using Knowledge Graph data.
This eliminates the need for slow filesystem scans by leveraging indexed data.

REQ-3.5.2: FilesystemNavigator class implementation
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import fnmatch

logger = logging.getLogger(__name__)


class FilesystemNavigator:
    """
    Filesystem navigation tool that leverages Knowledge Graph data for instant results.
    
    This class provides three core operations:
    - list: Show contents of a specific directory
    - find: Search for files matching patterns
    - tree: Display hierarchical directory structure
    """
    
    def __init__(self, db_manager):
        """
        Initialize with database manager.
        
        Args:
            db_manager: DatabaseManager instance with connection attribute
        """
        self.db = db_manager
        
    def execute(self, operation: str, path: str = None, pattern: str = None, 
                recursive: bool = False) -> Dict[str, Any]:
        """
        Execute filesystem navigation operation.
        
        Args:
            operation: One of "list", "find", "tree"
            path: Directory path (required for list/tree)
            pattern: File pattern (required for find, optional for list)
            recursive: Whether to search recursively
            
        Returns:
            Dictionary with operation results or error information
        """
        try:
            logger.info(f"Executing filesystem operation: {operation} path={path} pattern={pattern} recursive={recursive}")
            
            if operation == "list":
                if not path:
                    return {"error": "A 'path' is required for the 'list' operation."}
                return self._list_directory(path, pattern, recursive)
            elif operation == "find":
                if not pattern:
                    return {"error": "A 'pattern' is required for the 'find' operation."}
                return self._find_files(pattern, recursive)
            elif operation == "tree":
                if not path:
                    return {"error": "A 'path' is required for the 'tree' operation."}
                return self._show_tree(path)
            else:
                return {"error": f"Unknown operation: {operation}. Valid operations: list, find, tree"}
                
        except sqlite3.Error as e:
            logger.error(f"Database query failed for operation {operation}: {e}")
            return {"error": f"Database query failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in filesystem navigation: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _list_directory(self, path: str, pattern: Optional[str], recursive: bool) -> Dict[str, Any]:
        """
        List files in a specific directory.
        
        Args:
            path: Directory path to list
            pattern: Optional pattern to filter files
            recursive: Whether to search recursively
            
        Returns:
            Dictionary with files list and metadata
        """
        # Normalize path (remove leading/trailing slashes)
        path = path.strip('/')
        
        cursor = self.db.connection.cursor()
        
        if recursive:
            # Recursive search: include all subdirectories
            like_pattern = f"%{path}%"
            query = "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE ? ORDER BY file_path"
            results = cursor.execute(query, (like_pattern,)).fetchall()
        else:
            # Non-recursive: only immediate directory contents
            # Match files that start with path/ but don't have additional slashes
            query = """
                SELECT DISTINCT file_path FROM nodes 
                WHERE file_path LIKE ? 
                AND file_path NOT LIKE ?
                ORDER BY file_path
            """
            path_prefix = f"%{path}/%"
            path_with_subdir = f"%{path}/%/%"
            results = cursor.execute(query, (path_prefix, path_with_subdir)).fetchall()
        
        files = [row[0] for row in results]
        
        # Apply pattern filtering if specified
        if pattern:
            filtered_files = []
            for file_path in files:
                file_name = Path(file_path).name
                if fnmatch.fnmatch(file_name, pattern):
                    filtered_files.append(file_path)
            files = filtered_files
        
        return {
            "operation": "list",
            "path": path,
            "pattern": pattern,
            "recursive": recursive,
            "files": files,
            "count": len(files)
        }
    
    def _find_files(self, pattern: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Find files matching a pattern across the codebase.
        
        Args:
            pattern: Pattern to search for (supports wildcards)
            recursive: Always true for find operations
            
        Returns:
            Dictionary with matching files
        """
        cursor = self.db.connection.cursor()
        
        # Get all file paths from knowledge graph
        query = "SELECT DISTINCT file_path FROM nodes ORDER BY file_path"
        results = cursor.execute(query).fetchall()
        
        matching_files = []
        for row in results:
            file_path = row[0]
            file_name = Path(file_path).name
            
            # Support both filename and path pattern matching
            if (fnmatch.fnmatch(file_name, pattern) or 
                fnmatch.fnmatch(file_path, pattern) or
                pattern.lower() in file_name.lower()):
                matching_files.append(file_path)
        
        return {
            "operation": "find",
            "pattern": pattern,
            "files": matching_files,
            "count": len(matching_files)
        }
    
    def _show_tree(self, path: str) -> Dict[str, Any]:
        """
        Show hierarchical tree structure of a directory.
        
        Args:
            path: Root directory path for tree display
            
        Returns:
            Dictionary with tree structure
        """
        path = path.strip('/')
        
        cursor = self.db.connection.cursor()
        query = "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE ? ORDER BY file_path"
        like_pattern = f"%{path}%"
        results = cursor.execute(query, (like_pattern,)).fetchall()
        
        if not results:
            return {
                "operation": "tree",
                "path": path,
                "tree": f"No files found under path: {path}",
                "file_count": 0
            }
        
        # Build tree structure
        tree_structure = {}
        file_count = 0
        
        for row in results:
            file_path = row[0]
            file_count += 1
            
            # Create path relative to the requested path
            try:
                if path in file_path:
                    # Extract the part after the path
                    path_start = file_path.find(path)
                    relative_part = file_path[path_start + len(path):].lstrip('/')
                    
                    if relative_part:  # Skip empty relative paths
                        parts = relative_part.split('/')
                        current_level = tree_structure
                        
                        for part in parts:
                            if part not in current_level:
                                current_level[part] = {}
                            current_level = current_level[part]
            except Exception as e:
                logger.warning(f"Error processing path {file_path}: {e}")
                continue
        
        # Format tree as string
        tree_string = self._format_tree(tree_structure, path, "")
        
        return {
            "operation": "tree",
            "path": path,
            "tree": tree_string,
            "file_count": file_count
        }
    
    def _format_tree(self, structure: Dict, root: str, indent: str = "") -> str:
        """
        Format nested dictionary as tree structure string.
        
        Args:
            structure: Nested dictionary representing directory structure
            root: Root directory name
            indent: Current indentation level
            
        Returns:
            Formatted tree string
        """
        if not structure:
            return f"{indent}{root}/ (empty)"
        
        tree_lines = []
        if not indent:  # Root level
            tree_lines.append(f"{root}/")
        
        items = list(structure.items())
        for i, (name, children) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            
            tree_lines.append(f"{indent}{connector}{name}")
            
            if children:  # Has subdirectories
                next_indent = indent + ("    " if is_last else "│   ")
                subtree = self._format_tree(children, "", next_indent)
                if subtree:
                    tree_lines.append(subtree)
        
        return "\n".join(tree_lines)


# Export for easy import
__all__ = ["FilesystemNavigator"]