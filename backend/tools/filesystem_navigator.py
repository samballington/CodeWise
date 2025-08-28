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

# Import PathManager for consistent path handling
try:
    from storage.path_manager import get_path_manager
    PATH_MANAGER_AVAILABLE = True
except ImportError:
    PATH_MANAGER_AVAILABLE = False
    
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
        Initialize with database manager and PathManager for consistent path handling.
        
        Args:
            db_manager: DatabaseManager instance with connection attribute
        """
        self.db = db_manager
        
        # Initialize PathManager for consistent path operations
        if PATH_MANAGER_AVAILABLE:
            self.path_manager = get_path_manager()
            logger.info("FilesystemNavigator initialized with PathManager integration")
        else:
            self.path_manager = None
            logger.warning("PathManager not available - using legacy path handling")
        
    def _build_project_filter(self, base_query: str, params: list, project_scope: Optional[str] = None) -> tuple:
        """
        Add project scope filtering to SQL query using PathManager for accurate patterns.
        
        Args:
            base_query: Base SQL query string
            params: List of query parameters
            project_scope: Project name to filter by
            
        Returns:
            Tuple of (modified_query, modified_params)
        """
        if project_scope:
            # Use PathManager to build accurate search patterns if available
            if self.path_manager:
                try:
                    relative_pattern, absolute_pattern = self.path_manager.build_search_patterns(project_scope)
                    logger.debug(f"PathManager patterns: relative={relative_pattern}, absolute={absolute_pattern}")
                except Exception as e:
                    logger.warning(f"PathManager pattern building failed: {e}, using fallback")
                    relative_pattern = f"{project_scope}/%"
                    absolute_pattern = f"/workspace/{project_scope}/%"
            else:
                # Fallback to legacy patterns
                relative_pattern = f"{project_scope}/%"
                absolute_pattern = f"/workspace/{project_scope}/%"
            
            if "WHERE" in base_query.upper():
                # Already has WHERE clause - add OR condition for both patterns
                modified_query = base_query.replace("WHERE", f"WHERE (file_path LIKE ? OR file_path LIKE ?) AND", 1)
                modified_params = [relative_pattern, absolute_pattern] + params
            else:
                # No WHERE clause - insert OR condition before final clauses
                insertion_point = len(base_query)
                for clause in ["ORDER BY", "GROUP BY", "LIMIT", "OFFSET"]:
                    pos = base_query.upper().find(clause)
                    if pos != -1 and pos < insertion_point:
                        insertion_point = pos
                
                if insertion_point == len(base_query):
                    modified_query = base_query + " WHERE (file_path LIKE ? OR file_path LIKE ?)"
                else:
                    modified_query = base_query[:insertion_point].strip() + " WHERE (file_path LIKE ? OR file_path LIKE ?) " + base_query[insertion_point:]
                
                modified_params = [relative_pattern, absolute_pattern] + params
                
            logger.info(f"🎯 Applied project filter: {project_scope} (patterns: {relative_pattern}, {absolute_pattern})")
            return modified_query, modified_params
        else:
            logger.info("📂 No project filter applied - searching all projects")
            return base_query, params
    
    def validate_project_paths(self, project_name: str) -> Dict[str, Any]:
        """
        Validate path consistency for a project and provide diagnostic information.
        
        Args:
            project_name: Name of the project to validate
            
        Returns:
            Dictionary with validation results and suggestions
        """
        cursor = self.db.connection.cursor()
        
        # Check different path formats for the project
        relative_pattern = f"{project_name}/%"
        absolute_pattern = f"/workspace/{project_name}/%"
        
        # Count files by path format
        relative_count = cursor.execute(
            "SELECT COUNT(DISTINCT file_path) FROM nodes WHERE file_path LIKE ?", 
            [relative_pattern]
        ).fetchone()[0]
        
        absolute_count = cursor.execute(
            "SELECT COUNT(DISTINCT file_path) FROM nodes WHERE file_path LIKE ?", 
            [absolute_pattern]
        ).fetchone()[0]
        
        # Only check for files without project prefix if the project actually exists
        no_prefix_count = 0
        if relative_count > 0 or absolute_count > 0:
            # Project exists, check for files without project prefix (potential issues)
            no_prefix_patterns = [
                "frontend/%", "backend/%", "src/%", "lib/%", "components/%",
                "pages/%", "styles/%", "utils/%", "api/%", "models/%"
            ]
            
            for pattern in no_prefix_patterns:
                count = cursor.execute(
                    "SELECT COUNT(DISTINCT file_path) FROM nodes WHERE file_path LIKE ? AND file_path NOT LIKE ? AND file_path NOT LIKE ?",
                    [pattern, relative_pattern, absolute_pattern]
                ).fetchone()[0]
                no_prefix_count += count
        
        # Get all available projects
        all_projects = cursor.execute("""
            SELECT DISTINCT 
                CASE 
                    WHEN file_path LIKE '/workspace/%' THEN 
                        SUBSTR(file_path, 12, INSTR(SUBSTR(file_path, 12), '/') - 1)
                    ELSE
                        SUBSTR(file_path, 1, INSTR(file_path, '/') - 1)
                END as project
            FROM nodes 
            WHERE project != '' AND project IS NOT NULL
            ORDER BY project
            LIMIT 10
        """).fetchall()
        
        available_projects = [p[0] for p in all_projects if p[0] and p[0] != project_name]
        
        return {
            "project_name": project_name,
            "relative_path_files": relative_count,
            "absolute_path_files": absolute_count,
            "no_prefix_files": no_prefix_count,
            "total_files": relative_count + absolute_count + no_prefix_count,
            "path_consistency": "good" if relative_count > 0 and no_prefix_count == 0 else "issues",
            "available_projects": available_projects[:5],  # Top 5 suggestions
            "recommendations": self._get_path_recommendations(relative_count, absolute_count, no_prefix_count)
        }
    
    def _get_path_recommendations(self, relative_count: int, absolute_count: int, no_prefix_count: int) -> List[str]:
        """Generate recommendations based on path consistency analysis."""
        recommendations = []
        
        if relative_count == 0 and absolute_count == 0:
            recommendations.append("Project not found in database - check project name spelling")
        elif no_prefix_count > 0:
            recommendations.append(f"Found {no_prefix_count} files without project prefix - may need path migration")
        elif absolute_count > relative_count:
            recommendations.append("Most files use absolute paths - consider running path migration for consistency")
        elif relative_count > 0:
            recommendations.append("Project has consistent normalized paths - filesystem navigation should work correctly")
        
        return recommendations
    
    def execute(self, operation: str, path: str = None, pattern: str = None, 
                recursive: bool = False, project_scope: str = None) -> Dict[str, Any]:
        """
        Execute filesystem navigation operation.
        
        Args:
            operation: One of "list", "find", "tree"
            path: Directory path (required for list/tree)
            pattern: File pattern (required for find, optional for list)
            recursive: Whether to search recursively
            project_scope: Project name to filter results (e.g., 'infinite-kanvas')
            
        Returns:
            Dictionary with operation results or error information
        """
        try:
            logger.info(f"Executing filesystem operation: {operation} path={path} pattern={pattern} recursive={recursive} project_scope={project_scope}")
            
            if operation == "list":
                if not path:
                    return {"error": "A 'path' is required for the 'list' operation."}
                return self._list_directory(path, pattern, recursive, project_scope)
            elif operation == "find":
                if not pattern:
                    return {"error": "A 'pattern' is required for the 'find' operation."}
                return self._find_files(pattern, recursive, project_scope)
            elif operation == "tree":
                if not path:
                    return {"error": "A 'path' is required for the 'tree' operation."}
                return self._show_tree(path, project_scope)
            elif operation == "validate":
                if not project_scope:
                    return {"error": "A 'project_scope' is required for the 'validate' operation."}
                return self.validate_project_paths(project_scope)
            else:
                return {"error": f"Unknown operation: {operation}. Valid operations: list, find, tree, validate"}
                
        except sqlite3.Error as e:
            logger.error(f"Database query failed for operation {operation}: {e}")
            return {"error": f"Database query failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error in filesystem navigation: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _list_directory(self, path: str, pattern: Optional[str], recursive: bool, project_scope: Optional[str] = None) -> Dict[str, Any]:
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
        
        # Quick validation - check if path exists in any files before doing expensive queries
        cursor = self.db.connection.cursor()
        
        # Build path check query with project filtering
        base_query = "SELECT COUNT(*) FROM nodes WHERE file_path LIKE ?"
        params = [f"%{path}%"]
        query, query_params = self._build_project_filter(base_query, params, project_scope)
        
        path_check = cursor.execute(query, query_params).fetchone()[0]
        
        if path_check == 0:
            # Enhanced error handling with path validation
            if project_scope:
                # If project was specified, validate its path consistency
                validation = self.validate_project_paths(project_scope)
                
                if validation['total_files'] == 0:
                    return {
                        "error": f"Project '{project_scope}' not found in codebase",
                        "suggestions": f"Available projects: {', '.join(validation['available_projects'])}",
                        "files": [],
                        "operation": "list",
                        "path": path,
                        "validation": validation
                    }
                else:
                    return {
                        "error": f"Path '{path}' not found in project '{project_scope}'",
                        "project_info": f"Project has {validation['total_files']} files",
                        "recommendations": validation['recommendations'],
                        "files": [],
                        "operation": "list", 
                        "path": path,
                        "validation": validation
                    }
            else:
                # Generic suggestions when no project specified
                project_suggestions = cursor.execute(
                    "SELECT DISTINCT SUBSTR(file_path, 1, INSTR(file_path, '/') - 1) as project FROM nodes WHERE project != '' LIMIT 5"
                ).fetchall()
                suggestions = [s[0] for s in project_suggestions if s[0]]
                
                return {
                    "error": f"Path '{path}' not found in codebase",
                    "suggestions": f"Try specifying a project: {', '.join(suggestions)}",
                    "files": [],
                    "operation": "list",
                    "path": path
                }
        
        cursor = self.db.connection.cursor()
        
        if recursive:
            # Recursive search: include all subdirectories
            like_pattern = f"%{path}%"
            base_query = "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE ? ORDER BY file_path"
            query, query_params = self._build_project_filter(base_query, [like_pattern], project_scope)
            results = cursor.execute(query, query_params).fetchall()
        else:
            # Non-recursive: only immediate directory contents
            # Match files that start with path/ but don't have additional slashes
            base_query = """
                SELECT DISTINCT file_path FROM nodes 
                WHERE file_path LIKE ? 
                AND file_path NOT LIKE ?
                ORDER BY file_path
            """
            path_prefix = f"%{path}/%"
            path_with_subdir = f"%{path}/%/%"
            query, query_params = self._build_project_filter(base_query, [path_prefix, path_with_subdir], project_scope)
            results = cursor.execute(query, query_params).fetchall()
        
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
    
    def _find_files(self, pattern: str, recursive: bool = True, project_scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Find files matching a pattern across the codebase.
        
        Args:
            pattern: Pattern to search for (supports wildcards)
            recursive: Always true for find operations
            
        Returns:
            Dictionary with matching files
        """
        cursor = self.db.connection.cursor()
        
        # Get all file paths from knowledge graph with project filtering
        base_query = "SELECT DISTINCT file_path FROM nodes ORDER BY file_path"
        query, query_params = self._build_project_filter(base_query, [], project_scope)
        results = cursor.execute(query, query_params).fetchall()
        
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
    
    def _show_tree(self, path: str, project_scope: Optional[str] = None) -> Dict[str, Any]:
        """
        Show hierarchical tree structure of a directory.
        
        Args:
            path: Root directory path for tree display
            
        Returns:
            Dictionary with tree structure
        """
        path = path.strip('/')
        
        cursor = self.db.connection.cursor()
        base_query = "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE ? ORDER BY file_path"
        like_pattern = f"%{path}%"
        query, query_params = self._build_project_filter(base_query, [like_pattern], project_scope)
        results = cursor.execute(query, query_params).fetchall()
        
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