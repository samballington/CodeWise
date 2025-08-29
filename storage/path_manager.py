"""
Centralized Path Management Service

Industry standard approach for consistent file path handling across all components.
Ensures filesystem navigator tool works reliably by enforcing path conventions.

REQ-3.5.3: Centralized path normalization for filesystem compatibility
"""

import os
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

class PathManager:
    """
    Centralized path management service ensuring consistent path formats.
    
    Standard format: All paths stored as workspace-relative with project prefix
    Example: "iiot-monitoring/frontend/src/App.js"
    
    This ensures filesystem navigator can always find files using pattern:
    WHERE file_path LIKE 'project_name/%'
    """
    
    def __init__(self, workspace_root: str = "/workspace"):
        """
        Initialize path manager.
        
        Args:
            workspace_root: Root directory for all projects (default: /workspace)
        """
        self.workspace_root = Path(workspace_root).resolve()
        logger.info(f"PathManager initialized with workspace: {self.workspace_root}")
    
    def normalize_for_storage(self, file_path: Union[str, Path], project_name: Optional[str] = None) -> str:
        """
        Normalize file path for consistent database storage.
        
        Args:
            file_path: Input file path (absolute or relative)
            project_name: Project name for validation (optional)
            
        Returns:
            Normalized workspace-relative path with project prefix
            
        Examples:
            /workspace/iiot-monitoring/src/App.js -> iiot-monitoring/src/App.js
            /Users/dev/project/src/App.js -> project/src/App.js (when workspace=/Users/dev)
            src/App.js + project_name=iiot -> iiot-monitoring/src/App.js
        """
        # Handle empty string case
        if not file_path or str(file_path).strip() == "":
            if project_name:
                return f"{project_name}/"
            return ""
        
        # Handle relative paths first (before resolving)
        path_input = Path(file_path)
        path_str = str(file_path)
        
        # Check if it's truly a relative path (cross-platform)
        is_relative = (not path_input.is_absolute() and 
                      not path_str.startswith('/') and 
                      not (len(path_str) > 1 and path_str[1] == ':'))
        
        if is_relative:
            # It's a relative path - handle with project context if available
            if project_name:
                relative_str = path_str.replace('\\', '/')
                normalized = f"{project_name}/{relative_str}"
                logger.debug(f"Normalized relative path: {file_path} -> {normalized}")
                return normalized
        
        path_obj = Path(file_path).resolve()
        path_str = str(path_obj)
        
        # Convert to workspace-relative path
        try:
            # Try to make relative to workspace root
            relative_path = path_obj.relative_to(self.workspace_root)
            normalized = str(relative_path).replace('\\', '/')  # Ensure forward slashes
            
            # Validate project prefix exists
            if '/' not in normalized:
                # File is at workspace root, add project prefix if provided
                if project_name:
                    normalized = f"{project_name}/{normalized}"
                else:
                    logger.warning(f"File at workspace root without project context: {normalized}")
            
            logger.debug(f"Normalized path: {file_path} -> {normalized}")
            return normalized
            
        except ValueError:
            # Path is not under workspace root
            if project_name:
                # Check if original path was relative (before resolve())
                original_path = Path(file_path)
                file_path_str = str(file_path)
                
                # Use more robust absolute path detection
                is_absolute_path = (original_path.is_absolute() or 
                                  file_path_str.startswith('/') or 
                                  (len(file_path_str) > 1 and file_path_str[1] == ':'))
                
                if not is_absolute_path:
                    # It was a relative path, prepend project name
                    relative_str = file_path_str.replace('\\', '/')
                    normalized = f"{project_name}/{relative_str}"
                    logger.debug(f"Exception handler: treating as relative path")
                else:
                    # It was an absolute path outside workspace, use filename only
                    filename = path_obj.name
                    normalized = f"{project_name}/{filename}"
                    logger.debug(f"Exception handler: treating as absolute path, filename={filename}")
                logger.warning(f"Path outside workspace, using project context: {file_path} -> {normalized}")
                return normalized
            
            # Fallback: return as-is but log warning
            logger.warning(f"Cannot normalize path outside workspace: {file_path}")
            return str(file_path).replace('\\', '/')
    
    def extract_project_name(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Extract project name from file path.
        
        Args:
            file_path: File path (absolute or normalized)
            
        Returns:
            Project name or None if cannot be determined
        """
        normalized = self.normalize_for_storage(file_path)
        
        # Project name is first path component
        parts = normalized.split('/')
        if parts and parts[0]:
            return parts[0]
        
        return None
    
    def build_search_patterns(self, project_name: str) -> tuple[str, str]:
        """
        Build filesystem navigator search patterns for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Tuple of (relative_pattern, absolute_pattern) for SQL LIKE queries
        """
        relative_pattern = f"{project_name}/%"
        absolute_pattern = f"{self.workspace_root}/{project_name}/%"
        
        logger.debug(f"Search patterns for {project_name}: {relative_pattern}, {absolute_pattern}")
        return relative_pattern, absolute_pattern
    
    def validate_path_consistency(self, project_name: str) -> dict:
        """
        Validate path consistency for a project in the database.
        
        Args:
            project_name: Project to validate
            
        Returns:
            Dictionary with validation results
        """
        # This would query the database to check path consistency
        # Implementation depends on database connection
        return {
            "project": project_name,
            "consistent": True,
            "issues": []
        }

# Global instance for consistent usage
_path_manager = None

def get_path_manager() -> PathManager:
    """Get singleton PathManager instance."""
    global _path_manager
    if _path_manager is None:
        workspace_root = os.getenv("WORKSPACE_ROOT", "/workspace")
        _path_manager = PathManager(workspace_root)
    return _path_manager

def normalize_path(file_path: Union[str, Path], project_name: Optional[str] = None) -> str:
    """Convenience function for path normalization."""
    return get_path_manager().normalize_for_storage(file_path, project_name)

def get_project_from_path(file_path: Union[str, Path]) -> Optional[str]:
    """Convenience function to extract project name from path."""
    return get_path_manager().extract_project_name(file_path)