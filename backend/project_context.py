#!/usr/bin/env python3
"""
Project Context Management for CodeWise Agents
Prevents cross-contamination between different projects in workspace
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ProjectContext:
    """Context information for a specific project"""
    name: str
    base_path: str
    last_accessed: datetime = field(default_factory=datetime.now)
    mentioned_files: Set[str] = field(default_factory=set)
    search_history: List[str] = field(default_factory=list)
    framework_type: Optional[str] = None
    key_files: List[str] = field(default_factory=list)
    
    def update_access_time(self):
        """Update the last accessed timestamp"""
        self.last_accessed = datetime.now()
    
    def add_mentioned_file(self, file_path: str):
        """Add a file that was mentioned/accessed in this context"""
        self.mentioned_files.add(file_path)
        self.update_access_time()
    
    def add_search_query(self, query: str):
        """Add a search query to the history"""
        self.search_history.append(query)
        # Keep only last 10 queries
        if len(self.search_history) > 10:
            self.search_history = self.search_history[-10:]
        self.update_access_time()

class ProjectContextManager:
    """Manages project contexts to prevent cross-contamination"""
    
    def __init__(self):
        self.contexts: Dict[str, ProjectContext] = {}
        self.current_project: Optional[str] = None
        self.max_contexts = 10  # Limit number of cached contexts
        self.context_timeout = timedelta(hours=2)  # Auto-expire old contexts
    
    def set_project_context(self, project_name: str, mentioned_projects: List[str] = None) -> ProjectContext:
        """
        Set the current project context, creating if needed
        Args:
            project_name: Name of the project to set as current
            mentioned_projects: List of projects mentioned in the query
        Returns:
            ProjectContext for the current project
        """
        # Handle multiple mentioned projects - use the first one as primary
        if mentioned_projects and len(mentioned_projects) > 0:
            primary_project = mentioned_projects[0]
            if primary_project != project_name:
                logger.info(f"Switching primary project from {project_name} to {primary_project}")
                project_name = primary_project
        
        # Check if we're switching projects - if so, clear relevant caches
        if self.current_project and self.current_project != project_name:
            logger.info(f"Project context switch: {self.current_project} -> {project_name}")
            self._clear_context_cache()
        
        self.current_project = project_name
        
        # Create or update project context
        if project_name not in self.contexts:
            base_path = f"/workspace/{project_name}" if project_name != "workspace" else "/workspace"
            self.contexts[project_name] = ProjectContext(
                name=project_name,
                base_path=base_path
            )
            logger.info(f"Created new project context: {project_name}")
        else:
            # Update access time for existing context
            self.contexts[project_name].update_access_time()
        
        # Clean up old contexts if we have too many
        self._cleanup_old_contexts()
        
        return self.contexts[project_name]
    
    def get_current_context(self) -> Optional[ProjectContext]:
        """Get the current project context"""
        if self.current_project and self.current_project in self.contexts:
            return self.contexts[self.current_project]
        return None
    
    def add_file_to_context(self, file_path: str):
        """Add a file reference to the current project context"""
        context = self.get_current_context()
        if context:
            context.add_mentioned_file(file_path)
    
    def add_search_to_context(self, query: str):
        """Add a search query to the current project context"""
        context = self.get_current_context()
        if context:
            context.add_search_query(query)
    
    def get_context_for_file(self, file_path: str) -> Optional[str]:
        """
        Determine which project context a file belongs to
        Args:
            file_path: Path to the file
        Returns:
            Project name or None if not determinable
        """
        if not file_path.startswith('/workspace/'):
            return None
        
        # Handle root workspace path
        if file_path == '/workspace/' or file_path == '/workspace':
            return 'workspace'
        
        # Extract project from path: /workspace/project_name/...
        path_without_workspace = file_path.replace('/workspace/', '')
        if not path_without_workspace:  # Empty after removing workspace prefix
            return 'workspace'
            
        path_parts = path_without_workspace.split('/')
        
        # If path has parts and first part is not empty, it's a project directory
        if len(path_parts) > 0 and path_parts[0]:
            # Check if it's a direct workspace file (no subdirectories)
            if len(path_parts) == 1 and '.' in path_parts[0]:
                # It's a file directly in workspace root
                return 'workspace'
            else:
                # It's in a project subdirectory
                return path_parts[0]
        
        return 'workspace'  # Default to workspace
    
    def is_file_in_current_context(self, file_path: str) -> bool:
        """
        Check if a file belongs to the current project context
        Args:
            file_path: Path to check
        Returns:
            True if file is in current context
        """
        if not self.current_project:
            return True  # No context set, allow all files
        
        file_project = self.get_context_for_file(file_path)
        return file_project == self.current_project
    
    def filter_files_by_context(self, file_paths: List[str]) -> List[str]:
        """
        Filter file paths to only include those in the current project context
        Args:
            file_paths: List of file paths to filter
        Returns:
            Filtered list of file paths
        """
        if not self.current_project:
            return file_paths  # No context filtering if no context set
        
        filtered = []
        for file_path in file_paths:
            if self.is_file_in_current_context(file_path):
                filtered.append(file_path)
        
        return filtered
    
    def get_context_summary(self) -> str:
        """Get a summary of the current project context"""
        context = self.get_current_context()
        if not context:
            return "No active project context"
        
        summary = f"Project: {context.name}\n"
        summary += f"Base Path: {context.base_path}\n"
        summary += f"Last Accessed: {context.last_accessed.strftime('%H:%M:%S')}\n"
        summary += f"Files Referenced: {len(context.mentioned_files)}\n"
        summary += f"Search Queries: {len(context.search_history)}\n"
        
        if context.framework_type:
            summary += f"Framework: {context.framework_type}\n"
        
        if context.search_history:
            summary += f"Recent Searches: {', '.join(context.search_history[-3:])}\n"
        
        return summary
    
    def _clear_context_cache(self):
        """Clear context-specific caches when switching projects"""
        # This would clear any cached data that might cause cross-contamination
        # For now, we just log the action
        logger.info("Clearing context cache due to project switch")
    
    def _cleanup_old_contexts(self):
        """Remove old, unused contexts to prevent memory buildup"""
        if len(self.contexts) <= self.max_contexts:
            return
        
        # Sort contexts by last accessed time
        sorted_contexts = sorted(
            self.contexts.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest contexts beyond the limit
        contexts_to_remove = len(self.contexts) - self.max_contexts
        for i in range(contexts_to_remove):
            context_name = sorted_contexts[i][0]
            if context_name != self.current_project:  # Never remove current context
                del self.contexts[context_name]
                logger.info(f"Removed old project context: {context_name}")
    
    def reset_context(self):
        """Reset all context state"""
        self.contexts.clear()
        self.current_project = None
        logger.info("Reset all project contexts")

# Global instance for both agents to use
_global_context_manager = ProjectContextManager()

def get_context_manager() -> ProjectContextManager:
    """Get the global project context manager"""
    return _global_context_manager

def set_project_context(project_name: str, mentioned_projects: List[str] = None) -> ProjectContext:
    """Convenience function to set project context"""
    return get_context_manager().set_project_context(project_name, mentioned_projects)

def get_current_context() -> Optional[ProjectContext]:
    """Convenience function to get current context"""
    return get_context_manager().get_current_context()

def filter_files_by_context(file_paths: List[str]) -> List[str]:
    """Convenience function to filter files by current context"""
    return get_context_manager().filter_files_by_context(file_paths)