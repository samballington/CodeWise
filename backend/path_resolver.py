#!/usr/bin/env python3
"""
PathResolver - Fix Task 3 path resolution issues
Handles various input formats and resolves them to correct workspace paths
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PathResolver:
    """Resolves various file path formats to consistent workspace paths"""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.project_mappings = self._build_project_mappings()
        
    def _build_project_mappings(self) -> Dict[str, str]:
        """Build mapping of project names to their actual directories"""
        mappings = {}
        
        if self.workspace_root.exists():
            for item in self.workspace_root.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Map both the exact name and common variations
                    mappings[item.name] = str(item)
                    mappings[f"@{item.name}"] = str(item)
                    
        logger.info(f"Built project mappings: {list(mappings.keys())}")
        return mappings
    
    def resolve_file_path(self, input_path: str, project_context: str = None) -> Tuple[str, bool]:
        """
        Convert various path formats to absolute workspace paths
        Returns: (resolved_path, exists)
        """
        if not input_path or not input_path.strip():
            return "", False
            
        input_path = input_path.strip()
        logger.info(f"🔧 PATH RESOLVER: Resolving '{input_path}'")
        
        # Strategy 1: Handle @project-name prefix format
        if input_path.startswith("@"):
            resolved = self._resolve_project_prefix(input_path)
            if resolved:
                exists = Path(resolved).exists()
                logger.info(f"🔧 Project prefix resolved: {input_path} → {resolved} (exists: {exists})")
                return resolved, exists
        
        # Strategy 2: Handle absolute paths (already correct)
        if input_path.startswith("/workspace"):
            exists = Path(input_path).exists()
            logger.info(f"🔧 Absolute path: {input_path} (exists: {exists})")
            return input_path, exists
        
        # Strategy 3: Handle relative paths from search results
        if not input_path.startswith("/"):
            # Try direct workspace relative path
            full_path = self.workspace_root / input_path.lstrip('./')
            if full_path.exists():
                resolved = str(full_path)
                logger.info(f"🔧 Relative path resolved: {input_path} → {resolved}")
                return resolved, True
        
        # Strategy 4: Try to find the file using project context
        if project_context:
            resolved = self._resolve_with_project_context(input_path, project_context)
            if resolved:
                exists = Path(resolved).exists()
                logger.info(f"🔧 Project context resolved: {input_path} → {resolved} (exists: {exists})")
                return resolved, exists
        
        # Strategy 5: Search across all projects for the filename
        resolved = self._search_across_projects(input_path)
        if resolved:
            exists = Path(resolved).exists()
            logger.info(f"🔧 Cross-project search resolved: {input_path} → {resolved} (exists: {exists})")
            return resolved, exists
        
        # Fallback: Return as-is with workspace prefix
        fallback = str(self.workspace_root / input_path.lstrip('./'))
        exists = Path(fallback).exists()
        logger.info(f"🔧 Fallback resolution: {input_path} → {fallback} (exists: {exists})")
        return fallback, exists
    
    def _resolve_project_prefix(self, path: str) -> Optional[str]:
        """Convert @project-name file.ext to workspace/project-name/path/file.ext"""
        # Parse @project-name filename format (handle both space and no space)
        match = re.match(r'@([^\s]+)\s+(.+)', path)
        if not match:
            # Try without space: @project-name/path/file.ext
            match = re.match(r'@([^/]+)/(.+)', path)
        if not match:
            # Try just @project-name filename.ext
            match = re.match(r'@([^\s]+)\s*(.+)', path)
        if not match:
            return None
            
        project_name, filename = match.groups()
        
        # Look for the project directory
        project_dir = self.workspace_root / project_name
        if not project_dir.exists():
            logger.warning(f"Project directory not found: {project_dir}")
            return None
        
        # Search for the file in the project directory
        return self._find_file_in_project(project_dir, filename)
    
    def _find_file_in_project(self, project_dir: Path, filename: str) -> Optional[str]:
        """Find a file within a project directory"""
        # Try common locations first
        common_paths = [
            filename,  # Root of project
            f"src/{filename}",
            f"src/utils/{filename}",
            f"src/components/{filename}",
            f"src/lib/{filename}",
            f"backend/{filename}",
            f"frontend/{filename}",
        ]
        
        for common_path in common_paths:
            full_path = project_dir / common_path
            if full_path.exists():
                return str(full_path)
        
        # If not found in common locations, search recursively
        for root, dirs, files in os.walk(project_dir):
            if filename in files:
                return os.path.join(root, filename)
        
        return None
    
    def _resolve_with_project_context(self, input_path: str, project_context: str) -> Optional[str]:
        """Use project context to resolve relative paths"""
        if project_context in self.project_mappings:
            project_dir = Path(self.project_mappings[project_context])
            return self._find_file_in_project(project_dir, input_path)
        return None
    
    def _search_across_projects(self, filename: str) -> Optional[str]:
        """Search for a file across all workspace projects"""
        # Extract just the filename if it's a path
        if '/' in filename:
            filename = Path(filename).name
        
        for project_dir in self.workspace_root.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                found = self._find_file_in_project(project_dir, filename)
                if found:
                    return found
        
        return None
    
    def resolve_from_search_results(self, search_results: List[Dict]) -> List[str]:
        """Resolve file paths from smart_search results"""
        resolved_paths = []
        
        for result in search_results:
            if hasattr(result, 'file_path'):
                file_path = result.file_path
            elif isinstance(result, dict) and 'file_path' in result:
                file_path = result['file_path']
            else:
                continue
                
            resolved, exists = self.resolve_file_path(file_path)
            if exists:
                resolved_paths.append(resolved)
        
        return resolved_paths
    
    def get_project_from_path(self, file_path: str) -> Optional[str]:
        """Extract project name from a resolved file path"""
        path = Path(file_path)
        
        # Remove workspace root to get relative path
        try:
            relative = path.relative_to(self.workspace_root)
            return relative.parts[0] if relative.parts else None
        except ValueError:
            return None


# Test the PathResolver
if __name__ == "__main__":
    import asyncio
    
    async def test_path_resolver():
        print("=" * 60)
        print("TESTING PATH RESOLVER")
        print("=" * 60)
        
        resolver = PathResolver()
        
        test_cases = [
            "@infinite-kanvas canvas-utils.ts",
            "canvas-utils.ts",
            "infinite-kanvas/src/utils/canvas-utils.ts",
            "/workspace/infinite-kanvas/src/utils/canvas-utils.ts",
            "utils.ts",  # Should find multiple matches
        ]
        
        for test_case in test_cases:
            resolved, exists = resolver.resolve_file_path(test_case, project_context="infinite-kanvas")
            print(f"Input: '{test_case}'")
            print(f"Resolved: '{resolved}'")
            print(f"Exists: {exists}")
            print("-" * 40)
    
    asyncio.run(test_path_resolver())