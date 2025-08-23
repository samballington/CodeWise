"""
Knowledge Graph Startup Service - REQ-3.7.1

Automatically populates Knowledge Graph for all workspace projects on container startup.

Architecture:
- Runs during backend initialization (non-blocking)
- Scans /workspace for all project directories  
- Executes unified KG indexing for each project
- Provides comprehensive logging and status reporting
- Integrates with existing DatabaseManager and UnifiedIndexer

Success Criteria:
- All workspace projects indexed within 2 minutes of startup
- Zero manual intervention required
- Error isolation (one project failure doesn't block others)
- Complete KG data for symbol search, relationships, and code analysis
"""

import os
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProjectIndexingResult:
    """Results from indexing a single project."""
    project_name: str
    success: bool
    files_processed: int
    symbols_discovered: int
    relationships_found: int
    chunks_created: int
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class KGStartupResult:
    """Overall results from startup KG population."""
    total_projects: int
    successful_projects: int
    failed_projects: int
    total_processing_time: float
    project_results: List[ProjectIndexingResult]
    startup_timestamp: datetime


class KGStartupService:
    """
    Service that automatically populates Knowledge Graph on startup.
    
    REQ-3.7.1: Startup KG Population Service
    """
    
    def __init__(self, workspace_dir: str = "/workspace", db_path: str = "/app/storage/codewise.db"):
        """Initialize startup service with workspace configuration."""
        self.workspace_dir = Path(workspace_dir)
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # File patterns to index
        self.supported_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", 
            "*.java", "*.cpp", "*.c", "*.h", "*.hpp",
            "*.cs", "*.php", "*.rb", "*.go", "*.rs"
        ]
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '.vector_cache', '.discovery_cache', 'node_modules',
            '__pycache__', '.pytest_cache', 'dist', 'build', 'target',
            '.env', '.venv', 'venv', 'env'
        }
        
        self.logger.info(f"✅ KGStartupService initialized for workspace: {workspace_dir}")
    
    async def populate_all_projects(self, incremental: bool = True) -> KGStartupResult:
        """
        Main entry point - populate KG for all workspace projects.
        
        REQ-3.7.1: Automatic KG population with error isolation.
        REQ-3.9.1: Incremental indexing support
        """
        startup_start = time.time()
        mode = "incremental" if incremental else "full"
        self.logger.info(f"🚀 Starting {mode} Knowledge Graph population for all projects")
        
        # Discover all projects in workspace
        all_projects = self._discover_workspace_projects()
        self.logger.info(f"📁 Found {len(all_projects)} total projects: {[p.name for p in all_projects]}")
        
        if not all_projects:
            self.logger.warning("⚠️ No projects found in workspace - nothing to index")
            return KGStartupResult(
                total_projects=0,
                successful_projects=0,
                failed_projects=0,
                total_processing_time=time.time() - startup_start,
                project_results=[],
                startup_timestamp=datetime.now()
            )
        
        # Filter to only unindexed projects if incremental mode
        if incremental:
            indexed_projects = self._get_indexed_projects()
            projects_to_index = [p for p in all_projects if p.name not in indexed_projects]
            self.logger.info(f"📊 Incremental mode: {len(indexed_projects)} already indexed, "
                           f"{len(projects_to_index)} new projects to index")
            self.logger.info(f"🔄 New projects to index: {[p.name for p in projects_to_index]}")
        else:
            projects_to_index = all_projects
            self.logger.info(f"🔄 Full reindex mode: processing all {len(projects_to_index)} projects")
        
        if not projects_to_index:
            self.logger.info("✅ All projects already indexed - nothing to do")
            return KGStartupResult(
                total_projects=len(all_projects),
                successful_projects=len(all_projects),
                failed_projects=0,
                total_processing_time=time.time() - startup_start,
                project_results=[],
                startup_timestamp=datetime.now()
            )
        
        # Process each project (with error isolation)
        project_results = []
        successful_count = 0
        
        for project_path in projects_to_index:
            try:
                self.logger.info(f"🔄 Processing project: {project_path.name}")
                result = await self._index_single_project(project_path)
                project_results.append(result)
                
                if result.success:
                    successful_count += 1
                    self.logger.info(f"✅ {project_path.name}: {result.symbols_discovered} symbols, "
                                   f"{result.relationships_found} relationships in {result.processing_time:.2f}s")
                else:
                    self.logger.error(f"❌ {project_path.name}: {result.error_message}")
                    
            except Exception as e:
                # Error isolation - one project failure doesn't stop others
                self.logger.error(f"❌ Unexpected error processing {project_path.name}: {e}")
                project_results.append(ProjectIndexingResult(
                    project_name=project_path.name,
                    success=False,
                    files_processed=0,
                    symbols_discovered=0,
                    relationships_found=0,
                    chunks_created=0,
                    processing_time=0.0,
                    error_message=str(e)
                ))
        
        total_time = time.time() - startup_start
        failed_count = len(projects_to_index) - successful_count
        
        # Create summary result
        result = KGStartupResult(
            total_projects=len(all_projects),
            successful_projects=successful_count,
            failed_projects=failed_count,
            total_processing_time=total_time,
            project_results=project_results,
            startup_timestamp=datetime.now()
        )
        
        # Log final summary
        if incremental:
            self.logger.info(f"🎉 Incremental KG indexing completed: {successful_count}/{len(projects_to_index)} new projects indexed "
                           f"in {total_time:.2f}s")
        else:
            self.logger.info(f"🎉 Full KG reindexing completed: {successful_count}/{len(projects_to_index)} projects successful "
                           f"in {total_time:.2f}s")
        
        if failed_count > 0:
            self.logger.warning(f"⚠️ {failed_count} projects failed - check logs for details")
        
        return result
    
    def _discover_workspace_projects(self) -> List[Path]:
        """
        Discover all project directories in workspace.
        
        REQ-3.7.1: Scan /workspace for all project directories
        """
        projects = []
        
        if not self.workspace_dir.exists():
            self.logger.error(f"❌ Workspace directory not found: {self.workspace_dir}")
            return projects
        
        try:
            for item in self.workspace_dir.iterdir():
                if item.is_dir() and item.name not in self.skip_dirs:
                    # Check if directory contains code files
                    if self._has_code_files(item):
                        projects.append(item)
                        self.logger.debug(f"📂 Found project: {item.name}")
                    else:
                        self.logger.debug(f"⏭️ Skipping {item.name}: no code files found")
        
        except Exception as e:
            self.logger.error(f"❌ Error scanning workspace: {e}")
        
        return projects
    
    def _get_indexed_projects(self) -> set:
        """
        Get list of projects already indexed in Knowledge Graph.
        
        REQ-3.9.1: Determine which projects are already indexed
        """
        try:
            from storage.database_manager import DatabaseManager
            
            db = DatabaseManager()
            cursor = db.connection.cursor()
            
            # Get all file paths and extract project names
            query = "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE '/workspace/%'"
            results = cursor.execute(query).fetchall()
            
            indexed_projects = set()
            for row in results:
                file_path = row[0]
                # Extract project name from /workspace/ProjectName/...
                parts = file_path.split('/')
                if len(parts) >= 3 and parts[1] == 'workspace':
                    project_name = parts[2]
                    indexed_projects.add(project_name)
            
            self.logger.debug(f"🔍 Found {len(indexed_projects)} indexed projects: {sorted(indexed_projects)}")
            return indexed_projects
            
        except Exception as e:
            self.logger.error(f"❌ Error getting indexed projects: {e}")
            return set()
    
    def _has_code_files(self, project_path: Path) -> bool:
        """Check if directory contains code files worth indexing."""
        try:
            # Quick scan for supported file types
            for pattern in self.supported_patterns:
                if list(project_path.rglob(pattern)):
                    return True
            return False
        except Exception:
            return False
    
    async def _index_single_project(self, project_path: Path) -> ProjectIndexingResult:
        """
        Index a single project using UnifiedIndexer.
        
        REQ-3.7.1: Execute unified KG indexing for each project
        REQ-3.7.3: Graceful error handling
        """
        project_start = time.time()
        project_name = project_path.name
        
        try:
            # Import unified indexer (lazy import to avoid startup dependencies)
            import sys
            
            # Ensure path modifications are persistent and unique
            kg_path = Path("/app/knowledge_graph")
            storage_path = Path("/app/storage")
            
            # Add paths only if not already present
            if str(kg_path) not in sys.path:
                sys.path.insert(0, str(kg_path))
                self.logger.info(f"🔧 Added to sys.path: {kg_path}")
            
            if str(storage_path) not in sys.path:
                sys.path.insert(0, str(storage_path))
                self.logger.info(f"🔧 Added to sys.path: {storage_path}")
            
            # Verify paths before import
            relevant_paths = [p for p in sys.path if 'knowledge_graph' in p or 'storage' in p]
            self.logger.info(f"🔍 Current sys.path includes: {relevant_paths}")
            
            # Check if files actually exist
            unified_indexer_path = kg_path / "unified_indexer.py"
            self.logger.info(f"🔍 Checking unified_indexer.py exists: {unified_indexer_path.exists()}")
            
            # Try import with detailed error reporting
            try:
                from unified_indexer import UnifiedIndexer
                self.logger.info("✅ UnifiedIndexer import successful")
            except ImportError as import_err:
                self.logger.error(f"❌ UnifiedIndexer import failed: {import_err}")
                self.logger.error(f"Available sys.path: {sys.path[:5]}")
                self.logger.error(f"Working directory: {os.getcwd()}")
                raise
            
            # Create indexer instance
            indexer = UnifiedIndexer(db_path=self.db_path)
            
            # Execute indexing - force reindex on startup to ensure KG is populated
            self.logger.info(f"🔄 Starting indexing of {project_path.name} with Universal Dependency Indexer")
            result = await indexer.index_codebase(
                codebase_path=project_path,
                force_reindex=True,  # Force reindex during startup to ensure KG population
                file_patterns=self.supported_patterns
            )
            self.logger.info(f"📊 Indexing completed for {project_path.name}: {result}")
            
            # Clean up
            indexer.close()
            
            processing_time = time.time() - project_start
            
            # Return successful result
            return ProjectIndexingResult(
                project_name=project_name,
                success=result.success,
                files_processed=result.files_processed,
                symbols_discovered=result.symbols_discovered,
                relationships_found=result.relationships_found,
                chunks_created=result.chunks_created,
                processing_time=processing_time,
                error_message=None if result.success else "; ".join(result.error_details)
            )
            
        except Exception as e:
            processing_time = time.time() - project_start
            error_msg = f"Indexing failed: {str(e)}"
            
            self.logger.error(f"❌ {project_name} indexing error: {error_msg}")
            
            return ProjectIndexingResult(
                project_name=project_name,
                success=False,
                files_processed=0,
                symbols_discovered=0,
                relationships_found=0,
                chunks_created=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def get_kg_status(self) -> Dict[str, Any]:
        """
        Get current Knowledge Graph status for all projects.
        
        REQ-3.7.4: Status tracking per project
        """
        try:
            from storage.database_manager import DatabaseManager
            
            kg = DatabaseManager()
            all_nodes = kg.get_all_nodes()
            
            # Analyze project distribution
            project_stats = {}
            total_nodes = len(all_nodes)
            
            for node in all_nodes:
                file_path = str(node.get('file_path', '')) + str(node.get('path', ''))
                
                # Extract project name from path
                project = None
                for part in file_path.split('/'):
                    if part and not part.startswith('.') and part != 'workspace':
                        project = part
                        break
                
                if project:
                    project_stats[project] = project_stats.get(project, 0) + 1
            
            return {
                "total_nodes": total_nodes,
                "projects_indexed": len(project_stats),
                "project_distribution": project_stats,
                "status": "operational" if total_nodes > 0 else "empty"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting KG status: {e}")
            return {
                "total_nodes": 0,
                "projects_indexed": 0,
                "project_distribution": {},
                "status": "error",
                "error": str(e)
            }


# Global service instance
_startup_service = None


def get_kg_startup_service() -> KGStartupService:
    """Get singleton startup service instance."""
    global _startup_service
    if _startup_service is None:
        _startup_service = KGStartupService()
    return _startup_service


async def run_startup_kg_population() -> KGStartupResult:
    """
    Convenience function to run KG population.
    
    REQ-3.7.1: Main entry point for backend startup integration.
    """
    service = get_kg_startup_service()
    return await service.populate_all_projects()


# Export public interface
__all__ = [
    "KGStartupService", 
    "KGStartupResult", 
    "ProjectIndexingResult",
    "get_kg_startup_service", 
    "run_startup_kg_population"
]