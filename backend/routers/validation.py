"""
Path Validation API Endpoints

Provides REST endpoints for validating and managing path consistency
in the Knowledge Graph database.

REQ-PATH-001.5: Path validation endpoints for monitoring and debugging
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Import path validation components
try:
    from storage.database_manager import DatabaseManager
    from storage.path_migration import run_path_analysis, run_path_migration, MigrationResult
    from backend.tools.filesystem_navigator import FilesystemNavigator
    from storage.path_manager import get_path_manager
    PATH_COMPONENTS_AVAILABLE = True
except ImportError as e:
    PATH_COMPONENTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"Path validation components not available: {e}")

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/validation", tags=["path-validation"])

# Pydantic models for request/response
class PathValidationRequest(BaseModel):
    """Request model for path validation."""
    project_name: Optional[str] = None
    db_path: Optional[str] = "codewise.db"

class ProjectValidationResponse(BaseModel):
    """Response model for project validation."""
    project_name: str
    relative_path_files: int
    absolute_path_files: int
    no_prefix_files: int
    total_files: int
    path_consistency: str
    available_projects: List[str]
    recommendations: List[str]
    timestamp: str

class MigrationRequest(BaseModel):
    """Request model for database migration."""
    dry_run: bool = True
    max_paths: Optional[int] = None
    db_path: Optional[str] = "codewise.db"

class MigrationResponse(BaseModel):
    """Response model for migration results."""
    success: bool
    total_checked: int
    inconsistencies_found: int
    paths_migrated: int
    errors: List[str]
    warnings: List[str]
    backup_path: Optional[str]
    duration_seconds: float
    timestamp: str

class PathAnalysisResponse(BaseModel):
    """Response model for path analysis."""
    total_inconsistencies: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    inconsistencies: List[Dict[str, Any]]
    database_path: str
    timestamp: str

# Dependency to get database manager
def get_db_manager(db_path: str = "codewise.db") -> DatabaseManager:
    """Get DatabaseManager instance."""
    if not PATH_COMPONENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Path validation components not available")
    
    try:
        return DatabaseManager(db_path)
    except Exception as e:
        logger.error(f"Failed to initialize DatabaseManager: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@router.get("/health", summary="Check path validation service health")
async def health_check() -> Dict[str, Any]:
    """
    Check the health of path validation services.
    
    Returns:
        Health status and component availability
    """
    return {
        "status": "healthy" if PATH_COMPONENTS_AVAILABLE else "degraded",
        "components": {
            "path_manager": PATH_COMPONENTS_AVAILABLE,
            "migration_utilities": PATH_COMPONENTS_AVAILABLE,
            "filesystem_navigator": PATH_COMPONENTS_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    }

@router.post("/project/validate", response_model=ProjectValidationResponse, 
             summary="Validate path consistency for a project")
async def validate_project_paths(request: PathValidationRequest) -> ProjectValidationResponse:
    """
    Validate path consistency for a specific project.
    
    Args:
        request: Validation request with project name and database path
        
    Returns:
        Detailed validation results with recommendations
        
    Raises:
        HTTPException: If validation fails or project not found
    """
    if not request.project_name:
        raise HTTPException(status_code=400, detail="project_name is required")
    
    try:
        db_manager = get_db_manager(request.db_path)
        navigator = FilesystemNavigator(db_manager)
        
        # Perform validation
        validation_result = navigator.validate_project_paths(request.project_name)
        
        # Convert to response model
        response = ProjectValidationResponse(
            project_name=validation_result['project_name'],
            relative_path_files=validation_result['relative_path_files'],
            absolute_path_files=validation_result['absolute_path_files'],
            no_prefix_files=validation_result['no_prefix_files'],
            total_files=validation_result['total_files'],
            path_consistency=validation_result['path_consistency'],
            available_projects=validation_result['available_projects'],
            recommendations=validation_result['recommendations'],
            timestamp=datetime.now().isoformat()
        )
        
        db_manager.close()
        
        logger.info(f"Project validation completed for {request.project_name}: {response.path_consistency}")
        return response
        
    except Exception as e:
        logger.error(f"Project validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/analyze", response_model=PathAnalysisResponse,
             summary="Analyze database for path consistency issues")
async def analyze_database_paths(request: PathValidationRequest) -> PathAnalysisResponse:
    """
    Analyze the entire database for path consistency issues.
    
    Args:
        request: Analysis request with optional database path
        
    Returns:
        Comprehensive analysis of path inconsistencies
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        # Run path analysis
        analysis_result = run_path_analysis(request.db_path)
        
        response = PathAnalysisResponse(
            total_inconsistencies=analysis_result['total_inconsistencies'],
            critical_issues=analysis_result['critical_issues'],
            warning_issues=analysis_result['warning_issues'],
            info_issues=analysis_result.get('info_issues', 0),
            inconsistencies=analysis_result['inconsistencies'],
            database_path=request.db_path,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Database analysis completed: {response.total_inconsistencies} issues found")
        return response
        
    except Exception as e:
        logger.error(f"Database analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/migrate", response_model=MigrationResponse,
             summary="Run database path migration")
async def migrate_database_paths(request: MigrationRequest) -> MigrationResponse:
    """
    Run database path migration to fix inconsistencies.
    
    Args:
        request: Migration request with options
        
    Returns:
        Migration results and statistics
        
    Raises:
        HTTPException: If migration fails
    """
    try:
        # Run migration
        migration_result = run_path_migration(
            request.db_path,
            dry_run=request.dry_run
        )
        
        response = MigrationResponse(
            success=migration_result.success,
            total_checked=migration_result.total_checked,
            inconsistencies_found=migration_result.inconsistencies_found,
            paths_migrated=migration_result.paths_migrated,
            errors=migration_result.errors,
            warnings=migration_result.warnings,
            backup_path=migration_result.backup_path,
            duration_seconds=migration_result.duration_seconds,
            timestamp=datetime.now().isoformat()
        )
        
        action = "dry run" if request.dry_run else "migration"
        logger.info(f"Database {action} completed: {response.paths_migrated} paths processed")
        
        return response
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")

@router.get("/projects", summary="List all projects in database")
async def list_projects(db_path: str = "codewise.db") -> Dict[str, Any]:
    """
    List all projects found in the database with their file counts.
    
    Args:
        db_path: Path to database file
        
    Returns:
        List of projects with statistics
    """
    try:
        db_manager = get_db_manager(db_path)
        navigator = FilesystemNavigator(db_manager)
        
        # Get all projects by analyzing file paths
        cursor = db_manager.connection.cursor()
        
        projects_query = """
            SELECT 
                CASE 
                    WHEN file_path LIKE '/workspace/%' THEN 
                        SUBSTR(file_path, 12, INSTR(SUBSTR(file_path, 12), '/') - 1)
                    ELSE
                        SUBSTR(file_path, 1, INSTR(file_path, '/') - 1)
                END as project,
                COUNT(DISTINCT file_path) as file_count
            FROM nodes 
            WHERE project != '' AND project IS NOT NULL
            GROUP BY project
            ORDER BY file_count DESC
        """
        
        results = cursor.execute(projects_query).fetchall()
        
        projects = []
        for project_name, file_count in results:
            if project_name:
                # Get validation for each project
                validation = navigator.validate_project_paths(project_name)
                projects.append({
                    "name": project_name,
                    "file_count": file_count,
                    "path_consistency": validation['path_consistency'],
                    "relative_files": validation['relative_path_files'],
                    "absolute_files": validation['absolute_path_files'],
                    "no_prefix_files": validation['no_prefix_files']
                })
        
        db_manager.close()
        
        return {
            "projects": projects,
            "total_projects": len(projects),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Project listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

@router.get("/navigator/test/{project_name}", summary="Test filesystem navigator for project")
async def test_navigator_for_project(project_name: str, db_path: str = "codewise.db") -> Dict[str, Any]:
    """
    Test filesystem navigator functionality for a specific project.
    
    Args:
        project_name: Name of the project to test
        db_path: Path to database file
        
    Returns:
        Navigator test results
    """
    try:
        db_manager = get_db_manager(db_path)
        navigator = FilesystemNavigator(db_manager)
        
        # Test different navigator operations
        test_results = {}
        
        # Test 1: Validate project paths
        test_results["validation"] = navigator.validate_project_paths(project_name)
        
        # Test 2: Find files in project
        find_result = navigator.execute(
            operation="find",
            pattern="*",
            project_scope=project_name
        )
        test_results["find_all_files"] = {
            "found_files": len(find_result.get('files', [])),
            "has_error": 'error' in find_result
        }
        
        # Test 3: List root directory  
        list_result = navigator.execute(
            operation="list",
            path="",
            project_scope=project_name
        )
        test_results["list_root"] = {
            "found_files": len(list_result.get('files', [])),
            "has_error": 'error' in list_result
        }
        
        # Test 4: Validate operation
        validate_result = navigator.execute(
            operation="validate",
            project_scope=project_name
        )
        test_results["validate_operation"] = validate_result
        
        db_manager.close()
        
        return {
            "project_name": project_name,
            "test_results": test_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Navigator testing failed for {project_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Navigator test failed: {str(e)}")

# Additional utility endpoints

@router.get("/patterns/{project_name}", summary="Get search patterns for project")
async def get_project_patterns(project_name: str) -> Dict[str, Any]:
    """
    Get PathManager search patterns for a project.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Search patterns used by PathManager
    """
    try:
        if not PATH_COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="PathManager not available")
            
        path_manager = get_path_manager()
        relative_pattern, absolute_pattern = path_manager.build_search_patterns(project_name)
        
        return {
            "project_name": project_name,
            "relative_pattern": relative_pattern,
            "absolute_pattern": absolute_pattern,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pattern generation failed for {project_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern generation failed: {str(e)}")

# Debug endpoint for development
@router.get("/debug/database", summary="Get database debug information")
async def get_database_debug_info(db_path: str = "codewise.db") -> Dict[str, Any]:
    """
    Get debug information about the database structure and contents.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Debug information about database
    """
    try:
        db_manager = get_db_manager(db_path)
        cursor = db_manager.connection.cursor()
        
        # Get table info
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = cursor.execute(tables_query).fetchall()
        
        # Get sample paths from nodes table
        sample_paths_query = "SELECT DISTINCT file_path FROM nodes LIMIT 10"
        sample_paths = cursor.execute(sample_paths_query).fetchall()
        
        # Get path format statistics
        path_stats_query = """
            SELECT 
                COUNT(*) as total_nodes,
                COUNT(CASE WHEN file_path LIKE '/workspace/%' THEN 1 END) as absolute_paths,
                COUNT(CASE WHEN file_path LIKE '%/%' AND file_path NOT LIKE '/workspace/%' THEN 1 END) as relative_paths,
                COUNT(CASE WHEN file_path NOT LIKE '%/%' THEN 1 END) as single_files
            FROM nodes
        """
        path_stats = cursor.execute(path_stats_query).fetchone()
        
        db_manager.close()
        
        return {
            "database_path": db_path,
            "tables": [t[0] for t in tables],
            "sample_paths": [p[0] for p in sample_paths],
            "path_statistics": {
                "total_nodes": path_stats[0],
                "absolute_paths": path_stats[1],
                "relative_paths": path_stats[2],
                "single_files": path_stats[3]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database debug info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug info failed: {str(e)}")