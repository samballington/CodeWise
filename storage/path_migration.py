"""
Path Migration Utilities

Safe database migration tools for fixing inconsistent file paths in Knowledge Graph.
Ensures filesystem navigator compatibility without data loss.

REQ-PATH-001.4: Database Migration Strategy implementation
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import shutil

from storage.path_manager import PathManager, normalize_path

logger = logging.getLogger(__name__)


@dataclass
class PathInconsistency:
    """Represents a path inconsistency found in the database."""
    table: str
    row_id: str
    column: str
    current_path: str
    corrected_path: str
    project_name: Optional[str]
    severity: str  # 'critical', 'warning', 'info'
    description: str


@dataclass
class MigrationResult:
    """Results of a path migration operation."""
    success: bool
    total_checked: int
    inconsistencies_found: int
    paths_migrated: int
    errors: List[str]
    warnings: List[str]
    backup_path: Optional[str]
    duration_seconds: float


class PathMigrationAnalyzer:
    """Analyzes database for path consistency issues."""
    
    def __init__(self, db_path: str, workspace_root: str = "/workspace"):
        """
        Initialize migration analyzer.
        
        Args:
            db_path: Path to SQLite database
            workspace_root: Root directory for workspace
        """
        self.db_path = db_path
        self.path_manager = PathManager(workspace_root)
        self.inconsistencies = []
    
    def analyze_path_consistency(self) -> Dict[str, Any]:
        """
        Analyze all paths in the database for consistency issues.
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting path consistency analysis")
        start_time = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Analyze nodes table
            nodes_issues = self._analyze_nodes_table(conn)
            
            # Analyze chunks table  
            chunks_issues = self._analyze_chunks_table(conn)
            
            # Analyze edges table (if exists)
            edges_issues = self._analyze_edges_table(conn)
            
            # Generate summary
            total_issues = len(nodes_issues) + len(chunks_issues) + len(edges_issues)
            
            # Categorize by severity
            critical_issues = [i for i in self.inconsistencies if i.severity == 'critical']
            warning_issues = [i for i in self.inconsistencies if i.severity == 'warning']
            
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "database_path": self.db_path,
                "analysis_timestamp": start_time.isoformat(),
                "duration_seconds": analysis_duration,
                "total_inconsistencies": total_issues,
                "critical_issues": len(critical_issues),
                "warning_issues": len(warning_issues),
                "tables_analyzed": {
                    "nodes": len(nodes_issues),
                    "chunks": len(chunks_issues),
                    "edges": len(edges_issues)
                },
                "inconsistencies": [self._inconsistency_to_dict(i) for i in self.inconsistencies],
                "recommendations": self._generate_recommendations()
            }
            
        finally:
            conn.close()
    
    def _analyze_nodes_table(self, conn: sqlite3.Connection) -> List[PathInconsistency]:
        """Analyze nodes table for path inconsistencies."""
        logger.info("Analyzing nodes table paths")
        issues = []
        
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path FROM nodes WHERE file_path IS NOT NULL")
        
        for row_id, file_path in cursor.fetchall():
            if not file_path:
                continue
                
            # Extract project name from file path, use default if not found or invalid
            project_name = self.path_manager.extract_project_name(file_path)
            if not project_name or project_name in ['src', 'frontend', 'backend', 'lib', 'utils', 'components', 'pages', 'styles', 'public']:
                # For migration, infer project name from context or use default
                project_name = self._infer_project_name_from_context(conn, file_path) or "unknown-project"
            
            # Generate corrected path using migration-specific logic
            corrected_path = self._normalize_path_for_migration(file_path, project_name)
            
            # Check if correction is needed
            if file_path != corrected_path:
                severity = self._determine_severity(file_path, corrected_path)
                
                inconsistency = PathInconsistency(
                    table="nodes",
                    row_id=str(row_id),
                    column="file_path",
                    current_path=file_path,
                    corrected_path=corrected_path,
                    project_name=project_name,
                    severity=severity,
                    description=f"Node path needs normalization: {file_path} -> {corrected_path}"
                )
                
                issues.append(inconsistency)
                self.inconsistencies.append(inconsistency)
        
        logger.info(f"Found {len(issues)} path issues in nodes table")
        return issues
    
    def _analyze_chunks_table(self, conn: sqlite3.Connection) -> List[PathInconsistency]:
        """Analyze chunks table for path inconsistencies."""
        logger.info("Analyzing chunks table paths")
        issues = []
        
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_path FROM chunks WHERE file_path IS NOT NULL")
        
        for row_id, file_path in cursor.fetchall():
            if not file_path:
                continue
                
            project_name = self.path_manager.extract_project_name(file_path)
            if not project_name or project_name in ['src', 'frontend', 'backend', 'lib', 'utils', 'components', 'pages', 'styles', 'public']:
                # For migration, infer project name from context or use default
                project_name = self._infer_project_name_from_context(conn, file_path) or "unknown-project"
            corrected_path = self._normalize_path_for_migration(file_path, project_name)
            
            if file_path != corrected_path:
                severity = self._determine_severity(file_path, corrected_path)
                
                inconsistency = PathInconsistency(
                    table="chunks",
                    row_id=str(row_id),
                    column="file_path",
                    current_path=file_path,
                    corrected_path=corrected_path,
                    project_name=project_name,
                    severity=severity,
                    description=f"Chunk path needs normalization: {file_path} -> {corrected_path}"
                )
                
                issues.append(inconsistency)
                self.inconsistencies.append(inconsistency)
        
        logger.info(f"Found {len(issues)} path issues in chunks table")
        return issues
    
    def _analyze_edges_table(self, conn: sqlite3.Connection) -> List[PathInconsistency]:
        """Analyze edges table for path inconsistencies."""
        logger.info("Analyzing edges table paths")
        issues = []
        
        try:
            cursor = conn.cursor()
            # Check if edges table has file_path column
            cursor.execute("PRAGMA table_info(edges)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'file_path' in columns:
                cursor.execute("SELECT id, file_path FROM edges WHERE file_path IS NOT NULL")
                
                for row_id, file_path in cursor.fetchall():
                    if not file_path:
                        continue
                        
                    project_name = self.path_manager.extract_project_name(file_path)
                    if not project_name or project_name in ['src', 'frontend', 'backend', 'lib', 'utils', 'components', 'pages', 'styles', 'public']:
                        # For migration, infer project name from context or use default
                        project_name = self._infer_project_name_from_context(conn, file_path) or "unknown-project"
                    corrected_path = self._normalize_path_for_migration(file_path, project_name)
                    
                    if file_path != corrected_path:
                        severity = self._determine_severity(file_path, corrected_path)
                        
                        inconsistency = PathInconsistency(
                            table="edges",
                            row_id=str(row_id),
                            column="file_path",
                            current_path=file_path,
                            corrected_path=corrected_path,
                            project_name=project_name,
                            severity=severity,
                            description=f"Edge path needs normalization: {file_path} -> {corrected_path}"
                        )
                        
                        issues.append(inconsistency)
                        self.inconsistencies.append(inconsistency)
        except sqlite3.OperationalError:
            # edges table doesn't exist
            logger.info("Edges table not found - skipping")
        
        logger.info(f"Found {len(issues)} path issues in edges table")
        return issues
    
    def _determine_severity(self, current_path: str, corrected_path: str) -> str:
        """Determine severity of a path inconsistency."""
        # Warning: Workspace absolute paths (still findable by filesystem navigator but wrong format)
        if current_path.startswith('/workspace/') and not corrected_path.startswith('/workspace/'):
            return 'warning'
        
        # Critical: Path has no project prefix - filesystem navigator can't find these
        if not self._has_project_prefix(current_path):
            # Only if it's not a workspace absolute path (those are warnings)
            if not current_path.startswith('/workspace/'):
                return 'critical'
        
        # Info: Minor formatting differences (both paths have project prefixes)
        return 'info'
    
    def _normalize_path_for_migration(self, file_path: str, project_name: Optional[str] = None) -> str:
        """
        Migration-specific path normalization that doesn't require filesystem access.
        
        Args:
            file_path: Current path from database
            project_name: Extracted or provided project name
            
        Returns:
            Normalized path for consistent storage
        """
        # Handle empty or None paths
        if not file_path:
            return file_path
        
        path_str = str(file_path).replace('\\', '/')
        
        # If it's already in correct format, return as-is
        if self._has_project_prefix(path_str):
            return path_str
        
        # If it starts with /workspace/, convert to relative
        if path_str.startswith('/workspace/'):
            workspace_relative = path_str.replace('/workspace/', '', 1)
            return workspace_relative
        
        # If it's a relative path without project prefix, add project prefix
        if project_name and not path_str.startswith('/') and not self._has_project_prefix(path_str):
            # Extract project name from path if not provided
            if not project_name:
                # Use first directory component as project name if it looks like one
                parts = path_str.split('/')
                if len(parts) > 1 and parts[0] not in ['src', 'frontend', 'backend', 'lib', 'utils']:
                    project_name = parts[0]
                else:
                    project_name = 'unknown-project'
            
            return f"{project_name}/{path_str}"
        
        # For absolute paths outside workspace, extract filename
        if path_str.startswith('/') and project_name:
            from pathlib import Path
            filename = Path(path_str).name
            return f"{project_name}/{filename}"
        
        return path_str
    
    def _has_project_prefix(self, path: str) -> bool:
        """Check if path has project prefix format."""
        if not path or path.startswith('/'):
            return False
        
        parts = path.split('/')
        if len(parts) < 2:
            return False
        
        # First part should be project name (not common directory names)
        first_part = parts[0]
        common_dirs = {'src', 'frontend', 'backend', 'lib', 'utils', 'components', 'pages', 'styles', 'public'}
        
        # Project prefix exists if first part is not a common directory name
        return first_part not in common_dirs and first_part != ''
    
    def _infer_project_name_from_context(self, conn: sqlite3.Connection, file_path: str) -> Optional[str]:
        """
        Infer project name from database context when path doesn't have project prefix.
        
        Looks for other files in the database that do have project prefixes and tries
        to determine which project this file likely belongs to.
        """
        cursor = conn.cursor()
        
        # Look for existing files with project prefixes in same directory structure
        path_parts = file_path.replace('\\', '/').split('/')
        
        # Try to find similar paths with project prefixes
        if len(path_parts) > 1:
            # Look for files with same subdirectory structure
            pattern = f"%/{'/'.join(path_parts)}"
            cursor.execute(
                "SELECT file_path FROM nodes WHERE file_path LIKE ? AND file_path != ? LIMIT 5",
                (pattern, file_path)
            )
            results = cursor.fetchall()
            
            for (existing_path,) in results:
                if self._has_project_prefix(existing_path):
                    # Extract project name from the similar path
                    project_name = existing_path.split('/')[0]
                    if project_name and project_name not in ['src', 'frontend', 'backend', 'lib', 'utils', 'components', 'pages', 'styles', 'public']:
                        return project_name
        
        # Fallback: look for most common project prefix in database
        cursor.execute("""
            SELECT file_path FROM nodes 
            WHERE file_path LIKE '%/%' 
            AND file_path NOT LIKE '/workspace/%'
            LIMIT 10
        """)
        
        project_counts = {}
        for (path,) in cursor.fetchall():
            if self._has_project_prefix(path):
                project_name = path.split('/')[0]
                if project_name not in ['src', 'frontend', 'backend', 'lib', 'utils', 'components', 'pages', 'styles', 'public']:
                    project_counts[project_name] = project_counts.get(project_name, 0) + 1
        
        if project_counts:
            # Return most common project name
            return max(project_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _inconsistency_to_dict(self, inconsistency: PathInconsistency) -> Dict[str, Any]:
        """Convert PathInconsistency to dictionary."""
        return {
            "table": inconsistency.table,
            "row_id": inconsistency.row_id,
            "column": inconsistency.column,
            "current_path": inconsistency.current_path,
            "corrected_path": inconsistency.corrected_path,
            "project_name": inconsistency.project_name,
            "severity": inconsistency.severity,
            "description": inconsistency.description
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate migration recommendations based on analysis."""
        recommendations = []
        
        critical_count = len([i for i in self.inconsistencies if i.severity == 'critical'])
        warning_count = len([i for i in self.inconsistencies if i.severity == 'warning'])
        
        if critical_count > 0:
            recommendations.append(f"URGENT: {critical_count} critical path issues found that will break filesystem navigation")
        
        if warning_count > 0:
            recommendations.append(f"WARNING: {warning_count} path format issues found")
        
        if self.inconsistencies:
            recommendations.append("Run path migration to fix all inconsistencies")
            recommendations.append("Create database backup before running migration")
        else:
            recommendations.append("No path inconsistencies found - database is healthy")
        
        return recommendations


class PathDatabaseMigrator:
    """Executes safe database migrations to fix path inconsistencies."""
    
    def __init__(self, db_path: str, workspace_root: str = "/workspace"):
        """
        Initialize database migrator.
        
        Args:
            db_path: Path to SQLite database
            workspace_root: Root directory for workspace
        """
        self.db_path = db_path
        self.path_manager = PathManager(workspace_root)
        self.backup_path = None
    
    def create_backup(self) -> str:
        """
        Create backup of database before migration.
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.backup_{timestamp}"
        
        logger.info(f"Creating database backup: {backup_path}")
        shutil.copy2(self.db_path, backup_path)
        
        self.backup_path = backup_path
        return backup_path
    
    def migrate_paths(self, inconsistencies: List[PathInconsistency], 
                     dry_run: bool = False) -> MigrationResult:
        """
        Migrate paths to fix inconsistencies.
        
        Args:
            inconsistencies: List of inconsistencies to fix
            dry_run: If True, don't actually modify database
            
        Returns:
            Migration results
        """
        logger.info(f"Starting path migration (dry_run={dry_run})")
        start_time = datetime.now()
        
        if not dry_run:
            backup_path = self.create_backup()
        else:
            backup_path = None
        
        conn = sqlite3.connect(self.db_path)
        errors = []
        warnings = []
        paths_migrated = 0
        
        try:
            conn.execute("BEGIN TRANSACTION")
            
            for inconsistency in inconsistencies:
                try:
                    if not dry_run:
                        self._apply_path_fix(conn, inconsistency)
                    paths_migrated += 1
                    logger.debug(f"Fixed path: {inconsistency.current_path} -> {inconsistency.corrected_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to migrate {inconsistency.table}.{inconsistency.row_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            if not dry_run:
                conn.execute("COMMIT")
                logger.info("Migration committed successfully")
            else:
                conn.execute("ROLLBACK")
                logger.info("Dry run completed - no changes made")
                
        except Exception as e:
            conn.execute("ROLLBACK")
            error_msg = f"Migration failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            
        finally:
            conn.close()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return MigrationResult(
            success=len(errors) == 0,
            total_checked=len(inconsistencies),
            inconsistencies_found=len(inconsistencies),
            paths_migrated=paths_migrated if not dry_run else 0,
            errors=errors,
            warnings=warnings,
            backup_path=backup_path,
            duration_seconds=duration
        )
    
    def _apply_path_fix(self, conn: sqlite3.Connection, inconsistency: PathInconsistency):
        """Apply a single path fix to the database."""
        cursor = conn.cursor()
        
        update_sql = f"""
        UPDATE {inconsistency.table} 
        SET {inconsistency.column} = ? 
        WHERE id = ?
        """
        
        cursor.execute(update_sql, (inconsistency.corrected_path, inconsistency.row_id))
        
        if cursor.rowcount == 0:
            raise ValueError(f"No rows updated for {inconsistency.table}.{inconsistency.row_id}")
    
    def rollback_migration(self, backup_path: str) -> bool:
        """
        Rollback migration by restoring from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if rollback successful
        """
        try:
            logger.info(f"Rolling back database from backup: {backup_path}")
            shutil.copy2(backup_path, self.db_path)
            logger.info("Database rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False


def run_path_analysis(db_path: str, output_file: str = None) -> Dict[str, Any]:
    """
    Run complete path consistency analysis.
    
    Args:
        db_path: Path to database
        output_file: Optional file to save analysis results
        
    Returns:
        Analysis results dictionary
    """
    analyzer = PathMigrationAnalyzer(db_path)
    results = analyzer.analyze_path_consistency()
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Analysis results saved to {output_file}")
    
    return results


def run_path_migration(db_path: str, dry_run: bool = True) -> MigrationResult:
    """
    Run complete path migration process.
    
    Runs iteratively until no more inconsistencies are found, as some
    migrations may create intermediate states that need further correction.
    
    Args:
        db_path: Path to database
        dry_run: If True, don't actually modify database
        
    Returns:
        Migration results (consolidated from all iterations)
    """
    migrator = PathDatabaseMigrator(db_path)
    
    # For dry run, just do one iteration 
    if dry_run:
        analyzer = PathMigrationAnalyzer(db_path)
        analysis = analyzer.analyze_path_consistency()
        return migrator.migrate_paths(analyzer.inconsistencies, dry_run=True)
    
    # For real migration, run iteratively
    total_paths_migrated = 0
    all_errors = []
    all_warnings = []
    backup_path = None
    iteration = 0
    max_iterations = 5
    final_success = True
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Migration iteration {iteration}")
        
        # Analyze current state
        analyzer = PathMigrationAnalyzer(db_path)
        analysis = analyzer.analyze_path_consistency()
        
        if not analyzer.inconsistencies:
            logger.info(f"No more inconsistencies found after {iteration-1} iterations")
            break
        
        logger.info(f"Found {len(analyzer.inconsistencies)} inconsistencies in iteration {iteration}")
        
        # Migrate this iteration
        result = migrator.migrate_paths(analyzer.inconsistencies, dry_run=False)
        
        # Track backup path from first iteration
        if iteration == 1:
            backup_path = result.backup_path
        
        # Accumulate results
        total_paths_migrated += result.paths_migrated
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
        
        if not result.success:
            final_success = False
            logger.error(f"Migration iteration {iteration} failed")
            break
    
    if iteration >= max_iterations:
        warning_msg = f"Migration stopped after {max_iterations} iterations - may need manual review"
        all_warnings.append(warning_msg)
        logger.warning(warning_msg)
    
    # Create consolidated result
    duration = sum(getattr(result, 'duration_seconds', 0) for result in [result])
    
    return MigrationResult(
        success=final_success and len(all_errors) == 0,
        total_checked=total_paths_migrated,  # Use migrated paths as total checked 
        inconsistencies_found=total_paths_migrated,  # All migrated paths were inconsistencies
        paths_migrated=total_paths_migrated,
        errors=all_errors,
        warnings=all_warnings,
        backup_path=backup_path,
        duration_seconds=duration
    )


if __name__ == "__main__":
    # CLI interface for path migration
    import argparse
    
    parser = argparse.ArgumentParser(description="Path Migration Utilities")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--analyze", action="store_true", help="Analyze path consistency")
    parser.add_argument("--migrate", action="store_true", help="Run path migration")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't modify database)")
    parser.add_argument("--output", help="Output file for analysis results")
    parser.add_argument("--workspace", default="/workspace", help="Workspace root path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.analyze:
        print("Running path consistency analysis...")
        results = run_path_analysis(args.db_path, args.output)
        
        print(f"\\nAnalysis Results:")
        print(f"Total inconsistencies: {results['total_inconsistencies']}")
        print(f"Critical issues: {results['critical_issues']}")
        print(f"Warning issues: {results['warning_issues']}")
        print(f"Duration: {results['duration_seconds']:.2f}s")
        
        print("\\nRecommendations:")
        for rec in results['recommendations']:
            print(f"- {rec}")
    
    if args.migrate:
        print(f"Running path migration (dry_run={args.dry_run})...")
        result = run_path_migration(args.db_path, dry_run=args.dry_run)
        
        print(f"\\nMigration Results:")
        print(f"Success: {result.success}")
        print(f"Paths migrated: {result.paths_migrated}")
        print(f"Errors: {len(result.errors)}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        
        if result.backup_path:
            print(f"Backup created: {result.backup_path}")
        
        if result.errors:
            print("\\nErrors:")
            for error in result.errors:
                print(f"- {error}")