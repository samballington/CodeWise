"""
Unit Tests for Path Migration Utilities

Tests database migration functionality for path consistency fixes.
REQ-PATH-001.4: Database Migration Strategy testing
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
import json
import sys
import os

# Add parent directory to path to import storage modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.path_migration import (
    PathMigrationAnalyzer, PathDatabaseMigrator, PathInconsistency,
    MigrationResult, run_path_analysis, run_path_migration
)
from storage.path_manager import PathManager


class TestPathMigrationAnalyzer:
    """Test suite for PathMigrationAnalyzer class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database with test data."""
        db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = db_file.name
        db_file.close()
        
        # Create database with test schema
        conn = sqlite3.connect(db_path)
        
        # Create nodes table
        conn.execute('''
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                signature TEXT,
                docstring TEXT,
                properties JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chunks table
        conn.execute('''
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_type TEXT,
                line_start INTEGER,
                line_end INTEGER,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert test data with path inconsistencies
        test_nodes = [
            # Correct format
            ('node1', 'function', 'test_func', 'iiot-monitoring/src/app.py', 1, 10, None, None, '{}'),
            # Missing project prefix - CRITICAL
            ('node2', 'class', 'TestClass', 'src/components/Button.tsx', 5, 20, None, None, '{}'),
            # Absolute workspace path - WARNING  
            ('node3', 'variable', 'CONFIG', '/workspace/backend/config.py', 1, 1, None, None, '{}'),
            # Mixed format
            ('node4', 'import', 'requests', 'frontend/utils/api.js', 1, 1, None, None, '{}'),
        ]
        
        for node_data in test_nodes:
            conn.execute(
                'INSERT INTO nodes (id, type, name, file_path, line_start, line_end, signature, docstring, properties) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                node_data
            )
        
        test_chunks = [
            ('chunk1', 'Test content 1', 'iiot-monitoring/backend/main.py', 'function', 1, 20, '{}'),
            ('chunk2', 'Test content 2', 'backend/controllers/auth.py', 'class', 25, 50, '{}'),  # Missing prefix
            ('chunk3', 'Test content 3', '/workspace/frontend/src/index.js', 'module', 1, 100, '{}'),  # Absolute
        ]
        
        for chunk_data in test_chunks:
            conn.execute(
                'INSERT INTO chunks (id, content, file_path, chunk_type, line_start, line_end, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)',
                chunk_data
            )
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    def test_analyzer_initialization(self, temp_db):
        """Test PathMigrationAnalyzer initialization."""
        analyzer = PathMigrationAnalyzer(temp_db)
        assert analyzer.db_path == temp_db
        assert isinstance(analyzer.path_manager, PathManager)
        assert analyzer.inconsistencies == []
    
    def test_analyze_path_consistency(self, temp_db):
        """Test complete path consistency analysis."""
        analyzer = PathMigrationAnalyzer(temp_db)
        results = analyzer.analyze_path_consistency()
        
        # Validate result structure
        assert 'database_path' in results
        assert 'analysis_timestamp' in results
        assert 'total_inconsistencies' in results
        assert 'critical_issues' in results
        assert 'warning_issues' in results
        assert 'tables_analyzed' in results
        assert 'inconsistencies' in results
        assert 'recommendations' in results
        
        # Should find inconsistencies in test data
        assert results['total_inconsistencies'] > 0
        assert results['critical_issues'] > 0  # Missing project prefixes
        assert results['warning_issues'] > 0   # Absolute paths
        
        # Check tables analyzed
        assert 'nodes' in results['tables_analyzed']
        assert 'chunks' in results['tables_analyzed']
        
        # Verify recommendations
        assert len(results['recommendations']) > 0
        assert any('URGENT' in rec for rec in results['recommendations'])
    
    def test_analyze_nodes_table(self, temp_db):
        """Test nodes table analysis."""
        analyzer = PathMigrationAnalyzer(temp_db)
        
        conn = sqlite3.connect(temp_db)
        issues = analyzer._analyze_nodes_table(conn)
        conn.close()
        
        # Should find issues with nodes 2, 3, 4 (not node1)
        assert len(issues) >= 3
        
        # Check specific issues
        issue_paths = [issue.current_path for issue in issues]
        assert 'src/components/Button.tsx' in issue_paths  # Missing prefix
        assert '/workspace/backend/config.py' in issue_paths  # Absolute path
        assert 'frontend/utils/api.js' in issue_paths  # Missing prefix
    
    def test_analyze_chunks_table(self, temp_db):
        """Test chunks table analysis."""
        analyzer = PathMigrationAnalyzer(temp_db)
        
        conn = sqlite3.connect(temp_db)
        issues = analyzer._analyze_chunks_table(conn)
        conn.close()
        
        # Should find issues with chunks 2, 3 (not chunk1)
        assert len(issues) >= 2
        
        issue_paths = [issue.current_path for issue in issues]
        assert 'backend/controllers/auth.py' in issue_paths  # Missing prefix
        assert '/workspace/frontend/src/index.js' in issue_paths  # Absolute path
    
    def test_severity_determination(self, temp_db):
        """Test severity classification logic."""
        analyzer = PathMigrationAnalyzer(temp_db)
        
        # Critical: no project prefix -> has project prefix
        severity = analyzer._determine_severity('src/app.js', 'project/src/app.js')
        assert severity == 'critical'
        
        # Warning: absolute workspace -> relative
        severity = analyzer._determine_severity('/workspace/project/src/app.js', 'project/src/app.js')
        assert severity == 'warning'
        
        # Info: minor changes (backslash to forward slash)
        severity = analyzer._determine_severity('project/src/app.js', 'project/src/app.js')
        assert severity == 'info'
    
    def test_has_project_prefix(self, temp_db):
        """Test project prefix detection."""
        analyzer = PathMigrationAnalyzer(temp_db)
        
        # Has project prefix
        assert analyzer._has_project_prefix('project-name/src/app.js') == True
        assert analyzer._has_project_prefix('iiot-monitoring/backend/main.py') == True
        
        # No project prefix (starts with common directory names)
        assert analyzer._has_project_prefix('src/app.js') == False
        assert analyzer._has_project_prefix('frontend/components/Button.tsx') == False
        assert analyzer._has_project_prefix('/workspace/project/app.js') == False
        assert analyzer._has_project_prefix('/absolute/path/app.js') == False
    
    def test_inconsistency_to_dict(self, temp_db):
        """Test inconsistency serialization."""
        analyzer = PathMigrationAnalyzer(temp_db)
        
        inconsistency = PathInconsistency(
            table='nodes',
            row_id='test_id',
            column='file_path',
            current_path='src/app.js',
            corrected_path='project/src/app.js',
            project_name='project',
            severity='critical',
            description='Test inconsistency'
        )
        
        result_dict = analyzer._inconsistency_to_dict(inconsistency)
        
        assert result_dict['table'] == 'nodes'
        assert result_dict['current_path'] == 'src/app.js'
        assert result_dict['corrected_path'] == 'project/src/app.js'
        assert result_dict['severity'] == 'critical'


class TestPathDatabaseMigrator:
    """Test suite for PathDatabaseMigrator class."""
    
    @pytest.fixture
    def temp_db_with_issues(self):
        """Create temporary database with path issues for migration testing."""
        db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = db_file.name
        db_file.close()
        
        conn = sqlite3.connect(db_path)
        
        # Create test tables
        conn.execute('''
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                content TEXT,
                chunk_type TEXT
            )
        ''')
        
        # Insert data with path issues
        test_data = [
            ('node1', 'src/components/Button.tsx'),  # Needs prefix
            ('node2', '/workspace/project/main.py'),  # Needs normalization
            ('node3', 'correct-project/src/app.js'),  # Already correct
        ]
        
        for node_id, file_path in test_data:
            conn.execute('INSERT INTO nodes (id, file_path) VALUES (?, ?)', (node_id, file_path))
        
        conn.commit()
        conn.close()
        
        yield db_path
        os.unlink(db_path)
    
    def test_migrator_initialization(self, temp_db_with_issues):
        """Test PathDatabaseMigrator initialization."""
        migrator = PathDatabaseMigrator(temp_db_with_issues)
        assert migrator.db_path == temp_db_with_issues
        assert isinstance(migrator.path_manager, PathManager)
        assert migrator.backup_path is None
    
    def test_create_backup(self, temp_db_with_issues):
        """Test database backup creation."""
        migrator = PathDatabaseMigrator(temp_db_with_issues)
        
        backup_path = migrator.create_backup()
        
        # Verify backup was created
        assert os.path.exists(backup_path)
        assert backup_path.startswith(temp_db_with_issues + '.backup_')
        assert migrator.backup_path == backup_path
        
        # Verify backup contents
        original_size = os.path.getsize(temp_db_with_issues)
        backup_size = os.path.getsize(backup_path)
        assert original_size == backup_size
        
        # Cleanup
        os.unlink(backup_path)
    
    def test_migrate_paths_dry_run(self, temp_db_with_issues):
        """Test path migration dry run."""
        # First analyze to get inconsistencies
        analyzer = PathMigrationAnalyzer(temp_db_with_issues)
        analyzer.analyze_path_consistency()
        
        # Run dry migration
        migrator = PathDatabaseMigrator(temp_db_with_issues)
        result = migrator.migrate_paths(analyzer.inconsistencies, dry_run=True)
        
        # Validate result
        assert isinstance(result, MigrationResult)
        assert result.total_checked > 0
        assert result.inconsistencies_found > 0
        assert result.paths_migrated == 0  # Dry run shouldn't migrate
        assert result.backup_path is None  # No backup for dry run
        assert result.duration_seconds >= 0
        
        # Verify database unchanged
        conn = sqlite3.connect(temp_db_with_issues)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM nodes WHERE id = 'node1'")
        path = cursor.fetchone()[0]
        conn.close()
        
        # Should still have original path
        assert path == 'src/components/Button.tsx'
    
    def test_migrate_paths_real_migration(self, temp_db_with_issues):
        """Test real path migration."""
        # Analyze first
        analyzer = PathMigrationAnalyzer(temp_db_with_issues)
        analyzer.analyze_path_consistency()
        
        inconsistencies_count = len(analyzer.inconsistencies)
        
        # Run real migration
        migrator = PathDatabaseMigrator(temp_db_with_issues)
        result = migrator.migrate_paths(analyzer.inconsistencies, dry_run=False)
        
        # Validate result
        assert result.success == True
        assert result.paths_migrated == inconsistencies_count
        assert result.backup_path is not None
        assert os.path.exists(result.backup_path)
        
        # Verify database was changed
        conn = sqlite3.connect(temp_db_with_issues)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM nodes ORDER BY id")
        paths = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Paths should be corrected (exact values depend on PathManager logic)
        for path in paths:
            # All paths should now have proper format
            if '/' in path:
                parts = path.split('/')
                # Should not start with /workspace
                assert not path.startswith('/workspace/')
                # Should have project prefix for multi-part paths
                assert len(parts) >= 2
        
        # Cleanup backup
        os.unlink(result.backup_path)
    
    def test_rollback_migration(self, temp_db_with_issues):
        """Test migration rollback functionality."""
        # Create backup
        migrator = PathDatabaseMigrator(temp_db_with_issues)
        backup_path = migrator.create_backup()
        
        # Modify database
        conn = sqlite3.connect(temp_db_with_issues)
        conn.execute("UPDATE nodes SET file_path = 'modified/path.py' WHERE id = 'node1'")
        conn.commit()
        conn.close()
        
        # Verify modification
        conn = sqlite3.connect(temp_db_with_issues)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM nodes WHERE id = 'node1'")
        modified_path = cursor.fetchone()[0]
        conn.close()
        assert modified_path == 'modified/path.py'
        
        # Rollback
        success = migrator.rollback_migration(backup_path)
        assert success == True
        
        # Verify rollback
        conn = sqlite3.connect(temp_db_with_issues)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM nodes WHERE id = 'node1'")
        restored_path = cursor.fetchone()[0]
        conn.close()
        assert restored_path == 'src/components/Button.tsx'  # Original value
        
        # Cleanup
        os.unlink(backup_path)
    
    def test_apply_path_fix(self, temp_db_with_issues):
        """Test individual path fix application."""
        migrator = PathDatabaseMigrator(temp_db_with_issues)
        
        inconsistency = PathInconsistency(
            table='nodes',
            row_id='node1',
            column='file_path',
            current_path='src/components/Button.tsx',
            corrected_path='project/src/components/Button.tsx',
            project_name='project',
            severity='critical',
            description='Test fix'
        )
        
        conn = sqlite3.connect(temp_db_with_issues)
        conn.execute("BEGIN TRANSACTION")
        
        # Apply fix
        migrator._apply_path_fix(conn, inconsistency)
        
        # Verify fix
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM nodes WHERE id = 'node1'")
        updated_path = cursor.fetchone()[0]
        
        conn.execute("ROLLBACK")  # Don't commit for test
        conn.close()
        
        assert updated_path == 'project/src/components/Button.tsx'


class TestMigrationIntegration:
    """Integration tests for complete migration workflow."""
    
    @pytest.fixture
    def integration_db(self):
        """Create comprehensive test database."""
        db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = db_file.name
        db_file.close()
        
        conn = sqlite3.connect(db_path)
        
        # Create full schema
        conn.execute('''
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                file_path TEXT,
                line_start INTEGER,
                line_end INTEGER,
                signature TEXT,
                docstring TEXT,
                properties JSON
            )
        ''')
        
        conn.execute('''
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                file_path TEXT,
                chunk_type TEXT,
                line_start INTEGER,
                line_end INTEGER,
                metadata JSON
            )
        ''')
        
        # Insert mixed data - some correct, some incorrect
        nodes = [
            ('good1', 'function', 'func1', 'iiot-monitoring/src/app.py', 1, 10, None, None, '{}'),
            ('bad1', 'class', 'Class1', 'src/components/Button.tsx', 5, 20, None, None, '{}'),
            ('bad2', 'variable', 'VAR1', '/workspace/backend/config.py', 1, 1, None, None, '{}'),
            ('good2', 'import', 'module', 'project-x/utils/helper.js', 1, 1, None, None, '{}'),
        ]
        
        chunks = [
            ('chunk1', 'content1', 'correct-project/main.py', 'module', 1, 50, '{}'),
            ('chunk2', 'content2', 'backend/auth.py', 'function', 10, 30, '{}'),
            ('chunk3', 'content3', '/workspace/frontend/app.js', 'class', 1, 100, '{}'),
        ]
        
        for node in nodes:
            conn.execute(
                'INSERT INTO nodes (id, type, name, file_path, line_start, line_end, signature, docstring, properties) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                node
            )
        
        for chunk in chunks:
            conn.execute(
                'INSERT INTO chunks (id, content, file_path, chunk_type, line_start, line_end, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)',
                chunk
            )
        
        conn.commit()
        conn.close()
        
        yield db_path
        os.unlink(db_path)
    
    def test_complete_analysis_workflow(self, integration_db):
        """Test complete analysis workflow."""
        results = run_path_analysis(integration_db)
        
        # Should detect issues
        assert results['total_inconsistencies'] > 0
        assert results['critical_issues'] > 0
        
        # Should have both table types
        assert results['tables_analyzed']['nodes'] > 0
        assert results['tables_analyzed']['chunks'] > 0
        
        # Should provide recommendations
        assert len(results['recommendations']) > 0
    
    def test_complete_migration_workflow(self, integration_db):
        """Test complete migration workflow."""
        # Run dry run first
        dry_result = run_path_migration(integration_db, dry_run=True)
        assert dry_result.success
        assert dry_result.paths_migrated == 0
        assert dry_result.backup_path is None
        
        # Run real migration
        real_result = run_path_migration(integration_db, dry_run=False)
        assert real_result.success
        assert real_result.paths_migrated > 0
        assert real_result.backup_path is not None
        
        # Verify database consistency after migration
        post_analysis = run_path_analysis(integration_db)
        assert post_analysis['total_inconsistencies'] == 0  # Should be fixed
        
        # Cleanup
        if real_result.backup_path:
            os.unlink(real_result.backup_path)


class TestMigrationErrorHandling:
    """Test error handling in migration process."""
    
    def test_missing_database_file(self):
        """Test behavior with missing database file."""
        fake_db_path = "/non/existent/path/database.db"
        
        with pytest.raises(Exception):
            analyzer = PathMigrationAnalyzer(fake_db_path)
            analyzer.analyze_path_consistency()
    
    def test_corrupted_database(self):
        """Test behavior with corrupted database."""
        # Create invalid database file
        db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = db_file.name
        db_file.write(b"This is not a SQLite database")
        db_file.close()
        
        try:
            with pytest.raises(Exception):
                analyzer = PathMigrationAnalyzer(db_path)
                analyzer.analyze_path_consistency()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])