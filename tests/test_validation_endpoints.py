"""
Tests for Path Validation API Endpoints

Tests the REST API endpoints for path validation and migration management.
"""

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database_setup import setup_codewise_database


class TestValidationEndpoints:
    """Test path validation API endpoints."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Robust cleanup that handles locked files
        try:
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError):
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(temp_dir)
            except (PermissionError, OSError):
                pass
    
    @pytest.fixture
    def test_database(self, temp_workspace):
        """Create test database with sample data."""
        db_path = Path(temp_workspace) / "test.db"
        setup_codewise_database(str(db_path))
        
        # Insert sample data for testing
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        sample_nodes = [
            ('node1', 'function', 'func1', 'test-project/src/app.py', 1, 10, None, None, '{}'),
            ('node2', 'class', 'Component', 'test-project/components/Button.tsx', 15, 30, None, None, '{}'),
            ('node3', 'function', 'api', 'other-project/api/handler.py', 5, 20, None, None, '{}'),
            # Some inconsistent paths for testing
            ('node4', 'function', 'bad', 'frontend/BadComponent.js', 1, 5, None, None, '{}'),
            ('node5', 'variable', 'config', '/workspace/absolute-project/config.py', 1, 1, None, None, '{}'),
        ]
        
        for node in sample_nodes:
            cursor.execute('''
                INSERT INTO nodes (id, type, name, file_path, line_start, line_end, signature, docstring, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', node)
        
        conn.commit()
        conn.close()
        
        return str(db_path)
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        # Mock the FastAPI app with just the validation router
        from fastapi import FastAPI
        from backend.routers.validation import router
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/validation/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
        
        # Should indicate component availability
        assert data["components"]["path_manager"] is not None
    
    def test_project_validation_endpoint(self, client, test_database):
        """Test project validation endpoint."""
        request_data = {
            "project_name": "test-project",
            "db_path": test_database
        }
        
        response = client.post("/api/validation/project/validate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["project_name"] == "test-project"
        assert "relative_path_files" in data
        assert "absolute_path_files" in data
        assert "total_files" in data
        assert "path_consistency" in data
        assert "recommendations" in data
        assert "timestamp" in data
        
        # Should find the test-project files
        assert data["total_files"] > 0
    
    def test_project_validation_missing_project(self, client):
        """Test project validation with missing project name."""
        request_data = {
            "db_path": "nonexistent.db"
        }
        
        response = client.post("/api/validation/project/validate", json=request_data)
        
        assert response.status_code == 400
        assert "project_name is required" in response.json()["detail"]
    
    def test_database_analysis_endpoint(self, client, test_database):
        """Test database analysis endpoint."""
        request_data = {
            "db_path": test_database
        }
        
        response = client.post("/api/validation/analyze", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "total_inconsistencies" in data
        assert "critical_issues" in data
        assert "warning_issues" in data
        assert "inconsistencies" in data
        assert "database_path" in data
        assert "timestamp" in data
        
        # Should detect some inconsistencies from our test data
        assert data["database_path"] == test_database
    
    def test_migration_dry_run_endpoint(self, client, test_database):
        """Test migration dry run endpoint."""
        request_data = {
            "dry_run": True,
            "db_path": test_database
        }
        
        response = client.post("/api/validation/migrate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "success" in data
        assert "total_checked" in data
        assert "paths_migrated" in data
        assert "errors" in data
        assert "warnings" in data
        assert "duration_seconds" in data
        assert "timestamp" in data
        
        # Dry run should not migrate paths
        assert data["paths_migrated"] == 0
        assert data["backup_path"] is None
    
    def test_list_projects_endpoint(self, client, test_database):
        """Test list projects endpoint."""
        response = client.get(f"/api/validation/projects?db_path={test_database}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "projects" in data
        assert "total_projects" in data
        assert "timestamp" in data
        
        # Should find our test projects
        assert data["total_projects"] > 0
        
        # Check project structure
        if data["projects"]:
            project = data["projects"][0]
            assert "name" in project
            assert "file_count" in project
            assert "path_consistency" in project
    
    def test_navigator_test_endpoint(self, client, test_database):
        """Test navigator testing endpoint."""
        response = client.get(f"/api/validation/navigator/test/test-project?db_path={test_database}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["project_name"] == "test-project"
        assert "test_results" in data
        assert "timestamp" in data
        
        # Check test results structure
        test_results = data["test_results"]
        assert "validation" in test_results
        assert "find_all_files" in test_results
        assert "list_root" in test_results
        assert "validate_operation" in test_results
    
    def test_get_project_patterns_endpoint(self, client):
        """Test project patterns endpoint."""
        response = client.get("/api/validation/patterns/test-project")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["project_name"] == "test-project"
        assert "relative_pattern" in data
        assert "absolute_pattern" in data
        assert "timestamp" in data
        
        # Check pattern format
        assert data["relative_pattern"] == "test-project/%"
        assert "test-project" in data["absolute_pattern"]
    
    def test_database_debug_endpoint(self, client, test_database):
        """Test database debug endpoint."""
        response = client.get(f"/api/validation/debug/database?db_path={test_database}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["database_path"] == test_database
        assert "tables" in data
        assert "sample_paths" in data
        assert "path_statistics" in data
        assert "timestamp" in data
        
        # Check that required tables exist
        assert "nodes" in data["tables"]
        
        # Check path statistics structure
        stats = data["path_statistics"]
        assert "total_nodes" in stats
        assert "absolute_paths" in stats
        assert "relative_paths" in stats
    
    def test_error_handling_nonexistent_database(self, client):
        """Test error handling with non-existent database."""
        request_data = {
            "project_name": "any-project",
            "db_path": "nonexistent.db"
        }
        
        response = client.post("/api/validation/project/validate", json=request_data)
        
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]
    
    def test_migration_endpoint_validation(self, client, test_database):
        """Test migration endpoint input validation."""
        # Test with invalid parameters
        request_data = {
            "dry_run": "not_a_boolean",  # Invalid type
            "db_path": test_database
        }
        
        response = client.post("/api/validation/migrate", json=request_data)
        
        # Should fail validation
        assert response.status_code == 422
    
    @patch('backend.routers.validation.PATH_COMPONENTS_AVAILABLE', False)
    def test_endpoints_when_components_unavailable(self, client):
        """Test endpoint behavior when path components are unavailable."""
        # Health check should indicate degraded status
        response = client.get("/api/validation/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        
        # Other endpoints should return 503
        request_data = {"project_name": "test"}
        response = client.post("/api/validation/project/validate", json=request_data)
        assert response.status_code == 503


class TestValidationEndpointsIntegration:
    """Integration tests for validation endpoints."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        try:
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError):
            pass
    
    @pytest.fixture
    def populated_database(self, temp_workspace):
        """Create database with realistic test data."""
        db_path = Path(temp_workspace) / "integration_test.db"
        setup_codewise_database(str(db_path))
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Insert realistic project structure
        realistic_nodes = [
            # Good project with consistent paths
            ('n1', 'function', 'main', 'webapp/src/main.js', 1, 50, None, None, '{}'),
            ('n2', 'class', 'App', 'webapp/src/App.jsx', 10, 100, None, None, '{}'),
            ('n3', 'function', 'api', 'webapp/backend/api.py', 5, 30, None, None, '{}'),
            
            # Project with mixed path formats (common issue)
            ('n4', 'function', 'handler', 'api-service/handlers/main.py', 1, 20, None, None, '{}'),
            ('n5', 'class', 'Service', '/workspace/api-service/service.py', 15, 80, None, None, '{}'),
            
            # Files without project prefixes (problematic)
            ('n6', 'variable', 'config', 'config/settings.js', 1, 1, None, None, '{}'),
            ('n7', 'function', 'util', 'utils/helper.py', 3, 15, None, None, '{}'),
        ]
        
        for node in realistic_nodes:
            cursor.execute('''
                INSERT INTO nodes (id, type, name, file_path, line_start, line_end, signature, docstring, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', node)
        
        conn.commit()
        conn.close()
        
        return str(db_path)
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        from backend.routers.validation import router
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)
    
    def test_complete_validation_workflow(self, client, populated_database):
        """Test complete validation and potential migration workflow."""
        
        # Step 1: Check overall health
        health_response = client.get("/api/validation/health")
        assert health_response.status_code == 200
        
        # Step 2: Analyze database for issues
        analysis_response = client.post("/api/validation/analyze", json={
            "db_path": populated_database
        })
        assert analysis_response.status_code == 200
        analysis_data = analysis_response.json()
        
        # Should detect some inconsistencies
        assert analysis_data["total_inconsistencies"] > 0
        
        # Step 3: List all projects
        projects_response = client.get(f"/api/validation/projects?db_path={populated_database}")
        assert projects_response.status_code == 200
        projects_data = projects_response.json()
        
        assert projects_data["total_projects"] > 0
        project_names = [p["name"] for p in projects_data["projects"]]
        
        # Step 4: Validate specific projects
        for project_name in project_names[:2]:  # Test first 2 projects
            validation_response = client.post("/api/validation/project/validate", json={
                "project_name": project_name,
                "db_path": populated_database
            })
            assert validation_response.status_code == 200
            
            validation_data = validation_response.json()
            assert validation_data["project_name"] == project_name
            assert validation_data["total_files"] > 0
        
        # Step 5: Test navigator for a project
        if project_names:
            test_project = project_names[0]
            navigator_response = client.get(
                f"/api/validation/navigator/test/{test_project}?db_path={populated_database}"
            )
            assert navigator_response.status_code == 200
            
            navigator_data = navigator_response.json()
            assert navigator_data["project_name"] == test_project
            assert "test_results" in navigator_data
        
        # Step 6: Run migration dry run
        migration_response = client.post("/api/validation/migrate", json={
            "dry_run": True,
            "db_path": populated_database
        })
        assert migration_response.status_code == 200
        
        migration_data = migration_response.json()
        assert migration_data["success"] is not None
        
        # All steps completed successfully
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])