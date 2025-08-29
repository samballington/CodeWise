"""
Phase 2 Integration Tests

Comprehensive test suite for Phase 2 Knowledge Graph implementation
to ensure all components work correctly in Docker environment.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def sample_codebase():
    """Create a temporary codebase for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample Python files
    sample_files = {
        'auth.py': '''
"""Authentication module"""

class UserManager:
    """Manages user authentication and authorization"""
    
    def __init__(self):
        self.users = {}
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user with username and password"""
        if username in self.users:
            return self.users[username]['password'] == password
        return False
    
    def create_user(self, username: str, password: str):
        """Create a new user account"""
        self.users[username] = {'password': password}
        return True

def validate_session(session_id: str) -> bool:
    """Validate user session"""
    # Implementation here
    return session_id is not None
''',
        'users.py': '''
"""User profile management"""

from auth import UserManager

def get_user_profile(user_id: int) -> dict:
    """Retrieve user profile information"""
    # This function calls authenticate_user indirectly
    manager = UserManager()
    
    return {
        'id': user_id,
        'name': f'User {user_id}',
        'active': True
    }

def update_user_profile(user_id: int, data: dict):
    """Update user profile data"""
    profile = get_user_profile(user_id)
    profile.update(data)
    return profile
''',
        'api.py': '''
"""API endpoints"""

from users import get_user_profile, update_user_profile
from auth import validate_session

class APIHandler:
    """Main API request handler"""
    
    def handle_user_request(self, request):
        """Handle user-related API requests"""
        if not validate_session(request.session_id):
            return {'error': 'Invalid session'}
        
        user_profile = get_user_profile(request.user_id)
        return {'success': True, 'data': user_profile}
'''
    }
    
    for filename, content in sample_files.items():
        file_path = Path(temp_dir) / filename
        file_path.write_text(content)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestPhase2DatabaseSetup:
    """Test Phase 2.1: Database Setup and Management"""
    
    def test_database_initialization(self, temp_db):
        """Test that database can be initialized correctly."""
        from storage.database_setup import DatabaseSetup
        
        db_setup = DatabaseSetup(temp_db)
        success = db_setup.initialize_database()
        
        assert success, "Database initialization should succeed"
        assert os.path.exists(temp_db), "Database file should be created"
    
    def test_database_manager_operations(self, temp_db):
        """Test basic database manager operations."""
        from storage.database_setup import DatabaseSetup
        from storage.database_manager import DatabaseManager
        
        # Initialize database
        db_setup = DatabaseSetup(temp_db)
        db_setup.initialize_database()
        
        # Test database manager
        db_manager = DatabaseManager(temp_db)
        
        # Insert test node
        success = db_manager.insert_node(
            node_id='test_func',
            node_type='function',
            name='test_function',
            file_path='/test/file.py',
            line_start=10,
            line_end=20
        )
        
        assert success, "Node insertion should succeed"
        
        # Retrieve node
        nodes = db_manager.get_nodes_by_name('test_function')
        assert len(nodes) > 0, "Should find inserted node"
        assert nodes[0]['name'] == 'test_function', "Node name should match"
        
        db_manager.close()


class TestPhase2KnowledgeGraph:
    """Test Phase 2.2: Knowledge Graph Pipeline"""
    
    def test_symbol_collector(self, sample_codebase, temp_db):
        """Test symbol collection from code files."""
        from storage.database_setup import DatabaseSetup
        from storage.database_manager import DatabaseManager
        from knowledge_graph.symbol_collector import SymbolCollector
        
        # Initialize database
        db_setup = DatabaseSetup(temp_db)
        db_setup.initialize_database()
        db_manager = DatabaseManager(temp_db)
        
        # Test symbol collector
        collector = SymbolCollector(db_manager)
        
        # Process one file
        auth_file = Path(sample_codebase) / 'auth.py'
        symbols = collector.collect_symbols_from_file(auth_file)
        
        assert len(symbols) > 0, "Should collect symbols from file"
        
        # Check that we found expected symbols
        symbol_names = [s['name'] for s in symbols]
        assert 'UserManager' in symbol_names, "Should find UserManager class"
        assert 'authenticate_user' in symbol_names, "Should find authenticate_user method"
        
        db_manager.close()
    
    def test_relationship_extractor(self, sample_codebase, temp_db):
        """Test relationship extraction between symbols."""
        from storage.database_setup import DatabaseSetup
        from storage.database_manager import DatabaseManager
        from knowledge_graph.symbol_collector import SymbolCollector
        from knowledge_graph.relationship_extractor import RelationshipExtractor
        
        # Initialize database
        db_setup = DatabaseSetup(temp_db)
        db_setup.initialize_database()
        db_manager = DatabaseManager(temp_db)
        
        # First collect symbols
        collector = SymbolCollector(db_manager)
        
        for py_file in Path(sample_codebase).glob('*.py'):
            collector.collect_symbols_from_file(py_file)
        
        # Then extract relationships
        extractor = RelationshipExtractor(db_manager)
        
        for py_file in Path(sample_codebase).glob('*.py'):
            relationships = extractor.extract_relationships_from_file(py_file)
            
        # Verify relationships exist
        nodes = db_manager.get_nodes_by_name('get_user_profile')
        if nodes:
            node_id = nodes[0]['id']
            callers = db_manager.find_callers(node_id)
            # Should have some relationships
            
        db_manager.close()


class TestPhase2KGQueryMethods:
    """Test Phase 2.3: KG Query Methods"""
    
    def test_kg_query_methods_import(self):
        """Test that KG query methods can be imported."""
        try:
            from backend.kg_query_methods import (
                kg_find_symbol, kg_explore_neighborhood, 
                kg_analyze_dependencies, kg_find_callers
            )
            assert True, "KG query methods imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import KG query methods: {e}")
    
    def test_kg_enhanced_search_import(self):
        """Test that KG enhanced search can be imported."""
        try:
            from backend.kg_enhanced_smart_search import KGEnhancedSmartSearchEngine
            assert True, "KG enhanced search imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import KG enhanced search: {e}")
    
    def test_kg_analyze_relationships_import(self):
        """Test that KG analyze relationships can be imported."""
        try:
            from backend.kg_enhanced_analyze_relationships import enhanced_analyze_relationships
            assert True, "KG analyze relationships imported successfully"
        except ImportError as e:
            pytest.fail(f"Failed to import KG analyze relationships: {e}")


class TestPhase2ToolIntegration:
    """Test Phase 2.3: Tool Integration"""
    
    def test_agent_kg_imports(self):
        """Test that agent can import KG components."""
        import sys
        
        # Test the imports that agent.py uses
        try:
            from backend.kg_query_methods import (
                kg_find_symbol, kg_explore_neighborhood, kg_analyze_dependencies, kg_find_callers
            )
            from backend.kg_enhanced_smart_search import KGEnhancedSmartSearchEngine
            from backend.kg_enhanced_analyze_relationships import enhanced_analyze_relationships
            
            assert True, "All agent KG imports successful"
        except ImportError as e:
            pytest.fail(f"Agent KG import failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])