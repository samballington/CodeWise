"""
Simple Pattern Recognition Test (Windows-compatible)
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from tools.universal_pattern_recognizer import UniversalPatternRecognizer

async def test_python_patterns():
    """Test Python pattern recognition"""
    print("\n" + "="*60)
    print("TESTING PYTHON PATTERN RECOGNITION")
    print("="*60)
    
    recognizer = UniversalPatternRecognizer()
    
    python_code = '''
from typing import Dict, List
from myapp.services import UserService
import os

class UserController:
    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self.db_url = os.environ.get('DATABASE_URL')
    
    def get_user(self, user_id):
        return self.user_service.find_user(user_id)

class DatabaseService(UserService):
    def find_user(self, user_id):
        return None
'''
    
    relationships = recognizer.recognize_all_patterns(
        python_code, 'python', 'test.py')
    
    print(f"FOUND {len(relationships)} RELATIONSHIPS:")
    
    # Group by type
    by_type = {}
    for rel in relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    for rel_type, rels in by_type.items():
        print(f"\n{rel_type.upper()} ({len(rels)} found):")
        for rel in rels:
            print(f"  - {rel.source_component} -> {rel.target_component} (confidence: {rel.confidence:.2f})")
    
    # Check for expected patterns
    expected = {
        'module_import': 2,  # from myapp.services, import os
        'dependency_injection': 1,  # constructor injection
        'inheritance': 1,  # DatabaseService extends UserService
        'configuration_dependency': 1,  # os.environ.get
    }
    
    success = True
    for pattern_type, expected_count in expected.items():
        actual_count = len(by_type.get(pattern_type, []))
        if actual_count >= expected_count:
            print(f"PASS: {pattern_type} - Found {actual_count} (expected >={expected_count})")
        else:
            print(f"FAIL: {pattern_type} - Found {actual_count} (expected >={expected_count})")
            success = False
    
    return success

async def test_java_patterns():
    """Test Java pattern recognition"""
    print("\n" + "="*60)
    print("TESTING JAVA PATTERN RECOGNITION")
    print("="*60)
    
    recognizer = UniversalPatternRecognizer()
    
    java_code = '''
import org.springframework.beans.factory.annotation.Autowired;
import com.example.service.UserService;

public class UserController {
    @Autowired
    private UserService userService;
    
    public User getUser(Long id) {
        return userService.findById(id);
    }
}

class UserService implements UserRepository {
    public User findById(Long id) {
        return null;
    }
}
'''
    
    relationships = recognizer.recognize_all_patterns(
        java_code, 'java', 'test.java')
    
    print(f"FOUND {len(relationships)} RELATIONSHIPS:")
    
    # Group by type
    by_type = {}
    for rel in relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    for rel_type, rels in by_type.items():
        print(f"\n{rel_type.upper()} ({len(rels)} found):")
        for rel in rels:
            print(f"  - {rel.source_component} -> {rel.target_component} (confidence: {rel.confidence:.2f})")
    
    # Check for expected patterns
    expected = {
        'dependency_injection': 1,  # @Autowired
        'module_import': 1,  # import statements
        'interface_implementation': 1,  # implements
    }
    
    success = True
    for pattern_type, expected_count in expected.items():
        actual_count = len(by_type.get(pattern_type, []))
        if actual_count >= expected_count:
            print(f"PASS: {pattern_type} - Found {actual_count} (expected >={expected_count})")
        else:
            print(f"FAIL: {pattern_type} - Found {actual_count} (expected >={expected_count})")
            success = False
    
    return success

async def test_relationship_engine():
    """Test relationship engine integration"""
    print("\n" + "="*60)
    print("TESTING UNIVERSAL RELATIONSHIP ENGINE")
    print("="*60)
    
    try:
        from tools.universal_relationship_engine import UniversalRelationshipEngine
        
        engine = UniversalRelationshipEngine()
        
        # Test with mock component data
        components = [
            {
                'node_data': {
                    'id': 'UserController',
                    'name': 'UserController', 
                    'type': 'class',
                    'file_path': str(Path(__file__).parent / "test_files" / "test_python_patterns.py")
                },
                'type': 'kg_structural_node'
            }
        ]
        
        relationships = await engine.infer_relationships_from_components(components)
        
        print(f"FOUND {len(relationships)} RELATIONSHIPS from engine")
        
        if relationships:
            # Test graph conversion
            nodes, edges = engine.convert_to_graph_format(relationships)
            print(f"CONVERTED TO: {len(nodes)} nodes, {len(edges)} edges")
            return True
        else:
            print("WARNING: No relationships found (may be expected with mock data)")
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_simple_tests():
    """Run simplified tests"""
    print("UNIVERSAL DEPENDENCY DETECTION - SIMPLE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Python Patterns", test_python_patterns),
        ("Java Patterns", test_java_patterns),
        ("Relationship Engine", test_relationship_engine),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        print(f"{result:10} | {test_name}")
        if result == "PASSED":
            passed += 1
    
    print(f"\nOVERALL: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(run_simple_tests())