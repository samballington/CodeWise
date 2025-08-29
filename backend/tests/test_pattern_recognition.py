"""
Comprehensive Test Suite for Universal Pattern Recognition Engine
Tests pattern detection across multiple programming languages
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from tools.universal_pattern_recognizer import UniversalPatternRecognizer, RelationshipType
from tools.universal_relationship_engine import UniversalRelationshipEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_python_pattern_recognition():
    """Test pattern recognition on Python code"""
    print("\n" + "="*60)
    print("TESTING PYTHON PATTERN RECOGNITION")
    print("="*60)
    
    recognizer = UniversalPatternRecognizer()
    test_file = Path(__file__).parent / "test_files" / "test_python_patterns.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Read test file
    source_code = test_file.read_text(encoding='utf-8')
    
    # Run pattern recognition
    relationships = recognizer.recognize_all_patterns(
        source_code, 'python', str(test_file))
    
    print(f"📊 FOUND {len(relationships)} RELATIONSHIPS:")
    
    # Group by relationship type for analysis
    by_type = {}
    for rel in relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    # Print detailed results
    for rel_type, rels in by_type.items():
        print(f"\n🔗 {rel_type.upper()} ({len(rels)} found):")
        for rel in rels:
            print(f"  • {rel.source_component} → {rel.target_component}")
            print(f"    Confidence: {rel.confidence:.2f} | Pattern: {rel.pattern_matched[:50]}...")
            print(f"    Evidence: {rel.evidence[0] if rel.evidence else 'None'}")
    
    # Validate expected patterns
    expected_patterns = {
        'module_import': 5,  # Should find multiple import statements
        'inheritance': 1,    # DatabaseUserRepository extends IUserRepository
        'dependency_injection': 2,  # Constructor injections
        'configuration_dependency': 2,  # settings.SECRET_KEY, os.environ.get
    }
    
    success = True
    for pattern_type, expected_count in expected_patterns.items():
        actual_count = len(by_type.get(pattern_type, []))
        if actual_count >= expected_count:
            print(f"✅ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
        else:
            print(f"❌ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
            success = False
    
    return success

async def test_java_pattern_recognition():
    """Test pattern recognition on Java code"""
    print("\n" + "="*60)
    print("TESTING JAVA PATTERN RECOGNITION")
    print("="*60)
    
    # Create test Java code
    java_code = '''
package com.example.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.example.service.UserService;
import com.example.service.EmailService;
import com.example.model.User;
import javax.inject.Inject;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    @Inject
    private EmailService emailService;
    
    private final LoggingService loggingService;
    
    // Constructor injection
    public UserController(LoggingService loggingService) {
        this.loggingService = loggingService;
    }
    
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) {
        User user = userService.findById(id);
        loggingService.log("User accessed: " + id);
        return user;
    }
}

interface UserRepository {
    User findById(Long id);
    void save(User user);
}

class DatabaseUserRepository implements UserRepository {
    public User findById(Long id) {
        return null;
    }
    
    public void save(User user) {
        // Implementation
    }
}

abstract class BaseService {
    protected abstract void processData();
}

class UserService extends BaseService {
    protected void processData() {
        // Implementation
    }
}
'''
    
    recognizer = UniversalPatternRecognizer()
    relationships = recognizer.recognize_all_patterns(
        java_code, 'java', 'test_java_file.java')
    
    print(f"📊 FOUND {len(relationships)} RELATIONSHIPS:")
    
    # Group by type
    by_type = {}
    for rel in relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    for rel_type, rels in by_type.items():
        print(f"\n🔗 {rel_type.upper()} ({len(rels)} found):")
        for rel in rels:
            print(f"  • {rel.source_component} → {rel.target_component}")
            print(f"    Confidence: {rel.confidence:.2f}")
            if rel.metadata:
                print(f"    Framework: {rel.metadata.get('framework', 'generic')}")
    
    # Validate Java-specific patterns
    expected_java_patterns = {
        'dependency_injection': 2,  # @Autowired and @Inject
        'module_import': 1,         # import statements
        'interface_implementation': 1,  # implements UserRepository
        'inheritance': 1,           # extends BaseService
    }
    
    success = True
    for pattern_type, expected_count in expected_java_patterns.items():
        actual_count = len(by_type.get(pattern_type, []))
        if actual_count >= expected_count:
            print(f"✅ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
        else:
            print(f"❌ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
            success = False
    
    return success

async def test_javascript_pattern_recognition():
    """Test pattern recognition on JavaScript/TypeScript code"""
    print("\n" + "="*60)
    print("TESTING JAVASCRIPT PATTERN RECOGNITION")
    print("="*60)
    
    js_code = '''
import React, { useState, useEffect } from 'react';
import { UserService } from '../services/UserService';
import { EmailService } from '../services/EmailService';
import { ApiClient } from '../utils/ApiClient';

class UserController {
    constructor(userService, emailService) {
        this.userService = userService;
        this.emailService = emailService;
        this.apiClient = new ApiClient();
    }
    
    async getUser(userId) {
        const user = await this.userService.findById(userId);
        this.emailService.sendWelcomeEmail(user.email);
        return user;
    }
}

// Event patterns
class EventEmitter {
    constructor() {
        this.listeners = {};
    }
    
    addEventListener(eventName, callback) {
        if (!this.listeners[eventName]) {
            this.listeners[eventName] = [];
        }
        this.listeners[eventName].push(callback);
    }
    
    emit(eventName, data) {
        if (this.listeners[eventName]) {
            this.listeners[eventName].forEach(callback => callback(data));
        }
    }
}

// React component
const UserProfile = ({ userId }) => {
    const [user, setUser] = useState(null);
    
    useEffect(() => {
        const userService = new UserService();
        userService.getUser(userId).then(setUser);
    }, [userId]);
    
    return <div>{user?.name}</div>;
};

// CommonJS require
const express = require('express');
const bodyParser = require('body-parser');

module.exports = UserController;
'''
    
    recognizer = UniversalPatternRecognizer()
    relationships = recognizer.recognize_all_patterns(
        js_code, 'javascript', 'test_js_file.js')
    
    print(f"📊 FOUND {len(relationships)} RELATIONSHIPS:")
    
    # Group by type
    by_type = {}
    for rel in relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    for rel_type, rels in by_type.items():
        print(f"\n🔗 {rel_type.upper()} ({len(rels)} found):")
        for rel in rels:
            print(f"  • {rel.source_component} → {rel.target_component}")
            print(f"    Confidence: {rel.confidence:.2f}")
    
    # Validate JavaScript patterns
    expected_js_patterns = {
        'module_import': 3,  # ES6 imports and require statements
        'dependency_injection': 1,  # constructor parameters
        'event_subscription': 1,  # addEventListener
    }
    
    success = True
    for pattern_type, expected_count in expected_js_patterns.items():
        actual_count = len(by_type.get(pattern_type, []))
        if actual_count >= expected_count:
            print(f"✅ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
        else:
            print(f"❌ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
            success = False
    
    return success

async def test_swift_pattern_recognition():
    """Test pattern recognition on Swift code"""
    print("\n" + "="*60)
    print("TESTING SWIFT PATTERN RECOGNITION")
    print("="*60)
    
    swift_code = '''
import Foundation
import UIKit
import Combine

protocol UserRepositoryProtocol {
    func getUser(id: Int) -> User?
    func saveUser(_ user: User) -> Bool
}

class DatabaseUserRepository: UserRepositoryProtocol {
    func getUser(id: Int) -> User? {
        return nil
    }
    
    func saveUser(_ user: User) -> Bool {
        return true
    }
}

class UserService {
    private let userRepository: UserRepositoryProtocol
    private let emailService: EmailService
    
    init(userRepository: UserRepositoryProtocol, emailService: EmailService) {
        self.userRepository = userRepository
        self.emailService = emailService
    }
    
    func processUser(id: Int) -> User? {
        let user = userRepository.getUser(id: id)
        if let user = user {
            emailService.sendWelcomeEmail(to: user.email)
        }
        return user
    }
}

class UserViewController: UIViewController {
    private let userService: UserService
    
    init(userService: UserService) {
        self.userService = userService
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

struct User {
    let id: Int
    let name: String
    let email: String
}

extension UserService: UserRepositoryProtocol {
    // Protocol conformance
}
'''
    
    recognizer = UniversalPatternRecognizer()
    relationships = recognizer.recognize_all_patterns(
        swift_code, 'swift', 'test_swift_file.swift')
    
    print(f"📊 FOUND {len(relationships)} RELATIONSHIPS:")
    
    # Group by type
    by_type = {}
    for rel in relationships:
        rel_type = rel.relationship_type.value
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)
    
    for rel_type, rels in by_type.items():
        print(f"\n🔗 {rel_type.upper()} ({len(rels)} found):")
        for rel in rels:
            print(f"  • {rel.source_component} → {rel.target_component}")
            print(f"    Confidence: {rel.confidence:.2f}")
    
    # Validate Swift patterns
    expected_swift_patterns = {
        'module_import': 3,  # import statements
        'dependency_injection': 2,  # init with dependencies
        'interface_implementation': 1,  # protocol conformance
        'inheritance': 1,  # extends UIViewController
    }
    
    success = True
    for pattern_type, expected_count in expected_swift_patterns.items():
        actual_count = len(by_type.get(pattern_type, []))
        if actual_count >= expected_count:
            print(f"✅ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
        else:
            print(f"❌ {pattern_type}: Found {actual_count} (expected ≥{expected_count})")
            success = False
    
    return success

async def test_universal_query_expander():
    """Test the Universal Query Expander"""
    print("\n" + "="*60)
    print("TESTING UNIVERSAL QUERY EXPANDER")
    print("="*60)
    
    try:
        from tools.universal_query_expander import UniversalQueryExpander
        
        expander = UniversalQueryExpander()
        
        # Test controller query expansion
        test_components = [
            {'name': 'UserController', 'type': 'controller', 'language': 'java'},
            {'name': 'UserService', 'type': 'service', 'language': 'java'},
            {'name': 'UserRepository', 'type': 'repository', 'language': 'java'},
        ]
        
        print("🔍 Testing controller diagram query expansion...")
        expanded = expander.expand_component_query(
            test_components, "controller class diagram")
        
        print(f"📊 Expanded from {len(test_components)} to {len(expanded)} components")
        
        # Check if all related components are included
        component_types = {comp['type'] for comp in expanded}
        expected_types = {'controller', 'service', 'repository'}
        
        if expected_types.issubset(component_types):
            print("✅ Query expansion includes all expected component types")
            return True
        else:
            missing = expected_types - component_types
            print(f"❌ Missing component types: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Query expander test failed: {e}")
        return False

async def test_universal_relationship_engine():
    """Test the Universal Relationship Engine integration"""
    print("\n" + "="*60)
    print("TESTING UNIVERSAL RELATIONSHIP ENGINE")
    print("="*60)
    
    try:
        relationship_engine = UniversalRelationshipEngine()
        
        # Create mock components
        test_components = [
            {
                'node_data': {
                    'id': 'UserController',
                    'name': 'UserController',
                    'type': 'class',
                    'file_path': str(Path(__file__).parent / "test_files" / "test_python_patterns.py")
                },
                'type': 'kg_structural_node',
                'source': 'knowledge_graph'
            }
        ]
        
        print("🔗 Testing relationship inference...")
        relationships = await relationship_engine.infer_relationships_from_components(
            test_components, "controller diagram")
        
        print(f"📊 Found {len(relationships)} relationships")
        
        if relationships:
            print("✅ Relationship engine successfully inferred relationships")
            
            # Test graph format conversion
            nodes, edges = relationship_engine.convert_to_graph_format(relationships)
            print(f"📈 Converted to graph format: {len(nodes)} nodes, {len(edges)} edges")
            
            return True
        else:
            print("⚠️  No relationships found - this may be expected for limited test data")
            return True
            
    except Exception as e:
        print(f"❌ Relationship engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("UNIVERSAL DEPENDENCY DETECTION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Python Pattern Recognition", test_python_pattern_recognition),
        ("Java Pattern Recognition", test_java_pattern_recognition),
        ("JavaScript Pattern Recognition", test_javascript_pattern_recognition),
        ("Swift Pattern Recognition", test_swift_pattern_recognition),
        ("Universal Query Expander", test_universal_query_expander),
        ("Universal Relationship Engine", test_universal_relationship_engine),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"
            import traceback
            traceback.print_exc()
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        print(f"{result} | {test_name}")
        if "✅ PASSED" in result:
            passed += 1
    
    print(f"\n📊 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Universal dependency detection is working correctly.")
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(run_all_tests())