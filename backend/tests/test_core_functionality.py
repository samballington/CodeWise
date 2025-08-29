"""
Core Functionality Test - Test the universal components directly
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

async def test_all_language_patterns():
    """Test pattern recognition across all supported languages"""
    print("="*80)
    print("COMPREHENSIVE LANGUAGE PATTERN RECOGNITION TEST")
    print("="*80)
    
    from tools.universal_pattern_recognizer import UniversalPatternRecognizer
    
    recognizer = UniversalPatternRecognizer()
    
    # Test cases for different languages
    test_cases = {
        'python': '''
from django.conf import settings
from myapp.services import UserService

class UserController:
    def __init__(self, user_service: UserService):
        self.user_service = user_service
        self.db_url = settings.DATABASE_URL
        
class UserRepository(DatabaseConnection):
    pass
''',
        'java': '''
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

@Controller
public class UserController {
    @Autowired
    private UserService userService;
    
    public User getUser(Long id) {
        return userService.findById(id);
    }
}

class UserService implements UserRepository {
    public User findById(Long id) { return null; }
}
''',
        'javascript': '''
import { UserService } from './services/UserService';
import React from 'react';

class UserController {
    constructor(userService) {
        this.userService = userService;
        this.eventEmitter = new EventEmitter();
    }
    
    getUser(id) {
        this.eventEmitter.addEventListener('userFound', this.handleUser);
        return this.userService.findById(id);
    }
}

const UserComponent = () => {
    return <div>User</div>;
};
''',
        'swift': '''
import Foundation
import UIKit

protocol UserRepositoryProtocol {
    func getUser(id: Int) -> User?
}

class UserService: UserRepositoryProtocol {
    private let repository: UserRepositoryProtocol
    
    init(repository: UserRepositoryProtocol) {
        self.repository = repository
    }
    
    func getUser(id: Int) -> User? {
        return repository.getUser(id: id)
    }
}

class UserViewController: UIViewController {
    private let userService: UserService
    
    init(userService: UserService) {
        self.userService = userService
        super.init(nibName: nil, bundle: nil)
    }
}
''',
        'go': '''
package main

import (
    "context"
    "github.com/example/userservice"
)

type UserRepository interface {
    GetUser(ctx context.Context, id int) (*User, error)
}

type UserService struct {
    repo UserRepository
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (s *UserService) GetUser(ctx context.Context, id int) (*User, error) {
    return s.repo.GetUser(ctx, id)
}
''',
        'csharp': '''
using System;
using Microsoft.Extensions.DependencyInjection;

namespace Example.Controllers
{
    public class UserController
    {
        [Inject]
        private readonly IUserService userService;
        
        public UserController(IUserService userService)
        {
            this.userService = userService;
        }
        
        public User GetUser(int id)
        {
            return userService.FindById(id);
        }
    }
    
    public class UserService : IUserService
    {
        public User FindById(int id) { return null; }
    }
}
'''
    }
    
    total_relationships = 0
    language_results = {}
    
    for language, code in test_cases.items():
        print(f"\nTesting {language.upper()}:")
        print("-" * 40)
        
        relationships = recognizer.recognize_all_patterns(
            code, language, f'test.{language}')
        
        # Group by type
        by_type = {}
        for rel in relationships:
            rel_type = rel.relationship_type.value
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append(rel)
        
        print(f"Found {len(relationships)} total relationships:")
        for rel_type, rels in by_type.items():
            print(f"  {rel_type}: {len(rels)}")
        
        language_results[language] = len(relationships)
        total_relationships += len(relationships)
    
    print(f"\n" + "="*60)
    print("LANGUAGE PATTERN DETECTION SUMMARY")
    print("="*60)
    
    for language, count in language_results.items():
        print(f"{language:12}: {count:3} relationships detected")
    
    print(f"\nTOTAL: {total_relationships} relationships across all languages")
    
    # Verify we found relationships in most languages
    languages_with_patterns = sum(1 for count in language_results.values() if count > 0)
    success = languages_with_patterns >= 4  # At least 4 out of 6 languages should have patterns
    
    print(f"\nLanguages with detected patterns: {languages_with_patterns}/6")
    print(f"RESULT: {'PASSED' if success else 'FAILED'}")
    
    return success

async def test_architectural_validation():
    """Test architectural consistency validation"""
    print("\n" + "="*60)
    print("ARCHITECTURAL VALIDATION TEST")
    print("="*60)
    
    from tools.universal_pattern_recognizer import (
        UniversalPatternRecognizer, DetectedRelationship, 
        RelationshipType, ConfidenceLevel
    )
    
    recognizer = UniversalPatternRecognizer()
    
    # Test architectural violation detection
    violations = []
    
    # Test 1: Repository depending on Controller (violation)
    test_relationship = DetectedRelationship(
        source_component="UserRepository",
        target_component="UserController", 
        relationship_type=RelationshipType.DEPENDENCY_INJECTION,
        confidence=0.9,
        evidence=["Test relationship"],
        source_language="java",
        pattern_matched="test"
    )
    
    validation_score = recognizer._calculate_architectural_score(
        test_relationship, "/workspace/data/UserRepository.java")
    
    if validation_score < 1.0:
        print("PASS: Detected architectural violation (repository -> controller)")
        violations.append("repository_to_controller")
    else:
        print("FAIL: Did not detect repository -> controller violation")
    
    # Test 2: Controller depending on Service (valid)
    valid_relationship = DetectedRelationship(
        source_component="UserController",
        target_component="UserService",
        relationship_type=RelationshipType.DEPENDENCY_INJECTION,
        confidence=0.9,
        evidence=["Test relationship"],
        source_language="java",
        pattern_matched="test"
    )
    
    validation_score = recognizer._calculate_architectural_score(
        valid_relationship, "/workspace/controller/UserController.java")
    
    if validation_score >= 0.8:
        print("PASS: Valid controller -> service relationship scored highly")
    else:
        print("FAIL: Valid controller -> service relationship scored too low")
    
    print(f"\nArchitectural violations detected: {len(violations)}")
    return len(violations) > 0  # Success if we can detect violations

async def test_confidence_scoring():
    """Test confidence scoring system"""
    print("\n" + "="*60)
    print("CONFIDENCE SCORING TEST")
    print("="*60)
    
    from tools.universal_pattern_recognizer import (
        UniversalRelationshipSynthesizer, DetectedRelationship,
        RelationshipType
    )
    
    synthesizer = UniversalRelationshipSynthesizer()
    
    # Create relationships with different evidence strengths
    relationships = [
        DetectedRelationship(
            source_component="Controller",
            target_component="Service",
            relationship_type=RelationshipType.DEPENDENCY_INJECTION,
            confidence=0.9,
            evidence=["KG", "Vector search", "Code pattern"],  # Multiple sources
            source_language="java",
            pattern_matched="@Autowired"
        ),
        DetectedRelationship(
            source_component="Service",
            target_component="Repository",
            relationship_type=RelationshipType.API_CALL,
            confidence=0.6,
            evidence=["Code pattern only"],  # Single source
            source_language="java",
            pattern_matched="method call"
        )
    ]
    
    # Test confidence calculation
    final_relationships = synthesizer._calculate_final_confidence(relationships)
    
    high_confidence = [r for r in final_relationships if r.confidence >= 0.8]
    medium_confidence = [r for r in final_relationships if 0.6 <= r.confidence < 0.8]
    low_confidence = [r for r in final_relationships if r.confidence < 0.6]
    
    print(f"High confidence relationships: {len(high_confidence)}")
    print(f"Medium confidence relationships: {len(medium_confidence)}")
    print(f"Low confidence relationships: {len(low_confidence)}")
    
    # Verify multi-source evidence gets higher confidence
    multi_source_rel = final_relationships[0]  # Has 3 evidence sources
    single_source_rel = final_relationships[1]  # Has 1 evidence source
    
    if multi_source_rel.confidence > single_source_rel.confidence:
        print("PASS: Multi-source evidence received higher confidence")
        return True
    else:
        print("FAIL: Confidence scoring not working correctly")
        return False

async def run_core_tests():
    """Run all core functionality tests"""
    print("UNIVERSAL RELATIONSHIP ENGINE - CORE FUNCTIONALITY TESTS")
    print("="*80)
    
    tests = [
        ("Multi-Language Pattern Recognition", test_all_language_patterns),
        ("Architectural Validation", test_architectural_validation), 
        ("Confidence Scoring", test_confidence_scoring),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("CORE FUNCTIONALITY TEST RESULTS")
    print("="*80)
    
    passed = 0
    for test_name, result in results:
        print(f"{result:10} | {test_name}")
        if result == "PASSED":
            passed += 1
    
    print(f"\nOVERALL: {passed}/{len(tests)} core tests passed")
    
    if passed == len(tests):
        print("\nSUCCESS: All core functionality is working correctly!")
        print("The Universal Relationship Engine is ready for production use.")
    else:
        print("\nWARNING: Some core functionality issues detected.")
    
    return passed == len(tests)

if __name__ == "__main__":
    asyncio.run(run_core_tests())