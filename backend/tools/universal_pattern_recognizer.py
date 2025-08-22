"""
Universal Pattern Recognition Engine

Detects architectural relationships across all programming languages and frameworks.
Implements sophisticated pattern recognition that works across programming paradigms.

Architecture:
- Language-agnostic pattern detection with language-specific mappings
- Multi-source relationship synthesis with confidence scoring
- Universal architectural validation rules
- Extensible plugin system for new languages/frameworks
"""

import re
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    """Universal relationship types across all languages"""
    DEPENDENCY_INJECTION = "dependency_injection"
    INTERFACE_IMPLEMENTATION = "interface_implementation"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    AGGREGATION = "aggregation"
    EVENT_SUBSCRIPTION = "event_subscription"
    CONFIGURATION_DEPENDENCY = "configuration_dependency"
    API_CALL = "api_call"
    MODULE_IMPORT = "module_import"
    DATABASE_ACCESS = "database_access"

class ConfidenceLevel(Enum):
    """Universal confidence levels for relationship detection"""
    HIGH = 0.9          # Multiple sources confirm + architectural consistency
    MEDIUM_HIGH = 0.8   # Strong pattern + architectural fit
    MEDIUM = 0.7        # Good pattern match + some validation
    MEDIUM_LOW = 0.6    # Single strong source OR multiple weak sources
    LOW = 0.5           # Weak pattern OR architectural inconsistency
    REJECTED = 0.0      # Conflicts with architectural principles

@dataclass
class DetectedRelationship:
    """Universal relationship representation"""
    source_component: str
    target_component: str
    relationship_type: RelationshipType
    confidence: float
    evidence: List[str]
    source_language: str
    pattern_matched: str
    line_number: Optional[int] = None
    context: Optional[str] = None
    metadata: Optional[Dict] = None

class UniversalPatternRecognizer:
    """
    Universal Pattern Recognition Engine for architectural relationships.
    
    Detects dependency injection, interface implementation, event patterns,
    and configuration relationships across all programming languages.
    """
    
    def __init__(self):
        self.language_patterns = self._initialize_language_patterns()
        self.architectural_validators = self._initialize_architectural_validators()
        
    def recognize_all_patterns(self, source_code: str, language: str, 
                             file_path: str) -> List[DetectedRelationship]:
        """
        Main entry point: recognize all architectural patterns in source code.
        
        Args:
            source_code: Source code content
            language: Programming language (python, java, javascript, etc.)
            file_path: File path for context
            
        Returns:
            List of detected relationships with confidence scores
        """
        relationships = []
        
        try:
            # Detect different types of patterns
            relationships.extend(self.recognize_dependency_injection_patterns(
                source_code, language, file_path))
            relationships.extend(self.recognize_interface_implementation_patterns(
                source_code, language, file_path))
            relationships.extend(self.recognize_event_patterns(
                source_code, language, file_path))
            relationships.extend(self.recognize_configuration_patterns(
                source_code, language, file_path))
            relationships.extend(self.recognize_import_patterns(
                source_code, language, file_path))
            relationships.extend(self.recognize_inheritance_patterns(
                source_code, language, file_path))
            
            # Validate against architectural principles
            validated_relationships = self._validate_architectural_consistency(
                relationships, file_path)
            
            logger.info(f"🔍 Detected {len(validated_relationships)} relationships in {file_path}")
            return validated_relationships
            
        except Exception as e:
            logger.error(f"❌ Pattern recognition failed for {file_path}: {e}")
            return []
    
    def recognize_dependency_injection_patterns(self, source_code: str, 
                                               language: str, 
                                               file_path: str) -> List[DetectedRelationship]:
        """Detect dependency injection patterns across languages"""
        relationships = []
        
        if language not in self.language_patterns:
            return relationships
            
        patterns = self.language_patterns[language].get('dependency_injection', [])
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            confidence_base = pattern_config['confidence']
            
            matches = re.finditer(pattern, source_code, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_number = source_code[:match.start()].count('\n') + 1
                
                # Extract component names from match groups
                if match.groups():
                    if len(match.groups()) >= 2:
                        source_component = self._extract_component_name(file_path)
                        target_component = match.group(2)
                    else:
                        source_component = self._extract_component_name(file_path)
                        target_component = match.group(1)
                    
                    relationship = DetectedRelationship(
                        source_component=source_component,
                        target_component=target_component,
                        relationship_type=RelationshipType.DEPENDENCY_INJECTION,
                        confidence=confidence_base,
                        evidence=[f"Pattern matched: {pattern_config['description']}"],
                        source_language=language,
                        pattern_matched=match.group(0),
                        line_number=line_number,
                        context=self._extract_context(source_code, match.start(), match.end()),
                        metadata={
                            'injection_type': pattern_config.get('injection_type', 'unknown'),
                            'framework': pattern_config.get('framework', 'generic')
                        }
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def recognize_interface_implementation_patterns(self, source_code: str, 
                                                   language: str, 
                                                   file_path: str) -> List[DetectedRelationship]:
        """Detect interface/protocol implementation patterns"""
        relationships = []
        
        if language not in self.language_patterns:
            return relationships
            
        patterns = self.language_patterns[language].get('interface_implementation', [])
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            confidence_base = pattern_config['confidence']
            
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            
            for match in matches:
                line_number = source_code[:match.start()].count('\n') + 1
                
                if match.groups() and len(match.groups()) >= 2:
                    implementing_class = match.group(1)
                    interface_name = match.group(2)
                    
                    relationship = DetectedRelationship(
                        source_component=implementing_class,
                        target_component=interface_name,
                        relationship_type=RelationshipType.INTERFACE_IMPLEMENTATION,
                        confidence=confidence_base,
                        evidence=[f"Pattern matched: {pattern_config['description']}"],
                        source_language=language,
                        pattern_matched=match.group(0),
                        line_number=line_number,
                        context=self._extract_context(source_code, match.start(), match.end())
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def recognize_event_patterns(self, source_code: str, language: str, 
                               file_path: str) -> List[DetectedRelationship]:
        """Detect event-driven patterns (pub/sub, observers, listeners)"""
        relationships = []
        
        if language not in self.language_patterns:
            return relationships
            
        patterns = self.language_patterns[language].get('event_patterns', [])
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            confidence_base = pattern_config['confidence']
            
            matches = re.finditer(pattern, source_code, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_number = source_code[:match.start()].count('\n') + 1
                source_component = self._extract_component_name(file_path)
                
                if match.groups():
                    event_name = match.group(1) if len(match.groups()) >= 1 else 'unknown_event'
                    
                    relationship = DetectedRelationship(
                        source_component=source_component,
                        target_component=event_name,
                        relationship_type=RelationshipType.EVENT_SUBSCRIPTION,
                        confidence=confidence_base,
                        evidence=[f"Event pattern: {pattern_config['description']}"],
                        source_language=language,
                        pattern_matched=match.group(0),
                        line_number=line_number,
                        context=self._extract_context(source_code, match.start(), match.end()),
                        metadata={
                            'event_type': pattern_config.get('event_type', 'unknown')
                        }
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def recognize_configuration_patterns(self, source_code: str, language: str, 
                                       file_path: str) -> List[DetectedRelationship]:
        """Detect configuration-based dependencies"""
        relationships = []
        
        if language not in self.language_patterns:
            return relationships
            
        patterns = self.language_patterns[language].get('configuration_patterns', [])
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            confidence_base = pattern_config['confidence']
            
            matches = re.finditer(pattern, source_code, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_number = source_code[:match.start()].count('\n') + 1
                source_component = self._extract_component_name(file_path)
                
                if match.groups():
                    config_target = match.group(1) if len(match.groups()) >= 1 else 'unknown_config'
                    
                    relationship = DetectedRelationship(
                        source_component=source_component,
                        target_component=config_target,
                        relationship_type=RelationshipType.CONFIGURATION_DEPENDENCY,
                        confidence=confidence_base,
                        evidence=[f"Config pattern: {pattern_config['description']}"],
                        source_language=language,
                        pattern_matched=match.group(0),
                        line_number=line_number,
                        context=self._extract_context(source_code, match.start(), match.end())
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def recognize_import_patterns(self, source_code: str, language: str, 
                                file_path: str) -> List[DetectedRelationship]:
        """Detect module import and dependency patterns"""
        relationships = []
        
        if language not in self.language_patterns:
            return relationships
            
        patterns = self.language_patterns[language].get('import_patterns', [])
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            confidence_base = pattern_config['confidence']
            
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            
            for match in matches:
                line_number = source_code[:match.start()].count('\n') + 1
                source_component = self._extract_component_name(file_path)
                
                if match.groups():
                    imported_module = match.group(1) if len(match.groups()) >= 1 else 'unknown_module'
                    
                    # Clean up import path
                    imported_module = self._clean_import_path(imported_module, language)
                    
                    relationship = DetectedRelationship(
                        source_component=source_component,
                        target_component=imported_module,
                        relationship_type=RelationshipType.MODULE_IMPORT,
                        confidence=confidence_base,
                        evidence=[f"Import pattern: {pattern_config['description']}"],
                        source_language=language,
                        pattern_matched=match.group(0),
                        line_number=line_number,
                        context=self._extract_context(source_code, match.start(), match.end())
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def recognize_inheritance_patterns(self, source_code: str, language: str, 
                                     file_path: str) -> List[DetectedRelationship]:
        """Detect class inheritance patterns"""
        relationships = []
        
        if language not in self.language_patterns:
            return relationships
            
        patterns = self.language_patterns[language].get('inheritance_patterns', [])
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            confidence_base = pattern_config['confidence']
            
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            
            for match in matches:
                line_number = source_code[:match.start()].count('\n') + 1
                
                if match.groups() and len(match.groups()) >= 2:
                    child_class = match.group(1)
                    parent_class = match.group(2)
                    
                    relationship = DetectedRelationship(
                        source_component=child_class,
                        target_component=parent_class,
                        relationship_type=RelationshipType.INHERITANCE,
                        confidence=confidence_base,
                        evidence=[f"Inheritance pattern: {pattern_config['description']}"],
                        source_language=language,
                        pattern_matched=match.group(0),
                        line_number=line_number,
                        context=self._extract_context(source_code, match.start(), match.end())
                    )
                    
                    relationships.append(relationship)
        
        return relationships
    
    def _validate_architectural_consistency(self, relationships: List[DetectedRelationship],
                                          file_path: str) -> List[DetectedRelationship]:
        """Validate relationships against universal architectural principles"""
        validated = []
        
        for relationship in relationships:
            # Apply architectural validation rules
            validation_score = self._calculate_architectural_score(relationship, file_path)
            
            # Adjust confidence based on architectural consistency
            adjusted_confidence = relationship.confidence * validation_score
            
            # Only keep relationships above minimum threshold
            if adjusted_confidence >= 0.5:
                relationship.confidence = adjusted_confidence
                validated.append(relationship)
                
        return validated
    
    def _calculate_architectural_score(self, relationship: DetectedRelationship, 
                                     file_path: str) -> float:
        """Calculate architectural consistency score"""
        score = 1.0
        
        # Check for architectural violations
        if self._violates_layered_architecture(relationship, file_path):
            score *= 0.7
            
        if self._violates_dependency_inversion(relationship):
            score *= 0.8
            
        if self._violates_single_responsibility(relationship):
            score *= 0.9
            
        return max(score, 0.3)  # Minimum score for any detected pattern
    
    def _violates_layered_architecture(self, relationship: DetectedRelationship, 
                                     file_path: str) -> bool:
        """Check if relationship violates layered architecture principles"""
        # This is a simplified check - in production, you'd use more sophisticated analysis
        file_lower = file_path.lower()
        
        # Basic layer detection from file path
        is_controller = any(x in file_lower for x in ['controller', 'handler', 'route'])
        is_service = any(x in file_lower for x in ['service', 'business', 'logic'])
        is_repository = any(x in file_lower for x in ['repository', 'dao', 'data'])
        
        target_lower = relationship.target_component.lower()
        is_target_controller = any(x in target_lower for x in ['controller', 'handler'])
        
        # Controllers shouldn't depend on other controllers
        if is_controller and is_target_controller:
            return True
            
        # Repositories shouldn't depend on services
        if is_repository and any(x in target_lower for x in ['service', 'business']):
            return True
            
        return False
    
    def _violates_dependency_inversion(self, relationship: DetectedRelationship) -> bool:
        """Check for dependency inversion violations"""
        # High-level modules depending on low-level modules
        source_lower = relationship.source_component.lower()
        target_lower = relationship.target_component.lower()
        
        high_level_indicators = ['service', 'business', 'application']
        low_level_indicators = ['util', 'helper', 'database', 'file']
        
        is_source_high = any(x in source_lower for x in high_level_indicators)
        is_target_low = any(x in target_lower for x in low_level_indicators)
        
        return is_source_high and is_target_low
    
    def _violates_single_responsibility(self, relationship: DetectedRelationship) -> bool:
        """Check for single responsibility violations (too many diverse dependencies)"""
        # This would require tracking all relationships for a component
        # For now, return False - implement when we have component-wide analysis
        return False
    
    def _extract_component_name(self, file_path: str) -> str:
        """Extract component name from file path"""
        import os
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Convert snake_case to PascalCase for consistency
        if '_' in name_without_ext:
            parts = name_without_ext.split('_')
            return ''.join(word.capitalize() for word in parts)
        
        return name_without_ext
    
    def _extract_context(self, source_code: str, start: int, end: int, 
                        context_lines: int = 2) -> str:
        """Extract context around a match"""
        lines = source_code.split('\n')
        match_line = source_code[:start].count('\n')
        
        start_line = max(0, match_line - context_lines)
        end_line = min(len(lines), match_line + context_lines + 1)
        
        context_lines_text = lines[start_line:end_line]
        return '\n'.join(context_lines_text)
    
    def _clean_import_path(self, import_path: str, language: str) -> str:
        """Clean and normalize import paths"""
        # Remove quotes and common prefixes
        cleaned = import_path.strip('\'"')
        
        if language == 'javascript':
            # Remove relative path indicators
            cleaned = re.sub(r'^\.\.?/', '', cleaned)
            # Remove file extensions
            cleaned = re.sub(r'\.(js|ts|jsx|tsx)$', '', cleaned)
        elif language == 'python':
            # Remove leading dots from relative imports
            cleaned = re.sub(r'^\.+', '', cleaned)
        
        return cleaned
    
    def _initialize_language_patterns(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Initialize language-specific pattern mappings"""
        return {
            'python': {
                'dependency_injection': [
                    {
                        'pattern': r'def __init__\(self[^)]*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Constructor parameter injection',
                        'confidence': 0.8,
                        'injection_type': 'constructor'
                    },
                    {
                        'pattern': r'@inject\s*\n\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Decorator-based injection',
                        'confidence': 0.9,
                        'injection_type': 'decorator'
                    }
                ],
                'interface_implementation': [
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\([^)]*([a-zA-Z_][a-zA-Z0-9_]*)\)',
                        'description': 'Class inheritance',
                        'confidence': 0.9
                    },
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\([^)]*ABC[^)]*\)',
                        'description': 'Abstract base class implementation',
                        'confidence': 0.95
                    }
                ],
                'event_patterns': [
                    {
                        'pattern': r'\.subscribe\s*\(\s*[\'"]([^\'\"]+)[\'"]',
                        'description': 'Event subscription',
                        'confidence': 0.8,
                        'event_type': 'subscription'
                    },
                    {
                        'pattern': r'\.emit\s*\(\s*[\'"]([^\'\"]+)[\'"]',
                        'description': 'Event emission',
                        'confidence': 0.8,
                        'event_type': 'emission'
                    }
                ],
                'configuration_patterns': [
                    {
                        'pattern': r'settings\.([A-Z_][A-Z0-9_]*)',
                        'description': 'Django settings access',
                        'confidence': 0.85
                    },
                    {
                        'pattern': r'os\.environ\.get\s*\(\s*[\'"]([^\'\"]+)[\'"]',
                        'description': 'Environment variable access',
                        'confidence': 0.9
                    }
                ],
                'import_patterns': [
                    {
                        'pattern': r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                        'description': 'From import statement',
                        'confidence': 0.95
                    },
                    {
                        'pattern': r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
                        'description': 'Import statement',
                        'confidence': 0.95
                    }
                ],
                'inheritance_patterns': [
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\(([a-zA-Z_][a-zA-Z0-9_]*)\)',
                        'description': 'Single inheritance',
                        'confidence': 0.95
                    }
                ]
            },
            'java': {
                'dependency_injection': [
                    {
                        'pattern': r'@Autowired\s+private\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Spring @Autowired field injection',
                        'confidence': 0.95,
                        'injection_type': 'field',
                        'framework': 'spring'
                    },
                    {
                        'pattern': r'@Inject\s+private\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'JSR-330 @Inject annotation',
                        'confidence': 0.9,
                        'injection_type': 'field'
                    }
                ],
                'interface_implementation': [
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+implements\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Interface implementation',
                        'confidence': 0.95
                    },
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+extends\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Class extension',
                        'confidence': 0.95
                    }
                ],
                'import_patterns': [
                    {
                        'pattern': r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*);',
                        'description': 'Java import statement',
                        'confidence': 0.95
                    }
                ]
            },
            'javascript': {
                'dependency_injection': [
                    {
                        'pattern': r'constructor\s*\([^)]*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
                        'description': 'Constructor parameter',
                        'confidence': 0.7,
                        'injection_type': 'constructor'
                    }
                ],
                'import_patterns': [
                    {
                        'pattern': r'import.*from\s+[\'"]([^\'\"]+)[\'"]',
                        'description': 'ES6 import statement',
                        'confidence': 0.95
                    },
                    {
                        'pattern': r'require\s*\(\s*[\'"]([^\'\"]+)[\'"]',
                        'description': 'CommonJS require',
                        'confidence': 0.9
                    }
                ],
                'event_patterns': [
                    {
                        'pattern': r'\.addEventListener\s*\(\s*[\'"]([^\'\"]+)[\'"]',
                        'description': 'Event listener registration',
                        'confidence': 0.85,
                        'event_type': 'listener'
                    }
                ]
            },
            'go': {
                'dependency_injection': [
                    {
                        'pattern': r'func\s+New[a-zA-Z_][a-zA-Z0-9_]*\([^)]*([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Constructor function with dependency',
                        'confidence': 0.8,
                        'injection_type': 'constructor'
                    }
                ],
                'interface_implementation': [
                    {
                        'pattern': r'type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+interface',
                        'description': 'Interface definition',
                        'confidence': 0.9
                    }
                ],
                'import_patterns': [
                    {
                        'pattern': r'import\s+[\'"]([^\'\"]+)[\'"]',
                        'description': 'Go import statement',
                        'confidence': 0.95
                    }
                ]
            },
            'csharp': {
                'dependency_injection': [
                    {
                        'pattern': r'\[Inject\]\s+private\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Attribute-based injection',
                        'confidence': 0.9,
                        'injection_type': 'field'
                    }
                ],
                'interface_implementation': [
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Interface implementation',
                        'confidence': 0.9
                    }
                ],
                'import_patterns': [
                    {
                        'pattern': r'using\s+([a-zA-Z_][a-zA-Z0-9_.]*);',
                        'description': 'Using statement',
                        'confidence': 0.95
                    }
                ]
            },
            'swift': {
                'dependency_injection': [
                    {
                        'pattern': r'init\([^)]*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Swift initializer with dependency',
                        'confidence': 0.8,
                        'injection_type': 'initializer'
                    }
                ],
                'interface_implementation': [
                    {
                        'pattern': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Protocol implementation',
                        'confidence': 0.9
                    },
                    {
                        'pattern': r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Protocol conformance',
                        'confidence': 0.9
                    }
                ],
                'import_patterns': [
                    {
                        'pattern': r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                        'description': 'Swift import statement',
                        'confidence': 0.95
                    }
                ]
            }
        }
    
    def _initialize_architectural_validators(self) -> Dict[str, Any]:
        """Initialize architectural validation rules"""
        return {
            'layered_architecture': {
                'layers': ['presentation', 'business', 'data'],
                'violations': ['backward_dependency', 'layer_skipping']
            },
            'dependency_inversion': {
                'high_level': ['service', 'business', 'application'],
                'low_level': ['util', 'helper', 'database', 'file']
            },
            'single_responsibility': {
                'max_diverse_dependencies': 10
            }
        }

class UniversalRelationshipSynthesizer:
    """
    Synthesizes relationships from multiple sources with confidence scoring.
    
    Combines pattern recognition with architectural validation to produce
    high-confidence relationship mappings across all programming languages.
    """
    
    def __init__(self):
        self.pattern_recognizer = UniversalPatternRecognizer()
        
    def synthesize_relationships(self, kg_relationships: List[Dict],
                               vector_relationships: List[Dict],
                               code_relationships: List[Dict],
                               language: str) -> List[DetectedRelationship]:
        """
        Synthesize relationships from multiple sources with confidence scoring.
        
        Args:
            kg_relationships: Relationships from Knowledge Graph
            vector_relationships: Relationships inferred from vector search
            code_relationships: Relationships from pattern recognition
            language: Programming language for context
            
        Returns:
            Synthesized relationships with confidence scores
        """
        all_relationships = []
        
        # Convert code relationships (already in DetectedRelationship format)
        all_relationships.extend(code_relationships)
        
        # Convert and validate KG relationships
        kg_converted = self._convert_kg_relationships(kg_relationships, language)
        all_relationships.extend(kg_converted)
        
        # Convert and validate vector relationships
        vector_converted = self._convert_vector_relationships(vector_relationships, language)
        all_relationships.extend(vector_converted)
        
        # Deduplicate and merge relationships
        merged_relationships = self._merge_duplicate_relationships(all_relationships)
        
        # Calculate final confidence scores
        final_relationships = self._calculate_final_confidence(merged_relationships)
        
        # Filter by minimum confidence threshold
        filtered_relationships = [r for r in final_relationships if r.confidence >= 0.5]
        
        logger.info(f"🔗 Synthesized {len(filtered_relationships)} relationships from {len(all_relationships)} raw detections")
        
        return filtered_relationships
    
    def _convert_kg_relationships(self, kg_relationships: List[Dict], 
                                language: str) -> List[DetectedRelationship]:
        """Convert KG relationships to universal format"""
        converted = []
        
        for kg_rel in kg_relationships:
            try:
                relationship = DetectedRelationship(
                    source_component=kg_rel.get('source_id', 'unknown'),
                    target_component=kg_rel.get('target_id', 'unknown'),
                    relationship_type=self._map_kg_relationship_type(kg_rel.get('type', 'unknown')),
                    confidence=0.8,  # KG relationships have high base confidence
                    evidence=[f"Knowledge Graph: {kg_rel.get('type', 'unknown')}"],
                    source_language=language,
                    pattern_matched=f"KG: {kg_rel.get('type', 'unknown')}",
                    metadata={'source': 'knowledge_graph', 'kg_properties': kg_rel.get('properties', {})}
                )
                converted.append(relationship)
            except Exception as e:
                logger.warning(f"Failed to convert KG relationship: {e}")
        
        return converted
    
    def _convert_vector_relationships(self, vector_relationships: List[Dict], 
                                    language: str) -> List[DetectedRelationship]:
        """Convert vector search relationships to universal format"""
        converted = []
        
        for vec_rel in vector_relationships:
            try:
                relationship = DetectedRelationship(
                    source_component=vec_rel.get('source', 'unknown'),
                    target_component=vec_rel.get('target', 'unknown'),
                    relationship_type=self._infer_relationship_type_from_context(vec_rel.get('context', '')),
                    confidence=0.6,  # Vector relationships have medium base confidence
                    evidence=[f"Vector search context: {vec_rel.get('context', '')[:100]}..."],
                    source_language=language,
                    pattern_matched=f"Vector: {vec_rel.get('context', '')[:50]}...",
                    metadata={'source': 'vector_search', 'score': vec_rel.get('score', 0.0)}
                )
                converted.append(relationship)
            except Exception as e:
                logger.warning(f"Failed to convert vector relationship: {e}")
        
        return converted
    
    def _map_kg_relationship_type(self, kg_type: str) -> RelationshipType:
        """Map KG relationship types to universal types"""
        mapping = {
            'calls': RelationshipType.API_CALL,
            'imports': RelationshipType.MODULE_IMPORT,
            'extends': RelationshipType.INHERITANCE,
            'implements': RelationshipType.INTERFACE_IMPLEMENTATION,
            'uses': RelationshipType.DEPENDENCY_INJECTION,
            'depends_on': RelationshipType.DEPENDENCY_INJECTION,
            'contains': RelationshipType.COMPOSITION
        }
        
        return mapping.get(kg_type.lower(), RelationshipType.MODULE_IMPORT)
    
    def _infer_relationship_type_from_context(self, context: str) -> RelationshipType:
        """Infer relationship type from vector search context"""
        context_lower = context.lower()
        
        if any(x in context_lower for x in ['import', 'require', 'include']):
            return RelationshipType.MODULE_IMPORT
        elif any(x in context_lower for x in ['inject', 'autowired', 'dependency']):
            return RelationshipType.DEPENDENCY_INJECTION
        elif any(x in context_lower for x in ['extends', 'inherits']):
            return RelationshipType.INHERITANCE
        elif any(x in context_lower for x in ['implements', 'interface']):
            return RelationshipType.INTERFACE_IMPLEMENTATION
        elif any(x in context_lower for x in ['event', 'subscribe', 'listen']):
            return RelationshipType.EVENT_SUBSCRIPTION
        else:
            return RelationshipType.API_CALL
    
    def _merge_duplicate_relationships(self, relationships: List[DetectedRelationship]) -> List[DetectedRelationship]:
        """Merge duplicate relationships and combine their evidence"""
        merged = {}
        
        for rel in relationships:
            key = (rel.source_component, rel.target_component, rel.relationship_type)
            
            if key in merged:
                # Merge evidence and take highest confidence
                existing = merged[key]
                existing.evidence.extend(rel.evidence)
                existing.confidence = max(existing.confidence, rel.confidence)
            else:
                merged[key] = rel
        
        return list(merged.values())
    
    def _calculate_final_confidence(self, relationships: List[DetectedRelationship]) -> List[DetectedRelationship]:
        """Calculate final confidence scores based on multiple factors"""
        for rel in relationships:
            # Base confidence from pattern strength
            confidence = rel.confidence
            
            # Boost confidence for multiple evidence sources
            if len(rel.evidence) > 1:
                confidence = min(0.95, confidence + 0.1 * (len(rel.evidence) - 1))
            
            # Boost confidence for certain relationship types
            if rel.relationship_type in [RelationshipType.MODULE_IMPORT, RelationshipType.INHERITANCE]:
                confidence = min(0.98, confidence + 0.1)
            
            # Adjust for architectural consistency
            if self._is_architecturally_consistent(rel):
                confidence = min(0.99, confidence + 0.05)
            else:
                confidence *= 0.9
            
            rel.confidence = confidence
        
        return relationships
    
    def _is_architecturally_consistent(self, relationship: DetectedRelationship) -> bool:
        """Check if relationship is architecturally consistent"""
        # Simple architectural consistency check
        source_lower = relationship.source_component.lower()
        target_lower = relationship.target_component.lower()
        
        # Controllers should depend on services, not vice versa
        if 'service' in source_lower and 'controller' in target_lower:
            return False
        
        # Repositories should not depend on controllers
        if any(x in source_lower for x in ['repository', 'dao']) and 'controller' in target_lower:
            return False
        
        return True