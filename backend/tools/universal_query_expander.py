"""
Universal Query Expander - Architecture-Aware Query Intelligence

This module implements REQ-UNIVERSAL-QUERY: Cross-Language Query Intelligence
Automatically expands component queries to include related architectural elements.

REQ-UNIVERSAL-QUERY-1: Architecture-Aware Query Expansion
REQ-UNIVERSAL-QUERY-2: Multi-Source Architectural Data Fusion
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ArchitecturePattern(Enum):
    """Universal architectural patterns recognized across languages."""
    LAYERED = "layered"              # Controller -> Service -> Repository
    MVC = "mvc"                      # Model -> View -> Controller  
    MICROSERVICE = "microservice"    # Service boundaries and inter-service deps
    COMPONENT_BASED = "component"    # React/Vue components, modular architecture
    EVENT_DRIVEN = "event_driven"   # Publishers -> Subscribers -> Handlers
    MONOLITHIC = "monolithic"       # Single deployment unit with layers
    UNKNOWN = "unknown"              # Pattern not detected


@dataclass
class ComponentClassification:
    """Classification of a component's architectural role."""
    role: str                        # controller, service, model, etc.
    layer: str                       # presentation, business, data
    confidence: float                # 0.0 - 1.0
    language: str                    # Programming language
    framework: Optional[str] = None  # Detected framework (Django, Express, etc.)


class UniversalQueryExpander:
    """
    Architecture-aware query expansion engine that works across all languages.
    
    Automatically expands narrow queries like "controller diagram" to include
    related services, repositories, and architectural components.
    """
    
    # REQ-UNIVERSAL-QUERY-1: Universal Expansion Rules
    EXPANSION_RULES = {
        'controller': ['service', 'model', 'middleware', 'route', 'handler'],
        'handler': ['service', 'model', 'middleware', 'processor'],
        'route': ['controller', 'middleware', 'guard', 'service'],
        'service': ['repository', 'model', 'external_api', 'client'],
        'component': ['service', 'store', 'hook', 'util', 'provider'],
        'model': ['repository', 'service', 'entity'],
        'repository': ['model', 'entity', 'database'],
        'middleware': ['service', 'util', 'config'],
        'view': ['controller', 'service', 'model'],
        'widget': ['service', 'model', 'provider'],  # Flutter/mobile
    }
    
    # Architecture pattern indicators by component relationships
    ARCHITECTURE_INDICATORS = {
        ArchitecturePattern.LAYERED: {
            'required_roles': ['controller', 'service'],
            'layer_order': ['controller', 'service', 'repository'],
            'indicators': ['controller->service', 'service->repository']
        },
        ArchitecturePattern.MVC: {
            'required_roles': ['model', 'view', 'controller'],
            'layer_order': ['view', 'controller', 'model'],
            'indicators': ['view->controller', 'controller->model']
        },
        ArchitecturePattern.MICROSERVICE: {
            'required_roles': ['service', 'api'],
            'layer_order': ['api', 'service', 'repository'],
            'indicators': ['api->service', 'service->external']
        },
        ArchitecturePattern.COMPONENT_BASED: {
            'required_roles': ['component'],
            'layer_order': ['component', 'service', 'store'],
            'indicators': ['component->service', 'component->store']
        }
    }
    
    # Framework-specific architectural conventions
    FRAMEWORK_CONVENTIONS = {
        # Python frameworks
        'django': {
            'controllers': 'views', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.MVC
        },
        'fastapi': {
            'controllers': 'routes', 'services': 'dependencies', 'models': 'models',
            'pattern': ArchitecturePattern.LAYERED
        },
        'flask': {
            'controllers': 'routes', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.LAYERED
        },
        
        # JavaScript frameworks
        'express': {
            'controllers': 'routes', 'services': 'middleware', 'models': 'models',
            'pattern': ArchitecturePattern.LAYERED
        },
        'nestjs': {
            'controllers': 'controllers', 'services': 'services', 'models': 'entities',
            'pattern': ArchitecturePattern.LAYERED
        },
        'react': {
            'controllers': 'components', 'services': 'hooks', 'models': 'stores',
            'pattern': ArchitecturePattern.COMPONENT_BASED
        },
        
        # Java frameworks
        'spring': {
            'controllers': 'controllers', 'services': 'services', 'models': 'entities',
            'pattern': ArchitecturePattern.LAYERED
        },
        
        # Go frameworks
        'gin': {
            'controllers': 'handlers', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.LAYERED
        },
        'echo': {
            'controllers': 'handlers', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.LAYERED
        },
        
        # C# frameworks
        'aspnet': {
            'controllers': 'controllers', 'services': 'services', 'models': 'entities',
            'pattern': ArchitecturePattern.LAYERED
        },
        
        # Swift frameworks
        'swiftui': {
            'controllers': 'views', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.COMPONENT_BASED
        },
        'uikit': {
            'controllers': 'controllers', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.MVC
        },
        
        # Ruby frameworks
        'rails': {
            'controllers': 'controllers', 'services': 'services', 'models': 'models',
            'pattern': ArchitecturePattern.MVC
        }
    }
    
    def __init__(self):
        """Initialize the universal query expander."""
        logger.info("UniversalQueryExpander initialized")
    
    def expand_component_query(self, initial_components: List[Dict], 
                              max_depth: int = 2) -> List[Dict]:
        """
        Expand component query to include architecturally related components.
        
        Args:
            initial_components: Initial components discovered by query
            max_depth: Maximum expansion depth to prevent explosion
            
        Returns:
            Expanded list of components including architectural dependencies
        """
        if not initial_components:
            return []
        
        # Classify initial components
        classified_components = [
            self._classify_component(component) for component in initial_components
        ]
        
        # Detect architectural pattern
        architecture_pattern = self.detect_architecture_pattern(classified_components)
        logger.info(f"Detected architectural pattern: {architecture_pattern.value}")
        
        # Perform expansion based on pattern and relationships
        expanded_components = list(initial_components)  # Start with originals
        expansion_candidates = set()
        
        for depth in range(max_depth):
            current_expansion = set()
            
            for component_classification in classified_components:
                # Get expansion targets for this component role
                expansion_targets = self._get_expansion_targets(
                    component_classification, architecture_pattern
                )
                current_expansion.update(expansion_targets)
            
            # Find actual components matching expansion targets
            new_components = self._find_components_by_roles(
                current_expansion, initial_components
            )
            
            if not new_components:
                break  # No more components to expand
            
            expanded_components.extend(new_components)
            expansion_candidates.update(current_expansion)
            
            # Classify new components for next iteration
            new_classifications = [
                self._classify_component(comp) for comp in new_components
            ]
            classified_components.extend(new_classifications)
        
        # Remove duplicates while preserving order
        unique_components = []
        seen_ids = set()
        for component in expanded_components:
            comp_id = component.get('id') or component.get('name')
            if comp_id not in seen_ids:
                unique_components.append(component)
                seen_ids.add(comp_id)
        
        logger.info(f"Expanded query from {len(initial_components)} to {len(unique_components)} components")
        return unique_components
    
    def detect_architecture_pattern(self, components: List[ComponentClassification]) -> ArchitecturePattern:
        """
        Identify architectural pattern from component types and relationships.
        
        Args:
            components: List of classified components
            
        Returns:
            Detected architectural pattern
        """
        if not components:
            return ArchitecturePattern.UNKNOWN
        
        # Extract roles and frameworks
        roles = {comp.role for comp in components}
        frameworks = {comp.framework for comp in components if comp.framework}
        
        # Check framework-specific patterns first
        if frameworks:
            primary_framework = next(iter(frameworks))
            if primary_framework in self.FRAMEWORK_CONVENTIONS:
                return self.FRAMEWORK_CONVENTIONS[primary_framework]['pattern']
        
        # Pattern detection based on component roles
        for pattern, indicators in self.ARCHITECTURE_INDICATORS.items():
            required_roles = set(indicators['required_roles'])
            if required_roles.issubset(roles):
                logger.debug(f"Pattern {pattern.value} matched by required roles")
                return pattern
        
        # Fallback: simple heuristics
        if 'controller' in roles and 'service' in roles:
            return ArchitecturePattern.LAYERED
        elif 'model' in roles and 'view' in roles and 'controller' in roles:
            return ArchitecturePattern.MVC
        elif 'component' in roles:
            return ArchitecturePattern.COMPONENT_BASED
        
        return ArchitecturePattern.UNKNOWN
    
    def apply_expansion_boundaries(self, expanded_components: List[Dict], 
                                 complexity_limit: int = 20) -> List[Dict]:
        """
        Apply intelligent filtering to prevent diagram complexity explosion.
        
        Args:
            expanded_components: Components after expansion
            complexity_limit: Maximum number of components to include
            
        Returns:
            Filtered components within complexity boundaries
        """
        if len(expanded_components) <= complexity_limit:
            return expanded_components
        
        # Rank components by architectural importance
        ranked_components = self._rank_by_architectural_importance(expanded_components)
        
        # Keep most important components within limit
        filtered_components = ranked_components[:complexity_limit]
        
        logger.info(f"Applied complexity boundaries: {len(expanded_components)} -> {len(filtered_components)} components")
        return filtered_components
    
    def _classify_component(self, component: Dict) -> ComponentClassification:
        """
        Classify a component's architectural role using universal patterns.
        
        Args:
            component: Component dictionary with metadata
            
        Returns:
            Component classification with role, layer, and confidence
        """
        file_path = component.get('file_path', '').lower()
        component_name = component.get('name', '').lower()
        component_type = component.get('type', '').lower()
        
        # Extract language from file path or metadata
        language = self._detect_language_from_component(component)
        
        # Detect framework from imports or file structure
        framework = self._detect_framework_from_component(component)
        
        # Role classification based on file path patterns
        role = self._classify_role_from_path(file_path, component_name)
        
        # Layer classification based on role
        layer = self._classify_layer_from_role(role)
        
        # Calculate confidence based on multiple indicators
        confidence = self._calculate_classification_confidence(
            file_path, component_name, component_type, role
        )
        
        return ComponentClassification(
            role=role,
            layer=layer, 
            confidence=confidence,
            language=language,
            framework=framework
        )
    
    def _get_expansion_targets(self, component_classification: ComponentClassification,
                             architecture_pattern: ArchitecturePattern) -> Set[str]:
        """
        Get expansion targets for a classified component based on architectural pattern.
        
        Args:
            component_classification: Classified component
            architecture_pattern: Detected architectural pattern
            
        Returns:
            Set of role names to expand to
        """
        role = component_classification.role
        expansion_targets = set()
        
        # Get universal expansion rules
        if role in self.EXPANSION_RULES:
            expansion_targets.update(self.EXPANSION_RULES[role])
        
        # Apply pattern-specific expansion rules
        if architecture_pattern in self.ARCHITECTURE_INDICATORS:
            pattern_info = self.ARCHITECTURE_INDICATORS[architecture_pattern]
            layer_order = pattern_info['layer_order']
            
            # Expand to adjacent layers in architectural pattern
            if role in layer_order:
                role_index = layer_order.index(role)
                # Add adjacent layers
                if role_index > 0:
                    expansion_targets.add(layer_order[role_index - 1])
                if role_index < len(layer_order) - 1:
                    expansion_targets.add(layer_order[role_index + 1])
        
        # Framework-specific expansion rules
        if component_classification.framework in self.FRAMEWORK_CONVENTIONS:
            framework_info = self.FRAMEWORK_CONVENTIONS[component_classification.framework]
            # Add framework-specific related components
            for framework_role in framework_info.values():
                if isinstance(framework_role, str) and framework_role != role:
                    expansion_targets.add(framework_role)
        
        return expansion_targets
    
    def _find_components_by_roles(self, target_roles: Set[str], 
                                reference_components: List[Dict]) -> List[Dict]:
        """
        Find components matching target roles from available component pool.
        
        This would typically query the KG/vector store for components with matching roles.
        For now, returns empty list as placeholder for actual implementation.
        """
        # TODO: Implement actual component discovery
        # This would integrate with the existing KG and vector search systems
        return []
    
    def _classify_role_from_path(self, file_path: str, component_name: str) -> str:
        """Classify architectural role from file path and component name."""
        # Directory-based classification (universal)
        if any(pattern in file_path for pattern in ['controller', 'handler', 'route']):
            return 'controller'
        elif any(pattern in file_path for pattern in ['service', 'business', 'logic']):
            return 'service' 
        elif any(pattern in file_path for pattern in ['repository', 'model', 'data', 'entity']):
            return 'model'
        elif any(pattern in file_path for pattern in ['view', 'component', 'widget']):
            return 'view'
        elif any(pattern in file_path for pattern in ['middleware', 'guard', 'interceptor']):
            return 'middleware'
        elif any(pattern in file_path for pattern in ['config', 'setting', 'env']):
            return 'config'
        elif any(pattern in file_path for pattern in ['util', 'helper', 'tool', 'common']):
            return 'utility'
        
        # Name-based classification
        if any(pattern in component_name for pattern in ['controller', 'handler']):
            return 'controller'
        elif any(pattern in component_name for pattern in ['service', 'manager', 'provider']):
            return 'service'
        elif any(pattern in component_name for pattern in ['model', 'entity', 'repository']):
            return 'model'
        elif any(pattern in component_name for pattern in ['view', 'component', 'widget']):
            return 'view'
        
        return 'component'  # Default fallback
    
    def _classify_layer_from_role(self, role: str) -> str:
        """Classify architectural layer from component role."""
        layer_mapping = {
            'controller': 'presentation',
            'handler': 'presentation', 
            'route': 'presentation',
            'view': 'presentation',
            'component': 'presentation',
            'widget': 'presentation',
            'service': 'business',
            'manager': 'business',
            'provider': 'business',
            'logic': 'business',
            'model': 'data',
            'entity': 'data', 
            'repository': 'data',
            'dao': 'data',
            'middleware': 'infrastructure',
            'config': 'infrastructure',
            'utility': 'infrastructure'
        }
        
        return layer_mapping.get(role, 'unknown')
    
    def _calculate_classification_confidence(self, file_path: str, component_name: str,
                                          component_type: str, role: str) -> float:
        """Calculate confidence score for role classification."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for path-based classification
        if role in file_path:
            confidence += 0.3
        
        # Increase confidence for name-based classification  
        if role in component_name:
            confidence += 0.2
        
        # Increase confidence for type consistency
        if component_type in ['class', 'interface'] and role != 'utility':
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _detect_language_from_component(self, component: Dict) -> str:
        """Detect programming language from component metadata."""
        file_path = component.get('file_path', '')
        
        # Language detection by file extension
        extension_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.rb': 'ruby',
            '.php': 'php',
            '.kt': 'kotlin',
            '.rs': 'rust'
        }
        
        for ext, lang in extension_mapping.items():
            if file_path.endswith(ext):
                return lang
        
        return 'unknown'
    
    def _detect_framework_from_component(self, component: Dict) -> Optional[str]:
        """Detect framework from component imports or file structure."""
        file_path = component.get('file_path', '').lower()
        
        # Framework detection patterns
        framework_patterns = {
            'django': ['django', 'manage.py', 'settings.py'],
            'fastapi': ['fastapi', 'uvicorn'],
            'flask': ['flask', 'app.py'],
            'express': ['express', 'package.json'],
            'react': ['react', 'jsx', 'tsx'],
            'spring': ['springframework', 'springboot'],
            'gin': ['gin-gonic', 'gin'],
            'rails': ['rails', 'gemfile'],
            'swiftui': ['swiftui'],
            'uikit': ['uikit']
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in file_path for pattern in patterns):
                return framework
        
        return None
    
    def _rank_by_architectural_importance(self, components: List[Dict]) -> List[Dict]:
        """Rank components by architectural importance and relevance."""
        def importance_score(component: Dict) -> float:
            score = 0.0
            
            # Higher importance for core architectural components
            role = self._classify_role_from_path(
                component.get('file_path', '').lower(),
                component.get('name', '').lower()
            )
            
            role_importance = {
                'controller': 1.0,
                'service': 0.9,
                'model': 0.8,
                'repository': 0.7,
                'view': 0.6,
                'middleware': 0.5,
                'utility': 0.3,
                'config': 0.2
            }
            
            score += role_importance.get(role, 0.1)
            
            # Higher importance for components with more relationships
            # This would be enhanced with actual relationship data
            
            return score
        
        return sorted(components, key=importance_score, reverse=True)