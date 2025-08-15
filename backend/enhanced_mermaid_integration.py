"""
Enhanced Mermaid Generator Integration with Phase 1 Theming

Provides seamless integration between existing mermaid_generator tool
and Phase 1 semantic theming system while maintaining backward compatibility.
"""

import logging
from typing import Dict, List, Any, Optional

# Phase 1 imports
from ..backend.themes.theme_manager import ThemeManager
from ..backend.themes.schemas import SemanticRole
from ..backend.tools.mermaid_generator import RoleAwareMermaidGenerator

logger = logging.getLogger(__name__)


class EnhancedMermaidIntegration:
    """
    Enhanced mermaid generation with Phase 1 theming support.
    
    Maintains existing interface while adding theme capabilities and
    semantic role awareness for improved diagram consistency.
    """
    
    def __init__(self, default_theme: str = "dark_professional"):
        """
        Initialize enhanced mermaid integration.
        
        Args:
            default_theme: Default theme to use for diagram generation
        """
        # Phase 1 components
        self.theme_manager = ThemeManager()
        self.role_aware_generator = RoleAwareMermaidGenerator(default_theme)
        self.default_theme = default_theme
        
        # Performance tracking
        self.generation_stats = {
            'total_diagrams': 0,
            'themed_diagrams': 0,
            'semantic_role_assignments': 0,
            'theme_usage': {}
        }
        
        logger.info(f"Enhanced mermaid integration initialized with theme: {default_theme}")
    
    def create_mermaid_diagram(self, diagram_type: str, nodes: List[Dict], edges: List[Dict], 
                              theme: str = None, auto_assign_roles: bool = True) -> str:
        """
        Enhanced mermaid generation with Phase 1 theming support.
        
        Args:
            diagram_type: Type of diagram to generate
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            theme: Theme name (uses default if None)
            auto_assign_roles: Whether to auto-assign semantic roles
            
        Returns:
            Complete Mermaid diagram with semantic theming
        """
        self.generation_stats['total_diagrams'] += 1
        
        # Use default theme if none specified
        if theme is None:
            theme = self.default_theme
        
        # Validate theme exists
        if not self.theme_manager.get_theme(theme):
            logger.warning(f"Theme '{theme}' not found, using default")
            theme = self.default_theme
        
        # Track theme usage
        self.generation_stats['theme_usage'][theme] = \
            self.generation_stats['theme_usage'].get(theme, 0) + 1
        
        try:
            # Enhance nodes with semantic roles if needed
            enhanced_nodes = self._enhance_nodes_with_semantic_roles(
                nodes, auto_assign_roles
            )
            
            # Use Phase 1 role-aware generator
            diagram_data = {
                'diagram_type': self._normalize_diagram_type(diagram_type),
                'nodes': enhanced_nodes,
                'edges': edges,
                'theme': theme
            }
            
            diagram = self.role_aware_generator.generate_diagram(diagram_data)
            
            if enhanced_nodes != nodes:
                self.generation_stats['semantic_role_assignments'] += 1
            
            self.generation_stats['themed_diagrams'] += 1
            
            logger.debug(f"Generated themed diagram with {len(enhanced_nodes)} nodes using theme '{theme}'")
            return diagram
            
        except Exception as e:
            logger.error(f"Enhanced diagram generation failed: {e}")
            # Fallback to basic generation
            return self._fallback_diagram_generation(diagram_type, nodes, edges)
    
    def create_diagram_with_context(self, diagram_type: str, nodes: List[Dict], 
                                   edges: List[Dict], context_info: Dict[str, Any] = None,
                                   theme: str = None) -> str:
        """
        Create diagram with additional context information for better theming.
        
        Args:
            diagram_type: Type of diagram
            nodes: Node definitions
            edges: Edge definitions
            context_info: Additional context for semantic role assignment
            theme: Theme to use
            
        Returns:
            Enhanced mermaid diagram
        """
        # Use context to improve semantic role assignment
        if context_info:
            enhanced_nodes = self._enhance_nodes_with_context(nodes, context_info)
        else:
            enhanced_nodes = nodes
        
        return self.create_mermaid_diagram(diagram_type, enhanced_nodes, edges, theme)
    
    def get_available_themes(self) -> List[str]:
        """Get list of available themes."""
        return self.theme_manager.available_themes()
    
    def get_theme_preview(self, theme_name: str) -> Dict[str, str]:
        """
        Get preview of theme colors for each semantic role.
        
        Args:
            theme_name: Name of theme to preview
            
        Returns:
            Dictionary mapping semantic roles to their style information
        """
        theme = self.theme_manager.get_theme(theme_name)
        if not theme:
            return {}
        
        preview = {}
        for role in SemanticRole:
            if role in theme.roles:
                style_info = theme.roles[role]
                preview[role.value] = {
                    'fill_color': style_info.fill_color,
                    'stroke_color': style_info.stroke_color,
                    'text_color': getattr(style_info, 'text_color', None)
                }
        
        return preview
    
    def validate_diagram_data(self, diagram_type: str, nodes: List[Dict], 
                             edges: List[Dict]) -> Dict[str, Any]:
        """
        Validate diagram data and provide suggestions.
        
        Args:
            diagram_type: Type of diagram
            nodes: Node definitions
            edges: Edge definitions
            
        Returns:
            Validation results with suggestions
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'suggestions': [],
            'semantic_role_coverage': {}
        }
        
        # Check nodes have required fields
        for i, node in enumerate(nodes):
            if 'id' not in node:
                validation_results['warnings'].append(f"Node {i} missing 'id' field")
                validation_results['valid'] = False
            
            if 'label' not in node:
                validation_results['warnings'].append(f"Node {i} missing 'label' field")
            
            # Check semantic role assignment
            if 'semantic_role' not in node:
                validation_results['suggestions'].append(
                    f"Node {i} ({node.get('label', 'unlabeled')}) could benefit from semantic_role assignment"
                )
        
        # Analyze semantic role coverage
        roles_used = set()
        for node in nodes:
            role = node.get('semantic_role')
            if role:
                roles_used.add(role)
        
        validation_results['semantic_role_coverage'] = {
            'roles_used': list(roles_used),
            'total_roles_available': len(SemanticRole),
            'coverage_percentage': len(roles_used) / len(SemanticRole) * 100
        }
        
        # Suggestions for better semantic coverage
        if len(roles_used) < 3 and len(nodes) > 5:
            validation_results['suggestions'].append(
                "Consider using more diverse semantic roles for better visual distinction"
            )
        
        return validation_results
    
    def _enhance_nodes_with_semantic_roles(self, nodes: List[Dict], 
                                          auto_assign: bool) -> List[Dict]:
        """Enhance nodes with semantic role assignments."""
        if not auto_assign:
            return nodes
        
        enhanced_nodes = []
        for node in nodes:
            enhanced_node = node.copy()
            
            # Only assign role if not already present
            if 'semantic_role' not in enhanced_node:
                semantic_role = self._infer_semantic_role(enhanced_node)
                if semantic_role:
                    enhanced_node['semantic_role'] = semantic_role
            
            enhanced_nodes.append(enhanced_node)
        
        return enhanced_nodes
    
    def _enhance_nodes_with_context(self, nodes: List[Dict], 
                                   context_info: Dict[str, Any]) -> List[Dict]:
        """Enhance nodes using additional context information."""
        enhanced_nodes = []
        
        for node in nodes:
            enhanced_node = node.copy()
            
            # Use context to improve semantic role assignment
            if 'semantic_role' not in enhanced_node:
                role = self._infer_semantic_role_with_context(enhanced_node, context_info)
                if role:
                    enhanced_node['semantic_role'] = role
            
            enhanced_nodes.append(enhanced_node)
        
        return enhanced_nodes
    
    def _infer_semantic_role(self, node: Dict[str, Any]) -> Optional[str]:
        """
        Infer semantic role from node information.
        
        Uses keyword matching and heuristics to assign appropriate semantic roles.
        """
        node_id = str(node.get('id', '')).lower()
        label = str(node.get('label', '')).lower()
        shape = str(node.get('shape', '')).lower()
        
        # Combine all text for analysis
        text_content = f"{node_id} {label} {shape}"
        
        # Logic role keywords
        logic_keywords = [
            'service', 'controller', 'handler', 'processor', 'manager', 'engine',
            'algorithm', 'logic', 'business', 'calculate', 'process', 'transform'
        ]
        
        # IO role keywords
        io_keywords = [
            'input', 'output', 'file', 'stream', 'reader', 'writer', 'parser',
            'serializer', 'formatter', 'upload', 'download', 'import', 'export'
        ]
        
        # Interface role keywords
        interface_keywords = [
            'api', 'endpoint', 'route', 'gateway', 'proxy', 'adapter', 'facade',
            'interface', 'client', 'server', 'middleware', 'bridge'
        ]
        
        # Storage role keywords
        storage_keywords = [
            'database', 'db', 'storage', 'cache', 'repository', 'store', 'data',
            'model', 'entity', 'table', 'collection', 'index', 'query'
        ]
        
        # Test role keywords
        test_keywords = [
            'test', 'mock', 'stub', 'fake', 'fixture', 'spec', 'validation',
            'assertion', 'verify', 'check', 'unittest', 'integration'
        ]
        
        # External role keywords
        external_keywords = [
            'external', 'third', 'party', 'vendor', 'api', 'webhook', 'oauth',
            'payment', 'stripe', 'aws', 'google', 'microsoft', 'github'
        ]
        
        # Check for keyword matches
        keyword_groups = [
            (logic_keywords, 'logic'),
            (io_keywords, 'io'),
            (interface_keywords, 'interface'),
            (storage_keywords, 'storage'),
            (test_keywords, 'test'),
            (external_keywords, 'external')
        ]
        
        for keywords, role in keyword_groups:
            if any(keyword in text_content for keyword in keywords):
                return role
        
        # Shape-based inference
        if shape in ['database', 'cylinder']:
            return 'storage'
        elif shape in ['diamond', 'rhombus']:
            return 'logic'
        elif shape in ['circle', 'ellipse']:
            return 'interface'
        
        # Default to logic for unknown cases
        return 'logic'
    
    def _infer_semantic_role_with_context(self, node: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Optional[str]:
        """Infer semantic role using additional context."""
        # Start with basic inference
        role = self._infer_semantic_role(node)
        
        # Refine based on context
        context_type = context.get('diagram_context', '')
        architecture_layer = context.get('architecture_layer', '')
        
        if context_type == 'microservices':
            # In microservices context, services are often interface components
            if 'service' in str(node.get('label', '')).lower():
                return 'interface'
        
        if architecture_layer == 'data':
            # In data layer, most components are storage
            return 'storage'
        elif architecture_layer == 'presentation':
            # In presentation layer, most components are interface
            return 'interface'
        elif architecture_layer == 'business':
            # In business layer, most components are logic
            return 'logic'
        
        return role
    
    def _normalize_diagram_type(self, diagram_type: str) -> str:
        """Normalize diagram type for consistency."""
        diagram_type = diagram_type.strip().lower()
        
        # Map common variations
        mappings = {
            'flowchart': 'graph TD',
            'flow': 'graph TD',
            'graph': 'graph TD',
            'architecture': 'graph TD',
            'system': 'graph TD'
        }
        
        return mappings.get(diagram_type, 'graph TD')
    
    def _fallback_diagram_generation(self, diagram_type: str, nodes: List[Dict], 
                                    edges: List[Dict]) -> str:
        """Fallback diagram generation without theming."""
        try:
            # Basic mermaid generation
            lines = [self._normalize_diagram_type(diagram_type)]
            
            # Add nodes
            for node in nodes:
                node_id = str(node.get('id', 'node'))
                label = str(node.get('label', node_id))
                lines.append(f"    {node_id}[\"{label}\"]")
            
            # Add edges
            for edge in edges:
                source = str(edge.get('source', ''))
                target = str(edge.get('target', ''))
                if source and target:
                    lines.append(f"    {source} --> {target}")
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Fallback diagram generation failed: {e}")
            return "graph TD\n    Error[\"Diagram generation failed\"]"
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get diagram generation statistics."""
        return {
            **self.generation_stats,
            'semantic_role_assignment_rate': (
                self.generation_stats['semantic_role_assignments'] / 
                max(self.generation_stats['total_diagrams'], 1)
            ),
            'theming_success_rate': (
                self.generation_stats['themed_diagrams'] / 
                max(self.generation_stats['total_diagrams'], 1)
            )
        }
    
    def clear_generation_stats(self):
        """Clear generation statistics."""
        self.generation_stats = {
            'total_diagrams': 0,
            'themed_diagrams': 0,
            'semantic_role_assignments': 0,
            'theme_usage': {}
        }


# Backward compatibility function
def create_enhanced_mermaid_diagram(diagram_type: str, nodes: List[Dict], edges: List[Dict], 
                                   theme: str = "dark_professional") -> str:
    """
    Backward compatible entry point for enhanced mermaid generation.
    
    Args:
        diagram_type: Type of diagram to generate
        nodes: Node definitions
        edges: Edge definitions
        theme: Theme to use
        
    Returns:
        Enhanced mermaid diagram
    """
    integration = EnhancedMermaidIntegration(theme)
    return integration.create_mermaid_diagram(diagram_type, nodes, edges, theme)