"""
Role-Aware Mermaid Diagram Generator

Semantic theming system that replaces hardcoded templates with intelligent
role-based styling. Uses external themes and LLM-provided semantic roles
for consistent, maintainable diagram generation.
"""

from typing import Dict, List, Any, Optional
import re
import logging
from pathlib import Path

# Import our new theming system
from ..themes import ThemeManager, SemanticRole

logger = logging.getLogger(__name__)


class RoleAwareMermaidGenerator:
    """
    Next-generation Mermaid generator using semantic role-based theming.
    
    Replaces keyword-based styling with LLM-provided semantic roles,
    eliminating the 819-line hardcoded template system while providing
    superior styling accuracy and theme flexibility.
    """
    
    def __init__(self, default_theme: str = "dark_professional"):
        """
        Initialize with theme manager and default styling preferences.
        
        Args:
            default_theme: Theme to use when none specified
        """
        self.theme_manager = ThemeManager()
        self.default_theme = default_theme
        
        # Supported diagram types (simplified from original 6 types)
        self.supported_types = {
            "graph TD": "Top-Down Flow",
            "graph LR": "Left-Right Flow", 
            "flowchart TD": "Top-Down Flowchart",
            "flowchart LR": "Left-Right Flowchart",
            "graph": "Generic Graph"
        }
        
        logger.info(f"RoleAwareMermaidGenerator initialized with {len(self.theme_manager.available_themes())} themes")
    
    def generate_diagram(self, data: Dict[str, Any]) -> str:
        """
        Generate Mermaid diagram using semantic role-based theming.
        
        Args:
            data: Dictionary containing:
                - diagram_type: Mermaid diagram type (e.g., "graph TD")
                - nodes: List of nodes with semantic_role field
                - edges: List of edges
                - theme: Optional theme name (defaults to self.default_theme)
                
        Returns:
            Complete Mermaid diagram with semantic theming
            
        Raises:
            ValueError: If input validation fails
        """
        # Validate input
        validation_errors = self._validate_input(data)
        if validation_errors:
            raise ValueError(f"Input validation failed: {'; '.join(validation_errors)}")
        
        diagram_type = data["diagram_type"]
        nodes = data["nodes"]
        edges = data["edges"]
        theme_name = data.get("theme", self.default_theme)
        
        # Validate theme exists
        if not self.theme_manager.get_theme(theme_name):
            logger.warning(f"Theme '{theme_name}' not found, using default")
            theme_name = self.default_theme
        
        try:
            # Build diagram components
            diagram_lines = []
            
            # 1. Diagram declaration
            diagram_lines.append(self._get_diagram_declaration(diagram_type))
            
            # 2. Theme definitions
            theme_definitions = self.theme_manager.get_definitions(theme_name)
            if theme_definitions:
                diagram_lines.append(f"    {theme_definitions}")
            
            # 3. Node definitions with semantic styling
            node_lines = self._generate_nodes(nodes, theme_name)
            diagram_lines.extend(node_lines)
            
            # 4. Edge definitions
            edge_lines = self._generate_edges(edges)
            diagram_lines.extend(edge_lines)
            
            # 5. Style applications
            style_lines = self._apply_semantic_styles(nodes)
            diagram_lines.extend(style_lines)
            
            result = "\n".join(diagram_lines)
            
            logger.debug(f"Generated {len(diagram_lines)} line diagram with theme '{theme_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Diagram generation failed: {e}")
            return self._generate_error_diagram(str(e))
    
    def _validate_input(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate input data structure and content.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Input must be a dictionary")
            return errors
        
        # Required fields
        required_fields = ["diagram_type", "nodes", "edges"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate diagram type
        diagram_type = data.get("diagram_type", "")
        if diagram_type and not self._is_valid_diagram_type(diagram_type):
            valid_types = list(self.supported_types.keys())
            errors.append(f"Invalid diagram_type '{diagram_type}'. Supported: {valid_types}")
        
        # Validate nodes
        nodes = data.get("nodes")
        if nodes is not None:
            if not isinstance(nodes, list):
                errors.append("'nodes' must be a list")
            elif len(nodes) == 0:
                errors.append("At least one node is required")
            else:
                errors.extend(self._validate_nodes(nodes))
        
        # Validate edges
        edges = data.get("edges")
        if edges is not None:
            if not isinstance(edges, list):
                errors.append("'edges' must be a list")
            else:
                errors.extend(self._validate_edges(edges, nodes or []))
        
        return errors
    
    def _validate_nodes(self, nodes: List[Dict]) -> List[str]:
        """Validate node structure and semantic roles."""
        errors = []
        
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"Node {i} must be a dictionary")
                continue
            
            # Required fields
            if "id" not in node or not str(node["id"]).strip():
                errors.append(f"Node {i} missing or empty 'id'")
            
            if "label" not in node or not str(node["label"]).strip():
                errors.append(f"Node {i} missing or empty 'label'")
            
            # Validate semantic role if provided
            semantic_role = node.get("semantic_role")
            if semantic_role:
                try:
                    SemanticRole(semantic_role)
                except ValueError:
                    valid_roles = [role.value for role in SemanticRole]
                    errors.append(f"Node {i} has invalid semantic_role '{semantic_role}'. Valid roles: {valid_roles}")
        
        return errors
    
    def _validate_edges(self, edges: List[Dict], nodes: List[Dict]) -> List[str]:
        """Validate edge structure and node references."""
        errors = []
        node_ids = {str(node.get("id", "")) for node in nodes}
        
        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"Edge {i} must be a dictionary")
                continue
            
            # Required fields
            if "source" not in edge or not str(edge["source"]).strip():
                errors.append(f"Edge {i} missing or empty 'source'")
            elif str(edge["source"]) not in node_ids:
                errors.append(f"Edge {i} references unknown source node '{edge['source']}'")
            
            if "target" not in edge or not str(edge["target"]).strip():
                errors.append(f"Edge {i} missing or empty 'target'")
            elif str(edge["target"]) not in node_ids:
                errors.append(f"Edge {i} references unknown target node '{edge['target']}'")
        
        return errors
    
    def _is_valid_diagram_type(self, diagram_type: str) -> bool:
        """Check if diagram type is supported."""
        # Support exact matches and variations
        diagram_type = diagram_type.strip().lower()
        
        for supported_type in self.supported_types.keys():
            if diagram_type == supported_type.lower():
                return True
            # Also support partial matches like "graph" matching "graph TD"
            if diagram_type == "graph" or diagram_type == "flowchart":
                return True
        
        return False
    
    def _get_diagram_declaration(self, diagram_type: str) -> str:
        """Get the Mermaid diagram declaration line."""
        # Normalize diagram type
        diagram_type = diagram_type.strip()
        
        # Default to "graph TD" if just "graph" is specified
        if diagram_type.lower() == "graph":
            diagram_type = "graph TD"
        elif diagram_type.lower() == "flowchart":
            diagram_type = "flowchart TD"
        
        return diagram_type
    
    def _generate_nodes(self, nodes: List[Dict], theme_name: str) -> List[str]:
        """Generate Mermaid node definitions with semantic awareness."""
        node_lines = []
        
        for node in nodes:
            node_id = self._sanitize_id(str(node["id"]))
            label = self._escape_label(str(node["label"]))
            shape = node.get("shape", "rectangle")
            
            # Generate node with appropriate shape
            node_line = self._format_node_with_shape(node_id, label, shape)
            node_lines.append(f"    {node_line}")
        
        return node_lines
    
    def _format_node_with_shape(self, node_id: str, label: str, shape: str) -> str:
        """Format node with Mermaid shape syntax."""
        shape_mapping = {
            "rectangle": f'{node_id}["{label}"]',
            "round": f'{node_id}("{label}")',
            "diamond": f'{node_id}{{{label}}}',
            "rhombus": f'{node_id}{{{label}}}',
            "circle": f'{node_id}(("{label}"))',
            "ellipse": f'{node_id}(("{label}"))',
            "stadium": f'{node_id}(["{label}"])',
            "subroutine": f'{node_id}[["{label}"]]',
            "database": f'{node_id}[("{label}")]',
            "cylinder": f'{node_id}[("{label}")]'
        }
        
        return shape_mapping.get(shape, f'{node_id}["{label}"]')
    
    def _generate_edges(self, edges: List[Dict]) -> List[str]:
        """Generate Mermaid edge definitions."""
        edge_lines = []
        
        for edge in edges:
            source = self._sanitize_id(str(edge["source"]))
            target = self._sanitize_id(str(edge["target"]))
            label = edge.get("label", "")
            
            if label:
                # Edge with label
                escaped_label = self._escape_label(str(label))
                edge_line = f"{source} -->|{escaped_label}| {target}"
            else:
                # Simple edge
                edge_line = f"{source} --> {target}"
            
            edge_lines.append(f"    {edge_line}")
        
        return edge_lines
    
    def _apply_semantic_styles(self, nodes: List[Dict]) -> List[str]:
        """Apply semantic role-based styling to nodes."""
        style_lines = []
        
        for node in nodes:
            node_id = self._sanitize_id(str(node["id"]))
            semantic_role = node.get("semantic_role")
            
            if semantic_role:
                try:
                    role_enum = SemanticRole(semantic_role)
                    style_class = f"{role_enum.value}Style"
                    style_lines.append(f"    class {node_id} {style_class}")
                except ValueError:
                    logger.warning(f"Unknown semantic role: {semantic_role} for node {node_id}")
                    # Apply default styling
                    style_lines.append(f"    class {node_id} logicStyle")
        
        return style_lines
    
    def _sanitize_id(self, node_id: str) -> str:
        """Sanitize node ID for Mermaid compatibility."""
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
        
        # Ensure starts with letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"n_{sanitized}"
        
        return sanitized or "node"
    
    def _escape_label(self, label: str) -> str:
        """Escape special characters in labels for Mermaid."""
        # Escape quotes but preserve other characters
        escaped = label.replace('"', '\\"')
        
        # Don't escape > characters as they break Mermaid arrow syntax
        return escaped
    
    def _generate_error_diagram(self, error_message: str) -> str:
        """Generate error diagram when generation fails."""
        safe_message = self._escape_label(error_message)
        return f"""graph TD
    error["⚠️ Error: {safe_message}"]
    classDef errorStyle fill:#dc3545,stroke:#721c24,stroke-width:3px,color:white
    class error errorStyle"""
    
    def get_theme_info(self) -> Dict[str, Any]:
        """Get information about available themes and current configuration."""
        return {
            "available_themes": self.theme_manager.available_themes(),
            "default_theme": self.default_theme,
            "supported_roles": [role.value for role in SemanticRole],
            "supported_diagram_types": list(self.supported_types.keys())
        }


# Backward compatibility function for existing code
def create_mermaid_diagram(data: Dict[str, Any]) -> str:
    """
    Backward compatible entry point for existing mermaid generation.
    
    Now uses the new role-aware generator while maintaining the same interface.
    """
    generator = RoleAwareMermaidGenerator()
    return generator.generate_diagram(data)


# Utility functions for diagram inspection and debugging
def preview_node_styling(nodes: List[Dict], theme_name: str = "dark_professional") -> Dict[str, str]:
    """Preview what semantic styles will be applied to nodes."""
    result = {}
    
    for node in nodes:
        node_id = str(node.get("id", ""))
        semantic_role = node.get("semantic_role")
        
        if semantic_role:
            try:
                role_enum = SemanticRole(semantic_role)
                result[node_id] = f"{role_enum.value}Style"
            except ValueError:
                result[node_id] = "logicStyle (default)"
        else:
            result[node_id] = "no_style (missing semantic_role)"
    
    return result


def validate_semantic_roles(nodes: List[Dict]) -> Dict[str, List[str]]:
    """Validate semantic roles in node data."""
    valid_roles = []
    invalid_roles = []
    missing_roles = []
    
    for node in nodes:
        node_id = str(node.get("id", "unknown"))
        semantic_role = node.get("semantic_role")
        
        if not semantic_role:
            missing_roles.append(node_id)
        else:
            try:
                SemanticRole(semantic_role)
                valid_roles.append(f"{node_id}: {semantic_role}")
            except ValueError:
                invalid_roles.append(f"{node_id}: {semantic_role}")
    
    return {
        "valid": valid_roles,
        "invalid": invalid_roles,
        "missing": missing_roles
    }


def get_diagram_info() -> Dict[str, Any]:
    """Get information about the diagram generation system."""
    generator = RoleAwareMermaidGenerator()
    theme_info = generator.get_theme_info()
    
    return {
        "system": "Role-Aware Mermaid Generator v1.0",
        "features": [
            "Semantic role-based theming",
            "External theme configuration", 
            "LLM-guided styling",
            "Theme validation",
            "Backward compatibility"
        ],
        **theme_info
    }