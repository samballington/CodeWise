"""
Pure Rendering MermaidRenderer for Phase 3.2.1

Deterministic diagram generation that separates data discovery from visualization.
Contains ZERO logic for data discovery - input is structured data, output is 
deterministic Mermaid syntax.

This eliminates LLM decision paralysis by moving complex logic into reliable Python code.
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Import theming system
try:
    from ..themes.theme_manager import ThemeManager
except ImportError:
    try:
        # Fallback for Docker environment
        from backend.themes.theme_manager import ThemeManager
    except ImportError:
        # Fallback for standalone execution
        from themes.theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class MermaidRenderer:
    """
    Pure rendering engine for Mermaid diagrams. Contains ZERO logic for data discovery.
    Input is structured data, output is deterministic Mermaid syntax.
    
    This is a pure function approach - same input always produces same output.
    No decision making, no data fetching, no complex logic.
    """
    
    def __init__(self, theme_manager: Optional[ThemeManager] = None):
        """Initialize renderer with theme manager."""
        self.theme_manager = theme_manager or ThemeManager()
        logger.info("MermaidRenderer initialized (pure rendering mode)")
    
    def generate(self, diagram_type: str, graph_data: dict, theme_name: str = 'dark_professional') -> str:
        """
        Generates a Mermaid diagram from pre-defined set of nodes and edges.
        
        Args:
            diagram_type: Type of diagram ('graph TD', 'classDiagram', etc.)
            graph_data: Dict with 'nodes' and 'edges' lists
            theme_name: Theme to apply
            
        Returns:
            Complete Mermaid diagram syntax
        """
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        if not nodes:
            return self._generate_error_diagram("No nodes provided for diagram")
        
        logger.debug(f"Rendering {diagram_type} with {len(nodes)} nodes, {len(edges)} edges")
        
        # Pure deterministic rendering logic
        try:
            style_definitions = self.theme_manager.get_definitions(theme_name)
            
            if diagram_type == 'classDiagram':
                return self._render_class_diagram(nodes, edges, style_definitions)
            elif diagram_type.startswith('graph'):
                return self._render_graph_diagram(diagram_type, nodes, edges, style_definitions)
            else:
                return self._render_generic_diagram(diagram_type, nodes, edges, style_definitions)
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return self._generate_error_diagram(f"Rendering failed: {str(e)}")
    
    def _render_class_diagram(self, nodes: List[Dict], edges: List[Dict], style_defs: str) -> str:
        """
        Render STRICTLY VALID class diagram from structured data.
        Fixes all syntax issues identified in the analysis.
        """
        lines = ["classDiagram"]
        
        if style_defs:
            lines.append(f"    {style_defs}")
        
        # FIX 1: Define all classes first to ensure they exist before relationships
        all_class_names = {self._sanitize_id(node['id']) for node in nodes}
        for edge in edges:
            target = self._sanitize_id(edge['target'])
            # Only add target if it's not "(none)" or invalid
            if target.lower() != 'none' and target != '':
                all_class_names.add(target)
        
        # Define all classes first
        for class_name in sorted(list(all_class_names)):
            lines.append(f"    class {class_name}")
        
        # FIX 2: Apply stereotypes and members correctly
        for node in nodes:
            class_name = self._sanitize_id(node['id'])
            semantic_role = node.get('semantic_role', '')
            
            # FIX: Correct stereotype syntax - stereotype goes BEFORE class name
            if semantic_role:
                stereotype = self._infer_stereotype_from_role(semantic_role)
                if stereotype:
                    lines.append(f"    <<{stereotype}>> {class_name}")
            
            # Add methods/attributes if they exist
            methods = node.get('methods', [])
            attributes = node.get('attributes', [])
            
            if attributes:
                for attr in attributes[:3]:  # Limit to avoid clutter
                    lines.append(f"    {class_name} : +{attr}")
            
            if methods:
                for method in methods[:5]:  # Limit to avoid clutter
                    method_name = method.get('name', method) if isinstance(method, dict) else method
                    lines.append(f"    {class_name} : +{method_name}()")
            
            # Apply semantic role styling
            lines.append(f"    class {class_name} {semantic_role}Style")
        
        # FIX 3: Render relationships correctly, skip invalid targets
        for edge in edges:
            source = self._sanitize_id(edge['source'])
            target = self._sanitize_id(edge['target'])
            label = edge.get('label', '')
            relationship_type = edge.get('type', 'uses')
            
            # Skip invalid edges
            if target.lower() in ['none', '(none)', '', 'null']:
                continue
            
            # Use correct relationship syntax
            if relationship_type == 'inherits':
                lines.append(f"    {target} <|-- {source}")
            elif relationship_type == 'implements':
                lines.append(f"    {target} <|.. {source}")
            elif relationship_type == 'composition':
                lines.append(f"    {source} *-- {target}")
            elif relationship_type == 'aggregation':
                lines.append(f"    {source} o-- {target}")
            else:
                # For "uses" and other associations
                if label:
                    lines.append(f"    {source} --> {target} : {label}")
                else:
                    lines.append(f"    {source} --> {target}")
        
        return "\n".join(lines)
    
    def _infer_stereotype_from_role(self, semantic_role: str) -> str:
        """Convert semantic role to appropriate stereotype"""
        role_to_stereotype = {
            'controller': 'Controller',
            'service': 'Service', 
            'model': 'Entity',
            'interface': 'Interface',
            'logic': 'Component',
            'external': 'External'
        }
        return role_to_stereotype.get(semantic_role.lower(), '')
    
    def _render_graph_diagram(self, diagram_type: str, nodes: List[Dict], edges: List[Dict], style_defs: str) -> str:
        """Render graph diagram from structured data"""
        lines = [diagram_type]
        
        if style_defs:
            lines.append(f"    {style_defs}")
        
        # Add nodes
        for node in nodes:
            node_id = self._sanitize_id(node['id'])
            label = self._escape_label(node.get('label', node['id']))
            shape = node.get('shape', '[]')
            
            # Format node with shape
            if shape == '()':
                lines.append(f"    {node_id}((\"{label}\"))")
            elif shape == '{}':
                lines.append(f"    {node_id}{{\"{label}\"}}")
            elif shape == '[]':
                lines.append(f"    {node_id}[\"{label}\"]")
            elif shape == '[[]]':
                lines.append(f"    {node_id}[[\"{label}\"]]")
            elif shape == '([])':
                lines.append(f"    {node_id}([\"{label}\"])")
            else:
                lines.append(f"    {node_id}[\"{label}\"]")
            
            # Apply styling
            semantic_role = node.get('semantic_role', 'logic')
            lines.append(f"    class {node_id} {semantic_role}Style")
        
        # Add edges
        for edge in edges:
            source = self._sanitize_id(edge['source'])
            target = self._sanitize_id(edge['target'])
            edge_label = edge.get('label', '')
            edge_type = edge.get('type', '-->')
            
            if edge_label:
                label_escaped = self._escape_label(edge_label)
                if edge_type == '-->':
                    lines.append(f"    {source} -->|{label_escaped}| {target}")
                elif edge_type == '-.->':
                    lines.append(f"    {source} -.->|{label_escaped}| {target}")
                elif edge_type == '==>':
                    lines.append(f"    {source} ==>|{label_escaped}| {target}")
                else:
                    lines.append(f"    {source} {edge_type}|{label_escaped}| {target}")
            else:
                lines.append(f"    {source} {edge_type} {target}")
        
        return "\n".join(lines)
    
    def _render_generic_diagram(self, diagram_type: str, nodes: List[Dict], edges: List[Dict], style_defs: str) -> str:
        """Fallback renderer for any diagram type"""
        return self._render_graph_diagram(f"graph TD", nodes, edges, style_defs)
    
    def _generate_error_diagram(self, error_message: str) -> str:
        """Generate error diagram when rendering fails"""
        safe_message = self._escape_label(error_message)
        return f"""graph TD
    error["⚠️ Error: {safe_message}"]
    classDef errorStyle fill:#dc3545,stroke:#721c24,stroke-width:3px,color:white
    class error errorStyle"""
    
    def _sanitize_id(self, node_id: str) -> str:
        """Sanitize node ID for Mermaid compatibility"""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(node_id))
        
        # Ensure starts with letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"n_{sanitized}"
        
        return sanitized or "node"
    
    def _escape_label(self, label: str) -> str:
        """Escape special characters in labels for Mermaid"""
        # Escape quotes and other problematic characters
        escaped = str(label).replace('"', '\\"').replace('\\', '\\\\')
        return escaped


class KGDiagramGenerator:
    """
    Knowledge Graph-powered diagram generator that produces factually accurate
    diagrams by querying the relationship database directly.
    
    This handles the DATA DISCOVERY part, then uses MermaidRenderer for visualization.
    """
    
    def __init__(self, db_manager, theme_manager: ThemeManager):
        self.db_manager = db_manager
        self.theme_manager = theme_manager
        self.renderer = MermaidRenderer(theme_manager)
        self.diagram_queries = self._initialize_diagram_queries()
        
        logger.info("KGDiagramGenerator initialized with database and renderer")
    
    def _initialize_diagram_queries(self) -> Dict[str, str]:
        """Pre-built SQL queries for different diagram types"""
        return {
            'call_flow': """
                WITH RECURSIVE call_chain AS (
                    SELECT n.id, n.name, n.type, n.file_path, 0 as depth
                    FROM nodes n WHERE n.id = ?
                    
                    UNION ALL
                    
                    SELECT n.id, n.name, n.type, n.file_path, cc.depth + 1
                    FROM nodes n
                    JOIN edges e ON e.target_id = n.id
                    JOIN call_chain cc ON e.source_id = cc.id
                    WHERE e.type = 'calls' AND cc.depth < ?
                )
                SELECT * FROM call_chain ORDER BY depth, name
            """,
            
            'class_hierarchy': """
                WITH RECURSIVE inheritance AS (
                    SELECT n.id, n.name, n.file_path, 0 as level
                    FROM nodes n WHERE n.type = 'class' AND n.name = ?
                    
                    UNION ALL
                    
                    SELECT n.id, n.name, n.file_path, i.level + 1
                    FROM nodes n
                    JOIN edges e ON e.source_id = n.id
                    JOIN inheritance i ON e.target_id = i.id
                    WHERE e.type = 'inherits' AND i.level < 5
                )
                SELECT * FROM inheritance ORDER BY level, name
            """,
            
            'module_dependencies': """
                SELECT DISTINCT 
                    n1.name as source_module,
                    n1.file_path as source_path,
                    n2.name as target_module,
                    n2.file_path as target_path,
                    e.type as relationship_type
                FROM nodes n1
                JOIN edges e ON e.source_id = n1.id
                JOIN nodes n2 ON e.target_id = n2.id
                WHERE n1.file_path LIKE ? AND e.type IN ('imports', 'calls')
                ORDER BY n1.name, n2.name
            """
        }
    
    def generate_class_hierarchy(self, class_name: str, theme: str = "dark_professional") -> str:
        """Generate class inheritance diagram from Knowledge Graph"""
        try:
            # Query KG for inheritance relationships
            cursor = self.db_manager.connection.cursor()
            results = cursor.execute(
                self.diagram_queries['class_hierarchy'], 
                (class_name, 5)
            ).fetchall()
            
            if not results:
                logger.warning(f"No class hierarchy found for {class_name}")
                return self._generate_empty_diagram("No inheritance relationships found")
            
            # Convert KG data to renderer format
            graph_data = self._build_hierarchy_data(results)
            
            # Use pure renderer for visualization
            return self.renderer.generate('classDiagram', graph_data, theme)
            
        except Exception as e:
            logger.error(f"Class hierarchy generation failed: {e}")
            return self.renderer._generate_error_diagram(f"Generation failed: {str(e)}")
    
    def generate_call_flow(self, function_name: str, max_depth: int = 3, 
                          theme: str = "dark_professional") -> str:
        """Generate function call flow diagram from Knowledge Graph"""
        try:
            # Find the function node
            function_node = self._find_function_node(function_name)
            if not function_node:
                return self._generate_empty_diagram(f"Function '{function_name}' not found")
            
            # Query call relationships
            cursor = self.db_manager.connection.cursor()
            results = cursor.execute(
                self.diagram_queries['call_flow'],
                (function_node['id'], max_depth)
            ).fetchall()
            
            # Convert to renderer format
            graph_data = self._build_call_flow_data(results)
            
            # Use pure renderer
            return self.renderer.generate('graph TD', graph_data, theme)
            
        except Exception as e:
            logger.error(f"Call flow generation failed: {e}")
            return self.renderer._generate_error_diagram(f"Generation failed: {str(e)}")
    
    def generate_module_dependencies(self, module_path: str, theme: str = "dark_professional") -> str:
        """Generate module dependency diagram from Knowledge Graph"""
        try:
            cursor = self.db_manager.connection.cursor()
            results = cursor.execute(
                self.diagram_queries['module_dependencies'],
                (f"%{module_path}%",)
            ).fetchall()
            
            if not results:
                return self._generate_empty_diagram(f"No dependencies found for {module_path}")
            
            # Convert to renderer format
            graph_data = self._build_dependency_data(results)
            
            # Use pure renderer
            return self.renderer.generate('graph LR', graph_data, theme)
            
        except Exception as e:
            logger.error(f"Module dependency generation failed: {e}")
            return self.renderer._generate_error_diagram(f"Generation failed: {str(e)}")
    
    def _build_hierarchy_data(self, hierarchy_results: List[Dict]) -> Dict:
        """Convert KG hierarchy data to renderer format"""
        nodes = []
        edges = []
        
        for row in hierarchy_results:
            node_id = self._sanitize_node_id(row['name'])
            file_name = row['file_path'].split('/')[-1]
            
            nodes.append({
                'id': node_id,
                'label': f"{row['name']}\\n({file_name})",
                'semantic_role': 'logic',
                'shape': '[]'
            })
        
        # Add inheritance edges (derived from recursive query structure)
        for i in range(1, len(hierarchy_results)):
            parent_id = self._sanitize_node_id(hierarchy_results[i-1]['name'])
            child_id = self._sanitize_node_id(hierarchy_results[i]['name'])
            
            edges.append({
                'source': child_id,
                'target': parent_id,
                'type': 'inherits'
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def _build_call_flow_data(self, call_results: List[Dict]) -> Dict:
        """Convert KG call data to renderer format"""
        nodes = []
        edges = []
        
        nodes_by_depth = {}
        for row in call_results:
            depth = row['depth']
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(row)
        
        # Add nodes
        for depth, depth_nodes in nodes_by_depth.items():
            for node in depth_nodes:
                node_id = self._sanitize_node_id(f"{node['name']}_{node['id']}")
                file_name = node['file_path'].split('/')[-1]
                
                nodes.append({
                    'id': node_id,
                    'label': f"{node['name']}()\\n({file_name})",
                    'semantic_role': 'logic' if node['type'] == 'function' else 'interface',
                    'shape': '[]'
                })
        
        # Add call edges between depth levels
        for depth in range(len(nodes_by_depth) - 1):
            current_nodes = nodes_by_depth[depth]
            next_nodes = nodes_by_depth[depth + 1]
            
            for curr in current_nodes:
                for next_node in next_nodes:
                    curr_id = self._sanitize_node_id(f"{curr['name']}_{curr['id']}")
                    next_id = self._sanitize_node_id(f"{next_node['name']}_{next_node['id']}")
                    
                    edges.append({
                        'source': curr_id,
                        'target': next_id,
                        'type': '-->'
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def _build_dependency_data(self, dependency_results: List[Dict]) -> Dict:
        """Convert KG dependency data to renderer format"""
        nodes = []
        edges = []
        seen_modules = set()
        
        for row in dependency_results:
            # Add source module
            if row['source_module'] not in seen_modules:
                nodes.append({
                    'id': self._sanitize_node_id(row['source_module']),
                    'label': row['source_module'],
                    'semantic_role': 'interface',
                    'shape': '[]'
                })
                seen_modules.add(row['source_module'])
            
            # Add target module
            if row['target_module'] not in seen_modules:
                nodes.append({
                    'id': self._sanitize_node_id(row['target_module']),
                    'label': row['target_module'],
                    'semantic_role': 'interface',
                    'shape': '[]'
                })
                seen_modules.add(row['target_module'])
            
            # Add dependency edge
            edges.append({
                'source': self._sanitize_node_id(row['source_module']),
                'target': self._sanitize_node_id(row['target_module']),
                'type': '-->',
                'label': row['relationship_type']
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def _find_function_node(self, function_name: str) -> Optional[Dict]:
        """Find function node in KG"""
        try:
            cursor = self.db_manager.connection.cursor()
            result = cursor.execute("""
                SELECT id, name, file_path, line_start
                FROM nodes 
                WHERE type = 'function' AND name = ?
                ORDER BY file_path
                LIMIT 1
            """, (function_name,)).fetchone()
            
            if result:
                return {
                    'id': result['id'],
                    'name': result['name'],
                    'file_path': result['file_path'],
                    'line_start': result['line_start']
                }
        except Exception as e:
            logger.error(f"Function lookup failed: {e}")
        
        return None
    
    def _sanitize_node_id(self, name: str) -> str:
        """Sanitize node ID for Mermaid"""
        import re
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    
    def _generate_empty_diagram(self, message: str) -> str:
        """Generate empty state diagram"""
        return self.renderer._generate_error_diagram(message)


# Backward compatibility function that intelligently routes requests
def create_mermaid_diagram(diagram_type: str, nodes: List[Dict] = None, edges: List[Dict] = None, 
                          theme: str = "dark_professional", **kwargs) -> str:
    """
    Phase 3 Complete: Intelligent diagram generation.
    
    Auto-detects if this is a KG-powered request (no nodes/edges provided)
    or traditional LLM-based request (nodes/edges provided).
    """
    
    # Phase 3: Intelligent diagram detection
    if nodes is None and edges is None:
        # KG-powered diagram request
        return _create_kg_powered_diagram(diagram_type, theme, **kwargs)
    else:
        # Traditional LLM-based diagram (with pure renderer)
        return _create_traditional_diagram(diagram_type, nodes, edges, theme)


def _create_kg_powered_diagram(diagram_type: str, theme: str, **kwargs) -> str:
    """Create diagrams directly from Knowledge Graph data"""
    # This would need access to database manager - for now return placeholder
    logger.warning("KG-powered diagram generation requires database manager integration")
    renderer = MermaidRenderer()
    return renderer._generate_error_diagram(f"KG diagram type '{diagram_type}' requires database integration")


def _create_traditional_diagram(diagram_type: str, nodes: List[Dict], 
                               edges: List[Dict], theme: str) -> str:
    """Create diagrams from LLM-provided nodes and edges (Pure Renderer)"""
    
    renderer = MermaidRenderer()
    graph_data = {'nodes': nodes, 'edges': edges}
    
    return renderer.generate(diagram_type, graph_data, theme)