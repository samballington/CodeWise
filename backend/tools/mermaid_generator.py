"""
Mermaid Diagram Generator Tool

A deterministic tool for generating syntactically correct Mermaid.js diagrams
from structured JSON input, replacing prompt-based generation methods.

Based on the merprompt.md template system with proper error handling,
input validation, and style mapping logic.
"""

from typing import Dict, List, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

# Template configuration derived from merprompt.md
TEMPLATE_CONFIG = {
    "Full-Stack Application": {
        "diagram_type": "graph TD",
        "styles": [
            "classDef userStyle fill:#99d98c,stroke:#333,stroke-width:2px",
            "classDef frontendStyle fill:#76c893,stroke:#333,stroke-width:2px", 
            "classDef backendStyle fill:#52b69a,stroke:#333,stroke-width:2px",
            "classDef dbStyle fill:#34a0a4,stroke:#333,stroke-width:2px",
            "classDef externalStyle fill:#d9ed92,stroke:#333,stroke-width:2px",
            "classDef successStyle fill:#28a745,stroke:#155724,stroke-width:3px,color:white",
            "classDef warningStyle fill:#ffc107,stroke:#856404,stroke-width:3px,color:#212529",
            "classDef errorStyle fill:#dc3545,stroke:#721c24,stroke-width:3px,color:white"
        ],
        "style_mappings": {
            # Generic mappings based on keywords and shapes
            "user": "userStyle",
            "admin": "userStyle",
            "frontend": "frontendStyle", 
            "webapp": "frontendStyle",
            "client": "frontendStyle",
            "cdn": "frontendStyle",
            "backend": "backendStyle",
            "server": "backendStyle", 
            "api": "backendStyle",
            "database": "dbStyle",
            "db": "dbStyle",
            "cache": "dbStyle",
            "redis": "dbStyle",
            "postgresql": "dbStyle",
            "mongodb": "dbStyle",
            "external": "externalStyle",
            "payment": "externalStyle",
            "service": "externalStyle",
            "stripe": "externalStyle",
            "sendgrid": "externalStyle",
            # Status-based mappings (highest priority)
            "success": "successStyle",
            "successful": "successStyle",
            "healthy": "successStyle",
            "running": "successStyle",
            "active": "successStyle",
            "online": "successStyle",
            "ok": "successStyle",
            "completed": "successStyle",
            "warning": "warningStyle",
            "caution": "warningStyle",
            "pending": "warningStyle",
            "loading": "warningStyle",
            "busy": "warningStyle",
            "maintenance": "warningStyle",
            "degraded": "warningStyle",
            "error": "errorStyle",
            "failed": "errorStyle",
            "critical": "errorStyle",
            "down": "errorStyle",
            "offline": "errorStyle",
            "crashed": "errorStyle",
            "blocked": "errorStyle",
            "invalid": "errorStyle"
        },
        "shape_mappings": {
            "database": "dbStyle",
            "round-edge": "userStyle"
        }
    },
    
    "API / Microservice Interaction": {
        "diagram_type": "graph TD",
        "styles": [
            "classDef apiStyle fill:#1a759f,stroke:#333,stroke-width:2px,color:#fff",
            "classDef serviceStyle fill:#184e77,stroke:#333,stroke-width:2px,color:#fff", 
            "classDef externalStyle fill:#d9ed92,stroke:#333,stroke-width:2px"
        ],
        "style_mappings": {
            "gateway": "apiStyle",
            "api": "apiStyle",
            "graphql": "apiStyle",
            "service": "serviceStyle",
            "auth": "serviceStyle",
            "authentication": "serviceStyle",
            "products": "serviceStyle",
            "inventory": "serviceStyle",
            "microservice": "serviceStyle",
            "logging": "serviceStyle",
            "external": "externalStyle",
            "third": "externalStyle",
            "queue": "serviceStyle",
            "rabbitmq": "serviceStyle",
            "mongodb": "externalStyle"
        },
        "shape_mappings": {
            "database": "externalStyle",
            "round-edge": "externalStyle",
            "circle": "serviceStyle"
        }
    },
    
    "Database Schema": {
        "diagram_type": "erDiagram",
        "styles": [],  # ER diagrams don't use CSS styles
        "style_mappings": {},
        "shape_mappings": {}
    },
    
    "Internal Component Flow": {
        "diagram_type": "graph TD", 
        "styles": [
            "classDef entrypointStyle fill:#ef476f,stroke:#333,stroke-width:2px,color:white",
            "classDef controllerStyle fill:#f78c6b,stroke:#333,stroke-width:2px",
            "classDef serviceStyle fill:#ffd166,stroke:#333,stroke-width:2px", 
            "classDef modelStyle fill:#06d6a0,stroke:#333,stroke-width:2px",
            "classDef successStyle fill:#28a745,stroke:#155724,stroke-width:3px,color:white",
            "classDef warningStyle fill:#ffc107,stroke:#856404,stroke-width:3px,color:#212529",
            "classDef errorStyle fill:#dc3545,stroke:#721c24,stroke-width:3px,color:white"
        ],
        "style_mappings": {
            "request": "entrypointStyle",
            "input": "entrypointStyle",
            "controller": "controllerStyle",
            "login": "controllerStyle",
            "auth": "controllerStyle",
            "service": "serviceStyle",
            "logic": "serviceStyle", 
            "business": "serviceStyle",
            "model": "modelStyle",
            "data": "modelStyle",
            "database": "modelStyle",
            "users": "modelStyle",
            # Status-based mappings (highest priority)
            "success": "successStyle",
            "successful": "successStyle",
            "healthy": "successStyle",
            "running": "successStyle",
            "active": "successStyle",
            "online": "successStyle",
            "ok": "successStyle",
            "completed": "successStyle",
            "warning": "warningStyle",
            "caution": "warningStyle",
            "pending": "warningStyle",
            "loading": "warningStyle",
            "busy": "warningStyle",
            "maintenance": "warningStyle",
            "degraded": "warningStyle",
            "error": "errorStyle",
            "failed": "errorStyle",
            "critical": "errorStyle",
            "down": "errorStyle",
            "offline": "errorStyle",
            "crashed": "errorStyle",
            "blocked": "errorStyle",
            "invalid": "errorStyle"
        },
        "shape_mappings": {
            "rhombus": "controllerStyle",
            "database": "modelStyle"
        }
    },
    
    "CI/CD Pipeline": {
        "diagram_type": "graph LR",
        "styles": [
            "classDef vcsStyle fill:#fca311,stroke:#333,stroke-width:2px",
            "classDef buildStyle fill:#14213d,stroke:#333,stroke-width:2px,color:white",
            "classDef testStyle fill:#5a189a,stroke:#333,stroke-width:2px,color:white",
            "classDef deployStyle fill:#008000,stroke:#333,stroke-width:2px,color:white"
        ], 
        "style_mappings": {
            "commit": "vcsStyle",
            "push": "vcsStyle",
            "git": "vcsStyle",
            "github": "buildStyle",
            "actions": "buildStyle",
            "build": "buildStyle", 
            "compile": "buildStyle",
            "docker": "buildStyle",
            "test": "testStyle",
            "lint": "testStyle",
            "unit": "testStyle",
            "deploy": "deployStyle",
            "production": "deployStyle",
            "staging": "deployStyle",
            "kubernetes": "deployStyle",
            "k8s": "deployStyle",
            "approval": "buildStyle"
        },
        "shape_mappings": {
            "rhombus": "buildStyle",
            "circle": "deployStyle"
        }
    },
    
    "General System Architecture": {
        "diagram_type": "graph TD",
        "styles": [
            "classDef sourceStyle fill:#8ecae6,stroke:#333,stroke-width:2px", 
            "classDef processStyle fill:#219ebc,stroke:#333,stroke-width:2px,color:white",
            "classDef dataStyle fill:#023047,stroke:#333,stroke-width:2px,color:white",
            "classDef consumerStyle fill:#ffb703,stroke:#333,stroke-width:2px"
        ],
        "style_mappings": {
            "input": "sourceStyle",
            "source": "sourceStyle",
            "ingestion": "sourceStyle",
            "process": "processStyle", 
            "processing": "processStyle",
            "unit": "processStyle",
            "data": "dataStyle",
            "store": "dataStyle",
            "storage": "dataStyle",
            "database": "dataStyle",
            "consumer": "consumerStyle",
            "consumption": "consumerStyle",
            "output": "consumerStyle",
            "system": "consumerStyle"
        },
        "shape_mappings": {
            "round-edge": "sourceStyle",
            "database": "dataStyle",
            "hexagon": "processStyle"
        }
    }
}

def create_mermaid_diagram(data: Dict[str, Any]) -> str:
    """
    Generate a complete Mermaid.js diagram from structured JSON data.
    
    Args:
        data: Dictionary containing:
            - diagram_type: Template type key
            - nodes: List of node dictionaries with id, label, shape, group
            - edges: List of edge dictionaries with from, to, label
            
    Returns:
        Complete Mermaid diagram string with styling
        
    Raises:
        ValueError: If input data is invalid
        Exception: If diagram generation fails
    """
    try:
        # Validate input data
        validation_errors = validate_diagram_input(data)
        if validation_errors:
            error_msg = "; ".join(validation_errors)
            logger.error(f"Mermaid input validation failed: {error_msg}")
            raise ValueError(f"Input validation failed: {error_msg}")
            
        diagram_type = data.get("diagram_type", "General System Architecture")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        logger.info(f"Generating {diagram_type} diagram with {len(nodes)} nodes and {len(edges)} edges")
        
        # Get template configuration
        config = TEMPLATE_CONFIG.get(diagram_type, TEMPLATE_CONFIG["General System Architecture"])
        
        # Handle special case for ER diagrams
        if config["diagram_type"] == "erDiagram":
            result = _generate_er_diagram(nodes, edges)
        else:
            # Generate standard graph diagram
            result = _generate_graph_diagram(nodes, edges, config)
        
        logger.info(f"Successfully generated Mermaid diagram ({len(result)} characters)")
        return result
        
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        logger.error(f"Mermaid diagram generation failed: {e}")
        raise Exception(f"Diagram generation failed: {str(e)}")

def _generate_er_diagram(nodes: List[Dict], edges: List[Dict]) -> str:
    """Generate ER diagram for database schemas"""
    lines = ["erDiagram"]
    
    logger.debug(f"Generating ER diagram with {len(nodes)} entities")
    
    # Generate entity definitions
    for node in nodes:
        node_id = _sanitize_id(node.get("id", ""))
        if not node_id:
            continue
            
        lines.append(f"    {node_id} {{")
        
        # Add primary key field if not present
        label = node.get("label", "")
        if "id" not in label.lower() and "pk" not in label.lower():
            lines.append("        int id PK")
        
        # Add sample field based on entity name
        entity_type = node_id.lower()
        if "user" in entity_type:
            lines.append("        string name")
            lines.append("        string email")
        elif "post" in entity_type:
            lines.append("        string title")
            lines.append("        text content")
        elif "comment" in entity_type:
            lines.append("        text body")
        elif "profile" in entity_type:
            lines.append("        text bio")
        else:
            lines.append("        string name")
            
        lines.append("    }")
    
    # Generate relationships
    for edge in edges:
        from_id = _sanitize_id(edge.get("from", ""))
        to_id = _sanitize_id(edge.get("to", ""))
        label = _escape_label(edge.get("label", "relates"))
        
        if from_id and to_id:
            # Use appropriate relationship cardinality
            if "many" in label.lower() or "multiple" in label.lower():
                lines.append(f"    {from_id} ||--o{{ {to_id} : \"{label}\"")
            elif "one" in label.lower():
                lines.append(f"    {from_id} ||--|| {to_id} : \"{label}\"")
            else:
                lines.append(f"    {from_id} ||--o{{ {to_id} : \"{label}\"")
    
    return "\n".join(lines)

def _generate_graph_diagram(nodes: List[Dict], edges: List[Dict], config: Dict) -> str:
    """Generate flowchart/graph diagram with styling"""
    lines = []
    
    logger.debug(f"Generating {config['diagram_type']} with {len(nodes)} nodes")
    
    # Add diagram type declaration
    lines.append(config["diagram_type"])
    
    # Add style definitions
    if config["styles"]:
        lines.append("    %% --- Style Definitions ---")
        for style in config["styles"]:
            lines.append(f"    {style}")
        lines.append("")
    
    # Group nodes by subgraph
    subgraphs = {}
    standalone_nodes = []
    
    for node in nodes:
        group = node.get("group")
        if group:
            if group not in subgraphs:
                subgraphs[group] = []
            subgraphs[group].append(node)
        else:
            standalone_nodes.append(node)
    
    # Generate standalone nodes first
    for node in standalone_nodes:
        node_def = _generate_node_definition(node)
        if node_def:
            lines.append(f"    {node_def}")
    
    # Generate subgraphs
    for group_name, group_nodes in subgraphs.items():
        sanitized_group = _escape_label(group_name)
        lines.append(f"    subgraph \"{sanitized_group}\"")
        
        # Add direction hint if more than 2 nodes
        if len(group_nodes) > 2:
            lines.append("        direction LR")
            
        for node in group_nodes:
            node_def = _generate_node_definition(node)
            if node_def:
                lines.append(f"        {node_def}")
        lines.append("    end")
    
    if subgraphs or standalone_nodes:
        lines.append("")
    
    # Generate edges
    for edge in edges:
        edge_def = _generate_edge_definition(edge)
        if edge_def:
            lines.append(f"    {edge_def}")
    
    # Apply styles
    if config["styles"]:
        lines.append("")
        lines.append("    %% --- Apply Styles ---")
        
        # Group nodes by style for efficient application
        style_groups = {}
        for node in nodes:
            style = _determine_node_style(node, config)
            if style:
                if style not in style_groups:
                    style_groups[style] = []
                style_groups[style].append(_sanitize_id(node.get("id", "")))
        
        # Generate class applications
        for style, node_ids in style_groups.items():
            valid_ids = [nid for nid in node_ids if nid]
            if valid_ids:
                lines.append(f"    class {','.join(valid_ids)} {style}")
    
    return "\n".join(lines)

def _add_node_icon(label: str, node_id: str = "", shape: str = "") -> str:
    """
    Add appropriate text-based icons to node labels based on content and context.
    Maps node types to visual icons for better semantic representation.
    """
    if not label:
        return label
    
    # Convert to lowercase for keyword matching
    text_content = f"{label} {node_id}".lower()
    
    # Define icon mappings based on node functionality/type
    icon_mappings = [
        # Security/Authentication icons
        (["auth", "login", "security", "permission", "token", "certificate", "ssl", "oauth"], "🔒"),
        
        # Database/Storage icons  
        (["database", "db", "storage", "cache", "redis", "mongodb", "postgresql", "mysql", "data"], "💾"),
        
        # Web/API icons
        (["api", "gateway", "endpoint", "web", "http", "rest", "graphql", "service", "server"], "🌐"),
        
        # Analytics/Data processing icons
        (["analytics", "reporting", "metrics", "dashboard", "chart", "graph", "statistics", "data"], "📊"),
        
        # User/People icons
        (["user", "admin", "customer", "client", "person", "account", "profile"], "👤"),
        
        # Processing/Compute icons
        (["process", "processor", "compute", "calculation", "algorithm", "engine", "worker"], "⚙️"),
        
        # Event/Notification icons
        (["event", "notification", "alert", "message", "queue", "publish", "subscribe", "trigger"], "📢"),
        
        # External/Third-party icons
        (["external", "third-party", "payment", "stripe", "aws", "gcp", "azure", "cdn"], "🔗")
    ]
    
    # Find the first matching icon category
    for keywords, icon in icon_mappings:
        if any(keyword in text_content for keyword in keywords):
            # Only add icon if not already present
            if not any(existing_icon in label for existing_icon in ['🔒', '💾', '🌐', '📊', '👤', '⚙️', '📢', '🔗']):
                return f"{icon} {label}"
            break
    
    return label

def _generate_node_definition(node: Dict) -> str:
    """Generate Mermaid node definition with proper shape syntax and icon integration"""
    node_id = _sanitize_id(node.get("id", ""))
    if not node_id:
        return ""
        
    original_label = node.get("label", node_id)
    shape = node.get("shape", "square")
    
    # Add appropriate icon to the label
    enhanced_label = _add_node_icon(original_label, node_id, shape)
    escaped_label = _escape_label(enhanced_label)
    
    # Map shapes to Mermaid syntax (following merprompt.md specifications)
    shape_syntax = {
        "square": f"{node_id}[\"{escaped_label}\"]",
        "round-edge": f"{node_id}([{escaped_label}])",
        "database": f"{node_id}[({escaped_label})]", 
        "rhombus": f"{node_id}{{{escaped_label}}}",
        "circle": f"{node_id}(({escaped_label}))",
        "hexagon": f"{node_id}{{{{  {escaped_label}  }}}}"
    }
    
    result = shape_syntax.get(shape, f"{node_id}[\"{escaped_label}\"]")
    logger.debug(f"Generated node: {result}")
    return result

def _determine_arrow_type(edge: Dict) -> str:
    """
    Determine appropriate arrow type based on relationship context.
    Maps relationship keywords to semantic arrow styles.
    """
    label = edge.get("label", "").lower()
    
    # Data flow arrows (thick) - for data transfer, flow, sending
    data_flow_keywords = ["data flow", "sends", "transfers", "flows", "pushes", "streams", "pipes"]
    if any(keyword in label for keyword in data_flow_keywords):
        return "==>"
    
    # Optional/conditional arrows (dotted) - for optional, conditional, fallback
    optional_keywords = ["optional", "conditional", "fallback", "backup", "maybe", "if needed", "when available"]
    if any(keyword in label for keyword in optional_keywords):
        return "--.->"
    
    # Bidirectional arrows - for communication, sync, exchange
    bidirectional_keywords = ["communicates", "syncs", "exchanges", "handshake", "negotiates", "two-way"]
    if any(keyword in label for keyword in bidirectional_keywords):
        return "<-->"
    
    # Blocked/error arrows - for error, failure, blocked paths
    blocked_keywords = ["error", "fails", "blocks", "denies", "rejects", "aborts", "crashes"]
    if any(keyword in label for keyword in blocked_keywords):
        return "--x"
    
    # Default to basic arrow
    return "-->"

def _enhance_edge_label(label: str) -> str:
    """
    Enhance edge labels with more descriptive text and visual indicators (emojis).
    Replaces generic labels with specific ones and adds contextual emojis.
    """
    if not label:
        return label
    
    enhanced_label = label.lower()
    
    # Replace generic labels with more specific ones
    label_replacements = {
        "connects to": "communicates with",
        "connects": "communicates with", 
        "uses": "depends on",
        "calls": "invokes",
        "sends to": "transmits to",
        "gets from": "retrieves from",
        "updates": "modifies",
        "creates": "generates",
        "deletes": "removes"
    }
    
    # Apply label replacements
    for generic, specific in label_replacements.items():
        if generic in enhanced_label:
            enhanced_label = enhanced_label.replace(generic, specific)
    
    # Add visual indicators (emojis) based on relationship context
    emoji_mappings = [
        # Data flow indicators
        (["data flow", "sends", "transfers", "flows", "pushes", "streams", "pipes", "transmits"], "📊"),
        
        # Authentication/security indicators  
        (["auth", "authenticate", "login", "verify", "secure", "permission", "authorize"], "🔒"),
        
        # Event/notification indicators
        (["event", "notify", "alert", "trigger", "signal", "emit", "broadcast", "publish"], "⚡"),
        
        # API/communication indicators
        (["api", "request", "response", "communicates", "invokes", "call"], "🔗"),
        
        # Database/storage indicators
        (["database", "store", "save", "persist", "retrieve", "query", "fetch"], "💾"),
        
        # Processing indicators
        (["process", "compute", "calculate", "analyze", "transform", "generate"], "⚙️"),
        
        # Error/failure indicators
        (["error", "fail", "exception", "crash", "abort", "reject"], "❌"),
        
        # Success/completion indicators  
        (["success", "complete", "finish", "done", "confirm", "validate"], "✅"),
    ]
    
    # Find the first matching emoji category
    for keywords, emoji in emoji_mappings:
        if any(keyword in enhanced_label for keyword in keywords):
            enhanced_label = f"{emoji} {enhanced_label}"
            break
    
    # Restore original capitalization pattern
    if label.istitle() or (label and label[0].isupper()):
        enhanced_label = enhanced_label.title()
    elif label.isupper():
        enhanced_label = enhanced_label.upper()
    
    return enhanced_label

def _generate_edge_definition(edge: Dict) -> str:
    """Generate Mermaid edge definition with enhanced arrow types and richer labels"""
    from_id = _sanitize_id(edge.get("from", ""))
    to_id = _sanitize_id(edge.get("to", ""))
    
    if not from_id or not to_id:
        return ""
    
    # Determine appropriate arrow type based on context
    arrow_type = _determine_arrow_type(edge)
    
    label = edge.get("label")
    if label:
        # Enhance the label with better descriptors and emojis
        enhanced_label = _enhance_edge_label(label)
        escaped_label = _escape_label(enhanced_label)
        result = f"{from_id} {arrow_type}|{escaped_label}| {to_id}"
    else:
        result = f"{from_id} {arrow_type} {to_id}"
    
    logger.debug(f"Generated edge: {result} (arrow type: {arrow_type})")
    return result

def _determine_node_style(node: Dict, config: Dict) -> Optional[str]:
    """
    Determine appropriate style for a node based on keywords and shape.
    Uses intelligent matching with priority: shape > status keywords > generic keywords.
    """
    node_id = node.get("id", "").lower()
    label = node.get("label", "").lower() 
    shape = node.get("shape", "")
    
    # Check shape mappings first (highest priority)
    if shape in config["shape_mappings"]:
        return config["shape_mappings"][shape]
    
    # Define status keywords with highest priority
    status_keywords = {
        "success", "successful", "healthy", "running", "active", "online", "ok", "completed",
        "warning", "caution", "pending", "loading", "busy", "maintenance", "degraded",
        "error", "failed", "critical", "down", "offline", "crashed", "blocked", "invalid"
    }
    
    # Check status keywords first (second highest priority)
    for keyword, style in config["style_mappings"].items():
        if keyword in status_keywords and (keyword in node_id or keyword in label):
            return style
    
    # Check other keyword mappings (search in both id and label)
    for keyword, style in config["style_mappings"].items():
        if keyword not in status_keywords and (keyword in node_id or keyword in label):
            return style
    
    # Default to first available style if no match found
    if config["styles"]:
        # Extract style name from classDef definition
        first_style = config["styles"][0]
        # Parse "classDef styleName ..." to get "styleName"
        parts = first_style.split()
        if len(parts) >= 2 and parts[0] == "classDef":
            return parts[1]
        
    return None

def _sanitize_id(node_id: str) -> str:
    """
    Sanitize node ID to be Mermaid-safe.
    Following merprompt.md rules: alphanumeric only, must start with letter.
    """
    if not node_id:
        return ""
        
    # Remove special characters, keep alphanumeric and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', str(node_id))
    
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "N" + sanitized
        
    return sanitized or "Node"

def _escape_label(label: str) -> str:
    """
    Escape special characters in labels for Mermaid.
    Following merprompt.md specification: use HTML entities for quotes.
    """
    if not label:
        return ""
        
    # Convert to string and handle common escape cases
    escaped = str(label)
    
    # Replace quotes with HTML entities (per merprompt.md rules)
    escaped = escaped.replace('"', '&quot;')
    
    # Replace other problematic characters
    escaped = escaped.replace('<', '&lt;')
    escaped = escaped.replace('>', '&gt;')
    
    return escaped

def validate_diagram_input(data: Dict[str, Any]) -> List[str]:
    """
    Validate input data and return list of validation errors.
    Comprehensive validation following the tool schema specification.
    """
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Input must be a dictionary")
        return errors
    
    # Check required fields
    required_fields = ["diagram_type", "nodes", "edges"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate diagram type
    diagram_type = data.get("diagram_type")
    if diagram_type and diagram_type not in TEMPLATE_CONFIG:
        valid_types = list(TEMPLATE_CONFIG.keys())
        errors.append(f"Unknown diagram_type: '{diagram_type}'. Valid types: {valid_types}")
    
    # Validate nodes
    nodes = data.get("nodes")
    if nodes is not None:
        if not isinstance(nodes, list):
            errors.append("Field 'nodes' must be a list")
        elif len(nodes) == 0:
            errors.append("At least one node is required")
        else:
            # Validate each node
            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"Node {i} must be a dictionary")
                    continue
                    
                if "id" not in node:
                    errors.append(f"Node {i} missing required field: 'id'")
                elif not str(node["id"]).strip():
                    errors.append(f"Node {i} has empty 'id' field")
                    
                if "label" not in node:
                    errors.append(f"Node {i} missing required field: 'label'")
                elif not str(node["label"]).strip():
                    errors.append(f"Node {i} has empty 'label' field")
                
                # Validate shape if provided
                shape = node.get("shape")
                valid_shapes = ["square", "round-edge", "database", "rhombus", "circle", "hexagon"]
                if shape and shape not in valid_shapes:
                    errors.append(f"Node {i} has invalid shape: '{shape}'. Valid shapes: {valid_shapes}")
    
    # Validate edges
    edges = data.get("edges")
    if edges is not None:
        if not isinstance(edges, list):
            errors.append("Field 'edges' must be a list")
        else:
            # Collect node IDs for reference validation
            nodes_list = data.get("nodes", [])
            node_ids = {str(node.get("id", "")).strip() for node in nodes_list if isinstance(node, dict) and node.get("id")}
            
            # Validate each edge
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"Edge {i} must be a dictionary")
                    continue
                    
                if "from" not in edge:
                    errors.append(f"Edge {i} missing required field: 'from'")
                elif str(edge["from"]).strip() not in node_ids:
                    errors.append(f"Edge {i} references unknown source node: '{edge['from']}'")
                    
                if "to" not in edge:
                    errors.append(f"Edge {i} missing required field: 'to'")
                elif str(edge["to"]).strip() not in node_ids:
                    errors.append(f"Edge {i} references unknown target node: '{edge['to']}'")
    
    return errors

# Additional utility functions for advanced features

def get_available_diagram_types() -> List[str]:
    """Get list of available diagram types"""
    return list(TEMPLATE_CONFIG.keys())

def get_template_info(diagram_type: str) -> Dict[str, Any]:
    """Get detailed information about a specific template"""
    if diagram_type not in TEMPLATE_CONFIG:
        raise ValueError(f"Unknown diagram type: {diagram_type}")
    
    config = TEMPLATE_CONFIG[diagram_type]
    return {
        "diagram_type": diagram_type,
        "mermaid_type": config["diagram_type"],
        "supports_styling": bool(config["styles"]),
        "style_count": len(config["styles"]),
        "keyword_mappings": list(config["style_mappings"].keys()),
        "shape_mappings": list(config["shape_mappings"].keys())
    }

def preview_node_styling(nodes: List[Dict], diagram_type: str) -> Dict[str, str]:
    """Preview what styles will be applied to nodes"""
    if diagram_type not in TEMPLATE_CONFIG:
        return {}
    
    config = TEMPLATE_CONFIG[diagram_type]
    result = {}
    
    for node in nodes:
        node_id = node.get("id", "")
        style = _determine_node_style(node, config)
        result[node_id] = style or "default"
    
    return result