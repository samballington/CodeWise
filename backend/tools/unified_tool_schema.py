"""
Unified Tool Schema for Cerebras SDK Integration

This schema consolidates all CodeWise capabilities into a single, powerful tool
that eliminates LLM decision paralysis while preserving full functionality.
"""

UNIFIED_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_codebase",
            "description": (
                "Search and analyze the codebase using intelligent hybrid search. "
                "Combines semantic search, keyword search, and Knowledge Graph queries. "
                "Use this for code understanding, symbol lookup, concept exploration, "
                "and gathering structural data for diagrams."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language query describing what to find. "
                            "Examples: 'find authenticate_user function', "
                            "'how does user authentication work', "
                            "'show class hierarchy for UserManager'"
                        )
                    },
                    "analysis_mode": {
                        "type": "string",
                        "enum": ["auto", "structural_kg", "semantic_rag", "specific_symbol"],
                        "description": (
                            "Override automatic analysis to force a specific strategy:\n"
                            "- 'auto': Intelligent mode selection (recommended)\n"
                            "- 'structural_kg': Knowledge Graph queries for relationships/diagrams\n"
                            "- 'semantic_rag': Semantic search for conceptual understanding\n"
                            "- 'specific_symbol': Exact symbol search and neighborhood expansion"
                        ),
                        "default": "auto"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "file_type": {
                                "type": "string",
                                "description": "Filter by file extension (e.g., '.py', '.js')"
                            },
                            "directory": {
                                "type": "string", 
                                "description": "Filter by directory path"
                            },
                            "symbol_type": {
                                "type": "string",
                                "enum": ["function", "class", "method", "variable"],
                                "description": "Filter by programming symbol type"
                            }
                        },
                        "description": "Optional filters to narrow search scope"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "generate_diagram",
            "description": (
                "Generate Mermaid diagrams from structural data. This tool should "
                "ONLY be used after query_codebase with analysis_mode='structural_kg' "
                "has provided the factual relationship data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diagram_type": {
                        "type": "string",
                        "enum": ["flowchart", "class_diagram", "sequence", "graph"],
                        "description": "Type of Mermaid diagram to generate"
                    },
                    "structural_data": {
                        "type": "string",
                        "description": (
                            "JSON string containing the structural relationship data "
                            "obtained from query_codebase with structural_kg mode"
                        )
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the diagram"
                    },
                    "theme": {
                        "type": "string",
                        "enum": ["default", "dark", "forest", "neutral"],
                        "description": "Visual theme for the diagram",
                        "default": "default"
                    }
                },
                "required": ["diagram_type", "structural_data"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "examine_files",
            "description": (
                "Read and examine specific source files for detailed analysis. "
                "Use when you need to see actual file contents, understand "
                "implementation details, or analyze specific code sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to examine (max 10 files)",
                        "maxItems": 10
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around important sections",
                        "default": 5,
                        "minimum": 0,
                        "maximum": 20
                    }
                },
                "required": ["file_paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_symbols",
            "description": (
                "Search for specific programming symbols using Knowledge Graph. "
                "Returns symbol definitions with their relationships, callers, "
                "and dependencies for comprehensive analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name": {
                        "type": "string",
                        "description": "Name of symbol to search for (function, class, method, variable)"
                    },
                    "symbol_type": {
                        "type": "string",
                        "enum": ["function", "class", "method", "variable"],
                        "description": "Type of symbol to search for (optional - helps narrow results)"
                    },
                    "include_relationships": {
                        "type": "boolean",
                        "description": "Include caller and dependency relationships",
                        "default": True
                    }
                },
                "required": ["symbol_name"]
            }
        }
    }
]

# Export for easy import
__all__ = ["UNIFIED_TOOL_SCHEMA"]