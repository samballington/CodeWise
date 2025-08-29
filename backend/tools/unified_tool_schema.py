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
    }
]

# Export for easy import
__all__ = ["UNIFIED_TOOL_SCHEMA"]