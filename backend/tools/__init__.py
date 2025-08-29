"""
CodeWise Tools Package

This package contains specialized tools for the CodeWise agent system.
Each tool is designed to be deterministic, reliable, and maintainable.
"""

__version__ = "1.0.0"

# Defer imports to avoid circular dependencies and allow individual tool imports
__all__ = ["create_mermaid_diagram", "query_codebase", "QueryRouter"]