"""
CodeWise Tools Package

This package contains specialized tools for the CodeWise agent system.
Each tool is designed to be deterministic, reliable, and maintainable.
"""

__version__ = "1.0.0"

from .mermaid_generator import create_mermaid_diagram

__all__ = ["create_mermaid_diagram"]