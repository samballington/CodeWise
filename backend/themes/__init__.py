"""
Semantic Theming System for CodeWise Mermaid Generation

This module provides a complete theming system that replaces hardcoded
template styling with flexible, semantic role-based themes.
"""

from .schemas import SemanticRole, StyleRole, Theme, MermaidConfig
from .theme_manager import ThemeManager

__all__ = [
    'SemanticRole',
    'StyleRole', 
    'Theme',
    'MermaidConfig',
    'ThemeManager'
]

__version__ = "1.0.0"