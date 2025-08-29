"""
Parsers module for hierarchical chunking system.

This module provides unified parsing capabilities using tree-sitter
for consistent AST analysis across multiple programming languages.
"""

from .tree_sitter_parser import TreeSitterFactory

__all__ = ['TreeSitterFactory']