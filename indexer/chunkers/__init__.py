"""
Chunkers module for hierarchical code analysis.

This module provides the HierarchicalChunker that replaces the existing
flat AST chunking system with sophisticated bidirectional relationships.
"""

from .hierarchical_chunker import HierarchicalChunker

__all__ = ['HierarchicalChunker']