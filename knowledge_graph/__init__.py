"""
Knowledge Graph module for CodeWise.

Implements the two-pass AST-to-Graph pipeline that extracts structural 
relationships from code and populates the unified SQLite database.

Components:
- SymbolCollector: Pass 1 - Discovers all symbols across codebase
- RelationshipExtractor: Pass 2 - Extracts relationships between symbols
- KGAwareRAG: Enhanced retrieval using graph relationships
"""