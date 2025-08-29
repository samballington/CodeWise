"""
Unified Content Schemas for Component-Based UI System

This package provides Pydantic models for structured content blocks that enable
reliable, type-safe communication between the backend AI agent and frontend UI.

Architecture Philosophy:
- Data-first rendering: Backend generates structured data, frontend handles presentation
- Type safety: Full TypeScript/Pydantic type validation end-to-end  
- Component isolation: Each block type maps to a specific UI component
- Extensibility: New block types can be added without breaking existing functionality

Usage:
    from backend.schemas.ui_schemas import UnifiedAgentResponse, TextBlock
    
    response = UnifiedAgentResponse(
        response=[
            TextBlock(content="Analysis complete"),
            ComponentAnalysisBlock(title="Results", components=[...])
        ]
    )
"""

from .ui_schemas import (
    # Individual Content Blocks
    TextBlock,
    ComponentAnalysisBlock, 
    MermaidDiagramBlock,
    MarkdownTableBlock,
    CodeSnippetBlock,
    
    # Supporting Models
    CodeComponent,
    
    # Top-Level Response
    UnifiedAgentResponse,
    ContentBlock,
)

__all__ = [
    "TextBlock",
    "ComponentAnalysisBlock",
    "MermaidDiagramBlock", 
    "MarkdownTableBlock",
    "CodeSnippetBlock",
    "CodeComponent",
    "UnifiedAgentResponse",
    "ContentBlock",
]