"""
Unified Content Schemas for Component-Based UI System

This module provides Pydantic models for all content block types that can be
rendered in the unified UI system. Each schema corresponds to a specific 
React component in the frontend.

Design Principles:
- Validation-first: All data is validated at the boundary
- Component-mapping: Each block type maps 1:1 to a UI component
- Self-documenting: Rich field descriptions for API documentation
- Future-proof: Extensible design for new block types

Architecture:
- Individual block schemas inherit discriminated union pattern
- UnifiedAgentResponse is the top-level container
- ContentBlock union type enables type-safe rendering
"""

from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Literal, Optional
from enum import Enum


class ComponentType(str, Enum):
    """
    Semantic types for code components detected during analysis.
    
    These types drive both UI presentation and architectural validation,
    providing consistent categorization across different programming languages.
    """
    CLASS = "class"
    INTERFACE = "interface"
    SERVICE = "service"
    CONTROLLER = "controller"
    REPOSITORY = "repository"
    ENTITY = "entity"
    FUNCTION = "function"
    MODULE = "module"
    COMPONENT = "component"
    UTILITY = "utility"


class CodeComponent(BaseModel):
    """
    Represents a single analyzed code component with metadata.
    
    This model captures the essential information about a code component
    that enables both display and navigation in the UI.
    """
    name: str = Field(
        ..., 
        description="The name of the class, function, or component",
        min_length=1
    )
    path: str = Field(
        ..., 
        description="Relative file path from project root to the component's source",
        min_length=1
    )
    component_type: ComponentType = Field(
        ..., 
        description="Semantic type of the component for styling and grouping"
    )
    purpose: str = Field(
        ..., 
        description="Concise, one-sentence explanation of the component's responsibility",
        min_length=10,
        max_length=200
    )
    key_methods: List[str] = Field(
        default_factory=list, 
        description="List of important method signatures or key functionality"
    )
    line_start: Optional[int] = Field(
        None, 
        description="Starting line number for click-to-navigate functionality",
        ge=1
    )
    line_end: Optional[int] = Field(
        None, 
        description="Ending line number for component scope",
        ge=1
    )
    
    @validator('line_end')
    def validate_line_range(cls, v, values):
        """Ensure line_end is greater than line_start if both are provided."""
        if v is not None and 'line_start' in values and values['line_start'] is not None:
            if v <= values['line_start']:
                raise ValueError('line_end must be greater than line_start')
        return v


# === Individual Content Block Types ===

class TextBlock(BaseModel):
    """
    Standard text content block for explanations and narrative text.
    
    Used for introductions, conclusions, explanations, and any content
    that doesn't require specialized formatting or interaction.
    """
    block_type: Literal["text"] = "text"
    content: str = Field(
        ..., 
        description="Markdown-formatted text content",
        min_length=1
    )


class ComponentAnalysisBlock(BaseModel):
    """
    Structured display of analyzed code components.
    
    Renders as an interactive table or card layout showing component
    metadata with navigation capabilities to source code.
    """
    block_type: Literal["component_analysis"] = "component_analysis"
    title: str = Field(
        default="Key Components Analysis",
        description="Display title for the component analysis section"
    )
    components: List[CodeComponent] = Field(
        ...,
        description="List of analyzed components with metadata",
        min_items=1
    )
    show_line_numbers: bool = Field(
        default=True,
        description="Whether to display line numbers in the component list"
    )
    grouping: Optional[Literal["type", "path", "none"]] = Field(
        default="type",
        description="How to group components in the display"
    )


class MermaidDiagramBlock(BaseModel):
    """
    Mermaid.js diagram block for architectural visualizations.
    
    Renders interactive diagrams with zoom, pan, and export capabilities.
    Integrates with existing mermaid rendering infrastructure.
    """
    block_type: Literal["mermaid_diagram"] = "mermaid_diagram"
    title: str = Field(
        default="Architecture Diagram",
        description="Display title for the diagram"
    )
    mermaid_code: str = Field(
        ...,
        description="Complete, valid Mermaid syntax for the diagram",
        min_length=10
    )
    diagram_type: Optional[str] = Field(
        None,
        description="Type of Mermaid diagram (graph, flowchart, etc.) for UI optimization"
    )
    interactive: bool = Field(
        default=True,
        description="Whether the diagram supports interactive features"
    )
    
    @validator('mermaid_code')
    def validate_mermaid_syntax(cls, v):
        """Basic validation that content looks like Mermaid syntax."""
        v = v.strip()
        if not any(keyword in v for keyword in ['graph', 'flowchart', 'classDiagram', 'sequenceDiagram', 'erDiagram']):
            raise ValueError('Mermaid code must contain a valid diagram type declaration')
        return v


class MarkdownTableBlock(BaseModel):
    """
    Tabular data display with interactive features.
    
    Provides sorting, filtering, and export capabilities for structured data.
    Automatically handles responsive design and large datasets.
    """
    block_type: Literal["markdown_table"] = "markdown_table"
    title: str = Field(
        default="Data Table",
        description="Display title for the table"
    )
    headers: List[str] = Field(
        ...,
        description="Column headers for the table",
        min_items=1
    )
    rows: List[List[str]] = Field(
        ...,
        description="Table data rows, each row must match header count",
        min_items=1
    )
    sortable: bool = Field(
        default=True,
        description="Whether columns can be sorted"
    )
    searchable: bool = Field(
        default=True,
        description="Whether table content can be searched/filtered"
    )
    
    @validator('rows')
    def validate_row_columns(cls, v, values):
        """Ensure all rows have the same number of columns as headers."""
        if 'headers' in values:
            expected_cols = len(values['headers'])
            for i, row in enumerate(v):
                if len(row) != expected_cols:
                    raise ValueError(f'Row {i} has {len(row)} columns, expected {expected_cols}')
        return v


class CodeSnippetBlock(BaseModel):
    """
    Syntax-highlighted code display with interactive features.
    
    Supports multiple programming languages, copy-to-clipboard,
    and optional line highlighting for specific code sections.
    """
    block_type: Literal["code_snippet"] = "code_snippet"
    title: str = Field(
        ...,
        description="Descriptive title for the code snippet",
        min_length=1
    )
    language: str = Field(
        ...,
        description="Programming language for syntax highlighting",
        min_length=1
    )
    code: str = Field(
        ...,
        description="Raw code content without markdown fences",
        min_length=1
    )
    highlight_lines: Optional[List[int]] = Field(
        None,
        description="Line numbers to highlight (1-indexed)"
    )
    show_line_numbers: bool = Field(
        default=True,
        description="Whether to display line numbers"
    )
    copyable: bool = Field(
        default=True,
        description="Whether to show copy-to-clipboard button"
    )
    
    @validator('highlight_lines')
    def validate_highlight_lines(cls, v, values):
        """Ensure highlighted lines are within code bounds."""
        if v is not None and 'code' in values:
            max_line = len(values['code'].split('\n'))
            for line_num in v:
                if line_num < 1 or line_num > max_line:
                    raise ValueError(f'Highlight line {line_num} is out of range (1-{max_line})')
        return v


# === Union Type for All Content Blocks ===

ContentBlock = Union[
    TextBlock,
    ComponentAnalysisBlock,
    MermaidDiagramBlock,
    MarkdownTableBlock,
    CodeSnippetBlock
]


# === Top-Level Response Container ===

class UnifiedAgentResponse(BaseModel):
    """
    Top-level container for all agent responses in the unified UI system.
    
    This is the root object that gets serialized to JSON and sent to the
    frontend. It contains an ordered list of content blocks that will be
    rendered sequentially by the MasterRenderer component.
    
    The response structure is designed for:
    - Streaming support (blocks can be sent incrementally)
    - Type safety (all blocks are validated)
    - UI flexibility (presentation logic stays in frontend)
    - Extensibility (new block types can be added)
    """
    response: List[ContentBlock] = Field(
        ...,
        description="Ordered list of content blocks to render",
        min_items=1
    )
    
    def get_block_types(self) -> List[str]:
        """
        Get list of all block types in this response.
        
        Useful for frontend optimization and analytics.
        """
        return [block.block_type for block in self.response]
    
    def get_blocks_by_type(self, block_type: str) -> List[ContentBlock]:
        """
        Filter blocks by type.
        
        Args:
            block_type: The block type to filter for
            
        Returns:
            List of blocks matching the specified type
        """
        return [block for block in self.response if block.block_type == block_type]
    
    def validate_content_quality(self) -> Dict[str, List[str]]:
        """
        Validate content quality and return recommendations.
        
        Performs heuristic checks for content quality and structure,
        returning actionable feedback for improvement.
        
        Returns:
            Dictionary with 'warnings' and 'suggestions' keys
        """
        warnings = []
        suggestions = []
        
        # Check for logical flow
        block_types = self.get_block_types()
        if block_types and block_types[0] != "text":
            suggestions.append("Consider starting with a TextBlock for introduction")
        
        # Check for component analysis without diagrams
        has_components = "component_analysis" in block_types
        has_diagram = "mermaid_diagram" in block_types
        if has_components and not has_diagram:
            suggestions.append("Consider adding a MermaidDiagramBlock to visualize component relationships")
        
        # Check for overly long text blocks
        for i, block in enumerate(self.response):
            if block.block_type == "text" and len(block.content) > 1000:
                warnings.append(f"TextBlock {i} is very long ({len(block.content)} chars) - consider splitting")
        
        return {
            "warnings": warnings,
            "suggestions": suggestions
        }


# === Validation Utilities ===

def validate_response_structure(data: dict) -> UnifiedAgentResponse:
    """
    Validate and parse a raw response dictionary into a UnifiedAgentResponse.
    
    This function provides the main entry point for response validation
    with comprehensive error reporting for debugging.
    
    Args:
        data: Raw dictionary data to validate
        
    Returns:
        Validated UnifiedAgentResponse object
        
    Raises:
        ValidationError: If validation fails with detailed error information
    """
    try:
        return UnifiedAgentResponse.model_validate(data)
    except Exception as e:
        # Re-raise with additional context for debugging
        raise ValueError(f"Response validation failed: {str(e)}. Data structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")


def create_error_response(error_message: str, include_debug: bool = False) -> UnifiedAgentResponse:
    """
    Create a standardized error response for display to users.
    
    Args:
        error_message: User-friendly error description
        include_debug: Whether to include debugging information
        
    Returns:
        Valid UnifiedAgentResponse containing error information
    """
    content = f"## System Error\n\n{error_message}"
    
    if include_debug:
        import traceback
        content += f"\n\n### Debug Information\n```\n{traceback.format_exc()}\n```"
    
    return UnifiedAgentResponse(
        response=[
            TextBlock(content=content)
        ]
    )