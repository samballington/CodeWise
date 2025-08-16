"""
Pydantic models for hierarchical code chunking.

Defines the schema for representing code in a hierarchical structure
with bidirectional relationships and comprehensive metadata.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
import re


class ChunkType(str, Enum):
    """
    Types of code chunks in the hierarchical system.
    
    SYMBOL: Precise definitions (functions, classes, methods, variables)
    BLOCK: Logical code blocks with surrounding context (class bodies, modules)
    SUMMARY: High-level overviews containing references to child chunks
    """
    SYMBOL = "symbol"
    BLOCK = "block" 
    SUMMARY = "summary"


class ChunkBase(BaseModel):
    """
    Base class for all chunk types with common fields.
    
    Provides the foundation for hierarchical relationships and
    consistent metadata across all chunk types.
    """
    id: str = Field(..., description="Unique identifier for this chunk")
    type: ChunkType = Field(..., description="Type of chunk")
    content: str = Field(..., description="The actual code or text content")
    file_path: str = Field(..., description="Path to the source file")
    line_start: int = Field(..., ge=1, description="Starting line number (1-indexed)")
    line_end: int = Field(..., ge=1, description="Ending line number (1-indexed)")
    
    @validator('line_end')
    def line_end_must_be_after_start(cls, v, values):
        """Ensure line_end is not before line_start."""
        if 'line_start' in values and v < values['line_start']:
            raise ValueError('line_end must be >= line_start')
        return v
    
    @validator('id')
    def id_must_be_valid(cls, v):
        """Ensure ID follows consistent naming convention."""
        if not re.match(r'^[a-zA-Z0-9_\-:\.]+$', v):
            raise ValueError('Chunk ID must contain only alphanumeric characters, underscores, hyphens, colons, and dots')
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"


class SymbolChunk(ChunkBase):
    """
    Represents a precise code symbol (function, class, method, variable).
    
    These are the most granular chunks, representing specific code definitions
    that developers typically search for by name.
    """
    type: Literal[ChunkType.SYMBOL] = ChunkType.SYMBOL
    
    # Symbol-specific metadata
    symbol_name: str = Field(..., description="Name of the function/class/variable")
    symbol_type: str = Field(..., description="function|class|method|variable|import|constant")
    parent_chunk_id: Optional[str] = Field(None, description="ID of containing block/summary")
    
    # Function/method specific fields
    parameters: List[str] = Field(default_factory=list, description="Function parameters")
    return_type: Optional[str] = Field(None, description="Return type annotation")
    docstring: Optional[str] = Field(None, description="Documentation string")
    decorators: List[str] = Field(default_factory=list, description="Decorator names")
    
    # Quality metrics
    complexity_score: float = Field(0.0, ge=0.0, le=1.0, description="Complexity score (0-1)")
    
    @validator('symbol_type')
    def symbol_type_must_be_valid(cls, v):
        """Validate symbol type."""
        valid_types = {
            'function', 'class', 'method', 'variable', 
            'import', 'constant', 'property', 'async_function'
        }
        if v not in valid_types:
            raise ValueError(f'symbol_type must be one of: {valid_types}')
        return v
    
    @validator('symbol_name')
    def symbol_name_must_be_valid(cls, v):
        """Ensure symbol name is not empty."""
        if not v.strip():
            raise ValueError('symbol_name cannot be empty')
        return v.strip()


class BlockChunk(ChunkBase):
    """
    Represents a logical code block with surrounding context.
    
    These chunks capture code organization like class bodies, modules,
    or namespace blocks that contain multiple symbols.
    """
    type: Literal[ChunkType.BLOCK] = ChunkType.BLOCK
    
    # Block-specific metadata
    block_type: str = Field(..., description="class_body|module|namespace|package")
    parent_chunk_id: Optional[str] = Field(None, description="ID of containing summary")
    child_chunk_ids: List[str] = Field(default_factory=list, description="IDs of contained symbols")
    
    # Import/export tracking
    imports: List[str] = Field(default_factory=list, description="Import statements")
    exports: List[str] = Field(default_factory=list, description="Exported symbols")
    
    # Quality metrics
    complexity_score: float = Field(0.0, ge=0.0, le=1.0, description="Complexity score (0-1)")
    
    @validator('block_type')
    def block_type_must_be_valid(cls, v):
        """Validate block type."""
        valid_types = {
            'class_body', 'module', 'namespace', 'package',
            'interface', 'enum', 'struct', 'trait'
        }
        if v not in valid_types:
            raise ValueError(f'block_type must be one of: {valid_types}')
        return v
    
    @validator('child_chunk_ids')
    def child_chunk_ids_must_be_unique(cls, v):
        """Ensure child chunk IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError('child_chunk_ids must be unique')
        return v


class SummaryChunk(ChunkBase):
    """
    Represents a high-level overview containing references to child chunks.
    
    These chunks provide file-level or module-level summaries that help
    with contextual understanding and hierarchical navigation.
    """
    type: Literal[ChunkType.SUMMARY] = ChunkType.SUMMARY
    
    # Summary-specific metadata
    summary_type: str = Field(..., description="file|directory|package|module")
    child_chunk_ids: List[str] = Field(..., description="All chunks contained within this summary")
    
    # High-level interface tracking
    key_exports: List[str] = Field(default_factory=list, description="Main exported symbols")
    key_imports: List[str] = Field(default_factory=list, description="Important import dependencies")
    
    # Architectural context
    architecture_notes: str = Field("", description="High-level architectural observations")
    
    @validator('summary_type')
    def summary_type_must_be_valid(cls, v):
        """Validate summary type."""
        valid_types = {'file', 'directory', 'package', 'module', 'library'}
        if v not in valid_types:
            raise ValueError(f'summary_type must be one of: {valid_types}')
        return v
    
    @validator('child_chunk_ids')
    def child_chunk_ids_must_be_unique(cls, v):
        """Ensure child chunk IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError('child_chunk_ids must be unique')
        return v


# Type alias for all chunk types
AnyChunk = Union[SymbolChunk, BlockChunk, SummaryChunk]


class ChunkRelationship(BaseModel):
    """
    Represents a relationship between chunks for persistence.
    
    Used to store bidirectional relationships in metadata storage
    for efficient hierarchical traversal.
    """
    chunk_id: str = Field(..., description="ID of the chunk")
    parent_id: Optional[str] = Field(None, description="ID of parent chunk")
    child_ids: List[str] = Field(default_factory=list, description="IDs of child chunks")
    sibling_ids: List[str] = Field(default_factory=list, description="IDs of sibling chunks")
    chunk_type: ChunkType = Field(..., description="Type of the chunk")
    
    @validator('child_ids', 'sibling_ids')
    def ids_must_be_unique(cls, v):
        """Ensure ID lists are unique."""
        if len(v) != len(set(v)):
            raise ValueError('IDs must be unique')
        return v


class ChunkValidationError(Exception):
    """Custom exception for chunk validation errors."""
    pass


def validate_chunk_hierarchy(chunks: List[AnyChunk]) -> Dict[str, Any]:
    """
    Validate the consistency of a hierarchical chunk structure.
    
    Args:
        chunks: List of chunks to validate
        
    Returns:
        Dictionary with validation results and any errors found
        
    Raises:
        ChunkValidationError: If critical validation errors are found
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'chunk_count': len(chunks),
        'type_distribution': {}
    }
    
    # Build chunk index for efficient lookups
    chunk_index = {chunk.id: chunk for chunk in chunks}
    
    # Count chunk types
    for chunk in chunks:
        chunk_type = chunk.type.value
        validation_results['type_distribution'][chunk_type] = \
            validation_results['type_distribution'].get(chunk_type, 0) + 1
    
    # Validate relationships
    for chunk in chunks:
        # Check parent-child consistency
        if hasattr(chunk, 'parent_chunk_id') and chunk.parent_chunk_id:
            parent_id = chunk.parent_chunk_id
            if parent_id not in chunk_index:
                validation_results['errors'].append(
                    f"Chunk {chunk.id} references non-existent parent {parent_id}"
                )
                validation_results['valid'] = False
            else:
                parent = chunk_index[parent_id]
                if hasattr(parent, 'child_chunk_ids'):
                    if chunk.id not in parent.child_chunk_ids:
                        validation_results['warnings'].append(
                            f"Parent {parent_id} does not list {chunk.id} as child"
                        )
        
        # Check child-parent consistency
        if hasattr(chunk, 'child_chunk_ids'):
            for child_id in chunk.child_chunk_ids:
                if child_id not in chunk_index:
                    validation_results['errors'].append(
                        f"Chunk {chunk.id} references non-existent child {child_id}"
                    )
                    validation_results['valid'] = False
                else:
                    child = chunk_index[child_id]
                    if hasattr(child, 'parent_chunk_id'):
                        if child.parent_chunk_id != chunk.id:
                            validation_results['warnings'].append(
                                f"Child {child_id} does not reference {chunk.id} as parent"
                            )
    
    # Validate line number consistency
    for chunk in chunks:
        if chunk.line_start > chunk.line_end:
            validation_results['errors'].append(
                f"Chunk {chunk.id} has line_start > line_end"
            )
            validation_results['valid'] = False
    
    return validation_results


def create_chunk_id(file_path: str, symbol_name: str = None, 
                   chunk_type: ChunkType = ChunkType.SUMMARY) -> str:
    """
    Generate a consistent chunk ID from file path and symbol information.
    
    Args:
        file_path: Path to the source file
        symbol_name: Name of the symbol (for SYMBOL chunks)
        chunk_type: Type of chunk being created
        
    Returns:
        Unique chunk identifier
    """
    # Sanitize file path for ID usage
    sanitized_path = re.sub(r'[^\w\-_.]', '_', file_path.replace('\\', '/'))
    
    if chunk_type == ChunkType.SUMMARY:
        return f"summary_{sanitized_path}"
    elif chunk_type == ChunkType.SYMBOL and symbol_name:
        sanitized_symbol = re.sub(r'[^\w\-_]', '_', symbol_name)
        return f"symbol_{sanitized_path}::{sanitized_symbol}"
    elif chunk_type == ChunkType.BLOCK:
        return f"block_{sanitized_path}"
    else:
        # Fallback with timestamp
        import time
        timestamp = int(time.time() * 1000) % 10000
        return f"{chunk_type.value}_{sanitized_path}_{timestamp}"