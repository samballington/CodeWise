import sys, types

# Ensure 'ast_chunker' module exists for legacy imports (pytest expects it)
if 'ast_chunker' not in sys.modules:
    sys.modules['ast_chunker'] = types.ModuleType('ast_chunker')

# Import the real implementation
from indexer.ast_chunker import (
    ASTChunker as _Base,
    PythonASTChunker as _Py,
    ConfigChunker as _Cfg,
    MarkdownChunker as _Md,
    SmallFileChunker as _Sm,
    ChunkMetadata as _Meta,
    CodeChunk as _Ck,
)

# Legacy class stubs expected by old tests
class JavaScriptChunker(_Base):
    """Legacy stub inheriting from ASTChunker (JS)."""
    pass

class TypeScriptChunker(_Base):
    """Legacy stub inheriting from ASTChunker (TS)."""
    pass

class PythonChunker(_Base):
    """Legacy stub inheriting from ASTChunker (Py)."""
    pass

PythonASTChunker = _Py
ConfigChunker = _Cfg
MarkdownChunker = _Md
SmallFileChunker = _Sm
ChunkMetadata = _Meta
CodeChunk = _Ck

# Expose all classes on the module for wildcard imports
_current_mod = sys.modules['ast_chunker']
for _cls in (
        _Base,
        _Py,
        JavaScriptChunker,
        TypeScriptChunker,
        PythonChunker,
        _Cfg,
        _Md,
        _Sm,
        _Meta,
        _Ck,
    ):
    setattr(_current_mod, _cls.__name__, _cls)

# Clean export list
__all__ = [
    'ASTChunker',
    'PythonASTChunker',
    'JavaScriptChunker',
    'TypeScriptChunker',
    'PythonChunker',
    'ConfigChunker',
    'MarkdownChunker',
    'SmallFileChunker',
    'ChunkMetadata',
    'CodeChunk',
] 