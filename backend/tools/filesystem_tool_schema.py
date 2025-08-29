"""
Filesystem Tool Schema for CodeWise Agent

This schema defines the navigate_filesystem tool that provides deterministic,
factual exploration of project structure using Knowledge Graph data.

REQ-3.5.1: Tool schema for filesystem navigation operations
"""

FILESYSTEM_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "navigate_filesystem",
            "description": (
                "Your primary tool for listing files, finding files, and exploring the project's "
                "directory structure. Use this for any questions about file locations, directory "
                "contents, or project organization. This tool queries the Knowledge Graph for "
                "instant, accurate results. IMPORTANT: Always check if a path exists before "
                "exploring subdirectories. Use 'find' with patterns instead of guessing paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["list", "find", "tree"],
                        "description": (
                            "'list' to show contents of a specific directory path, "
                            "'find' to search for files matching a pattern anywhere in the codebase, "
                            "'tree' to show hierarchical directory structure overview."
                        )
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "The directory path to list or show tree for (e.g., "
                            "'SWE_Project/obs/src/main/java/com/example/obs/controller'). "
                            "Required for 'list' and 'tree' operations."
                        )
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "A pattern to filter files (e.g., '*Controller.java', '*.py', 'Auth*'). "
                            "Used with 'list' and 'find' operations. For 'find', this is required."
                        )
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Set to true to search recursively through subdirectories. "
                            "Applies to 'list' and 'find' operations."
                        )
                    }
                },
                "required": ["operation"]
            }
        }
    }
]

# Export for easy import
__all__ = ["FILESYSTEM_TOOL_SCHEMA"]