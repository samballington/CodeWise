"""
Theme and Role Schemas for Mermaid Rendering System

This module provides Pydantic models for semantic role-based theming,
eliminating hardcoded styling and enabling flexible, validated themes.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class SemanticRole(str, Enum):
    """
    Semantic roles for code components that drive intelligent styling.
    
    These roles are assigned by the LLM based on component function,
    not keywords, providing more accurate and consistent styling.
    """
    LOGIC = "logic"
    IO = "io" 
    INTERFACE = "interface"
    STORAGE = "storage"
    TEST = "test"
    EXTERNAL = "external"


class StyleRole(BaseModel):
    """
    Visual styling configuration for a semantic role.
    
    Provides validated color schemes and styling options with
    proper hex color validation and optional text color override.
    """
    fill_color: str = Field(..., pattern=r'^#[0-9a-fA-F]{6}$', 
                           description="Background fill color in hex format")
    stroke_color: str = Field(..., pattern=r'^#[0-9a-fA-F]{6}$',
                             description="Border stroke color in hex format")
    stroke_width: str = Field(default="2px", 
                             description="Border width (e.g., '2px', '3px')")
    text_color: Optional[str] = Field(None, pattern=r'^#[0-9a-fA-F]{6}$',
                                     description="Optional text color override")
    
    def get_mermaid_classdef(self, role: SemanticRole) -> str:
        """
        Generate Mermaid classDef string for this style role.
        
        Args:
            role: The semantic role this style applies to
            
        Returns:
            Mermaid classDef string ready for diagram injection
        """
        color_attr = f",color:{self.text_color}" if self.text_color else ""
        return (f"classDef {role.value}Style "
                f"fill:{self.fill_color},"
                f"stroke:{self.stroke_color},"
                f"stroke-width:{self.stroke_width}"
                f"{color_attr}")


class Theme(BaseModel):
    """
    Complete theme configuration with semantic role mappings.
    
    A theme defines the visual appearance for all semantic roles,
    enabling consistent styling across all diagram types.
    """
    name: str = Field(..., description="Human-readable theme name")
    description: str = Field(..., description="Theme description and use case")
    roles: Dict[SemanticRole, StyleRole] = Field(..., 
                                                description="Role to style mappings")
    
    def get_mermaid_classdef(self, role: SemanticRole) -> str:
        """
        Get Mermaid classDef for a specific role.
        
        Args:
            role: The semantic role to get styling for
            
        Returns:
            Mermaid classDef string
            
        Raises:
            KeyError: If role is not defined in this theme
        """
        if role not in self.roles:
            raise KeyError(f"Role {role.value} not defined in theme {self.name}")
        
        return self.roles[role].get_mermaid_classdef(role)
    
    def get_all_classdefs(self) -> str:
        """
        Get all Mermaid classDef strings for this theme.
        
        Returns:
            All classDef strings joined with newlines
        """
        classdefs = []
        for role in self.roles:
            classdefs.append(self.get_mermaid_classdef(role))
        return "\n    ".join(classdefs)


class MermaidConfig(BaseModel):
    """
    Configuration for Mermaid diagram generation.
    
    Controls diagram behavior and feature enablement while
    maintaining separation between content and presentation.
    """
    diagram_type: str = Field(..., description="Type of diagram to generate")
    theme_name: str = Field(..., description="Theme to apply")
    enable_icons: bool = Field(default=True, 
                              description="Whether to include icons in nodes")
    enable_enhanced_labels: bool = Field(default=True,
                                        description="Whether to use enhanced node labels")