"""
Theme Manager for Mermaid Rendering System

Provides centralized theme loading, validation, and Mermaid generation
with caching for performance and error handling for robustness.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, List
from .schemas import Theme, SemanticRole, MermaidConfig

logger = logging.getLogger(__name__)


class ThemeManager:
    """
    Centralized theme management with caching and validation.
    
    Handles loading themes from YAML, validating them against Pydantic schemas,
    and providing efficient access to theme definitions for Mermaid generation.
    """
    
    def __init__(self, themes_file: str = None):
        """
        Initialize ThemeManager with theme file location.
        
        Args:
            themes_file: Path to themes.yaml file. If None, uses default location.
        """
        if themes_file is None:
            # Default to themes.yaml in same directory as this module
            themes_file = Path(__file__).parent / "themes.yaml"
        
        self.themes_file = Path(themes_file)
        self._themes_cache: Dict[str, Theme] = {}
        self._load_themes()
    
    def _load_themes(self) -> None:
        """
        Load and validate themes from YAML file.
        
        Raises:
            FileNotFoundError: If themes file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValidationError: If theme data doesn't match schema
        """
        if not self.themes_file.exists():
            raise FileNotFoundError(
                f"Themes file not found: {self.themes_file}. "
                "Please ensure themes.yaml exists in the themes directory."
            )
        
        try:
            with open(self.themes_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data or 'themes' not in data:
                raise ValueError("Invalid themes file: missing 'themes' key")
            
            # Validate and cache each theme
            for theme_name, theme_data in data['themes'].items():
                try:
                    validated_theme = Theme(**theme_data)
                    self._themes_cache[theme_name] = validated_theme
                    logger.debug(f"Loaded theme: {theme_name}")
                except Exception as e:
                    logger.error(f"Failed to load theme '{theme_name}': {e}")
                    # Continue loading other themes even if one fails
                    continue
            
            logger.info(f"Successfully loaded {len(self._themes_cache)} themes")
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {self.themes_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading themes from {self.themes_file}: {e}")
            raise
    
    def get_theme(self, theme_name: str) -> Optional[Theme]:
        """
        Get a theme by name with case-insensitive lookup.
        
        Args:
            theme_name: Name of the theme to retrieve
            
        Returns:
            Theme object if found, None otherwise
        """
        # Try exact match first
        if theme_name in self._themes_cache:
            return self._themes_cache[theme_name]
        
        # Try case-insensitive match
        lower_name = theme_name.lower()
        for name, theme in self._themes_cache.items():
            if name.lower() == lower_name:
                return theme
        
        logger.warning(f"Theme '{theme_name}' not found. Available themes: {self.available_themes()}")
        return None
    
    def get_definitions(self, theme_name: str) -> str:
        """
        Get complete Mermaid theme definitions for a theme.
        
        DEPRECATED: This method was designed for the old format.
        New format uses simple classDef at the end without init blocks.
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            Empty string (classDef definitions now handled by mermaid generator)
        """
        # Return empty - we now generate classDef definitions in the mermaid generator
        # to match the user's example format
        return ""
    
    def get_role_style_class(self, role: SemanticRole, theme_name: str) -> str:
        """
        Get the CSS class name for a semantic role in a theme.
        
        Args:
            role: The semantic role
            theme_name: Name of the theme
            
        Returns:
            CSS class name (e.g., "logicStyle")
        """
        return f"{role.value}Style"
    
    def available_themes(self) -> List[str]:
        """
        Get list of available theme names.
        
        Returns:
            List of theme names
        """
        return list(self._themes_cache.keys())
    
    def get_default_theme(self) -> Optional[Theme]:
        """
        Get the default theme (dark_professional if available).
        
        Returns:
            Default theme or first available theme
        """
        # Try preferred default
        default = self.get_theme("dark_professional")
        if default:
            return default
        
        # Fallback to first available theme
        if self._themes_cache:
            first_theme_name = next(iter(self._themes_cache))
            return self._themes_cache[first_theme_name]
        
        return None
    
    def validate_theme_completeness(self, theme_name: str) -> Dict[str, bool]:
        """
        Validate that a theme defines all required semantic roles.
        
        Args:
            theme_name: Name of the theme to validate
            
        Returns:
            Dictionary mapping role names to whether they're defined
        """
        theme = self.get_theme(theme_name)
        if not theme:
            return {role.value: False for role in SemanticRole}
        
        return {role.value: role in theme.roles for role in SemanticRole}
    
    def reload_themes(self) -> bool:
        """
        Reload themes from file (useful for development).
        
        Returns:
            True if reload successful, False otherwise
        """
        try:
            self._themes_cache.clear()
            self._load_themes()
            logger.info("Themes reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload themes: {e}")
            return False
    
    def get_theme_info(self, theme_name: str) -> Optional[Dict[str, str]]:
        """
        Get theme metadata for display purposes.
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            Dictionary with theme name, description, and role count
        """
        theme = self.get_theme(theme_name)
        if not theme:
            return None
        
        return {
            'name': theme.name,
            'description': theme.description,
            'roles_defined': len(theme.roles),
            'total_roles': len(SemanticRole),
            'completeness': f"{len(theme.roles)}/{len(SemanticRole)}"
        }