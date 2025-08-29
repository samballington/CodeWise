"""
Universal Dependency Indexer - Language-Agnostic Import and Call Analysis

This module implements REQ-UNIVERSAL-DEPS: Universal Dependency Detection System
Supports multiple programming languages with unified relationship extraction.

REQ-UNIVERSAL-DEPS-1: Language-Agnostic Import Analysis
REQ-UNIVERSAL-DEPS-2: Cross-Language Call Graph Analysis  
REQ-UNIVERSAL-DEPS-3: Configuration-Based Dependency Analysis
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class UniversalDependencyExtractor:
    """
    Language-agnostic dependency extraction engine.
    
    Implements sophisticated import/require/include analysis that works
    across programming languages and frameworks.
    """
    
    # REQ-UNIVERSAL-DEPS-1: Language-Agnostic Import Patterns
    LANGUAGE_PATTERNS = {
        'python': [
            r'from\s+([\w.]+)\s+import\s+([\w,\s*]+)',     # from module import Class, func
            r'import\s+([\w.]+)(?:\s+as\s+\w+)?',          # import module [as alias]
        ],
        'javascript': [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',   # import ... from 'module'
            r'const\s+[\w{},\s]+\s*=\s*require\([\'"]([^\'"]+)[\'"]\)',  # const x = require('module')
            r'import\([\'"]([^\'"]+)[\'"]\)',              # dynamic import('module')
            r'export\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',   # export ... from 'module'
        ],
        'typescript': [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',   # import ... from 'module'
            r'import\s+[\'"]([^\'"]+)[\'"]',               # import 'module'
            r'export\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',   # export ... from 'module'
        ],
        'java': [
            r'import\s+([\w.]+);',                         # import package.Class;
            r'import\s+static\s+([\w.]+);',                # import static package.method;
        ],
        'go': [
            r'import\s+"([^"]+)"',                         # import "package"
            r'import\s+(\w+)\s+"([^"]+)"',                 # import alias "package"
            r'import\s+\(\s*([^)]+)\s*\)',                 # import ( ... ) block
        ],
        'csharp': [
            r'using\s+([\w.]+);',                          # using Namespace;
            r'using\s+static\s+([\w.]+);',                 # using static Class;
            r'using\s+(\w+)\s*=\s*([\w.]+);',             # using Alias = Namespace;
        ],
        'swift': [
            r'import\s+(\w+)',                             # import Module
            r'import\s+([\w.]+)',                          # import Module.SubModule
            r'import\s+(class|struct|enum|protocol)\s+(\w+\.\w+)',  # import class Module.Class
        ],
        'rust': [
            r'use\s+([\w:]+);',                            # use module::item;
            r'extern\s+crate\s+(\w+);',                    # extern crate name;
            r'mod\s+(\w+);',                               # mod module_name;
        ],
        'php': [
            r'use\s+([\\\w]+);',                           # use Namespace\Class;
            r'require_once\s+[\'"]([^\'"]+)[\'"]',         # require_once 'file'
            r'include_once\s+[\'"]([^\'"]+)[\'"]',         # include_once 'file'
        ],
        'ruby': [
            r'require\s+[\'"]([^\'"]+)[\'"]',              # require 'module'
            r'require_relative\s+[\'"]([^\'"]+)[\'"]',     # require_relative 'module'
            r'load\s+[\'"]([^\'"]+)[\'"]',                 # load 'file'
        ],
        'kotlin': [
            r'import\s+([\w.]+)',                          # import package.Class
            r'import\s+([\w.]+)\.\*',                      # import package.*
        ]
    }
    
    # REQ-UNIVERSAL-DEPS-2: Cross-Language Call Patterns
    CALL_PATTERNS = {
        'object_method': r'(\w+)\.(\w+)\s*\(',           # object.method() - universal
        'function_call': r'(\w+)\s*\(',                  # function() - universal  
        'static_call': r'(\w+)::(\w+)\s*\(',            # Class::method() - multiple languages
        'async_call': r'await\s+(\w+)',                 # await function - async languages
        'chained_call': r'(\w+(?:\.\w+)+)\s*\(',        # obj.method1().method2()
    }
    
    # Language detection by file extension
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.cs': 'csharp',
        '.swift': 'swift',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.kt': 'kotlin',
        '.kts': 'kotlin'
    }
    
    # REQ-UNIVERSAL-DEPS-3: Configuration File Patterns
    CONFIG_EXTRACTORS = {
        'package.json': {
            'patterns': ['dependencies', 'devDependencies', 'peerDependencies'],
            'language': 'javascript',
            'parser': 'json'
        },
        'requirements.txt': {
            'patterns': [r'(\w+)'],
            'language': 'python',
            'parser': 'text'
        },
        'go.mod': {
            'patterns': [r'require\s+([\w./]+)\s+v([\d.]+)'],
            'language': 'go',
            'parser': 'text'
        },
        'Cargo.toml': {
            'patterns': ['dependencies', 'dev-dependencies'],
            'language': 'rust',
            'parser': 'toml'
        },
        'pom.xml': {
            'patterns': [r'<groupId>(.*?)</groupId>.*?<artifactId>(.*?)</artifactId>'],
            'language': 'java',
            'parser': 'xml'
        },
        'packages.config': {
            'patterns': [r'<package\s+id="([^"]+)"'],
            'language': 'csharp',
            'parser': 'xml'
        },
        'composer.json': {
            'patterns': ['require', 'require-dev'],
            'language': 'php',
            'parser': 'json'
        },
        'Gemfile': {
            'patterns': [r'gem\s+[\'"]([^\'"]+)[\'"]'],
            'language': 'ruby',
            'parser': 'text'
        },
        'Package.swift': {
            'patterns': [r'\.package\(url:\s*"([^"]+)"'],
            'language': 'swift',
            'parser': 'text'
        }
    }
    
    def __init__(self):
        """Initialize the universal dependency extractor."""
        self.supported_languages = set(self.LANGUAGE_PATTERNS.keys())
        logger.info(f"UniversalDependencyExtractor initialized with {len(self.supported_languages)} languages")
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Language identifier or None if unsupported
        """
        suffix = Path(file_path).suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(suffix)
    
    def extract_imports_for_language(self, source_code: str, language: str, file_path: str) -> List[Dict]:
        """
        Extract import dependencies using language-specific patterns.
        
        Args:
            source_code: Source code content
            language: Programming language identifier
            file_path: Path to the source file
            
        Returns:
            List of import dependency dictionaries
        """
        if language not in self.LANGUAGE_PATTERNS:
            logger.warning(f"Unsupported language: {language}")
            return []
        
        dependencies = []
        patterns = self.LANGUAGE_PATTERNS[language]
        
        for pattern in patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Extract import path from regex groups
                import_path = self._extract_import_path(match, pattern, language)
                if import_path:
                    dependencies.append({
                        'type': 'import',
                        'source_file': file_path,
                        'target_module': import_path,
                        'language': language,
                        'line_number': source_code[:match.start()].count('\n') + 1,
                        'confidence': self._calculate_import_confidence(import_path, language),
                        'dependency_type': self._classify_dependency_type(import_path, file_path)
                    })
        
        logger.debug(f"Extracted {len(dependencies)} import dependencies from {file_path}")
        return dependencies
    
    def extract_call_relationships(self, source_code: str, language: str, 
                                 file_path: str, available_components: List[str]) -> List[Dict]:
        """
        Extract function/method call relationships using universal patterns.
        
        Args:
            source_code: Source code content
            language: Programming language identifier  
            file_path: Path to the source file
            available_components: List of known components/modules
            
        Returns:
            List of call relationship dictionaries
        """
        relationships = []
        
        for call_type, pattern in self.CALL_PATTERNS.items():
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            for match in matches:
                call_target = self._extract_call_target(match, call_type)
                if call_target and self._is_cross_module_call(call_target, available_components):
                    relationships.append({
                        'type': 'call',
                        'call_type': call_type,
                        'source_file': file_path,
                        'target_function': call_target,
                        'language': language,
                        'line_number': source_code[:match.start()].count('\n') + 1,
                        'confidence': self._calculate_call_confidence(call_target, call_type, language)
                    })
        
        logger.debug(f"Extracted {len(relationships)} call relationships from {file_path}")
        return relationships
    
    def extract_configuration_dependencies(self, config_file_path: str) -> List[Dict]:
        """
        Extract dependencies from configuration files.
        
        Args:
            config_file_path: Path to configuration file
            
        Returns:
            List of configuration dependency dictionaries
        """
        config_name = Path(config_file_path).name
        if config_name not in self.CONFIG_EXTRACTORS:
            logger.debug(f"No extractor for config file: {config_name}")
            return []
        
        extractor_config = self.CONFIG_EXTRACTORS[config_name]
        dependencies = []
        
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if extractor_config['parser'] == 'json':
                dependencies = self._extract_json_dependencies(content, extractor_config)
            elif extractor_config['parser'] == 'text':
                dependencies = self._extract_text_dependencies(content, extractor_config)
            elif extractor_config['parser'] == 'xml':
                dependencies = self._extract_xml_dependencies(content, extractor_config)
            elif extractor_config['parser'] == 'toml':
                dependencies = self._extract_toml_dependencies(content, extractor_config)
            
            # Add metadata to each dependency
            for dep in dependencies:
                dep.update({
                    'type': 'configuration',
                    'source_file': config_file_path,
                    'language': extractor_config['language'],
                    'confidence': 0.95  # High confidence for explicit config dependencies
                })
                
        except Exception as e:
            logger.error(f"Failed to extract dependencies from {config_file_path}: {e}")
        
        logger.debug(f"Extracted {len(dependencies)} config dependencies from {config_file_path}")
        return dependencies
    
    def resolve_import_to_component(self, import_path: str, project_structure: Dict, 
                                   source_language: str) -> Optional[str]:
        """
        Resolve import path to actual component/file in project.
        
        Args:
            import_path: Import/require path from source code
            project_structure: Dictionary of project files and structure
            source_language: Language of the source file
            
        Returns:
            Resolved component name or None if not found
        """
        # Language-specific resolution rules
        resolution_strategies = {
            'python': self._resolve_python_import,
            'javascript': self._resolve_js_import,
            'typescript': self._resolve_js_import,  # Similar to JS
            'java': self._resolve_java_import,
            'go': self._resolve_go_import,
            'swift': self._resolve_swift_import,
            'csharp': self._resolve_csharp_import
        }
        
        resolver = resolution_strategies.get(source_language)
        if resolver:
            return resolver(import_path, project_structure)
        
        # Fallback: simple string matching
        return self._resolve_generic_import(import_path, project_structure)
    
    def classify_architectural_role(self, file_path: str, imports: List[str], 
                                  exports: List[str], language: str) -> str:
        """
        Classify component's architectural role based on universal patterns.
        
        Args:
            file_path: Path to the component file
            imports: List of imported modules/components
            exports: List of exported functions/classes
            language: Programming language
            
        Returns:
            Architectural role classification
        """
        path_lower = file_path.lower()
        
        # Directory-based classification (universal)
        if any(pattern in path_lower for pattern in ['controller', 'handler', 'route', 'view']):
            return 'controller'
        elif any(pattern in path_lower for pattern in ['service', 'business', 'logic', 'manager']):
            return 'service'
        elif any(pattern in path_lower for pattern in ['repository', 'model', 'data', 'entity']):
            return 'model'
        elif any(pattern in path_lower for pattern in ['config', 'setting', 'env']):
            return 'configuration'
        elif any(pattern in path_lower for pattern in ['util', 'helper', 'tool', 'common']):
            return 'utility'
        
        # Import pattern-based classification
        if self._imports_many_services(imports):
            return 'controller'
        elif self._imports_data_layer(imports):
            return 'service'
        elif self._imports_external_only(imports):
            return 'model'
        
        return 'component'  # Default classification
    
    # Private helper methods
    
    def _extract_import_path(self, match: re.Match, pattern: str, language: str) -> Optional[str]:
        """Extract import path from regex match based on language patterns."""
        groups = match.groups()
        if not groups:
            return None
        
        # Language-specific extraction logic
        if language == 'go' and len(groups) > 1:
            # Handle Go import aliases: import alias "package"
            return groups[1] if groups[1] else groups[0]
        elif language == 'swift' and len(groups) > 1:
            # Handle Swift qualified imports: import class Module.Class
            return f"{groups[1]}" if groups[1] else groups[0]
        else:
            # Most languages: return first capture group
            return groups[0]
    
    def _extract_call_target(self, match: re.Match, call_type: str) -> Optional[str]:
        """Extract call target from regex match."""
        groups = match.groups()
        if not groups:
            return None
        
        if call_type == 'object_method' and len(groups) >= 2:
            return f"{groups[0]}.{groups[1]}"
        elif call_type == 'static_call' and len(groups) >= 2:
            return f"{groups[0]}::{groups[1]}"
        else:
            return groups[0]
    
    def _calculate_import_confidence(self, import_path: str, language: str) -> float:
        """Calculate confidence score for import relationship."""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for explicit imports
        if import_path and not import_path.startswith('.'):
            confidence += 0.1
        
        # Language-specific adjustments
        if language in ['java', 'csharp'] and '.' in import_path:
            confidence += 0.1  # Qualified imports more reliable
        
        return min(confidence, 1.0)
    
    def _calculate_call_confidence(self, call_target: str, call_type: str, language: str) -> float:
        """Calculate confidence score for call relationship."""
        confidence = 0.6  # Base confidence (lower than imports)
        
        # Higher confidence for qualified calls
        if call_type in ['object_method', 'static_call']:
            confidence += 0.2
        
        # Language-specific adjustments
        if language in ['java', 'csharp', 'swift'] and '.' in call_target:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _classify_dependency_type(self, import_path: str, source_file: str) -> str:
        """Classify dependency as internal, external, or cross-layer."""
        if import_path.startswith('.') or import_path.startswith('/'):
            return 'internal_relative'
        elif any(external in import_path for external in ['http', 'api', 'client']):
            return 'external'
        elif any(layer in import_path for layer in ['service', 'repository', 'model']):
            return 'cross_layer'
        else:
            return 'internal'
    
    def _is_cross_module_call(self, call_target: str, available_components: List[str]) -> bool:
        """Check if call target represents cross-module dependency."""
        # Simple heuristic: if call target matches known component names
        return any(component in call_target for component in available_components)
    
    def _extract_json_dependencies(self, content: str, config: Dict) -> List[Dict]:
        """Extract dependencies from JSON configuration files."""
        try:
            data = json.loads(content)
            dependencies = []
            
            for pattern in config['patterns']:
                if pattern in data and isinstance(data[pattern], dict):
                    for dep_name, version in data[pattern].items():
                        dependencies.append({
                            'target_module': dep_name,
                            'version': version,
                            'dependency_type': 'external'
                        })
            
            return dependencies
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON configuration")
            return []
    
    def _extract_text_dependencies(self, content: str, config: Dict) -> List[Dict]:
        """Extract dependencies from text-based configuration files."""
        dependencies = []
        
        for pattern in config['patterns']:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if match.groups():
                    dependencies.append({
                        'target_module': match.group(1),
                        'dependency_type': 'external'
                    })
        
        return dependencies
    
    def _extract_xml_dependencies(self, content: str, config: Dict) -> List[Dict]:
        """Extract dependencies from XML configuration files."""
        # Simple regex-based XML parsing for dependency extraction
        dependencies = []
        
        for pattern in config['patterns']:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    dependencies.append({
                        'target_module': f"{groups[0]}.{groups[1]}",
                        'dependency_type': 'external'
                    })
        
        return dependencies
    
    def _extract_toml_dependencies(self, content: str, config: Dict) -> List[Dict]:
        """Extract dependencies from TOML configuration files."""
        # Simple regex-based TOML parsing for dependency extraction
        dependencies = []
        
        in_dependencies_section = False
        for line in content.split('\n'):
            if line.strip().startswith('[dependencies]'):
                in_dependencies_section = True
                continue
            elif line.strip().startswith('[') and in_dependencies_section:
                in_dependencies_section = False
                continue
            
            if in_dependencies_section:
                match = re.match(r'(\w+)\s*=', line.strip())
                if match:
                    dependencies.append({
                        'target_module': match.group(1),
                        'dependency_type': 'external'
                    })
        
        return dependencies
    
    # Language-specific import resolvers
    
    def _resolve_python_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Resolve Python import to actual file/module."""
        # Convert module path to file path
        file_path = import_path.replace('.', '/') + '.py'
        if file_path in project_structure:
            return import_path
        return None
    
    def _resolve_js_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Resolve JavaScript/TypeScript import to actual file."""
        # Handle relative imports
        if import_path.startswith('./') or import_path.startswith('../'):
            # Relative path resolution would need source file context
            return import_path
        
        # Check for file extensions
        for ext in ['.js', '.ts', '.jsx', '.tsx']:
            file_path = import_path + ext
            if file_path in project_structure:
                return import_path
        
        return None
    
    def _resolve_java_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Resolve Java import to actual class file."""
        # Convert package.Class to file path
        file_path = import_path.replace('.', '/') + '.java'
        if file_path in project_structure:
            return import_path.split('.')[-1]  # Return class name
        return None
    
    def _resolve_go_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Resolve Go import to actual package."""
        # Go imports are typically package paths
        return import_path.split('/')[-1] if import_path else None
    
    def _resolve_swift_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Resolve Swift import to actual module."""
        # Swift imports are typically module names
        return import_path
    
    def _resolve_csharp_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Resolve C# using statement to actual namespace."""
        # C# using statements are namespace paths
        return import_path.split('.')[-1] if import_path else None
    
    def _resolve_generic_import(self, import_path: str, project_structure: Dict) -> Optional[str]:
        """Generic import resolution for unsupported languages."""
        # Simple string matching fallback
        for file_path in project_structure:
            if import_path in file_path:
                return import_path
        return None
    
    # Architectural classification helpers
    
    def _imports_many_services(self, imports: List[str]) -> bool:
        """Check if component imports many service-like modules."""
        service_imports = sum(1 for imp in imports if any(
            keyword in imp.lower() for keyword in ['service', 'manager', 'provider']
        ))
        return service_imports >= 2
    
    def _imports_data_layer(self, imports: List[str]) -> bool:
        """Check if component imports data layer modules."""
        return any(
            keyword in imp.lower() for imp in imports 
            for keyword in ['repository', 'model', 'entity', 'dao']
        )
    
    def _imports_external_only(self, imports: List[str]) -> bool:
        """Check if component only imports external dependencies."""
        if not imports:
            return False
        
        internal_imports = sum(1 for imp in imports if imp.startswith('.') or '/' in imp)
        return internal_imports == 0