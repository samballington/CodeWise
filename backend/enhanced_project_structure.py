#!/usr/bin/env python3
"""
Enhanced project structure analysis for CodeWise agents
Provides better context awareness, framework detection, and @ annotations
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from directory_filters import get_find_filter_args, filter_file_list
from project_context import get_current_context, get_context_manager

logger = logging.getLogger(__name__)

@dataclass
class ProjectInfo:
    """Information about a detected project"""
    name: str
    framework: Optional[str] = None
    entry_points: List[str] = None
    key_directories: List[str] = None
    config_files: List[str] = None
    confidence_score: float = 0.0

class FrameworkDetector:
    """Detects project frameworks based on file signatures"""
    
    FRAMEWORK_SIGNATURES = {
        'django': {
            'files': ['manage.py', 'settings.py', 'wsgi.py'],
            'patterns': ['*/settings.py', '*/models.py', '*/views.py'],
            'directories': ['migrations', 'templates', 'static'],
            'confidence_boost': 0.3
        },
        'fastapi': {
            'files': ['main.py', 'requirements.txt'],
            'patterns': ['*uvicorn*', '*fastapi*'],
            'dependencies': ['fastapi', 'uvicorn'],
            'confidence_boost': 0.2
        },
        'flask': {
            'files': ['app.py', 'requirements.txt'],
            'patterns': ['*flask*'],
            'dependencies': ['flask'],
            'confidence_boost': 0.2
        },
        'react': {
            'files': ['package.json', 'src/App.js', 'src/App.jsx', 'src/App.tsx'],
            'patterns': ['src/components/*', 'public/index.html'],
            'dependencies': ['react', 'react-dom'],
            'confidence_boost': 0.3
        },
        'nextjs': {
            'files': ['next.config.js', 'package.json'],
            'patterns': ['pages/*', 'app/*', 'components/*'],
            'dependencies': ['next'],
            'confidence_boost': 0.3
        },
        'vue': {
            'files': ['vue.config.js', 'package.json'],
            'patterns': ['src/components/*', 'src/views/*'],
            'dependencies': ['vue'],
            'confidence_boost': 0.2
        },
        'spring': {
            'files': ['pom.xml', 'build.gradle'],
            'patterns': ['src/main/java/*', 'src/main/resources/*'],
            'directories': ['src/main/java', 'src/test/java'],
            'confidence_boost': 0.3
        },
        'nodejs': {
            'files': ['package.json', 'index.js', 'server.js'],
            'patterns': ['node_modules/*'],
            'dependencies': ['express', 'koa', 'hapi'],
            'confidence_boost': 0.2
        }
    }
    
    @classmethod
    def detect_framework(cls, files: List[str], directories: List[str], 
                        package_contents: Dict[str, str] = None) -> Tuple[Optional[str], float]:
        """
        Detect project framework based on files and directories
        Returns: (framework_name, confidence_score)
        """
        scores = {}
        
        for framework, signature in cls.FRAMEWORK_SIGNATURES.items():
            score = 0.0
            
            # Check for signature files
            if 'files' in signature:
                for sig_file in signature['files']:
                    if any(sig_file in f for f in files):
                        score += 0.2
            
            # Check for signature patterns
            if 'patterns' in signature:
                for pattern in signature['patterns']:
                    clean_pattern = pattern.replace('*', '')
                    if any(clean_pattern in f for f in files):
                        score += 0.1
            
            # Check for signature directories
            if 'directories' in signature:
                for sig_dir in signature['directories']:
                    if any(sig_dir in d for d in directories):
                        score += 0.15
            
            # Check dependencies in package files
            if 'dependencies' in signature and package_contents:
                for dep in signature['dependencies']:
                    if any(dep in content.lower() for content in package_contents.values()):
                        score += 0.25
            
            # Apply confidence boost
            if score > 0:
                score += signature.get('confidence_boost', 0)
            
            scores[framework] = score
        
        # Return framework with highest score
        if scores:
            best_framework = max(scores.keys(), key=lambda k: scores[k])
            if scores[best_framework] > 0.3:  # Minimum confidence threshold
                return best_framework, scores[best_framework]
        
        return None, 0.0

class EnhancedProjectStructure:
    """Enhanced project structure analyzer"""
    
    def __init__(self, mcp_tool_caller):
        self.call_mcp_tool = mcp_tool_caller
        
    async def analyze_project(self, directory: str = ".", max_depth: int = 4, 
                            include_files: bool = True, project_name: str = None) -> str:
        """
        Generate enhanced project structure analysis with context awareness
        """
        # Use project context for scoped analysis
        current_context = get_current_context()
        if current_context and not project_name:
            project_name = current_context.name
            directory = current_context.base_path
        
        # Resolve directory path
        search_dir = self._resolve_directory_path(directory, project_name)
        
        logger.info(f"Analyzing project structure: {search_dir} (depth: {max_depth})")
        
        # Get directory structure with standardized filtering
        directories = await self._get_directories(search_dir, max_depth)
        if not directories:
            return f"Error: Could not access directory structure for {search_dir}"
        
        # Get important files
        important_files = await self._get_important_files(search_dir, max_depth)
        
        # Get source files if requested
        source_files = []
        if include_files:
            source_files = await self._get_source_files(search_dir, max_depth)
        
        # Read package/config files for framework detection
        package_contents = await self._read_package_files(important_files)
        
        # Detect framework
        framework_name, confidence = FrameworkDetector.detect_framework(
            important_files + source_files, directories, package_contents
        )
        
        # Build project info
        project_info = ProjectInfo(
            name=project_name or os.path.basename(search_dir),
            framework=framework_name,
            entry_points=self._identify_entry_points(important_files, source_files),
            key_directories=self._identify_key_directories(directories),
            config_files=self._identify_config_files(important_files),
            confidence_score=confidence
        )
        
        # Generate enhanced tree structure
        tree_structure = self._build_enhanced_tree(
            directories, important_files, source_files, search_dir, project_info
        )
        
        # Create context-aware summary
        summary = self._create_project_summary(project_info, len(directories), 
                                             len(important_files) + len(source_files))
        
        # Add @ annotations for codebase highlighting
        annotated_result = self._add_annotations(summary + "\n\n" + tree_structure, project_name)
        
        # Update project context with discovered information
        if current_context and framework_name:
            current_context.framework_type = framework_name
            current_context.key_files = project_info.entry_points + project_info.config_files
        
        return annotated_result
    
    def _resolve_directory_path(self, directory: str, project_name: str = None) -> str:
        """Resolve directory path within workspace context"""
        if project_name and project_name != "workspace":
            return f"/workspace/{project_name}"
        elif directory == "." or not directory.startswith("/"):
            if directory == ".":
                return "/workspace"
            else:
                return f"/workspace/{directory.lstrip('./')}"
        else:
            return directory
    
    async def _get_directories(self, search_dir: str, max_depth: int) -> List[str]:
        """Get directory structure with standardized filtering"""
        filter_args = get_find_filter_args()
        dirs_cmd = f"""find {search_dir} -maxdepth {max_depth} -type d {filter_args} | sort"""
        
        result = await self.call_mcp_tool("run_command", {"command": dirs_cmd})
        if not result or "Error" in result:
            return []
        
        directories = [d.strip() for d in result.split('\n') if d.strip()]
        return filter_file_list(directories)  # Additional filtering
    
    async def _get_important_files(self, search_dir: str, max_depth: int) -> List[str]:
        """Get important configuration and documentation files"""
        important_files_cmd = f"""find {search_dir} -maxdepth {max_depth} -type f \\( \\
            -name 'README*' -o -name 'readme*' -o -name 'LICENSE*' -o \\
            -name 'package.json' -o -name 'requirements.txt' -o -name 'Pipfile' -o \\
            -name 'poetry.lock' -o -name 'pipfile.lock' -o -name 'yarn.lock' -o \\
            -name 'manage.py' -o -name 'app.py' -o -name 'main.py' -o -name 'index.js' -o \\
            -name 'server.js' -o -name 'index.html' -o -name 'index.tsx' -o \\
            -name 'Dockerfile' -o -name 'docker-compose.yml' -o -name '.env*' -o \\
            -name '*.config.js' -o -name 'tsconfig.json' -o -name 'setup.py' -o \\
            -name 'pom.xml' -o -name 'build.gradle' -o -name 'Makefile' -o \\
            -name 'pyproject.toml' -o -name 'next.config.js' -o -name 'vue.config.js' \\
        \\) | head -40"""
        
        result = await self.call_mcp_tool("run_command", {"command": important_files_cmd})
        if not result or "Error" in result:
            return []
        
        files = [f.strip() for f in result.split('\n') if f.strip()]
        return filter_file_list(files)
    
    async def _get_source_files(self, search_dir: str, max_depth: int) -> List[str]:
        """Get source code files for context"""
        filter_args = get_find_filter_args()
        source_files_cmd = f"""find {search_dir} -maxdepth {max_depth} -type f \\( \\
            -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.jsx' -o -name '*.tsx' -o \\
            -name '*.java' -o -name '*.go' -o -name '*.rs' -o -name '*.php' -o -name '*.rb' \\
        \\) {filter_args} | head -25"""
        
        result = await self.call_mcp_tool("run_command", {"command": source_files_cmd})
        if not result or "Error" in result:
            return []
        
        files = [f.strip() for f in result.split('\n') if f.strip()]
        return filter_file_list(files)
    
    async def _read_package_files(self, files: List[str]) -> Dict[str, str]:
        """Read package/config files for framework detection"""
        package_contents = {}
        
        package_files = ['package.json', 'requirements.txt', 'pom.xml', 'pyproject.toml']
        
        for file_path in files:
            filename = os.path.basename(file_path)
            if filename in package_files:
                try:
                    result = await self.call_mcp_tool("read_file", file_path)
                    if result and "Error" not in result:
                        package_contents[filename] = result[:1000]  # Limit content size
                except Exception as e:
                    logger.warning(f"Could not read package file {file_path}: {e}")
        
        return package_contents
    
    def _identify_entry_points(self, important_files: List[str], source_files: List[str]) -> List[str]:
        """Identify main entry points of the project"""
        entry_patterns = [
            'main.py', 'app.py', 'manage.py', 'server.js', 'index.js', 
            'index.html', 'index.tsx', 'App.js', 'App.tsx'
        ]
        
        entry_points = []
        all_files = important_files + source_files
        
        for pattern in entry_patterns:
            matching_files = [f for f in all_files if pattern in os.path.basename(f)]
            entry_points.extend(matching_files)
        
        return list(set(entry_points))  # Remove duplicates
    
    def _identify_key_directories(self, directories: List[str]) -> List[str]:
        """Identify key project directories"""
        key_patterns = [
            'src', 'lib', 'app', 'components', 'pages', 'views', 'templates',
            'static', 'public', 'assets', 'config', 'tests', 'test', 'spec',
            'docs', 'api', 'routes', 'models', 'controllers', 'services'
        ]
        
        key_dirs = []
        for directory in directories:
            dir_name = os.path.basename(directory)
            if dir_name in key_patterns:
                key_dirs.append(directory)
        
        return key_dirs
    
    def _identify_config_files(self, files: List[str]) -> List[str]:
        """Identify configuration files"""
        config_patterns = [
            'package.json', 'requirements.txt', 'Dockerfile', 'docker-compose.yml',
            '.env', 'tsconfig.json', 'next.config.js', 'vue.config.js',
            'pom.xml', 'build.gradle', 'pyproject.toml'
        ]
        
        config_files = []
        for file_path in files:
            filename = os.path.basename(file_path)
            if any(pattern in filename for pattern in config_patterns):
                config_files.append(file_path)
        
        return config_files
    
    def _build_enhanced_tree(self, directories: List[str], important_files: List[str], 
                           source_files: List[str], base_dir: str, project_info: ProjectInfo) -> str:
        """Build enhanced tree structure with context highlighting"""
        # Combine all files and sort
        all_files = important_files + source_files
        
        # Create tree structure
        tree_lines = []
        tree_lines.append(f"📁 {project_info.name}/")
        
        # Group files by directory
        dir_files = {}
        for file_path in all_files:
            if file_path.startswith(base_dir):
                rel_path = file_path.replace(base_dir, '').lstrip('/')
                dir_name = os.path.dirname(rel_path) or '.'
                filename = os.path.basename(rel_path)
                
                if dir_name not in dir_files:
                    dir_files[dir_name] = []
                dir_files[dir_name].append(filename)
        
        # Sort directories - show root files first, then subdirectories
        sorted_dirs = sorted(dir_files.keys(), key=lambda x: (x != '.', x))
        
        for dir_name in sorted_dirs[:15]:  # Limit number of directories shown
            files = sorted(dir_files[dir_name])
            
            if dir_name == '.':
                # Root files
                for filename in files[:10]:  # Limit files shown
                    icon = self._get_file_icon(filename)
                    priority = "🌟 " if filename in [os.path.basename(f) for f in project_info.entry_points] else ""
                    tree_lines.append(f"  {icon} {priority}{filename}")
            else:
                # Subdirectory
                tree_lines.append(f"  📁 {dir_name}/")
                for filename in files[:5]:  # Limit files per directory
                    icon = self._get_file_icon(filename)
                    tree_lines.append(f"    {icon} {filename}")
                
                if len(files) > 5:
                    tree_lines.append(f"    ... ({len(files) - 5} more files)")
        
        if len(sorted_dirs) > 15:
            tree_lines.append(f"  ... ({len(sorted_dirs) - 15} more directories)")
        
        return "\n".join(tree_lines)
    
    def _get_file_icon(self, filename: str) -> str:
        """Get appropriate icon for file type"""
        lower_name = filename.lower()
        
        if lower_name.startswith('readme'):
            return "📖"
        elif lower_name in ['package.json', 'requirements.txt', 'pom.xml']:
            return "📦"
        elif lower_name.startswith('dockerfile'):
            return "🐳"
        elif lower_name.endswith('.py'):
            return "🐍"
        elif lower_name.endswith(('.js', '.jsx', '.ts', '.tsx')):
            return "⚡"
        elif lower_name.endswith('.java'):
            return "☕"
        elif lower_name.endswith(('.html', '.css')):
            return "🌐"
        elif lower_name.endswith('.md'):
            return "📝"
        elif lower_name.endswith(('.json', '.yml', '.yaml', '.toml')):
            return "⚙️"
        else:
            return "📄"
    
    def _create_project_summary(self, project_info: ProjectInfo, dir_count: int, file_count: int) -> str:
        """Create context-aware project summary"""
        summary_lines = []
        
        # Project header
        framework_info = f" ({project_info.framework})" if project_info.framework else ""
        summary_lines.append(f"🏗️  Project: {project_info.name}{framework_info}")
        
        # Framework confidence
        if project_info.framework and project_info.confidence_score > 0:
            confidence_pct = int(project_info.confidence_score * 100)
            summary_lines.append(f"🎯 Framework Detection: {confidence_pct}% confidence")
        
        # Structure overview
        summary_lines.append(f"📊 Structure: {dir_count} directories, {file_count} files")
        
        # Entry points
        if project_info.entry_points:
            entry_names = [os.path.basename(ep) for ep in project_info.entry_points[:3]]
            summary_lines.append(f"🚀 Entry Points: {', '.join(entry_names)}")
        
        # Key directories
        if project_info.key_directories:
            key_names = [os.path.basename(kd) for kd in project_info.key_directories[:5]]
            summary_lines.append(f"📁 Key Directories: {', '.join(key_names)}")
        
        return "\n".join(summary_lines)
    
    def _add_annotations(self, content: str, project_name: str = None) -> str:
        """Add @ annotations for codebase highlighting"""
        if not project_name or project_name == "workspace":
            return content
        
        # Add project annotation at the beginning
        annotated_content = f"@{project_name}: {content}"
        
        return annotated_content