"""
Enhanced File Discovery Engine for CodeWise Indexer

This module provides comprehensive file discovery with expanded file type support,
content-based detection, symlink handling, and detailed statistics.
"""

import os
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a discovered file"""
    path: Path
    relative_path: str
    file_type: str
    size: int
    is_text: bool
    detection_method: str  # 'extension', 'content', 'dotfile'


@dataclass
class FileDiscoveryStats:
    """Statistics from file discovery process"""
    total_files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skipped_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    binary_files_skipped: int = 0
    symlinks_followed: int = 0
    symlinks_skipped: int = 0
    dotfiles_included: int = 0
    content_detected_files: int = 0
    
    def get_coverage_percentage(self) -> float:
        """Calculate file coverage percentage"""
        if self.total_files_scanned == 0:
            return 0.0
        return (self.files_indexed / self.total_files_scanned) * 100


class FileDiscoveryEngine:
    """Enhanced file discovery engine with comprehensive filtering and statistics"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.valid_extensions = {
            # Code files - Modern Languages
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".swift", ".go", ".rs", ".dart",
            ".scala", ".clj", ".cljs", ".rb", ".php", ".lua", ".r", ".jl",
            # JavaScript/Node.js variants
            ".mjs", ".cjs", ".es6", ".es", ".jsm",
            # Code files - Systems Programming  
            ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx", ".c++", ".h++", ".hh", ".hxx",
            ".m", ".mm", ".asm", ".s", ".S",
            # Code files - Functional/Academic
            ".hs", ".lhs", ".ml", ".mli", ".fs", ".fsi", ".fsx", ".elm", ".ex", ".exs",
            ".erl", ".hrl", ".nim", ".cr", ".d", ".zig", ".v",
            # Web Frameworks
            ".vue", ".svelte", ".jsp", ".asp", ".aspx", ".erb", ".haml", ".slim",
            # Mobile/iOS Development
            ".plist", ".storyboard", ".entitlements", ".xcconfig", ".xcworkspace",
            # Documentation
            ".md", ".txt", ".rst", ".adoc", ".tex", ".org",
            # Data/Config files
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".pom",
            # Web files
            ".html", ".htm", ".css", ".scss", ".sass", ".less", ".stylus",
            # Lock/dependency files
            ".lock", ".frozen", ".resolved",
            # Environment/config files
            ".env", ".config", ".properties", ".gitignore", ".gitattributes",
            # Build files
            ".gradle", ".maven", ".sbt", ".dockerfile", ".containerfile", ".cmake", 
            ".meson", ".ninja", ".bazel", ".bzl", ".buck", ".pants", ".nix", ".rake",
            # Other text formats
            ".xml", ".csv", ".sql", ".sh", ".bat", ".ps1", ".fish", ".zsh", ".bash",
            ".makefile", ".mk", ".am", ".in", ".m4", ".spec", ".desktop", ".patch"
        }
        
        # Files to include even without extensions (by name patterns)
        self.special_files = {
            "dockerfile", "makefile", "rakefile", "gemfile", "procfile",
            "readme", "license", "changelog", "contributing", "authors",
            "requirements", "pipfile", "poetry", "package", "composer",
            # Configuration files
            "eslint", "prettier", "babel", "webpack", "rollup", "vite", 
            "postcss", "tailwind", "tsconfig", "jsconfig", "next",
            "jest", "vitest", "cypress", "playwright",
            # Package/dependency files
            "package-lock", "yarn.lock", "pnpm-lock", "bun.lockb"
        }
        
        # Directories to skip entirely
        self.skip_directories = {
            ".git", ".svn", ".hg", ".bzr",  # VCS
            "node_modules", "__pycache__", ".pytest_cache",  # Dependencies/cache
            ".venv", "venv", "env", ".env",  # Virtual environments
            "build", "dist", "target", "out",  # Build outputs
            ".idea", ".vscode", ".vs",  # IDE files
            "coverage", ".coverage", ".nyc_output",  # Coverage
            "logs", "log", "tmp", "temp"  # Temporary files
        }
        
        self.stats = FileDiscoveryStats()
        self._visited_paths: Set[Path] = set()  # For symlink cycle detection
        
    def discover_files(self) -> List[FileInfo]:
        """Discover all indexable files with comprehensive filtering"""
        logger.info(f"Starting file discovery in: {self.workspace_path}")
        
        if not self.workspace_path.exists():
            logger.error(f"Workspace path does not exist: {self.workspace_path}")
            return []
            
        discovered_files = []
        self._visited_paths.clear()
        
        try:
            discovered_files = self._scan_directory(self.workspace_path)
        except Exception as e:
            logger.error(f"Error during file discovery: {e}")
            
        # Log comprehensive statistics
        self._log_discovery_stats()
        
        logger.info(f"File discovery complete: {len(discovered_files)} files indexed")
        return discovered_files
    
    def _scan_directory(self, directory: Path) -> List[FileInfo]:
        """Recursively scan directory for indexable files"""
        discovered_files = []
        
        try:
            for item in directory.iterdir():
                self.stats.total_files_scanned += 1
                
                # Skip hidden directories but allow hidden files
                if item.is_dir() and item.name.startswith('.'):
                    if item.name in self.skip_directories:
                        self.stats.skipped_reasons["hidden_directory"] += 1
                        continue
                
                # Skip known directories to avoid
                if item.is_dir() and item.name in self.skip_directories:
                    self.stats.skipped_reasons["skip_directory"] += 1
                    continue
                
                if item.is_dir():
                    # Recursively scan subdirectories
                    discovered_files.extend(self._scan_directory(item))
                elif item.is_file():
                    file_info = self._process_file(item)
                    if file_info:
                        discovered_files.append(file_info)
                        self.stats.files_indexed += 1
                    else:
                        self.stats.files_skipped += 1
                        
        except PermissionError:
            logger.warning(f"Permission denied accessing: {directory}")
            self.stats.skipped_reasons["permission_denied"] += 1
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            
        return discovered_files
    
    def _process_file(self, file_path: Path) -> Optional[FileInfo]:
        """Process individual file and determine if it should be indexed"""
        try:
            # Handle symlinks
            if file_path.is_symlink():
                return self._handle_symlink(file_path)
            
            # Check if file should be indexed
            if not self.should_index_file(file_path):
                return None
                
            # Get file info
            file_size = file_path.stat().st_size
            relative_path = str(file_path.relative_to(self.workspace_path))
            
            # Determine file type and detection method
            file_type, detection_method = self._determine_file_type(file_path)
            
            # Verify it's a text file
            is_text = self._is_text_file(file_path)
            if not is_text:
                self.stats.binary_files_skipped += 1
                self.stats.skipped_reasons["binary_file"] += 1
                return None
            
            # Update statistics
            self.stats.files_by_type[file_type] += 1
            if detection_method == "content":
                self.stats.content_detected_files += 1
            elif file_path.name.startswith('.'):
                self.stats.dotfiles_included += 1
            
            return FileInfo(
                path=file_path,
                relative_path=relative_path,
                file_type=file_type,
                size=file_size,
                is_text=is_text,
                detection_method=detection_method
            )
            
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            self.stats.skipped_reasons["processing_error"] += 1
            return None
    
    def _handle_symlink(self, symlink_path: Path) -> Optional[FileInfo]:
        """Handle symlink with cycle detection"""
        try:
            # Resolve the symlink
            resolved_path = symlink_path.resolve()
            
            # Check for cycles
            if resolved_path in self._visited_paths:
                self.stats.symlinks_skipped += 1
                self.stats.skipped_reasons["symlink_cycle"] += 1
                return None
            
            # Check if target exists and is a file
            if not resolved_path.exists() or not resolved_path.is_file():
                self.stats.symlinks_skipped += 1
                self.stats.skipped_reasons["broken_symlink"] += 1
                return None
            
            # Add to visited paths to prevent cycles
            self._visited_paths.add(resolved_path)
            
            # Process the resolved file
            self.stats.symlinks_followed += 1
            return self._process_file(resolved_path)
            
        except Exception as e:
            logger.warning(f"Error handling symlink {symlink_path}: {e}")
            self.stats.symlinks_skipped += 1
            self.stats.skipped_reasons["symlink_error"] += 1
            return None
    
    def should_index_file(self, path: Path) -> bool:
        """Determine if file should be indexed based on multiple criteria"""
        # Skip if file is too large (>10MB)
        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                self.stats.skipped_reasons["too_large"] += 1
                return False
        except OSError:
            self.stats.skipped_reasons["stat_error"] += 1
            return False
        
        # Check by extension
        if path.suffix.lower() in self.valid_extensions:
            return True
        
        # Check special files by name (case-insensitive)
        file_name_lower = path.name.lower()
        for special_name in self.special_files:
            if special_name in file_name_lower:
                return True
        
        # Check dotfiles with valid extensions
        if path.name.startswith('.') and path.suffix.lower() in self.valid_extensions:
            return True
        
        # For files without extensions, use content detection
        if not path.suffix:
            return self.detect_text_file(path)
        
        self.stats.skipped_reasons["unknown_extension"] += 1
        return False
    
    def detect_text_file(self, path: Path) -> bool:
        """Detect text files without extensions using content analysis"""
        try:
            # Skip if file is too large for content detection
            if path.stat().st_size > 100 * 1024:  # 100KB limit
                return False
            
            # Read first 1KB to check for binary markers
            with open(path, 'rb') as f:
                chunk = f.read(1024)
                
            # Check for null bytes (binary indicator)
            if b'\x00' in chunk:
                return False
            
            # Try to decode as UTF-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                # Try other common encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        chunk.decode(encoding)
                        return True
                    except UnicodeDecodeError:
                        continue
                return False
                
        except Exception as e:
            logger.debug(f"Content detection failed for {path}: {e}")
            return False
    
    def _determine_file_type(self, path: Path) -> tuple[str, str]:
        """Determine file type and detection method"""
        extension = path.suffix.lower()
        
        if extension in self.valid_extensions:
            return extension[1:] if extension else "unknown", "extension"
        
        # Check special files
        file_name_lower = path.name.lower()
        for special_name in self.special_files:
            if special_name in file_name_lower:
                return special_name, "special_name"
        
        # Content-based detection
        if not extension:
            return "text", "content"
        
        return "unknown", "unknown"
    
    def _is_text_file(self, path: Path) -> bool:
        """Verify that file is actually a text file"""
        try:
            # For known extensions, assume text
            if path.suffix.lower() in self.valid_extensions:
                return True
            
            # For unknown files, use content detection
            return self.detect_text_file(path)
            
        except Exception:
            return False
    
    def _log_discovery_stats(self):
        """Log comprehensive discovery statistics"""
        stats = self.stats
        coverage = stats.get_coverage_percentage()
        
        logger.info("=== File Discovery Statistics ===")
        logger.info(f"Total files scanned: {stats.total_files_scanned}")
        logger.info(f"Files indexed: {stats.files_indexed}")
        logger.info(f"Files skipped: {stats.files_skipped}")
        logger.info(f"Coverage percentage: {coverage:.2f}%")
        
        if stats.files_by_type:
            logger.info("Files by type:")
            for file_type, count in sorted(stats.files_by_type.items()):
                logger.info(f"  {file_type}: {count}")
        
        if stats.skipped_reasons:
            logger.info("Skip reasons:")
            for reason, count in sorted(stats.skipped_reasons.items()):
                logger.info(f"  {reason}: {count}")
        
        logger.info(f"Binary files skipped: {stats.binary_files_skipped}")
        logger.info(f"Symlinks followed: {stats.symlinks_followed}")
        logger.info(f"Symlinks skipped: {stats.symlinks_skipped}")
        logger.info(f"Dotfiles included: {stats.dotfiles_included}")
        logger.info(f"Content-detected files: {stats.content_detected_files}")
        logger.info("=== End Statistics ===")
    
    def get_discovery_stats(self) -> FileDiscoveryStats:
        """Return comprehensive discovery statistics"""
        return self.stats