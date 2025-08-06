#!/usr/bin/env python3
"""
Performance Configuration - Centralized performance settings and constants

This module provides production-ready configuration management for performance
optimization features with proper environment variable integration and validation.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class PerformanceConfig:
    """Centralized performance configuration with proper validation"""
    
    # Path resolution caching
    PATH_RESOLUTION_CACHE_SIZE: int = 1000
    PATH_RESOLUTION_CACHE_TTL_SECONDS: int = 300  # 5 minutes
    
    # Discovery pipeline timeouts
    DISCOVERY_PIPELINE_TIMEOUT_SECONDS: float = 5.0
    DISCOVERY_FILE_READ_TIMEOUT_SECONDS: float = 2.0
    
    # Discovery pipeline limits
    MAX_AUTO_EXAMINATIONS: int = 3
    MIN_CONFIDENCE_THRESHOLD: float = 0.6
    MAX_PATH_RESOLUTION_CANDIDATES: int = 3
    
    # Logging configuration for performance
    ENABLE_PERFORMANCE_LOGGING: bool = False
    ENABLE_DEBUG_PATH_RESOLUTION: bool = False
    
    # Cache configuration
    ENABLE_PATH_RESOLUTION_CACHE: bool = True
    ENABLE_FILE_CONTENT_CACHE: bool = True
    FILE_CONTENT_CACHE_SIZE: int = 50
    FILE_CONTENT_CACHE_TTL_SECONDS: int = 180  # 3 minutes
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.PATH_RESOLUTION_CACHE_SIZE < 0:
            raise ValueError("PATH_RESOLUTION_CACHE_SIZE must be non-negative")
        
        if self.DISCOVERY_PIPELINE_TIMEOUT_SECONDS <= 0:
            raise ValueError("DISCOVERY_PIPELINE_TIMEOUT_SECONDS must be positive")
        
        if not 0 <= self.MIN_CONFIDENCE_THRESHOLD <= 1:
            raise ValueError("MIN_CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if self.MAX_AUTO_EXAMINATIONS < 0:
            raise ValueError("MAX_AUTO_EXAMINATIONS must be non-negative")


class PerformanceConfigManager:
    """Production-ready configuration manager with environment variable support"""
    
    _instance: Optional['PerformanceConfigManager'] = None
    _config: Optional[PerformanceConfig] = None
    
    def __new__(cls) -> 'PerformanceConfigManager':
        """Singleton pattern for configuration management"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_config(self) -> PerformanceConfig:
        """Get configuration with environment variable overrides"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def reload_config(self) -> PerformanceConfig:
        """Force reload configuration from environment"""
        self._config = None
        return self.get_config()
    
    def _load_config(self) -> PerformanceConfig:
        """Load configuration from environment variables and config files"""
        # Start with default configuration
        config_dict = {}
        
        # Load from config file if exists
        config_file_path = Path(__file__).parent / "performance_config.json"
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r') as f:
                    file_config = json.load(f)
                config_dict.update(file_config)
            except (json.JSONDecodeError, IOError) as e:
                # Log warning but continue with defaults
                pass
        
        # Environment variable overrides
        env_overrides = {
            'PATH_RESOLUTION_CACHE_SIZE': self._get_env_int('CODEWISE_PATH_CACHE_SIZE'),
            'PATH_RESOLUTION_CACHE_TTL_SECONDS': self._get_env_int('CODEWISE_PATH_CACHE_TTL'),
            'DISCOVERY_PIPELINE_TIMEOUT_SECONDS': self._get_env_float('CODEWISE_DISCOVERY_TIMEOUT'),
            'DISCOVERY_FILE_READ_TIMEOUT_SECONDS': self._get_env_float('CODEWISE_FILE_READ_TIMEOUT'),
            'MAX_AUTO_EXAMINATIONS': self._get_env_int('CODEWISE_MAX_AUTO_EXAMS'),
            'MIN_CONFIDENCE_THRESHOLD': self._get_env_float('CODEWISE_MIN_CONFIDENCE'),
            'MAX_PATH_RESOLUTION_CANDIDATES': self._get_env_int('CODEWISE_MAX_PATH_CANDIDATES'),
            'ENABLE_PERFORMANCE_LOGGING': self._get_env_bool('CODEWISE_ENABLE_PERF_LOGGING'),
            'ENABLE_DEBUG_PATH_RESOLUTION': self._get_env_bool('CODEWISE_DEBUG_PATH_RESOLUTION'),
            'ENABLE_PATH_RESOLUTION_CACHE': self._get_env_bool('CODEWISE_ENABLE_PATH_CACHE'),
            'ENABLE_FILE_CONTENT_CACHE': self._get_env_bool('CODEWISE_ENABLE_FILE_CACHE'),
            'FILE_CONTENT_CACHE_SIZE': self._get_env_int('CODEWISE_FILE_CACHE_SIZE'),
            'FILE_CONTENT_CACHE_TTL_SECONDS': self._get_env_int('CODEWISE_FILE_CACHE_TTL'),
        }
        
        # Apply non-None overrides
        for key, value in env_overrides.items():
            if value is not None:
                config_dict[key] = value
        
        return PerformanceConfig(**config_dict)
    
    def _get_env_int(self, key: str) -> Optional[int]:
        """Get integer environment variable with error handling"""
        value = os.getenv(key)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    def _get_env_float(self, key: str) -> Optional[float]:
        """Get float environment variable with error handling"""
        value = os.getenv(key)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    
    def _get_env_bool(self, key: str) -> Optional[bool]:
        """Get boolean environment variable with error handling"""
        value = os.getenv(key)
        if value is None:
            return None
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')


# Global configuration instance
config_manager = PerformanceConfigManager()