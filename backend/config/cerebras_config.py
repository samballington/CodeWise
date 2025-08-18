"""
Cerebras SDK Configuration for Native Integration

Centralized configuration for Cerebras SDK with Docker compatibility
and secure API key management for production environments.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class CerebrasConfig:
    """Centralized configuration for Cerebras SDK with Docker compatibility"""
    
    def __init__(self):
        """Initialize configuration with environment variables and secure defaults"""
        self.api_key = self._get_api_key()
        self.model = os.getenv("CEREBRAS_MODEL", "llama3.1-70b")
        self.reasoning_effort = os.getenv("CEREBRAS_REASONING_EFFORT", "medium")
        self.max_tokens = int(os.getenv("CEREBRAS_MAX_TOKENS", "4096"))
        self.timeout = float(os.getenv("CEREBRAS_TIMEOUT", "30.0"))
        
        # Validate configuration on initialization
        self._validate_config()
        
    def _get_api_key(self) -> Optional[str]:
        """Secure API key retrieval with fallback methods"""
        # Priority order: environment variable, .env file, config file
        api_key = os.getenv("CEREBRAS_API_KEY")
        
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("CEREBRAS_API_KEY")
            except ImportError:
                logger.warning("python-dotenv not available, using direct env vars only")
        
        if not api_key:
            logger.error("CEREBRAS_API_KEY not found in environment")
            return None
            
        # Mask API key in logs for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info(f"✅ Cerebras API key loaded: {masked_key}")
        return api_key
    
    def _validate_config(self) -> None:
        """Validate configuration is complete and correct"""
        if not self.api_key:
            logger.error("❌ Cerebras API key missing - SDK will not function")
            raise ValueError("CEREBRAS_API_KEY environment variable is required")
            
        if not self.model:
            logger.error("❌ Cerebras model not specified")
            raise ValueError("CEREBRAS_MODEL must be specified")
        
        if self.reasoning_effort not in ["low", "medium", "high"]:
            logger.warning(f"⚠️ Invalid reasoning effort '{self.reasoning_effort}', using 'medium'")
            self.reasoning_effort = "medium"
            
        if self.max_tokens < 100 or self.max_tokens > 32000:
            logger.warning(f"⚠️ Invalid max_tokens '{self.max_tokens}', using 4096")
            self.max_tokens = 4096
            
        logger.info(f"✅ Cerebras config validated: {self.model} with {self.reasoning_effort} reasoning")
    
    def validate_config(self) -> bool:
        """Public method to validate configuration - returns True if valid"""
        try:
            self._validate_config()
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def get_client_config(self) -> dict:
        """Get configuration dictionary for Cerebras client initialization"""
        return {
            "api_key": self.api_key,
            "timeout": self.timeout
        }
    
    def get_completion_config(self) -> dict:
        """Get configuration dictionary for completion requests"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens
        }

# Global configuration instance
cerebras_config = CerebrasConfig()

# Export for easy access
__all__ = ["cerebras_config", "CerebrasConfig"]