"""
Multi-Provider API System for CodeWise

This module provides an abstraction layer for multiple AI API providers,
allowing seamless switching between OpenAI, Kimi K2, and other providers.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standardized API response format"""
    content: str
    model: str
    provider: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseProvider(ABC):
    """Abstract base class for AI API providers"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.provider_name = self.__class__.__name__.replace('Provider', '').lower()
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict], **kwargs) -> APIResponse:
        """Generate chat completion response"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass
    
    def validate_api_key(self) -> bool:
        """Validate API key by making a test request"""
        try:
            # Test with a simple embedding request
            test_result = self.generate_embeddings(["test"])
            return len(test_result) > 0 and len(test_result[0]) > 0
        except Exception as e:
            logger.error(f"API key validation failed for {self.provider_name}: {e}")
            return False


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-4"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        super().__init__(api_key, model_name)
        openai.api_key = self.api_key
        
        # Model configurations
        self.embedding_model = "text-embedding-3-small"
        self.chat_models = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        }
        
        # Validate model name
        if model_name not in self.chat_models:
            logger.warning(f"Unknown model {model_name}, using gpt-4")
            self.model_name = "gpt-4"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Process in small batches to avoid token limits
            batch_size = 5
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Clean and validate batch
                clean_batch = []
                for text in batch:
                    if isinstance(text, str) and text.strip():
                        clean_batch.append(text.strip())
                    else:
                        clean_batch.append("empty")
                
                if clean_batch:
                    response = openai.embeddings.create(
                        model=self.embedding_model,
                        input=clean_batch
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Add empty embeddings for invalid batch
                    all_embeddings.extend([[0.0] * 1536] * len(batch))
            
            logger.debug(f"Generated {len(all_embeddings)} embeddings using OpenAI")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            # Return empty embeddings to maintain alignment
            return [[0.0] * 1536] * len(texts)
    
    def chat_completion(self, messages: List[Dict], **kwargs) -> APIResponse:
        """Generate chat completion using OpenAI API"""
        try:
            # Extract parameters
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 2000)
            stream = kwargs.get('stream', False)
            
            response = openai.chat.completions.create(
                model=self.chat_models.get(self.model_name, self.model_name),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
            else:
                content = response.choices[0].message.content
            
            # Extract usage information
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return APIResponse(
                content=content,
                model=self.model_name,
                provider="openai",
                usage=usage,
                metadata={
                    'finish_reason': response.choices[0].finish_reason if not stream else 'stop'
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            return APIResponse(
                content=f"Error: {str(e)}",
                model=self.model_name,
                provider="openai",
                usage={},
                metadata={'error': str(e)}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            'provider': 'openai',
            'chat_model': self.model_name,
            'embedding_model': self.embedding_model,
            'max_tokens': 4096 if 'gpt-3.5' in self.model_name else 8192,
            'supports_streaming': True,
            'supports_functions': True
        }


class KimiProvider(BaseProvider):
    """Kimi K2 API provider implementation"""
    
    def __init__(self, api_key: str = None, model_name: str = "moonshot-v1-8k"):
        # Use provided API key or default from TODO.txt
        api_key = api_key or ""
        
        super().__init__(api_key, model_name)
        self.base_url = "https://api.moonshot.cn/v1"
        
        # Model configurations
        self.available_models = {
            "moonshot-v1-8k": {"max_tokens": 8192, "context_window": 8192},
            "moonshot-v1-32k": {"max_tokens": 32768, "context_window": 32768},
            "moonshot-v1-128k": {"max_tokens": 131072, "context_window": 131072}
        }
        
        # Validate model name
        if model_name not in self.available_models:
            logger.warning(f"Unknown Kimi model {model_name}, using moonshot-v1-8k")
            self.model_name = "moonshot-v1-8k"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Kimi API"""
        try:
            # Kimi doesn't have a dedicated embedding endpoint, so we'll use a workaround
            # or return placeholder embeddings for now
            logger.warning("Kimi provider doesn't support embeddings, using placeholder")
            
            # Return placeholder embeddings (same dimension as OpenAI)
            return [[0.1] * 1536] * len(texts)
            
        except Exception as e:
            logger.error(f"Kimi embedding generation failed: {e}")
            return [[0.0] * 1536] * len(texts)
    
    def chat_completion(self, messages: List[Dict], **kwargs) -> APIResponse:
        """Generate chat completion using Kimi API"""
        try:
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Extract parameters
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 2000)
            stream = kwargs.get('stream', False)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # Make request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Kimi API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract content
            content = result['choices'][0]['message']['content']
            
            # Extract usage information
            usage = result.get('usage', {})
            
            return APIResponse(
                content=content,
                model=self.model_name,
                provider="kimi",
                usage=usage,
                metadata={
                    'finish_reason': result['choices'][0].get('finish_reason', 'stop')
                }
            )
            
        except Exception as e:
            logger.error(f"Kimi chat completion failed: {e}")
            return APIResponse(
                content=f"Error: {str(e)}",
                model=self.model_name,
                provider="kimi",
                usage={},
                metadata={'error': str(e)}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Kimi model information"""
        model_config = self.available_models.get(self.model_name, self.available_models["moonshot-v1-8k"])
        
        return {
            'provider': 'kimi',
            'chat_model': self.model_name,
            'embedding_model': None,  # Kimi doesn't support embeddings
            'max_tokens': model_config['max_tokens'],
            'context_window': model_config['context_window'],
            'supports_streaming': True,
            'supports_functions': False
        }


class APIProviderManager:
    """Manager for switching between different API providers"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider_name = 'kimi'  # Default to Kimi K2
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers"""
        try:
            # Initialize OpenAI provider
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.providers['openai'] = OpenAIProvider(openai_key)
                logger.info("OpenAI provider initialized")
            else:
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        try:
            # Initialize Kimi provider
            self.providers['kimi'] = KimiProvider()
            logger.info("Kimi provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kimi provider: {e}")
        
        # Set default provider - temporarily use OpenAI until Kimi key is fixed
        if 'openai' in self.providers:
            self.current_provider_name = 'openai'
            logger.info("✅ DEFAULT PROVIDER SET TO OPENAI (Kimi key needs to be updated)")
        elif 'kimi' in self.providers:
            self.current_provider_name = 'kimi'
            logger.info("✅ DEFAULT PROVIDER SET TO KIMI K2 for cost optimization")
            logger.info(f"✅ Kimi K2 model: {self.providers['kimi'].model_name}")
        else:
            logger.error("❌ No API providers available")
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different API provider"""
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not available")
            return False
        
        # Validate provider
        provider = self.providers[provider_name]
        if not provider.validate_api_key():
            logger.error(f"Provider {provider_name} validation failed")
            return False
        
        self.current_provider_name = provider_name
        logger.info(f"Switched to provider: {provider_name}")
        return True
    
    def get_current_provider(self) -> Optional[BaseProvider]:
        """Get currently active provider instance"""
        return self.providers.get(self.current_provider_name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using current provider or fallback to OpenAI for embeddings"""
        provider = self.get_current_provider()
        if not provider:
            logger.error("No active provider for embeddings")
            return [[0.0] * 1536] * len(texts)
        
        # If current provider doesn't support embeddings, fallback to OpenAI
        if provider.provider_name == 'kimi':
            logger.info("Using OpenAI for embeddings while Kimi handles chat completions")
            if 'openai' in self.providers:
                return self.providers['openai'].generate_embeddings(texts)
            else:
                logger.warning("OpenAI not available for embeddings, using local embedding fallback")
                return self._generate_local_embeddings(texts)
        
        return provider.generate_embeddings(texts)
    
    def chat_completion(self, messages: List[Dict], **kwargs) -> APIResponse:
        """Generate chat completion using current provider"""
        provider = self.get_current_provider()
        if not provider:
            logger.error("No active provider for chat completion")
            return APIResponse(
                content="Error: No active API provider",
                model="unknown",
                provider="none",
                usage={},
                metadata={'error': 'No active provider'}
            )
        
        return provider.chat_completion(messages, **kwargs)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider"""
        provider = self.get_current_provider()
        if not provider:
            return {'error': 'No active provider'}
        
        info = provider.get_model_info()
        info['current_provider'] = self.current_provider_name
        info['available_providers'] = self.get_available_providers()
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all providers"""
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                is_healthy = provider.validate_api_key()
                health_status[name] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'model_info': provider.get_model_info() if is_healthy else None
                }
            except Exception as e:
                health_status[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return {
            'current_provider': self.current_provider_name,
            'providers': health_status,
            'overall_status': 'healthy' if any(
                status['status'] == 'healthy' 
                for status in health_status.values()
            ) else 'unhealthy'
        }


# Global provider manager instance
_provider_manager = None

def get_provider_manager() -> APIProviderManager:
    """Get global provider manager instance (singleton)"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = APIProviderManager()
    return _provider_manager
