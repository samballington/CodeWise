"""
Tests for the API Provider System

This module tests the multi-provider API abstraction layer including
OpenAI and Kimi K2 providers, provider switching, and error handling.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from api_providers import (
    APIProviderManager, 
    OpenAIProvider, 
    KimiProvider, 
    BaseProvider,
    APIResponse
)


class TestAPIResponse:
    """Test the APIResponse dataclass"""
    
    def test_api_response_creation(self):
        """Test creating an APIResponse object"""
        response = APIResponse(
            content="Test response",
            model="gpt-4",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"finish_reason": "stop"}
        )
        
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.usage["prompt_tokens"] == 10
        assert response.metadata["finish_reason"] == "stop"


class TestOpenAIProvider:
    """Test the OpenAI provider implementation"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization"""
        provider = OpenAIProvider()
        
        assert provider.api_key == "test-key"
        assert provider.model_name == "gpt-4"
        assert provider.provider_name == "openai"
        assert provider.embedding_model == "text-embedding-3-small"
    
    def test_openai_provider_invalid_model(self):
        """Test OpenAI provider with invalid model name"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider(model_name="invalid-model")
            # Should fallback to gpt-4
            assert provider.model_name == "gpt-4"
    
    def test_openai_provider_missing_api_key(self):
        """Test OpenAI provider without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIProvider()
    
    @patch('openai.embeddings.create')
    def test_generate_embeddings_success(self, mock_create):
        """Test successful embedding generation"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_create.return_value = mock_response
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            embeddings = provider.generate_embeddings(["text1", "text2"])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        mock_create.assert_called_once()
    
    @patch('openai.embeddings.create')
    def test_generate_embeddings_failure(self, mock_create):
        """Test embedding generation failure handling"""
        mock_create.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            embeddings = provider.generate_embeddings(["text1", "text2"])
        
        # Should return empty embeddings on failure
        assert len(embeddings) == 2
        assert embeddings[0] == [0.0] * 1536
        assert embeddings[1] == [0.0] * 1536
    
    @patch('openai.chat.completions.create')
    def test_chat_completion_success(self, mock_create):
        """Test successful chat completion"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_create.return_value = mock_response
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            response = provider.chat_completion([{"role": "user", "content": "Hello"}])
        
        assert response.content == "Test response"
        assert response.provider == "openai"
        assert response.usage["prompt_tokens"] == 10
        assert response.metadata["finish_reason"] == "stop"
    
    @patch('openai.chat.completions.create')
    def test_chat_completion_failure(self, mock_create):
        """Test chat completion failure handling"""
        mock_create.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIProvider()
            response = provider.chat_completion([{"role": "user", "content": "Hello"}])
        
        assert "Error: API Error" in response.content
        assert response.provider == "openai"
        assert "error" in response.metadata


class TestKimiProvider:
    """Test the Kimi K2 provider implementation"""
    
    def test_kimi_provider_initialization(self):
        """Test Kimi provider initialization"""
        provider = KimiProvider()
        
        assert provider.api_key.startswith("sk-or-v1-")
        assert provider.model_name == "moonshot-v1-8k"
        assert provider.provider_name == "kimi"
        assert provider.base_url == "https://api.moonshot.cn/v1"
    
    def test_kimi_provider_invalid_model(self):
        """Test Kimi provider with invalid model name"""
        provider = KimiProvider(model_name="invalid-model")
        # Should fallback to moonshot-v1-8k
        assert provider.model_name == "moonshot-v1-8k"
    
    def test_generate_embeddings_placeholder(self):
        """Test Kimi embedding generation (placeholder implementation)"""
        provider = KimiProvider()
        embeddings = provider.generate_embeddings(["text1", "text2"])
        
        # Should return placeholder embeddings
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert embeddings[0] == [0.1] * 1536
    
    @patch('requests.post')
    def test_chat_completion_success(self, mock_post):
        """Test successful Kimi chat completion"""
        # Mock requests response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Kimi response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25}
        }
        mock_post.return_value = mock_response
        
        provider = KimiProvider()
        response = provider.chat_completion([{"role": "user", "content": "Hello"}])
        
        assert response.content == "Kimi response"
        assert response.provider == "kimi"
        assert response.usage["prompt_tokens"] == 15
        assert response.metadata["finish_reason"] == "stop"
    
    @patch('requests.post')
    def test_chat_completion_failure(self, mock_post):
        """Test Kimi chat completion failure handling"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        provider = KimiProvider()
        response = provider.chat_completion([{"role": "user", "content": "Hello"}])
        
        assert "Error: Kimi API error: 400" in response.content
        assert response.provider == "kimi"
        assert "error" in response.metadata


class TestAPIProviderManager:
    """Test the API provider manager"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_provider_manager_initialization(self):
        """Test provider manager initialization"""
        manager = APIProviderManager()
        
        assert "openai" in manager.providers
        assert "kimi" in manager.providers
        assert manager.current_provider_name in ["openai", "kimi"]
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_available_providers(self):
        """Test getting available providers"""
        manager = APIProviderManager()
        providers = manager.get_available_providers()
        
        assert "openai" in providers
        assert "kimi" in providers
        assert len(providers) >= 2
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_get_current_provider(self):
        """Test getting current provider"""
        manager = APIProviderManager()
        provider = manager.get_current_provider()
        
        assert provider is not None
        assert isinstance(provider, BaseProvider)
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(OpenAIProvider, 'validate_api_key', return_value=True)
    def test_switch_provider_success(self, mock_validate):
        """Test successful provider switching"""
        manager = APIProviderManager()
        
        # Switch to OpenAI
        success = manager.switch_provider("openai")
        
        assert success is True
        assert manager.current_provider_name == "openai"
        mock_validate.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_switch_provider_invalid(self):
        """Test switching to invalid provider"""
        manager = APIProviderManager()
        
        success = manager.switch_provider("invalid-provider")
        
        assert success is False
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(OpenAIProvider, 'validate_api_key', return_value=False)
    def test_switch_provider_validation_failure(self, mock_validate):
        """Test provider switching with validation failure"""
        manager = APIProviderManager()
        
        success = manager.switch_provider("openai")
        
        assert success is False
        mock_validate.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(OpenAIProvider, 'generate_embeddings', return_value=[[0.1, 0.2]])
    def test_generate_embeddings_delegation(self, mock_generate):
        """Test embedding generation delegation to current provider"""
        manager = APIProviderManager()
        manager.current_provider_name = "openai"
        
        embeddings = manager.generate_embeddings(["test"])
        
        assert embeddings == [[0.1, 0.2]]
        mock_generate.assert_called_once_with(["test"])
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(OpenAIProvider, 'chat_completion')
    def test_chat_completion_delegation(self, mock_chat):
        """Test chat completion delegation to current provider"""
        mock_response = APIResponse(
            content="Test", model="gpt-4", provider="openai", 
            usage={}, metadata={}
        )
        mock_chat.return_value = mock_response
        
        manager = APIProviderManager()
        manager.current_provider_name = "openai"
        
        response = manager.chat_completion([{"role": "user", "content": "Hello"}])
        
        assert response.content == "Test"
        mock_chat.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(OpenAIProvider, 'get_model_info', return_value={"provider": "openai"})
    def test_get_provider_info(self, mock_info):
        """Test getting provider information"""
        manager = APIProviderManager()
        manager.current_provider_name = "openai"
        
        info = manager.get_provider_info()
        
        assert info["provider"] == "openai"
        assert "current_provider" in info
        assert "available_providers" in info
        mock_info.assert_called_once()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch.object(OpenAIProvider, 'validate_api_key', return_value=True)
    @patch.object(KimiProvider, 'validate_api_key', return_value=True)
    def test_health_check(self, mock_kimi_validate, mock_openai_validate):
        """Test health check for all providers"""
        manager = APIProviderManager()
        
        health = manager.health_check()
        
        assert "current_provider" in health
        assert "providers" in health
        assert "overall_status" in health
        assert health["overall_status"] == "healthy"
        
        # Check individual provider health
        assert "openai" in health["providers"]
        assert "kimi" in health["providers"]
        assert health["providers"]["openai"]["status"] == "healthy"
        assert health["providers"]["kimi"]["status"] == "healthy"


# Integration tests
class TestAPIProviderIntegration:
    """Integration tests for the API provider system"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_provider_manager_singleton(self):
        """Test that get_provider_manager returns singleton"""
        from api_providers import get_provider_manager
        
        manager1 = get_provider_manager()
        manager2 = get_provider_manager()
        
        assert manager1 is manager2
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('openai.embeddings.create')
    @patch('requests.post')
    def test_end_to_end_provider_switching(self, mock_requests, mock_openai):
        """Test complete provider switching workflow"""
        # Mock OpenAI
        mock_openai_response = Mock()
        mock_openai_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.return_value = mock_openai_response
        
        # Mock Kimi
        mock_kimi_response = Mock()
        mock_kimi_response.status_code = 200
        mock_kimi_response.json.return_value = {
            "choices": [{"message": {"content": "Kimi response"}}],
            "usage": {}
        }
        mock_requests.return_value = mock_kimi_response
        
        manager = APIProviderManager()
        
        # Test OpenAI
        manager.switch_provider("openai")
        embeddings = manager.generate_embeddings(["test"])
        assert len(embeddings) > 0
        
        # Test Kimi
        manager.switch_provider("kimi")
        response = manager.chat_completion([{"role": "user", "content": "Hello"}])
        assert response.content == "Kimi response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])