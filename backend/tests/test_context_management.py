"""
Comprehensive test suite for context management and summarization architecture.

Tests the "Summarize-then-Synthesize" implementation including:
- Core summarization engine
- Specialized compression prompts
- Content classification and quality preservation
- Context monitoring and analytics
- Graceful degradation strategies
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from cerebras_agent import CerebrasNativeAgent


class TestContextManagement:
    """Test suite for context management functionality."""
    
    @pytest.fixture
    def agent(self):
        """Create a CerebrasNativeAgent instance for testing."""
        with patch('cerebras_agent.Cerebras'):
            agent = CerebrasNativeAgent()
            agent.client = Mock()
            return agent
    
    @pytest.fixture
    def large_tool_result(self):
        """Create a large tool result that exceeds summarization threshold."""
        # Generate content larger than 20,000 characters
        large_content = {
            "files": [
                {
                    "path": f"src/component_{i}.py",
                    "content": "class Component{}:\n    def __init__(self):\n        self.data = '{}'\n    def process(self):\n        return self.data.upper()".format(i, "x" * 500)
                } for i in range(50)
            ],
            "analysis": "This is a comprehensive codebase analysis. " + "Detailed analysis content. " * 1000,
            "recommendations": ["Recommendation {}".format(i) for i in range(100)]
        }
        return large_content
    
    @pytest.fixture
    def small_tool_result(self):
        """Create a small tool result that stays below summarization threshold."""
        return {
            "files": ["src/main.py", "src/utils.py"],
            "summary": "Simple project structure"
        }

    async def test_core_summarization_engine_threshold_detection(self, agent, large_tool_result, small_tool_result):
        """Test REQ-CTX-UNIFIED: Core summarization engine with proper threshold detection."""
        
        # Mock the summarization method
        agent._summarize_with_llm = AsyncMock(return_value="## Summarized Content\n- Key point 1\n- Key point 2")
        
        # Test large content triggers summarization
        large_result_str = json.dumps(large_tool_result, indent=2)
        assert len(large_result_str) > 20000, "Test data should exceed threshold"
        
        result = await agent._execute_tool_and_summarize("query_codebase", large_tool_result, "analyze the codebase")
        
        # Verify summarization was called
        agent._summarize_with_llm.assert_called_once()
        assert result == "## Summarized Content\n- Key point 1\n- Key point 2"
        
        # Reset mock
        agent._summarize_with_llm.reset_mock()
        
        # Test small content bypasses summarization
        small_result_str = json.dumps(small_tool_result, indent=2)
        assert len(small_result_str) <= 20000, "Test data should be below threshold"
        
        result = await agent._execute_tool_and_summarize("simple_tool", small_tool_result, "simple query")
        
        # Verify summarization was NOT called
        agent._summarize_with_llm.assert_not_called()
        assert result == small_result_str

    async def test_specialized_compression_prompts(self, agent):
        """Test REQ-CTX-PROMPT: Specialized compression prompts for different tool types."""
        
        # Test query_codebase prompt
        prompt = agent._get_compression_prompt("query_codebase", "analyze the architecture")
        assert "CONTENT CLASSIFICATION" in prompt
        assert "ESSENTIAL: Class definitions, main functions, API endpoints" in prompt
        assert "## Key Components" in prompt
        
        # Test navigate_filesystem prompt
        prompt = agent._get_compression_prompt("navigate_filesystem", "explore the project structure")
        assert "DIRECTORY STRUCTURE COMPRESSION" in prompt
        assert "ESSENTIAL: Main directories, config files, entry points" in prompt
        assert "## Directory Structure" in prompt
        
        # Test default prompt
        prompt = agent._get_compression_prompt("unknown_tool", "generic query")
        assert "GENERAL TOOL OUTPUT COMPRESSION" in prompt
        assert "ESSENTIAL: Core functionality, main results, key data" in prompt
        assert "## Summary" in prompt

    async def test_content_classification_and_quality(self, agent, large_tool_result):
        """Test REQ-CTX-QUALITY: Content classification and structured output formatting."""
        
        # Mock the Cerebras client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = """## Key Components
- **AuthenticationService** (`src/auth/service.py:23-67`)
  - Type: class
  - Purpose: Handles user authentication and session management
  - Key methods: login(), logout(), validate_token()
  - Dependencies: UserRepository, TokenManager

## Architecture Patterns
- **Pattern**: Repository Pattern
- **Implementation**: Separates data access from business logic
- **Benefits**: Improved testability and maintainability

## Critical Issues
- **Issue**: Missing input validation in authentication endpoints
- **Impact**: Security vulnerability
- **Recommendation**: Implement comprehensive input sanitization"""
        
        agent.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await agent._summarize_with_llm(
            json.dumps(large_tool_result), 
            "analyze the codebase architecture", 
            "query_codebase"
        )
        
        # Verify structured output format
        assert "## Key Components" in result
        assert "## Architecture Patterns" in result
        assert "## Critical Issues" in result
        
        # Verify content classification is preserved
        assert "Type:" in result and "Purpose:" in result
        assert "Dependencies:" in result
        assert "Security vulnerability" in result

    async def test_context_monitoring_and_analytics(self, agent, large_tool_result):
        """Test REQ-CTX-MONITORING: Comprehensive context usage analytics and logging."""
        
        # Mock the summarization to capture monitoring calls
        agent._summarize_with_llm = AsyncMock(return_value="Summarized content")
        
        # Capture log calls
        with patch('cerebras_agent.logger') as mock_logger:
            large_result_str = json.dumps(large_tool_result, indent=2)
            
            result = await agent._execute_tool_and_summarize(
                "query_codebase", 
                large_tool_result, 
                "analyze the codebase"
            )
            
            # Verify monitoring logs were called
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            
            # Check for size monitoring
            size_log = next((log for log in log_calls if "Tool output size:" in log and "above threshold" in log), None)
            assert size_log is not None, "Should log when content exceeds threshold"
            
            # Check for compression metrics
            compression_log = next((log for log in log_calls if "Compression applied:" in log), None)
            assert compression_log is not None, "Should log compression metrics"

    async def test_graceful_degradation_strategy(self, agent):
        """Test REQ-CTX-FALLBACK: Four-tier graceful degradation strategy."""
        
        # Test context health assessment
        context_health = {
            "total_tokens": 60000,
            "max_tokens": 65536,
            "utilization_percent": 91.6,
            "messages_count": 10
        }
        
        # Mock methods for different degradation tiers
        agent._progressive_context_trimming = AsyncMock(return_value=[{"role": "user", "content": "trimmed"}])
        agent._multi_turn_synthesis = AsyncMock(return_value="Multi-turn synthesis result")
        agent._partial_response_with_explanation = Mock(return_value="Partial response with explanation")
        
        # Test Tier 2: Progressive trimming (90-95% utilization)
        context_health["utilization_percent"] = 93.0
        result = await agent._apply_graceful_degradation([{"role": "user", "content": "test"}], context_health)
        agent._progressive_context_trimming.assert_called_once()
        
        # Test Tier 3: Multi-turn synthesis (95-98% utilization)
        agent._progressive_context_trimming.reset_mock()
        context_health["utilization_percent"] = 96.5
        result = await agent._apply_graceful_degradation([{"role": "user", "content": "test"}], context_health)
        agent._multi_turn_synthesis.assert_called_once()
        
        # Test Tier 4: Partial response (98%+ utilization)
        agent._multi_turn_synthesis.reset_mock()
        context_health["utilization_percent"] = 99.0
        result = await agent._apply_graceful_degradation([{"role": "user", "content": "test"}], context_health)
        agent._partial_response_with_explanation.assert_called_once()

    async def test_context_utilization_calculation(self, agent):
        """Test context utilization calculation accuracy."""
        
        messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Test message 2"},
        ]
        
        # Mock token counting
        with patch.object(agent, '_count_tokens', return_value=1000):
            context_health = agent._assess_context_health(messages)
            
            assert context_health["total_tokens"] == 1000
            assert context_health["messages_count"] == 3
            assert "utilization_percent" in context_health
            assert 0 <= context_health["utilization_percent"] <= 100

    async def test_compression_ratio_calculation(self, agent, large_tool_result):
        """Test compression ratio calculation and logging."""
        
        # Mock summarization to return shorter content
        compressed_content = "## Summary\n- Key point 1\n- Key point 2"
        agent._summarize_with_llm = AsyncMock(return_value=compressed_content)
        
        original_size = len(json.dumps(large_tool_result, indent=2))
        
        with patch('cerebras_agent.logger') as mock_logger:
            result = await agent._execute_tool_and_summarize(
                "query_codebase", 
                large_tool_result, 
                "analyze the codebase"
            )
            
            # Verify compression ratio is calculated and logged
            compression_log = next(
                (call.args[0] for call in mock_logger.info.call_args_list 
                 if "Compression applied:" in call.args[0]), 
                None
            )
            
            assert compression_log is not None
            assert "compression ratio:" in compression_log
            assert "%" in compression_log

    async def test_error_handling_in_summarization(self, agent, large_tool_result):
        """Test error handling when summarization fails."""
        
        # Mock summarization to raise an exception
        agent._summarize_with_llm = AsyncMock(side_effect=Exception("Summarization failed"))
        
        with patch('cerebras_agent.logger') as mock_logger:
            # Should fallback to original content when summarization fails
            result = await agent._execute_tool_and_summarize(
                "query_codebase", 
                large_tool_result, 
                "analyze the codebase"
            )
            
            # Should return original content as fallback
            original_content = json.dumps(large_tool_result, indent=2)
            assert result == original_content
            
            # Should log the error
            error_logged = any(
                "Failed to summarize tool output" in str(call.args[0]) 
                for call in mock_logger.error.call_args_list
            )
            assert error_logged

    def test_compression_prompt_consistency(self, agent):
        """Test that compression prompts are consistent and well-formed."""
        
        tool_types = ["query_codebase", "navigate_filesystem", "unknown_tool"]
        
        for tool_type in tool_types:
            prompt = agent._get_compression_prompt(tool_type, "test query")
            
            # Verify prompt structure
            assert len(prompt) > 100, f"Prompt for {tool_type} should be substantial"
            assert "ESSENTIAL:" in prompt, f"Prompt for {tool_type} should have ESSENTIAL classification"
            assert "##" in prompt, f"Prompt for {tool_type} should have structured output format"
            
            # Verify tool-specific content
            if tool_type == "query_codebase":
                assert "Class definitions" in prompt
                assert "API endpoints" in prompt
            elif tool_type == "navigate_filesystem":
                assert "DIRECTORY STRUCTURE" in prompt
                assert "config files" in prompt

    async def test_integration_with_existing_workflow(self, agent):
        """Test integration of context management with existing agent workflow."""
        
        # Mock the complete workflow
        mock_tool_result = {"analysis": "test analysis", "files": ["file1.py"]}
        
        # Mock dependencies
        agent._summarize_with_llm = AsyncMock(return_value="Summarized analysis")
        agent._assess_context_health = Mock(return_value={
            "total_tokens": 50000,
            "utilization_percent": 85.0,
            "messages_count": 5
        })
        
        # Test that summarization integrates smoothly
        result = await agent._execute_tool_and_summarize(
            "query_codebase", 
            mock_tool_result, 
            "analyze the code"
        )
        
        # Should return summarized content for integration
        assert result == "Summarized analysis"
        
        # Verify context health was assessed
        agent._assess_context_health.assert_called()


# Performance and stress tests
class TestContextManagementPerformance:
    """Performance tests for context management."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for performance testing."""
        with patch('cerebras_agent.Cerebras'):
            agent = CerebrasNativeAgent()
            agent.client = Mock()
            return agent
    
    async def test_summarization_performance(self, agent):
        """Test that summarization completes within reasonable time."""
        
        # Create very large content
        large_content = {"data": "x" * 100000}
        
        agent._summarize_with_llm = AsyncMock(return_value="Summarized")
        
        import time
        start_time = time.time()
        
        result = await agent._execute_tool_and_summarize(
            "query_codebase",
            large_content,
            "test query"
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within 5 seconds (generous for testing)
        assert execution_time < 5.0, f"Summarization took too long: {execution_time}s"

    def test_memory_efficiency(self, agent):
        """Test that context management doesn't cause memory leaks."""
        
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run multiple summarization cycles
        for i in range(10):
            large_content = {"iteration": i, "data": "x" * 10000}
            # Simulate processing without actual async calls
            content_str = json.dumps(large_content)
            prompt = agent._get_compression_prompt("query_codebase", f"test {i}")
        
        # Force garbage collection after test
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be minimal
        growth = final_objects - initial_objects
        assert growth < 1000, f"Excessive memory growth: {growth} objects"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])