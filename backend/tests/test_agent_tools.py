import os
import asyncio
import pytest

from agent import CodeWiseAgent

@pytest.mark.asyncio
async def test_tools_callable(tmp_path, monkeypatch):
    """Ensure code_search, file_glimpse, list_entities tools execute without error."""
    # Prepare dummy workspace file for file_glimpse using tmp_path
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    dummy_path = workspace_dir / "test_dummy.txt"
    dummy_path.write_text("hello world\n" * 5)
    
    # Set workspace environment variable for the agent
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    agent = CodeWiseAgent(openai_api_key="", mcp_server_url="http://mcp_server:8001")

    # Map tool names to callables
    tool_map = {tool.name: tool.func for tool in agent.tools}

    # code_search should return string (even if "No results")
    code_search_result = tool_map["code_search"]("dummy search")
    assert isinstance(code_search_result, str)

    # file_glimpse should return file content
    glimpse_result = tool_map["file_glimpse"]("test_dummy.txt")
    assert "hello world" in glimpse_result

    # list_entities returns str (may be empty)
    entities_result = tool_map["list_entities"](None)
    assert isinstance(entities_result, str) 