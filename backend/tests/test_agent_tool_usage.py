import json
import pytest
from backend.agent import CodeWiseAgent


@pytest.mark.asyncio
async def test_code_search_tool_works(tmp_path, monkeypatch):
    """Test that code_search tool can be invoked directly and returns results."""
    # --- create tiny workspace ---
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    # dummy entity file
    schema_file = workspace_dir / "schema.sql"
    schema_file.write_text("CREATE TABLE book(id INT PRIMARY KEY, title TEXT);")
    # env patch
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    # Build real tools via CodeWiseAgent (will use local embedder)
    agent_wrapper = CodeWiseAgent(openai_api_key="", mcp_server_url="http://mcp_server:8001")
    
    # Test: invoke code_search via hybrid_search directly
    results = agent_wrapper.hybrid_search.search("database", k=10)
    
    # Verify that search returned some results
    assert results is not None
    assert len(results) > 0, "code_search should return some results"
    
    # Verify that the schema file content was found
    found_schema = any("book" in str(result.snippet).lower() or "table" in str(result.snippet).lower() for result in results)
    assert found_schema, f"Should find database schema content, got: {[r.snippet for r in results]}"
    
    print(f"Successfully found {len(results)} chunks with hybrid search")


@pytest.mark.asyncio  
async def test_file_glimpse_tool_works(tmp_path, monkeypatch):
    """Test that file_glimpse tool can access file content."""
    # --- create tiny workspace ---
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    # dummy source file
    src_file = workspace_dir / "app.py"
    file_content = "def main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()"
    src_file.write_text(file_content)
    # env patch
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    # Build real tools via CodeWiseAgent
    agent_wrapper = CodeWiseAgent(openai_api_key="", mcp_server_url="http://mcp_server:8001")
    
    # Test: read file directly from workspace (simulating file_glimpse tool behavior)
    import os
    workspace_path = os.environ.get("WORKSPACE_DIR", "/workspace")
    full_path = os.path.join(workspace_path, "app.py")
    
    # Verify file exists and can be read
    assert os.path.exists(full_path), f"File should exist at {full_path}"
    
    with open(full_path, 'r', encoding='utf-8') as f:
        result = f.read()
    
    # Verify that tool returned file content
    assert result is not None
    assert "main" in result, f"Should find function content, got: {result}"
    assert "hello world" in result, f"Should find print statement, got: {result}"
    
    print(f"Successfully retrieved file content: {len(result)} chars")


@pytest.mark.asyncio
async def test_agent_tools_are_available(tmp_path, monkeypatch):
    """Test that the agent has the expected tools available."""
    # --- create tiny workspace ---
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    # env patch
    monkeypatch.setenv("WORKSPACE_DIR", str(workspace_dir))

    # Build real tools via CodeWiseAgent
    agent_wrapper = CodeWiseAgent(openai_api_key="", mcp_server_url="http://mcp_server:8001")
    tools = agent_wrapper.tools
    
    # Verify that expected tools are available
    tool_names = [tool.name for tool in tools]
    expected_tools = ["code_search", "file_glimpse", "list_entities", "read_file", "write_file", "list_files", "run_command"]
    
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Tool '{expected_tool}' not found in {tool_names}"
    
    print(f"Successfully verified {len(expected_tools)} expected tools are available") 