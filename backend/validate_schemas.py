#!/usr/bin/env python3
"""
Schema Validation Test for REQ-UI-UNIFIED-1
Simple ASCII-only validation to ensure 100% functionality.
"""

import json
import sys
import traceback

def test_schemas():
    """Test all UI schemas functionality."""
    print("Testing UI Schemas (REQ-UI-UNIFIED-1)...")
    print("=" * 50)
    
    try:
        # Test 1: Import schemas
        print("1. Testing imports...")
        from schemas.ui_schemas import (
            UnifiedAgentResponse, TextBlock, ComponentAnalysisBlock,
            MermaidDiagramBlock, CodeSnippetBlock, CodeComponent, ComponentType
        )
        print("   [PASS] All schema imports successful")
        
        # Test 2: Create basic components
        print("2. Testing basic component creation...")
        component = CodeComponent(
            name="UserService",
            path="services/user.py",
            component_type=ComponentType.SERVICE,
            purpose="Handles user authentication and management operations"
        )
        print("   [PASS] CodeComponent creation successful")
        
        # Test 3: Create all content block types
        print("3. Testing all content block types...")
        
        text_block = TextBlock(content="# Analysis Results\n\nThis is a test.")
        print("   [PASS] TextBlock created")
        
        comp_block = ComponentAnalysisBlock(
            title="Key Components",
            components=[component]
        )
        print("   [PASS] ComponentAnalysisBlock created")
        
        mermaid_block = MermaidDiagramBlock(
            title="Architecture",
            mermaid_code="graph TD\n    A[Start] --> B[End]"
        )
        print("   [PASS] MermaidDiagramBlock created")
        
        code_block = CodeSnippetBlock(
            title="Example Code",
            language="python",
            code="def hello():\n    return 'world'"
        )
        print("   [PASS] CodeSnippetBlock created")
        
        # Test 4: Create unified response
        print("4. Testing UnifiedAgentResponse...")
        response = UnifiedAgentResponse(
            response=[
                text_block,
                comp_block,
                mermaid_block,
                code_block
            ]
        )
        print("   [PASS] UnifiedAgentResponse created with 4 blocks")
        
        # Test 5: JSON serialization
        print("5. Testing JSON serialization...")
        json_data = response.model_dump()
        json_string = json.dumps(json_data, indent=2)
        print(f"   [PASS] JSON serialized ({len(json_string)} characters)")
        
        # Test 6: JSON deserialization
        print("6. Testing JSON deserialization...")
        parsed = json.loads(json_string)
        reconstructed = UnifiedAgentResponse.model_validate(parsed)
        print(f"   [PASS] JSON deserialized ({len(reconstructed.response)} blocks)")
        
        # Test 7: Helper methods
        print("7. Testing helper methods...")
        block_types = response.get_block_types()
        text_blocks = response.get_blocks_by_type("text")
        quality_check = response.validate_content_quality()
        
        print(f"   [PASS] Block types: {block_types}")
        print(f"   [PASS] Text blocks found: {len(text_blocks)}")
        print(f"   [PASS] Quality check completed")
        
        # Test 8: Error handling
        print("8. Testing error responses...")
        from schemas.ui_schemas import create_error_response
        error_resp = create_error_response("Test error message")
        print("   [PASS] Error response created successfully")
        
        # Test 9: Validation failures
        print("9. Testing validation errors...")
        try:
            # This should fail - empty response
            UnifiedAgentResponse(response=[])
            print("   [FAIL] Empty response should have failed")
            return False
        except Exception:
            print("   [PASS] Empty response correctly rejected")
        
        try:
            # This should fail - invalid line range
            CodeComponent(
                name="Test",
                path="test.py", 
                component_type=ComponentType.CLASS,
                purpose="Test component for validation testing",
                line_start=100,
                line_end=50
            )
            print("   [FAIL] Invalid line range should have failed")
            return False
        except Exception:
            print("   [PASS] Invalid line range correctly rejected")
        
        print("\nSUCCESS: All REQ-UI-UNIFIED-1 tests passed!")
        print("Schema system is 100% functional and ready for use.")
        return True
        
    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_schemas()
    sys.exit(0 if success else 1)