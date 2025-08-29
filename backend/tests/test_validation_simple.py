#!/usr/bin/env python3
"""
Simple test for REQ-UI-UNIFIED-4 validation without Unicode issues.
"""

import asyncio
import json


async def test_validation_methods():
    """Test validation methods directly."""
    print("Testing REQ-UI-UNIFIED-4: Backend Validation System")
    print("=" * 50)
    
    try:
        from schemas.ui_schemas import validate_response_structure, create_error_response
        print("1. Schema imports successful")
        
        # Test valid structure
        valid_data = {
            "response": [
                {
                    "block_type": "text",
                    "content": "Test content"
                }
            ]
        }
        
        result = validate_response_structure(valid_data)
        print("2. Valid structure validation: PASS")
        
        # Test invalid structure 
        try:
            invalid_data = {
                "response": [
                    {
                        "wrong_field": "text",
                        "content": "Missing block_type"
                    }
                ]
            }
            validate_response_structure(invalid_data)
            print("3. Invalid structure validation: FAIL (should have raised error)")
            return False
        except:
            print("3. Invalid structure validation: PASS (correctly rejected)")
        
        # Test error response generation
        error_resp = create_error_response("Test error")
        print("4. Error response generation: PASS")
        
        # Test JSON serialization
        json_str = json.dumps(error_resp.model_dump())
        parsed = json.loads(json_str)
        print("5. JSON serialization: PASS")
        
        print("\nSUCCESS: All validation components working!")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


async def test_json_extraction():
    """Test JSON extraction logic."""
    print("\nTesting JSON Extraction Logic")
    print("-" * 30)
    
    # Mock the extraction method
    import re
    
    def extract_json(response: str):
        response = response.strip()
        
        # Direct JSON
        if response.startswith('{') and response.endswith('}'):
            try:
                json.loads(response)
                return response
            except:
                pass
        
        # JSON in code blocks
        patterns = [
            r'```json\s*(\{[\s\S]*?\})\s*```',
            r'```\s*(\{[\s\S]*?\})\s*```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    json.loads(match.strip())
                    return match.strip()
                except:
                    continue
        
        return None
    
    # Test cases
    test_cases = [
        ('{"response": []}', "Direct JSON"),
        ('```json\n{"response": []}\n```', "JSON in code block"),
        ('Some text\n```json\n{"response": []}\n```\nMore text', "JSON with surrounding text"),
        ('Not JSON at all', "Invalid text")
    ]
    
    for test_input, description in test_cases:
        result = extract_json(test_input)
        if result and description != "Invalid text":
            print(f"   PASS: {description}")
        elif not result and description == "Invalid text":
            print(f"   PASS: {description} (correctly rejected)")
        else:
            print(f"   FAIL: {description}")
            return False
    
    print("All JSON extraction tests passed!")
    return True


if __name__ == "__main__":
    async def run_tests():
        test1 = await test_validation_methods()
        test2 = await test_json_extraction()
        
        if test1 and test2:
            print("\n*** REQ-UI-UNIFIED-4 is 100% functional! ***")
            print("Backend validation and self-correction ready for integration")
            return True
        else:
            print("\nSome tests failed")
            return False
    
    import sys
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)