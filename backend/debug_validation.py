#!/usr/bin/env python3
"""
Debug validation behavior.
"""

from schemas.ui_schemas import validate_response_structure, TextBlock
import json

# Test 1: Valid data
print("Testing valid data...")
valid_data = {
    "response": [
        {
            "block_type": "text",
            "content": "Test content"
        }
    ]
}

try:
    result = validate_response_structure(valid_data)
    print("Valid data accepted:", type(result))
except Exception as e:
    print("Valid data failed:", e)

# Test 2: Invalid block_type
print("\nTesting invalid block_type...")
invalid_data = {
    "response": [
        {
            "wrong_field": "text",  # Should be block_type
            "content": "Missing block_type"
        }
    ]
}

try:
    result = validate_response_structure(invalid_data)
    print("Invalid data incorrectly accepted:", type(result))
    print("Response blocks:", result.response[0])
except Exception as e:
    print("Invalid data correctly rejected:", e)

# Test 3: Check what TextBlock accepts
print("\nTesting TextBlock directly...")
try:
    block = TextBlock(wrong_field="test", content="test")  
    print("TextBlock incorrectly created:", block)
except Exception as e:
    print("TextBlock correctly rejected:", e)