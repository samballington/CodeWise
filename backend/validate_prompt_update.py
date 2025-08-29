#!/usr/bin/env python3
"""
Simple validation test for REQ-UI-UNIFIED-2 system prompt update.
Tests the prompt content directly without complex imports.
"""

import re

def extract_system_prompt_from_file():
    """Extract the system prompt string directly from the source file."""
    try:
        with open('cerebras_agent.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the system prompt method
        start_marker = 'def _get_native_system_prompt(self) -> str:'
        end_marker = 'def _detect_sdk_version(self) -> str:'
        
        start_pos = content.find(start_marker)
        end_pos = content.find(end_marker)
        
        if start_pos == -1 or end_pos == -1:
            raise ValueError("Could not locate system prompt method")
        
        method_content = content[start_pos:end_pos]
        
        # Extract the return statement content (the actual prompt)
        return_match = re.search(r'return\s+"""(.*?)"""', method_content, re.DOTALL)
        if return_match:
            return return_match.group(1).strip()
        else:
            raise ValueError("Could not extract prompt from return statement")
    
    except Exception as e:
        print(f"Error extracting prompt: {e}")
        return None

def test_prompt_content():
    """Test the system prompt content for REQ-UI-UNIFIED-2 requirements."""
    print("Validating REQ-UI-UNIFIED-2: System Prompt Update")
    print("=" * 50)
    
    prompt = extract_system_prompt_from_file()
    if not prompt:
        print("FAILED: Could not extract system prompt")
        return False
    
    print(f"1. Prompt extraction successful ({len(prompt)} characters)")
    
    # Test 2: Required sections
    print("2. Testing required sections...")
    required_sections = [
        "## 8. Unified Response Format",
        "Critical Output Requirement", 
        "EVERY response MUST be a single, valid JSON object",
        "UnifiedAgentResponse schema"
    ]
    
    all_sections_found = True
    for section in required_sections:
        if section in prompt:
            print(f"   [PASS] {section}")
        else:
            print(f"   [FAIL] Missing: {section}")
            all_sections_found = False
    
    # Test 3: Content block types
    print("3. Testing content block types...")
    required_blocks = [
        "TextBlock",
        "ComponentAnalysisBlock", 
        "MermaidDiagramBlock",
        "CodeSnippetBlock", 
        "MarkdownTableBlock"
    ]
    
    all_blocks_found = True
    for block in required_blocks:
        if block in prompt:
            print(f"   [PASS] {block}")
        else:
            print(f"   [FAIL] Missing: {block}")
            all_blocks_found = False
    
    # Test 4: Chain of thought example
    print("4. Testing chain-of-thought example...")
    chain_elements = [
        "Chain-of-Thought Example",
        "Your Internal Process",
        "Your JSON Response"
    ]
    
    chain_complete = True
    for element in chain_elements:
        if element in prompt:
            print(f"   [PASS] {element}")
        else:
            print(f"   [FAIL] Missing: {element}")
            chain_complete = False
    
    # Test 5: JSON format requirements
    print("5. Testing JSON format requirements...")
    json_reqs = [
        "valid JSON object",
        "no preamble, no postamble", 
        "structured JSON",
        "frontend UI system"
    ]
    
    json_complete = True
    for req in json_reqs:
        if req in prompt:
            print(f"   [PASS] {req}")
        else:
            print(f"   [FAIL] Missing: {req}")
            json_complete = False
    
    # Test 6: Block selection rules
    print("6. Testing block selection rules...")
    selection_rules = [
        "Use TextBlock for",
        "Use ComponentAnalysisBlock for",
        "Use MermaidDiagramBlock for"
    ]
    
    rules_complete = True
    for rule in selection_rules:
        if rule in prompt:
            print(f"   [PASS] {rule}")
        else:
            print(f"   [FAIL] Missing: {rule}")
            rules_complete = False
    
    # Test 7: Validation checkpoint
    print("7. Testing validation checkpoint...")
    validation_elements = [
        "Validation Checkpoint",
        "Before sending your response, verify",
        "Response is valid JSON starting with"
    ]
    
    validation_complete = True
    for element in validation_elements:
        if element in prompt:
            print(f"   [PASS] {element}")
        else:
            print(f"   [FAIL] Missing: {element}")
            validation_complete = False
    
    # Test 8: Original content preservation
    print("8. Testing original content preservation...")
    original_content = [
        "Workflow A: Standard Code Query",
        "Workflow B: Codebase Onboarding",
        "navigate_filesystem",
        "query_codebase"
    ]
    
    preservation_complete = True
    for content in original_content:
        if content in prompt:
            print(f"   [PASS] Preserved: {content}")
        else:
            print(f"   [FAIL] Lost: {content}")
            preservation_complete = False
    
    # Test 9: JSON examples present
    print("9. Testing JSON examples...")
    json_examples = re.findall(r'```json.*?```', prompt, re.DOTALL)
    if len(json_examples) >= 3:
        print(f"   [PASS] Found {len(json_examples)} JSON examples")
    else:
        print(f"   [FAIL] Only {len(json_examples)} JSON examples (need >= 3)")
        return False
    
    # Final assessment
    all_tests_passed = (
        all_sections_found and
        all_blocks_found and  
        chain_complete and
        json_complete and
        rules_complete and
        validation_complete and
        preservation_complete
    )
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("SUCCESS: REQ-UI-UNIFIED-2 is 100% complete!")
        print("✓ System prompt updated with unified response format")
        print("✓ All content block types documented")
        print("✓ Chain-of-thought example included")
        print("✓ JSON format requirements specified")
        print("✓ Block selection rules defined")
        print("✓ Validation checkpoint added")
        print("✓ Original workflow content preserved")
        print("✓ Comprehensive JSON examples provided")
        return True
    else:
        print("FAILED: Some requirements not met")
        return False

if __name__ == "__main__":
    success = test_prompt_content()
    
    if success:
        print("\nREQ-UI-UNIFIED-2 validation complete - system is ready!")
    else:
        print("\nREQ-UI-UNIFIED-2 validation failed - needs fixes")
    
    import sys
    sys.exit(0 if success else 1)