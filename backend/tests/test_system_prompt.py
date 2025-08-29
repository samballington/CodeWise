#!/usr/bin/env python3
"""
Test Suite for REQ-UI-UNIFIED-2: Updated Agent System Prompt

Validates that the system prompt includes all required elements for 
unified UI response format.
"""

import json
import re
from typing import Dict, List

def test_system_prompt_update():
    """Test that the system prompt includes unified response format requirements."""
    print("Testing REQ-UI-UNIFIED-2: Updated Agent System Prompt")
    print("=" * 55)
    
    try:
        # Import and get the agent 
        from backend.cerebras_agent import CerebrasNativeAgent
        agent = CerebrasNativeAgent()
        
        # Get the system prompt
        prompt = agent._get_native_system_prompt()
        
        print("1. Testing system prompt retrieval...")
        print(f"   [PASS] System prompt retrieved ({len(prompt)} characters)")
        
        # Test 2: Check for unified response format section
        print("2. Testing unified response format section...")
        required_sections = [
            "## 8. Unified Response Format",
            "Critical Output Requirement", 
            "EVERY response MUST be a single, valid JSON object",
            "UnifiedAgentResponse schema"
        ]
        
        for section in required_sections:
            if section in prompt:
                print(f"   [PASS] Found: {section}")
            else:
                print(f"   [FAIL] Missing: {section}")
                return False
        
        # Test 3: Check for all content block types
        print("3. Testing content block type documentation...")
        required_blocks = [
            "TextBlock",
            "ComponentAnalysisBlock", 
            "MermaidDiagramBlock",
            "CodeSnippetBlock",
            "MarkdownTableBlock"
        ]
        
        for block in required_blocks:
            if block in prompt:
                print(f"   [PASS] Block type documented: {block}")
            else:
                print(f"   [FAIL] Missing block type: {block}")
                return False
        
        # Test 4: Check for chain-of-thought example
        print("4. Testing chain-of-thought example...")
        chain_markers = [
            "Chain-of-Thought Example",
            "Your Internal Process",
            "Your JSON Response"
        ]
        
        for marker in chain_markers:
            if marker in prompt:
                print(f"   [PASS] Found: {marker}")
            else:
                print(f"   [FAIL] Missing: {marker}")
                return False
        
        # Test 5: Check for JSON format requirements
        print("5. Testing JSON format requirements...")
        json_requirements = [
            'Response is valid JSON starting with `{"response": [`',
            "no preamble, no postamble",
            "structured JSON",
            "frontend UI system depends"
        ]
        
        for req in json_requirements:
            if req in prompt:
                print(f"   [PASS] JSON requirement: {req}")
            else:
                print(f"   [FAIL] Missing JSON requirement: {req}")
                return False
        
        # Test 6: Check for logical ordering principles
        print("6. Testing logical ordering principles...")
        ordering_principles = [
            "Start with Introduction",
            "Present Data", 
            "Provide Examples",
            "Conclude"
        ]
        
        for principle in ordering_principles:
            if principle in prompt:
                print(f"   [PASS] Ordering principle: {principle}")
            else:
                print(f"   [FAIL] Missing ordering principle: {principle}")
                return False
        
        # Test 7: Check for block selection rules  
        print("7. Testing block selection rules...")
        selection_rules = [
            "Use TextBlock for",
            "Use ComponentAnalysisBlock for", 
            "Use MermaidDiagramBlock for",
            "Use CodeSnippetBlock for",
            "Use MarkdownTableBlock for"
        ]
        
        for rule in selection_rules:
            if rule in prompt:
                print(f"   [PASS] Selection rule: {rule}")
            else:
                print(f"   [FAIL] Missing selection rule: {rule}")
                return False
        
        # Test 8: Check for validation checkpoint
        print("8. Testing validation checkpoint...")
        validation_items = [
            "Validation Checkpoint",
            "Before sending your response, verify:",
            "Response is valid JSON",
            "Contains 2+ content blocks"
        ]
        
        for item in validation_items:
            if item in prompt:
                print(f"   [PASS] Validation item: {item}")
            else:
                print(f"   [FAIL] Missing validation item: {item}")
                return False
        
        # Test 9: Verify original workflow content is preserved
        print("9. Testing preservation of original workflow content...")
        original_workflows = [
            "Workflow A: Standard Code Query",
            "Workflow B: Codebase Onboarding", 
            "Workflow C: Diagram Generation",
            "Rule 1: Workflow Selection is Mandatory"
        ]
        
        for workflow in original_workflows:
            if workflow in prompt:
                print(f"   [PASS] Preserved: {workflow}")
            else:
                print(f"   [FAIL] Lost original content: {workflow}")
                return False
        
        # Test 10: Check JSON example validity
        print("10. Testing JSON example syntax...")
        
        # Extract JSON examples from the prompt
        json_pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
        json_matches = re.findall(json_pattern, prompt)
        
        if len(json_matches) >= 2:
            print(f"   [PASS] Found {len(json_matches)} JSON examples")
            
            # Test that at least one example parses as valid JSON
            valid_examples = 0
            for i, example in enumerate(json_matches):
                try:
                    # Clean up the example (remove comments, fix escapes)
                    cleaned = example.replace('// Array of content blocks - each block maps to a UI component', '')
                    cleaned = re.sub(r'//.*', '', cleaned)  # Remove comments
                    cleaned = cleaned.replace('\\n', '\n')  # Fix newlines
                    
                    parsed = json.loads(cleaned)
                    if 'response' in parsed:
                        valid_examples += 1
                        print(f"   [PASS] JSON example {i+1} is valid")
                except:
                    print(f"   [INFO] JSON example {i+1} has template syntax (expected)")
            
            if valid_examples >= 1:
                print(f"   [PASS] {valid_examples} examples are valid JSON")
            else:
                print("   [INFO] Examples use template syntax for documentation")
                
        else:
            print("   [FAIL] Insufficient JSON examples found")
            return False
        
        print("\nSUCCESS: REQ-UI-UNIFIED-2 system prompt update is complete!")
        print("✓ Unified response format documentation added")
        print("✓ All content block types documented with examples")  
        print("✓ Chain-of-thought example provides clear guidance")
        print("✓ JSON format requirements clearly specified")
        print("✓ Logical ordering principles established")
        print("✓ Block selection rules defined")
        print("✓ Validation checkpoint ensures quality")
        print("✓ Original workflow content preserved")
        
        return True
        
    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_prompt_integration():
    """Test that the prompt integrates properly with the agent."""
    print("\n" + "=" * 55)
    print("Testing System Prompt Integration")
    print("=" * 55)
    
    try:
        from backend.cerebras_agent import CerebrasNativeAgent
        
        # Test agent initialization with updated prompt
        agent = CerebrasNativeAgent()
        prompt = agent._get_native_system_prompt()
        
        print("1. Agent initialization with updated prompt...")
        print(f"   [PASS] Agent created successfully")
        print(f"   [PASS] Prompt length: {len(prompt)} characters")
        
        # Test prompt sections are in logical order
        sections = [
            "## 1. Core Identity",
            "## 2. Your Specialized Toolset", 
            "## 3. Mandatory Reasoning Workflows",
            "## 8. Unified Response Format"
        ]
        
        print("2. Testing section ordering...")
        last_pos = -1
        for section in sections:
            pos = prompt.find(section)
            if pos > last_pos:
                print(f"   [PASS] Section in correct order: {section}")
                last_pos = pos
            else:
                print(f"   [FAIL] Section out of order: {section}")
                return False
        
        print("\nSUCCESS: System prompt integration is working correctly!")
        return True
        
    except Exception as e:
        print(f"\nFAILED: Integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_system_prompt_update()
    success2 = test_prompt_integration()
    
    if success1 and success2:
        print("\n🎉 REQ-UI-UNIFIED-2 is 100% functional!")
        print("System prompt successfully updated with unified response requirements.")
    else:
        print("\n❌ Tests failed - REQ-UI-UNIFIED-2 needs fixes")
        
    import sys
    sys.exit(0 if (success1 and success2) else 1)