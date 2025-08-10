#!/usr/bin/env python3
"""
Mermaid Diagram Utilities for CodeWise
Provides syntax validation, correction, and error handling for Mermaid diagrams.
Implements a validation-correction loop using actual Mermaid parser validation.
"""

import re
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MermaidDiagramType(Enum):
    """Supported Mermaid diagram types"""
    GRAPH = "graph"
    FLOWCHART = "flowchart"
    SEQUENCE = "sequenceDiagram"
    CLASS = "classDiagram"
    STATE = "stateDiagram"
    PIE = "pie"
    GITGRAPH = "gitgraph"
    ER = "erDiagram"
    JOURNEY = "journey"


@dataclass
class MermaidValidationResult:
    """Result of Mermaid syntax validation"""
    is_valid: bool
    final_code: str
    original_code: str
    validation_error: Optional[str] = None
    correction_attempts: int = 0
    fixes_applied: List[str] = None
    diagram_type: Optional[MermaidDiagramType] = None
    
    def __post_init__(self):
        if self.fixes_applied is None:
            self.fixes_applied = []


class MermaidValidator:
    """
    Handles Mermaid diagram validation and correction using actual Mermaid parser.
    Implements validation-correction loop for robust syntax checking.
    """
    
    def __init__(self):
        self.max_length = 10000
        self.max_correction_attempts = 3
        self.validation_timeout = 10  # seconds
        
        # Common syntax corrections that can be applied before validation
        self.quick_fixes = [
            # Replace problematic Unicode characters
            (re.compile(r'[‑–—]'), '-', "Replace Unicode dashes with hyphens"),
            (re.compile(r'[\u2018\u2019\u201A]'), "'", "Replace smart single quotes with regular quotes"),  
            (re.compile(r'[\u201C\u201D\u201E]'), '"', "Replace smart double quotes with regular quotes"),
            (re.compile(r'\u2026'), '...', "Replace ellipsis character"),
            
            # Fix common node label issues
            (re.compile(r'\[([^\]]*)\(([^)]*)\)([^\]]*)\]'), r'[\1\2\3]', "Remove parentheses from node labels"),
            
            # Clean up whitespace
            (re.compile(r'\s+'), ' ', "Normalize whitespace"),
            (re.compile(r'\n\s*\n\s*\n+'), '\n\n', "Reduce excessive line breaks"),
        ]
        
        # Valid diagram type patterns for detection
        self.diagram_patterns = {
            MermaidDiagramType.GRAPH: re.compile(r'^\s*graph\s+(TD|LR|BT|RL|TB)', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.FLOWCHART: re.compile(r'^\s*flowchart\s+(TD|LR|BT|RL|TB)', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.SEQUENCE: re.compile(r'^\s*sequenceDiagram', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.CLASS: re.compile(r'^\s*classDiagram', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.STATE: re.compile(r'^\s*stateDiagram', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.PIE: re.compile(r'^\s*pie\s+title', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.GITGRAPH: re.compile(r'^\s*gitgraph', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.ER: re.compile(r'^\s*erDiagram', re.IGNORECASE | re.MULTILINE),
            MermaidDiagramType.JOURNEY: re.compile(r'^\s*journey', re.IGNORECASE | re.MULTILINE),
        }
        
        # Create validator script content for Node.js execution
        self.validator_script = '''
const mermaid = require('mermaid');

process.stdin.resume();
process.stdin.setEncoding('utf8');

let input = '';
process.stdin.on('data', function(chunk) {
    input += chunk;
});

process.stdin.on('end', function() {
    try {
        const code = input.trim();
        if (!code) {
            console.log(JSON.stringify({valid: false, error: "Empty input"}));
            return;
        }
        
        // Initialize mermaid
        mermaid.initialize({startOnLoad: false, securityLevel: 'strict'});
        
        // Try to parse the diagram
        const result = mermaid.parse(code);
        console.log(JSON.stringify({valid: true, parsed: result}));
    } catch (error) {
        console.log(JSON.stringify({valid: false, error: error.message}));
    }
});
'''
    
    def validate_and_correct(self, mermaid_code: str, llm_correction_func: Optional[callable] = None) -> MermaidValidationResult:
        """
        Validate and correct Mermaid diagram code using validation-correction loop.
        
        Args:
            mermaid_code: Raw Mermaid diagram code
            llm_correction_func: Optional function to call LLM for corrections
            
        Returns:
            MermaidValidationResult with validation results and corrected code
        """
        original_code = mermaid_code.strip()
        
        # Input validation
        if not original_code:
            return MermaidValidationResult(
                is_valid=False,
                final_code="",
                original_code="",
                validation_error="Empty or null input"
            )
        
        # Check length limits
        if len(original_code) > self.max_length:
            logger.warning(f"Mermaid code too long ({len(original_code)} > {self.max_length}), truncating")
            original_code = original_code[:self.max_length] + "..."
        
        current_code = original_code
        fixes_applied = []
        correction_attempts = 0
        
        # Apply quick fixes first
        current_code, quick_fixes = self._apply_quick_fixes(current_code)
        fixes_applied.extend(quick_fixes)
        
        # Validation-correction loop
        for attempt in range(self.max_correction_attempts):
            validation_result = self._validate_with_parser(current_code)
            
            if validation_result["valid"]:
                logger.info(f"✅ Mermaid validation successful after {attempt} corrections")
                return MermaidValidationResult(
                    is_valid=True,
                    final_code=current_code,
                    original_code=original_code,
                    correction_attempts=correction_attempts,
                    fixes_applied=fixes_applied,
                    diagram_type=self._detect_diagram_type(current_code)
                )
            
            error_message = validation_result.get("error", "Unknown validation error")
            logger.info(f"🔄 Mermaid validation failed (attempt {attempt + 1}): {error_message}")
            
            # Try to correct the error
            if llm_correction_func and attempt < self.max_correction_attempts - 1:
                try:
                    corrected_code = llm_correction_func(current_code, error_message)
                    if corrected_code and corrected_code.strip() != current_code.strip():
                        current_code = corrected_code.strip()
                        correction_attempts += 1
                        fixes_applied.append(f"LLM correction for: {error_message}")
                        continue
                except Exception as e:
                    logger.warning(f"LLM correction failed: {e}")
            
            # Apply heuristic fixes based on error message
            corrected_code, fix_description = self._apply_heuristic_fixes(current_code, error_message)
            if corrected_code != current_code:
                current_code = corrected_code
                correction_attempts += 1
                fixes_applied.append(fix_description)
            else:
                # No more corrections possible
                break
        
        # If we get here, validation failed after all attempts
        logger.warning(f"❌ Mermaid validation failed after {self.max_correction_attempts} attempts")
        return MermaidValidationResult(
            is_valid=False,
            final_code=current_code,
            original_code=original_code,
            validation_error=validation_result.get("error", "Validation failed"),
            correction_attempts=correction_attempts,
            fixes_applied=fixes_applied,
            diagram_type=self._detect_diagram_type(current_code)
        )
    
    def _apply_quick_fixes(self, code: str) -> Tuple[str, List[str]]:
        """Apply quick syntax fixes before validation"""
        fixes_applied = []
        current_code = code
        
        for pattern, replacement, description in self.quick_fixes:
            matches = pattern.findall(current_code)
            if matches:
                current_code = pattern.sub(replacement, current_code)
                fixes_applied.append(f"{description} ({len(matches)} instances)")
        
        return current_code.strip(), fixes_applied
    
    def _validate_with_parser(self, code: str) -> Dict[str, Union[bool, str]]:
        """
        Validate Mermaid code using the actual Mermaid parser via Node.js
        Falls back to basic syntax checking if Node.js/mermaid is not available
        """
        try:
            # Try to validate with actual Mermaid parser via Node.js
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as script_file:
                script_file.write(self.validator_script)
                script_file.flush()
                
                # Run Node.js validation
                result = subprocess.run(
                    ['node', script_file.name],
                    input=code,
                    text=True,
                    capture_output=True,
                    timeout=self.validation_timeout
                )
                
                Path(script_file.name).unlink()  # Clean up temp file
                
                if result.returncode == 0:
                    try:
                        validation_result = json.loads(result.stdout)
                        return validation_result
                    except json.JSONDecodeError:
                        return {"valid": False, "error": "Invalid JSON response from validator"}
                else:
                    return {"valid": False, "error": f"Validation process error: {result.stderr}"}
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Node.js Mermaid validation failed: {e}, falling back to basic validation")
            return self._basic_syntax_validation(code)
        except Exception as e:
            logger.warning(f"Mermaid validation error: {e}, falling back to basic validation")
            return self._basic_syntax_validation(code)
    
    def _basic_syntax_validation(self, code: str) -> Dict[str, Union[bool, str]]:
        """Basic syntax validation as fallback when Node.js validation is unavailable"""
        issues = []
        
        # Check for valid diagram type
        diagram_type = self._detect_diagram_type(code)
        if not diagram_type:
            return {"valid": False, "error": "Unknown or unsupported diagram type"}
        
        # Check for balanced brackets and parentheses
        if not self._check_balanced_brackets(code):
            issues.append("Unbalanced brackets or parentheses")
        
        # Basic graph syntax validation
        if diagram_type in [MermaidDiagramType.GRAPH, MermaidDiagramType.FLOWCHART]:
            graph_issues = self._validate_basic_graph_syntax(code)
            issues.extend(graph_issues)
        
        if issues:
            return {"valid": False, "error": "; ".join(issues)}
        
        return {"valid": True}
    
    def _detect_diagram_type(self, code: str) -> Optional[MermaidDiagramType]:
        """Detect the type of Mermaid diagram from code"""
        for diagram_type, pattern in self.diagram_patterns.items():
            if pattern.search(code):
                return diagram_type
        return None
    
    def _check_balanced_brackets(self, code: str) -> bool:
        """Check if brackets and parentheses are balanced"""
        brackets = {'[': ']', '(': ')', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                last_open = stack.pop()
                if brackets.get(last_open) != char:
                    return False
        
        return len(stack) == 0
    
    def _validate_basic_graph_syntax(self, code: str) -> List[str]:
        """Basic validation for graph/flowchart syntax"""
        issues = []
        
        # Check for connections
        connection_pattern = re.compile(r'(\w+)\s*(-->|---|\.\.\>|==\>)\s*(\w+)')
        connections = connection_pattern.findall(code)
        
        if not connections:
            issues.append("No valid connections found in graph")
        
        return issues
    
    def _apply_heuristic_fixes(self, code: str, error_message: str) -> Tuple[str, str]:
        """Apply heuristic fixes based on validation error message"""
        error_lower = error_message.lower()
        
        # Fix common issues based on error patterns
        if "unexpected" in error_lower or "invalid" in error_lower:
            # Try removing problematic characters
            fixed_code = re.sub(r'[^\w\s\[\]\-\>\.\n]', '', code)
            if fixed_code != code:
                return fixed_code, f"Removed special characters due to: {error_message}"
        
        if "bracket" in error_lower or "parenthes" in error_lower:
            # Try to fix bracket issues
            fixed_code = self._fix_bracket_issues(code)
            if fixed_code != code:
                return fixed_code, f"Fixed bracket issues due to: {error_message}"
        
        # If no specific fix could be applied
        return code, "No heuristic fix available"
    
    def _fix_bracket_issues(self, code: str) -> str:
        """Try to fix common bracket issues"""
        # Remove unmatched brackets at the end
        while code and code[-1] in '])}':
            code = code[:-1]
        
        return code.strip()


# Singleton instance for reuse
_mermaid_validator = None

def get_mermaid_validator() -> MermaidValidator:
    """Get singleton MermaidValidator instance"""
    global _mermaid_validator
    if _mermaid_validator is None:
        _mermaid_validator = MermaidValidator()
    return _mermaid_validator


def validate_and_correct_mermaid(code: str, llm_correction_func: Optional[callable] = None) -> MermaidValidationResult:
    """
    Convenience function to validate and correct Mermaid diagram code
    
    Args:
        code: Raw Mermaid diagram code
        llm_correction_func: Optional function to call LLM for corrections
        
    Returns:
        MermaidValidationResult with validation results and corrected code
    """
    validator = get_mermaid_validator()
    return validator.validate_and_correct(code, llm_correction_func)


def validate_mermaid_simple(code: str) -> Tuple[str, bool]:
    """
    Simple validation function that returns corrected code and validity
    
    Args:
        code: Raw Mermaid diagram code
        
    Returns:
        Tuple of (corrected_code, is_valid)
    """
    result = validate_and_correct_mermaid(code)
    return result.final_code, result.is_valid